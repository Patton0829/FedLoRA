import os
import random
import re
import shutil
from typing import List
from tqdm import tqdm
import fire
import torch
import numpy as np
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from peft import (
    LoraConfig,
    cast_mixed_precision_params,
    get_peft_model,
)
from fed_utils import (
    FedAvg,
    SCAFFOLD,
    HAA,
    client_selection,
    evaluate_dataset_records,
    global_evaluation,
    save_acc_history,
    save_confusion_matrix,
    save_prediction_samples,
    plot_acc_curve,
    plot_confusion_matrix_heatmap,
    GeneralClient,
)
import datasets
from utils.prompter import Prompter
import json

file_path = './HF_key.json'
with open(file_path, 'r') as file:
    keys = json.load(file)

datasets.utils.logging.set_verbosity_error()


def _is_scaffold_like_algorithm(aggregation_algorithm: str) -> bool:
    algo = aggregation_algorithm.lower()
    return algo in {
        "scaffold",
        "haa",
    }


def setup_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def fl_finetune(
        # model/data params
        global_model: str = '',
        data_path: str = './data',
        dev_data_path: str = './data/10/global_test.json',
        output_dir: str = './lora-Qwen2.5-1.5B-Instruct/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        aggregation_algorithm: str = 'scaffold',
        haa_tau: float = 5.0,
        heterogeneity_tau: float = None,
        client_selection_frac: float = 0.5,
        num_communication_rounds: int = 200,
        num_clients: int = 10,
        # Local training hyperparams
        local_batch_size: int = 32,  # 64,
        local_micro_batch_size: int = 16,
        local_num_epochs: int = 1,
        local_learning_rate: float = 1e-4,
        local_lr_scheduler_type: str = "cosine",
        local_val_set_size: int = 0,
        eval_batch_size: int = 16,
        global_eval_every_rounds: int = 10,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = False,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        resume_round: int = None,  # 0-based epoch index to resume from; inferred from checkpoint path when omitted
        use_gradient_checkpointing: bool = True,
        gradient_checkpointing_use_reentrant: bool = False,
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        seed: int = 42,
):
    setup_reproducibility(seed)
    if heterogeneity_tau is not None:
        # Backward compatibility for older arg name.
        haa_tau = float(heterogeneity_tau)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"aggregation_algorithm: {aggregation_algorithm}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"haa_tau: {haa_tau}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_lr_scheduler_type: {local_lr_scheduler_type}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"eval_batch_size: {eval_batch_size}\n"
            f"global_eval_every_rounds: {global_eval_every_rounds}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"resume_round: {resume_round if resume_round is not None else 'auto'}\n"
            f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
            f"gradient_checkpointing_use_reentrant: {gradient_checkpointing_use_reentrant}\n"
            f"prompt template: {prompt_template_name}\n"
            f"seed: {seed}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), "Please generate the data files for each client"

    # set up the global model & toknizer
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
    if use_cuda:
        print(f"Mixed precision mode: {'bf16' if use_bf16 else 'fp16'}")
    device_map = {"": 0} if use_cuda else None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    elif not use_cuda:
        print("CUDA is not available. Training will run on CPU and will be very slow.")

    # model = LlamaForCausalLM.from_pretrained(
    #     global_model,
    #     load_in_8bit=True,
    #     torch_dtype=torch.float16,
    #     device_map=device_map,
    # )

    # tokenizer = LlamaTokenizer.from_pretrained(global_model)

    if global_model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=load_dtype,
            device_map=device_map,
        )
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b':
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            torch_dtype=load_dtype,
            device_map=device_map,
            token=keys["hf_token"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            torch_dtype=load_dtype,
            device_map=device_map,
            token=keys["hf_token"],
        )

    if global_model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(global_model)
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b':
        tokenizer = AutoTokenizer.from_pretrained(global_model, token=keys["hf_token"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(global_model, token=keys["hf_token"])

    # Prefer model/tokenizer native special-token configuration.
    # Fallback to eos token only when pad token is missing.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif model.config.eos_token_id is not None:
            tokenizer.pad_token_id = model.config.eos_token_id
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    print(
        f"Tokenizer special tokens: pad_token_id={tokenizer.pad_token_id}, "
        f"eos_token_id={tokenizer.eos_token_id}, bos_token_id={tokenizer.bos_token_id}"
    )

    

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point.get("context", data_point.get("input", "")),
            data_point.get("response", data_point.get("output", "")),
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point.get("context", data_point.get("input", ""))
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    if use_gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": gradient_checkpointing_use_reentrant}
            )
        except TypeError:
            # Older transformers versions may not support kwargs.
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    if use_cuda and load_dtype in (torch.float16, torch.bfloat16):
        cast_mixed_precision_params(model, load_dtype)
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    local_step_count_dict = dict()
    scaffold_state = None
    output_dir = os.path.join(output_dir, str(num_clients))
    result_dir = os.path.join("./resault", str(num_clients))
    final_output_dir = os.path.join(output_dir, "final")

    acc_list = []
    round_records = []
    start_epoch = 0

    if resume_from_checkpoint:
        checkpoint_path = resume_from_checkpoint
        if os.path.isdir(checkpoint_path):
            candidate = os.path.join(checkpoint_path, "adapter_model.bin")
            if os.path.exists(candidate):
                checkpoint_path = candidate
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"resume_from_checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded adapter from {checkpoint_path}")
        if load_result.missing_keys:
            print(f"Missing keys while loading: {len(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            print(f"Unexpected keys while loading: {len(load_result.unexpected_keys)}")

        if resume_round is not None:
            start_epoch = int(resume_round)
        else:
            # Infer from .../<epoch_idx>/adapter_model.bin
            parent_name = os.path.basename(os.path.dirname(checkpoint_path))
            if re.fullmatch(r"\d+", parent_name):
                start_epoch = int(parent_name) + 1
        print(f"Resuming from epoch index {start_epoch} (human-readable round {start_epoch + 1})")

        history_path = os.path.join(result_dir, "acc_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
                if isinstance(history_data, list):
                    round_records = history_data
                    acc_list = [float(item["accuracy"]) for item in round_records if "accuracy" in item]
                    print(f"Loaded existing accuracy history entries: {len(round_records)}")
            except Exception as exc:
                print(f"Warning: failed to load acc history from {history_path}: {exc}")

    for epoch in tqdm(range(start_epoch, num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)
        local_client_metrics = []

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size, seed)
            server_control = None
            client_control = None
            if _is_scaffold_like_algorithm(aggregation_algorithm):
                if scaffold_state is not None:
                    server_control = scaffold_state.get("server_control")
                    client_control = scaffold_state.get("client_controls", {}).get(client_id)
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       local_lr_scheduler_type,
                                       group_by_length,
                                       ddp,
                                       fl_algorithm=aggregation_algorithm,
                                       server_control=server_control,
                                       client_control=client_control,
                                       seed=seed)

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            if local_val_set_size > 0:
                local_acc = evaluate_dataset_records(
                    client.model,
                    tokenizer,
                    prompter,
                    client.local_eval_records,
                    dataset_name=f"Client_{client_id} Local Evaluation",
                    eval_batch_size=eval_batch_size,
                )
                local_client_metrics.append(
                    {
                        "client_id": int(client_id),
                        "eval_source": client.local_eval_source,
                        "eval_file": client.local_data_path,
                        "eval_samples": int(len(client.local_eval_records)),
                        "local_accuracy": float(local_acc),
                        "local_eval_skipped": False,
                    }
                )
            else:
                print(f"Skip local generative evaluation for Client_{client_id} because local_val_set_size=0")
                local_client_metrics.append(
                    {
                        "client_id": int(client_id),
                        "eval_source": client.local_eval_source,
                        "eval_file": client.local_data_path,
                        "eval_samples": 0,
                        "local_accuracy": None,
                        "local_eval_skipped": True,
                    }
                )

            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, local_step_count_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, local_step_count_dict, previously_selected_clients_set)
            del client

        print("Collecting the weights of clients and performing aggregation")
        if aggregation_algorithm.lower() == "scaffold":
            model, scaffold_state = SCAFFOLD(
                model,
                selected_clients_set,
                output_dir,
                local_dataset_len_dict,
                epoch,
                scaffold_state=scaffold_state,
                num_clients=num_clients,
                local_steps=local_step_count_dict,
                local_lr=local_learning_rate,
            )
        elif aggregation_algorithm.lower() == "haa":
            model, scaffold_state = HAA(
                model,
                selected_clients_set,
                output_dir,
                local_dataset_len_dict,
                epoch,
                scaffold_state=scaffold_state,
                num_clients=num_clients,
                local_steps=local_step_count_dict,
                local_lr=local_learning_rate,
                tau=haa_tau,
            )
        else:
            model = FedAvg(model,
                           selected_clients_set,
                           output_dir,
                           local_dataset_len_dict,
                           epoch,
                           )

        should_eval = ((epoch + 1) % max(1, int(global_eval_every_rounds)) == 0) or (
            epoch == num_communication_rounds - 1
        )
        if should_eval:
            # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
            global_eval_result = global_evaluation(
                model,
                tokenizer,
                prompter,
                dev_data_path,
                return_details=True,
                sample_size=0,
                mistake_sample_size=20,
                eval_batch_size=eval_batch_size,
            )
            acc = global_eval_result["accuracy"]
            round_result_dir = os.path.join(result_dir, f"round_{epoch + 1}")
            os.makedirs(result_dir, exist_ok=True)
            for dirname in os.listdir(result_dir):
                if dirname.startswith("round_"):
                    stale_round_dir = os.path.join(result_dir, dirname)
                    if os.path.isdir(stale_round_dir) and stale_round_dir != round_result_dir:
                        shutil.rmtree(stale_round_dir, ignore_errors=True)

            mistakes_filename = "mistake_samples.json"
            confusion_filename = "confusion_matrix.json"
            confusion_png_filename = "confusion_matrix.png"
            save_prediction_samples(global_eval_result["mistake_samples"], round_result_dir, mistakes_filename)
            save_confusion_matrix(global_eval_result["confusion_matrix"], round_result_dir, confusion_filename)
            plot_confusion_matrix_heatmap(
                global_eval_result["confusion_matrix"],
                round_result_dir,
                filename=confusion_png_filename,
                title=f"Round {epoch + 1} Global Confusion Matrix",
            )
            print('Acc of Epoch', str(epoch), 'is:', acc)
            acc_list.append(acc)
            round_records.append(
                {
                    "round": int(epoch + 1),
                    "selected_clients": sorted(int(client_id) for client_id in selected_clients_set),
                    "local_client_metrics": local_client_metrics,
                    "global_eval_file": dev_data_path,
                    "global_prediction_samples_file": None,
                    "global_mistakes_file": os.path.join(round_result_dir, mistakes_filename),
                    "global_confusion_matrix_file": os.path.join(round_result_dir, confusion_filename),
                    "global_confusion_matrix_png_file": os.path.join(round_result_dir, confusion_png_filename),
                    "global_per_label_accuracy": global_eval_result["per_label_accuracy"],
                    "accuracy": float(acc),
                }
            )
            save_acc_history(acc_list, result_dir, round_records=round_records)
            plot_acc_curve(acc_list, result_dir)
        else:
            print(f"Skip global evaluation at round {epoch + 1} (evaluate every {global_eval_every_rounds} rounds).")

        # Keep only the final global adapter/config; remove round artifacts after aggregation.
        round_output_dir = os.path.join(output_dir, str(epoch))
        if os.path.isdir(round_output_dir):
            shutil.rmtree(round_output_dir, ignore_errors=True)


    print(acc_list)
    os.makedirs(final_output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_output_dir, "adapter_model.bin"))
    config.save_pretrained(final_output_dir)
    print("Accuracy history and curve saved.")
    print(f"Final global adapter saved to {final_output_dir}")


if __name__ == "__main__":
    fire.Fire(fl_finetune)

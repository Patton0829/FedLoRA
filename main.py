import os
import random
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
    client_selection,
    evaluate_dataset_records,
    global_evaluation,
    save_acc_history,
    plot_acc_curve,
    GeneralClient,
)
import datasets
from utils.prompter import Prompter
import json

file_path = './HF_key.json'
with open(file_path, 'r') as file:
    keys = json.load(file)

datasets.utils.logging.set_verbosity_error()


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
        dev_data_path: str = './dataset/jnu/jnu_test_with_labels.json',
        output_dir: str = './lora-shepherd/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.1,
        num_communication_rounds: int = 10,
        num_clients: int = 100,
        # Local training hyperparams
        local_batch_size: int = 64,  # 64,
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 10,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
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
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        seed: int = 42,
):
    setup_reproducibility(seed)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
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
    load_dtype = torch.float16 if use_cuda else torch.float32
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
            dtype=load_dtype,
            device_map=device_map,
        )
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b':
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=False,
            dtype=load_dtype,
            device_map=device_map,
            token=keys["hf_token"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            dtype=load_dtype,
            device_map=device_map,
            token=keys["hf_token"],
        )

    if global_model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(global_model)
    elif global_model == 'google/gemma-2b' or global_model == 'google/gemma-7b':
        tokenizer = AutoTokenizer.from_pretrained(global_model, token=keys["hf_token"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(global_model, token=keys["hf_token"])

    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    

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

    model.gradient_checkpointing_enable()
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
    output_dir = os.path.join(output_dir, str(num_clients))
    result_dir = os.path.join("./resault", str(num_clients))
    final_output_dir = os.path.join(output_dir, "final")

    acc_list = []
    round_records = []

    for epoch in tqdm(range(num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=epoch)
        local_client_metrics = []

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size, seed)
            client.build_local_trainer(tokenizer,
                                       local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       local_num_epochs,
                                       local_learning_rate,
                                       group_by_length,
                                       ddp,
                                       seed)

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            local_acc = evaluate_dataset_records(
                client.model,
                tokenizer,
                prompter,
                client.local_eval_records,
                dataset_name=f"Client_{client_id} Local Evaluation",
            )
            local_client_metrics.append(
                {
                    "client_id": int(client_id),
                    "eval_source": client.local_eval_source,
                    "eval_file": client.local_data_path,
                    "eval_samples": int(len(client.local_eval_records)),
                    "local_accuracy": float(local_acc),
                }
            )

            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client

        print("Collecting the weights of clients and performing aggregation")
        model = FedAvg(model,
                       selected_clients_set,
                       output_dir,
                       local_dataset_len_dict,
                       epoch,
                       )
        torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        config.save_pretrained(output_dir)

        # Please design the evaluation method based on your specific requirements in the fed_utils/evaluation.py file.
        acc = global_evaluation(model, tokenizer, prompter, dev_data_path)
        print('Acc of Epoch', str(epoch), 'is:', acc)
        acc_list.append(acc)
        round_records.append(
            {
                "round": int(epoch + 1),
                "selected_clients": sorted(int(client_id) for client_id in selected_clients_set),
                "local_client_metrics": local_client_metrics,
                "global_eval_file": dev_data_path,
                "accuracy": float(acc),
            }
        )
        save_acc_history(acc_list, result_dir, round_records=round_records)
        plot_acc_curve(acc_list, result_dir)


    print(acc_list)
    os.makedirs(final_output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_output_dir, "adapter_model.bin"))
    config.save_pretrained(final_output_dir)
    print("Accuracy history and curve saved.")
    print(f"Final global adapter saved to {final_output_dir}")


if __name__ == "__main__":
    fire.Fire(fl_finetune)

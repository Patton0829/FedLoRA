import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


def _control_key_from_param_name(param_name):
    return param_name.replace(".default", "")


def _use_scaffold_like_training(fl_algorithm):
    algo = fl_algorithm.lower()
    return algo in {
        "scaffold",
        "haa",
    }


class SCAFFOLDTrainer(transformers.Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        param
                        for name, param in self.model.named_parameters()
                        if param.requires_grad and name in decay_parameters
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        param
                        for name, param in self.model.named_parameters()
                        if param.requires_grad and name not in decay_parameters
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        return self.optimizer


class SCAFFOLDCallback(transformers.TrainerCallback):
    def __init__(self, server_control=None, client_control=None):
        self.server_control = server_control or {}
        self.client_control = client_control or {}

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None or "default" not in name:
                continue

            control_key = _control_key_from_param_name(name)
            server_value = self.server_control.get(control_key)
            client_value = self.client_control.get(control_key)
            if server_value is None or client_value is None:
                continue

            correction = server_value.to(param.grad.device, dtype=param.grad.dtype) - client_value.to(
                param.grad.device, dtype=param.grad.dtype
            )
            param.grad.add_(correction)

        return control


class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.local_num_steps = 0

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size, seed=42):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=seed
            )
            self.local_eval_records = list(local_train_val["test"])
            self.local_eval_source = "held_out_local_split"
            self.local_train_dataset = (
                local_train_val["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
            )
        else:
            self.local_eval_records = list(self.local_data["train"])
            self.local_eval_source = "local_training_full_set"
            self.local_train_dataset = self.local_data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp,
                            fl_algorithm="fedavg",
                            server_control=None,
                            client_control=None,
                            seed=42):
        use_cuda = torch.cuda.is_available()
        is_scaffold = _use_scaffold_like_training(fl_algorithm)
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=use_cuda,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="no",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            output_dir=self.local_output_dir,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            dataloader_drop_last=False,
            dataloader_pin_memory=use_cuda,
            lr_scheduler_type="linear",
            max_grad_norm=0.0 if is_scaffold else 1.0,
            weight_decay=0.0 if is_scaffold else 0.0,
            seed=seed,
            data_seed=seed,
            full_determinism=True,
        )
        trainer_callbacks = []
        if is_scaffold:
            trainer_callbacks.append(
                SCAFFOLDCallback(
                    server_control=server_control,
                    client_control=client_control,
                )
            )

        trainer_cls = SCAFFOLDTrainer if is_scaffold else transformers.Trainer
        self.local_trainer = trainer_cls(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            callbacks=trainer_callbacks,
        )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        train_result = self.local_trainer.train()
        self.local_num_steps = int(getattr(train_result, "global_step", self.local_trainer.state.global_step) or 0)

    def _restore_model_forward_after_trainer(self):
        """
        Prevent recursive wrapper accumulation from repeated Trainer/Accelerate construction.
        When the same model instance is prepared many times, accelerate may keep wrapping
        model.forward (autocast + convert_to_fp32), eventually causing RecursionError.
        """
        accelerator = getattr(self.local_trainer, "accelerator", None)
        if accelerator is not None:
            try:
                self.model = accelerator.unwrap_model(self.model)
            except Exception:
                pass

        original_forward = getattr(self.model, "_original_forward", None)
        if original_forward is not None:
            self.model.forward = original_forward
            try:
                delattr(self.model, "_original_forward")
            except Exception:
                pass

    def terminate_local_training(self, epoch, local_dataset_len_dict, local_step_count_dict, previously_selected_clients_set):

        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        local_step_count_dict[self.client_id] = int(self.local_num_steps)
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        self._restore_model_forward_after_trainer()
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, local_step_count_dict, previously_selected_clients_set, last_client_id

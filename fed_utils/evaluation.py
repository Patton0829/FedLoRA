import json
import os
import random
import re

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import GenerationConfig


model_type = "llama"
datasets.utils.logging.set_verbosity_error()
device_map = "auto"
max_new_token: int = 32
verbose: bool = False


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1)


def normalize_label(text):
    if text is None:
        return ""

    normalized = str(text).strip().lower()
    prefixes = (
        "the diagnosis result is ",
        "diagnosis result is ",
        "the result is ",
        "the answer is: ",
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    normalized = normalized.splitlines()[0].strip()
    normalized = re.sub(r"^[a-z]\.\s*", "", normalized)
    normalized = re.sub(r"[.\s]+$", "", normalized)
    return normalized


def save_acc_history(acc_list, save_dir, filename="acc_history.json"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    history = []
    for round_idx, acc in enumerate(acc_list, start=1):
        history.append({"round": round_idx, "acc": float(acc)})

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return save_path


def plot_acc_curve(acc_list, save_dir, filename="acc_curve.png"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    rounds = list(range(1, len(acc_list) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, acc_list, marker="o", linewidth=2)
    plt.title("Global Evaluation Accuracy")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.xticks(rounds)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def global_evaluation(model, tokenizer, prompter, dev_data_path):
    with open(dev_data_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    label_set = sorted(
        {
            normalize_label(data_point.get("output", data_point.get("response", "")))
            for data_point in test_set
            if normalize_label(data_point.get("output", data_point.get("response", "")))
        }
    )
    right_count_dict = dict.fromkeys(label_set, 0)
    total_count_dict = dict.fromkeys(label_set, 0)
    acc_count_dict = dict.fromkeys(label_set, 0)

    sampling_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_token,
    }
    if model_type == "gpt2":
        sampling_kwargs.update(
            {
                "bos_token_id": 50256,
                "eos_token_id": 50256,
                "_from_model_config": True,
            }
        )
    sampling = GenerationConfig(**sampling_kwargs)

    model_device = next(model.parameters()).device
    autocast_enabled = model_device.type == "cuda"

    for data_point in tqdm(test_set):
        target = normalize_label(data_point.get("output", data_point.get("response", "")))
        if not target:
            continue

        model_input = data_point.get("input", data_point.get("context", ""))
        test_prompt = prompter.generate_prompt(
            data_point["instruction"],
            model_input,
            None,
        )

        with torch.autocast(device_type="cuda", enabled=autocast_enabled):
            inputs = tokenizer(test_prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model_device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)

            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=sampling,
                    return_dict_in_generate=True,
                    output_scores=False,
                    max_new_tokens=max_new_token,
                    pad_token_id=tokenizer.eos_token_id,
                )

        generated_tokens = generation_output.sequences[0][input_ids.shape[-1] :]
        predicted_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        predicted_label = normalize_label(predicted_text)

        if verbose:
            print("-------------------")
            print(test_prompt)
            print("target:", target)
            print("predicted raw:", predicted_text)
            print("predicted normalized:", predicted_label)

        if predicted_label == target:
            right_count_dict[target] += 1
        total_count_dict[target] += 1

    mean_acc = 0.0

    for key in acc_count_dict.keys():
        if total_count_dict[key] == 0:
            acc_count_dict[key] = 0.0
            continue
        tmp = right_count_dict[key] / total_count_dict[key]
        mean_acc += tmp
        acc_count_dict[key] = tmp

    if acc_count_dict:
        mean_acc /= len(acc_count_dict.keys())

    if verbose:
        print(right_count_dict)
    print("Acc: ", acc_count_dict)
    print()
    print("========== Accuracy ==========")
    print(mean_acc)

    return mean_acc

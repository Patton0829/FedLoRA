import json
import os
import random
import re
from difflib import SequenceMatcher

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
KNOWN_LABELS = [
    "normal",
    "ball fault",
    "inner race fault",
    "outer race fault",
]
STRICT_DIAGNOSIS_INSTRUCTION = (
    "You are an expert in bearing fault diagnosis. Based on the provided time-domain features, "
    "identify the bearing condition and output only one label from the following options: "
    "inner race fault, normal, outer race fault, ball fault. "
    "Do not output any explanation, punctuation, or extra words."
)


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

    alias_map = {
        "bearing is normal": "normal",
        "healthy": "normal",
        "healthy bearing": "normal",
        "ball": "ball fault",
        "ballfault": "ball fault",
        "inner race": "inner race fault",
        "inner-race fault": "inner race fault",
        "inner ring fault": "inner race fault",
        "outer race": "outer race fault",
        "outer-race fault": "outer race fault",
        "outer ring fault": "outer race fault",
    }

    if normalized in alias_map:
        return alias_map[normalized]

    for label in KNOWN_LABELS:
        if label in normalized:
            return label

    for alias, label in alias_map.items():
        if alias in normalized:
            return label

    return normalized


def coerce_to_known_label(predicted_text):
    normalized = normalize_label(predicted_text)
    if normalized in KNOWN_LABELS:
        return normalized

    if not normalized:
        return "normal"

    best_label = KNOWN_LABELS[0]
    best_score = -1.0
    for label in KNOWN_LABELS:
        score = SequenceMatcher(None, normalized, label).ratio()
        if label.split()[0] in normalized or normalized.split()[0] in label:
            score += 0.15
        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def save_acc_history(acc_list, save_dir, filename="acc_history.json", round_records=None):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    if round_records is not None:
        history = round_records
    else:
        history = []
        for round_idx, acc in enumerate(acc_list, start=1):
            history.append({"round": round_idx, "accuracy": float(acc)})

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


def save_prediction_samples(samples, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    return save_path


def save_confusion_matrix(confusion_matrix, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(confusion_matrix, f, ensure_ascii=False, indent=2)
    return save_path


def plot_confusion_matrix_heatmap(confusion_matrix, save_dir, filename="confusion_matrix.png", title="Confusion Matrix"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    labels = [label for label in KNOWN_LABELS if label in confusion_matrix]
    labels.extend([label for label in confusion_matrix if label not in labels and label in KNOWN_LABELS])
    pred_labels = labels

    matrix = np.array([[confusion_matrix.get(true_label, {}).get(pred_label, 0) for pred_label in pred_labels] for true_label in labels])

    display_map = {
        "normal": "Normal",
        "ball fault": "Ball Fault",
        "inner race fault": "Inner Race Fault",
        "outer race fault": "Outer Race Fault",
    }
    display_true_labels = [display_map.get(label, label) for label in labels]
    display_pred_labels = [display_map.get(label, label) for label in pred_labels]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xticks(np.arange(len(display_pred_labels)))
    ax.set_yticks(np.arange(len(display_true_labels)))
    ax.set_xticklabels(display_pred_labels, rotation=25, ha="right", fontsize=11)
    ax.set_yticklabels(display_true_labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)

    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def evaluate_dataset_records(
    model,
    tokenizer,
    prompter,
    test_set,
    dataset_name=None,
    return_details=False,
    sample_size=5,
    mistake_sample_size=20,
):
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
    prediction_samples = []
    mistake_samples = []
    confusion_labels = sorted(set(label_set) | set(KNOWN_LABELS))
    confusion_matrix = {label: {pred_label: 0 for pred_label in confusion_labels} for label in confusion_labels}

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
            STRICT_DIAGNOSIS_INSTRUCTION,
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
        predicted_label = coerce_to_known_label(predicted_text)

        if verbose:
            print("-------------------")
            print(test_prompt)
            print("target:", target)
            print("predicted raw:", predicted_text)
            print("predicted normalized:", predicted_label)

        is_correct = predicted_label == target
        if is_correct:
            right_count_dict[target] += 1
        total_count_dict[target] += 1
        confusion_matrix[target][predicted_label] += 1

        if len(prediction_samples) < sample_size:
            prediction_samples.append(
                {
                    "instruction": data_point.get("instruction", ""),
                    "input": model_input,
                    "target": target,
                    "predicted_raw": predicted_text,
                    "predicted_normalized": predicted_label,
                    "is_correct": is_correct,
                }
            )
        if not is_correct and len(mistake_samples) < mistake_sample_size:
            mistake_samples.append(
                {
                    "instruction": data_point.get("instruction", ""),
                    "input": model_input,
                    "target": target,
                    "predicted_raw": predicted_text,
                    "predicted_normalized": predicted_label,
                    "is_correct": False,
                }
            )

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
    if dataset_name:
        print(f"========== {dataset_name} ==========")
    print("Acc: ", acc_count_dict)
    print()
    print("========== Accuracy ==========")
    print(mean_acc)

    if return_details:
        return {
            "accuracy": mean_acc,
            "per_label_accuracy": acc_count_dict,
            "prediction_samples": prediction_samples,
            "mistake_samples": mistake_samples,
            "confusion_matrix": confusion_matrix,
        }

    return mean_acc


def global_evaluation(model, tokenizer, prompter, dev_data_path, return_details=False, sample_size=5, mistake_sample_size=20):
    with open(dev_data_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    return evaluate_dataset_records(
        model,
        tokenizer,
        prompter,
        test_set,
        dataset_name="Global Evaluation",
        return_details=return_details,
        sample_size=sample_size,
        mistake_sample_size=mistake_sample_size,
    )

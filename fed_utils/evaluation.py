import json
import math
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
LABEL_ABBREVIATIONS = {
    "normal": "N",
    "ball fault": "BF",
    "inner race fault": "IRF",
    "outer race fault": "ORF",
}
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
        "n": "normal",
        "bf": "ball fault",
        "irf": "inner race fault",
        "orf": "outer race fault",
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
    if not rounds:
        return save_path

    max_xticks = min(10, len(rounds))
    if len(rounds) <= max_xticks:
        xticks = rounds
    else:
        step = max(1, math.ceil((len(rounds) - 1) / (max_xticks - 1)))
        xticks = list(range(1, len(rounds) + 1, step))
        if xticks[-1] != rounds[-1]:
            xticks.append(rounds[-1])

    figure_width = max(8, min(14, len(rounds) * 0.18))
    plt.figure(figsize=(figure_width, 5))
    plt.plot(rounds, acc_list, marker="o", linewidth=2)
    plt.title("Global Evaluation Accuracy")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.xticks(xticks)
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


def _get_confusion_matrix_label_order(confusion_matrix):
    labels = [label for label in KNOWN_LABELS if label in confusion_matrix]
    labels.extend([label for label in confusion_matrix if label not in labels])
    return labels


def build_confusion_matrix_payload(confusion_matrix):
    labels = _get_confusion_matrix_label_order(confusion_matrix)
    count_confusion_matrix = {
        true_label: {
            pred_label: int(confusion_matrix.get(true_label, {}).get(pred_label, 0))
            for pred_label in labels
        }
        for true_label in labels
    }
    row_totals = {
        true_label: int(sum(count_confusion_matrix[true_label].values()))
        for true_label in labels
    }
    percentage_confusion_matrix = {}
    for true_label in labels:
        row_total = row_totals[true_label]
        percentage_confusion_matrix[true_label] = {}
        for pred_label in labels:
            count = count_confusion_matrix[true_label][pred_label]
            percentage_confusion_matrix[true_label][pred_label] = (
                float(count * 100.0 / row_total) if row_total else 0.0
            )
    col_totals = {
        pred_label: int(sum(count_confusion_matrix[true_label][pred_label] for true_label in labels))
        for pred_label in labels
    }
    total_samples = int(sum(row_totals.values()))
    correct_samples = int(sum(count_confusion_matrix[label][label] for label in labels))

    per_class_metrics = {}
    for label in labels:
        tp = float(count_confusion_matrix[label][label])
        fp = float(col_totals[label] - tp)
        fn = float(row_totals[label] - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class_metrics[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(row_totals[label]),
        }

    macro_precision = float(np.mean([per_class_metrics[label]["precision"] for label in labels])) if labels else 0.0
    macro_recall = float(np.mean([per_class_metrics[label]["recall"] for label in labels])) if labels else 0.0
    macro_f1 = float(np.mean([per_class_metrics[label]["f1"] for label in labels])) if labels else 0.0
    accuracy = float(correct_samples / total_samples) if total_samples else 0.0

    return {
        "label_order": labels,
        "label_abbreviations": {
            label: LABEL_ABBREVIATIONS.get(label, label) for label in labels
        },
        "count_confusion_matrix": count_confusion_matrix,
        "percentage_confusion_matrix": percentage_confusion_matrix,
        "row_totals": row_totals,
        "col_totals": col_totals,
        "total_samples": total_samples,
        "metrics": {
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
        },
        "per_class_metrics": per_class_metrics,
    }


def save_confusion_matrix(confusion_matrix, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    payload = (
        confusion_matrix
        if isinstance(confusion_matrix, dict) and "count_confusion_matrix" in confusion_matrix
        else build_confusion_matrix_payload(confusion_matrix)
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return save_path


def plot_confusion_matrix_heatmap(confusion_matrix, save_dir, filename="confusion_matrix.png", title="Confusion Matrix"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    payload = (
        confusion_matrix
        if isinstance(confusion_matrix, dict) and "count_confusion_matrix" in confusion_matrix
        else build_confusion_matrix_payload(confusion_matrix)
    )
    labels = payload["label_order"]
    display_labels = [payload["label_abbreviations"].get(label, label) for label in labels]
    count_matrix = np.array(
        [
            [payload["count_confusion_matrix"][true_label][pred_label] for pred_label in labels]
            for true_label in labels
        ],
        dtype=float,
    )
    percentage_matrix = np.array(
        [
            [payload["percentage_confusion_matrix"][true_label][pred_label] for pred_label in labels]
            for true_label in labels
        ],
        dtype=float,
    )

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(percentage_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)

    threshold = percentage_matrix.max() / 2 if percentage_matrix.size and percentage_matrix.max() > 0 else 0
    for i in range(count_matrix.shape[0]):
        for j in range(count_matrix.shape[1]):
            count_value = int(count_matrix[i, j])
            percentage_value = float(percentage_matrix[i, j])
            ax.text(
                j,
                i,
                f"{count_value}\n{percentage_value:.1f}%",
                ha="center",
                va="center",
                color="white" if percentage_value > threshold else "black",
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
        confusion_matrix_payload = build_confusion_matrix_payload(confusion_matrix)
        return {
            "accuracy": mean_acc,
            "per_label_accuracy": acc_count_dict,
            "prediction_samples": prediction_samples,
            "mistake_samples": mistake_samples,
            "confusion_matrix": confusion_matrix_payload,
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

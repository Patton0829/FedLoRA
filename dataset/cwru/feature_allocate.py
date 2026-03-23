import argparse
import json
import os
import random

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISPLAY_LABELS = {
    "ball fault": "BF",
    "inner race fault": "IRF",
    "outer race fault": "ORF",
    "normal": "N",
}
DISPLAY_ORDER = ["normal", "ball fault", "inner race fault", "outer race fault"]
STRICT_DIAGNOSIS_INSTRUCTION = (
    "You are an expert in bearing fault diagnosis. Based on the provided time-domain features, "
    "identify the bearing condition and output only one label from the following options: "
    "inner race fault, normal, outer race fault, ball fault. "
    "Do not output any explanation, punctuation, or extra words."
)


def canonicalize_output_label(label):
    normalized = str(label).strip().lower()
    alias_map = {
        "healthy": "normal",
        "bearing is normal": "normal",
        "ball": "ball fault",
        "ballfault": "ball fault",
        "inner race": "inner race fault",
        "inner ring fault": "inner race fault",
        "outer race": "outer race fault",
        "outer ring fault": "outer race fault",
    }
    return alias_map.get(normalized, normalized)


def standardize_record(record):
    model_input = record.get("input", record.get("context", ""))
    model_output = canonicalize_output_label(record.get("output", record.get("response", "")))
    return {
        "instruction": record.get("instruction", STRICT_DIAGNOSIS_INSTRUCTION),
        "input": model_input,
        "output": model_output,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Allocate CWRU multi-condition data to federated clients under one-condition-per-client constraints."
    )
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients.")
    parser.add_argument(
        "--variant",
        type=str,
        default="feature",
        choices=["raw", "feature", "adaptive"],
        help="Dataset variant. Default: 24-feature dataset.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="condition0,condition1,condition2,condition3",
        help="Comma-separated condition folders under dataset/cwru.",
    )
    parser.add_argument("--train_file", type=str, default=None, help="Custom train json filename for every condition.")
    parser.add_argument("--test_file", type=str, default=None, help="Custom test json filename for every condition.")
    parser.add_argument(
        "--output_root",
        type=str,
        default="data",
        help="Output root dir; files will be saved into <output_root>/<num_clients>/",
    )
    parser.add_argument(
        "--samples_per_client",
        type=int,
        default=1400,
        help="Training samples per client from its assigned condition.",
    )
    parser.add_argument(
        "--test_samples_per_condition",
        type=int,
        default=500,
        help="Number of test samples drawn from each condition to build global_test.json.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def resolve_variant_files(variant):
    mapping = {
        "raw": ("train_data.json", "test_data.json"),
        "feature": ("train_data_feature.json", "test_data_feature.json"),
        "adaptive": ("train_data_adaptive.json", "test_data_adaptive.json"),
    }
    return mapping[variant]


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_condition_records(base_dir, condition, train_file, test_file):
    condition_dir = os.path.join(base_dir, condition)
    train_path = os.path.join(condition_dir, train_file)
    test_path = os.path.join(condition_dir, test_file)
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if not isinstance(train_data, list) or not isinstance(test_data, list):
        raise ValueError(f"Expected list records for {condition} but got invalid json format.")

    train_records = [standardize_record(rec) for rec in train_data]
    test_records = [standardize_record(rec) for rec in test_data]
    return train_records, test_records


def plot_label_distribution(distribution_by_client, save_dir):
    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    observed_labels = {
        label
        for client_dist in distribution_by_client.values()
        for label in client_dist.keys()
    }
    labels = [label for label in DISPLAY_ORDER if label in observed_labels]
    labels.extend(sorted(observed_labels - set(labels)))
    display_labels = [DISPLAY_LABELS.get(label, label) for label in labels]

    num_plots = len(distribution_by_client)
    num_cols = min(5, num_plots)
    num_rows = int(np.ceil(num_plots / num_cols))
    global_max_count = max(max(client_dist.values(), default=0) for client_dist in distribution_by_client.values())
    unified_ymax = max(global_max_count * 1.15, 1)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6.0 * num_cols, 4.9 * num_rows))
    axes = np.array(axes).reshape(-1)
    palette = ["#9ecae1", "#fdae6b", "#a1d99b", "#f4a6a6", "#c7b9ff", "#d9c2a7"]
    colors = palette[: len(labels)] if len(labels) <= len(palette) else plt.cm.Set3(np.linspace(0, 1, len(labels)))

    for client_id, ax in zip(sorted(distribution_by_client.keys()), axes):
        counts = [distribution_by_client[client_id].get(label, 0) for label in labels]
        total = sum(counts)
        bars = ax.bar(range(len(labels)), counts, color=colors[: len(labels)], width=0.65)

        ax.set_facecolor("white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(display_labels, fontsize=15, rotation=0, ha="center")
        ax.set_ylim(0, unified_ymax)
        ax.tick_params(axis="y", labelsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)

        for bar, count in zip(bars, counts):
            if count == 0:
                continue
            percentage = 100.0 * count / total if total else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count}\n{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=14,
            )
        ax.text(0.5, -0.2, f"Client_{client_id}", transform=ax.transAxes, ha="center", va="top", fontsize=16)

    for ax in axes[num_plots:]:
        ax.axis("off")

    fig.suptitle("Label Distribution Across Client Datasets", fontsize=22, y=0.985)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(save_dir, "client_label_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_condition_distribution(distribution_by_client, conditions, save_dir):
    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    num_plots = len(distribution_by_client)
    num_cols = min(5, num_plots)
    num_rows = int(np.ceil(num_plots / num_cols))
    global_max_count = max(max(client_dist.values(), default=0) for client_dist in distribution_by_client.values())
    unified_ymax = max(global_max_count * 1.15, 1)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6.0 * num_cols, 4.9 * num_rows))
    axes = np.array(axes).reshape(-1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

    for client_id, ax in zip(sorted(distribution_by_client.keys()), axes):
        counts = [distribution_by_client[client_id].get(cond, 0) for cond in conditions]
        total = sum(counts)
        bars = ax.bar(range(len(conditions)), counts, color=colors, width=0.65)

        ax.set_facecolor("white")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, fontsize=15, rotation=0, ha="center")
        ax.set_ylim(0, unified_ymax)
        ax.tick_params(axis="y", labelsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.set_axisbelow(True)

        for bar, count in zip(bars, counts):
            if count == 0:
                continue
            percentage = 100.0 * count / total if total else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count}\n{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=14,
            )
        ax.text(0.5, -0.25, f"Client_{client_id}", transform=ax.transAxes, ha="center", va="top", fontsize=16)

    for ax in axes[num_plots:]:
        ax.axis("off")

    fig.suptitle("Condition Distribution Across Client Datasets", fontsize=22, y=0.985)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(save_dir, "client_condition_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    if not conditions:
        raise ValueError("No valid conditions were provided.")
    if args.num_clients < len(conditions):
        raise ValueError(
            f"num_clients={args.num_clients} is smaller than number of conditions={len(conditions)}. "
            "Each condition must have at least one client."
        )

    base_dir = os.path.join("dataset", "cwru")
    default_train_file, default_test_file = resolve_variant_files(args.variant)
    train_file = args.train_file or default_train_file
    test_file = args.test_file or default_test_file

    all_train_records = []
    all_test_records = []
    train_condition_tags = []
    test_condition_tags = []

    for cond in conditions:
        train_records, test_records = load_condition_records(base_dir, cond, train_file, test_file)
        all_train_records.extend(train_records)
        all_test_records.extend(test_records)
        train_condition_tags.extend([cond] * len(train_records))
        test_condition_tags.extend([cond] * len(test_records))
        print(f"Loaded {cond}: train={len(train_records)}, test={len(test_records)}")

    data_path = os.path.join(args.output_root, str(args.num_clients))
    os.makedirs(data_path, exist_ok=True)

    # Reorganize records by condition for constrained allocation.
    train_by_condition = {cond: [] for cond in conditions}
    test_by_condition = {cond: [] for cond in conditions}
    for rec, cond in zip(all_train_records, train_condition_tags):
        train_by_condition[cond].append(rec)
    for rec, cond in zip(all_test_records, test_condition_tags):
        test_by_condition[cond].append(rec)

    # Assign one condition to each client, and guarantee each condition appears at least once.
    client_conditions = list(conditions)
    remaining = args.num_clients - len(conditions)
    for i in range(remaining):
        client_conditions.append(conditions[i % len(conditions)])
    rng.shuffle(client_conditions)

    # Validate per-condition capacity and prepare non-overlapping chunks.
    condition_client_count = {cond: client_conditions.count(cond) for cond in conditions}
    for cond in conditions:
        required = condition_client_count[cond] * args.samples_per_client
        available = len(train_by_condition[cond])
        if required > available:
            raise ValueError(
                f"{cond} needs {required} train samples "
                f"({condition_client_count[cond]} clients x {args.samples_per_client}) but only has {available}."
            )
        rng.shuffle(train_by_condition[cond])

    # Allocate local training files: each client gets exactly samples_per_client from one condition.
    client_label_distribution = {}
    client_condition_distribution = {}
    condition_offset = {cond: 0 for cond in conditions}
    selected_train_records = []
    client_condition_map = {}
    for client_id, cond in enumerate(client_conditions):
        start = condition_offset[cond]
        end = start + args.samples_per_client
        local_records = train_by_condition[cond][start:end]
        condition_offset[cond] = end
        save_json(os.path.join(data_path, f"local_training_{client_id}.json"), local_records)
        selected_train_records.extend(local_records)
        client_condition_map[str(client_id)] = cond

        label_counts = {}
        for rec in local_records:
            label = rec["output"]
            label_counts[label] = label_counts.get(label, 0) + 1
        client_label_distribution[client_id] = label_counts
        client_condition_distribution[client_id] = {cond: len(local_records)}
        print(f"Generated local_training_{client_id}.json with {len(local_records)} samples from {cond}.")

    # Build a compact global test set: fixed samples from each condition.
    global_test_records = []
    for cond in conditions:
        if len(test_by_condition[cond]) < args.test_samples_per_condition:
            raise ValueError(
                f"{cond} test samples are insufficient: need {args.test_samples_per_condition}, got {len(test_by_condition[cond])}."
            )
        rng.shuffle(test_by_condition[cond])
        global_test_records.extend(test_by_condition[cond][: args.test_samples_per_condition])
    rng.shuffle(global_test_records)

    save_json(os.path.join(data_path, "global_training.json"), selected_train_records)
    save_json(os.path.join(data_path, "global_test.json"), global_test_records)
    save_json(os.path.join(data_path, "client_condition_map.json"), client_condition_map)

    metadata = {
        "variant": args.variant,
        "conditions": conditions,
        "num_clients": int(args.num_clients),
        "samples_per_client": int(args.samples_per_client),
        "test_samples_per_condition": int(args.test_samples_per_condition),
        "train_file": train_file,
        "test_file": test_file,
        "total_train_samples": len(selected_train_records),
        "total_test_samples": len(global_test_records),
        "condition_train_counts": {cond: int(train_condition_tags.count(cond)) for cond in conditions},
        "condition_test_counts": {cond: int(test_condition_tags.count(cond)) for cond in conditions},
        "assigned_client_counts_per_condition": condition_client_count,
    }
    save_json(os.path.join(data_path, "allocation_metadata.json"), metadata)

    plot_label_distribution(client_label_distribution, data_path)
    plot_condition_distribution(client_condition_distribution, conditions, data_path)

    print(f"Done. Output dir: {data_path}")
    print(f"Train file used per condition: {train_file}")
    print(f"Test file used per condition: {test_file}")


if __name__ == "__main__":
    main()

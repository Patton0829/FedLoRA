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
    if normalized in alias_map:
        return alias_map[normalized]
    return normalized


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
        description="Allocate CWRU train/test data to federated clients."
    )
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients.")
    parser.add_argument(
        "--diff_quantity",
        type=int,
        default=1,
        help="1 for Dirichlet non-IID split, 0 for shard split.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="condition0",
        help="Condition folder under dataset/cwru, e.g. condition0.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="feature",
        choices=["raw", "feature", "adaptive"],
        help="Dataset variant. Default is 24-feature dataset.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Custom train json filename. Overrides --variant if provided.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Custom test json filename. Overrides --variant if provided.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data",
        help="Output root dir; data will be saved into <output_root>/<num_clients>/",
    )
    parser.add_argument("--dirichlet_alpha", type=float, default=None, help="Dirichlet alpha for non-IID split.")
    parser.add_argument(
        "--train_sample_size",
        type=int,
        default=1400,
        help="Total train samples to keep from the selected condition for fast training.",
    )
    parser.add_argument(
        "--test_sample_size",
        type=int,
        default=None,
        help="Optional total test samples to keep from the selected condition. Default keeps full test set.",
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


def load_json_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}, but got {type(data)}")
    return [standardize_record(record) for record in data]


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def stratified_sample(records, sample_size, seed):
    if sample_size is None or sample_size <= 0 or sample_size >= len(records):
        return list(records)

    rng = random.Random(seed)
    groups = {}
    for rec in records:
        label = rec.get("output", "")
        groups.setdefault(label, []).append(rec)

    labels = sorted(groups.keys())
    for label in labels:
        rng.shuffle(groups[label])

    base = sample_size // max(1, len(labels))
    remainder = sample_size % max(1, len(labels))

    sampled = []
    for i, label in enumerate(labels):
        take = base + (1 if i < remainder else 0)
        take = min(take, len(groups[label]))
        sampled.extend(groups[label][:take])

    if len(sampled) < sample_size:
        leftovers = []
        for label in labels:
            leftovers.extend(groups[label][len([x for x in sampled if x.get("output", "") == label]) :])
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: sample_size - len(sampled)])

    rng.shuffle(sampled)
    return sampled[:sample_size]


def plot_client_distribution(distribution_by_client, save_dir):
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

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5.6 * num_cols, 4.6 * num_rows))
    axes = np.array(axes).reshape(-1)
    palette = ["#9ecae1", "#fdae6b", "#a1d99b", "#f4a6a6", "#c7b9ff", "#d9c2a7"]
    colors = palette[: len(labels)] if len(labels) <= len(palette) else plt.cm.Set3(np.linspace(0, 1, len(labels)))

    for client_id, ax in zip(sorted(distribution_by_client.keys()), axes):
        counts = [distribution_by_client[client_id].get(label, 0) for label in labels]
        total = sum(counts)
        bars = ax.bar(range(len(labels)), counts, color=colors[: len(labels)], width=0.65)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(display_labels, fontsize=12)
        ax.set_ylim(0, unified_ymax)
        ax.tick_params(axis="y", labelsize=12)
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
                fontsize=13,
            )

        ax.text(
            0.5,
            -0.18,
            f"Client_{client_id}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=16,
        )

    for ax in axes[num_plots:]:
        ax.axis("off")

    fig.suptitle("Label Distribution Across Client Datasets", fontsize=22, y=0.975)
    legend_text = "Label abbreviations: N = normal, BF = ball fault, IRF = inner race fault, ORF = outer race fault"
    fig.text(0.5, 0.045, legend_text, ha="center", fontsize=16)
    fig.tight_layout(rect=[0, 0.07, 1, 0.945])

    png_path = os.path.join(save_dir, "client_data_distribution.png")
    pdf_path = os.path.join(save_dir, "client_data_distribution.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved client distribution figure to {png_path}")
    print(f"Saved client distribution figure to {pdf_path}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    base_dir = os.path.join("dataset", "cwru", args.condition)
    default_train_file, default_test_file = resolve_variant_files(args.variant)
    train_file = args.train_file or default_train_file
    test_file = args.test_file or default_test_file
    train_path = os.path.join(base_dir, train_file)
    test_path = os.path.join(base_dir, test_file)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_records = load_json_records(train_path)
    test_records = load_json_records(test_path)
    train_records = stratified_sample(train_records, args.train_sample_size, args.seed)
    test_records = stratified_sample(test_records, args.test_sample_size, args.seed + 1)

    data_path = os.path.join(args.output_root, str(args.num_clients))
    os.makedirs(data_path, exist_ok=True)

    save_json(os.path.join(data_path, "global_training.json"), train_records)
    save_json(os.path.join(data_path, "global_test.json"), test_records)

    labels = [record["output"] for record in train_records]
    categories = sorted(list(set(labels)))
    category_indices = {c: [] for c in categories}
    for idx, label in enumerate(labels):
        category_indices[label].append(idx)

    if args.diff_quantity:
        alpha = args.dirichlet_alpha
        if alpha is None:
            alpha = 5.0 if args.num_clients <= 20 else 10.0
        print(
            f"Using Dirichlet non-IID split with alpha={alpha:.2f}. "
            "Larger alpha means weaker heterogeneity."
        )

        idx_partition = [[] for _ in range(args.num_clients)]
        dataset_size = len(train_records)
        min_require_size = max(1, min(40, dataset_size // max(1, args.num_clients) // 2))
        min_size = 0
        while min_size < min_require_size:
            idx_partition = [[] for _ in range(args.num_clients)]
            for category in categories:
                category_ids = np.array(category_indices[category], dtype=int)
                np.random.shuffle(category_ids)
                proportions = np.random.dirichlet(np.repeat(alpha, args.num_clients))
                proportions = np.array(
                    [p * (len(idx_j) < dataset_size / args.num_clients) for p, idx_j in zip(proportions, idx_partition)]
                )
                proportions = proportions / proportions.sum()
                split_points = (np.cumsum(proportions) * len(category_ids)).astype(int)[:-1]
                splits = np.split(category_ids, split_points)
                idx_partition = [idx_j + idx.tolist() for idx_j, idx in zip(idx_partition, splits)]
            min_size = min(len(idx_j) for idx_j in idx_partition)
            print(f"Current minimum client sample count: {min_size}")
    else:
        num_shards_per_client = 2
        all_indices = np.arange(len(train_records))
        shards = np.array_split(all_indices, int(num_shards_per_client * args.num_clients))
        random.shuffle(shards)
        shards = [shards[i : i + num_shards_per_client] for i in range(0, len(shards), num_shards_per_client)]
        idx_partition = [np.concatenate(shards[n]).tolist() for n in range(args.num_clients)]

    client_distribution = {}
    for client_id, idx in enumerate(idx_partition):
        random.shuffle(idx)
        local_records = [train_records[i] for i in idx]
        save_json(os.path.join(data_path, f"local_training_{client_id}.json"), local_records)

        counts = {}
        for rec in local_records:
            label = rec["output"]
            counts[label] = counts.get(label, 0) + 1
        client_distribution[client_id] = counts

        print(f"Generated local_training_{client_id}.json with {len(local_records)} samples.")

    plot_client_distribution(client_distribution, data_path)
    print(f"Done. Output dir: {data_path}")
    print(f"Train source: {train_path}")
    print(f"Test source: {test_path}")
    print(f"Train samples used: {len(train_records)}")
    print(f"Test samples used: {len(test_records)}")


if __name__ == "__main__":
    main()

import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


num_clients = int(sys.argv[1])
diff_quantity = int(sys.argv[2])
dataset_name = sys.argv[3] if len(sys.argv) > 3 else "jnu"
train_data_path = (
    sys.argv[4]
    if len(sys.argv) > 4
    else os.path.join("dataset", dataset_name, f"{dataset_name}_train.json")
)

seed = 42
np.random.seed(seed)
random.seed(seed)

DISPLAY_LABELS = {
    "ball fault": "BF",
    "inner race fault": "IRF",
    "outer race fault": "ORF",
    "normal": "N",
}

DISPLAY_ORDER = ["normal", "ball fault", "inner race fault", "outer race fault"]


def standardize_record(record):
    instruction = record.get("instruction", "")
    model_input = record.get("input", record.get("context", ""))
    model_output = record.get("output", record.get("response", ""))
    category = record.get("category", model_output)

    return {
        "instruction": instruction,
        "input": model_input,
        "output": model_output,
    }


df = pd.read_json(train_data_path, orient="records")
records = [standardize_record(record) for record in df.to_dict(orient="records")]
df = pd.DataFrame(records)
df["category"] = [record.get("output", "") for record in records]


def remove_category_column(dataframe):
    return dataframe.drop(columns=["category"], errors="ignore")

sorted_df = df.sort_values(by=["category"]).reset_index(drop=True)
grouped = sorted_df.groupby("category", group_keys=False)
sampled_df = grouped.apply(lambda x: x.sample(n=min(10, len(x)), random_state=seed))
sampled_df = sampled_df.reset_index(drop=True)
remaining_df = sorted_df.drop(index=sampled_df.index).reset_index(drop=True)

data_path = os.path.join("data", str(num_clients))
os.makedirs(data_path, exist_ok=True)

with open(os.path.join(data_path, "global_training.json"), "w", encoding="utf-8") as outfile:
    json.dump(
        remove_category_column(remaining_df).to_dict(orient="records"),
        outfile,
        ensure_ascii=False,
        indent=2,
    )

with open(os.path.join(data_path, "global_test.json"), "w", encoding="utf-8") as outfile:
    json.dump(
        remove_category_column(sampled_df).to_dict(orient="records"),
        outfile,
        ensure_ascii=False,
        indent=2,
    )

if diff_quantity:
    min_size = 0
    min_require_size = 40
    alpha = 0.5

    dataset_size = len(remaining_df)
    categories = remaining_df["category"].unique().tolist()

    while min_size < min_require_size:
        idx_partition = [[] for _ in range(num_clients)]
        for category in categories:
            category_rows = remaining_df.loc[remaining_df["category"] == category]
            category_indices = category_rows.index.values
            np.random.shuffle(category_indices)

            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [p * (len(idx_j) < dataset_size / num_clients) for p, idx_j in zip(proportions, idx_partition)]
            )
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(category_indices)).astype(int)[:-1]
            idx_partition = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_partition, np.split(category_indices, split_points))
            ]
            min_size = min(len(idx_j) for idx_j in idx_partition)

        print(min_size)
else:
    num_shards_per_client = 2
    remaining_df_index = remaining_df.index.values
    shards = np.array_split(remaining_df_index, int(num_shards_per_client * num_clients))
    random.shuffle(shards)
    shards = [shards[i : i + num_shards_per_client] for i in range(0, len(shards), num_shards_per_client)]
    idx_partition = [np.concatenate(shards[n]).tolist() for n in range(num_clients)]

client_distribution = {}
for client_id, idx in enumerate(idx_partition):
    print(f"\n Generating the local training dataset of Client_{client_id}")
    sub_remaining_df = remaining_df.loc[idx].reset_index(drop=True)
    sub_remaining_df_records = remove_category_column(sub_remaining_df).to_dict(orient="records")
    print(f"Local sample count of Client_{client_id}: {len(sub_remaining_df_records)}")
    client_distribution[client_id] = sub_remaining_df["category"].value_counts().to_dict()

    with open(
        os.path.join(data_path, f"local_training_{client_id}.json"),
        "w",
        encoding="utf-8",
    ) as outfile:
        json.dump(sub_remaining_df_records, outfile, ensure_ascii=False, indent=2)


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
    global_max_count = max(
        max(client_dist.values(), default=0)
        for client_dist in distribution_by_client.values()
    )
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
                fontsize=14,
            )

        ax.text(
            0.5,
            -0.18,
            f"Client {client_id + 1}",
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
    print(f"\nSaved client distribution figure to {png_path}")
    print(f"Saved client distribution figure to {pdf_path}")


plot_client_distribution(client_distribution, data_path)

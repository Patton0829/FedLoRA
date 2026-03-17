import json
from pathlib import Path


def extract_label(label_item: dict) -> str:
    if "output" in label_item:
        return str(label_item["output"]).strip()
    if "label" in label_item:
        return str(label_item["label"]).strip()
    raise KeyError("Each label item must contain either 'output' or 'label'.")


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    test_path = current_dir / "jnu_test.json"
    labels_path = current_dir / "jnu_labels.json"
    output_path = current_dir / "jnu_test_with_labels.json"

    with test_path.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    with labels_path.open("r", encoding="utf-8") as f:
        labels_data = json.load(f)

    if len(test_data) != len(labels_data):
        raise ValueError(
            f"Length mismatch: {test_path.name} has {len(test_data)} items, "
            f"but {labels_path.name} has {len(labels_data)} items."
        )

    merged_data = []
    for sample, label_item in zip(test_data, labels_data):
        merged_sample = dict(sample)
        merged_sample["output"] = extract_label(label_item)
        merged_data.append(merged_sample)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(merged_data)} labeled test samples to {output_path.name}")


if __name__ == "__main__":
    main()

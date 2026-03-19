import json
from pathlib import Path

SMALL_TEST_SIZE = 100
DIAGNOSIS_INSTRUCTION = (
    "You are an expert in bearing fault diagnosis. Based on the provided time-domain "
    "features, identify the bearing condition and output only one label from the following "
    "options: inner race fault, normal, outer race fault, ball fault. Do not output any "
    "explanation, punctuation, or extra words."
)


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
    small_output_path = current_dir / "jnu_test_with_labels_100.json"

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
        merged_sample["instruction"] = DIAGNOSIS_INSTRUCTION
        merged_sample["output"] = extract_label(label_item)
        merged_data.append(merged_sample)

    if len(merged_data) < SMALL_TEST_SIZE:
        raise ValueError(
            f"Expected at least {SMALL_TEST_SIZE} test samples, but only found {len(merged_data)}."
        )

    small_merged_data = merged_data[:SMALL_TEST_SIZE]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    with small_output_path.open("w", encoding="utf-8") as f:
        json.dump(small_merged_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(merged_data)} labeled test samples to {output_path.name}")
    print(f"Saved {len(small_merged_data)} labeled test samples to {small_output_path.name}")


if __name__ == "__main__":
    main()

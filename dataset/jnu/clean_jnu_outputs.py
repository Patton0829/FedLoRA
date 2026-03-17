import json
import re
from pathlib import Path


PREFIXES = (
    "The diagnosis result is ",
    "Diagnosis result is ",
    "The result is ",
)


def normalize_label(text: str) -> str:
    if text is None:
        return ""

    cleaned = str(text).strip()
    for prefix in PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    cleaned = cleaned.strip()
    cleaned = re.sub(r"[.\s]+$", "", cleaned)
    return cleaned


def process_file(file_path: Path) -> None:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    changed_count = 0
    for item in data:
        old_output = item.get("output", "")
        new_output = normalize_label(old_output)
        if new_output != old_output:
            item["output"] = new_output
            changed_count += 1

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{file_path.name}: updated {changed_count} records")


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    target_files = [
        current_dir / "jnu_train.json",
        current_dir / "jnu_test.json",
    ]

    for file_path in target_files:
        if file_path.exists():
            process_file(file_path)
        else:
            print(f"{file_path.name}: file not found, skipped")


if __name__ == "__main__":
    main()

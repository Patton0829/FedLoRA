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


def main() -> None:
    file_path = Path(__file__).resolve().parent / "jnu_labels.json"

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


if __name__ == "__main__":
    main()

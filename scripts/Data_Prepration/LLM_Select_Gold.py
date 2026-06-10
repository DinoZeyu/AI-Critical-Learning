from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import mimetypes
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DATA_DIR = REPO_ROOT / "Image_Data"
TRAIN_CLEAN_DIR = IMAGE_DATA_DIR / "Train_Clean_Data"
GOLD_DATA_DIR = IMAGE_DATA_DIR / "Gold_Data"
PROMPT_PATH = REPO_ROOT / "Prompt" / "Gold_Data_Pick.txt"
ENV_PATH = REPO_ROOT / ".env"
LABELS_FILENAME = "labels.csv"
BACKUP_LABELS_FILENAME = "labels_before_gold_split.csv"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_GOLD_RATIO = 0.20
DEFAULT_QUALITY_THRESHOLD = 0.93


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use OpenAI vision models to score train-clean image-label pairs and "
            "select a class-balanced, quality-filtered gold dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder under Image_Data/Train_Clean_Data, e.g. STL or Flower_102.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to Image_Data/Gold_Data/<dataset>.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=PROMPT_PATH,
        help="Prompt template with {dataset_name} and {class_list} placeholders.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="OpenAI model name. Defaults to OPENAI_MODEL or gpt-4o-mini.",
    )
    parser.add_argument(
        "--gold-ratio",
        type=float,
        default=DEFAULT_GOLD_RATIO,
        help="Target fraction to keep per class before quality filtering.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLD,
        help="Minimum data_reliance score required for gold data.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Annotate only the first N unprocessed rows. Useful for API smoke tests.",
    )
    parser.add_argument(
        "--select-partial",
        action="store_true",
        help="Allow gold selection from partial annotations, mainly for debugging.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for a failed API call.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing annotations and selected gold output before running.",
    )
    return parser.parse_args()


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.is_file():
        return

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.gold_ratio <= 1.0:
        raise ValueError("--gold-ratio must be in the range (0, 1].")
    if not 0.0 <= args.quality_threshold <= 1.0:
        raise ValueError("--quality-threshold must be in the range [0, 1].")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    if args.sleep < 0:
        raise ValueError("--sleep cannot be negative.")


def read_labels_file(labels_path: Path) -> list[dict[str, str]]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    with labels_path.open(newline="") as labels_file:
        rows = list(csv.DictReader(labels_file))

    if not rows:
        raise ValueError(f"No rows found in {labels_path}")

    required_columns = {"index", "relative_path", "label", "class_name"}
    missing = required_columns - set(rows[0])
    if missing:
        raise ValueError(f"{labels_path} is missing required columns: {sorted(missing)}")

    return rows


def read_source_labels(dataset_dir: Path) -> list[dict[str, str]]:
    backup_path = dataset_dir / BACKUP_LABELS_FILENAME
    labels_path = backup_path if backup_path.is_file() else dataset_dir / LABELS_FILENAME
    return read_labels_file(labels_path)


def format_class_list(rows: list[dict[str, str]]) -> str:
    label_to_name = {int(row["label"]): row["class_name"] for row in rows}
    return "\n".join(
        f"- {label}: {label_to_name[label]}" for label in sorted(label_to_name)
    )


def load_prompt(prompt_path: Path, dataset_name: str, rows: list[dict[str, str]]) -> str:
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")

    prompt_template = prompt_path.read_text()
    return (
        prompt_template.replace("{dataset_name}", dataset_name).replace(
            "{class_list}", format_class_list(rows)
        )
    )


def encode_image_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def parse_model_json(content: str) -> dict[str, Any]:
    parsed = json.loads(content)
    expected_keys = {"data_reliance", "rationale", "alternative_class"}
    if set(parsed) != expected_keys:
        raise ValueError(f"Unexpected response keys: {sorted(parsed)}")

    data_reliance = float(parsed["data_reliance"])
    if not 0.0 <= data_reliance <= 1.0:
        raise ValueError(f"data_reliance out of range: {data_reliance}")

    alternative_class = parsed["alternative_class"]
    if alternative_class == "":
        alternative_class = None

    return {
        "data_reliance": round(data_reliance, 2),
        "rationale": str(parsed["rationale"]),
        "alternative_class": alternative_class,
    }


def annotate_one(
    client: OpenAI,
    model: str,
    prompt: str,
    dataset_dir: Path,
    row: dict[str, str],
) -> dict[str, Any]:
    image_path = dataset_dir / row["relative_path"]
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")

    target_label_text = (
        f"Target label id: {row['label']}\n"
        f"Target label name: {row['class_name']}\n"
        "Score this image-label pair for gold data suitability."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=250,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": target_label_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_data_url(image_path)},
                    },
                ],
            },
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Model response content is empty.")

    annotation = parse_model_json(content)
    return {
        **row,
        "image_path": str(image_path),
        "model": model,
        "data_reliance": f"{annotation['data_reliance']:.2f}",
        "rationale": annotation["rationale"],
        "alternative_class": annotation["alternative_class"],
    }


def annotate_with_retries(
    client: OpenAI,
    model: str,
    prompt: str,
    dataset_dir: Path,
    row: dict[str, str],
    max_retries: int,
) -> dict[str, Any]:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return annotate_one(
                client=client,
                model=model,
                prompt=prompt,
                dataset_dir=dataset_dir,
                row=row,
            )
        except Exception as error:
            last_error = error
            if attempt >= max_retries:
                break
            time.sleep(2**attempt)

    raise RuntimeError(
        f"Failed to annotate {row['relative_path']} after {max_retries + 1} attempts."
    ) from last_error


def load_existing_annotations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}

    annotations = {}
    with path.open() as annotations_file:
        for line_number, line in enumerate(annotations_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "relative_path" not in record:
                raise ValueError(f"Missing relative_path at {path}:{line_number}")
            annotations[record["relative_path"]] = record
    return annotations


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def rows_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    return grouped


def allocate_target_counts(
    rows: list[dict[str, Any]],
    gold_ratio: float,
) -> dict[str, int]:
    grouped = rows_by_label(rows)
    total_gold = round(len(rows) * gold_ratio)
    base_counts = {}
    remainders = []

    for label, class_rows in grouped.items():
        exact_count = len(class_rows) * gold_ratio
        base_count = math.floor(exact_count)
        base_counts[label] = base_count
        remainders.append((exact_count - base_count, label))

    remaining = total_gold - sum(base_counts.values())
    for _, label in sorted(remainders, reverse=True)[:remaining]:
        base_counts[label] += 1

    return base_counts


def sort_gold_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["data_reliance"]),
            row["alternative_class"] is None,
            -int(row["index"]),
        ),
        reverse=True,
    )


def select_gold_rows(
    annotations: list[dict[str, Any]],
    original_rows: list[dict[str, str]],
    gold_ratio: float,
    quality_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_counts = allocate_target_counts(original_rows, gold_ratio=gold_ratio)
    annotated_by_label = rows_by_label(annotations)
    summary_rows = []
    gold_rows = []

    for label in sorted(target_counts, key=lambda value: int(value)):
        class_annotations = annotated_by_label.get(label, [])
        target_count = target_counts[label]
        above_threshold = [
            row
            for row in class_annotations
            if float(row["data_reliance"]) >= quality_threshold
        ]
        selected = sort_gold_candidates(above_threshold)[:target_count]
        gold_rows.extend(selected)

        class_name = selected[0]["class_name"] if selected else None
        if class_name is None and class_annotations:
            class_name = class_annotations[0]["class_name"]

        summary_rows.append(
            {
                "label": label,
                "class_name": class_name,
                "annotated_count": len(class_annotations),
                "target_gold_count": target_count,
                "above_threshold_count": len(above_threshold),
                "selected_count": len(selected),
                "shortfall_count": max(target_count - len(selected), 0),
                "quality_threshold": f"{quality_threshold:.2f}",
            }
        )

    gold_rows = sorted(
        gold_rows,
        key=lambda row: (int(row["label"]), -float(row["data_reliance"]), int(row["index"])),
    )
    return gold_rows, summary_rows


def clean_selected_outputs(output_dir: Path) -> None:
    for output_path in (
        output_dir / LABELS_FILENAME,
        output_dir / "gold_labels.csv",
        output_dir / "selection_summary.csv",
        output_dir / "selection_summary.json",
    ):
        if output_path.exists():
            output_path.unlink()

    images_dir = output_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    write_csv_with_fieldnames(path, rows, list(rows[0].keys()))


def write_csv_with_fieldnames(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_gold_label_rows(
    dataset_dir: Path,
    output_dir: Path,
    gold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for row in gold_rows:
        source_path = dataset_dir / row["relative_path"]
        destination_path = images_dir / row["image_filename"]
        shutil.copy2(source_path, destination_path)

        output_row = dict(row)
        output_row["original_relative_path"] = row["relative_path"]
        output_row["original_image_path"] = str(source_path)
        output_row["relative_path"] = str(destination_path.relative_to(output_dir))
        output_row["image_path"] = str(destination_path)
        output_rows.append(output_row)

    return output_rows


def write_train_labels_without_gold(
    dataset_dir: Path,
    source_rows: list[dict[str, str]],
    gold_rows: list[dict[str, Any]],
) -> None:
    labels_path = dataset_dir / LABELS_FILENAME
    backup_path = dataset_dir / BACKUP_LABELS_FILENAME
    if not backup_path.exists():
        shutil.copy2(labels_path, backup_path)

    gold_relative_paths = {row["original_relative_path"] for row in gold_rows}
    remaining_rows = [
        row for row in source_rows if row["relative_path"] not in gold_relative_paths
    ]
    write_csv_with_fieldnames(labels_path, remaining_rows, list(source_rows[0].keys()))


def write_selection_outputs(
    dataset_dir: Path,
    output_dir: Path,
    source_rows: list[dict[str, str]],
    gold_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    if not gold_rows:
        raise ValueError(
            "No gold rows selected. Try lowering --quality-threshold after reviewing "
            "annotations.jsonl."
        )

    clean_selected_outputs(output_dir)
    gold_label_rows = build_gold_label_rows(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        gold_rows=gold_rows,
    )
    write_csv(output_dir / LABELS_FILENAME, gold_label_rows)
    write_train_labels_without_gold(
        dataset_dir=dataset_dir,
        source_rows=source_rows,
        gold_rows=gold_label_rows,
    )
    write_csv(output_dir / "selection_summary.csv", summary_rows)

    summary = {
        "total_gold_selected": len(gold_label_rows),
        "classes_with_shortfall": [
            row for row in summary_rows if int(row["shortfall_count"]) > 0
        ],
        "per_class": summary_rows,
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    validate_args(args)

    dataset_dir = TRAIN_CLEAN_DIR / args.dataset
    output_dir = args.output_dir or (GOLD_DATA_DIR / args.dataset)
    annotations_path = output_dir / "annotations.jsonl"

    if args.overwrite and annotations_path.exists():
        annotations_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_source_labels(dataset_dir)
    prompt = load_prompt(args.prompt_path, args.dataset, rows)
    existing_annotations = load_existing_annotations(annotations_path)

    rows_to_process = [
        row for row in rows if row["relative_path"] not in existing_annotations
    ]
    if args.limit is not None:
        rows_to_process = rows_to_process[: args.limit]

    if rows_to_process:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or export it in your shell."
            )
        client = OpenAI()

        for row in tqdm(rows_to_process, desc=f"Annotating {args.dataset}"):
            record = annotate_with_retries(
                client=client,
                model=args.model,
                prompt=prompt,
                dataset_dir=dataset_dir,
                row=row,
                max_retries=args.max_retries,
            )
            append_jsonl(annotations_path, record)
            existing_annotations[row["relative_path"]] = record

            if args.sleep > 0:
                time.sleep(args.sleep)

    all_annotations = list(existing_annotations.values())
    expected_count = len(rows)
    if len(all_annotations) < expected_count and not args.select_partial:
        print(
            f"Annotated {len(all_annotations)}/{expected_count} rows. "
            "Run again without --limit to finish before selecting final gold data, "
            "or pass --select-partial for a debug selection.",
            file=sys.stderr,
        )
        return

    gold_rows, summary_rows = select_gold_rows(
        annotations=all_annotations,
        original_rows=rows,
        gold_ratio=args.gold_ratio,
        quality_threshold=args.quality_threshold,
    )
    write_selection_outputs(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        source_rows=rows,
        gold_rows=gold_rows,
        summary_rows=summary_rows,
    )

    print(f"Saved annotations to {annotations_path}")
    print(f"Saved {len(gold_rows)} gold rows to {output_dir / LABELS_FILENAME}")
    print(f"Copied gold images to {output_dir / 'images'}")
    print(f"Updated train labels at {dataset_dir / LABELS_FILENAME}")
    print(f"Original train labels backup: {dataset_dir / BACKUP_LABELS_FILENAME}")
    print(f"Saved selection summary to {output_dir / 'selection_summary.json'}")


if __name__ == "__main__":
    main()

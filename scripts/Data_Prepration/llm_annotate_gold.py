from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DATA_DIR = REPO_ROOT / "Image_Data"
PROMPT_PATH = REPO_ROOT / "Prompt" / "Gold_Data_Pick.txt"
MODEL = "gpt-4o-mini"
GOLD_RATIO = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use GPT-4o mini to score train-clean images and select gold data."
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
        "--limit",
        type=int,
        help="Annotate only the first N unprocessed rows. Useful for testing.",
    )
    parser.add_argument(
        "--gold-ratio",
        type=float,
        default=GOLD_RATIO,
        help="Fraction of train-clean images to select as gold data.",
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
        help="Delete existing annotations and gold output before running.",
    )
    return parser.parse_args()


def read_labels(dataset_dir: Path) -> list[dict[str, str]]:
    labels_path = dataset_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    with labels_path.open(newline="") as labels_file:
        rows = list(csv.DictReader(labels_file))

    required_columns = {"index", "relative_path", "label", "class_name"}
    missing = required_columns - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{labels_path} is missing required columns: {sorted(missing)}")

    return rows


def format_class_list(rows: list[dict[str, str]]) -> str:
    label_to_name = {
        int(row["label"]): row["class_name"]
        for row in rows
    }
    return "\n".join(
        f"- {label}: {label_to_name[label]}"
        for label in sorted(label_to_name)
    )


def load_prompt(prompt_path: Path, dataset_name: str, rows: list[dict[str, str]]) -> str:
    prompt_template = prompt_path.read_text()
    return (
        prompt_template
        .replace("{dataset_name}", dataset_name)
        .replace("{class_list}", format_class_list(rows))
    )


def encode_image_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"

    image_bytes = image_path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def parse_model_json(content: str) -> dict[str, Any]:
    parsed = json.loads(content)

    if set(parsed) != {"data_reliance", "rationale", "alternative_class"}:
        raise ValueError(f"Unexpected response keys: {sorted(parsed)}")

    data_reliance = float(parsed["data_reliance"])
    if not 0.0 <= data_reliance <= 1.0:
        raise ValueError(f"data_reliance out of range: {data_reliance}")

    parsed["data_reliance"] = round(data_reliance, 2)
    if parsed["alternative_class"] == "":
        parsed["alternative_class"] = None

    return parsed


def annotate_one(
    client: OpenAI,
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
        model=MODEL,
        temperature=0,
        max_tokens=250,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
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
        "model": MODEL,
        "data_reliance": f"{annotation['data_reliance']:.2f}",
        "rationale": annotation["rationale"],
        "alternative_class": annotation["alternative_class"],
    }


def annotate_with_retries(
    client: OpenAI,
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
                prompt=prompt,
                dataset_dir=dataset_dir,
                row=row,
            )
        except Exception as error:
            last_error = error
            if attempt >= max_retries:
                break
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Failed to annotate {row['relative_path']} after {max_retries + 1} attempts."
    ) from last_error


def load_existing_annotations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}

    annotations = {}
    with path.open() as annotations_file:
        for line in annotations_file:
            if not line.strip():
                continue
            record = json.loads(line)
            annotations[record["relative_path"]] = record
    return annotations


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def allocate_gold_counts(rows: list[dict[str, Any]], gold_ratio: float) -> dict[str, int]:
    rows_by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_class.setdefault(row["label"], []).append(row)

    total_gold = round(len(rows) * gold_ratio)
    base_counts = {}
    remainders = []

    for label, class_rows in rows_by_class.items():
        exact_count = len(class_rows) * gold_ratio
        base_count = math.floor(exact_count)
        base_counts[label] = base_count
        remainders.append((exact_count - base_count, label))

    remaining = total_gold - sum(base_counts.values())
    for _, label in sorted(remainders, reverse=True)[:remaining]:
        base_counts[label] += 1

    return base_counts


def write_gold_csv(
    annotations: list[dict[str, Any]],
    output_path: Path,
    gold_ratio: float,
) -> list[dict[str, Any]]:
    counts_by_class = allocate_gold_counts(annotations, gold_ratio=gold_ratio)
    rows_by_class: dict[str, list[dict[str, Any]]] = {}

    for row in annotations:
        rows_by_class.setdefault(row["label"], []).append(row)

    gold_rows = []
    for label, class_rows in rows_by_class.items():
        sorted_rows = sorted(
            class_rows,
            key=lambda row: (
                float(row["data_reliance"]),
                row["alternative_class"] is None,
            ),
            reverse=True,
        )
        gold_rows.extend(sorted_rows[:counts_by_class[label]])

    gold_rows = sorted(
        gold_rows,
        key=lambda row: (row["label"], -float(row["data_reliance"]), int(row["index"])),
    )

    if not gold_rows:
        raise ValueError("No gold rows selected.")

    fieldnames = list(gold_rows[0].keys())
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gold_rows)

    return gold_rows


def main() -> None:
    args = parse_args()

    dataset_dir = IMAGE_DATA_DIR / "Train_Clean_Data" / args.dataset
    output_dir = args.output_dir or (IMAGE_DATA_DIR / "Gold_Data" / args.dataset)
    annotations_path = output_dir / "annotations.jsonl"
    gold_csv_path = output_dir / "gold_labels.csv"

    if args.overwrite and output_dir.exists():
        for output_path in (annotations_path, gold_csv_path):
            if output_path.exists():
                output_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_labels(dataset_dir)
    prompt = load_prompt(args.prompt_path, args.dataset, rows)
    existing_annotations = load_existing_annotations(annotations_path)

    rows_to_process = [
        row for row in rows
        if row["relative_path"] not in existing_annotations
    ]
    if args.limit is not None:
        rows_to_process = rows_to_process[:args.limit]

    client = None
    if rows_to_process:
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI()

    for row in tqdm(rows_to_process, desc=f"Annotating {args.dataset}"):
        if client is None:
            raise RuntimeError("OpenAI client was not initialized.")

        record = annotate_with_retries(
            client=client,
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
    if len(all_annotations) < expected_count:
        print(
            f"Annotated {len(all_annotations)}/{expected_count} rows. "
            f"Run again without --limit to finish before selecting final gold data.",
            file=sys.stderr,
        )
        return

    gold_rows = write_gold_csv(
        annotations=all_annotations,
        output_path=gold_csv_path,
        gold_ratio=args.gold_ratio,
    )

    print(f"Saved all annotations to {annotations_path}")
    print(f"Saved {len(gold_rows)} gold rows to {gold_csv_path}")


if __name__ == "__main__":
    main()

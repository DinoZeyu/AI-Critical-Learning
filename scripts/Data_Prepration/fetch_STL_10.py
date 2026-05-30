from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from torchvision.datasets import STL10


STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DATA_DIR = REPO_ROOT / "Image_Data"
RAW_CACHE_DIR = IMAGE_DATA_DIR / "_stl10_raw"


def load_labeled_stl10() -> list[dict]:
    samples = []

    for source_split in ("train", "test"):
        dataset = STL10(
            root=str(RAW_CACHE_DIR),
            split=source_split,
            download=True,
        )

        for source_index, (image, label) in enumerate(dataset):
            samples.append(
                {
                    "image": image,
                    "label": label,
                    "class_name": STL10_CLASSES[label],
                    "source_split": source_split,
                    "source_index": source_index,
                }
            )

    return samples


def make_balanced_split(
    samples: list[dict],
    train_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if train_size % len(STL10_CLASSES) != 0:
        raise ValueError("train_size must be divisible by the number of classes.")

    train_per_class = train_size // len(STL10_CLASSES)
    rng = random.Random(seed)
    samples_by_label = defaultdict(list)

    for sample in samples:
        samples_by_label[sample["label"]].append(sample)

    train_samples = []
    test_samples = []

    for label in range(len(STL10_CLASSES)):
        class_samples = samples_by_label[label]
        rng.shuffle(class_samples)

        if len(class_samples) < train_per_class:
            raise ValueError(
                f"Class {label} only has {len(class_samples)} samples, "
                f"but train split needs {train_per_class}."
            )

        train_samples.extend(class_samples[:train_per_class])
        test_samples.extend(class_samples[train_per_class:])

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    return train_samples, test_samples


def export_split(samples: list[dict], split: str, output_dir: Path) -> None:
    stl_dir = output_dir / "STL"
    images_dir = stl_dir / "images"
    labels_path = stl_dir / "labels.csv"

    images_dir.mkdir(parents=True, exist_ok=True)

    with labels_path.open("w", newline="") as labels_file:
        writer = csv.DictWriter(
            labels_file,
            fieldnames=[
                "index",
                "split",
                "image_filename",
                "relative_path",
                "label",
                "class_name",
                "source_split",
                "source_index",
            ],
        )
        writer.writeheader()

        for index, sample in enumerate(samples):
            image_filename = f"{split}_{index:05d}.png"
            image_path = images_dir / image_filename

            sample["image"].save(image_path)

            writer.writerow(
                {
                    "index": index,
                    "split": split,
                    "image_filename": image_filename,
                    "relative_path": image_path.relative_to(stl_dir),
                    "label": sample["label"],
                    "class_name": sample["class_name"],
                    "source_split": sample["source_split"],
                    "source_index": sample["source_index"],
                }
            )

    print(f"Saved {len(samples)} {split} images to {images_dir}")
    print(f"Saved labels to {labels_path}")


def recreate_output_dirs() -> None:
    for output_dir in (
        IMAGE_DATA_DIR / "Train_Clean_Data" / "STL",
        IMAGE_DATA_DIR / "Test_Clean_Data" / "STL",
    ):
        if output_dir.exists():
            shutil.rmtree(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download STL-10 labeled data and export a balanced 80/20 split."
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=10400,
        help="Number of labeled images to place in Train_Clean_Data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible balanced splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    recreate_output_dirs()

    samples = load_labeled_stl10()
    train_samples, test_samples = make_balanced_split(
        samples=samples,
        train_size=args.train_size,
        seed=args.seed,
    )

    export_split(
        samples=train_samples,
        split="train",
        output_dir=IMAGE_DATA_DIR / "Train_Clean_Data",
    )
    export_split(
        samples=test_samples,
        split="test",
        output_dir=IMAGE_DATA_DIR / "Test_Clean_Data",
    )


if __name__ == "__main__":
    main()

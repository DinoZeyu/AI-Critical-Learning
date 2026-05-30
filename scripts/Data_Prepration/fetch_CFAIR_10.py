from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from torchvision.datasets import CIFAR10


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DATA_DIR = REPO_ROOT / "Image_Data"
RAW_CACHE_DIR = IMAGE_DATA_DIR / "_cifar_10_raw"
DATASET_DIR_NAME = "CIFAR10_STL_Shared"
DEFAULT_SAMPLES_PER_CLASS = 260


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


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


CIFAR_TO_STL = {
    0: 0,  # airplane -> airplane
    1: 2,  # automobile -> car
    2: 1,  # bird -> bird
    3: 3,  # cat -> cat
    4: 4,  # deer -> deer
    5: 5,  # dog -> dog
    7: 6,  # horse -> horse
    8: 8,  # ship -> ship
    9: 9,  # truck -> truck
}


def load_cifar10_test_shared_classes() -> list[dict]:
    dataset = CIFAR10(
        root=str(RAW_CACHE_DIR),
        train=False,
        download=True,
    )

    samples = []
    for source_index, (image, source_label) in enumerate(dataset):
        if source_label not in CIFAR_TO_STL:
            continue

        stl_label = CIFAR_TO_STL[source_label]
        samples.append(
            {
                "image": image,
                "label": stl_label,
                "class_name": STL10_CLASSES[stl_label],
                "source_dataset": "CIFAR10",
                "source_split": "test",
                "source_index": source_index,
                "source_label": source_label,
                "source_class_name": CIFAR10_CLASSES[source_label],
            }
        )

    return samples


def make_balanced_subset(
    samples: list[dict],
    samples_per_class: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    samples_by_label = defaultdict(list)

    for sample in samples:
        samples_by_label[sample["label"]].append(sample)

    selected_samples = []
    for label in sorted(samples_by_label):
        class_samples = samples_by_label[label]
        rng.shuffle(class_samples)

        if len(class_samples) < samples_per_class:
            raise ValueError(
                f"Class {label} only has {len(class_samples)} samples, "
                f"but {samples_per_class} are required."
            )

        selected_samples.extend(class_samples[:samples_per_class])

    rng.shuffle(selected_samples)
    return selected_samples


def export_test_noise(samples: list[dict]) -> None:
    dataset_dir = IMAGE_DATA_DIR / "Test_Noise_Data" / DATASET_DIR_NAME
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / "labels.csv"

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

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
                "source_dataset",
                "source_split",
                "source_index",
                "source_label",
                "source_class_name",
            ],
        )
        writer.writeheader()

        for index, sample in enumerate(samples):
            image_filename = f"test_noise_{index:05d}.png"
            image_path = images_dir / image_filename
            sample["image"].save(image_path)

            writer.writerow(
                {
                    "index": index,
                    "split": "test_noise",
                    "image_filename": image_filename,
                    "relative_path": image_path.relative_to(dataset_dir),
                    "label": sample["label"],
                    "class_name": sample["class_name"],
                    "source_dataset": sample["source_dataset"],
                    "source_split": sample["source_split"],
                    "source_index": sample["source_index"],
                    "source_label": sample["source_label"],
                    "source_class_name": sample["source_class_name"],
                }
            )

    print(f"Saved {len(samples)} CIFAR10/STL shared test-noise images to {images_dir}")
    print(f"Saved labels to {labels_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download CIFAR-10 test images and export the 9 classes shared with "
            "STL-10 as an extra robustness test set."
        )
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help="Number of CIFAR-10 test images to keep for each shared STL-10 class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible per-class sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = load_cifar10_test_shared_classes()
    selected_samples = make_balanced_subset(
        samples=samples,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    export_test_noise(selected_samples)


if __name__ == "__main__":
    main()

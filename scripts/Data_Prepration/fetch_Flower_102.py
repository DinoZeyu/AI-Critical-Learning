from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from torchvision.datasets import Flowers102


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DATA_DIR = REPO_ROOT / "Image_Data"
RAW_CACHE_DIR = IMAGE_DATA_DIR / "_flower_102_raw"
DATASET_DIR_NAME = "Flower_102"
FLOWER_102_NUM_CLASSES = 102
FLOWER_102_CLASSES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]


def validate_class_mapping() -> None:
    if len(FLOWER_102_CLASSES) != FLOWER_102_NUM_CLASSES:
        raise ValueError(
            f"Flower-102 class mapping has {len(FLOWER_102_CLASSES)} names, "
            f"expected {FLOWER_102_NUM_CLASSES}."
        )
    if len(set(FLOWER_102_CLASSES)) != FLOWER_102_NUM_CLASSES:
        raise ValueError("Flower-102 class mapping contains duplicate names.")


def load_labeled_flowers102() -> list[dict]:
    validate_class_mapping()
    samples = []

    for source_split in ("train", "val", "test"):
        dataset = Flowers102(
            root=str(RAW_CACHE_DIR),
            split=source_split,
            download=True,
        )

        for source_index, (source_path, label) in enumerate(
            zip(dataset._image_files, dataset._labels, strict=True)
        ):
            if not 0 <= label < FLOWER_102_NUM_CLASSES:
                raise ValueError(f"Unexpected Flower-102 label: {label}")

            samples.append(
                {
                    "source_path": Path(source_path),
                    "label": label,
                    "class_name": FLOWER_102_CLASSES[label],
                    "source_split": source_split,
                    "source_index": source_index,
                    "source_filename": Path(source_path).name,
                }
            )

    return samples


def make_stratified_split(
    samples: list[dict],
    train_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0 < train_size < len(samples):
        raise ValueError("train_size must be between 1 and total sample count - 1.")

    rng = random.Random(seed)
    samples_by_label = defaultdict(list)

    for sample in samples:
        samples_by_label[sample["label"]].append(sample)

    if len(samples_by_label) != FLOWER_102_NUM_CLASSES:
        raise ValueError(
            f"Expected {FLOWER_102_NUM_CLASSES} classes, found {len(samples_by_label)}."
        )

    total_size = len(samples)
    train_counts = {}
    remainders = []

    for label, class_samples in samples_by_label.items():
        exact_count = len(class_samples) * train_size / total_size
        base_count = int(exact_count)
        train_counts[label] = base_count
        remainders.append((exact_count - base_count, label))

    remaining = train_size - sum(train_counts.values())
    for _, label in sorted(remainders, reverse=True)[:remaining]:
        train_counts[label] += 1

    train_samples = []
    test_samples = []

    for label in sorted(samples_by_label):
        class_samples = samples_by_label[label]
        rng.shuffle(class_samples)

        train_count = train_counts[label]
        train_samples.extend(class_samples[:train_count])
        test_samples.extend(class_samples[train_count:])

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    return train_samples, test_samples


def export_split(samples: list[dict], split: str, output_dir: Path) -> None:
    dataset_dir = output_dir / DATASET_DIR_NAME
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / "labels.csv"

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
                "source_filename",
            ],
        )
        writer.writeheader()

        for index, sample in enumerate(samples):
            image_filename = f"{split}_{index:05d}.jpg"
            image_path = images_dir / image_filename
            shutil.copy2(sample["source_path"], image_path)

            writer.writerow(
                {
                    "index": index,
                    "split": split,
                    "image_filename": image_filename,
                    "relative_path": image_path.relative_to(dataset_dir),
                    "label": sample["label"],
                    "class_name": sample["class_name"],
                    "source_split": sample["source_split"],
                    "source_index": sample["source_index"],
                    "source_filename": sample["source_filename"],
                }
            )

    print(f"Saved {len(samples)} {split} images to {images_dir}")
    print(f"Saved labels to {labels_path}")


def recreate_output_dirs() -> None:
    for output_dir in (
        IMAGE_DATA_DIR / "Train_Clean_Data" / DATASET_DIR_NAME,
        IMAGE_DATA_DIR / "Test_Clean_Data" / DATASET_DIR_NAME,
    ):
        if output_dir.exists():
            shutil.rmtree(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Flower-102 and export a stratified 6552/1637 split."
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=6552,
        help="Number of labeled images to place in Train_Clean_Data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = load_labeled_flowers102()
    train_samples, test_samples = make_stratified_split(
        samples=samples,
        train_size=args.train_size,
        seed=args.seed,
    )

    recreate_output_dirs()

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

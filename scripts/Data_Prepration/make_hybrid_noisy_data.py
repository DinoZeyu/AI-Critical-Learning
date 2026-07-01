from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cl.data.noise import FEATURE_NOISE_TYPES, make_hybrid_noisy_dataset


IMAGE_DATA_DIR = REPO_ROOT / "Image_Data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create hybrid noisy image datasets by applying feature noise to "
            "images and label_shuffle noise to labels."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder name, e.g. STL or Flower_102.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Clean split to read and noisy split to write. Defaults to train.",
    )
    parser.add_argument(
        "--feature-noise-type",
        choices=FEATURE_NOISE_TYPES,
        required=True,
        help="Feature noise transform to apply to every image.",
    )
    parser.add_argument(
        "--feature-value",
        type=float,
        required=True,
        help="Feature noise strength. blur: radius, brightness: factor, gaussian: pixel sigma.",
    )
    parser.add_argument(
        "--label-shuffle-fraction",
        type=float,
        default=0.2,
        help="Fraction of labels to randomly replace. Defaults to 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by stochastic feature noise and label shuffle.",
    )
    parser.add_argument(
        "--label-seed",
        type=int,
        help="Optional separate seed for label shuffle. Defaults to --seed.",
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Optional output leaf folder name. Defaults to "
            "<feature-noise>_<feature-value>_label_shuffle_<fraction>."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output dataset folder before writing.",
    )
    return parser.parse_args()


def format_value(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "neg")


def main() -> None:
    args = parse_args()

    clean_root = IMAGE_DATA_DIR / f"{args.split.capitalize()}_Clean_Data"
    noise_root = IMAGE_DATA_DIR / f"{args.split.capitalize()}_Noise_Data"
    input_dataset_dir = clean_root / args.dataset

    output_name = args.output_name
    if output_name is None:
        output_name = (
            f"{args.feature_noise_type}_{format_value(args.feature_value)}"
            f"_label_shuffle_{format_value(args.label_shuffle_fraction)}"
        )

    output_dataset_dir = noise_root / args.dataset / "hybrid_noise" / output_name

    if args.overwrite and output_dataset_dir.exists():
        shutil.rmtree(output_dataset_dir)
    if output_dataset_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dataset_dir}. Use --overwrite to replace it."
        )

    make_hybrid_noisy_dataset(
        input_dataset_dir=input_dataset_dir,
        output_dataset_dir=output_dataset_dir,
        feature_noise_type=args.feature_noise_type,
        feature_noise_value=args.feature_value,
        label_shuffle_fraction=args.label_shuffle_fraction,
        seed=args.seed,
        label_seed=args.label_seed,
    )

    print(f"Saved hybrid noisy dataset to {output_dataset_dir}")


if __name__ == "__main__":
    main()

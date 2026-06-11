from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cl.data.noise import SUPPORTED_NOISE_TYPES, make_noisy_dataset


IMAGE_DATA_DIR = REPO_ROOT / "Image_Data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create synthetic noisy image datasets from clean image datasets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder name, e.g. STL or Flower_102.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        required=True,
        help="Clean split to read and noisy split to write.",
    )
    parser.add_argument(
        "--noise-type",
        choices=SUPPORTED_NOISE_TYPES,
        required=True,
        help="Noise transform to apply.",
    )
    parser.add_argument(
        "--value",
        type=float,
        required=True,
        help=(
            "Noise strength. blur: radius, brightness: factor, "
            "gaussian: pixel sigma, label_shuffle: fraction of labels to change."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by stochastic noise transforms.",
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Optional output leaf folder name. Defaults to "
            "<noise-type>_<value> under <dataset>/feature_noise or "
            "<dataset>/label_noise."
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


def noise_group(noise_type: str) -> str:
    if noise_type == "label_shuffle":
        return "label_noise"
    return "feature_noise"


def main() -> None:
    args = parse_args()

    clean_root = IMAGE_DATA_DIR / f"{args.split.capitalize()}_Clean_Data"
    noise_root = IMAGE_DATA_DIR / f"{args.split.capitalize()}_Noise_Data"
    input_dataset_dir = clean_root / args.dataset

    output_name = args.output_name
    if output_name is None:
        output_name = f"{args.noise_type}_{format_value(args.value)}"

    output_dataset_dir = noise_root / args.dataset / noise_group(args.noise_type) / output_name

    if args.overwrite and output_dataset_dir.exists():
        shutil.rmtree(output_dataset_dir)
    if output_dataset_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dataset_dir}. Use --overwrite to replace it."
        )

    make_noisy_dataset(
        input_dataset_dir=input_dataset_dir,
        output_dataset_dir=output_dataset_dir,
        noise_type=args.noise_type,
        value=args.value,
        seed=args.seed,
    )

    print(f"Saved noisy dataset to {output_dataset_dir}")


if __name__ == "__main__":
    main()

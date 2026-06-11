from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


SUPPORTED_NOISE_TYPES = ("blur", "brightness", "gaussian", "label_shuffle")


def apply_blur(image: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur to a PIL image."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Adjust image brightness. factor < 1 darkens, factor > 1 brightens."""
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_gaussian_noise(
    image: Image.Image,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Add pixel-space Gaussian noise to a PIL image."""
    if rng is None:
        rng = np.random.default_rng()

    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=sigma, size=image_array.shape)
    noisy_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array, mode="RGB")


def apply_noise(
    image: Image.Image,
    noise_type: str,
    value: float,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    if noise_type == "blur":
        return apply_blur(image, radius=value)
    if noise_type == "brightness":
        return apply_brightness(image, factor=value)
    if noise_type == "gaussian":
        return apply_gaussian_noise(image, sigma=value, rng=rng)

    supported = ", ".join(SUPPORTED_NOISE_TYPES)
    raise ValueError(f"Unsupported noise_type '{noise_type}'. Choose from: {supported}.")


def shuffle_label_rows(
    rows: list[dict[str, str]],
    shuffle_fraction: float,
    seed: int,
) -> list[dict[str, str]]:
    if not 0.0 <= shuffle_fraction <= 1.0:
        raise ValueError("label_shuffle value must be in the range [0, 1].")
    if not rows:
        return rows

    rng = np.random.default_rng(seed)
    labels = sorted({row["label"] for row in rows})
    label_to_names: dict[str, list[str]] = {}
    for row in rows:
        label_to_names.setdefault(row["label"], [])
        if row["class_name"] not in label_to_names[row["label"]]:
            label_to_names[row["label"]].append(row["class_name"])

    if len(labels) < 2:
        raise ValueError("label_shuffle requires at least two classes.")

    noisy_rows = [dict(row) for row in rows]
    num_to_shuffle = round(len(noisy_rows) * shuffle_fraction)
    selected_indices = set(
        rng.choice(len(noisy_rows), size=num_to_shuffle, replace=False).tolist()
    )

    for index, row in enumerate(noisy_rows):
        row["original_label"] = row["label"]
        row["original_class_name"] = row["class_name"]
        row["label_noise_applied"] = int(index in selected_indices)
        if index not in selected_indices:
            continue

        current_label = row["label"]
        candidate_labels = [label for label in labels if label != current_label]
        new_label = str(rng.choice(candidate_labels))
        row["label"] = new_label
        row["class_name"] = label_to_names[new_label][0]

    return noisy_rows


def make_noisy_dataset(
    input_dataset_dir: Path | str,
    output_dataset_dir: Path | str,
    noise_type: str,
    value: float,
    seed: int = 42,
) -> None:
    """Create a noisy copy of an image dataset that has images/ and labels.csv.

    The input labels.csv must include a relative_path column. The output keeps all
    original metadata columns and appends noise_type/noise_value.
    """
    input_dataset_dir = Path(input_dataset_dir)
    output_dataset_dir = Path(output_dataset_dir)
    input_labels_path = input_dataset_dir / "labels.csv"
    output_labels_path = output_dataset_dir / "labels.csv"

    if not input_labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {input_labels_path}")

    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    with input_labels_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"labels.csv has no header: {input_labels_path}")
        if "relative_path" not in reader.fieldnames:
            raise ValueError(f"labels.csv must include relative_path: {input_labels_path}")

        fieldnames = list(reader.fieldnames)
        for extra_field in ("noise_type", "noise_value"):
            if extra_field not in fieldnames:
                fieldnames.append(extra_field)
        rows = list(reader)

        if noise_type == "label_shuffle":
            for extra_field in (
                "original_label",
                "original_class_name",
                "label_noise_applied",
            ):
                if extra_field not in fieldnames:
                    fieldnames.append(extra_field)
            rows = shuffle_label_rows(
                rows=rows,
                shuffle_fraction=value,
                seed=seed,
            )

        with output_labels_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                source_image_path = input_dataset_dir / row["relative_path"]
                if not source_image_path.is_file():
                    raise FileNotFoundError(f"Missing source image: {source_image_path}")

                output_relative_path = Path(row["relative_path"])
                output_image_path = output_dataset_dir / output_relative_path
                output_image_path.parent.mkdir(parents=True, exist_ok=True)

                if noise_type == "label_shuffle":
                    shutil.copy2(source_image_path, output_image_path)
                else:
                    with Image.open(source_image_path) as image:
                        noisy_image = apply_noise(
                            image=image.convert("RGB"),
                            noise_type=noise_type,
                            value=value,
                            rng=rng,
                        )
                        noisy_image.save(output_image_path)

                row["noise_type"] = noise_type
                row["noise_value"] = value
                writer.writerow(row)

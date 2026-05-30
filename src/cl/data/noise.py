from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# In order to apply feature noise for the train and test datasetm, we used 3 different nose strengths for each noise type. 
# For blur, we used a radius of 1, 2, and 3. For brightness, we used factors of 0.5, 0.75, and 1.25. For Gaussian noise, we used standard deviations of 10, 20, and 30.
SUPPORTED_NOISE_TYPES = ("blur", "brightness", "gaussian")


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
    output_images_dir = output_dataset_dir / "images"
    output_labels_path = output_dataset_dir / "labels.csv"

    if not input_labels_path.is_file():
        raise FileNotFoundError(f"Missing labels file: {input_labels_path}")

    output_images_dir.mkdir(parents=True, exist_ok=True)
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

        with output_labels_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                source_image_path = input_dataset_dir / row["relative_path"]
                if not source_image_path.is_file():
                    raise FileNotFoundError(f"Missing source image: {source_image_path}")

                output_relative_path = Path(row["relative_path"])
                output_image_path = output_dataset_dir / output_relative_path
                output_image_path.parent.mkdir(parents=True, exist_ok=True)

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

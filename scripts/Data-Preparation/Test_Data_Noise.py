import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def resolve_default_paths(args_out_dir: str | None) -> Path:
    cl_data_raw = os.environ.get("CL_DATA_RAW")
    if args_out_dir is None and not cl_data_raw:
        raise RuntimeError(
            "CL_DATA_RAW is not set. Please run: source ./env_superpod.sh\n"
            "Or pass --out_dir explicitly."
        )

    out_dir = Path(args_out_dir).resolve() if args_out_dir else (Path(cl_data_raw) / "cifar10" / "v1_seed42")
    return out_dir


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def apply_blur(img: Image.Image, blur_radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def apply_gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    arr_noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_noisy)


def apply_brightness(img: Image.Image, brightness_factor: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(brightness_factor)


def build_suffix(corruption: str, severity: float) -> str:
    if corruption == "blur":
        return f"blur_r{severity:g}"
    if corruption == "gaussian_noise":
        return f"gaussian_noise_s{severity:g}"
    if corruption == "brightness":
        return f"brightness_f{severity:g}"
    raise ValueError(f"Unsupported corruption: {corruption}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=None,
                    help="Dataset root. Default: $CL_DATA_RAW/cifar10/v1_seed42")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--corruption", type=str, required=True,
                    choices=["blur", "gaussian_noise", "brightness"],
                    help="Type of corruption to generate")

    ap.add_argument("--blur_radius", type=float, default=1.2,
                    help="Gaussian blur radius, used when corruption=blur")
    ap.add_argument("--noise_sigma", type=float, default=12.0,
                    help="Gaussian noise sigma in pixel space, used when corruption=gaussian_noise")
    ap.add_argument("--brightness_factor", type=float, default=0.65,
                    help="Brightness factor, used when corruption=brightness. <1 means darker")

    args = ap.parse_args()

    out_dir = resolve_default_paths(args.out_dir)
    ann_dir = out_dir / "annotations"
    src_jsonl = ann_dir / "test_index_map.jsonl"

    if not src_jsonl.exists():
        raise FileNotFoundError(f"Missing source test jsonl: {src_jsonl}")

    rows = read_jsonl(src_jsonl)
    if len(rows) == 0:
        raise ValueError(f"Empty jsonl: {src_jsonl}")

    rng = np.random.default_rng(args.seed)

    if args.corruption == "blur":
        severity = args.blur_radius
    elif args.corruption == "gaussian_noise":
        severity = args.noise_sigma
    elif args.corruption == "brightness":
        severity = args.brightness_factor
    else:
        raise ValueError(f"Unsupported corruption: {args.corruption}")

    suffix = build_suffix(args.corruption, severity)
    dst_img_dir_name = f"test_images_{suffix}"
    dst_img_dir = out_dir / dst_img_dir_name
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PATH] out_dir = {out_dir}")
    print(f"[PATH] src_jsonl = {src_jsonl}")
    print(f"[MODE] corruption = {args.corruption}")
    print(f"[MODE] suffix = {suffix}")
    print(f"[PATH] dst_img_dir = {dst_img_dir}")

    dst_rows: List[Dict] = []

    for r in rows:
        src_relpath = r["relpath"]               # e.g. test_images/000000.png
        filename = Path(src_relpath).name        # e.g. 000000.png
        src_img_path = out_dir / src_relpath
        dst_relpath = f"{dst_img_dir_name}/{filename}"
        dst_img_path = out_dir / dst_relpath

        if not src_img_path.exists():
            raise FileNotFoundError(f"Missing source image: {src_img_path}")

        img = Image.open(src_img_path).convert("RGB")

        if args.corruption == "blur":
            img_out = apply_blur(img, blur_radius=args.blur_radius)
        elif args.corruption == "gaussian_noise":
            img_out = apply_gaussian_noise(img, sigma=args.noise_sigma, rng=rng)
        elif args.corruption == "brightness":
            img_out = apply_brightness(img, brightness_factor=args.brightness_factor)
        else:
            raise ValueError(f"Unsupported corruption: {args.corruption}")

        img_out.save(dst_img_path)

        r_new = dict(r)
        r_new["relpath"] = dst_relpath
        r_new["is_noisy"] = True
        r_new["test_corruption"] = args.corruption

        if args.corruption == "blur":
            r_new["test_corruption_severity"] = args.blur_radius
        elif args.corruption == "gaussian_noise":
            r_new["test_corruption_severity"] = args.noise_sigma
        elif args.corruption == "brightness":
            r_new["test_corruption_severity"] = args.brightness_factor

        dst_rows.append(r_new)

    dst_jsonl = ann_dir / f"test_index_map_{suffix}.jsonl"
    write_jsonl(dst_jsonl, dst_rows)

    print(f"[OK] Corrupted test set generated")
    print(f"[OK] Images: {dst_img_dir} (count={len(dst_rows)})")
    print(f"[OK] Mapping: {dst_jsonl}")


if __name__ == "__main__":
    main()
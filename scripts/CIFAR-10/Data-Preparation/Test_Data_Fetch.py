import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from torchvision.datasets import CIFAR10


CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def stratified_pick(labels: np.ndarray, total_n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick total_n indices with near-equal count per class."""
    num_classes = int(labels.max()) + 1
    base = total_n // num_classes
    rem = total_n % num_classes
    per = {c: base + (1 if c < rem else 0) for c in range(num_classes)}

    picked = []
    for c in range(num_classes):
        idxs = np.where(labels == c)[0]
        if len(idxs) < per[c]:
            raise ValueError(f"class {c} not enough: need {per[c]}, have {len(idxs)}")
        chosen = rng.choice(idxs, size=per[c], replace=False)
        picked.append(chosen)

    picked = np.concatenate(picked)
    rng.shuffle(picked)
    return picked


def resolve_default_paths(args_out_dir: str | None, args_tv_root: str | None) -> Tuple[Path, Path]:
    cl_data_raw = os.environ.get("CL_DATA_RAW")
    if (args_out_dir is None or args_tv_root is None) and not cl_data_raw:
        raise RuntimeError(
            "CL_DATA_RAW is not set. Please run: source ./env_superpod.sh\n"
            "Or pass --out_dir and --tv_root explicitly."
        )

    out_dir = Path(args_out_dir).resolve() if args_out_dir else (Path(cl_data_raw) / "cifar10" / "v1_seed42")
    tv_root = Path(args_tv_root).resolve() if args_tv_root else (Path(cl_data_raw) / "cifar10" / "_torchvision_cache")
    return out_dir, tv_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=None,
                    help="Output dir. Default: $CL_DATA_RAW/cifar10/v1_seed42")
    ap.add_argument("--tv_root", default=None,
                    help="Torchvision cache root. Default: $CL_DATA_RAW/cifar10/_torchvision_cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total_n", type=int, default=1800)
    args = ap.parse_args()

    out_dir, tv_root = resolve_default_paths(args.out_dir, args.tv_root)

    img_dir = out_dir / "test_images"
    ann_dir = out_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    tv_root.mkdir(parents=True, exist_ok=True)

    print(f"[PATH] out_dir = {out_dir}")
    print(f"[PATH] tv_root = {tv_root}")

    rng = np.random.default_rng(args.seed)

    ds_test = CIFAR10(root=str(tv_root), train=False, download=True, transform=None)
    labels = np.array(ds_test.targets, dtype=np.int64)

    if args.total_n > len(ds_test):
        raise ValueError(f"total_n={args.total_n} > available={len(ds_test)}")

    pick = stratified_pick(labels, args.total_n, rng)

    rows: List[Dict] = []
    for new_id, orig_idx in enumerate(pick):
        img, y = ds_test[int(orig_idx)]

        relpath = f"test_images/{new_id:06d}.png"
        img.save(out_dir / relpath)

        rows.append({
            "id": int(new_id),
            "relpath": relpath,
            "orig_split": "test",
            "orig_index": int(orig_idx),
            "orig_label": int(y),
            "orig_class": CIFAR10_CLASS_NAMES[int(y)],
            "label": int(y),
            "class_name": CIFAR10_CLASS_NAMES[int(y)],
            "is_noisy": False,
            "seed": int(args.seed),
        })

    map_path = ann_dir / "test_index_map.jsonl"
    with map_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    label_hist = {name: 0 for name in CIFAR10_CLASS_NAMES}
    for r in rows:
        label_hist[r["class_name"]] += 1

    meta = {
        "dataset": "CIFAR-10",
        "step": "test_export_1800",
        "seed": args.seed,
        "total_n": args.total_n,
        "noisy_k": 0,
        "noisy_rate": 0.0,
        "source": "test",
        "out_dir": str(out_dir),
        "tv_root": str(tv_root),
        "label_hist": label_hist,
        "files": {
            "index_map_jsonl": str(map_path),
        },
    }

    meta_path = ann_dir / "test_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Test exported into: {out_dir}")
    print(f"[OK] Images: {img_dir} (count={len(rows)})")
    print(f"[OK] Mapping: {map_path}")
    print(f"[OK] Meta: {meta_path}")


if __name__ == "__main__":
    main()
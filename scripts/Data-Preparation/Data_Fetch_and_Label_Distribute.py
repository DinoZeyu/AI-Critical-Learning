"""
Step-1: Fetch 10k samples from public CIFAR-10 and export to RAW directory.

- Create a raw 10k pool.
- Optionally corrupt (flip) labels for noisy_k samples (default 2000), ensuring new_label != orig_label.
- Save exported images as PNG and write an index mapping file for traceability.

Output structure (under out_dir):
  images/000000.png ... 009999.png
  annotations/index_map.jsonl
  annotations/meta.json

Superpod recommended defaults (if you sourced env_superpod.sh):
  out_dir = $CL_DATA_RAW/cifar10/v1_seed42
  tv_root = $CL_DATA_RAW/cifar10/_torchvision_cache
"""

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


def make_noisy_labels(orig_labels: np.ndarray, noisy_k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (new_labels, noisy_mask).
    Noisy labels are guaranteed != original labels.
    """
    n = len(orig_labels)
    if noisy_k < 0 or noisy_k > n:
        raise ValueError(f"noisy_k must be in [0, {n}] but got {noisy_k}")

    noisy_mask = np.zeros(n, dtype=bool)
    if noisy_k == 0:
        return orig_labels.copy(), noisy_mask

    noisy_idx = rng.choice(np.arange(n), size=noisy_k, replace=False)
    noisy_mask[noisy_idx] = True

    new_labels = orig_labels.copy()
    for i in noisy_idx:
        y = int(orig_labels[i])
        # choose a wrong label among other 9 classes
        candidates = list(range(10))
        candidates.remove(y)
        new_labels[i] = int(rng.choice(candidates))

    return new_labels, noisy_mask


def resolve_default_paths(args_out_dir: str | None, args_tv_root: str | None) -> Tuple[Path, Path]:
    """
    If args_out_dir/tv_root are None, default to paths under $CL_DATA_RAW.
    """
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
                    help="RAW output dir. Default: $CL_DATA_RAW/cifar10/v1_seed42")
    ap.add_argument("--tv_root", default=None,
                    help="Torchvision cache root. Default: $CL_DATA_RAW/cifar10/_torchvision_cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total_n", type=int, default=10000)
    ap.add_argument("--noisy_k", type=int, default=2000)
    ap.add_argument("--source", choices=["train", "test", "both"], default="train",
                    help="Which CIFAR-10 split(s) to sample from. Recommend train.")
    args = ap.parse_args()

    out_dir, tv_root = resolve_default_paths(args.out_dir, args.tv_root)

    img_dir = out_dir / "images"
    ann_dir = out_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    tv_root.mkdir(parents=True, exist_ok=True)

    print(f"[PATH] out_dir = {out_dir}")
    print(f"[PATH] tv_root = {tv_root}")

    rng = np.random.default_rng(args.seed)

    ds_train = CIFAR10(root=str(tv_root), train=True, download=True, transform=None)
    ds_test = CIFAR10(root=str(tv_root), train=False, download=True, transform=None)

    # Build a combined index space depending on source
    if args.source == "train":
        items = [("train", i) for i in range(len(ds_train))]
        labels = np.array(ds_train.targets, dtype=np.int64)
    elif args.source == "test":
        items = [("test", i) for i in range(len(ds_test))]
        labels = np.array(ds_test.targets, dtype=np.int64)
    else:
        items = [("train", i) for i in range(len(ds_train))] + [("test", i) for i in range(len(ds_test))]
        labels = np.array(list(ds_train.targets) + list(ds_test.targets), dtype=np.int64)

    if args.total_n > len(items):
        raise ValueError(f"total_n={args.total_n} > available={len(items)} for source={args.source}")

    # Pick indices into `items`
    pick = stratified_pick(labels, args.total_n, rng)
    picked_items = [items[int(i)] for i in pick]
    orig_labels = labels[pick]

    # Corrupt labels for noisy samples
    new_labels, noisy_mask = make_noisy_labels(orig_labels, args.noisy_k, rng)

    # Export images + build mapping
    rows: List[Dict] = []
    for new_id, ((orig_split, orig_idx), y0, y1, is_noisy) in enumerate(
        zip(picked_items, orig_labels, new_labels, noisy_mask)
    ):
        if orig_split == "train":
            img, _ = ds_train[int(orig_idx)]
        else:
            img, _ = ds_test[int(orig_idx)]

        relpath = f"images/{new_id:06d}.png"
        img.save(out_dir / relpath)

        rows.append({
            "id": int(new_id),
            "relpath": relpath,
            "orig_split": orig_split,
            "orig_index": int(orig_idx),          # index within that split (train or test)
            "orig_label": int(y0),
            "orig_class": CIFAR10_CLASS_NAMES[int(y0)],
            "label": int(y1),                     # possibly corrupted label (raw label)
            "class_name": CIFAR10_CLASS_NAMES[int(y1)],
            "is_noisy": bool(is_noisy),
            "seed": int(args.seed),
        })

    # Write index_map.jsonl (one JSON per line)
    map_path = ann_dir / "train_index_map.jsonl"
    with map_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Meta summary
    label_hist_after_noise = {name: 0 for name in CIFAR10_CLASS_NAMES}
    noisy_count = int(noisy_mask.sum())
    for r in rows:
        label_hist_after_noise[r["class_name"]] += 1

    meta = {
        "dataset": "CIFAR-10",
        "step": "raw_fetch_10k",
        "seed": args.seed,
        "total_n": args.total_n,
        "noisy_k": args.noisy_k,
        "noisy_rate": noisy_count / args.total_n,
        "source": args.source,
        "out_dir": str(out_dir),
        "tv_root": str(tv_root),
        "label_hist_after_noise": label_hist_after_noise,
        "files": {
            "index_map_jsonl": str(map_path),
        },
    }
    (ann_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Raw exported: {out_dir}")
    print(f"[OK] Images: {img_dir} (count={len(rows)})")
    print(f"[OK] Mapping: {map_path}")
    print(f"[OK] Meta: {ann_dir / 'train_meta.json'}")


if __name__ == "__main__":
    main()
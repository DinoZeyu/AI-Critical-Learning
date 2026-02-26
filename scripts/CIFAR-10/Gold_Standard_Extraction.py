#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a 1,000-sample clean gold standard set from the RAW 10k pool.

Clean criteria:
  - is_noisy == false
  - label == orig_label

Inputs (default, if you sourced env_superpod.sh):
  RAW index map:
    $CL_DATA_RAW/cifar10/v1_seed42/annotations/index_map.jsonl
  RAW images:
    $CL_DATA_RAW/cifar10/v1_seed42/images/

Outputs (default):
  Images:
    $CL_DATA_PROCESSED/cifar10/v1_seed42/gold/standard/images/000000.png ... 000999.png
  Annotations:
    $CL_DATA_PROCESSED/cifar10/v1_seed42/annotations/gold_index_map.jsonl
    $CL_DATA_PROCESSED/cifar10/v1_seed42/annotations/gold_meta.json
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np


CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold_n", type=int, default=1000)
    ap.add_argument("--balanced", action="store_true",
                    help="Sample equally per class (recommended). Requires gold_n divisible by 10.")
    ap.add_argument("--copy_mode", choices=["copy", "symlink"], default="copy",
                    help="copy = duplicate pngs; symlink = faster/less space (if allowed).")
    ap.add_argument("--raw_dir", default=None,
                    help="Default: $CL_DATA_RAW/cifar10/v1_seed42")
    ap.add_argument("--processed_dir", default=None,
                    help="Default: $CL_DATA_PROCESSED/cifar10/v1_seed42")
    args = ap.parse_args()

    cl_raw = os.environ.get("CL_DATA_RAW")
    cl_proc = os.environ.get("CL_DATA_PROCESSED")
    if (args.raw_dir is None or args.processed_dir is None) and (not cl_raw or not cl_proc):
        raise RuntimeError("CL_DATA_RAW / CL_DATA_PROCESSED not set. Please run: source ./env_superpod.sh")

    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else (Path(cl_raw) / "cifar10" / "v1_seed42")
    proc_dir = Path(args.processed_dir).resolve() if args.processed_dir else (Path(cl_proc) / "cifar10" / "v1_seed42")

    raw_map = raw_dir / "annotations" / "index_map.jsonl"
    raw_images = raw_dir / "images"
    if not raw_map.exists():
        raise FileNotFoundError(f"Missing {raw_map}")
    if not raw_images.exists():
        raise FileNotFoundError(f"Missing {raw_images}")

    rng = np.random.default_rng(args.seed)
    rows = read_jsonl(raw_map)

    # Clean criteria
    clean = [
        r for r in rows
        if (not r.get("is_noisy", False)) and (int(r["label"]) == int(r["orig_label"]))
    ]

    if len(clean) < args.gold_n:
        raise ValueError(f"Not enough clean samples: have {len(clean)}, need {args.gold_n}")

    # Sample gold
    if args.balanced:
        if args.gold_n % 10 != 0:
            raise ValueError("gold_n must be divisible by 10 when --balanced is set.")
        per = args.gold_n // 10

        picked: List[Dict] = []
        for c in range(10):
            pool = [r for r in clean if int(r["orig_label"]) == c]
            if len(pool) < per:
                raise ValueError(f"Not enough clean samples in class {c}: have {len(pool)}, need {per}")
            idx = rng.choice(len(pool), size=per, replace=False)
            picked.extend([pool[i] for i in idx])
        rng.shuffle(picked)
    else:
        idx = rng.choice(len(clean), size=args.gold_n, replace=False)
        picked = [clean[i] for i in idx]

    # Output locations
    out_img_dir = proc_dir / "gold" / "images"
    out_ann_dir = proc_dir / "annotations"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    def link_or_copy(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.copy_mode == "symlink":
            if dst.exists():
                return
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)

    gold_rows: List[Dict] = []
    for gold_id, r in enumerate(picked):
        raw_id = int(r["id"])
        src = raw_dir / r["relpath"]  # images/xxxxxx.png
        dst_rel = f"gold/images/{gold_id:06d}.png"
        dst = proc_dir / dst_rel
        link_or_copy(src, dst)

        gold_rows.append({
            "gold_id": gold_id,
            "split": "gold_standard",
            "relpath": dst_rel,
            "raw_id": raw_id,
            "raw_relpath": r["relpath"],
            "label": int(r["label"]),
            "class_name": CIFAR10_CLASS_NAMES[int(r["label"])],
            "orig_label": int(r["orig_label"]),
            "orig_class": r.get("orig_class", CIFAR10_CLASS_NAMES[int(r["orig_label"])]),
            "orig_split": r["orig_split"],
            "orig_index": int(r["orig_index"]),
            "is_noisy": False,
            "seed": int(args.seed),
        })

    gold_map = out_ann_dir / "gold_index_map.jsonl"
    write_jsonl(gold_map, gold_rows)

    # Meta summary
    hist = {name: 0 for name in CIFAR10_CLASS_NAMES}
    for gr in gold_rows:
        hist[gr["class_name"]] += 1

    meta = {
        "dataset": "CIFAR-10",
        "step": "gold_standard_1k",
        "seed": args.seed,
        "gold_n": args.gold_n,
        "balanced": bool(args.balanced),
        "copy_mode": args.copy_mode,
        "raw_dir": str(raw_dir),
        "processed_dir": str(proc_dir),
        "clean_criteria": "is_noisy==false AND label==orig_label",
        "label_hist": hist,
        "files": {
            "gold_index_map_jsonl": str(gold_map),
            "gold_images_dir": str(out_img_dir),
        },
    }
    (out_ann_dir / "gold_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"[OK] gold images: {out_img_dir} (count={len(gold_rows)})")
    print(f"[OK] gold map: {gold_map}")
    print(f"[OK] gold meta: {out_ann_dir / 'gold_meta.json'}")


if __name__ == "__main__":
    main()
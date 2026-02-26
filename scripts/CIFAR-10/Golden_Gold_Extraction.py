#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Golden-Gold Standard extraction (LOCAL ONLY)

Goal:
- Build a "Golden_Gold_Standard" subset from RAW 10k using:
  (1) raw index_map: is_noisy == false
  (2) reliance score: data_reliance >= threshold
- Save ONLY a jsonl list to local_annotations/ (no images copied)
- Do NOT write any orig_* ground-truth fields (avoid leakage)

Inputs:
- RAW index map (read-only):
  $CL_DATA_RAW/cifar10/v1_seed42/annotations/index_map.jsonl
- Reliance results (local):
  local_annotations/reliance_raw10k_seed42.jsonl

Outputs (local):
- local_annotations/Golden_Gold_Standard_seed42.jsonl
- local_annotations/Golden_Gold_Standard_seed42_meta.json

Each output row:
{ gg_id, raw_id, raw_relpath, label, class_name, data_reliance, seed, threshold }
"""

import argparse
import json
import os
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
    ap.add_argument("--n", type=int, default=1000, help="How many samples to extract.")
    ap.add_argument("--threshold", type=float, default=0.90, help="Minimum data_reliance.")
    ap.add_argument("--balanced", action="store_true",
                    help="If set, sample equally per class (requires n divisible by 10).")
    ap.add_argument("--reliance_path", type=str, default="local_annotations/reliance_raw10k_seed42.jsonl")

    ap.add_argument("--out_jsonl", type=str, default="local_annotations/Golden_Gold_Standard_seed42.jsonl")
    ap.add_argument("--out_meta", type=str, default="local_annotations/Golden_Gold_Standard_seed42_meta.json")
    args = ap.parse_args()

    cl_raw = os.environ.get("CL_DATA_RAW")
    if not cl_raw:
        raise RuntimeError("CL_DATA_RAW not set. Run: source ./env_superpod.sh")

    raw_dir = Path(cl_raw) / "cifar10" / "v1_seed42"
    raw_map = raw_dir / "annotations" / "index_map.jsonl"
    if not raw_map.exists():
        raise FileNotFoundError(f"Missing raw index map: {raw_map}")

    reliance_path = Path(args.reliance_path)
    if not reliance_path.exists():
        raise FileNotFoundError(f"Missing reliance file: {reliance_path}")

    rng = np.random.default_rng(args.seed)

    # Load raw index_map and keep only fields we need (NO orig_*)
    raw_rows = read_jsonl(raw_map)
    raw_by_id: Dict[int, Dict] = {}
    for r in raw_rows:
        rid = int(r["id"])
        raw_by_id[rid] = {
            "id": rid,
            "relpath": r["relpath"],  # images/xxxxxx.png
            "label": int(r["label"]),
            "class_name": r.get("class_name") or CIFAR10_CLASS_NAMES[int(r["label"])],
            "is_noisy": bool(r.get("is_noisy", False)),
        }

    # Load reliance results
    rel_rows = read_jsonl(reliance_path)

    # Filter: is_noisy==false AND data_reliance>=threshold
    candidates: List[Dict] = []
    for rr in rel_rows:
        rid = int(rr["id"])
        if rid not in raw_by_id:
            continue

        base = raw_by_id[rid]
        if base["is_noisy"]:
            continue

        dr = float(rr["data_reliance"])
        if dr < args.threshold:
            continue

        candidates.append({
            "raw_id": rid,
            "raw_relpath": base["relpath"],
            "label": base["label"],
            "class_name": base["class_name"],
            "data_reliance": dr,
        })

    if len(candidates) < args.n:
        raise ValueError(
            f"Not enough candidates with is_noisy=false and data_reliance>={args.threshold}: "
            f"have {len(candidates)}, need {args.n}. "
            f"Try lowering threshold or n."
        )

    # Sample
    if args.balanced:
        if args.n % 10 != 0:
            raise ValueError("n must be divisible by 10 when --balanced is set.")
        per = args.n // 10
        picked: List[Dict] = []
        for c in range(10):
            pool = [x for x in candidates if int(x["label"]) == c]
            if len(pool) < per:
                raise ValueError(
                    f"Not enough candidates in class {c} at threshold {args.threshold}: "
                    f"have {len(pool)}, need {per}. "
                    f"Lower threshold or disable --balanced."
                )
            idx = rng.choice(len(pool), size=per, replace=False)
            picked.extend([pool[i] for i in idx])
        rng.shuffle(picked)
    else:
        idx = rng.choice(len(candidates), size=args.n, replace=False)
        picked = [candidates[i] for i in idx]

    # Build output rows
    out_rows: List[Dict] = []
    for gg_id, x in enumerate(picked):
        out_rows.append({
            "gg_id": gg_id,
            "raw_id": int(x["raw_id"]),
            "raw_relpath": x["raw_relpath"],  # relative to $CL_DATA_RAW/cifar10/v1_seed42/
            "label": int(x["label"]),
            "class_name": x["class_name"],
            "data_reliance": float(x["data_reliance"]),
            "seed": args.seed,
            "threshold": args.threshold,
        })

    out_jsonl = Path(args.out_jsonl)
    out_meta = Path(args.out_meta)
    write_jsonl(out_jsonl, out_rows)

    # meta + histogram
    hist = {name: 0 for name in CIFAR10_CLASS_NAMES}
    for r in out_rows:
        hist[r["class_name"]] += 1

    meta = {
        "name": "Golden_Gold_Standard",
        "step": "extract_by_is_noisy_and_reliance",
        "seed": args.seed,
        "n": args.n,
        "threshold": args.threshold,
        "balanced": bool(args.balanced),
        "filters": ["is_noisy==false", f"data_reliance>={args.threshold}"],
        "raw_index_map": str(raw_map),
        "reliance_path": str(reliance_path),
        "output_jsonl": str(out_jsonl),
        "label_hist": hist,
        "note": "Local-only output. No images copied. No orig_* fields stored. raw_relpath is relative to raw_dir.",
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote: {out_jsonl} (count={len(out_rows)})")
    print(f"[OK] meta:  {out_meta}")


if __name__ == "__main__":
    main()
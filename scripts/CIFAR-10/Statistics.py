#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistics & visualization for CIFAR-10 data_reliance (RAW10K only).

Input:
  local_annotations/reliance_raw10k_seed42.jsonl   (required)

Output:
  Plots ONLY:
    notebooks/CIFAR10_images/hist_reliance_raw10k.png
    notebooks/CIFAR10_images/stage_counts_raw10k.png

  Statistics ONLY (json/txt) under notebooks/statistics/:
    notebooks/statistics/stats_raw10k.json
    notebooks/statistics/stats_raw10k.txt

Usage:
  python scripts/Statistics.py
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["data_reliance"] = df["data_reliance"].astype(float)
    if "is_noisy" in df.columns:
        df["is_noisy"] = df["is_noisy"].astype(bool)
    return df


def make_bin_labels(bins: List[float]) -> List[str]:
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            labels.append(f"[{lo:.1f}, {hi:.1f}]")
        else:
            labels.append(f"[{lo:.1f}, {hi:.1f})")
    return labels


def bin_counts(vals: np.ndarray, bins: List[float]) -> Dict[str, int]:
    labels = make_bin_labels(bins)
    bins2 = bins[:-1] + [bins[-1] + 1e-9]  # include 1.0
    counts, _ = np.histogram(vals, bins=bins2)
    return {labels[i]: int(counts[i]) for i in range(len(labels))}


def summary_stats(vals: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
    }


def write_stats(stats: Dict, out_json: Path, out_txt: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    for k, v in stats.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        else:
            lines.append(f"{k}: {v}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw10k_jsonl", default="local_annotations/reliance_raw10k_seed42.jsonl")

    # your directory convention
    ap.add_argument("--out_img_dir", default="notebooks/CIFAR10_images")
    ap.add_argument("--out_stats_dir", default="notebooks/statistics")

    ap.add_argument(
        "--bins",
        nargs="*",
        type=float,
        default=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        help="Bin edges for stage counting (default: 0.0,0.1,...,1.0)."
    )
    ap.add_argument(
        "--low_thresholds",
        nargs="*",
        type=float,
        default=[0.2, 0.3, 0.5],
        help="Thresholds for counting low-reliance among clean (is_noisy==False)."
    )
    ap.add_argument("--hist_bins", type=int, default=30)
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    raw_path = (repo_root / args.raw10k_jsonl).resolve()
    out_img_dir = (repo_root / args.out_img_dir).resolve()
    out_stats_dir = (repo_root / args.out_stats_dir).resolve()

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing RAW10K reliance file: {raw_path}")

    raw = read_jsonl(raw_path)
    vals = raw["data_reliance"].to_numpy(dtype=float)

    stats = {
        "name": "raw10k",
        "file": str(raw_path),
        "N": int(len(raw)),
        "summary": summary_stats(vals),
        "bin_counts_overall": bin_counts(vals, args.bins),
    }

    if "is_noisy" in raw.columns:
        clean = raw[raw["is_noisy"] == False]
        noisy = raw[raw["is_noisy"] == True]
        clean_vals = clean["data_reliance"].to_numpy(dtype=float)
        noisy_vals = noisy["data_reliance"].to_numpy(dtype=float)

        stats["split_counts"] = {
            "clean_is_noisy_false": int(len(clean)),
            "noisy_is_noisy_true": int(len(noisy)),
        }
        stats["bin_counts_clean"] = bin_counts(clean_vals, args.bins)
        stats["bin_counts_noisy"] = bin_counts(noisy_vals, args.bins)

        low = {}
        for t in args.low_thresholds:
            n_low = int((clean["data_reliance"] < t).sum())
            low[f"clean_low_lt_{t:.2f}"] = {
                "count": n_low,
                "ratio": float(n_low / len(clean)) if len(clean) else 0.0
            }
        stats["clean_low_counts"] = low

    write_stats(
        stats,
        out_stats_dir / "stats_raw10k.json",
        out_stats_dir / "stats_raw10k.txt"
    )

    # plots
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=args.hist_bins)
    plt.xlabel("data_reliance")
    plt.ylabel("count")
    plt.title("Reliance distribution (raw10k)")
    p1 = out_img_dir / "hist_reliance_raw10k.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    stage_labels = list(stats["bin_counts_overall"].keys())
    stage_counts = list(stats["bin_counts_overall"].values())
    plt.figure(figsize=(10, 5))
    plt.bar(stage_labels, stage_counts)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("reliance range")
    plt.ylabel("count")
    plt.title("Reliance stage counts (raw10k)")
    p2 = out_img_dir / "stage_counts_raw10k.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    print("[OK] saved plots:")
    print(" -", p1)
    print(" -", p2)
    print("[OK] saved stats:")
    print(" -", out_stats_dir / "stats_raw10k.json")
    print(" -", out_stats_dir / "stats_raw10k.txt")


if __name__ == "__main__":
    main()
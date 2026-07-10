#!/usr/bin/env python3
"""Generate a LaTeX-ready figure from the multi-seed result CSV files.

Default output is a PDF sized for a single LaTeX column. The figure uses the
local SciencePlots-inspired style file at
Visualization/styles/gcl_paper.mplstyle.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGURES = ROOT / "Visualization" / "Figures"
STYLE_FILE = ROOT / "Visualization" / "styles" / "gcl_paper.mplstyle"

# Matplotlib tries to write under ~/.config by default on this machine.  Keep
# its cache inside the repo's ignored .cache folder so normal runs are quiet.
MPLCONFIGDIR = ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


DATASETS = ["STL", "Flower_102"]
PANEL_TITLES = {"STL": "(a) STL-10", "Flower_102": "(b) Flower-102"}

NOISE_ORDER = [
    ("feature_noise/blur_3p0", "Blur"),
    ("feature_noise/brightness_0p75", "Brt."),
    ("feature_noise/gaussian_30p0", "Gau."),
    ("label_noise/label_shuffle_0p2", "Lab."),
    ("hybrid_noise/blur_3p0_label_shuffle_0p2", "Blur+L"),
    ("hybrid_noise/brightness_0p75_label_shuffle_0p2", "Brt.+L"),
    ("hybrid_noise/gaussian_30p0_label_shuffle_0p2", "Gau.+L"),
]
NOISE_RANK = {rel: i for i, (rel, _) in enumerate(NOISE_ORDER)}
NOISE_LABEL = dict(NOISE_ORDER)
COMPACT_NOISE_LABEL = {
    "feature_noise/blur_3p0": "Blur",
    "feature_noise/brightness_0p75": "Bright.",
    "feature_noise/gaussian_30p0": "Gauss.",
    "label_noise/label_shuffle_0p2": "Label",
    "hybrid_noise/blur_3p0_label_shuffle_0p2": "Blur\n+Label",
    "hybrid_noise/brightness_0p75_label_shuffle_0p2": "Bright.\n+Label",
    "hybrid_noise/gaussian_30p0_label_shuffle_0p2": "Gauss.\n+Label",
}

COL = {
    "ink": "#2B2B2B",
    "muted": "#676B6F",
    "line": "#DED7CD",
    "grid": "#EFE9DF",
    "panel": "#FCFAF6",
    "ggcl": "#C99B2E",
    "ggcl_dark": "#7A5A19",
    "ggcl_soft": "#F3DCA0",
    "clean": "#1E8E4A",
    "clean_dark": "#146233",
    "clean_soft": "#D2E6D6",
    "noise": "#D9988E",
    "noise_dark": "#9A5D55",
    "noise_soft": "#F2CDC7",
    "accent": "#F1C86A",
    "accent_dark": "#7A5A19",
}

BAR_COL = {
    "ggcl": "#F9CCE0",
    "ggcl_dark": "#A83B72",
    "noise": "#FEF1F3",
    "noise_dark": "#C87C91",
}

SINGLE_COLUMN_WIDTH_IN = 3.48
STACKED_BAR_PANEL_SIZE = (SINGLE_COLUMN_WIDTH_IN, SINGLE_COLUMN_WIDTH_IN * 1.25)


def configure_style() -> None:
    plt.style.use(str(STYLE_FILE))
    sns.set_style(
        "ticks",
        rc={
            "axes.edgecolor": COL["ink"],
            "grid.color": COL["grid"],
            "xtick.color": COL["muted"],
            "ytick.color": COL["muted"],
        },
    )
    mpl.rcParams.update(
        {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.titleweight": "normal",
            "savefig.bbox": "standard",
        }
    )


def load_main_stats() -> pd.DataFrame:
    baseline_df = pd.read_csv(DOCS / "multi_seed_main_setting_stats.csv")
    exact_df = pd.read_csv(DOCS / "multi_seed_ggcl_exact_run_stats.csv")

    exact_numeric_cols = [
        "selected_acc_mean",
        "selected_acc_std",
        "peak_acc_mean",
        "peak_acc_std",
        "final_acc_mean",
        "final_acc_std",
        "selected_epoch_mean",
        "selected_epoch_std",
    ]
    baseline_numeric_cols = [
        "clean_selected_acc",
        "noise_selected_acc_mean",
        "noise_selected_acc_std",
    ]
    for col in exact_numeric_cols:
        exact_df[col] = pd.to_numeric(exact_df[col])
    for col in baseline_numeric_cols:
        baseline_df[col] = pd.to_numeric(baseline_df[col])

    # Match the main table: choose one best exact hyperparameter setting across
    # all three seeds, rather than mixing different best runs seed-by-seed.
    exact_df = exact_df.sort_values(
        ["dataset", "train_rel", "selected_acc_mean", "selected_acc_std"],
        ascending=[True, True, False, True],
    )
    df = exact_df.groupby(["dataset", "train_rel"], as_index=False).first()
    df = df.merge(
        baseline_df[
            [
                "dataset",
                "train_rel",
                "clean_selected_acc",
                "noise_selected_acc_mean",
                "noise_selected_acc_std",
            ]
        ],
        on=["dataset", "train_rel"],
        how="left",
        validate="one_to_one",
    )
    df["noise_rank"] = df["train_rel"].map(NOISE_RANK)
    df["noise_label"] = df["train_rel"].map(NOISE_LABEL)
    return df.sort_values(["dataset", "noise_rank"])


def ordered_dataset_rows(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    return df[df["dataset"] == dataset].sort_values("noise_rank").reset_index(drop=True)


def save_figure(fig: mpl.figure.Figure, stem: str, formats: list[str], dpi: int) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = FIGURES / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches=None, pad_inches=0)
        paths.append(path)
    plt.close(fig)
    return paths


def shrink_legend_text(legend: mpl.legend.Legend) -> None:
    for text in legend.get_texts():
        text.set_fontsize(4.4)


def accuracy_legend_handles() -> list[Patch | Line2D]:
    return [
        Patch(facecolor=BAR_COL["noise"], edgecolor="none", label="Noisy baseline"),
        Patch(facecolor=BAR_COL["ggcl"], edgecolor="none", label="GCL result"),
        Line2D([0], [0], color=COL["clean"], lw=0.76, ls=(0, (4, 2)), label="Clean baseline"),
    ]


def accuracy_y_max(main_df: pd.DataFrame) -> float:
    return min(
        100.0,
        max(
            main_df["selected_acc_mean"].max() + main_df["selected_acc_std"].max(),
            main_df["noise_selected_acc_mean"].max() + main_df["noise_selected_acc_std"].max(),
            main_df["clean_selected_acc"].max(),
        )
        * 100.0
        + 5.5,
    )


def draw_accuracy_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    dataset: str,
    y_max: float,
    *,
    show_title: bool,
    show_x_labels: bool = True,
) -> None:
    bar_width = 0.34
    x = np.arange(len(rows))
    noisy = rows["noise_selected_acc_mean"].to_numpy() * 100.0
    noisy_std = rows["noise_selected_acc_std"].to_numpy() * 100.0
    ggcl = rows["selected_acc_mean"].to_numpy() * 100.0
    ggcl_std = rows["selected_acc_std"].to_numpy() * 100.0
    clean = float(rows["clean_selected_acc"].iloc[0]) * 100.0
    clean_x0, clean_x1 = -0.55, len(rows) - 0.45

    error_kw = {
        "elinewidth": 0.36,
        "capsize": 0.75,
        "capthick": 0.36,
    }
    ax.bar(
        x - bar_width / 2,
        noisy,
        width=bar_width,
        color=BAR_COL["noise"],
        edgecolor=BAR_COL["noise_dark"],
        linewidth=0.18,
        yerr=noisy_std,
        error_kw={**error_kw, "ecolor": BAR_COL["noise_dark"]},
        zorder=2,
    )
    ax.bar(
        x + bar_width / 2,
        ggcl,
        width=bar_width,
        color=BAR_COL["ggcl"],
        edgecolor=BAR_COL["ggcl_dark"],
        linewidth=0.20,
        yerr=ggcl_std,
        error_kw={**error_kw, "ecolor": BAR_COL["ggcl_dark"]},
        zorder=3,
    )
    ax.hlines(clean, clean_x0, clean_x1, color=COL["clean"], lw=0.76, ls=(0, (4, 2)), zorder=6)

    if show_title:
        ax.set_title(PANEL_TITLES[dataset], pad=2.2, fontsize=4.8)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    if show_x_labels:
        compact_labels = rows["train_rel"].map(COMPACT_NOISE_LABEL).to_numpy()
        for xi, label in zip(x, compact_labels):
            ax.text(
                xi,
                -0.058,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=4.2,
                color=COL["ink"],
                linespacing=0.86,
                clip_on=False,
            )
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0.0, y_max)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)
    ax.tick_params(axis="both", width=0.42, length=1.7, pad=1.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COL["ink"])
        spine.set_linewidth(0.36)


def figure_accuracy(main_df: pd.DataFrame, formats: list[str], dpi: int) -> list[Path]:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=STACKED_BAR_PANEL_SIZE, sharex=True)
    axes = np.array([ax1, ax2])
    handles = accuracy_legend_handles()
    y_max = accuracy_y_max(main_df)

    for ax, dataset, show_x_labels in zip(axes, DATASETS, [False, True]):
        rows = ordered_dataset_rows(main_df, dataset)
        draw_accuracy_panel(ax, rows, dataset, y_max, show_title=True, show_x_labels=show_x_labels)

    for ax in axes:
        ax.set_ylabel("Selected Accuracy (%)", fontsize=4.4, labelpad=2.1)
        ax.tick_params(axis="y", labelsize=4.3)
    fig.text(0.56, 0.04, "Noise Type", ha="center", va="center", fontsize=4.4, color=COL["ink"])
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.99), ncol=3, frameon=False, handlelength=0.9)
    shrink_legend_text(legend)
    for text in legend.get_texts():
        text.set_fontsize(4.4)
    fig.subplots_adjust(left=0.105, right=0.997, bottom=0.095, top=0.91, hspace=0.28)
    return save_figure(fig, "accuracy_with_baselines_singlecol", formats, dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
        choices=["pdf", "png", "svg"],
        help="Output formats. Default: pdf.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="DPI for raster outputs. Default: 600.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    main_df = load_main_stats()

    outputs: list[Path] = []

    outputs.extend(figure_accuracy(main_df, args.formats, args.dpi))

    print("Generated figures:")
    for path in outputs:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

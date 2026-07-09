#!/usr/bin/env python3
"""Generate LaTeX-ready figures from the multi-seed result CSV files.

Default outputs are PDF and high-DPI PNG. The figures are sized for a single
LaTeX column and use the local SciencePlots-inspired style file at
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
DATASET_LABELS = {"STL": "STL-10", "Flower_102": "Flower-102"}

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
    "feature_noise/brightness_0p75": "Brightness",
    "feature_noise/gaussian_30p0": "Gaussian",
    "label_noise/label_shuffle_0p2": "Label",
    "hybrid_noise/blur_3p0_label_shuffle_0p2": "Blur\n+Label",
    "hybrid_noise/brightness_0p75_label_shuffle_0p2": "Brightness\n+Label",
    "hybrid_noise/gaussian_30p0_label_shuffle_0p2": "Gaussian\n+Label",
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
SINGLE_COLUMN_16_9_SIZE = (SINGLE_COLUMN_WIDTH_IN, SINGLE_COLUMN_WIDTH_IN * 9 / 16)
BAR_PANEL_SIZE = SINGLE_COLUMN_16_9_SIZE
SUBFIG_PANEL_SIZE = (SINGLE_COLUMN_WIDTH_IN * 0.49, SINGLE_COLUMN_WIDTH_IN * 0.44)


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
    df = pd.read_csv(DOCS / "multi_seed_main_setting_stats.csv")
    numeric_cols = [
        "selected_acc_mean",
        "selected_acc_std",
        "clean_selected_acc",
        "noise_selected_acc_mean",
        "noise_selected_acc_std",
        "recovery_mean",
        "recovery_std",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
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


def add_panel_labels(axes: np.ndarray) -> None:
    for label, ax in zip(["a", "b"], axes):
        ax.text(
            -0.10,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=7.2,
            fontweight="bold",
            color=COL["ink"],
            va="top",
            ha="left",
        )


def shrink_legend_text(legend: mpl.legend.Legend) -> None:
    for text in legend.get_texts():
        text.set_fontsize(4.8)


def common_axis_format(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)
    ax.tick_params(axis="both", width=0.55, length=2.2, pad=1.5)
    sns.despine(ax=ax, offset=2, trim=False)


def common_horizontal_format(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", width=0.55, length=2.2, pad=1.4)
    sns.despine(ax=ax, offset=2, trim=False)


def hide_inner_y_axis(axes: np.ndarray) -> None:
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)


def accuracy_legend_handles() -> list[Patch | Line2D]:
    return [
        Patch(facecolor=BAR_COL["noise"], edgecolor="none", label="Noisy baseline"),
        Patch(facecolor=BAR_COL["ggcl"], edgecolor="none", label="GCL"),
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
        ax.set_title(DATASET_LABELS[dataset], pad=1.4, fontsize=4.2)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    compact_labels = rows["train_rel"].map(COMPACT_NOISE_LABEL).to_numpy()
    for xi, label in zip(x, compact_labels):
        ax.text(
            xi,
            -0.058,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=2.35,
            color=COL["ink"],
            linespacing=0.86,
            clip_on=False,
        )
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
    fig, axes = plt.subplots(1, 2, figsize=BAR_PANEL_SIZE, sharey=True)
    handles = accuracy_legend_handles()
    y_max = accuracy_y_max(main_df)

    for ax, dataset in zip(axes, DATASETS):
        rows = ordered_dataset_rows(main_df, dataset)
        draw_accuracy_panel(ax, rows, dataset, y_max, show_title=True)

    hide_inner_y_axis(axes)
    axes[0].set_ylabel("Selected Accuracy (%)", fontsize=3.8, labelpad=1.7)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=3.75)
    fig.text(0.56, 0.055, "Noise Type", ha="center", va="center", fontsize=3.8, color=COL["ink"])
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.985), ncol=3, frameon=False, handlelength=0.9)
    shrink_legend_text(legend)
    for text in legend.get_texts():
        text.set_fontsize(3.9)
    fig.subplots_adjust(left=0.095, right=0.997, bottom=0.145, top=0.855, wspace=0.075)
    return save_figure(fig, "accuracy_with_baselines_singlecol", formats, dpi)


def figure_accuracy_subfigures(main_df: pd.DataFrame, formats: list[str], dpi: int) -> list[Path]:
    outputs: list[Path] = []
    y_max = accuracy_y_max(main_df)
    stems = {
        "STL": "accuracy_stl10_subfigure",
        "Flower_102": "accuracy_flower102_subfigure",
    }

    for dataset in DATASETS:
        fig, ax = plt.subplots(1, 1, figsize=SUBFIG_PANEL_SIZE)
        rows = ordered_dataset_rows(main_df, dataset)
        draw_accuracy_panel(ax, rows, dataset, y_max, show_title=False)
        ax.set_ylabel("Selected Accuracy (%)", fontsize=3.8, labelpad=1.7)
        ax.tick_params(axis="y", labelsize=3.75)
        fig.text(0.56, 0.045, "Noise Type", ha="center", va="center", fontsize=3.8, color=COL["ink"])
        fig.subplots_adjust(left=0.175, right=0.995, bottom=0.185, top=0.985)
        outputs.extend(save_figure(fig, stems[dataset], formats, dpi))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
        help="Output formats. Default: pdf png.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="DPI for raster outputs. Default: 600.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    main_df = load_main_stats()

    outputs: list[Path] = []

    # Only generate the individual subfigure panels.
    outputs.extend(figure_accuracy_subfigures(main_df, args.formats, args.dpi))

    print("Generated figures:")
    for path in outputs:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
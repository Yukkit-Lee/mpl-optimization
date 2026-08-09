#!/usr/bin/env python3
"""Build manuscript Figure 4 (initializer-source grouped bars).

Run from the repository root:

    python new_scripts/new_paper_figures/new_build_figure4.py

Input:
    new_expTab/new_stage_b_anchor_postagg/
    new_stage_b_anchor_postagg_init_source_ablation.csv
Output:
    new_expFig/paper_experiment/new_figure4_initializer_source_bars.{png,pdf}

Safe visual parameters below include figure size, colors, bar width, label
offsets and y-axis display limits.  The method order, dataset/distribution grid,
accuracy column and test/holdout gates are experimental semantics and must not
be changed to make a method appear better.  Source rows are seed42 (n=1), so no
error bar may be added without new independent runs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from new_paper_figure_common import DATASET_LABELS, OUT, ROOT, read_csv, save_figure


# Safe visual parameters.
FIGSIZE = (14.6, 7.7)
BAR_WIDTH = 0.68
COLORS = ("#2C7BB6", "#FDAE61", "#8C8C8C", "#D7191C")
Y_LIMITS = {"cifar10": 66, "cifar100": 25}
VALUE_LABEL_OFFSETS = {"cifar10": 0.8, "cifar100": 0.3}

# Audited panel/method definitions; changing these changes the scientific plot.
DATASETS = ("cifar10", "cifar100")
DISTRIBUTIONS = ("iid", "mild", "severe")
METHODS = (
    "C1_random_plus_anchor",
    "C3_fixed_local0_plus_anchor",
    "C4_secure_average",
    "C5_secure_average_plus_anchor",
)
LABELS = (
    "Random + Anchor",
    "Local-0 + Anchor",
    "Secure Average",
    "Average + Anchor",
)
SHORT_LABELS = (
    "Random\n+Anchor",
    "Local-0\n+Anchor",
    "Secure\nAverage",
    "Average\n+Anchor",
)
SOURCE = (
    ROOT
    / "new_expTab/new_stage_b_anchor_postagg/"
    "new_stage_b_anchor_postagg_init_source_ablation.csv"
)


def load_rows():
    rows = read_csv(SOURCE)
    if any(
        row["test_read"] != "False" or row["locked_holdout_read"] != "False"
        for row in rows
    ):
        raise ValueError("Stage-B plot source read test/holdout")
    return rows


def lookup(rows, dataset, distribution, method):
    matches = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["distribution"] == distribution
        and row["method"] == method
    ]
    if len(matches) != 1:
        raise ValueError("non-unique ablation row")
    return float(matches[0]["accuracy"])


def main():
    rows = load_rows()
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex="row", sharey="row")
    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, distribution in enumerate(DISTRIBUTIONS):
            ax = axes[row_idx, col_idx]
            values = np.asarray(
                [
                    100.0 * lookup(rows, dataset, distribution, method)
                    for method in METHODS
                ]
            )
            xpos = np.arange(len(METHODS))
            bars = ax.bar(
                xpos,
                values,
                color=COLORS,
                width=BAR_WIDTH,
                edgecolor="white",
                linewidth=0.7,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + VALUE_LABEL_OFFSETS[dataset],
                    "{:.1f}".format(value),
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                )
            ax.set_title(distribution.capitalize(), fontweight="bold")
            ax.grid(axis="y", alpha=0.16)
            if col_idx == 0:
                ax.set_ylabel(
                    DATASET_LABELS[dataset] + "\nDevelopment accuracy (%)"
                )
            ax.set_xticks(xpos)
            ax.set_xticklabels(SHORT_LABELS, fontsize=7)
            ax.set_ylim(0, Y_LIMITS[dataset])
            ax.text(
                0.98, 0.04, "n=1", transform=ax.transAxes, ha="right",
                fontsize=7, color="#666666",
            )
    fig.suptitle(
        "Initializer-source ablation under heterogeneous partitions",
        fontweight="bold",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for color, label in zip(COLORS, LABELS)
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.text(
        0.5,
        -0.042,
        "Seed 42 (n=1); no uncertainty estimate. Red denotes the proposed secure-average-plus-anchor initializer.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    save_figure(fig, "new_figure4_initializer_source_bars")
    print("PASS built Figure 4 in", OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build manuscript Figure 5 (same-budget anchor convergence).

Run from the repository root:

    python new_scripts/new_paper_figures/new_build_figure5.py

Input:
    new_log/new_stage_b_anchor_postagg/
    new_stage_b_anchor_postagg_trajectories.json
Output:
    new_expFig/paper_experiment/new_figure5_anchor_convergence_pea.{png,pdf}

Safe visual parameters below include figure size, method colors, line width and
marker size.  The selected methods, history steps, development accuracies,
dataset/distribution panels and privacy gates are experimental semantics.  Do
not smooth, interpolate or truncate histories in this plotting script.  Only
seed42 trajectories exist, so adding uncertainty bars would be scientifically
invalid without new independent runs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from new_paper_figure_common import (
    DATASET_LABELS,
    FULL_BLUE,
    OURS_RED,
    OUT,
    ROOT,
    read_json,
    save_figure,
)


# Safe visual parameters.
FIGSIZE = (14.2, 7.0)
LINE_WIDTH = 2.0
ENDPOINT_SIZE = 18

# Audited methods/panels; changing these changes the scientific comparison.
DATASETS = ("cifar10", "cifar100")
DISTRIBUTIONS = ("iid", "mild", "severe")
METHOD_COLORS = {
    "C1_random_plus_anchor": "#636363",
    "C3_fixed_local0_plus_anchor": FULL_BLUE,
    "C5_secure_average_plus_anchor": OURS_RED,
}
METHOD_LABELS = {
    "C1_random_plus_anchor": "Random + Anchor",
    "C3_fixed_local0_plus_anchor": "Local-0 + Anchor",
    "C5_secure_average_plus_anchor": "Average + Anchor",
}
SOURCE = (
    ROOT
    / "new_log/new_stage_b_anchor_postagg/"
    "new_stage_b_anchor_postagg_trajectories.json"
)


def load_trajectories():
    trajectories = read_json(SOURCE)
    if (
        trajectories["status"] != "COMPLETE"
        or trajectories["test_read"]
        or trajectories["locked_holdout_read"]
    ):
        raise ValueError("invalid Stage-B trajectories")
    return trajectories


def main():
    trajectories = load_trajectories()
    fig, axes = plt.subplots(
        2, 3, figsize=FIGSIZE, sharex=True, sharey="row"
    )
    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, distribution in enumerate(DISTRIBUTIONS):
            ax = axes[row_idx, col_idx]
            matches = [
                item
                for item in trajectories["rows"]
                if item["dataset"] == dataset
                and item["distribution"] == distribution
            ]
            if len(matches) != 3:
                raise ValueError("expected three Stage-B methods per panel")
            for item in matches:
                steps = [point["step"] for point in item["history"]]
                values = [
                    100.0 * point["development_accuracy"]
                    for point in item["history"]
                ]
                ax.plot(
                    steps,
                    values,
                    color=METHOD_COLORS[item["method"]],
                    linewidth=LINE_WIDTH,
                    label=METHOD_LABELS[item["method"]],
                )
                ax.scatter(
                    [steps[-1]], [values[-1]],
                    color=METHOD_COLORS[item["method"]], s=ENDPOINT_SIZE,
                )
            ax.set_title(distribution.capitalize(), fontweight="bold")
            ax.set_xlabel("Anchor-refinement steps")
            ax.grid(alpha=0.17)
            if col_idx == 0:
                ax.set_ylabel(
                    DATASET_LABELS[dataset] + "\nDevelopment accuracy (%)"
                )
            ax.text(
                0.98, 0.04, "seed 42 (n=1)", transform=ax.transAxes,
                ha="right", fontsize=7, color="#666666",
            )
    handles = [
        Line2D(
            [0], [0], color=color, lw=2.1, label=METHOD_LABELS[method]
        )
        for method, color in METHOD_COLORS.items()
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle("Same-budget anchor-refinement convergence", fontweight="bold")
    fig.text(
        0.5,
        -0.045,
        "Only seed 42 trajectories exist, so no uncertainty bars are drawn.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, "new_figure5_anchor_convergence_pea")
    print("PASS built Figure 5 in", OUT)


if __name__ == "__main__":
    main()

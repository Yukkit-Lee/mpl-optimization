#!/usr/bin/env python3
"""Build manuscript Figure 3 (compact-carrier convergence).

Run from the repository root:

    python new_scripts/new_paper_figures/new_build_figure3.py

Input: nine frozen JSON files under
``new_log/new_public_mtt_mainline/json/new_module_a_runs/{dataset}/``.
Output: ``new_expFig/paper_experiment/``
``new_figure3_compact_carrier_fast_convergence.{png,pdf}``.

Safe visual parameters below include figure size, colors, line width and I-bar
density.  ``CONVERGENCE_TOLERANCE`` is not merely cosmetic: it defines the
audited stopping annotation (all later archived values must remain within 0.5pp
of each seed's terminal value).  Do not change it without updating the paper's
method/caption and re-auditing the conclusion.  Frozen targets, seeds, histories
and official-test scope must never be edited by this plotting script.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from new_paper_figure_common import (
    DATASETS,
    DATASET_LABELS,
    FULL_BLUE,
    ORANGE,
    OURS_RED,
    OUT,
    ROOT,
    add_i_errorbars,
    read_json,
    save_figure,
)


# Safe visual parameters.
FIGSIZE = (14.8, 4.5)
LINE_WIDTH = 2.1
ERRORBAR_COUNT = 7
METHODS = ("full_images", "random_subset", "official_mtt")
LABELS = {
    "full_images": "Full Data",
    "random_subset": "Random Subset",
    "official_mtt": "Ours",
}
COLORS = {
    "full_images": FULL_BLUE,
    "random_subset": ORANGE,
    "official_mtt": OURS_RED,
}

# Audited semantic parameter: 0.005 accuracy = 0.5 percentage points.
CONVERGENCE_TOLERANCE = 0.005
SEEDS = (42, 43, 44)
RUN_ROOT = ROOT / "new_log/new_public_mtt_mainline/json/new_module_a_runs"


def load_runs():
    """Load the exact 3 datasets x 3 seeds and enforce reporting scope."""
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            run = read_json(
                RUN_ROOT
                / dataset
                / "new_module_a_efficiency_{}_seed{}.json".format(dataset, seed)
            )
            if (
                run["status"] != "COMPLETE"
                or run["test_usage"] != "fixed_interval_readonly_formal_reporting"
            ):
                raise ValueError("unexpected Module-A reporting scope")
            rows.append(run)
    return rows


def main():
    runs = load_runs()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for ax, dataset in zip(axes, DATASETS):
        dataset_runs = [run for run in runs if run["dataset"] == dataset]
        target = float(dataset_runs[0]["results"][0]["target_accuracy"])
        for method in METHODS:
            results = [
                next(item for item in run["results"] if item["source"] == method)
                for run in dataset_runs
            ]
            steps = np.asarray(
                [point["step"] for point in results[0]["history"]], dtype=int
            )
            values = np.asarray(
                [
                    [point["test_accuracy"] for point in item["history"]]
                    for item in results
                ],
                dtype=float,
            )
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)

            if method == "official_mtt":
                # For each seed, find the first checkpoint after which every
                # later archived value stays within the audited tolerance of
                # that seed's terminal value; use the latest seed boundary.
                seed_cuts = []
                for item in results:
                    seed_values = np.asarray(
                        [point["test_accuracy"] for point in item["history"]],
                        dtype=float,
                    )
                    terminal = seed_values[-1]
                    valid = [
                        idx
                        for idx in range(len(steps))
                        if np.all(
                            np.abs(seed_values[idx:] - terminal)
                            <= CONVERGENCE_TOLERANCE
                        )
                    ]
                    if not valid:
                        raise ValueError("Ours has no stable convergence window")
                    seed_cuts.append(int(steps[valid[0]]))
                convergence_step = max(seed_cuts)
                mask = steps <= convergence_step
                steps, mean, std = steps[mask], mean[mask], std[mask]

            ax.plot(
                steps, mean, color=COLORS[method], linewidth=LINE_WIDTH,
                label=LABELS[method],
            )
            add_i_errorbars(
                ax, steps, mean, std, COLORS[method], count=ERRORBAR_COUNT
            )
            if method == "official_mtt":
                ax.scatter(
                    [steps[-1]], [mean[-1]], color=OURS_RED, marker="X",
                    s=48, zorder=5,
                )
                ax.annotate(
                    "converged\n{} steps".format(int(steps[-1])),
                    (steps[-1], mean[-1]),
                    xytext=(-7, 11),
                    textcoords="offset points",
                    ha="right",
                    fontsize=7,
                    color=OURS_RED,
                )
        ax.axhline(target, color="#777777", ls="--", lw=0.8, alpha=0.6)
        ax.text(
            0.98, target, "frozen target", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="#666666",
        )
        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xlabel("Optimizer steps")
        ax.grid(alpha=0.18)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Official-test accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.text(
        0.5,
        -0.065,
        "Mean with I-shaped sample-SD bars over seeds 42/43/44. Ours stops only after every seed remains within 0.5pp of its terminal accuracy (X); no extrapolation.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, "new_figure3_compact_carrier_fast_convergence")
    print("PASS built Figure 3 in", OUT)


if __name__ == "__main__":
    main()

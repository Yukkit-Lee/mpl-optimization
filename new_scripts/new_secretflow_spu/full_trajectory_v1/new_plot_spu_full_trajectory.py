#!/usr/bin/env python3
"""PEA-style SPU full-trajectory Figure 2 candidate (handoff §10).

Continuous real Stage-B + Stage-C trajectory, n=1, no interpolation/smoothing.
x-axis = cumulative_training_online_secure_seconds. Stage-B->C boundary shown
with a vertical line. True Ours red #D7191C, Full blue #2C7BB6.
Writes to new_expFig/paper_experiment/spu_full_trajectory_v1/ (never overwrites
the current paper figure).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "new_expFig/paper_experiment/spu_full_trajectory_v1"
PLOT_DATA = ROOT / "new_output/new_secretflow_spu/new_figure2_spu_full_trajectory_plot_data.csv"

DATASETS = ("cifar10", "cifar100", "tiny")
DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tiny": "Tiny-ImageNet"}
METHODS = ("full_data", "true_ours")
COLORS = {"full_data": "#2C7BB6", "true_ours": "#D7191C"}
METHOD_LABELS = {"full_data": "Full Data", "true_ours": "True Ours"}


def load_plot_data():
    rows = {}
    with PLOT_DATA.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault((row["dataset"], row["method"]), []).append(row)
    return rows


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_plot_data()
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.6), sharey=False)

    for axis, dataset in zip(axes, DATASETS):
        targets = {
            "T@80": float(data[(dataset, "true_ours")][0]["target_T80"]),
            "T@90": float(data[(dataset, "true_ours")][0]["target_T90"]),
            "T@95": float(data[(dataset, "true_ours")][0]["target_T95"]),
        }
        for method in METHODS:
            rows = data[(dataset, method)]
            # Ordered by stage then epoch/step.
            ordered = sorted(
                rows,
                key=lambda r: (
                    0 if r["stage"].startswith("stage_b") else 1,
                    int(float(r["epoch_or_step"])),
                ),
            )
            x = np.asarray(
                [float(r["cumulative_training_online_secure_seconds"]) for r in ordered]
            )
            y = np.asarray([float(r["development_accuracy"]) for r in ordered])
            color = COLORS[method]
            axis.plot(
                x, y, color=color, linewidth=2.2,
                label="{} (n=1)".format(METHOD_LABELS[method]),
                marker="o", markersize=3.5,
            )
            axis.scatter([x[-1]], [y[-1]], color=color, s=32, zorder=5)
            # Stage B->C transition: first point whose stage == stage_c.
            transition_x = next(
                (float(r["cumulative_training_online_secure_seconds"]) for r in ordered
                 if r["stage"].startswith("stage_c")),
                None,
            )
            if transition_x is not None:
                axis.axvline(
                    transition_x, color=color, linestyle=":", linewidth=1.2, alpha=0.7
                )
        for label, value in targets.items():
            axis.axhline(value, color="#777777", linestyle="--", linewidth=0.7, alpha=0.45)
            axis.text(
                0.015, value, label, transform=axis.get_yaxis_transform(),
                fontsize=7, color="#666666", va="bottom",
            )
        axis.set_xscale("log")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_title(DATASET_LABELS[dataset], fontweight="bold")
        axis.set_xlabel("Cumulative online secure runtime (seconds, log scale)")
        axis.grid(True, which="both", alpha=0.18)
        axis.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Development validation accuracy")
    fig.suptitle(
        "Development accuracy versus cumulative online secure runtime",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        (
            "Continuous Stage-B + Stage-C trajectory (seed42, n=1). Dotted "
            "vertical lines: Stage B→C transition. Evaluation overhead is "
            "measured separately and not included in training online time. "
            "ε is conditional Stage C only, not end-to-end DP."
        ),
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 220
        fig.savefig(
            str(FIG_DIR / "new_figure2_spu_full_trajectory_seed42") + "." + suffix,
            **kwargs,
        )
    plt.close(fig)
    print("saved full-trajectory figures to", FIG_DIR)


if __name__ == "__main__":
    main()

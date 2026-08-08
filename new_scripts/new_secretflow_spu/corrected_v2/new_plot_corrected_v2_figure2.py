#!/usr/bin/env python3
"""corrected-v2 SPU Figure 2 in the original PEA visual grammar (handoff §8).

Replicates new_expFig/new_true_ours/new_figure2_pea_style_secure_time.png's
draw_figure: 1x3 aligned panels, % y-axis, log-seconds x-axis, Full blue
#2C7BB6 / True Ours red #D7191C, summary boxes, T@80/90/95 gray dashed lines,
Stage B->C dotted vertical lines, terminal point markers, n=1 legend, NO
variance shading. Reads only corrected-v2 plot data (hash-registered).
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
FIG_DIR = ROOT / "new_expFig/new_secretflow_spu/corrected_v2"
PLOT_DATA = ROOT / "new_output/new_secretflow_spu/new_figure2_spu_corrected_v2_plot_data.csv"

DATASETS = ("cifar10", "cifar100", "tiny")
DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tiny": "Tiny-ImageNet"}
METHODS = ("full_data", "true_ours")
COLORS = {"full_data": "#2C7BB6", "true_ours": "#D7191C"}
METHOD_LABELS = {"full_data": "Full Data", "true_ours": "True Ours"}


def display_time(seconds):
    if seconds is None:
        return "Not Reached"
    if seconds < 60:
        return "{:.1f}s".format(seconds)
    if seconds < 3600:
        return "{:.0f}m".format(seconds / 60)
    return "{:.1f}h".format(seconds / 3600)


def load_plot_data():
    rows = {}
    with PLOT_DATA.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["method"])
            rows.setdefault(key, []).append(row)
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
            x = np.asarray(
                [float(r["cumulative_online_secure_seconds"]) for r in rows]
            )
            y = np.asarray([float(r["development_accuracy"]) for r in rows])
            color = COLORS[method]
            axis.plot(
                x, y, color=color, linewidth=2.2,
                label="{} (n={})".format(METHOD_LABELS[method], 1),
                marker="o", markersize=3.5,
            )
            # Terminal point marker.
            axis.scatter([x[-1]], [y[-1]], color=color, s=32, zorder=5)
            # Stage B->C transition dotted vertical line.
            transition = float(rows[0]["stage_b_to_c_transition_seconds"])
            axis.axvline(
                transition, color=color, linestyle=":", linewidth=1.2, alpha=0.7
            )
            # Summary box (mirrors PEA draw_figure).
            acc = y[-1]
            t90 = None
            for r in rows:
                if float(r["development_accuracy"]) >= targets["T@90"]:
                    t90 = float(r["cumulative_online_secure_seconds"])
                    break
            annotation = (
                "{}\nAcc {:.1%} | features=…\nT@90={} | conditional ε=3.0"
            ).format(METHOD_LABELS[method], acc, display_time(t90))
            axis.text(
                0.02 if method == "full_data" else 0.98,
                0.97,
                annotation,
                transform=axis.transAxes,
                ha="left" if method == "full_data" else "right",
                va="top",
                fontsize=7.0,
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.82,
                },
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
            "Dotted vertical lines: Stage B→C transition. ε is conditional "
            "Stage C only, not end-to-end DP. SPU seed42 only (n=1), no "
            "variance shading. x-axis = cumulative online secure."
        ),
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 220
        fig.savefig(str(FIG_DIR / "new_figure2_spu_corrected_v2_secure_time") + "." + suffix, **kwargs)
    plt.close(fig)
    print("saved figures to", FIG_DIR)


if __name__ == "__main__":
    main()

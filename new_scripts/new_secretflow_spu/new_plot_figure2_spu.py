#!/usr/bin/env python3
"""Generate the SPU three-panel Figure 2 (seed42).

Development accuracy vs cumulative online secure runtime (log x-axis) for
CIFAR-10 / CIFAR-100 / Tiny-ImageNet, Full (blue) vs Ours (red). Tiny Full is
a negative stop on SPU and is marked explicitly rather than plotted from a
fabricated run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "new_output/new_secretflow_spu"
FIG = ROOT / "new_expFig/new_secretflow_spu"

# Validated palette (dataviz): categorical slot 1 = Full blue, slot 8 = Ours red.
FULL = "#2a78d6"
OURS = "#e34948"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"

DATASETS = ("cifar10", "cifar100", "tiny")
TARGETS = {
    "cifar10": (0.58128, 0.65394, 0.69027),
    "cifar100": (0.34464, 0.38772, 0.40926),
    "tiny": (0.22120, 0.24885, 0.262675),
}


def load_series(dataset, method):
    path = EXP / "new_figure2_spu_seed42_plot_data.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == dataset and row["method"] == method:
                rows.append(row)
    x = [float(r["cumulative_online_secure_seconds"]) for r in rows]
    y = [float(r["development_accuracy"]) for r in rows]
    return np.asarray(x), np.asarray(y)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), sharey=True)
    fig.suptitle(
        "Development accuracy vs cumulative online secure runtime",
        fontsize=13,
        color=INK,
    )
    for ax, dataset in zip(axes, DATASETS):
        ax.set_facecolor("#ffffff")
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color(MUTED)

        # Frozen T@80/90/95 targets (horizontal muted dashed lines).
        t80, t90, t95 = TARGETS[dataset]
        for tval, label in ((t80, "T@80"), (t90, "T@90"), (t95, "T@95")):
            ax.axhline(
                tval, color=MUTED, linestyle="--", linewidth=1.0, alpha=0.7
            )
            ax.text(
                0.015,
                tval,
                label,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=7,
                color=MUTED,
            )

        # Full (blue) and Ours (red).
        x_full, y_full = load_series(dataset, "full_data")
        x_ours, y_ours = load_series(dataset, "true_ours")
        if len(x_full):
            ax.plot(
                x_full, y_full, color=FULL, linewidth=2.0, label="Full", zorder=3
            )
            ax.scatter(x_full[-1], y_full[-1], color=FULL, s=28, zorder=4)
        if len(x_ours):
            ax.plot(
                x_ours, y_ours, color=OURS, linewidth=2.0, label="Ours", zorder=3
            )
            ax.scatter(x_ours[-1], y_ours[-1], color=OURS, s=28, zorder=4)

        ax.set_xscale("log")
        ax.set_xlim(1e2, 8e3)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_major_locator(
            mticker.LogLocator(base=10, numticks=5)
        )
        ax.set_xlabel("Cumulative online secure runtime (seconds)", fontsize=10)
        ax.set_title(dataset, fontsize=12, color=INK)
        ax.tick_params(colors=INK)

        # Tiny Full negative-stop annotation.
        if dataset == "tiny":
            ax.text(
                0.97,
                0.05,
                "Tiny Full: NEGATIVE STOP\n(not reproducible on SPU)",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color=MUTED,
            )

        if dataset == "cifar10":
            ax.set_ylabel("Development validation accuracy", fontsize=10)
            ax.legend(loc="lower right", fontsize=9, frameon=False)

    fig.tight_layout()
    png = FIG / "new_figure2_spu_seed42_secure_time.png"
    pdf = FIG / "new_figure2_spu_seed42_secure_time.pdf"
    fig.savefig(png, dpi=200, facecolor="#ffffff")
    fig.savefig(pdf, facecolor="#ffffff")
    print("saved", png)
    print("saved", pdf)


if __name__ == "__main__":
    main()

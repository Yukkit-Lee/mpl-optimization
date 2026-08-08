#!/usr/bin/env python3
"""Generate the optional 2x3 dual-backend candidate figure (seed42).

Row 1 = TFE / ABY3 (frozen timelines, read-only), row 2 = SecretFlow-SPU
(this task). Columns = CIFAR-10 / CIFAR-100 / Tiny-ImageNet. Full blue, Ours
red, cumulative online secure runtime on a log x-axis.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "new_output/new_secretflow_spu"
FIG = ROOT / "new_expFig/new_secretflow_spu"

FULL = "#2a78d6"
OURS = "#e34948"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"

DATASETS = ("cifar10", "cifar100", "tiny")
TFE_TL = {
    "cifar10": {
        "true_ours": "new_log/new_true_ours/formal_timelines/cifar10/true_ours/new_true_ours_timeline_seed42.json",
        "full_data": "new_log/new_true_ours/formal_timelines/cifar10/full_data/new_true_ours_timeline_seed42.json",
    },
    "cifar100": {
        "true_ours": "new_log/new_true_ours/formal_timelines/cifar100/true_ours/new_true_ours_timeline_seed42.json",
        "full_data": "new_log/new_true_ours/formal_timelines/cifar100/full_data/new_true_ours_timeline_seed42.json",
    },
    "tiny": {
        "true_ours": "new_log/new_true_ours/formal_timelines/tiny/true_ours/new_true_ours_timeline_seed42.json",
        "full_data": "new_log/new_true_ours/formal_timelines/tiny/full_data_corrected_v2/new_true_ours_timeline_seed42.json",
    },
}


def tfe_series(dataset, method):
    path = ROOT / TFE_TL[dataset][method]
    if not path.exists():
        return np.asarray([]), np.asarray([])
    data = json.loads(path.read_text(encoding="utf-8"))
    pts = data["points"]
    x = [float(p["cumulative_online_secure_seconds"]) for p in pts]
    y = [float(p["development_validation_accuracy"]) for p in pts]
    return np.asarray(x), np.asarray(y)


def spu_series(dataset, method):
    path = EXP / "new_figure2_spu_seed42_plot_data.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == dataset and row["method"] == method:
                rows.append(row)
    if not rows:
        return np.asarray([]), np.asarray([])
    x = [float(r["cumulative_online_secure_seconds"]) for r in rows]
    y = [float(r["development_accuracy"]) for r in rows]
    return np.asarray(x), np.asarray(y)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5), sharey="row")
    for row_i, (row_label, series_fn) in enumerate(
        (("TFE / ABY3", tfe_series), ("SecretFlow-SPU", spu_series))
    ):
        for col_i, dataset in enumerate(DATASETS):
            ax = axes[row_i][col_i]
            ax.set_facecolor("#ffffff")
            ax.grid(True, color=GRID, linewidth=0.8, alpha=0.6)
            for spine in ax.spines.values():
                spine.set_color(MUTED)
            for method, color, label in (
                ("full_data", FULL, "Full"),
                ("true_ours", OURS, "Ours"),
            ):
                x, y = series_fn(dataset, method)
                if len(x):
                    ax.plot(
                        x, y, color=color, linewidth=2.0, label=label, zorder=3
                    )
                    ax.scatter(x[-1], y[-1], color=color, s=26, zorder=4)
            ax.set_xscale("log")
            ax.set_xlim(1e2, 3e4)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_major_locator(
                mticker.LogLocator(base=10, numticks=4)
            )
            ax.set_xlabel(
                "Cumulative online secure runtime (seconds)", fontsize=9
            )
            ax.tick_params(labelsize=8, colors=INK)
            if row_i == 0:
                ax.set_title(dataset, fontsize=12, color=INK)
            if col_i == 0:
                ax.set_ylabel(
                    "Development validation accuracy", fontsize=9
                )
            if row_i == 0 and col_i == 0:
                ax.legend(loc="lower right", fontsize=9, frameon=False)
            if dataset == "tiny" and row_label == "SecretFlow-SPU":
                ax.text(
                    0.97,
                    0.05,
                    "Tiny Full: NEGATIVE STOP on SPU",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color=MUTED,
                )
    fig.suptitle(
        "Development accuracy vs cumulative online secure runtime (seed42)",
        fontsize=13,
        color=INK,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = FIG / "new_figure2_dual_backend_2x3_seed42.png"
    pdf = FIG / "new_figure2_dual_backend_2x3_seed42.pdf"
    fig.savefig(png, dpi=200, facecolor="#ffffff")
    fig.savefig(pdf, facecolor="#ffffff")
    print("saved", png)
    print("saved", pdf)


if __name__ == "__main__":
    main()

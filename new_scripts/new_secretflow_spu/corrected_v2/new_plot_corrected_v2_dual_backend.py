#!/usr/bin/env python3
"""corrected-v2 dual-backend 2x3 candidate (handoff §8).

Row 1 = TFE / ABY3 (frozen timelines, read-only), Row 2 = SecretFlow-SPU
(corrected-v2). Columns = CIFAR-10 / CIFAR-100 / Tiny-ImageNet.
Full blue #2C7BB6 / True Ours red #D7191C. n=1 for both rows.
"""

from __future__ import annotations

import csv
import json
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
COLORS = {"full_data": "#2C7BB6", "true_ours": "#D7191C"}
METHOD_LABELS = {"full_data": "Full Data", "true_ours": "True Ours"}

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
    rows = []
    with PLOT_DATA.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == dataset and row["method"] == method:
                rows.append(row)
    if not rows:
        return np.asarray([]), np.asarray([])
    x = [float(r["cumulative_online_secure_seconds"]) for r in rows]
    y = [float(r["development_accuracy"]) for r in rows]
    return np.asarray(x), np.asarray(y)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 10.5), sharey="row")
    for row_i, (row_label, series_fn) in enumerate(
        (("TFE / ABY3", tfe_series), ("SecretFlow-SPU", spu_series))
    ):
        for col_i, dataset in enumerate(DATASETS):
            ax = axes[row_i][col_i]
            for method in ("full_data", "true_ours"):
                x, y = series_fn(dataset, method)
                if len(x):
                    ax.plot(
                        x, y, color=COLORS[method], linewidth=2.2,
                        label="{} (n=1)".format(METHOD_LABELS[method]),
                        marker="o", markersize=3,
                    )
                    ax.scatter([x[-1]], [y[-1]], color=COLORS[method], s=30, zorder=5)
            ax.set_xscale("log")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_xlabel("Cumulative online secure runtime (seconds, log scale)")
            ax.grid(True, which="both", alpha=0.18)
            ax.tick_params(labelsize=8)
            if row_i == 0:
                ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(row_label + "\n\nDevelopment validation accuracy", fontsize=9)
            if row_i == 0 and col_i == 0:
                ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Development accuracy versus cumulative online secure runtime (seed42)",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        (
            "Backend-specific measured online boundaries, not a strict "
            "hardware-normalized speedup comparison. ε is conditional Stage C "
            "only. SPU n=1, no variance shading."
        ),
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for suffix in ("png", "pdf"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 220
        fig.savefig(str(FIG_DIR / "new_figure2_dual_backend_corrected_v2_2x3") + "." + suffix, **kwargs)
    plt.close(fig)
    print("saved dual-backend figures to", FIG_DIR)


if __name__ == "__main__":
    main()

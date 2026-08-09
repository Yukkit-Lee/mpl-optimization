#!/usr/bin/env python3
"""Build paper Figure 6 and its summary table from the frozen formal gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "new_log/new_paper_adaptive/new_adaptive_latest_initializer_formal_plaintext.json"
OUT = ROOT / "new_expFig/paper_experiment"
TABLE = ROOT / "new_expTab/new_paper_adaptive/new_adaptive_latest_initializer_formal_summary.csv"
FIXED_BLUE = "#2C7BB6"
ADAPTIVE_RED = "#D7191C"


def sample_mean_std(values):
    array = np.asarray(values, dtype=float)
    return array.mean(axis=0), array.std(axis=0, ddof=1)


def sparse_i_bars(ax, x, mean, std, color, count=11):
    idx = np.unique(np.linspace(0, len(x) - 1, min(count, len(x)), dtype=int))
    ax.errorbar(
        np.asarray(x)[idx], np.asarray(mean)[idx], yerr=np.asarray(std)[idx],
        fmt="none", ecolor=color, elinewidth=1.0, capsize=3.0,
        capthick=1.0, alpha=0.9, zorder=3,
    )


def rolling_mean(values, window=10):
    values = np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    if payload["status"] != "COMPLETE" or payload["official_test_read"] or payload["locked_holdout_read"]:
        raise ValueError("formal result has an invalid status or evaluation boundary")
    if payload["tfe_formal_authorized_by_gate"]:
        raise ValueError("this builder is for the failed plaintext gate; inspect TFE results instead")

    runs = {}
    for item in payload["detailed"]:
        mode = item["row"]["mode"]
        runs.setdefault(mode, []).append(item)
    if set(runs) != {"fixed_dp", "adaptive_dp"} or any(len(items) != 3 for items in runs.values()):
        raise ValueError("expected exactly three formal seeds for each clipping mode")

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.35))
    for mode, color, label in (
        ("fixed_dp", FIXED_BLUE, "Fixed DP"),
        ("adaptive_dp", ADAPTIVE_RED, "Adaptive DP"),
    ):
        histories = [item["history"] for item in sorted(runs[mode], key=lambda x: x["row"]["seed"])]
        steps = np.asarray([point["step"] for point in histories[0]], dtype=int)

        eval_idx = [i for i, point in enumerate(histories[0]) if point["development_validation_accuracy"] is not None]
        eval_steps = steps[eval_idx]
        accuracy = [[history[i]["development_validation_accuracy"] for i in eval_idx] for history in histories]
        acc_mean, acc_std = sample_mean_std(accuracy)
        axes[0].plot(eval_steps, acc_mean, color=color, linewidth=2.15, label=label)
        sparse_i_bars(axes[0], eval_steps, acc_mean, acc_std, color, count=8)

        bounds = [[point["clipping_bound"] for point in history] for history in histories]
        bound_mean, bound_std = sample_mean_std(bounds)
        axes[1].plot(steps, bound_mean, color=color, linewidth=2.15)
        sparse_i_bars(axes[1], steps, bound_mean, bound_std, color)

        clipped = [[point["fraction_clipped_nonprivate_diagnostic"] for point in history] for history in histories]
        clipped_rolling = [rolling_mean(row) for row in clipped]
        clipped_mean, clipped_std = sample_mean_std(clipped_rolling)
        axes[2].plot(steps, clipped_mean, color=color, linewidth=2.15)
        sparse_i_bars(axes[2], steps, clipped_mean, clipped_std, color)

    titles = ("Development utility", "Clipping-bound trajectory", "Clipped-example diagnostic")
    ylabels = ("Development-validation accuracy", r"Clipping bound $C_t$", "Fraction clipped (10-step mean)")
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Private-update step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylim(0.574, 0.5925)
    axes[1].set_ylim(1.2, 2.08)
    axes[2].set_ylim(-0.015, None)
    handles = [
        Line2D([0], [0], color=FIXED_BLUE, lw=2.2, label="Fixed DP"),
        Line2D([0], [0], color=ADAPTIVE_RED, lw=2.2, label="Adaptive DP"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(r"Matched clipping comparison on CIFAR-10 ($\epsilon=3$)", fontweight="bold")
    fig.text(
        0.5, -0.07,
        "Mean with I-shaped sample-SD bars over frozen RNG seeds 42/43/44. "
        "Clipped fraction is a non-private diagnostic; official test and locked holdout were not read.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "new_figure6_adaptive_clipping_matched_pea"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("fixed_dp", "adaptive_dp"):
        items = sorted(runs[mode], key=lambda x: x["row"]["seed"])
        final = np.asarray([100.0 * item["row"]["final_accuracy"] for item in items])
        auc = np.asarray([100.0 * item["row"]["early_auc"] for item in items])
        final_c = np.asarray([item["row"]["final_c"] for item in items])
        rows.append({
            "mode": mode,
            "n": 3,
            "final_accuracy_mean_percent": float(final.mean()),
            "final_accuracy_sample_std_pp": float(final.std(ddof=1)),
            "early_auc_mean_percent": float(auc.mean()),
            "early_auc_sample_std_pp": float(auc.std(ddof=1)),
            "final_c_mean": float(final_c.mean()),
            "final_c_sample_std": float(final_c.std(ddof=1)),
            "accounted_epsilon": float(items[0]["row"]["accounted_epsilon"]),
            "count_releases": int(items[0]["row"]["count_releases"]),
        })
    with TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

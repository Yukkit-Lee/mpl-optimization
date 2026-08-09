#!/usr/bin/env python3
"""Build the split manuscript Figure 2(a) TFE and Figure 2(b) SPU.

Run from the repository root:

    python new_scripts/new_paper_figures/new_build_figure2.py

Inputs (frozen experimental evidence):
    new_log/new_true_ours/new_true_ours_master_results.json
    new_output/new_secretflow_spu/new_figure2_spu_corrected_v2_plot_data.csv

Outputs:
    new_expFig/paper_experiment/new_figure2a_tfe_pea_i_errorbars.{png,pdf}
    new_expFig/paper_experiment/new_figure2b_spu_short_axis_deltas.{png,pdf}

Safe visual parameters are grouped immediately below.  Figure size, spacing,
line width, marker size, colors, tick precision and error-bar density may be
adjusted without changing results.  Do NOT change input paths, expected release
counts, target definitions, time fields, n values, delta calculation, or the
Stage-B transition field merely to improve appearance: those define the plotted
experimental evidence.  Tiny TFE and all SPU series are n=1, so uncertainty
bars must not be invented for them.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, PercentFormatter

from new_paper_figure_common import (
    DATASETS,
    DATASET_LABELS,
    FULL_BLUE,
    OURS_RED,
    OUT,
    ROOT,
    add_i_errorbars,
    read_csv,
    read_json,
    save_figure,
)


# ------------------------- Safe visual parameters -------------------------
TFE_FIGSIZE = (15.8, 4.5)       # width, height in inches
SPU_FIGSIZE = (16.0, 4.8)
TFE_PANEL_WSPACE = 0.28         # horizontal space between datasets
SPU_DATASET_WSPACE = 0.28
SPU_SHORT_WINDOW_WSPACE = 0.08  # gap between Ours/Full short windows
TFE_ERRORBAR_COUNT = 9          # sparse I-bars per n=3 curve
LINE_WIDTH = 2.15
SPU_MARKER_SIZE = 4.7
SPU_TICK_DECIMALS = 2

# PEA Figure 2 SampleNet-NP (pink) and SampleNet-DP (purple).
SPU_FULL_COLOR = "#E377C2"
SPU_OURS_COLOR = "#9467BD"

# ------------------------- Evidence invariants ----------------------------
TFE_MASTER = ROOT / "new_log/new_true_ours/new_true_ours_master_results.json"
SPU_PLOT_DATA = (
    ROOT
    / "new_output/new_secretflow_spu/new_figure2_spu_corrected_v2_plot_data.csv"
)
EXPECTED_SPU_RELEASES = {"cifar10": 8, "cifar100": 2, "tiny": 18}


def load_and_validate():
    """Load Figure 2 sources and enforce their audited cardinality/scope."""
    master = read_json(TFE_MASTER)
    if master["status"] != "COMPLETE_DEVELOPMENT_ONLY_14_OF_14":
        raise ValueError("TFE master is not complete")
    if master["accuracy_scope"] != "development_validation_only":
        raise ValueError("unexpected TFE accuracy scope")

    grouped = defaultdict(list)
    for row in read_csv(SPU_PLOT_DATA):
        if int(row["n"]) != 1:
            raise ValueError("corrected-v2 SPU plot data must be n=1")
        grouped[(row["dataset"], row["method"])].append(row)
    for dataset in DATASETS:
        for method in ("full_data", "true_ours"):
            rows = grouped[(dataset, method)]
            if len(rows) != EXPECTED_SPU_RELEASES[dataset]:
                raise ValueError(
                    "SPU release cardinality mismatch: {} {}".format(dataset, method)
                )
            times = [float(row["cumulative_online_secure_seconds"]) for row in rows]
            if any(right <= left for left, right in zip(times, times[1:])):
                raise ValueError("SPU cumulative time is not strictly increasing")
    return master, grouped


def build_tfe(master):
    """Build Figure 2(a): original TFE log-seconds accuracy/time curves."""
    fig, axes = plt.subplots(1, 3, figsize=TFE_FIGSIZE)
    for col_idx, dataset in enumerate(DATASETS):
        ax = axes[col_idx]
        targets = list(master["datasets"][dataset]["targets"].values())
        for method, color in (("full_data", FULL_BLUE), ("true_ours", OURS_RED)):
            result = master["datasets"][dataset]["methods"][method]
            curve = result["curve"]
            x = np.asarray(curve["time_seconds"], dtype=float)
            y = np.asarray(curve["accuracy_mean"], dtype=float)
            std = curve["accuracy_sample_std"]
            ax.plot(x, y, color=color, linewidth=LINE_WIDTH)
            ax.scatter([x[-1]], [y[-1]], s=25, color=color, zorder=4)
            if std is not None:
                add_i_errorbars(
                    ax, x, y, np.asarray(std, dtype=float), color,
                    count=TFE_ERRORBAR_COUNT,
                )
            # Experimental field: median transition over the available seeds.
            ax.axvline(
                float(np.median(result["stage_transition_seconds"])),
                color=color,
                linestyle=":",
                linewidth=1.05,
                alpha=0.55,
            )
        for target in targets:
            ax.axhline(target, color="#808080", linestyle="--", linewidth=0.65, alpha=0.42)
        ax.set_xscale("log")
        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xlabel("Cumulative online secure time (s, log scale)")
        ax.set_ylabel("Development-validation accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.16)
        ax.text(
            0.025,
            0.955,
            "TFE / ABY3",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="#444444",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="#BBBBBB",
                alpha=0.88,
            ),
        )

    handles = [
        Line2D([0], [0], color=FULL_BLUE, lw=2.2, label="Full Data"),
        Line2D([0], [0], color=OURS_RED, lw=2.2, label="Ours"),
        Line2D([0], [0], color="#808080", lw=0.8, ls="--", label="T@80/90/95"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        "TFE / ABY3: development accuracy versus cumulative online secure time",
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.07, right=0.985, top=0.86, bottom=0.20,
        wspace=TFE_PANEL_WSPACE,
    )
    save_figure(fig, "new_figure2a_tfe_pea_i_errorbars")


def build_spu(grouped):
    """Build Figure 2(b): SPU short windows and exact local accuracy deltas."""
    fig = plt.figure(figsize=SPU_FIGSIZE)
    outer = fig.add_gridspec(1, 3, wspace=SPU_DATASET_WSPACE)
    for col_idx, dataset in enumerate(DATASETS):
        short_grid = outer[0, col_idx].subgridspec(
            1, 2, wspace=SPU_SHORT_WINDOW_WSPACE
        )
        short_left = fig.add_subplot(short_grid[0, 0])
        short_right = fig.add_subplot(short_grid[0, 1], sharey=short_left)
        delta_series = []

        for ax, method, color, marker, window_label in (
            (short_left, "true_ours", SPU_OURS_COLOR, "D", "Ours short window"),
            (short_right, "full_data", SPU_FULL_COLOR, "s", "Full Data short window"),
        ):
            rows = grouped[(dataset, method)]
            x = np.asarray(
                [float(item["cumulative_online_secure_seconds"]) for item in rows]
            ) / 60.0
            absolute_y = np.asarray(
                [float(item["development_accuracy"]) for item in rows]
            )
            # Experimental transformation: exact percentage-point difference
            # from the first archived release; never smooth or fit this series.
            y = (absolute_y - absolute_y[0]) * 100.0
            delta_series.append(y)
            transition = float(rows[0]["stage_b_to_c_transition_seconds"]) / 60.0
            ax.plot(
                x,
                y,
                color=color,
                linewidth=1.9,
                marker=marker,
                markersize=SPU_MARKER_SIZE,
                markeredgecolor="white",
                markeredgewidth=0.45,
            )
            ax.axvline(transition, color=color, linestyle=":", linewidth=1.05, alpha=0.55)
            window_min = min(float(x.min()), transition)
            window_max = float(x.max())
            span = max(window_max - window_min, 1.0 / 60.0)
            ax.set_xlim(window_min - 0.08 * span, window_max + 0.08 * span)
            ax.axhline(0.0, color="#808080", linestyle="--", linewidth=0.7, alpha=0.5)
            ax.set_xlabel(window_label + "\nCumulative time (min)", fontsize=8)

            # Visual-only tick placement: omit the two labels nearest the break.
            candidate_ticks = np.linspace(window_min, window_max, 4)
            ax.set_xticks(
                candidate_ticks[:-1] if ax is short_left else candidate_ticks[1:]
            )
            ax.xaxis.set_major_formatter(
                FormatStrFormatter("%.{}f".format(SPU_TICK_DECIMALS))
            )
            ax.text(
                0.97, 0.955, "{} releases".format(len(rows)),
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="#555555",
            )
            ax.text(
                0.5,
                0.045,
                "absolute: {:.2%} -> {:.2%}".format(absolute_y[0], absolute_y[-1]),
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=color,
            )

        # Shared delta range inside a dataset preserves Ours/Full comparability.
        delta_min = min(0.0, *(float(values.min()) for values in delta_series))
        delta_max = max(0.0, *(float(values.max()) for values in delta_series))
        delta_pad = max(0.03, 0.18 * max(delta_max - delta_min, 0.01))
        short_left.set_ylim(delta_min - delta_pad, delta_max + delta_pad)
        short_left.text(
            0.03,
            0.955,
            "SecretFlow-SPU",
            transform=short_left.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="#444444",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="#BBBBBB",
                alpha=0.88,
            ),
        )
        short_left.spines["right"].set_visible(False)
        short_right.spines["left"].set_visible(False)
        short_right.tick_params(axis="y", left=False, labelleft=False)
        break_style = dict(color="#555555", clip_on=False, linewidth=1.0)
        short_left.plot((0.975, 1.025), (-0.012, 0.012), transform=short_left.transAxes, **break_style)
        short_left.plot((0.975, 1.025), (0.988, 1.012), transform=short_left.transAxes, **break_style)
        short_right.plot((-0.025, 0.025), (-0.012, 0.012), transform=short_right.transAxes, **break_style)
        short_right.plot((-0.025, 0.025), (0.988, 1.012), transform=short_right.transAxes, **break_style)
        for ax in (short_left, short_right):
            ax.grid(True, alpha=0.16)
        short_left.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        short_left.set_ylabel("Accuracy change from first release (pp)", fontsize=8)
        short_left.set_title(DATASET_LABELS[dataset], fontweight="bold", x=1.04)

    handles = [
        Line2D([0], [0], color=SPU_FULL_COLOR, lw=2.0, marker="s", markersize=5, label="Full Data"),
        Line2D([0], [0], color=SPU_OURS_COLOR, lw=2.0, marker="D", markersize=5, label="Ours"),
        Line2D([0], [0], color="#808080", lw=0.8, ls="--", label="First-release baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("SecretFlow-SPU: local post-Stage-B accuracy change", fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.22)
    save_figure(fig, "new_figure2b_spu_short_axis_deltas")


def main():
    master, grouped = load_and_validate()
    build_tfe(master)
    build_spu(grouped)
    print("PASS built Figure 2(a)/(b) in", OUT)


if __name__ == "__main__":
    main()

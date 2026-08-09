#!/usr/bin/env python3
"""Build manuscript Figures 2--5 from frozen, audited artifacts only."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "new_expFig/paper_experiment"
DATASETS = ("cifar10", "cifar100", "tiny")
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny": "Tiny ImageNet",
}
FULL_BLUE = "#2C7BB6"
OURS_RED = "#D7191C"
ORANGE = "#FDAE61"
SPU_FULL_SAMPLENET_NP = "#E377C2"
SPU_OURS_SAMPLENET_DP = "#9467BD"
DARK = "#333333"


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (stem + ".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUT / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def sparse_indices(length: int, count: int = 9):
    return np.unique(np.linspace(0, length - 1, min(count, length), dtype=int))


def i_errorbars(ax, x, y, std, color, count=9):
    idx = sparse_indices(len(x), count)
    ax.errorbar(
        np.asarray(x)[idx],
        np.asarray(y)[idx],
        yerr=np.asarray(std)[idx],
        fmt="none",
        ecolor=color,
        elinewidth=1.0,
        capsize=3.0,
        capthick=1.0,
        alpha=0.9,
        zorder=3,
    )


def figure2():
    master = read_json(ROOT / "new_log/new_true_ours/new_true_ours_master_results.json")
    if master["status"] != "COMPLETE_DEVELOPMENT_ONLY_14_OF_14":
        raise ValueError("TFE master is not complete")
    if master["accuracy_scope"] != "development_validation_only":
        raise ValueError("unexpected TFE accuracy scope")
    spu_rows = read_csv(
        ROOT / "new_output/new_secretflow_spu/new_figure2_spu_corrected_v2_plot_data.csv"
    )
    grouped = defaultdict(list)
    for row in spu_rows:
        if int(row["n"]) != 1:
            raise ValueError("corrected-v2 SPU plot data must be n=1")
        grouped[(row["dataset"], row["method"])].append(row)
    expected_releases = {"cifar10": 8, "cifar100": 2, "tiny": 18}
    for dataset in DATASETS:
        for method in ("full_data", "true_ours"):
            rows = grouped[(dataset, method)]
            if len(rows) != expected_releases[dataset]:
                raise ValueError("SPU release cardinality mismatch: {} {}".format(dataset, method))
            times = [float(row["cumulative_online_secure_seconds"]) for row in rows]
            if any(right <= left for left, right in zip(times, times[1:])):
                raise ValueError("SPU cumulative time is not strictly increasing")

    # Figure 2(a): retain the original TFE accuracy/time visual grammar.
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.5))
    for col_idx, dataset in enumerate(DATASETS):
        top = axes[col_idx]
        targets = list(master["datasets"][dataset]["targets"].values())
        for method, color in (("full_data", FULL_BLUE), ("true_ours", OURS_RED)):
            result = master["datasets"][dataset]["methods"][method]
            curve = result["curve"]
            x = np.asarray(curve["time_seconds"], dtype=float)
            y = np.asarray(curve["accuracy_mean"], dtype=float)
            std = curve["accuracy_sample_std"]
            top.plot(x, y, color=color, linewidth=2.15, linestyle="-")
            top.scatter([x[-1]], [y[-1]], s=25, color=color, zorder=4)
            if std is not None:
                i_errorbars(top, x, y, np.asarray(std, dtype=float), color)
            top.axvline(
                float(np.median(result["stage_transition_seconds"])),
                color=color, linestyle=":", linewidth=1.05, alpha=0.55,
            )
        for target in targets:
            top.axhline(target, color="#808080", linestyle="--", linewidth=0.65, alpha=0.42)
        top.set_xscale("log")
        top.set_title(DATASET_LABELS[dataset], fontweight="bold")
        top.set_xlabel("Cumulative online secure time (s, log scale)")
        top.text(
            0.025, 0.955, "TFE / ABY3", transform=top.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold", color="#444444",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#BBBBBB", alpha=0.88),
        )
        top.yaxis.set_major_formatter(PercentFormatter(1.0))
        top.grid(True, alpha=0.16)
        top.set_ylabel("Development-validation accuracy")
    tfe_handles = [
        Line2D([0], [0], color=FULL_BLUE, lw=2.2, label="Full Data"),
        Line2D([0], [0], color=OURS_RED, lw=2.2, label="Ours"),
        Line2D([0], [0], color="#808080", lw=0.8, ls="--", label="T@80/90/95"),
    ]
    fig.legend(tfe_handles, [h.get_label() for h in tfe_handles], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("TFE / ABY3: development accuracy versus cumulative online secure time", fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.20, wspace=0.28)
    save(fig, "new_figure2a_tfe_pea_i_errorbars")

    # Figure 2(b): SPU gets its own y semantics and layout.  PEA Figure 2's
    # SampleNet-NP pink and SampleNet-DP purple are used for Full and Ours.
    fig = plt.figure(figsize=(16.0, 4.8))
    outer = fig.add_gridspec(1, 3, wspace=0.28)
    for col_idx, dataset in enumerate(DATASETS):
        short_grid = outer[0, col_idx].subgridspec(1, 2, wspace=0.08)
        short_left = fig.add_subplot(short_grid[0, 0])
        short_right = fig.add_subplot(short_grid[0, 1], sharey=short_left)
        # SPU's Stage-B cost separates Ours and Full by tens to hundreds of
        # minutes, while each genuine Stage-C trace spans only a few minutes.
        # Two adjacent short windows preserve the absolute time values and make
        # every archived release visible.  The diagonal marks explicitly denote
        # the omitted interval; no observation is interpolated across it.
        delta_series = []
        for ax, method, color, marker, window_label in (
            (short_left, "true_ours", SPU_OURS_SAMPLENET_DP, "D", "Ours short window"),
            (short_right, "full_data", SPU_FULL_SAMPLENET_NP, "s", "Full Data short window"),
        ):
            rows = grouped[(dataset, method)]
            if not rows:
                raise ValueError("missing SPU series {} {}".format(dataset, method))
            x = np.asarray(
                [float(item["cumulative_online_secure_seconds"]) for item in rows]
            ) / 60.0
            absolute_y = np.asarray(
                [float(item["development_accuracy"]) for item in rows]
            )
            # The absolute SPU accuracy changes by only 0.02--0.44pp after
            # Stage B, so a full-scale accuracy axis necessarily looks flat.
            # Show the genuine local change in percentage points relative to
            # the first archived release, while printing the absolute start and
            # final accuracies in each window.  No smoothing or rescaling of x.
            y = (absolute_y - absolute_y[0]) * 100.0
            delta_series.append(y)
            transition = float(rows[0]["stage_b_to_c_transition_seconds"]) / 60.0
            ax.plot(
                x, y, color=color, linewidth=1.9, linestyle="-", marker=marker,
                markersize=4.7, markeredgecolor="white", markeredgewidth=0.45,
            )
            ax.axvline(transition, color=color, linestyle=":", linewidth=1.05, alpha=0.55)
            window_min = min(float(x.min()), transition)
            window_max = float(x.max())
            span = max(window_max - window_min, 1.0 / 60.0)
            ax.set_xlim(window_min - 0.08 * span, window_max + 0.08 * span)
            ax.axhline(0.0, color="#808080", linestyle="--", linewidth=0.7, alpha=0.5)
            ax.set_xlabel(window_label + "\nCumulative time (min)", fontsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.text(
                0.97, 0.955, "{} releases".format(len(rows)), transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#555555",
            )
            ax.text(
                0.5, 0.045,
                "absolute: {:.2%} -> {:.2%}".format(absolute_y[0], absolute_y[-1]),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8,
                color=color,
            )
        delta_min = min(0.0, *(float(values.min()) for values in delta_series))
        delta_max = max(0.0, *(float(values.max()) for values in delta_series))
        delta_pad = max(0.03, 0.18 * max(delta_max - delta_min, 0.01))
        short_left.set_ylim(delta_min - delta_pad, delta_max + delta_pad)
        short_left.text(
            0.03, 0.955, "SecretFlow-SPU", transform=short_left.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold", color="#444444",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#BBBBBB", alpha=0.88),
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
    spu_handles = [
        Line2D([0], [0], color=SPU_FULL_SAMPLENET_NP, lw=2.0, marker="s", markersize=5, label="Full Data"),
        Line2D([0], [0], color=SPU_OURS_SAMPLENET_DP, lw=2.0, marker="D", markersize=5, label="Ours"),
        Line2D([0], [0], color="#808080", lw=0.8, ls="--", label="First-release baseline"),
    ]
    fig.legend(spu_handles, [h.get_label() for h in spu_handles], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("SecretFlow-SPU: local post-Stage-B accuracy change", fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.22)
    save(fig, "new_figure2b_spu_short_axis_deltas")


def module_a_runs():
    rows = []
    root = ROOT / "new_log/new_public_mtt_mainline/json/new_module_a_runs"
    for dataset in DATASETS:
        for seed in (42, 43, 44):
            run = read_json(root / dataset / "new_module_a_efficiency_{}_seed{}.json".format(dataset, seed))
            if run["status"] != "COMPLETE" or run["test_usage"] != "fixed_interval_readonly_formal_reporting":
                raise ValueError("unexpected Module-A reporting scope")
            rows.append(run)
    return rows


def figure3():
    runs = module_a_runs()
    methods = ("full_images", "random_subset", "official_mtt")
    labels = {"full_images": "Full Data", "random_subset": "Random Subset", "official_mtt": "Ours"}
    colors = {"full_images": FULL_BLUE, "random_subset": ORANGE, "official_mtt": OURS_RED}
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5))
    for ax, dataset in zip(axes, DATASETS):
        dataset_runs = [run for run in runs if run["dataset"] == dataset]
        target = float(dataset_runs[0]["results"][0]["target_accuracy"])
        for method in methods:
            results = [next(item for item in run["results"] if item["source"] == method) for run in dataset_runs]
            steps = np.asarray([point["step"] for point in results[0]["history"]], dtype=int)
            values = np.asarray([[point["test_accuracy"] for point in item["history"]] for item in results], dtype=float)
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            if method == "official_mtt":
                # A convergence stop is stronger than merely crossing the
                # frozen target.  For each seed, find the first checkpoint
                # after which every archived future value stays within 0.5pp
                # of that seed's terminal accuracy; use the latest of the
                # three boundaries.
                seed_cuts = []
                for item in results:
                    seed_values = np.asarray([point["test_accuracy"] for point in item["history"]], dtype=float)
                    terminal = seed_values[-1]
                    valid = [
                        idx for idx in range(len(steps))
                        if np.all(np.abs(seed_values[idx:] - terminal) <= 0.005)
                    ]
                    if not valid:
                        raise ValueError("Ours has no stable 0.5pp convergence window")
                    seed_cuts.append(int(steps[valid[0]]))
                convergence_step = max(seed_cuts)
                mask = steps <= convergence_step
                steps, mean, std = steps[mask], mean[mask], std[mask]
            ax.plot(steps, mean, color=colors[method], linewidth=2.1, label=labels[method])
            i_errorbars(ax, steps, mean, std, colors[method], count=7)
            if method == "official_mtt":
                ax.scatter([steps[-1]], [mean[-1]], color=OURS_RED, marker="X", s=48, zorder=5)
                ax.annotate(
                    "converged\n{} steps".format(int(steps[-1])),
                    (steps[-1], mean[-1]), xytext=(-7, 11), textcoords="offset points",
                    ha="right", fontsize=7, color=OURS_RED,
                )
        ax.axhline(target, color="#777777", ls="--", lw=0.8, alpha=0.6)
        ax.text(0.98, target, "frozen target", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color="#666666")
        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xlabel("Optimizer steps")
        ax.grid(alpha=0.18)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Official-test accuracy")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(
        0.5,
        -0.065,
        "Mean with I-shaped sample-SD bars over seeds 42/43/44. Ours stops only after every seed remains within 0.5pp of its terminal accuracy (X); no extrapolation.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "new_figure3_compact_carrier_fast_convergence")


def anchor_sources():
    rows = read_csv(ROOT / "new_expTab/new_stage_b_anchor_postagg/new_stage_b_anchor_postagg_init_source_ablation.csv")
    if any(row["test_read"] != "False" or row["locked_holdout_read"] != "False" for row in rows):
        raise ValueError("Stage-B plot source read test/holdout")
    return rows


def lookup(rows, dataset, distribution, method):
    matches = [row for row in rows if row["dataset"] == dataset and row["distribution"] == distribution and row["method"] == method]
    if len(matches) != 1:
        raise ValueError("non-unique ablation row")
    return float(matches[0]["accuracy"])


def figure4():
    rows = anchor_sources()
    methods = (
        "C1_random_plus_anchor",
        "C3_fixed_local0_plus_anchor",
        "C4_secure_average",
        "C5_secure_average_plus_anchor",
    )
    labels = ("Random + Anchor", "Local-0 + Anchor", "Secure Average", "Average + Anchor")
    colors = ("#2C7BB6", "#FDAE61", "#8C8C8C", "#D7191C")
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 7.7), sharex="row", sharey="row")
    for row_idx, dataset in enumerate(("cifar10", "cifar100")):
        for col_idx, distribution in enumerate(("iid", "mild", "severe")):
            ax = axes[row_idx, col_idx]
            values = np.asarray([100.0 * lookup(rows, dataset, distribution, method) for method in methods])
            xpos = np.arange(len(methods))
            bars = ax.bar(xpos, values, color=colors, width=0.68, edgecolor="white", linewidth=0.7)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2.0, value + (0.8 if dataset == "cifar10" else 0.3), "{:.1f}".format(value), ha="center", va="bottom", fontsize=7.2)
            ax.set_title(distribution.capitalize(), fontweight="bold")
            ax.grid(axis="y", alpha=0.16)
            if col_idx == 0:
                ax.set_ylabel(DATASET_LABELS[dataset] + "\nDevelopment accuracy (%)")
            ax.set_xticks(xpos)
            ax.set_xticklabels(("Random\n+Anchor", "Local-0\n+Anchor", "Secure\nAverage", "Average\n+Anchor"), fontsize=7)
            ax.set_ylim(0, 66 if dataset == "cifar10" else 25)
            ax.text(0.98, 0.04, "n=1", transform=ax.transAxes, ha="right", fontsize=7, color="#666666")
    fig.suptitle("Initializer-source ablation under heterogeneous partitions", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for color, label in zip(colors, labels)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.text(0.5, -0.042, "Seed 42 (n=1); no uncertainty estimate. Red denotes the proposed secure-average-plus-anchor initializer.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    save(fig, "new_figure4_initializer_source_bars")


def figure5():
    trajectories = read_json(ROOT / "new_log/new_stage_b_anchor_postagg/new_stage_b_anchor_postagg_trajectories.json")
    if trajectories["status"] != "COMPLETE" or trajectories["test_read"] or trajectories["locked_holdout_read"]:
        raise ValueError("invalid Stage-B trajectories")
    method_colors = {
        "C1_random_plus_anchor": "#636363",
        "C3_fixed_local0_plus_anchor": FULL_BLUE,
        "C5_secure_average_plus_anchor": OURS_RED,
    }
    method_labels = {
        "C1_random_plus_anchor": "Random + Anchor",
        "C3_fixed_local0_plus_anchor": "Local-0 + Anchor",
        "C5_secure_average_plus_anchor": "Average + Anchor",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.0), sharex=True, sharey="row")
    for row_idx, dataset in enumerate(("cifar10", "cifar100")):
        for col_idx, distribution in enumerate(("iid", "mild", "severe")):
            ax = axes[row_idx, col_idx]
            matches = [item for item in trajectories["rows"] if item["dataset"] == dataset and item["distribution"] == distribution]
            if len(matches) != 3:
                raise ValueError("expected three Stage-B methods per panel")
            for item in matches:
                steps = [point["step"] for point in item["history"]]
                values = [100.0 * point["development_accuracy"] for point in item["history"]]
                ax.plot(steps, values, color=method_colors[item["method"]], linewidth=2.0, label=method_labels[item["method"]])
                ax.scatter([steps[-1]], [values[-1]], color=method_colors[item["method"]], s=18)
            ax.set_title(distribution.capitalize(), fontweight="bold")
            ax.set_xlabel("Anchor-refinement steps")
            ax.grid(alpha=0.17)
            if col_idx == 0:
                ax.set_ylabel(DATASET_LABELS[dataset] + "\nDevelopment accuracy (%)")
            ax.text(0.98, 0.04, "seed 42 (n=1)", transform=ax.transAxes, ha="right", fontsize=7, color="#666666")
    handles = [Line2D([0], [0], color=color, lw=2.1, label=method_labels[method]) for method, color in method_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Same-budget anchor-refinement convergence", fontweight="bold")
    fig.text(0.5, -0.045, "Only seed 42 trajectories exist, so no uncertainty bars are drawn.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save(fig, "new_figure5_anchor_convergence_pea")


def main():
    figure2()
    figure3()
    figure4()
    figure5()
    print("PASS built manuscript Figures 2--5 in", OUT)


if __name__ == "__main__":
    main()

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
from matplotlib.ticker import MaxNLocator, PercentFormatter


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
SPU_FULL_ORANGE = "#FF7F0E"
SPU_OURS_CYAN = "#17BECF"
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

    fig, axes = plt.subplots(2, 3, figsize=(15.8, 8.6), sharey="col")
    for col_idx, dataset in enumerate(DATASETS):
        top = axes[0, col_idx]
        bottom = axes[1, col_idx]
        targets = list(master["datasets"][dataset]["targets"].values())
        for method, color in (("full_data", FULL_BLUE), ("true_ours", OURS_RED)):
            result = master["datasets"][dataset]["methods"][method]
            curve = result["curve"]
            x = np.asarray(curve["time_seconds"], dtype=float) / 60.0
            y = np.asarray(curve["accuracy_mean"], dtype=float)
            std = curve["accuracy_sample_std"]
            top.plot(x, y, color=color, linewidth=2.15, linestyle="-")
            top.scatter([x[-1]], [y[-1]], s=25, color=color, zorder=4)
            if std is not None:
                i_errorbars(top, x, y, np.asarray(std, dtype=float), color)
        for target in targets:
            top.axhline(target, color="#808080", linestyle="--", linewidth=0.65, alpha=0.42)
        top.set_title(DATASET_LABELS[dataset], fontweight="bold")
        top.set_xlabel("Cumulative online secure time (min)")
        top.text(
            0.025, 0.955, "TFE / ABY3", transform=top.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold", color="#444444",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#BBBBBB", alpha=0.88),
        )

        t90_minutes = {}
        for method, color, marker, label in (
            ("full_data", SPU_FULL_ORANGE, "s", "Full Data"),
            ("true_ours", SPU_OURS_CYAN, "D", "Ours"),
        ):
            rows = grouped[(dataset, method)]
            if not rows:
                raise ValueError("missing SPU series {} {}".format(dataset, method))
            times = np.asarray(
                [float(item["cumulative_online_secure_seconds"]) for item in rows]
            ) / 60.0
            accuracies = np.asarray(
                [float(item["development_accuracy"]) for item in rows]
            )
            target_t90 = float(rows[0]["target_T90"])
            reached = times[accuracies >= target_t90]
            if len(reached) == 0:
                raise ValueError("SPU series does not reach T@90: {} {}".format(dataset, method))
            t90_minutes[method] = float(reached[0])

            # The corrected SPU traces begin after the expensive Stage B and
            # are already essentially converged (only 0.02--0.44pp variation).
            # Plot the genuine final utility/time endpoint rather than joining
            # near-constant releases into a visually misleading curve.
            final_x = float(times[-1])
            final_y = float(accuracies[-1])
            bottom.scatter(
                [final_x], [final_y], color=color, marker=marker, s=78,
                edgecolor="white", linewidth=0.65, zorder=5,
            )
            offset = (-8, 0) if method == "full_data" else (8, 0)
            bottom.annotate(
                "{}\n{:.1f} min, {:.2%}".format(label, final_x, final_y),
                (final_x, final_y), xytext=offset, textcoords="offset points",
                ha="right" if method == "full_data" else "left",
                va="center", fontsize=7.2, color=color, fontweight="bold",
            )
        for target in [
            float(grouped[(dataset, "full_data")][0][key])
            for key in ("target_T80", "target_T90", "target_T95")
        ]:
            bottom.axhline(target, color="#808080", linestyle="--", linewidth=0.65, alpha=0.42)
        speedup = t90_minutes["full_data"] / t90_minutes["true_ours"]
        bottom.set_xlabel("Total cumulative online secure time (min; lower is better)")
        bottom.text(
            0.025, 0.955, "SecretFlow-SPU endpoints", transform=bottom.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold", color="#444444",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#BBBBBB", alpha=0.88),
        )
        bottom.text(
            0.975, 0.955, "Ours T@90: {:.2f}x faster".format(speedup),
            transform=bottom.transAxes, ha="right", va="top", fontsize=8,
            fontweight="bold", color=SPU_OURS_CYAN,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=SPU_OURS_CYAN, alpha=0.88),
        )

        for ax in (top, bottom):
            # Both rows use measured cumulative online secure time in minutes.
            # Plain linear ticks avoid broken axes and scientific-offset labels.
            ax.set_xscale("linear")
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(True, alpha=0.16)
        if col_idx == 0:
            top.set_ylabel("Development-validation accuracy")
            bottom.set_ylabel("Development-validation accuracy")
    handles = [
        Line2D([0], [0], color=FULL_BLUE, lw=2.2, ls="-", label="TFE Full Data"),
        Line2D([0], [0], color=OURS_RED, lw=2.2, ls="-", label="TFE Ours"),
        Line2D([0], [0], color=SPU_FULL_ORANGE, lw=0, marker="s", markersize=7, label="SPU Full Data endpoint"),
        Line2D([0], [0], color=SPU_OURS_CYAN, lw=0, marker="D", markersize=7, label="SPU Ours endpoint"),
        Line2D([0], [0], color="#808080", lw=0.8, ls="--", label="T@80/90/95"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.002))
    fig.suptitle("Secure-training utility versus cumulative online time", fontweight="bold")
    fig.text(
        0.5,
        -0.047,
        "I-bars show sample standard deviation for TFE n=3; Tiny TFE and all SPU panels are n=1. "
        "Top: complete TFE accuracy-time trajectories. Bottom: genuine final SPU utility/time endpoints; near-flat post-Stage-B releases are not joined into pseudo-convergence curves. "
        "All x values are measured cumulative online secure minutes on linear, unbroken axes; T@Target values are in the audited companion table. Backend rows are not hardware-normalized.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.13)
    save(fig, "new_figure2_dual_backend_pea_i_errorbars")


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

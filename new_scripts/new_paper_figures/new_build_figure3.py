#!/usr/bin/env python3
"""生成论文 Figure 3（紧凑载体的收敛曲线）。

在仓库根目录运行：

    python new_scripts/new_paper_figures/new_build_figure3.py

输入：以下目录中的 9 个冻结 JSON 文件：
``new_log/new_public_mtt_mainline/json/new_module_a_runs/{dataset}/``.
输出：``new_expFig/paper_experiment/`` 下的
``new_figure3_compact_carrier_fast_convergence.{png,pdf}``.

下方可安全调整的视觉参数包括图尺寸、颜色、线宽和 I 型误差条密度。
``CONVERGENCE_TOLERANCE`` 不是纯视觉参数：它定义了经审计的停止标注，
即每个 seed 此后的全部归档值都必须处于其终值的 0.5 个百分点范围内。
若不同时修改论文方法/图注并重新审计结论，不得修改该值。本绘图脚本也
不得改动冻结目标、seed、历史轨迹或 official-test 的使用范围。
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


# 可安全调整的视觉参数。
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
ANNOTATION_POSITIONS = {
    "cifar10": (0.31, 0.96),
    "cifar100": (0.32, 0.96),
    "tiny": (0.24, 0.95),
}

# 已审计的科学语义参数：0.005 accuracy 等于 0.5 个百分点。
CONVERGENCE_TOLERANCE = 0.005
SEEDS = (42, 43, 44)
RUN_ROOT = ROOT / "new_log/new_public_mtt_mainline/json/new_module_a_runs"


def load_runs():
    """读取严格限定的 3 个数据集 x 3 个 seed，并检查报告适用范围。"""
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
                # 对每个 seed，找到首个满足以下条件的 checkpoint：其后的全部
                # 归档值都保持在该 seed 终值的审计容差内；最后取三个 seed 中
                # 最晚的边界作为图中 Ours 的收敛停止点。
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
                    xytext=ANNOTATION_POSITIONS[dataset],
                    textcoords="axes fraction",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=OURS_RED,
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.92,
                    ),
                    arrowprops=dict(
                        arrowstyle="-",
                        color=OURS_RED,
                        linewidth=0.7,
                        shrinkA=1.5,
                        shrinkB=3.0,
                    ),
                )
        ax.axhline(target, color="#777777", ls="--", lw=0.8, alpha=0.6)
        ax.text(
            0.98, target, "frozen target", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="#666666",
        )
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 0.08 * (ymax - ymin))
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
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "new_figure3_compact_carrier_fast_convergence")
    print("PASS built Figure 3 in", OUT)


if __name__ == "__main__":
    main()

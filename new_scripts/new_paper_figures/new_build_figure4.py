#!/usr/bin/env python3
"""生成论文 Figure 4（初始化来源分组柱状图）。

在仓库根目录运行：

    python new_scripts/new_paper_figures/new_build_figure4.py

输入：
    new_expTab/new_stage_b_anchor_postagg/
    new_stage_b_anchor_postagg_init_source_ablation.csv
输出：
    new_expFig/paper_experiment/new_figure4_initializer_source_bars.{png,pdf}

下方可安全调整的视觉参数包括图尺寸、颜色、柱宽、数值标签偏移和纵轴显示
范围。方法顺序、数据集/分布面板、准确率字段以及 test/holdout 门属于实验
语义，禁止为使某个方法看起来更好而修改。源数据只有 seed42（n=1），若无
新的独立重复实验，不得添加误差条。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from new_paper_figure_common import DATASET_LABELS, OUT, ROOT, read_csv, save_figure


# 可安全调整的视觉参数。
FIGSIZE = (14.6, 7.7)
BAR_WIDTH = 0.68
COLORS = ("#2C7BB6", "#FDAE61", "#8C8C8C", "#D7191C")
Y_LIMITS = {"cifar10": 66, "cifar100": 25}
VALUE_LABEL_OFFSETS = {"cifar10": 0.8, "cifar100": 0.3}

# 已审计的面板与方法定义；修改这些内容会改变图的科学比较对象。
DATASETS = ("cifar10", "cifar100")
DISTRIBUTIONS = ("iid", "mild", "severe")
METHODS = (
    "C1_random_plus_anchor",
    "C3_fixed_local0_plus_anchor",
    "C4_secure_average",
    "C5_secure_average_plus_anchor",
)
LABELS = (
    "Random Init. + Anchor",
    "Single-Party Init. + Anchor",
    "Secure Average Only",
    "Secure Average + Anchor",
)
SHORT_LABELS = (
    "Random Init.\n+ Anchor",
    "Single-Party Init.\n+ Anchor",
    "Secure Average\nOnly",
    "Secure Average\n+ Anchor",
)
SOURCE = (
    ROOT
    / "new_expTab/new_stage_b_anchor_postagg/"
    "new_stage_b_anchor_postagg_init_source_ablation.csv"
)


def load_rows():
    """读取消融表，并确认绘图数据未读取 test 或 locked holdout。"""
    rows = read_csv(SOURCE)
    if any(
        row["test_read"] != "False" or row["locked_holdout_read"] != "False"
        for row in rows
    ):
        raise ValueError("Stage-B plot source read test/holdout")
    return rows


def lookup(rows, dataset, distribution, method):
    """按数据集、分布和方法精确查找唯一一条准确率记录。"""
    matches = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["distribution"] == distribution
        and row["method"] == method
    ]
    if len(matches) != 1:
        raise ValueError("non-unique ablation row")
    return float(matches[0]["accuracy"])


def main():
    rows = load_rows()
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, sharex="row", sharey="row")
    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, distribution in enumerate(DISTRIBUTIONS):
            ax = axes[row_idx, col_idx]
            values = np.asarray(
                [
                    100.0 * lookup(rows, dataset, distribution, method)
                    for method in METHODS
                ]
            )
            xpos = np.arange(len(METHODS))
            bars = ax.bar(
                xpos,
                values,
                color=COLORS,
                width=BAR_WIDTH,
                edgecolor="white",
                linewidth=0.7,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + VALUE_LABEL_OFFSETS[dataset],
                    "{:.1f}".format(value),
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                )
            ax.set_title(distribution.capitalize(), fontweight="bold")
            ax.grid(axis="y", alpha=0.16)
            if col_idx == 0:
                ax.set_ylabel(
                    DATASET_LABELS[dataset] + "\nDevelopment accuracy (%)"
                )
            ax.set_xticks(xpos)
            ax.set_xticklabels(SHORT_LABELS, fontsize=7)
            ax.set_ylim(0, Y_LIMITS[dataset])
    fig.suptitle(
        "Initializer-source ablation under heterogeneous partitions",
        fontweight="bold",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for color, label in zip(COLORS, LABELS)
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    save_figure(fig, "new_figure4_initializer_source_bars")
    print("PASS built Figure 4 in", OUT)


if __name__ == "__main__":
    main()

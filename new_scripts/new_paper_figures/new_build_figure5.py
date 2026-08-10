#!/usr/bin/env python3
"""生成论文 Figure 5（相同预算下的 anchor 收敛曲线）。

在仓库根目录运行：

    python new_scripts/new_paper_figures/new_build_figure5.py

输入：
    new_log/new_stage_b_anchor_postagg/
    new_stage_b_anchor_postagg_trajectories.json
输出：
    new_expFig/paper_experiment/new_figure5_anchor_convergence_pea.{png,pdf}

下方可安全调整的视觉参数包括图尺寸、方法颜色、线宽和终点标记大小。所选
方法、历史 step、开发集准确率、数据集/分布面板以及隐私门属于实验语义。
本绘图脚本不得对历史轨迹进行平滑、插值或截断。当前只有 seed42 的轨迹，
因此在没有新增独立重复实验时，添加不确定性误差条在科学上是不成立的。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from new_paper_figure_common import (
    DATASET_LABELS,
    FULL_BLUE,
    OURS_RED,
    OUT,
    ROOT,
    read_json,
    save_figure,
)


# 可安全调整的视觉参数。
FIGSIZE = (14.2, 7.0)
LINE_WIDTH = 2.0
ENDPOINT_SIZE = 18

# 已审计的方法与面板；修改这些内容会改变科学比较对象。
DATASETS = ("cifar10", "cifar100")
DISTRIBUTIONS = ("iid", "mild", "severe")
METHOD_COLORS = {
    "C1_random_plus_anchor": "#636363",
    "C3_fixed_local0_plus_anchor": FULL_BLUE,
    "C5_secure_average_plus_anchor": OURS_RED,
}
METHOD_LABELS = {
    "C1_random_plus_anchor": "Random Init. + Anchor",
    "C3_fixed_local0_plus_anchor": "Single-Party Init. + Anchor",
    "C5_secure_average_plus_anchor": "Secure Average + Anchor",
}
SOURCE = (
    ROOT
    / "new_log/new_stage_b_anchor_postagg/"
    "new_stage_b_anchor_postagg_trajectories.json"
)


def load_trajectories():
    """读取 Stage-B 轨迹，并检查完成状态及 test/holdout 隐私门。"""
    trajectories = read_json(SOURCE)
    if (
        trajectories["status"] != "COMPLETE"
        or trajectories["test_read"]
        or trajectories["locked_holdout_read"]
    ):
        raise ValueError("invalid Stage-B trajectories")
    return trajectories


def main():
    trajectories = load_trajectories()
    fig, axes = plt.subplots(
        2, 3, figsize=FIGSIZE, sharex=True, sharey="row"
    )
    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, distribution in enumerate(DISTRIBUTIONS):
            ax = axes[row_idx, col_idx]
            matches = [
                item
                for item in trajectories["rows"]
                if item["dataset"] == dataset
                and item["distribution"] == distribution
            ]
            if len(matches) != 3:
                raise ValueError("expected three Stage-B methods per panel")
            for item in matches:
                steps = [point["step"] for point in item["history"]]
                values = [
                    100.0 * point["development_accuracy"]
                    for point in item["history"]
                ]
                ax.plot(
                    steps,
                    values,
                    color=METHOD_COLORS[item["method"]],
                    linewidth=LINE_WIDTH,
                    label=METHOD_LABELS[item["method"]],
                )
                ax.scatter(
                    [steps[-1]], [values[-1]],
                    color=METHOD_COLORS[item["method"]], s=ENDPOINT_SIZE,
                )
            ax.set_title(distribution.capitalize(), fontweight="bold")
            ax.set_xlabel("Anchor-refinement steps")
            ax.grid(alpha=0.17)
            if col_idx == 0:
                ax.set_ylabel(
                    DATASET_LABELS[dataset] + "\nDevelopment accuracy (%)"
                )
    handles = [
        Line2D(
            [0], [0], color=color, lw=2.1, label=METHOD_LABELS[method]
        )
        for method, color in METHOD_COLORS.items()
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle("Same-budget anchor-refinement convergence", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, "new_figure5_anchor_convergence_pea")
    print("PASS built Figure 5 in", OUT)


if __name__ == "__main__":
    main()

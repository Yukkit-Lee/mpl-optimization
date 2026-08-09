#!/usr/bin/env python3
"""绘制论文 Figure 6：三个数据集的独立配置 Fixed/Adaptive DP 对比。

运行方式（在仓库根目录执行）：

    /root/autodl-tmp/new_envs/distillation/bin/python \
        new_scripts/new_paper_figures/new_build_figure6.py

默认输入：

    new_log/new_paper_adaptive/
        new_user_tuned_fixed_vs_adaptive_multidataset_formal.json

默认输出：

    new_expFig/paper_experiment/
        new_figure6_user_tuned_three_datasets.{png,pdf}
    new_expTab/new_paper_adaptive/
        new_figure6_user_tuned_three_datasets_summary.csv

图的含义：每个数据集一个面板；左轴画 Fixed/Adaptive 的开发集准确率，
右轴只画 Adaptive 的动态裁剪阈值 C_t。横轴是私有更新步，不是运行时间。
脚本不校验输入 JSON 的 SHA256，但会检查状态、数据集、方法、seed、epsilon
和 test/holdout 门，防止误画不完整或越界数据。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from new_paper_figure_common import FULL_BLUE, OURS_RED, ROOT, read_json, save_figure


# ---------------------------- 可安全调整的视觉参数 ---------------------------
FIGSIZE = (15.0, 4.45)       # 整张图的宽和高，单位为英寸。
ACCURACY_LINE_WIDTH = 2.15   # 蓝/红准确率曲线宽度。
C_LINE_WIDTH = 1.65          # 灰色 Adaptive C_t 曲线宽度。
ERRORBAR_COUNT = 7           # 每条准确率曲线最多显示多少组 I 型误差条。
FIXED_COLOR = FULL_BLUE      # Fixed DP 准确率颜色。
ADAPTIVE_COLOR = OURS_RED    # Adaptive DP 准确率颜色。
C_COLOR = "#777777"          # Adaptive C_t 右轴及曲线颜色。
C_LINESTYLE = "--"           # C_t 使用虚线，避免与准确率曲线混淆。

# ---------------------------- 数据边界，不要为美化修改 -------------------------
DATASETS = ("cifar10", "cifar100", "tiny")
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny": "Tiny ImageNet",
}
MODES = ("fixed_dp", "adaptive_dp")
SEEDS = (42, 43, 44)
EPSILON = 3.0
DEFAULT_INPUT = (
    ROOT / "new_log/new_paper_adaptive/"
    "new_user_tuned_fixed_vs_adaptive_multidataset_formal.json"
)
DEFAULT_OUTPUT_STEM = "new_figure6_user_tuned_three_datasets"
DEFAULT_TABLE = (
    ROOT / "new_expTab/new_paper_adaptive/"
    "new_figure6_user_tuned_three_datasets_summary.csv"
)


def resolve_path(value: str) -> Path:
    """将命令行中的相对路径按仓库根目录解析。"""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_runs(path: Path) -> dict[tuple[str, str], list[dict]]:
    """读取18条轨迹并执行必要的结构/边界检查；这里没有哈希门。"""
    payload = read_json(path)
    if payload.get("status") != "COMPLETE":
        raise ValueError(f"输入结果未完成: {path}")
    if payload.get("official_test_read") is not False:
        raise ValueError("输入结果读取了 official test，拒绝绘图")
    if payload.get("locked_holdout_read") is not False:
        raise ValueError("输入结果读取了 locked holdout，拒绝绘图")

    grouped = {(dataset, mode): [] for dataset in DATASETS for mode in MODES}
    for item in payload.get("detailed", []):
        row = item.get("row", {})
        key = (row.get("dataset"), row.get("mode"))
        if key not in grouped:
            raise ValueError(f"出现未注册的数据集/方法: {key}")
        if int(row.get("seed", -1)) not in SEEDS:
            raise ValueError(f"出现未注册seed: {row.get('seed')}")
        if abs(float(row.get("accounted_epsilon")) - EPSILON) > 1e-8:
            raise ValueError(f"epsilon不是{EPSILON}: {key}/seed{row.get('seed')}")
        if not item.get("history"):
            raise ValueError(f"轨迹为空: {key}/seed{row.get('seed')}")
        grouped[key].append(item)

    for key, runs in grouped.items():
        runs.sort(key=lambda item: int(item["row"]["seed"]))
        if [int(item["row"]["seed"]) for item in runs] != list(SEEDS):
            raise ValueError(f"{key}没有且仅有seed 42/43/44")
    return grouped


def accuracy_trajectory(run: dict) -> tuple[np.ndarray, np.ndarray]:
    """只提取真正执行开发集评价的点；不插值、不补造中间准确率。"""
    points = [
        point for point in run["history"]
        if point.get("development_validation_accuracy") is not None
    ]
    return (
        np.asarray([point["step"] for point in points], dtype=int),
        np.asarray(
            [point["development_validation_accuracy"] for point in points],
            dtype=float,
        ),
    )


def adaptive_c_trajectory(run: dict) -> tuple[np.ndarray, np.ndarray]:
    """提取 Adaptive 每个私有更新步的 C_t；按要求不绘制其误差条。"""
    history = run["history"]
    return (
        np.asarray([point["step"] for point in history], dtype=int),
        np.asarray([point["clipping_bound"] for point in history], dtype=float),
    )


def mean_sample_std(trajectories):
    """检查三个seed的横轴一致，并计算均值与样本标准差。"""
    steps = trajectories[0][0]
    if any(not np.array_equal(item[0], steps) for item in trajectories[1:]):
        raise ValueError("同一数据集/方法的三个seed评价步不一致")
    values = np.asarray([item[1] for item in trajectories], dtype=float)
    return steps, values.mean(axis=0), values.std(axis=0, ddof=1)


def add_accuracy_i_bars(ax, x, mean, std, color):
    """只给准确率添加稀疏的三seed样本标准差 I 型误差条。"""
    indices = np.unique(
        np.linspace(0, len(x) - 1, min(ERRORBAR_COUNT, len(x)), dtype=int)
    )
    ax.errorbar(
        x[indices], mean[indices], yerr=std[indices], fmt="none",
        ecolor=color, elinewidth=1.0, capsize=3.0, capthick=1.0,
        alpha=0.9, zorder=3,
    )


def build_summary_row(dataset: str, mode: str, runs: list[dict]) -> dict:
    """生成 Table 5 和配套 CSV 所需的三seed汇总数值。"""
    rows = [run["row"] for run in runs]
    final = 100.0 * np.asarray([row["final_accuracy"] for row in rows])
    best = 100.0 * np.asarray([row["best_accuracy"] for row in rows])
    early_auc = 100.0 * np.asarray([row["early_auc"] for row in rows])
    final_c = np.asarray([row["final_c"] for row in rows])
    return {
        "dataset": dataset,
        "mode": mode,
        "n": len(runs),
        "epsilon": EPSILON,
        "steps": int(rows[0]["steps"]),
        "learning_rate": float(rows[0]["learning_rate"]),
        "expected_batch_size": int(rows[0]["expected_batch_size"]),
        "count_releases": int(rows[0]["count_releases"]),
        "final_accuracy_mean_percent": float(final.mean()),
        "final_accuracy_sample_std_pp": float(final.std(ddof=1)),
        "best_accuracy_mean_percent": float(best.mean()),
        "best_accuracy_sample_std_pp": float(best.std(ddof=1)),
        "early_auc_mean_percent": float(early_auc.mean()),
        "early_auc_sample_std_pp": float(early_auc.std(ddof=1)),
        "final_c_mean": float(final_c.mean()),
        "final_c_sample_std": float(final_c.std(ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制三数据集独立配置 Figure 6")
    parser.add_argument(
        "--input", default=str(DEFAULT_INPUT.relative_to(ROOT)),
        help="18条正式轨迹JSON；相对路径以仓库根目录为准",
    )
    parser.add_argument(
        "--output-stem", default=DEFAULT_OUTPUT_STEM,
        help="PNG/PDF文件名（不含扩展名）",
    )
    parser.add_argument(
        "--table", default=str(DEFAULT_TABLE.relative_to(ROOT)),
        help="汇总CSV路径；相对路径以仓库根目录为准",
    )
    args = parser.parse_args()
    grouped = load_runs(resolve_path(args.input))

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    summary_rows = []
    for ax, dataset in zip(axes, DATASETS):
        # Fixed 和 Adaptive 使用各自真实的更新步范围；二者没有被强行拉伸到同一终点。
        for mode, color in (("fixed_dp", FIXED_COLOR), ("adaptive_dp", ADAPTIVE_COLOR)):
            runs = grouped[(dataset, mode)]
            steps, mean, std = mean_sample_std(
                [accuracy_trajectory(run) for run in runs]
            )
            ax.plot(steps, mean, color=color, linewidth=ACCURACY_LINE_WIDTH)
            add_accuracy_i_bars(ax, steps, mean, std, color)
            summary_rows.append(build_summary_row(dataset, mode, runs))

        # 右轴只画 Adaptive C_t 均值：不画 Fixed C，也不画 C_t 误差条。
        c_axis = ax.twinx()
        c_steps, c_mean, _ = mean_sample_std(
            [adaptive_c_trajectory(run) for run in grouped[(dataset, "adaptive_dp")]]
        )
        c_axis.plot(
            c_steps, c_mean, color=C_COLOR, linestyle=C_LINESTYLE,
            linewidth=C_LINE_WIDTH, zorder=2,
        )
        c_axis.set_ylabel(r"Adaptive clipping bound $C_t$", color=C_COLOR)
        c_axis.tick_params(axis="y", colors=C_COLOR)
        c_axis.spines["right"].set_color(C_COLOR)

        ax.set_title(DATASET_LABELS[dataset], fontweight="bold")
        ax.set_xlabel("Private-update step")
        ax.set_ylabel("Development-validation accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(alpha=0.18)

    handles = [
        Line2D([0], [0], color=FIXED_COLOR, lw=2.2, label="Fixed DP accuracy"),
        Line2D([0], [0], color=ADAPTIVE_COLOR, lw=2.2, label="Adaptive DP accuracy"),
        Line2D([0], [0], color=C_COLOR, lw=1.7, ls=C_LINESTYLE,
               label=r"Adaptive $C_t$"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(r"Adaptive clipping across datasets ($\epsilon=3$)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), w_pad=2.0)
    save_figure(fig, args.output_stem)

    table_path = resolve_path(args.table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print("PASS built Figure 6:", args.output_stem)
    print("summary_table=", table_path)


if __name__ == "__main__":
    main()

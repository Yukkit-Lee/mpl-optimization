#!/usr/bin/env python3
"""生成论文 Figure 6：可独立配置的 Fixed DP / Adaptive DP 消融对比。

在仓库根目录运行：

    python new_scripts/new_paper_figures/new_build_figure6.py \
        --input new_log/new_paper_adaptive/new_user_tuned_fixed_vs_adaptive_formal.json \
        --output-stem new_figure6_user_tuned \
        --table new_expTab/new_paper_adaptive/new_figure6_user_tuned_summary.csv

输入：

默认仍读取已归档结果；你自行调参后应通过 ``--input`` 指向新 JSON。

输出：

    new_expFig/paper_experiment/
    <你传给 --output-stem 的名称>.{png,pdf}

下方“视觉参数”可安全修改，不会改变实验结果。不得为了美化图形修改输入结果、
seed、epsilon、history、方法参数或误差条数值。脚本不再要求 Fixed 冻结；它会
忠实绘制输入 JSON 中各自独立设置的 Fixed/Adaptive 参数。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from new_paper_figure_common import FULL_BLUE, OURS_RED, OUT, ROOT, read_json, save_figure


# ------------------------- 可安全调整的视觉参数 -------------------------
FIGSIZE = (14.8, 4.35)       # 图宽、图高（英寸）
LINE_WIDTH = 2.15            # 曲线宽度
ERRORBAR_COUNT = 9           # 每条曲线最多显示多少个稀疏I型误差条
ROLLING_WINDOW = 10          # clipped fraction仅用于显示的滑动平均窗口
FIXED_COLOR = FULL_BLUE
ADAPTIVE_COLOR = OURS_RED

# ------------------------- 默认输入输出（可被命令行覆盖） -------------------
DEFAULT_SOURCE = ROOT / "new_log/new_paper_adaptive/new_adaptive_independent_formal.json"
DEFAULT_TABLE = ROOT / "new_expTab/new_paper_adaptive/new_adaptive_independent_formal_summary.csv"
DEFAULT_OUTPUT_STEM = "new_figure6_adaptive_independently_tuned_pea"


def mean_std(values):
    """按 seed 计算均值与样本标准差；n=1 时返回零但不绘制误差条。"""
    array = np.asarray(values, dtype=float)
    std = array.std(axis=0, ddof=1) if len(array) >= 2 else np.zeros(array.shape[1])
    return array.mean(axis=0), std


def add_i_bars(ax, x, mean, std, color, run_count):
    """在少量真实归档点绘制样本标准差 I 型误差条；n=1 时不伪造。"""
    if run_count < 2:
        return
    idx = np.unique(
        np.linspace(0, len(x) - 1, min(ERRORBAR_COUNT, len(x)), dtype=int)
    )
    ax.errorbar(
        np.asarray(x)[idx], np.asarray(mean)[idx], yerr=np.asarray(std)[idx],
        fmt="none", ecolor=color, elinewidth=1.0, capsize=3.0,
        capthick=1.0, alpha=0.9, zorder=3,
    )


def rolling_mean(values):
    """仅平滑非私有诊断比例，不修改准确率或C轨迹。"""
    values = np.asarray(values, dtype=float)
    kernel = np.ones(ROLLING_WINDOW, dtype=float) / ROLLING_WINDOW
    padded = np.pad(values, (ROLLING_WINDOW - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_runs(source: Path):
    """读取指定结果，只检查结构、评价边界和两种方法是否完整。"""
    payload = read_json(source)
    if payload["status"] != "COMPLETE":
        raise ValueError("Figure 6正式结果尚未完成")
    if payload["official_test_read"] or payload["locked_holdout_read"]:
        raise ValueError("Figure 6来源越过test/holdout边界")
    runs = {}
    for item in payload["detailed"]:
        runs.setdefault(item["row"]["mode"], []).append(item)
    if set(runs) != {"fixed_dp", "adaptive_dp"}:
        raise ValueError("Figure 6必须同时包含Fixed与Adaptive")
    if any(not items for items in runs.values()):
        raise ValueError("Figure 6每种方法至少需要一个seed")
    fixed_seeds = {int(item["row"]["seed"]) for item in runs["fixed_dp"]}
    adaptive_seeds = {int(item["row"]["seed"]) for item in runs["adaptive_dp"]}
    if fixed_seeds != adaptive_seeds:
        raise ValueError("Fixed与Adaptive必须使用同一组seed")
    epsilons = {
        round(float(item["row"]["accounted_epsilon"]), 10)
        for items in runs.values() for item in items
    }
    if len(epsilons) != 1:
        raise ValueError("输入结果不是相同epsilon比较")
    return payload, runs


def main():
    parser = argparse.ArgumentParser(description="绘制用户可配置的 Figure 6")
    parser.add_argument(
        "--input", default=str(DEFAULT_SOURCE.relative_to(ROOT)),
        help="实验JSON路径；相对路径以仓库根目录为准",
    )
    parser.add_argument(
        "--output-stem", default=DEFAULT_OUTPUT_STEM,
        help="PNG/PDF文件名（不要带扩展名），输出目录固定为new_expFig/paper_experiment",
    )
    parser.add_argument(
        "--table", default=str(DEFAULT_TABLE.relative_to(ROOT)),
        help="Figure 6汇总CSV路径；相对路径以仓库根目录为准",
    )
    args = parser.parse_args()
    source = Path(args.input)
    table = Path(args.table)
    if not source.is_absolute():
        source = ROOT / source
    if not table.is_absolute():
        table = ROOT / table

    payload, runs = load_runs(source)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    summary_rows = []
    for mode, color, label in (
        ("fixed_dp", FIXED_COLOR, "Fixed DP (independently configured)"),
        ("adaptive_dp", ADAPTIVE_COLOR, "Adaptive DP (independently configured)"),
    ):
        items = sorted(runs[mode], key=lambda item: item["row"]["seed"])
        run_count = len(items)
        histories = [item["history"] for item in items]
        steps = np.asarray([point["step"] for point in histories[0]], dtype=int)
        if any(len(history) != len(steps) for history in histories):
            raise ValueError("同一方法的三个seed轨迹长度不一致")

        eval_idx = [
            idx for idx, point in enumerate(histories[0])
            if point["development_validation_accuracy"] is not None
        ]
        eval_steps = steps[eval_idx]
        accuracy = [
            [history[idx]["development_validation_accuracy"] for idx in eval_idx]
            for history in histories
        ]
        acc_mean, acc_std = mean_std(accuracy)
        axes[0].plot(eval_steps, acc_mean, color=color, linewidth=LINE_WIDTH)
        add_i_bars(axes[0], eval_steps, acc_mean, acc_std, color, run_count)

        bounds = [[point["clipping_bound"] for point in history] for history in histories]
        bound_mean, bound_std = mean_std(bounds)
        axes[1].plot(steps, bound_mean, color=color, linewidth=LINE_WIDTH)
        add_i_bars(axes[1], steps, bound_mean, bound_std, color, run_count)

        clipped = [
            rolling_mean(
                [point["fraction_clipped_nonprivate_diagnostic"] for point in history]
            )
            for history in histories
        ]
        clipped_mean, clipped_std = mean_std(clipped)
        axes[2].plot(steps, clipped_mean, color=color, linewidth=LINE_WIDTH)
        add_i_bars(axes[2], steps, clipped_mean, clipped_std, color, run_count)

        final = np.asarray([100.0 * item["row"]["final_accuracy"] for item in items])
        auc = np.asarray([100.0 * item["row"]["early_auc"] for item in items])
        final_c = np.asarray([item["row"]["final_c"] for item in items])
        summary_rows.append({
            "mode": mode,
            "n": run_count,
            "final_accuracy_mean_percent": float(final.mean()),
            "final_accuracy_sample_std_pp": (
                float(final.std(ddof=1)) if run_count >= 2 else ""
            ),
            "early_auc_mean_percent": float(auc.mean()),
            "early_auc_sample_std_pp": (
                float(auc.std(ddof=1)) if run_count >= 2 else ""
            ),
            "final_c_mean": float(final_c.mean()),
            "final_c_sample_std": (
                float(final_c.std(ddof=1)) if run_count >= 2 else ""
            ),
            "accounted_epsilon": float(items[0]["row"]["accounted_epsilon"]),
            "count_releases": int(items[0]["row"]["count_releases"]),
        })

    titles = ("Development utility", "Clipping-bound trajectory", "Clipped-example diagnostic")
    ylabels = ("Development-validation accuracy", r"Clipping bound $C_t$", "Fraction clipped (10-step mean)")
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Private-update step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    handles = [
        Line2D([0], [0], color=FIXED_COLOR, lw=2.2, label="Fixed DP"),
        Line2D([0], [0], color=ADAPTIVE_COLOR, lw=2.2, label="Adaptive DP"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.015))
    epsilon = float(next(iter(runs["fixed_dp"]))["row"]["accounted_epsilon"])
    fig.suptitle(
        rf"Plaintext independently configured clipping comparison on CIFAR-10 ($\epsilon={epsilon:g}$)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save_figure(fig, args.output_stem)

    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    print("PASS built Figure 6 in", OUT / (args.output_stem + ".png"))
    print("summary_table=", table)
    print("adaptive_minus_fixed_mean_pp=", payload["mean_adaptive_minus_fixed_pp"])


if __name__ == "__main__":
    main()

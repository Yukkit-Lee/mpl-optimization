#!/usr/bin/env python3
"""生成与论文 Figure 2(a) 视觉语法一致的 SPU 完整轨迹图。

运行方式（在仓库根目录）：

    /root/autodl-tmp/new_envs/distillation/bin/python \
      new_scripts/new_secretflow_spu/full_trajectory_v1/new_plot_spu_full_trajectory.py

输入为真实 SPU Stage-B + Stage-C seed42 轨迹，不插值、不平滑、不补点。
横轴为 cumulative_training_online_secure_seconds，竖直点线为 Stage B→C
转换时刻。本图 n=1，因此不绘制伪造误差。

输出使用 ``*_unified_notitle_bold`` 新文件名，不覆盖旧候选图。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "new_expFig/paper_experiment/spu_full_trajectory_v1"
PLOT_DATA = ROOT / "new_output/new_secretflow_spu/new_figure2_spu_full_trajectory_plot_data.csv"

DATASETS = ("cifar10", "cifar100", "tiny")
DATASET_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tiny": "Tiny ImageNet"}
METHODS = ("full_data", "true_ours")
# SPU 后端使用既定的粉色/紫色方法配色。
COLORS = {"full_data": "#E377C2", "true_ours": "#9467BD"}

# 只影响排版，不改变任何实验数值。
# 略微补偿 tight-bbox 的横向差异，使导出的 PDF 尺寸与 Figure 2(a) 对齐。
FIGSIZE = (15.833, 4.5)
PANEL_WSPACE = 0.28
LINE_WIDTH = 2.2
PNG_DPI = 600


def apply_bold_typography(fig):
    """复制远端最新论文 Figure 2 的全局粗体层级。"""
    for text in fig.findobj(match=mpl.text.Text):
        text.set_fontweight("bold")
        if text.get_fontsize() < 9.5:
            text.set_fontsize(9.5)
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(max(11.5, ax.xaxis.label.get_fontsize()))
        ax.yaxis.label.set_fontsize(max(11.5, ax.yaxis.label.get_fontsize()))
        ax.title.set_fontsize(max(13, ax.title.get_fontsize()))
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
            tick.set_fontsize(max(9.5, tick.get_fontsize()))


def load_plot_data():
    rows = {}
    with PLOT_DATA.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault((row["dataset"], row["method"]), []).append(row)
    return rows


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_plot_data()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=False)

    for axis, dataset in zip(axes, DATASETS):
        targets = {
            "T@80": float(data[(dataset, "true_ours")][0]["target_T80"]),
            "T@90": float(data[(dataset, "true_ours")][0]["target_T90"]),
            "T@95": float(data[(dataset, "true_ours")][0]["target_T95"]),
        }
        visible_y = []
        for method in METHODS:
            rows = data[(dataset, method)]
            # Ordered by stage then epoch/step.
            ordered = sorted(
                rows,
                key=lambda r: (
                    0 if r["stage"].startswith("stage_b") else 1,
                    int(float(r["epoch_or_step"])),
                ),
            )
            x = np.asarray(
                [float(r["cumulative_training_online_secure_seconds"]) for r in ordered]
            )
            y = np.asarray([float(r["development_accuracy"]) for r in ordered])
            # epoch0 的安全训练时间为 0，数学上无法放在对数横轴上。
            # 该点仍保留在原始 CSV 中；绘图仅显示 x>0 的真实观测点，
            # 同时避免不可见的 epoch0 将纵轴拉到接近 0%。
            visible = x > 0.0
            x_visible = x[visible]
            y_visible = y[visible]
            visible_y.extend(y_visible.tolist())
            color = COLORS[method]
            axis.plot(x_visible, y_visible, color=color, linewidth=LINE_WIDTH)
            axis.scatter([x_visible[-1]], [y_visible[-1]], color=color, s=25, zorder=5)
            # Stage B->C transition: first point whose stage == stage_c.
            transition_x = next(
                (float(r["cumulative_training_online_secure_seconds"]) for r in ordered
                 if r["stage"].startswith("stage_c")),
                None,
            )
            if transition_x is not None:
                axis.axvline(
                    transition_x, color=color, linestyle=":", linewidth=1.2, alpha=0.7
                )
        for label, value in targets.items():
            axis.axhline(value, color="#777777", linestyle="--", linewidth=0.7, alpha=0.45)

        # 仿照论文 Figure 2：每个数据集使用独立紧凑纵轴。
        # 范围由真正可见的轨迹点和 T@80/90/95 目标线共同决定。
        y_values = np.asarray(visible_y + list(targets.values()), dtype=float)
        y_span = max(float(y_values.max() - y_values.min()), 0.01)
        y_pad = max(0.006, 0.06 * y_span)
        axis.set_ylim(float(y_values.min() - y_pad), float(y_values.max() + y_pad))
        axis.set_xscale("log")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_title(DATASET_LABELS[dataset], fontweight="bold")
        axis.set_xlabel("Cumulative online secure time (s, log scale)")
        axis.grid(True, which="both", alpha=0.16)
        axis.text(
            0.025, 0.955, "SPU / ABY3",
            transform=axis.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold", color="#444444",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white",
                edgecolor="#BBBBBB", alpha=0.88,
            ),
        )
    axes[0].set_ylabel("Development-validation accuracy")

    # 图例文字、线型和位置与 Figure 2(a) 保持一致。
    handles = [
        Line2D([0], [0], color=COLORS["full_data"], lw=2.2, label="Full Data"),
        Line2D([0], [0], color=COLORS["true_ours"], lw=2.2, label="Ours"),
        Line2D([0], [0], color="#777777", lw=0.8, ls="--",
               label="T@80/90/95"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    apply_bold_typography(fig)
    fig.subplots_adjust(
        left=0.07, right=0.985, top=0.96, bottom=0.20,
        wspace=PANEL_WSPACE,
    )
    output_stem = "new_figure2b_spu_full_trajectory_seed42_pink_purple_notitle_bold"
    png_path = FIG_DIR / (output_stem + ".png")
    pdf_path = FIG_DIR / (output_stem + ".pdf")
    svg_path = FIG_DIR / (output_stem + ".svg")
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    # Matplotlib 的 SVG path 行默认带尾随空格；机械清理以保持
    # git diff --check 可通过，不改变任何向量内容或渲染结果。
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    print("saved full-trajectory figures to", FIG_DIR)


if __name__ == "__main__":
    main()

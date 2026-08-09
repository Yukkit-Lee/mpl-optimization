#!/usr/bin/env python3
"""论文独立绘图脚本共用的辅助工具。

本模块只提供路径、颜色、文件读取、图片保存和 I 型误差条等公共能力，
故意不包含任何具体 Figure 的数据选择逻辑。需要修改某张图的科学语义时，
应进入 ``new_build_figure2.py`` 至 ``new_build_figure5.py`` 中对应的脚本。
PNG 分辨率和共享颜色等安全视觉默认值放在这里，以保持各图风格一致。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 仓库与输出目录。除非另一个实验目录具有完全相同且经过审计的数据结构，
# 否则不要把 ROOT 指向其他目录。
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "new_expFig/paper_experiment"

DATASETS = ("cifar10", "cifar100", "tiny")
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny": "Tiny ImageNet",
}

# 可安全修改的视觉参数：颜色不会改变任何实验数值。
FULL_BLUE = "#2C7BB6"
OURS_RED = "#D7191C"
ORANGE = "#FDAE61"

# PNG 输出分辨率。提高 PNG_DPI 会让位图更清晰，但也会增大文件体积；
# PDF 始终是矢量输出，不受该参数影响。
PNG_DPI = 240


def read_json(path: Path):
    """读取必需的 JSON，但不计算或比对 SHA256。

    绘图只要求文件存在且 JSON 格式可以解析，因此用户修改 JSON 数值后可以
    直接重新绘图，无需同步更新任何 hash manifest。各 Figure 脚本仍会检查
    必需字段、记录数量和 test/holdout 等科学边界，避免画出结构不完整的数据。
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    """读取必需的 CSV，并将每一行返回为字典。"""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, stem: str):
    """在 ``new_expFig/paper_experiment`` 下保存同名 PNG/PDF。"""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (stem + ".png"), dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(OUT / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def sparse_indices(length: int, count: int = 9):
    """均匀选择少量 checkpoint，避免 I 型误差条过密而遮挡曲线。"""
    return np.unique(np.linspace(0, length - 1, min(count, length), dtype=int))


def add_i_errorbars(ax, x, y, std, color, count: int = 9):
    """在稀疏归档点绘制样本标准差 I 型误差条。

    ``count`` 只控制视觉密度，可以安全调整；``std`` 属于实验依据，
    对 n=1 的曲线绝对不能编造标准差。
    """
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

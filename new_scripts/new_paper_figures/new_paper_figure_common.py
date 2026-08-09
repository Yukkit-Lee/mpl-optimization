#!/usr/bin/env python3
"""Shared helpers for the independently runnable manuscript figure scripts.

This module intentionally contains no Figure-specific data selection.  Change
Figure semantics only in ``new_build_figure2.py`` ... ``new_build_figure5.py``.
Safe visual defaults such as PNG DPI and shared colors live here so all figures
remain visually consistent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Repository/output locations.  Do not point ROOT at a different experiment
# tree unless that tree contains the same audited artifact schema.
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "new_expFig/paper_experiment"

DATASETS = ("cifar10", "cifar100", "tiny")
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny": "Tiny ImageNet",
}

# Safe-to-change visual parameters: colors do not change numerical results.
FULL_BLUE = "#2C7BB6"
OURS_RED = "#D7191C"
ORANGE = "#FDAE61"

# Output resolution.  Raising PNG_DPI improves raster sharpness but increases
# file size; PDF output remains vector-based.
PNG_DPI = 240


def read_json(path: Path):
    """Read one required JSON artifact and fail rather than silently fallback."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    """Read one required CSV artifact as dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, stem: str):
    """Save matching PNG/PDF outputs under ``new_expFig/paper_experiment``."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (stem + ".png"), dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(OUT / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def sparse_indices(length: int, count: int = 9):
    """Choose evenly spaced checkpoints for readable I-shaped error bars."""
    return np.unique(np.linspace(0, length - 1, min(count, length), dtype=int))


def add_i_errorbars(ax, x, y, std, color, count: int = 9):
    """Draw sample-standard-deviation I-bars at sparse archived checkpoints.

    ``count`` is a safe visual-density parameter.  ``std`` is experimental
    evidence and must never be invented for an n=1 series.
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

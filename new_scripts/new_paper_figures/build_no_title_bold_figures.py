#!/usr/bin/env python3
"""Rebuild Figures 2--6 without redundant figure-level titles.

The original plotting modules and outputs are left unchanged. This runner reuses
their validated data-loading and plotting logic, suppresses only ``suptitle``
text, applies publication-scale bold typography, and redirects every export to
``new_expFig/paper_experiment/no_title_bold``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.text


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = ROOT / "new_expFig" / "paper_experiment" / "no_title_bold"
PNG_DPI = 600


def suppress_suptitle(self, *args, **kwargs):
    """Record a suppressed title so the saved layout can reclaim its space."""
    self._suppressed_suptitle = True
    return None


def apply_bold_typography(fig, stem: str) -> None:
    """Make all visible text bold and readable after LNCS-width scaling."""
    for text in fig.findobj(match=matplotlib.text.Text):
        text.set_fontweight("bold")
        if text.get_fontsize() < 9.5:
            text.set_fontsize(9.5)

    for ax in fig.axes:
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(max(11.5, ax.xaxis.label.get_fontsize()))
        ax.yaxis.label.set_fontsize(max(11.5, ax.yaxis.label.get_fontsize()))
        ax.title.set_fontweight("bold")
        ax.title.set_fontsize(max(13, ax.title.get_fontsize()))
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
            tick.set_fontsize(max(9.5, tick.get_fontsize()))

    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_fontsize(max(10.5, text.get_fontsize()))

    if "figure4_initializer" in stem:
        compact_labels = {
            "Random Init.\n+ Anchor": "Rand. Init.\n+ Anchor",
            "Single-Party Init.\n+ Anchor": "Local Init.\n+ Anchor",
            "Secure Average\nOnly": "SecAvg\nOnly",
            "Secure Average\n+ Anchor": "SecAvg\n+ Anchor",
            "Random Init. + Anchor": "Rand. Init. + Anchor",
            "Single-Party Init. + Anchor": "Local Init. + Anchor",
            "Secure Average Only": "SecAvg Only",
            "Secure Average + Anchor": "SecAvg + Anchor",
        }
        for text in fig.findobj(match=matplotlib.text.Text):
            if text.get_text() in compact_labels:
                text.set_text(compact_labels[text.get_text()])
                text.set_fontsize(9.5)

    if "figure6_" in stem:
        accuracy_axes = [
            ax for ax in fig.axes if "validation accuracy" in ax.get_ylabel().lower()
        ]
        clipping_axes = [
            ax for ax in fig.axes if "clipping bound" in ax.get_ylabel().lower()
        ]
        for ax in accuracy_axes[1:]:
            ax.set_ylabel("")
        for ax in clipping_axes[:-1]:
            ax.set_ylabel("")

    if getattr(fig, "_suppressed_suptitle", False):
        fig.subplots_adjust(top=0.96)


def redirected_save(fig, stem: str) -> None:
    """Export a new no-title/bold variant without touching original figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_bold_typography(fig, stem)
    output_stem = stem + "_notitle_bold"
    fig.savefig(OUTPUT_DIR / f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{output_stem}.svg", bbox_inches="tight")
    fig.savefig(
        OUTPUT_DIR / f"{output_stem}.png",
        dpi=PNG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_module(module_name: str, argv: list[str] | None = None) -> None:
    module = importlib.import_module(module_name)
    if hasattr(module, "save_figure"):
        module.save_figure = redirected_save
    previous_argv = sys.argv
    try:
        sys.argv = [module_name] + (argv or [])
        module.main()
    finally:
        sys.argv = previous_argv


def main() -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    import new_paper_figure_common as common

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    matplotlib.figure.Figure.suptitle = suppress_suptitle
    common.OUT = OUTPUT_DIR
    common.PNG_DPI = PNG_DPI
    common.save_figure = redirected_save

    run_module("new_build_figure2")
    run_module("new_build_figure3")
    run_module("new_build_figure4")
    run_module("new_build_figure5")
    run_module(
        "new_build_figure6",
        [
            "--output-stem",
            "new_figure6_user_tuned_three_datasets",
            "--table",
            str(OUTPUT_DIR / "new_figure6_summary.csv"),
        ],
    )
    print(f"PASS no-title bold figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

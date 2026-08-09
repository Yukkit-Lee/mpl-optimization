#!/usr/bin/env python3
"""Compatibility wrapper that rebuilds manuscript Figures 2--5.

Each Figure now has its own documented, independently runnable script:

    python new_scripts/new_paper_figures/new_build_figure2.py
    python new_scripts/new_paper_figures/new_build_figure3.py
    python new_scripts/new_paper_figures/new_build_figure4.py
    python new_scripts/new_paper_figures/new_build_figure5.py

Run this wrapper only when all four figures should be regenerated together:

    python new_scripts/new_paper_figures/new_build_figures_2_to_5.py

Do not add Figure-specific plotting logic here; put it in the corresponding
individual script so data sources and adjustable parameters remain discoverable.
"""

from new_build_figure2 import main as build_figure2
from new_build_figure3 import main as build_figure3
from new_build_figure4 import main as build_figure4
from new_build_figure5 import main as build_figure5
from new_paper_figure_common import OUT


def main():
    build_figure2()
    build_figure3()
    build_figure4()
    build_figure5()
    print("PASS built manuscript Figures 2--5 in", OUT)


if __name__ == "__main__":
    main()

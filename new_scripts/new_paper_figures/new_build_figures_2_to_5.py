#!/usr/bin/env python3
"""一次性重建论文 Figure 2--5 的兼容入口。

每张图现在都有带中文说明、可独立运行的脚本：

    python new_scripts/new_paper_figures/new_build_figure2.py
    python new_scripts/new_paper_figures/new_build_figure3.py
    python new_scripts/new_paper_figures/new_build_figure4.py
    python new_scripts/new_paper_figures/new_build_figure5.py

只有需要同时重新生成四张图时才运行本入口：

    python new_scripts/new_paper_figures/new_build_figures_2_to_5.py

不要在本文件中加入任何具体 Figure 的绘图逻辑；应将逻辑放进对应的独立
脚本，使每张图的数据来源、参数含义和可调范围都能被直接找到。
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

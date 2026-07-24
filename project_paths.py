#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：统一解析论文项目在本机与 AutoDL 上的源码、数据、MTT 和输出路径。

路径优先读取环境变量；未配置时自动寻找项目同级目录和当前用户的
``tmpWork``。本模块只处理路径，不修改算法、随机种子或实验超参数。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _env_path(name: str):
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else None


def _first_existing(candidates: Iterable[Path], fallback: Path) -> Path:
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate.exists():
            return candidate.resolve()
    return fallback.expanduser().resolve()


SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = _env_path("PROJECT_ROOT") or SOURCE_ROOT
DATA_ROOT = _env_path("DATA_ROOT") or _first_existing(
    [PROJECT_ROOT.parent / "tmpWork", Path.home() / "tmpWork"],
    PROJECT_ROOT / "runtime_data",
)
MTT_ROOT = _env_path("MTT_ROOT") or _first_existing(
    [PROJECT_ROOT.parent / "mtt-distillation", DATA_ROOT / "mtt-distillation"],
    PROJECT_ROOT.parent / "mtt-distillation",
)
OFFICIAL_MTT_ROOT = _env_path("OFFICIAL_MTT_ROOT") or (
    DATA_ROOT / "mtt_data" / "georgecazenavette"
)
OUTPUT_ROOT = _env_path("OUTPUT_ROOT") or PROJECT_ROOT


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path = PROJECT_ROOT
    source_root: Path = SOURCE_ROOT
    mtt_root: Path = MTT_ROOT
    data_root: Path = DATA_ROOT
    official_mtt_root: Path = OFFICIAL_MTT_ROOT
    output_root: Path = OUTPUT_ROOT

    @property
    def cifar10_data_root(self) -> Path:
        return self.data_root / "mtt_cifar10_work" / "data"

    @property
    def cifar100_data_root(self) -> Path:
        return self.data_root / "mtt_cifar100_work" / "data"

    @property
    def tiny_imagenet_root(self) -> Path:
        return self.data_root / "mtt_tiny_work" / "data" / "tiny-imagenet-200"

    def as_dict(self):
        return {
            "PROJECT_ROOT": str(self.project_root),
            "SOURCE_ROOT": str(self.source_root),
            "MTT_ROOT": str(self.mtt_root),
            "DATA_ROOT": str(self.data_root),
            "OFFICIAL_MTT_ROOT": str(self.official_mtt_root),
            "OUTPUT_ROOT": str(self.output_root),
        }


PATHS = ProjectPaths()


def print_resolved_paths() -> None:
    """打印最终生效路径，便于迁移后排错。"""
    for key, value in PATHS.as_dict().items():
        print(f"{key:<18}: {value}")


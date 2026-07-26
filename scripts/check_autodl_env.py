#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用途：在 AutoDL 上检查路径、核心依赖、CUDA 和关键资产是否就绪。"""

import json
import os
import platform
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from new_runtime_paths import DATA_ROOT, MTT_ROOT, OFFICIAL_MTT_ROOT, OUTPUT_ROOT, PROJECT_ROOT


def _version(module_name):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:
        return f"ERROR: {exc}"


def main() -> int:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "mtt_root": str(MTT_ROOT),
            "data_root": str(DATA_ROOT),
            "official_mtt_root": str(OFFICIAL_MTT_ROOT),
            "output_root": str(OUTPUT_ROOT),
        },
        "versions": {
            "numpy": _version("numpy"),
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
            "sklearn": _version("sklearn"),
            "matplotlib": _version("matplotlib"),
        },
    }
    try:
        import torch
        report["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        report["cuda"] = {"available": False, "error": str(exc)}

    required = {
        "MTT repo": MTT_ROOT / "utils.py",
        "official CIFAR-10 ipc50": OFFICIAL_MTT_ROOT / "cifar10_50" / "images_best.pt",
        "official CIFAR-100 ipc50": OFFICIAL_MTT_ROOT / "cifar100_50" / "images_best.pt",
        "official Tiny ipc50": OFFICIAL_MTT_ROOT / "tiny_50" / "images_best_0.pt",
        "CIFAR-10 archive": DATA_ROOT / "mtt_cifar10_work/data/cifar-10-python.tar.gz",
        "CIFAR-100 archive": DATA_ROOT / "mtt_cifar100_work/data/cifar-100-python.tar.gz",
        "Tiny ImageNet archive": DATA_ROOT / "mtt_tiny_work/data/tiny-imagenet-200.zip",
        "portable Stage-B initializer manifest": PROJECT_ROOT / "portable_artifacts/stage_b_selected/manifest.json",
    }
    report["assets"] = {
        name: {"path": str(path), "exists": path.exists()} for name, path in required.items()
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    missing = [name for name, info in report["assets"].items() if not info["exists"]]
    if missing:
        print("\n缺失资产：" + "、".join(missing), file=sys.stderr)
        return 2
    if not report.get("cuda", {}).get("available"):
        print("\nCUDA 未就绪。", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用途：在 AutoDL 上检查路径、核心依赖、CUDA 和关键资产是否就绪。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from project_paths import PATHS


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
        "paths": PATHS.as_dict(),
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
        "MTT repo": PATHS.mtt_root / "utils.py",
        "official CIFAR-10 ipc50": PATHS.official_mtt_root / "cifar10_50" / "images_best.pt",
        "official CIFAR-100 ipc50": PATHS.official_mtt_root / "cifar100_50" / "images_best.pt",
        "official Tiny ipc50": PATHS.official_mtt_root / "tiny_50" / "images_best_0.pt",
        "CIFAR-10 data": PATHS.cifar10_data_root,
        "CIFAR-100 data": PATHS.cifar100_data_root,
        "Tiny ImageNet data": PATHS.tiny_imagenet_root,
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


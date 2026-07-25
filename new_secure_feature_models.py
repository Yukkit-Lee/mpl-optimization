#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：new_secure_feature_pipeline 共用的 MTT ConvNet、随机状态和文件校验工具。

该文件只定义模型与通用工具，不读取任何原始数据。CIFAR-10/CIFAR-100 使用
三层 ConvNet，Tiny ImageNet 使用四层 ConvNet；两者分类器前表示均严格为
2048 维，供 Stage A/B/C 共享同一个冻结编码器。
"""

import hashlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


DATASET_SPECS = {
    "cifar10": {
        "channels": 3,
        "image_size": (32, 32),
        "num_classes": 10,
        "depth": 3,
        "width": 128,
    },
    "cifar100": {
        "channels": 3,
        "image_size": (32, 32),
        "num_classes": 100,
        "depth": 3,
        "width": 128,
    },
    "tiny": {
        "channels": 3,
        "image_size": (64, 64),
        "num_classes": 200,
        "depth": 4,
        "width": 128,
    },
}


class NewMTTConvNet(nn.Module):
    """与 MTT 默认 ConvNet 对齐、且分类器前表示为 2048 维的网络。"""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        image_size: Tuple[int, int],
        depth: int,
        width: int = 128,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = channels
        height, width_px = image_size
        for _ in range(depth):
            layers.extend(
                [
                    nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
                    # MTT 的 instancenorm 对应每通道一组的 GroupNorm。
                    nn.GroupNorm(width, width, affine=True),
                    nn.ReLU(inplace=False),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = width
            height //= 2
            width_px //= 2
        self.features = nn.Sequential(*layers)
        self.feature_dim = width * height * width_px
        if self.feature_dim != 2048:
            raise ValueError("共享编码器表示维度必须为 2048，实际为 {}".format(self.feature_dim))
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))


def configure_single_thread_runtime(seed: int) -> None:
    """固定所有显式线程和随机源；重复调用时不重设 inter-op 线程池。"""

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["MALLOC_ARENA_MAX"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_mtt_convnet(dataset: str, seed: int, device: torch.device) -> NewMTTConvNet:
    if dataset not in DATASET_SPECS:
        raise ValueError("不支持的数据集: {}".format(dataset))
    torch.manual_seed(seed)
    spec = DATASET_SPECS[dataset]
    return NewMTTConvNet(
        channels=spec["channels"],
        num_classes=spec["num_classes"],
        image_size=spec["image_size"],
        depth=spec["depth"],
        width=spec["width"],
    ).to(device)


def unwrap_private_model(model: nn.Module) -> nn.Module:
    """获取 Opacus 包装模型内部的原始模型。"""

    return getattr(model, "_module", model)


def parameter_snapshot(model: nn.Module):
    return [parameter.detach().cpu().clone() for parameter in unwrap_private_model(model).parameters()]


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="new_tmp_", suffix=".pt", dir=str(path.parent), delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        torch.save(payload, str(tmp_path))
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_schema_hash(model: nn.Module) -> str:
    entries = []
    for name, tensor in unwrap_private_model(model).state_dict().items():
        entries.append("{}:{}:{}".format(name, tuple(tensor.shape), tensor.dtype))
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()

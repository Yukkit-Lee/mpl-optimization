#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：为 Opacus 提供基于 ChaCha20 CSPRNG 的可恢复高斯噪声后端。

当前 PyTorch/CUDA 环境无法加载旧版 torchcsprng。本文件不改变 Opacus 的
逐样本裁剪与 RDP accountant，只替换其高斯噪声采样函数。均匀随机字节由
ChaCha20 生成，再通过 Box-Muller 映射为高斯变量；每次输出叠加四份独立
高斯并除以 2，以保留 Opacus secure_mode 对浮点离散攻击的缓解方式。
"""

import hashlib
import math
import os
import struct
from typing import Any, Dict, Optional

import numpy as np
import torch
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms


class ChaCha20GaussianSampler:
    """使用 ChaCha20 字节流生成可 checkpoint 的安全高斯噪声。"""

    SCHEMA = "new_chacha20_gaussian_sampler_v1"

    def __init__(self, key: Optional[bytes] = None, request_counter: int = 0) -> None:
        self._key = key if key is not None else os.urandom(32)
        if len(self._key) != 32:
            raise ValueError("ChaCha20 key 必须为 32 bytes")
        self._request_counter = int(request_counter)
        if self._request_counter < 0:
            raise ValueError("request_counter 不能为负数")

    @property
    def request_counter(self) -> int:
        return self._request_counter

    @property
    def key_fingerprint(self) -> str:
        """只返回不可逆指纹，允许审计且不暴露 CSPRNG key。"""

        return hashlib.sha256(self._key).hexdigest()

    def state_dict(self) -> Dict[str, Any]:
        """checkpoint 专用状态；含密钥的文件必须限制为当前用户可读。"""

        return {
            "schema": self.SCHEMA,
            "key": self._key,
            "request_counter": self._request_counter,
        }

    def counter_state_dict(self) -> Dict[str, Any]:
        """返回可写入普通 checkpoint 的非敏感恢复状态。"""

        return {
            "schema": self.SCHEMA,
            "request_counter": self._request_counter,
            "key_fingerprint": self.key_fingerprint,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("schema") != self.SCHEMA:
            raise ValueError("不支持的 CSPRNG checkpoint schema")
        key = state["key"]
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("CSPRNG checkpoint key 无效")
        counter = int(state["request_counter"])
        if counter < 0:
            raise ValueError("CSPRNG checkpoint counter 无效")
        self._key = key
        self._request_counter = counter

    def load_counter_state_dict(self, state: Dict[str, Any]) -> None:
        """在从权限受限 sidecar 加载 key 后，恢复公开 counter。"""

        if state.get("schema") != self.SCHEMA:
            raise ValueError("不支持的 CSPRNG counter schema")
        if state.get("key_fingerprint") != self.key_fingerprint:
            raise ValueError("CSPRNG key sidecar 与 checkpoint 指纹不一致")
        counter = int(state["request_counter"])
        if counter < 0:
            raise ValueError("CSPRNG checkpoint counter 无效")
        self._request_counter = counter

    def _stream_bytes(self, num_bytes: int) -> bytes:
        """每次请求使用唯一 128-bit nonce，避免复用 ChaCha20 key stream。"""

        if num_bytes < 0:
            raise ValueError("num_bytes 不能为负数")
        if self._request_counter >= 2 ** 128:
            raise OverflowError("ChaCha20 request counter 已耗尽")
        nonce = struct.pack("<QQ", self._request_counter & ((1 << 64) - 1), self._request_counter >> 64)
        self._request_counter += 1
        cipher = Cipher(
            algorithms.ChaCha20(self._key, nonce),
            mode=None,
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()
        return encryptor.update(b"\x00" * num_bytes)

    def _standard_normal_numpy(self, count: int) -> np.ndarray:
        """将 CSPRNG uint64 均匀变量通过 Box-Muller 转为标准高斯。"""

        if count <= 0:
            return np.empty((0,), dtype=np.float64)
        pair_count = (count + 1) // 2
        raw = self._stream_bytes(pair_count * 16)
        integers = np.frombuffer(raw, dtype="<u8").reshape(pair_count, 2)
        # 使用最高 53 bit 构造开区间 (0, 1)，避免 log(0)。
        uniforms = ((integers >> 11).astype(np.float64) + 0.5) / float(1 << 53)
        radius = np.sqrt(-2.0 * np.log(uniforms[:, 0]))
        angle = 2.0 * math.pi * uniforms[:, 1]
        values = np.empty(pair_count * 2, dtype=np.float64)
        values[0::2] = radius * np.cos(angle)
        values[1::2] = radius * np.sin(angle)
        return values[:count]

    def generate(
        self,
        std: float,
        reference: torch.Tensor,
        secure_mode: bool = True,
    ) -> torch.Tensor:
        """生成与 reference 同形状、设备和 dtype 的高斯噪声。"""

        if std < 0:
            raise ValueError("noise std 不能为负数")
        if std == 0 or reference.numel() == 0:
            return torch.zeros_like(reference)
        count = int(reference.numel())
        if secure_mode:
            # 丢弃首个样本，并使用 4 份独立 N(0, std^2) 的和除以 2。
            self._standard_normal_numpy(1)
            values = np.zeros(count, dtype=np.float64)
            for _ in range(4):
                values += self._standard_normal_numpy(count)
            values *= float(std) / 2.0
        else:
            values = self._standard_normal_numpy(count) * float(std)
        tensor = torch.from_numpy(values.reshape(tuple(reference.shape)))
        return tensor.to(device=reference.device, dtype=reference.dtype, non_blocking=False)


def install_opacus_chacha20_noise(sampler: ChaCha20GaussianSampler):
    """
    将 Opacus 的噪声函数替换为当前 sampler，并返回原函数供测试恢复。

    Opacus 仍负责 per-sample clipping、采样率记录和 RDP accountant。本函数
    仅替换 DPOptimizer.add_noise() 在运行时查找的模块级 `_generate_noise`。
    """

    import opacus.optimizers.optimizer as optimizer_module

    original = optimizer_module._generate_noise

    def new_generate_noise(std, reference, generator=None, secure_mode=False):
        del generator
        return sampler.generate(std=float(std), reference=reference, secure_mode=True)

    optimizer_module._generate_noise = new_generate_noise
    return original

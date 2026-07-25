#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：验证 Stage-A ChaCha20 CSPRNG 高斯噪声的分布、恢复与 Opacus 接入。

本测试仅使用小张量和单线程，不读取数据集，也不执行正式训练。
"""

import copy
import os

import numpy as np
import torch

from new_cryptographic_gaussian import (
    ChaCha20GaussianSampler,
    install_opacus_chacha20_noise,
)
from new_secure_feature_models import configure_single_thread_runtime


def test_distribution() -> None:
    sampler = ChaCha20GaussianSampler(key=bytes(range(32)))
    reference = torch.zeros(200000, dtype=torch.float32)
    noise = sampler.generate(std=1.7, reference=reference)
    mean = float(noise.mean())
    std = float(noise.std(unbiased=True))
    if abs(mean) > 0.02:
        raise AssertionError("噪声均值异常: {}".format(mean))
    if abs(std - 1.7) > 0.025:
        raise AssertionError("噪声标准差异常: {}".format(std))
    print("distribution mean={:.6f} std={:.6f}".format(mean, std))


def test_checkpoint_replay() -> None:
    sampler = ChaCha20GaussianSampler(key=b"k" * 32)
    reference = torch.zeros(257)
    sampler.generate(std=0.8, reference=reference)
    state = copy.deepcopy(sampler.state_dict())
    counter_state = copy.deepcopy(sampler.counter_state_dict())
    expected = sampler.generate(std=0.8, reference=reference)
    restored = ChaCha20GaussianSampler()
    restored.load_state_dict(state)
    restored.load_counter_state_dict(counter_state)
    actual = restored.generate(std=0.8, reference=reference)
    if not torch.equal(expected, actual):
        raise AssertionError("CSPRNG checkpoint 重放不一致")
    print("checkpoint replay counter={}".format(restored.request_counter))


def test_opacus_patch_does_not_use_torch_normal() -> None:
    import opacus.optimizers.optimizer as optimizer_module

    sampler = ChaCha20GaussianSampler(key=b"p" * 32)
    original_opacus = install_opacus_chacha20_noise(sampler)
    original_torch_normal = torch.normal

    def forbidden_normal(*args, **kwargs):
        raise AssertionError("安全后端不应调用 torch.normal")

    torch.normal = forbidden_normal
    try:
        output = optimizer_module._generate_noise(
            std=0.5,
            reference=torch.zeros(64),
            generator=None,
            secure_mode=False,
        )
    finally:
        torch.normal = original_torch_normal
        optimizer_module._generate_noise = original_opacus
    if output.shape != (64,) or not torch.isfinite(output).all():
        raise AssertionError("Opacus CSPRNG hook 输出无效")
    print("opacus patch finite=True counter={}".format(sampler.request_counter))


def test_independent_keys() -> None:
    first = ChaCha20GaussianSampler()
    second = ChaCha20GaussianSampler()
    if first.key_fingerprint == second.key_fingerprint:
        raise AssertionError("独立 CSPRNG 实例不应使用相同密钥")
    print("independent keys=True")


def main() -> int:
    os.environ["PYTHONNOUSERSITE"] = "1"
    configure_single_thread_runtime(42)
    test_distribution()
    test_checkpoint_replay()
    test_opacus_patch_does_not_use_torch_normal()
    test_independent_keys()
    print("new cryptographic Gaussian tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

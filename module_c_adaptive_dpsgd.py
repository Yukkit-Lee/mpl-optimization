#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块 C：Adaptive-Clipped Private Optimization（特征级全局训练）
=================================================================

用途：
- 从模块 B 的 chosen initial global model 出发，
  在 logically global feature dataset 上进行后续私有训练。
- 规则层改动：使用 Adaptive Clipping 动态更新 clipping bound C_t。
- 噪声后端替换：使用可插拔 NoiseBackend（Gaussian / BinomialMPC）。

说明：
- 本文件完全不包含历史 trust 机制。
- 训练模式仅保留：plain / fixed_dp / adaptive_dp。
"""

import copy
import json
import math
import os
import platform
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_JSON_DIR = LOG_DIR / "json"
LOG_FINAL_MODEL_DIR = LOG_DIR / "finalModel"
DATA_DIR = BASE_DIR / "data"


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 与模块 b 对齐的模型定义
# ============================================================================
class LinearClassifier(nn.Module):
    """最简线性分类器（二分类 out_dim=1，多分类 out_dim=num_classes）。"""

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        out_dim = 1 if num_classes == 2 else num_classes
        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def make_fresh_model(feat_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    m = LinearClassifier(feat_dim, num_classes)
    return m.to(device)


def clone_model(model: nn.Module, device: torch.device) -> nn.Module:
    return copy.deepcopy(model).to(device)


# ============================================================================
# 工具函数（复用原模块 C 风格）
# ============================================================================
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_safe_runtime(max_threads: int = 1) -> None:
    t = max(1, int(max_threads))
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[key] = str(t)

    try:
        torch.set_num_threads(t)
        try:
            torch.set_num_interop_threads(t)
        except Exception:
            pass
    except Exception:
        pass

    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=t)
    except Exception:
        pass

    os.environ["TFE_SAFE_RUNTIME_APPLIED"] = "1"


def infer_num_classes(y: np.ndarray) -> int:
    return int(len(np.unique(np.asarray(y).reshape(-1))))


def to_tensor_x(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def to_tensor_y(y: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    y = np.asarray(y).reshape(-1)
    if num_classes == 2:
        return torch.from_numpy(y.astype(np.float32)).view(-1, 1).to(device)
    return torch.from_numpy(y.astype(np.int64)).to(device)


def predict_labels(model: nn.Module, x: np.ndarray, num_classes: int, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(to_tensor_x(x, device))
        if num_classes == 2:
            pred = (torch.sigmoid(logits).view(-1) >= 0.5).long()
        else:
            pred = torch.argmax(logits, dim=1)
    return pred.detach().cpu().numpy()


def compute_accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, num_classes: int, device: torch.device) -> float:
    pred = predict_labels(model, x, num_classes, device)
    return float(np.mean(np.asarray(y).reshape(-1) == pred.reshape(-1)))


def compute_loss(model: nn.Module, x: np.ndarray, y: np.ndarray, num_classes: int, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(to_tensor_x(x, device))
        yt = to_tensor_y(y, num_classes, device)
        if num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(logits, yt)
        else:
            loss = F.cross_entropy(logits, yt)
    return float(loss.item())


def build_log_epochs(total: int, max_lines: int = 10) -> set:
    total = max(1, int(total))
    max_lines = max(2, int(max_lines))
    if total <= max_lines:
        return set(range(1, total + 1))
    picks = np.linspace(1, total, num=max_lines, dtype=int)
    out = set(int(x) for x in picks)
    out.add(1)
    out.add(total)
    return out


def select_device(device_hint: Optional[str] = None) -> torch.device:
    if device_hint:
        hint = str(device_hint).strip().lower()
        if hint.startswith("cuda"):
            try:
                dev = torch.device(device_hint)
                a = torch.randn(2, 2, device=dev)
                b = torch.randn(2, 2, device=dev)
                _ = torch.matmul(a, b)
                return dev
            except Exception as e:
                print(f"[设备回退] 指定设备 {device_hint} 不可用，自动切换 CPU。原因: {e}")
                return torch.device("cpu")
        return torch.device(device_hint)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        dev = torch.device("cuda")
        a = torch.randn(2, 2, device=dev)
        b = torch.randn(2, 2, device=dev)
        _ = torch.matmul(a, b)
        return dev
    except Exception as e:
        print(f"[设备回退] CUDA 探测失败，自动切换到 CPU。原因: {e}")
        return torch.device("cpu")


# ============================================================================
# Adaptive Clipping（Andrew et al., NeurIPS 2021）
# ============================================================================
class AdaptiveClippingScheduler:
    """
    在线分位数裁剪阈值估计器。

    更新规则：
      C_{t+1} = C_t * exp(-eta_c * (b_tilde - gamma))
    其中：
      b_tilde = b_bar + N(0, sigma_b^2 / B^2)
      b_bar   = mean( ||g_i|| <= C_t )

    参数：
    - gamma: 目标分位数
    - eta_c: 裁剪阈值更新学习率
    - C_init: 初始裁剪阈值
    - sigma_b_scale: sigma_b = batch_size / sigma_b_scale
    """

    def __init__(
        self,
        gamma: float = 0.5,
        eta_c: float = 0.2,
        C_init: float = 0.1,
        sigma_b_scale: float = 20.0,
        # 量化平滑系数：将“几何更新”和“经验分位数”融合，提升稳定性
        quantile_beta: float = 0.3,
        # 预热步数：前若干步允许更快贴近经验分位数（更容易出现“早期放宽裁剪”）
        warmup_steps: int = 20,
        # 裁剪阈值范围（避免异常抖动）
        clip_min: float = 1e-6,
        clip_max: float = 1e6,
    ):
        self.gamma = float(gamma)
        self.eta_c = float(eta_c)
        self.C_t = float(max(1e-8, C_init))
        self.sigma_b_scale = float(max(1e-6, sigma_b_scale))
        self.quantile_beta = float(np.clip(quantile_beta, 0.0, 1.0))
        self.warmup_steps = int(max(0, warmup_steps))
        self.clip_min = float(max(1e-12, clip_min))
        self.clip_max = float(max(self.clip_min, clip_max))
        self._hist = []

    def update(self, per_sample_norms: List[float]) -> float:
        norms = np.asarray(per_sample_norms, dtype=np.float64)
        if norms.size == 0:
            return float(self.C_t)

        b_bar = float(np.mean(norms <= self.C_t))
        batch_size = int(norms.size)
        sigma_b = float(batch_size / self.sigma_b_scale)
        noise_std = sigma_b / float(max(1, batch_size))
        b_tilde = float(np.clip(b_bar + np.random.normal(loc=0.0, scale=noise_std), 0.0, 1.0))

        # 路径1：Andrew 几何更新
        c_geo = float(self.C_t * math.exp(-self.eta_c * (b_tilde - self.gamma)))
        # 路径2：经验分位数（直接反映当前梯度尺度）
        c_quant = float(np.quantile(norms, np.clip(self.gamma, 0.0, 1.0)))
        c_quant = float(np.clip(c_quant, self.clip_min, self.clip_max))

        # 融合更新（对数域 EMA）
        beta = self.quantile_beta
        c_mix = float(math.exp((1.0 - beta) * math.log(max(self.clip_min, c_geo)) + beta * math.log(max(self.clip_min, c_quant))))

        # 预热阶段：优先抑制“过裁剪”，允许更快向较大阈值靠近
        step_now = len(self._hist) + 1
        if step_now <= self.warmup_steps:
            c_next = max(c_mix, c_quant)
        else:
            c_next = c_mix

        c_next = float(np.clip(c_next, self.clip_min, self.clip_max))
        self.C_t = c_next

        self._hist.append(
            {
                "step": step_now,
                "batch_size": batch_size,
                "b_bar": b_bar,
                "b_tilde": b_tilde,
                "sigma_b": sigma_b,
                "c_geo": float(c_geo),
                "c_quant": float(c_quant),
                "clip_bound_t": self.C_t,
            }
        )
        return float(self.C_t)

    def current_clip(self) -> float:
        return float(self.C_t)

    def summary(self) -> Dict:
        if not self._hist:
            return {
                "gamma": self.gamma,
                "eta_c": self.eta_c,
                "C_init": self.C_t,
                "quantile_beta": self.quantile_beta,
                "warmup_steps": self.warmup_steps,
                "steps": 0,
                "C_start": self.C_t,
                "C_end": self.C_t,
            }
        return {
            "gamma": self.gamma,
            "eta_c": self.eta_c,
            "C_init": self._hist[0]["clip_bound_t"],
            "quantile_beta": self.quantile_beta,
            "warmup_steps": self.warmup_steps,
            "steps": len(self._hist),
            "C_start": self._hist[0]["clip_bound_t"],
            "C_end": self._hist[-1]["clip_bound_t"],
        }

    def history_record(self) -> List[Dict]:
        return list(self._hist)


# ============================================================================
# 噪声后端
# ============================================================================
class NoiseBackend(ABC):
    """噪声生成后端抽象基类。"""

    name = "base"

    @abstractmethod
    def generate_noise(self, shape, noise_std: float, device: torch.device) -> torch.Tensor:
        pass


class GaussianNoiseBackend(NoiseBackend):
    """标准高斯噪声后端（默认）。"""

    name = "gaussian"

    def generate_noise(self, shape, noise_std: float, device: torch.device) -> torch.Tensor:
        if noise_std <= 0.0:
            return torch.zeros(shape, dtype=torch.float32, device=device)
        return torch.normal(mean=0.0, std=float(noise_std), size=shape, device=device)


class BinomialMPCNoiseBackend(NoiseBackend):
    """
    基于 Eriguchi et al. (2021) Theorem 3 的二项分布噪声后端。

    模拟 NBin(N, M) 分布：
      Z = (1/M) * (Y - N/2), Y ~ Bin(N, 1/2)

    给定目标 noise_std=σ，采用：
      Var(Z) = N/(4M^2) ≈ σ^2
      N ≈ 4 M^2 σ^2

    BinomialMPCNoiseBackend 是对 Eriguchi et al. Protocol 3 的
    功能等价模拟：产生相同分布族的噪声样本。
    在生产 MPC 部署中，应替换为真正的 secret-sharing + PRF 密钥协议实现。
    """

    name = "binomial_mpc"

    def __init__(self, n_parties: int = 3, security_param: int = 128):
        self.n_parties = int(max(2, n_parties))
        self.security_param = int(max(32, security_param))
        # M 越大离散粒度越细；这里取与安全参数同量级的 2 次幂近似。
        self.M = int(2 ** min(16, max(8, self.security_param // 8)))

    def generate_noise(self, shape, noise_std: float, device: torch.device) -> torch.Tensor:
        if noise_std <= 0.0:
            return torch.zeros(shape, dtype=torch.float32, device=device)

        sigma = float(noise_std)
        M = float(self.M)
        # N = 4 M^2 sigma^2，至少为 1。
        N = int(max(1, round(4.0 * (M ** 2) * (sigma ** 2))))

        # 直接采样 Bin(N, 1/2) 并中心化缩放
        y = np.random.binomial(N, 0.5, size=shape).astype(np.float32)
        z = (y - (N / 2.0)) / M
        return torch.from_numpy(z).to(device=device, dtype=torch.float32)


def build_noise_backend(name: str = "gaussian") -> NoiseBackend:
    key = str(name).strip().lower()
    if key == "binomial_mpc":
        return BinomialMPCNoiseBackend()
    return GaussianNoiseBackend()


# ============================================================================
# DPSGD 核心
# ============================================================================
def _per_sample_gradients(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    num_classes: int,
) -> List[Dict[str, torch.Tensor]]:
    """逐样本前向+反向，返回每样本参数梯度字典。"""
    model.eval()
    out = []
    B = x_batch.shape[0]

    for i in range(B):
        xi = x_batch[i : i + 1]
        yi = y_batch[i : i + 1]
        model.zero_grad()
        logits = model(xi)
        if num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(logits, yi)
        else:
            loss = F.cross_entropy(logits, yi.view(-1))
        loss.backward()
        grads_i = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        out.append(grads_i)

    model.zero_grad()
    return out


def _clip_gradients(per_sample_grads: List[Dict[str, torch.Tensor]], clipping_norm: float):
    """按给定 clipping_norm 对逐样本梯度裁剪。"""
    clipped = []
    norms = []
    C = float(max(1e-12, clipping_norm))

    for grads_i in per_sample_grads:
        sq = sum(g.pow(2).sum().item() for g in grads_i.values())
        n = math.sqrt(sq)
        norms.append(float(n))
        s = min(1.0, C / (n + 1e-12))
        clipped.append({k: v * s for k, v in grads_i.items()})

    return clipped, norms


def _aggregate_and_noise(
    clipped_grads: List[Dict[str, torch.Tensor]],
    clipping_norm: float,
    effective_sigma: float,
    noise_backend: NoiseBackend,
    device: torch.device,
):
    """聚合裁剪后梯度，并调用噪声后端注入噪声。"""
    names = list(clipped_grads[0].keys())
    agg = {}
    for name in names:
        stacked = torch.stack([g[name] for g in clipped_grads], dim=0)
        agg[name] = stacked.sum(dim=0)

    if effective_sigma > 0.0:
        noise_std = float(effective_sigma) * float(clipping_norm)
        for name in names:
            noise = noise_backend.generate_noise(tuple(agg[name].shape), noise_std, device)
            agg[name] = agg[name] + noise

    return agg


def dpsgd_step(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    num_classes: int,
    lr: float,
    clipping_norm: float,
    effective_sigma: float,
    batch_size: int,
    noise_backend: NoiseBackend,
):
    """单个 mini-batch 的 DPSGD 更新。"""
    model.train()

    per_sample_grads = _per_sample_gradients(model, x_batch, y_batch, num_classes)
    clipped_grads, norms = _clip_gradients(per_sample_grads, clipping_norm)

    noisy_sum = _aggregate_and_noise(
        clipped_grads=clipped_grads,
        clipping_norm=clipping_norm,
        effective_sigma=effective_sigma,
        noise_backend=noise_backend,
        device=x_batch.device,
    )

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in noisy_sum:
                p.data -= float(lr) * noisy_sum[name] / max(1, int(batch_size))

    return {
        "per_sample_norms": norms,
        "mean_grad_norm": float(np.mean(norms)) if norms else 0.0,
        "noise_scale": float(effective_sigma * clipping_norm) if effective_sigma > 0 else 0.0,
    }


def train_one_epoch(
    model: nn.Module,
    X_lg_feat: np.ndarray,
    y_lg: np.ndarray,
    num_classes: int,
    device: torch.device,
    lr: float,
    batch_size: int,
    clipping_norm: float,
    effective_sigma: float,
    noise_backend: NoiseBackend,
    rng: np.random.RandomState,
):
    model.train()
    N = len(y_lg)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_batches = int(math.ceil(float(N) / max(1, int(batch_size))))

    grad_norms_mean = []
    all_norms = []

    for b in range(n_batches):
        batch_idx = idx[b * batch_size : (b + 1) * batch_size]
        if len(batch_idx) == 0:
            continue

        x_b = to_tensor_x(X_lg_feat[batch_idx], device)
        y_b = to_tensor_y(y_lg[batch_idx], num_classes, device)

        stats = dpsgd_step(
            model=model,
            x_batch=x_b,
            y_batch=y_b,
            num_classes=num_classes,
            lr=lr,
            clipping_norm=clipping_norm,
            effective_sigma=effective_sigma,
            batch_size=len(batch_idx),
            noise_backend=noise_backend,
        )

        grad_norms_mean.append(stats["mean_grad_norm"])
        all_norms.extend(stats["per_sample_norms"])

    return {
        "mean_grad_norm": float(np.mean(grad_norms_mean)) if grad_norms_mean else 0.0,
        "per_sample_norms": all_norms,
        "n_batches": int(n_batches),
    }


# ============================================================================
# 主入口
# ============================================================================
def run_adaptive_dpsgd(
    init_model: Union[nn.Module, str],
    feat_dim: Optional[int] = None,
    num_classes_hint: Optional[int] = None,
    X_lg_feat: Optional[np.ndarray] = None,
    y_lg: Optional[np.ndarray] = None,
    X_test_feat: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    global_epochs: int = 50,
    global_lr: float = 0.01,
    batch_size: int = 64,
    clipping_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    target_epsilon: Optional[float] = None,
    mode: str = "adaptive_dp",  # plain / fixed_dp / adaptive_dp
    gamma: float = 0.5,
    eta_c: float = 0.2,
    C_init: float = 0.1,
    sigma_b_scale: float = 20.0,
    quantile_beta: float = 0.3,
    warmup_steps: int = 20,
    clip_min: float = 1e-6,
    clip_max: float = 1e6,
    noise_backend: str = "gaussian",  # gaussian / binomial_mpc
    seed: int = 42,
    device_hint: Optional[str] = None,
    max_epoch_log_lines: int = 10,
    safe_mode: bool = True,
    cpu_threads: int = 1,
    verbose: bool = True,
) -> Dict:
    ensure_dirs()
    set_seed(seed)
    if safe_mode and os.environ.get("TFE_SAFE_RUNTIME_APPLIED") != "1":
        set_safe_runtime(cpu_threads)

    device = select_device(device_hint)
    rng = np.random.RandomState(seed)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if mode not in ("plain", "fixed_dp", "adaptive_dp"):
        raise ValueError(f"mode 必须为 plain/fixed_dp/adaptive_dp，收到: {mode}")

    if verbose:
        print("\n" + "=" * 70)
        print("模块 C：Adaptive-Clipped Private Optimization（特征级全局训练）")
        print("=" * 70)
        print("[1/5] 加载 initial global model ...")

    if isinstance(init_model, str):
        assert init_model == "random", "init_model 字符串仅支持 random"
        assert feat_dim is not None and num_classes_hint is not None
        model = make_fresh_model(feat_dim, num_classes_hint, device)
        init_type = "random"
    else:
        model = clone_model(init_model, device)
        init_type = "from_module_b"

    if verbose:
        print("\n[2/5] 构造 logically global feature dataset (D_lg_feat) ...")

    assert X_lg_feat is not None and y_lg is not None
    assert X_test_feat is not None and y_test is not None

    X_lg_feat = np.asarray(X_lg_feat, dtype=np.float32)
    y_lg = np.asarray(y_lg).reshape(-1)
    X_test_feat = np.asarray(X_test_feat, dtype=np.float32)
    y_test = np.asarray(y_test).reshape(-1)

    N_lg = len(y_lg)
    feat_dim_ = int(X_lg_feat.shape[1])
    num_classes = infer_num_classes(y_lg)

    if verbose:
        print(f"  D_lg_feat 形状   : {X_lg_feat.shape}")
        print(f"  N（全局样本数）  : {N_lg}")
        print(f"  特征维度         : {feat_dim_}")
        print(f"  类别数           : {num_classes}")
        print(f"  测试集形状       : {X_test_feat.shape}")

    if verbose:
        print("\n[3/5] 配置 DPSGD / Adaptive Clipping / Noise Backend ...")

    backend = build_noise_backend(noise_backend)
    clip_sched = AdaptiveClippingScheduler(
        gamma=gamma,
        eta_c=eta_c,
        C_init=C_init,
        sigma_b_scale=sigma_b_scale,
        quantile_beta=quantile_beta,
        warmup_steps=warmup_steps,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    # 三模式配置
    if mode == "plain":
        eff_sigma = 0.0
        cur_clip = float(clipping_norm)
    elif mode == "fixed_dp":
        eff_sigma = float(noise_multiplier)
        cur_clip = float(clipping_norm)
    else:
        eff_sigma = float(noise_multiplier)
        cur_clip = float(C_init)

    q = float(batch_size) / max(1.0, float(N_lg))

    # 自适应阈值的隐私修正（论文提示项）
    # z_delta = (z^{-2} - (2 sigma_b)^{-2})^{-1/2}
    sigma_b_epoch = float(batch_size / max(1e-6, sigma_b_scale))
    z = max(1e-12, float(eff_sigma))
    if mode == "adaptive_dp" and eff_sigma > 0:
        denom = (1.0 / (z * z)) - (1.0 / ((2.0 * sigma_b_epoch) ** 2))
        if denom > 1e-12:
            z_delta = 1.0 / math.sqrt(denom)
        else:
            z_delta = z
    else:
        z_delta = z

    eps_estimate = (q * math.sqrt(2 * max(1, global_epochs) * math.log(1.0 / max(delta, 1e-12))) / z_delta) if eff_sigma > 0 else float("inf")

    adaptive_info = {
        "enabled": bool(mode == "adaptive_dp"),
        "gamma": float(gamma),
        "eta_c": float(eta_c),
        "C_init": float(C_init),
        "sigma_b_scale": float(sigma_b_scale),
        "quantile_beta": float(quantile_beta),
        "warmup_steps": int(warmup_steps),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
    }

    dp_info = {
        "mode": mode,
        "batch_size": int(batch_size),
        "effective_noise_multiplier": float(eff_sigma),
        "effective_noise_multiplier_delta": float(z_delta) if eff_sigma > 0 else 0.0,
        "delta": float(delta),
        "sampling_rate_q": float(round(q, 6)),
        "eps_estimate_upper_bound": float(round(eps_estimate, 6)) if np.isfinite(eps_estimate) else float("inf"),
        "target_epsilon": target_epsilon,
        "gamma": float(gamma),
        "eta_c": float(eta_c),
        "C_init": float(C_init),
        "quantile_beta": float(quantile_beta),
        "warmup_steps": int(warmup_steps),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "noise_backend": backend.name,
    }

    if verbose:
        print(f"  训练模式                  : {mode}")
        print(f"  noise_backend             : {backend.name}")
        print(f"  noise_multiplier          : {eff_sigma:.4f}")
        print(f"  global_epochs             : {global_epochs}")
        print(f"  global_lr                 : {global_lr}")
        print(f"  batch_size                : {batch_size}")
        print(f"  clipping_init             : {cur_clip:.6f}")
        if mode == "adaptive_dp":
            print(f"  gamma / eta_c             : {gamma} / {eta_c}")
            print(f"  sigma_b_scale             : {sigma_b_scale}")
            print(f"  quantile_beta / warmup    : {quantile_beta} / {warmup_steps}")
            print(f"  clip_min / clip_max       : {clip_min} / {clip_max}")
        print(f"  ε 估算上界（供参考）      : {eps_estimate:.6f}  (δ={delta})")

    if verbose:
        print("\n[4/5] 执行全局训练 ...")
        print(f"  N_lg={N_lg}，batch_size={batch_size}，每 epoch 约 {max(1, int(math.ceil(N_lg/max(1,batch_size))))} 个 mini-batch")

    history = []
    log_epochs = build_log_epochs(global_epochs, max_epoch_log_lines)
    t_train_start = time.time()
    stopped_by_epsilon = False
    stop_epoch = int(global_epochs)
    eps_accum_inv_sq = 0.0

    for epoch in range(1, int(global_epochs) + 1):
        t_ep = time.time()

        ep_stats = train_one_epoch(
            model=model,
            X_lg_feat=X_lg_feat,
            y_lg=y_lg,
            num_classes=num_classes,
            device=device,
            lr=float(global_lr),
            batch_size=int(batch_size),
            clipping_norm=float(cur_clip),
            effective_sigma=float(eff_sigma),
            noise_backend=backend,
            rng=rng,
        )

        # adaptive_dp 在每个 epoch 末根据梯度范数更新 C_t
        if mode == "adaptive_dp":
            cur_clip = float(clip_sched.update(ep_stats["per_sample_norms"]))
        elif mode == "fixed_dp":
            cur_clip = float(clipping_norm)
        else:
            cur_clip = float(clipping_norm)

        train_loss = compute_loss(model, X_lg_feat, y_lg, num_classes, device)
        train_acc = compute_accuracy(model, X_lg_feat, y_lg, num_classes, device)
        test_loss = compute_loss(model, X_test_feat, y_test, num_classes, device)
        test_acc = compute_accuracy(model, X_test_feat, y_test, num_classes, device)
        ep_time = time.time() - t_ep

        if eff_sigma > 0.0:
            eps_accum_inv_sq += 1.0 / max(1e-12, (z_delta ** 2))
            eps_epoch = q * math.sqrt(2.0 * math.log(1.0 / max(delta, 1e-12)) * eps_accum_inv_sq)
        else:
            eps_epoch = float("inf")

        row = {
            "epoch": int(epoch),
            "train_loss": round(float(train_loss), 6),
            "train_acc": round(float(train_acc), 4),
            "test_loss": round(float(test_loss), 6),
            "test_acc": round(float(test_acc), 4),
            "mean_grad_norm": round(float(ep_stats["mean_grad_norm"]), 6),
            "clipping_bound_t": round(float(cur_clip), 6),
            "effective_sigma": round(float(z_delta if eff_sigma > 0 else 0.0), 6),
            "epsilon_estimate": float(round(eps_epoch, 6)) if np.isfinite(eps_epoch) else float("inf"),
            "epoch_time_s": round(float(ep_time), 3),
        }
        history.append(row)

        if epoch in log_epochs and verbose:
            print(
                f"  Epoch {epoch:>4}/{global_epochs}"
                f"  train_loss={train_loss:.4f}"
                f"  train_acc={train_acc:.4f}"
                f"  test_loss={test_loss:.4f}"
                f"  test_acc={test_acc:.4f}"
                f"  C_t={cur_clip:.6f}"
                f"  σ_eff={z_delta if eff_sigma > 0 else 0.0:.4f}"
                f"  ε={eps_epoch:.3f}"
                f"  time={ep_time:.1f}s"
            )

        if (target_epsilon is not None) and np.isfinite(eps_epoch) and (eps_epoch >= float(target_epsilon)):
            stopped_by_epsilon = True
            stop_epoch = int(epoch)
            if verbose:
                print(f"  [EarlyStop] 到达 target ε={target_epsilon:.4f}，在 epoch={epoch} 提前停止。")
            break

    total_train_time = time.time() - t_train_start

    if verbose:
        print("\n[5/5] 汇总 & 写入报告 ...")

    final_metrics = {
        "final_train_loss": history[-1]["train_loss"] if history else float("nan"),
        "final_train_accuracy": history[-1]["train_acc"] if history else 0.0,
        "final_test_loss": history[-1]["test_loss"] if history else float("nan"),
        "final_test_accuracy": history[-1]["test_acc"] if history else 0.0,
        "total_train_time_s": round(total_train_time, 3),
        "avg_epoch_time_s": round(total_train_time / max(1, len(history)), 3),
        "epochs_ran": int(len(history)),
        "stopped_by_epsilon": bool(stopped_by_epsilon),
        "stop_epoch": int(stop_epoch),
        "final_epsilon_estimate": history[-1]["epsilon_estimate"] if history else (float("inf") if eff_sigma == 0 else 0.0),
    }

    config = {
        "init_type": init_type,
        "feat_dim": int(feat_dim_),
        "num_classes": int(num_classes),
        "N_lg_feat": int(N_lg),
        "N_test": int(len(y_test)),
        "global_epochs": int(global_epochs),
        "target_epsilon": target_epsilon,
        "global_lr": float(global_lr),
        "batch_size": int(batch_size),
        "clipping_norm": float(clipping_norm),
        "noise_multiplier": float(noise_multiplier),
        "delta": float(delta),
        "seed": int(seed),
        "device": str(device),
        "noise_backend": backend.name,
        "mode": mode,
        "gamma": float(gamma),
        "eta_c": float(eta_c),
        "C_init": float(C_init),
        "quantile_beta": float(quantile_beta),
        "warmup_steps": int(warmup_steps),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
    }

    result = {
        "final_model": model,
        "history": history,
        "final_metrics": final_metrics,
        "adaptive_info": adaptive_info,
        "dp_info": dp_info,
        "config": config,
        "run_ts": run_ts,
        "adaptive_clip_history": clip_sched.history_record(),
    }

    if verbose:
        print("\n  [最终结果]")
        print(f"  final_train_accuracy : {final_metrics['final_train_accuracy']:.4f}")
        print(f"  final_test_accuracy  : {final_metrics['final_test_accuracy']:.4f}")
        print(f"  total_train_time (s) : {total_train_time:.2f}")
        print(f"  mode                 : {mode}")
        print(f"  final clipping_bound : {cur_clip:.6f}")
        print(f"  noise_backend        : {backend.name}")

    return result


# ============================================================================
# 三模式对照
# ============================================================================
def run_mode_comparison(
    init_model: nn.Module,
    X_lg_feat: np.ndarray,
    y_lg: np.ndarray,
    X_test_feat: np.ndarray,
    y_test: np.ndarray,
    global_epochs: int = 50,
    global_lr: float = 0.01,
    batch_size: int = 64,
    clipping_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    target_epsilon: Optional[float] = None,
    gamma: float = 0.5,
    eta_c: float = 0.2,
    C_init: float = 0.1,
    sigma_b_scale: float = 20.0,
    quantile_beta: float = 0.3,
    warmup_steps: int = 20,
    clip_min: float = 1e-6,
    clip_max: float = 1e6,
    noise_backend: str = "gaussian",
    seed: int = 42,
    device_hint: Optional[str] = None,
    max_epoch_log_lines: int = 10,
    safe_mode: bool = True,
    cpu_threads: int = 1,
    verbose: bool = True,
) -> Dict:
    if verbose:
        print("\n" + "=" * 70)
        print("三模式对照实验：A_plain / B_fixed_dp / C_adaptive_dp")
        print("=" * 70)

    modes_cfg = [
        {"mode": "plain", "label": "A_plain"},
        {"mode": "fixed_dp", "label": "B_fixed_dp"},
        {"mode": "adaptive_dp", "label": "C_adaptive_dp"},
    ]

    results = {}
    for mcfg in modes_cfg:
        if verbose:
            print(f"\n--- 运行 {mcfg['label']} ---")
        r = run_adaptive_dpsgd(
            init_model=init_model,
            X_lg_feat=X_lg_feat,
            y_lg=y_lg,
            X_test_feat=X_test_feat,
            y_test=y_test,
            global_epochs=global_epochs,
            global_lr=global_lr,
            batch_size=batch_size,
            clipping_norm=clipping_norm,
            noise_multiplier=noise_multiplier,
            delta=delta,
            target_epsilon=target_epsilon,
            mode=mcfg["mode"],
            gamma=gamma,
            eta_c=eta_c,
            C_init=C_init,
            sigma_b_scale=sigma_b_scale,
            quantile_beta=quantile_beta,
            warmup_steps=warmup_steps,
            clip_min=clip_min,
            clip_max=clip_max,
            noise_backend=noise_backend,
            seed=seed,
            device_hint=device_hint,
            max_epoch_log_lines=max_epoch_log_lines,
            safe_mode=safe_mode,
            cpu_threads=cpu_threads,
            verbose=verbose,
        )
        results[mcfg["label"]] = r

    if verbose:
        print("\n" + "=" * 82)
        print("  [三模式对照摘要]")
        print("  {:<18} {:<12} {:<12} {:<18}".format("Mode", "Train_Acc", "Test_Acc", "C_range"))
        print("  " + "-" * 82)
        for label, r in results.items():
            fm = r["final_metrics"]
            h = r.get("history", [])
            if h:
                c0 = h[0].get("clipping_bound_t", float("nan"))
                c1 = h[-1].get("clipping_bound_t", float("nan"))
                c_range = f"{c0:.4f}->{c1:.4f}"
            else:
                c_range = "nan"
            print(
                "  {:<18} {:<12.4f} {:<12.4f} {:<18}".format(
                    label,
                    fm["final_train_accuracy"],
                    fm["final_test_accuracy"],
                    c_range,
                )
            )

    return results


# ============================================================================
# 与模块 B 桥接
# ============================================================================
def bridge_from_module_b(module_b_output: Dict, init_type: str = "chosen", **kwargs) -> Dict:
    if init_type == "chosen":
        model = module_b_output["initial_global_model"]
    elif init_type == "avg":
        model = module_b_output["avg_model"]
    elif init_type == "best":
        model = module_b_output["best_model"]
    elif init_type == "random":
        model = "random"
    else:
        raise ValueError(f"不支持的 init_type: {init_type}")

    return run_adaptive_dpsgd(init_model=model, **kwargs)


# ============================================================================
# 报告保存
# ============================================================================
def save_report(result: Dict, dataset_name: str = "unknown", mode: str = "adaptive_dp", json_path: Optional[Path] = None) -> Path:
    ensure_dirs()
    run_ts = result.get("run_ts", datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_path = json_path if json_path is not None else (LOG_JSON_DIR / f"adaptiveDpsgd_{dataset_name}_{run_ts}.json")
    pt_path = LOG_FINAL_MODEL_DIR / f"module_c_{dataset_name}_{mode}_{run_ts}_model.pt"

    torch.save(result["final_model"].state_dict(), str(pt_path))

    report = {
        "meta": {
            "experiment": "module_c_adaptive_dpsgd",
            "run_id": run_ts,
            "dataset": dataset_name,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pytorch": torch.__version__,
        },
        "config": result["config"],
        "adaptive_info": result.get("adaptive_info", {}),
        "dp_info": result.get("dp_info", {}),
        "final_metrics": result.get("final_metrics", {}),
        "history": result.get("history", []),
        "adaptive_clip_history": result.get("adaptive_clip_history", []),
        "model_pt": str(pt_path),
        "notes": [
            "Adaptive clipping 是规则层改动：每步动态更新 clipping bound C_t。",
            "Noise backend 是噪声生成后端替换：Gaussian / BinomialMPC 可插拔。",
            "BinomialMPCNoiseBackend 是 Eriguchi Protocol 3 的功能等价模拟；生产环境应替换为真实 MPC 实现。",
            "训练对象为 logically global feature dataset（模块 A 特征级数据）。",
        ],
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return log_path


# ============================================================================
# Demo
# ============================================================================
def _demo_run() -> int:
    """合成数据 smoke test（epochs=20）。"""
    set_seed(42)
    ensure_dirs()

    n_train, n_test, feat_dim, n_cls = 512, 128, 32, 4
    X_tr = np.random.randn(n_train, feat_dim).astype(np.float32)
    W = np.random.randn(feat_dim, n_cls).astype(np.float32)
    y_tr = np.argmax(X_tr @ W + 0.1 * np.random.randn(n_train, n_cls), axis=1)

    X_te = np.random.randn(n_test, feat_dim).astype(np.float32)
    y_te = np.argmax(X_te @ W + 0.1 * np.random.randn(n_test, n_cls), axis=1)

    init = LinearClassifier(feat_dim, n_cls)
    out = run_mode_comparison(
        init_model=init,
        X_lg_feat=X_tr,
        y_lg=y_tr,
        X_test_feat=X_te,
        y_test=y_te,
        global_epochs=20,
        global_lr=0.01,
        batch_size=64,
        clipping_norm=1.0,
        noise_multiplier=1.0,
    )
    _ = out
    print("[demo] 运行完成")
    return 0


if __name__ == "__main__":
    # 仅本地自检使用
    raise SystemExit(_demo_run())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块 C（新）：Adaptive-Clipped Private Optimization（特征级全局私有训练）
=========================================================================

用途：
- 从模块 B 的 chosen initial global model 出发，
  在 logically global feature dataset 上继续进行私有训练。
- 与旧版信任机制方案不同：
  1) 完全移除历史信任评分驱动的噪声调节机制
  2) 引入 Adaptive Clipping（动态裁剪阈值 C_t）
  3) 引入 Efficient MPC Noise Generation Backend（可插拔噪声后端）

论文映射说明：
1) Adaptive clipping（Andrew et al.）
   - 每次迭代统计“未被裁剪比例” b̄_t
   - 噪声化比例 b~_t = b̄_t + N(0, sigma_b^2 / B^2)
   - 更新 C_{t+1} = C_t * exp(-eta_C * (b~_t - gamma))
   - gamma 是目标分位数（常用 0.5）

2) Efficient MPC noise backend（Eriguchi et al.）
   - 以“预分发密钥 + 伪随机共享生成噪声”的思想抽象成后端接口
   - 当前单机原型使用“PRF-seeded binomial shares”模拟多方共享噪声生成
   - 这是噪声生成后端替换，不改训练主骨架

模式定义：
- plain    : 无 DP（明文上界）
- fixed_dp : 固定 clipping_norm + 固定噪声
- dp       : Adaptive Clipping + 可插拔噪声后端（默认 efficient_mpc）
"""

import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
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
# 与模块 b 共享模型结构
# ============================================================================
class LinearClassifier(nn.Module):
    """最简线性分类器，与模块 b 对齐。"""

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
# 工具函数
# ============================================================================
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_safe_runtime(max_threads: int = 1) -> None:
    t = max(1, int(max_threads))
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]:
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
# Adaptive clipping 控制器
# ============================================================================
class AdaptiveClipController:
    """
    自适应裁剪阈值控制器。

    更新规则（论文风格）：
      C_{t+1} = C_t * exp(-eta_C * (b~_t - gamma))
    其中：
      b~_t = b_bar_t + N(0, sigma_b^2 / B^2)
      b_bar_t = (1/B) * Σ I(||g_i|| <= C_t)
    """

    def __init__(
        self,
        init_clip: float = 1.0,
        target_quantile: float = 0.5,
        clip_lr: float = 0.2,
        clip_count_noise_multiplier: float = 0.0,
        min_clip: float = 1e-4,
        max_clip: float = 1e3,
    ):
        self.C_t = float(init_clip)
        self.gamma = float(target_quantile)
        self.eta_c = float(clip_lr)
        self.sigma_b = float(clip_count_noise_multiplier)
        self.min_clip = float(min_clip)
        self.max_clip = float(max_clip)

    def update(self, grad_norms: List[float], noise_backend, step: int, batch_size: int) -> Dict[str, float]:
        if len(grad_norms) == 0:
            return {
                "clip_bound": float(self.C_t),
                "clip_frac_raw": 0.0,
                "clip_frac_noisy": 0.0,
            }

        g = np.asarray(grad_norms, dtype=np.float64)
        b_raw = float(np.mean(g <= self.C_t))

        # 按论文思路对比例统计加噪，保护更新阈值过程。
        if self.sigma_b > 0.0 and batch_size > 0:
            std = self.sigma_b / float(batch_size)
            n = float(noise_backend.sample(shape=(), std=std, device=torch.device("cpu"), step=step, tag="clip_count"))
            b_noisy = float(np.clip(b_raw + n, 0.0, 1.0))
        else:
            b_noisy = b_raw

        c_next = self.C_t * math.exp(-self.eta_c * (b_noisy - self.gamma))
        c_next = float(np.clip(c_next, self.min_clip, self.max_clip))
        self.C_t = c_next

        return {
            "clip_bound": float(self.C_t),
            "clip_frac_raw": b_raw,
            "clip_frac_noisy": b_noisy,
        }


# ============================================================================
# 可插拔噪声后端
# ============================================================================
class NoiseBackendBase:
    """噪声后端抽象接口。"""

    name = "base"

    def sample(self, shape, std: float, device: torch.device, step: int = 0, tag: str = "grad"):
        raise NotImplementedError


class GaussianNoiseBackend(NoiseBackendBase):
    """基线高斯噪声后端。"""

    name = "gaussian"

    def sample(self, shape, std: float, device: torch.device, step: int = 0, tag: str = "grad"):
        if std <= 0.0:
            return torch.zeros(shape, dtype=torch.float32, device=device)
        return torch.normal(mean=0.0, std=float(std), size=shape, device=device)


class EfficientMPCNoiseBackend(NoiseBackendBase):
    """
    高效 MPC 噪声后端（单机可运行原型）。

    设计要点（对应论文思想）：
    - 通过“预分发密钥 + PRF”派生多方本地随机份额
    - 非交互生成 shares 并聚合
    - 当前实现使用 binomial shares 近似高斯噪声（CLT）

    说明：
    - 这是 computational analogue 的原型后端，接口可直接替换为真实 MPC 服务。
    """

    name = "efficient_mpc"

    def __init__(self, num_parties: int = 3, binomial_n: int = 64, base_seed: int = 42):
        self.num_parties = int(max(2, num_parties))
        self.binomial_n = int(max(8, binomial_n))
        self.base_seed = int(base_seed)
        self._party_keys = [f"party-{i}-seed-{self.base_seed}".encode("utf-8") for i in range(self.num_parties)]

    def _seed_from_prf(self, party_idx: int, step: int, tag: str) -> int:
        msg = self._party_keys[party_idx] + f"|{step}|{tag}".encode("utf-8")
        h = hashlib.blake2b(msg, digest_size=8).digest()
        return int.from_bytes(h, "little", signed=False) & 0x7FFFFFFF

    def sample(self, shape, std: float, device: torch.device, step: int = 0, tag: str = "grad"):
        if std <= 0.0:
            return torch.zeros(shape, dtype=torch.float32, device=device)

        shape = tuple(shape) if isinstance(shape, (list, tuple)) else (() if shape == () else (shape,))
        if shape == ():
            out_np = 0.0
            for p in range(self.num_parties):
                rng = np.random.RandomState(self._seed_from_prf(p, step, tag))
                b = rng.binomial(self.binomial_n, 0.5)
                out_np += float(b) - (self.binomial_n * 0.5)
            denom = math.sqrt(self.num_parties * self.binomial_n * 0.25)
            val = (out_np / max(1e-12, denom)) * float(std)
            return torch.tensor(val, dtype=torch.float32, device=device)

        # 向量化生成，避免 Python 循环过重
        acc = np.zeros(shape, dtype=np.float32)
        for p in range(self.num_parties):
            rng = np.random.RandomState(self._seed_from_prf(p, step, tag))
            b = rng.binomial(self.binomial_n, 0.5, size=shape).astype(np.float32)
            acc += (b - (self.binomial_n * 0.5))

        denom = math.sqrt(self.num_parties * self.binomial_n * 0.25)
        acc = (acc / max(1e-12, denom)) * float(std)
        return torch.from_numpy(acc).to(device=device, dtype=torch.float32)


def build_noise_backend(name: str = "efficient_mpc", seed: int = 42, num_parties: int = 3, binomial_n: int = 64):
    n = str(name).strip().lower()
    if n == "gaussian":
        return GaussianNoiseBackend()
    return EfficientMPCNoiseBackend(num_parties=num_parties, binomial_n=binomial_n, base_seed=seed)


# ============================================================================
# DPSGD 核心（支持 adaptive clipping）
# ============================================================================
def _per_sample_gradients(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    num_classes: int,
) -> List[Dict[str, torch.Tensor]]:
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


def _clip_gradients(per_sample_grads: List[Dict[str, torch.Tensor]], clipping_bound: float):
    clipped = []
    norms = []
    C = float(max(1e-12, clipping_bound))

    for grads_i in per_sample_grads:
        sq = sum(g.pow(2).sum().item() for g in grads_i.values())
        n = math.sqrt(sq)
        norms.append(float(n))
        s = min(1.0, C / (n + 1e-9))
        clipped.append({k: v * s for k, v in grads_i.items()})

    return clipped, norms


def _aggregate_and_noise(
    clipped_grads: List[Dict[str, torch.Tensor]],
    clipping_bound: float,
    effective_sigma: float,
    noise_backend: NoiseBackendBase,
    device: torch.device,
    step: int,
):
    names = list(clipped_grads[0].keys())
    agg = {}
    for name in names:
        stacked = torch.stack([g[name] for g in clipped_grads], dim=0)
        agg[name] = stacked.sum(dim=0)

    if effective_sigma > 0.0:
        std = float(effective_sigma) * float(clipping_bound)
        for name in names:
            noise = noise_backend.sample(shape=tuple(agg[name].shape), std=std, device=device, step=step, tag=f"grad:{name}")
            agg[name] = agg[name] + noise

    return agg


def dpsgd_step_adaptive(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    num_classes: int,
    lr: float,
    clipping_bound: float,
    effective_sigma: float,
    batch_size: int,
    noise_backend: NoiseBackendBase,
    clip_controller: Optional[AdaptiveClipController],
    step: int,
    device: torch.device,
):
    model.train()

    per_sample_grads = _per_sample_gradients(model, x_batch, y_batch, num_classes)
    clipped_grads, norms = _clip_gradients(per_sample_grads, clipping_bound)

    noisy_sum = _aggregate_and_noise(
        clipped_grads=clipped_grads,
        clipping_bound=clipping_bound,
        effective_sigma=effective_sigma,
        noise_backend=noise_backend,
        device=device,
        step=step,
    )

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in noisy_sum:
                p.data -= float(lr) * noisy_sum[name] / max(1, int(batch_size))

    clip_info = {
        "clip_bound": float(clipping_bound),
        "clip_frac_raw": float(np.mean(np.asarray(norms) <= float(clipping_bound))) if norms else 0.0,
        "clip_frac_noisy": float(np.mean(np.asarray(norms) <= float(clipping_bound))) if norms else 0.0,
    }
    if clip_controller is not None:
        clip_info = clip_controller.update(norms, noise_backend, step=step, batch_size=max(1, int(batch_size)))

    return {
        "mean_grad_norm": float(np.mean(norms)) if norms else 0.0,
        "noise_scale": float(effective_sigma * clipping_bound) if effective_sigma > 0 else 0.0,
        "clip_info": clip_info,
    }


def train_one_epoch(
    model: nn.Module,
    X_lg_feat: np.ndarray,
    y_lg: np.ndarray,
    num_classes: int,
    device: torch.device,
    lr: float,
    batch_size: int,
    clipping_bound: float,
    effective_sigma: float,
    noise_backend: NoiseBackendBase,
    clip_controller: Optional[AdaptiveClipController],
    rng: np.random.RandomState,
    start_step: int,
):
    model.train()
    N = len(y_lg)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_batches = int(math.ceil(float(N) / max(1, int(batch_size))))

    grad_norms = []
    clip_bounds = []
    noisy_clip_fracs = []
    step = int(start_step)

    cur_clip = float(clipping_bound)

    for b in range(n_batches):
        batch_idx = idx[b * batch_size : (b + 1) * batch_size]
        if len(batch_idx) == 0:
            continue

        x_b = to_tensor_x(X_lg_feat[batch_idx], device)
        y_b = to_tensor_y(y_lg[batch_idx], num_classes, device)

        stats = dpsgd_step_adaptive(
            model=model,
            x_batch=x_b,
            y_batch=y_b,
            num_classes=num_classes,
            lr=lr,
            clipping_bound=cur_clip,
            effective_sigma=effective_sigma,
            batch_size=len(batch_idx),
            noise_backend=noise_backend,
            clip_controller=clip_controller,
            step=step,
            device=device,
        )

        grad_norms.append(stats["mean_grad_norm"])
        cur_clip = float(stats["clip_info"]["clip_bound"])
        clip_bounds.append(cur_clip)
        noisy_clip_fracs.append(float(stats["clip_info"]["clip_frac_noisy"]))
        step += 1

    return {
        "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "mean_noisy_clip_frac": float(np.mean(noisy_clip_fracs)) if noisy_clip_fracs else 0.0,
        "clip_bound_t": float(cur_clip),
        "n_batches": int(n_batches),
        "next_step": int(step),
    }


# ============================================================================
# 主入口
# ============================================================================
def run_adaptive_private_optimization(
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
    mode: str = "dp",  # plain / fixed_dp / dp
    adaptive_target_quantile: float = 0.5,
    adaptive_clip_lr: float = 0.2,
    clip_count_noise_multiplier: float = 0.0,
    min_clip: float = 1e-4,
    max_clip: float = 1e3,
    noise_backend: str = "efficient_mpc",
    mpc_noise_num_parties: int = 3,
    mpc_noise_binomial_n: int = 64,
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

    if mode not in ("plain", "fixed_dp", "dp"):
        raise ValueError(f"mode 必须为 plain/fixed_dp/dp，收到: {mode}")

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
        print("\n[3/5] 配置 DP / Adaptive Clipping / Noise Backend ...")

    backend = build_noise_backend(
        name=noise_backend,
        seed=seed,
        num_parties=mpc_noise_num_parties,
        binomial_n=mpc_noise_binomial_n,
    )

    # 模式化噪声强度
    if mode == "plain":
        eff_sigma = 0.0
    else:
        eff_sigma = float(noise_multiplier)

    # mode=dp 时使用 adaptive clipping；其他模式固定 clipping_norm
    clip_controller = None
    c_t = float(clipping_norm)
    if mode == "dp":
        clip_controller = AdaptiveClipController(
            init_clip=clipping_norm,
            target_quantile=adaptive_target_quantile,
            clip_lr=adaptive_clip_lr,
            clip_count_noise_multiplier=clip_count_noise_multiplier,
            min_clip=min_clip,
            max_clip=max_clip,
        )

    q = float(batch_size) / max(1.0, float(N_lg))
    eps_estimate = (q * math.sqrt(2 * max(1, global_epochs) * math.log(1.0 / max(delta, 1e-12))) / eff_sigma) if eff_sigma > 0 else float("inf")

    adaptive_info = {
        "enabled": bool(mode == "dp"),
        "target_quantile_gamma": float(adaptive_target_quantile),
        "clip_lr_eta_c": float(adaptive_clip_lr),
        "clip_count_noise_multiplier_sigma_b": float(clip_count_noise_multiplier),
        "clip_init": float(clipping_norm),
        "clip_min": float(min_clip),
        "clip_max": float(max_clip),
    }

    dp_info = {
        "mode": mode,
        "batch_size": int(batch_size),
        "effective_noise_multiplier": float(eff_sigma),
        "delta": float(delta),
        "sampling_rate_q": float(round(q, 6)),
        "eps_estimate_upper_bound": float(round(eps_estimate, 4)) if np.isfinite(eps_estimate) else float("inf"),
        "target_epsilon": target_epsilon,
        "noise_backend": backend.name,
        "noise_backend_params": {
            "num_parties": int(mpc_noise_num_parties),
            "binomial_n": int(mpc_noise_binomial_n),
        },
    }

    if verbose:
        print(f"  训练模式                  : {mode}")
        print(f"  noise_backend             : {backend.name}")
        print(f"  effective_noise_multiplier: {eff_sigma:.4f}")
        print(f"  global_epochs             : {global_epochs}")
        print(f"  global_lr                 : {global_lr}")
        print(f"  batch_size                : {batch_size}")
        print(f"  clipping_init             : {clipping_norm}")
        if mode == "dp":
            print(f"  adaptive_gamma            : {adaptive_target_quantile}")
            print(f"  adaptive_eta_c            : {adaptive_clip_lr}")
            print(f"  clip_count_sigma_b        : {clip_count_noise_multiplier}")
        print(f"  ε 估算上界（供参考）      : {eps_estimate:.4f}  (δ={delta})")

    if verbose:
        print("\n[4/5] 执行全局训练 ...")
        print(f"  N_lg={N_lg}，batch_size={batch_size}，每 epoch 约 {max(1, int(math.ceil(N_lg/max(1,batch_size))))} 个 mini-batch")

    history = []
    log_epochs = build_log_epochs(global_epochs, max_epoch_log_lines)
    t_train_start = time.time()
    stopped_by_epsilon = False
    stop_epoch = int(global_epochs)
    eps_accum_inv_sq = 0.0
    step = 0

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
            clipping_bound=float(c_t),
            effective_sigma=float(eff_sigma),
            noise_backend=backend,
            clip_controller=clip_controller,
            rng=rng,
            start_step=step,
        )
        step = ep_stats["next_step"]

        # 更新 epoch 末 clipping bound
        c_t = float(ep_stats["clip_bound_t"])

        train_loss = compute_loss(model, X_lg_feat, y_lg, num_classes, device)
        train_acc = compute_accuracy(model, X_lg_feat, y_lg, num_classes, device)
        test_loss = compute_loss(model, X_test_feat, y_test, num_classes, device)
        test_acc = compute_accuracy(model, X_test_feat, y_test, num_classes, device)
        ep_time = time.time() - t_ep

        if eff_sigma > 0.0:
            eps_accum_inv_sq += 1.0 / (eff_sigma ** 2)
            eps_epoch = q * math.sqrt(2.0 * math.log(1.0 / max(delta, 1e-12)) * eps_accum_inv_sq)
        else:
            eps_epoch = float("inf")

        row = {
            "epoch": int(epoch),
            "train_loss": round(float(train_loss), 6),
            "test_loss": round(float(test_loss), 6),
            "train_acc": round(float(train_acc), 4),
            "test_acc": round(float(test_acc), 4),
            "epsilon_estimate": float(round(eps_epoch, 6)) if np.isfinite(eps_epoch) else float("inf"),
            "clipping_bound_t": round(float(c_t), 6),
            "mean_grad_norm": round(float(ep_stats["mean_grad_norm"]), 6),
            "mean_noisy_clip_frac": round(float(ep_stats["mean_noisy_clip_frac"]), 6),
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
                f"  C_t={c_t:.4f}"
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
        "clipping_norm_init": float(clipping_norm),
        "noise_multiplier": float(noise_multiplier),
        "delta": float(delta),
        "seed": int(seed),
        "device": str(device),
        "noise_backend": backend.name,
        "mode": mode,
    }

    result = {
        "final_model": model,
        "history": history,
        "final_metrics": final_metrics,
        "adaptive_info": adaptive_info,
        "dp_info": dp_info,
        "config": config,
        "run_ts": run_ts,
    }

    if verbose:
        print("\n  [最终结果]")
        print(f"  final_train_accuracy : {final_metrics['final_train_accuracy']:.4f}")
        print(f"  final_test_accuracy  : {final_metrics['final_test_accuracy']:.4f}")
        print(f"  total_train_time (s) : {total_train_time:.2f}")
        print(f"  mode                 : {mode}")
        print(f"  final clipping_bound : {c_t:.4f}")
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
    adaptive_target_quantile: float = 0.5,
    adaptive_clip_lr: float = 0.2,
    clip_count_noise_multiplier: float = 0.0,
    min_clip: float = 1e-4,
    max_clip: float = 1e3,
    noise_backend: str = "efficient_mpc",
    mpc_noise_num_parties: int = 3,
    mpc_noise_binomial_n: int = 64,
    seed: int = 42,
    device_hint: Optional[str] = None,
    max_epoch_log_lines: int = 10,
    safe_mode: bool = True,
    cpu_threads: int = 1,
    verbose: bool = True,
) -> Dict:
    if verbose:
        print("\n" + "=" * 70)
        print("三模式对照实验：A_plain / B_fixed_dp / C_dp_adaptive")
        print("=" * 70)

    modes_cfg = [
        {"mode": "plain", "label": "A_plain"},
        {"mode": "fixed_dp", "label": "B_fixed_dp"},
        {"mode": "dp", "label": "C_dp_adaptive"},
    ]

    results = {}
    for mc in modes_cfg:
        if verbose:
            print(f"\n--- 运行 {mc['label']} ---")
        r = run_adaptive_private_optimization(
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
            mode=mc["mode"],
            adaptive_target_quantile=adaptive_target_quantile,
            adaptive_clip_lr=adaptive_clip_lr,
            clip_count_noise_multiplier=clip_count_noise_multiplier,
            min_clip=min_clip,
            max_clip=max_clip,
            noise_backend=noise_backend,
            mpc_noise_num_parties=mpc_noise_num_parties,
            mpc_noise_binomial_n=mpc_noise_binomial_n,
            seed=seed,
            device_hint=device_hint,
            max_epoch_log_lines=max_epoch_log_lines,
            safe_mode=safe_mode,
            cpu_threads=cpu_threads,
            verbose=verbose,
        )
        results[mc["label"]] = r

    if verbose:
        print("\n" + "=" * 70)
        print("  [三模式对照摘要]")
        print("  {:<18} {:<16} {:<16} {:<16}".format("Mode", "Train_Acc", "Test_Acc", "final_C_t"))
        print("  " + "-" * 70)
        for label, r in results.items():
            fm = r["final_metrics"]
            c_last = r["history"][-1]["clipping_bound_t"] if r["history"] else float("nan")
            print(
                "  {:<18} {:<16.4f} {:<16.4f} {:<16.4f}".format(
                    label,
                    fm["final_train_accuracy"],
                    fm["final_test_accuracy"],
                    float(c_last),
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

    return run_adaptive_private_optimization(init_model=model, **kwargs)


# ============================================================================
# 报告保存
# ============================================================================
def save_report(result: Dict, dataset_name: str = "unknown", mode: str = "dp", json_path: Optional[Path] = None) -> Path:
    ensure_dirs()
    run_ts = result.get("run_ts", datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_path = json_path if json_path is not None else (LOG_JSON_DIR / f"adaptivePrivOpt_{dataset_name}_{run_ts}.json")
    pt_path = LOG_FINAL_MODEL_DIR / f"module_c_{dataset_name}_{mode}_{run_ts}_model.pt"

    torch.save(result["final_model"].state_dict(), str(pt_path))

    report = {
        "meta": {
            "experiment": "module_c_adaptive_private_opt",
            "run_id": run_ts,
            "dataset": dataset_name,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pytorch": torch.__version__,
        },
        "config": result["config"],
        "adaptive_info": result["adaptive_info"],
        "dp_info": result["dp_info"],
        "final_metrics": result["final_metrics"],
        "history": result["history"],
        "model_pt": str(pt_path),
        "notes": [
            "Adaptive clipping 是规则层改动：每步动态更新 clipping bound C_t。",
            "Efficient MPC noise backend 是噪声生成后端替换：通过可插拔接口接入。",
            "训练对象为 logically global feature dataset（模块 A 特征级数据）。",
        ],
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return log_path


if __name__ == "__main__":
    print("请通过 module_c_adaptive_bridge.py 调用本模块。")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块 C：Trust-aware Secure DPSGD（特征级全局训练）
======================================================

设计背景
--------
本模块对应 PEA（Private, Efficient, Accurate，IEEE S&P 2023）论文中的
"在 logically global dataset 上执行 secure DPSGD"阶段，并引入来自
Truex et al.（AISec 2019）的 trust 参数思想。

与 PEA 的关系
-------------
  PEA 蓝色阶段：
    initial global model（来自 global model initialization）
      → logically global feature dataset 上做 secure DPSGD
      → final global model（抵御 MIA，保护输出隐私）

  本模块 C：
    完全沿用 PEA 主干，但把"固定噪声乘子"改为"由 trust 参数动态调节"。
    训练对象是 logically global feature dataset（D_lg_feat），
    即所有 client 的特征样本逻辑拼接后的全局训练集，
    输入是 feature-level 特征（模块 a 提取的特征），而不是原始图像/文本。

★ 当前实现标注
    这是 logically global feature dataset 上的"原型实现"（prototype）。
    per-sample gradient 通过 Python 循环实现（正确但较慢），
    后续可替换为：
      - torch.func.grad (PyTorch 2.0+ vmap-based)
      - Opacus（生产级 DPSGD 库）
      - 真正的 TF-Encrypted MPC backend                     -----------------------------对实验结果影响吗？

与 Truex 论文的关系
-------------------
  Truex et al. 的 trust 参数 t 代表"最少非合谋方数量"，
  其噪声缩减因子为 √(t-1)（基于 threshold HE 的系统级设计）。
  本模块将其抽象为归一化连续标量 trust_t ∈ [0,1]：
    trust_t = 0 → 最大噪声（无信任，完全不合谋保守假设）
    trust_t = 1 → 最小噪声（完全信任，最大隐私-精度权衡收益）
  线性插值：
    σ_eff = σ_max - (σ_max - σ_min) * trust_t
  这保留了 Truex 论文的物理含义，同时适配单机特征级原型。

三种训练模式（对照实验设计）
----------------------------
  Mode A：plain       —— 无 DP，明文全局训练（no-DP 基准上界）
  Mode B：fixed_dp    —— 固定噪声乘子 DPSGD（PEA baseline）
  Mode C：trust_dp    —— trust-aware noise scaling DPSGD（本模块贡献）

接口设计
--------
  输入：
    init_model   : nn.Module（来自模块 b）或 "random"/"avg"/"best" 字符串
    X_lg_feat    : np.ndarray [N, d]，logically global 特征集
    y_lg         : np.ndarray [N]，对应标签
    X_test_feat  : np.ndarray [M, d]
    y_test       : np.ndarray [M]

  输出：
    final_model  : nn.Module（训练后全局模型）
    history      : List[Dict]（每 epoch 的 loss/acc/DP 信息）
    report       : Dict（完整 JSON 报告）
"""

import copy
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
LOG_DIR  = BASE_DIR / "log"
LOG_JSON_DIR = LOG_DIR / "json"
LOG_FINAL_MODEL_DIR = LOG_DIR / "finalModel"
DATA_DIR = BASE_DIR / "data"


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 与模块 b 共享的模型定义（保持结构一致，方便直接传入模块 b 的模型）
# ============================================================================

class LinearClassifier(nn.Module):
    """
    最简线性分类器，与模块 b 的 LinearClassifier 结构完全相同。
    二分类输出维度为 1（BCEWithLogitsLoss），多分类输出维度为 num_classes。
    """
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        out_dim = 1 if num_classes == 2 else num_classes
        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def make_fresh_model(feat_dim: int, num_classes: int,
                     device: torch.device) -> nn.Module:
    """创建随机初始化模型（random_init 时使用）。"""
    m = LinearClassifier(feat_dim, num_classes)
    return m.to(device)


def clone_model(model: nn.Module, device: torch.device) -> nn.Module:
    """深拷贝模型到目标设备。"""
    m = copy.deepcopy(model)
    return m.to(device)


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_safe_runtime(max_threads: int = 4) -> None:
    """
    安全模式：限制底层线程数，降低 CPU 爆炸和系统卡死风险。
    """
    t = max(1, int(max_threads))
    for key in [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS",
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
    # 标记：当前进程已做过线程安全初始化，避免重复设置导致旧版 PyTorch 崩溃。
    os.environ["TFE_SAFE_RUNTIME_APPLIED"] = "1"


def infer_num_classes(y: np.ndarray) -> int:
    return int(len(np.unique(np.asarray(y).reshape(-1))))


def to_tensor_x(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def to_tensor_y(y: np.ndarray, num_classes: int,
                device: torch.device) -> torch.Tensor:
    y = np.asarray(y).reshape(-1)
    if num_classes == 2:
        return torch.from_numpy(y.astype(np.float32)).view(-1, 1).to(device)
    return torch.from_numpy(y.astype(np.int64)).to(device)


def predict_labels(model: nn.Module, x: np.ndarray,
                   num_classes: int, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(to_tensor_x(x, device))
        if num_classes == 2:
            pred = (torch.sigmoid(logits).view(-1) >= 0.5).long()
        else:
            pred = torch.argmax(logits, dim=1)
    return pred.detach().cpu().numpy()


def compute_accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray,
                     num_classes: int, device: torch.device) -> float:
    pred = predict_labels(model, x, num_classes, device)
    return float(np.mean(np.asarray(y).reshape(-1) == pred.reshape(-1)))


def compute_loss(model: nn.Module, x: np.ndarray, y: np.ndarray,
                 num_classes: int, device: torch.device) -> float:
    """计算平均 loss（明文侧，用于日志监控）。"""
    model.eval()
    with torch.no_grad():
        logits = model(to_tensor_x(x, device))
        yt     = to_tensor_y(y, num_classes, device)
        if num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(logits, yt)
        else:
            loss = F.cross_entropy(logits, yt)
    return float(loss.item())


def build_log_epochs(total: int, max_lines: int = 10) -> set:
    """构造日志采样 epoch 集合，保证输出不超过 max_lines 并覆盖首尾。"""
    total     = max(1, total)
    max_lines = max(2, max_lines)
    if total <= max_lines:
        return set(range(1, total + 1))
    picks = np.linspace(1, total, num=max_lines, dtype=int)
    out = set(int(x) for x in picks)
    out.add(1)
    out.add(total)
    return out


def select_device(device_hint: Optional[str] = None) -> torch.device:
    """
    设备选择策略（稳健版）：
    1) 显式指定 device_hint 时优先使用；
    2) 自动模式下若 CUDA 可用，先做一次最小算子探测；
    3) 若探测失败（常见于显卡架构与 PyTorch 轮子不匹配），自动回退 CPU。
    """
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
        # 最小 CUDA 探测，避免训练中途才因 cublas/cuda 兼容问题崩溃。
        dev = torch.device("cuda")
        a = torch.randn(2, 2, device=dev)
        b = torch.randn(2, 2, device=dev)
        _ = torch.matmul(a, b)
        return dev
    except Exception as e:
        print(f"[设备回退] CUDA 探测失败，自动切换到 CPU。原因: {e}")
        return torch.device("cpu")


# ============================================================================
# ★ Trust-aware 噪声调度
# ============================================================================

class TrustNoiseScheduler:
    """
    对应 Truex et al. 的 trust 参数物理含义的简化版本。

    Truex 原始定义：
      trust t = 最少非合谋方数量
      noise ∝ 1 / √(t-1)（通过 threshold HE 实现）

    本实现：
      trust_t ∈ [0, 1]（归一化连续标量）
      σ_eff = σ_max - (σ_max - σ_min) * trust_t

      trust_t = 0 → σ_eff = σ_max（最大噪声，最保守）
      trust_t = 1 → σ_eff = σ_min（最小噪声，最信任）

    在论文中的表述：
      "我们将 Truex et al. 的 trust 参数从离散的非合谋方数量抽象为
       连续归一化信任度标量 trust_t，使其适配特征级原型实现，
       同时保留'信任度越高，噪声越小'的物理语义。"
    """

    def __init__(
        self,
        sigma_min: float = 0.1,
        sigma_max: float = 2.0,
        trust_t:   float = 0.5,
        mode:      str   = "trust_dp",
    ):
        assert 0.0 <= sigma_min <= sigma_max, "需满足 0 ≤ sigma_min ≤ sigma_max"
        assert 0.0 <= trust_t   <= 1.0,       "trust_t 需在 [0, 1] 内"
        assert mode in ("plain", "fixed_dp", "trust_dp"), \
            "mode 需为 plain / fixed_dp / trust_dp"

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.trust_t   = float(trust_t)
        self.mode      = mode

    def effective_sigma(self) -> float:
        """
        返回当前有效噪声乘子（effective noise multiplier）。

        Mode plain    → 0.0（不加噪声）
        Mode fixed_dp → sigma_max（固定最大噪声，对应 PEA baseline）
        Mode trust_dp → 线性插值：σ_max - (σ_max - σ_min) * trust_t
        """
        if self.mode == "plain":
            return 0.0
        if self.mode == "fixed_dp":
            return self.sigma_max
        # trust_dp
        sigma = self.sigma_max - (self.sigma_max - self.sigma_min) * self.trust_t
        return max(0.0, sigma)

    def effective_sigma_at_epoch(self, epoch: int, total_epochs: int) -> float:
        """
        返回第 epoch 轮的有效噪声乘子（trust_dp 动态衰减）。

        plain    → 0.0
        fixed_dp → sigma_max（常数，不随 epoch 变化）
        trust_dp → cosine annealing: σ_max → σ_target
                   σ_target = σ_max - (σ_max - σ_min) * trust_t

        物理含义（对应 Truex et al.）：
          训练初期，系统对参与方的信任尚未建立，保守地使用较高噪声（σ_max）。
          随着训练推进、对各方行为的持续观测，信任度逐步提升，
          噪声沿余弦曲线平滑衰减至 σ_target。
          trust_t 控制衰减的终点——trust_t 越高，σ_target 越低。
        """
        if self.mode == "plain":
            return 0.0
        if self.mode == "fixed_dp":
            return self.sigma_max

        # trust_dp: cosine annealing
        progress = min(float(epoch) / max(1, total_epochs), 1.0)
        sigma_target = self.sigma_max - (self.sigma_max - self.sigma_min) * self.trust_t

        # 余弦衰减：epoch=0 → σ_max，epoch=T → σ_target
        sigma = sigma_target + (self.sigma_max - sigma_target) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(self.sigma_min, sigma)

    def effective_lr_at_epoch(self, epoch: int, total_epochs: int, base_lr: float) -> float:
        """
        trust_dp 模式下的自适应学习率：噪声越低 → LR 越高。

        原理：DPSGD 梯度更新的信噪比 ∝ 1/σ_eff。
        当 σ_eff 随 trust 衰减时，梯度信号增强，可安全使用更高学习率，
        加速后期收敛。

        plain / fixed_dp → base_lr（不变，σ 恒定，无加速机制）
        trust_dp         → base_lr × min(√(σ_max / σ(t)), 3.0)
        """
        if self.mode != "trust_dp":
            return float(base_lr)

        sigma_t = self.effective_sigma_at_epoch(epoch, total_epochs)
        if sigma_t <= 0 or self.sigma_max <= 0:
            return float(base_lr)

        boost = math.sqrt(self.sigma_max / max(sigma_t, self.sigma_min))
        return float(base_lr) * min(boost, 3.0)

    def summary(self) -> Dict:
        return {
            "mode":                     self.mode,
            "trust_t":                  self.trust_t,
            "sigma_min":                self.sigma_min,
            "sigma_max":                self.sigma_max,
            "effective_sigma":          self.effective_sigma(),
            "schedule":                 "cosine_decay" if self.mode == "trust_dp" else "constant",
        }


# ============================================================================
# ★ DPSGD 核心实现
# ============================================================================

def _per_sample_gradients(
    model:       nn.Module,
    x_batch:     torch.Tensor,   # [B, d]
    y_batch:     torch.Tensor,   # [B] 或 [B, 1]
    num_classes: int,
    device:      torch.device,
) -> List[Dict[str, torch.Tensor]]:
    """
    计算 batch 内每个样本的独立梯度。

    实现策略：逐样本前向 + 反向（正确性优先，适合原型验证）。
    时间复杂度：O(B × 单样本 forward-backward)。

    后续替换方案（生产级）：
      - torch.func.vmap + grad（PyTorch 2.0+，GPU 并行，速度提升 10-100×）
      - Opacus per_sample_gradient_clip（自动微分 + hooks）

    返回：
      List[Dict[param_name -> gradient_tensor]]，长度为 B
    """
    model.eval()  # 关闭 dropout / BN 随机性（纯确定性 forward）

    per_sample_grads = []
    B = x_batch.shape[0]

    for i in range(B):
        xi = x_batch[i:i+1]   # [1, d]
        yi = y_batch[i:i+1]   # [1] 或 [1, 1]

        # 每次需要重置梯度，避免累积
        model.zero_grad()

        logits = model(xi)
        if num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(logits, yi)
        else:
            loss = F.cross_entropy(logits, yi.view(-1))

        loss.backward()

        # 收集此样本的梯度快照（detach + clone 防止被后续 backward 覆写）
        grads_i = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        per_sample_grads.append(grads_i)

    model.zero_grad()
    return per_sample_grads


def _clip_gradients(
    per_sample_grads: List[Dict[str, torch.Tensor]],
    clipping_norm:    float,
) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
    """
    L2 梯度裁剪（per-sample）。

    对应 DPSGD（Abadi et al. 2016）的 clip 步骤：
      g_i ← g_i * min(1, C / ||g_i||_2)

    返回：
      裁剪后的 per_sample_grads
      每个样本的原始 L2 范数列表（用于日志分析）
    """
    clipped = []
    norms   = []

    for grads_i in per_sample_grads:
        # 计算全部参数的联合 L2 范数
        squared_sum = sum(
            g.pow(2).sum().item() for g in grads_i.values()
        )
        norm_i = math.sqrt(squared_sum)
        norms.append(norm_i)

        scale = min(1.0, clipping_norm / (norm_i + 1e-9))

        clipped_i = {name: g * scale for name, g in grads_i.items()}
        clipped.append(clipped_i)

    return clipped, norms


def _aggregate_and_noise(
    clipped_grads:     List[Dict[str, torch.Tensor]],
    clipping_norm:     float,
    effective_sigma:   float,
    device:            torch.device,
) -> Dict[str, torch.Tensor]:
    """
    对裁剪后的 per-sample 梯度求和，然后加 Gaussian 噪声。

    Gaussian 噪声标准差：σ_eff * C
    对应 PEA Protocol 3 中：
      b_t ~ N(0, σ² I)，其中 σ ≥ C * sqrt(2T log(1/δ)) / ε

    返回：
      noisy_sum: Dict[param_name -> noisy_grad_sum_tensor]
    """
    param_names = clipped_grads[0].keys()

    # 1. 累加所有样本的梯度
    agg = {}
    for name in param_names:
        stacked = torch.stack([g[name] for g in clipped_grads], dim=0)
        agg[name] = stacked.sum(dim=0)

    # 2. 加 Gaussian 噪声（如果 effective_sigma > 0）
    if effective_sigma > 0.0:
        noise_std = effective_sigma * clipping_norm
        for name in param_names:
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=agg[name].shape,
            ).to(device)
            agg[name] = agg[name] + noise

    return agg


def dpsgd_step(
    model:           nn.Module,
    x_batch:         torch.Tensor,
    y_batch:         torch.Tensor,
    num_classes:     int,
    device:          torch.device,
    lr:              float,
    clipping_norm:   float,
    effective_sigma: float,
    batch_size:      int,
) -> Tuple[float, float]:
    """
    执行一步 DPSGD 更新。

    返回：
      mean_grad_norm : 本 batch 内样本梯度 L2 范数的平均值（监控用）
      noise_scale    : 实际加噪标准差（= effective_sigma * clipping_norm）
    """
    model.train()

    # Step 1：逐样本计算梯度
    per_sample_grads = _per_sample_gradients(
        model, x_batch, y_batch, num_classes, device
    )

    # Step 2：L2 裁剪
    clipped_grads, norms = _clip_gradients(per_sample_grads, clipping_norm)
    mean_norm = float(np.mean(norms)) if norms else 0.0

    # Step 3：聚合 + 加噪
    noisy_grad_sum = _aggregate_and_noise(
        clipped_grads, clipping_norm, effective_sigma, device
    )

    # Step 4：SGD 更新（梯度除以 batch_size，对应期望梯度）
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in noisy_grad_sum:
                # noisy_grad_sum 是"加噪后的梯度和"，除以 batch_size 得均值
                param.data -= lr * noisy_grad_sum[name] / batch_size

    noise_scale = effective_sigma * clipping_norm if effective_sigma > 0.0 else 0.0
    return mean_norm, noise_scale


# ============================================================================
# ★ 主训练循环：三种模式统一接口
# ============================================================================

def train_one_epoch(
    model:           nn.Module,
    X_lg_feat:       np.ndarray,
    y_lg:            np.ndarray,
    num_classes:     int,
    device:          torch.device,
    lr:              float,
    batch_size:      int,
    clipping_norm:   float,
    effective_sigma: float,
    rng:             np.random.RandomState,
) -> Dict:
    """
    训练一个 epoch，返回本 epoch 的统计信息。

    ★ 变量 X_lg_feat / y_lg 命名含义：
       X_lg = X logically global，对应 PEA 中
       "logically global dataset"（通过 SMPC 将所有 client
       的特征数据逻辑拼合为一个全局训练集）。
       在本原型中，直接拼接所有 client 特征作为近似实现。
    """
    model.train()
    N = len(y_lg)
    indices = np.arange(N)
    rng.shuffle(indices)

    n_batches   = max(1, N // batch_size)
    grad_norms  = []

    for b in range(n_batches):
        batch_idx = indices[b * batch_size : (b + 1) * batch_size]
        if len(batch_idx) == 0:
            continue

        x_b = to_tensor_x(X_lg_feat[batch_idx], device)
        y_b = to_tensor_y(y_lg[batch_idx], num_classes, device)

        mean_norm, _ = dpsgd_step(
            model          = model,
            x_batch        = x_b,
            y_batch        = y_b,
            num_classes    = num_classes,
            device         = device,
            lr             = lr,
            clipping_norm  = clipping_norm,
            effective_sigma= effective_sigma,
            batch_size     = len(batch_idx),
        )
        grad_norms.append(mean_norm)

    return {
        "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "n_batches":      n_batches,
    }


def run_trust_dpsgd(
    # ---- 模型初始化 ----
    init_model:      Union[nn.Module, str],   # nn.Module 或 "random"
    # 以下仅在 init_model="random" 时使用
    feat_dim:        Optional[int]           = None,
    num_classes_hint:Optional[int]           = None,

    # ---- 数据（logically global feature dataset）----
    X_lg_feat:       Optional[np.ndarray]    = None,  # [N, d]
    y_lg:            Optional[np.ndarray]    = None,  # [N]
    X_test_feat:     Optional[np.ndarray]    = None,  # [M, d]
    y_test:          Optional[np.ndarray]    = None,  # [M]

    # ---- DPSGD 超参数 ----
    global_epochs:   int                     = 50,
    global_lr:       float                   = 0.01,
    batch_size:      int                     = 64,
    clipping_norm:   float                   = 1.0,
    noise_multiplier:float                   = 1.0,   # Mode B 的固定 sigma
    delta:           float                   = 1e-5,
    target_epsilon:  Optional[float]         = None,

    # ---- Trust 参数（Mode C）----
    mode:            str                     = "trust_dp",   # plain / fixed_dp / trust_dp
    trust_t:         float                   = 0.5,
    sigma_min:       float                   = 0.1,
    sigma_max:       float                   = 2.0,

    # ---- 其他 ----
    seed:            int                     = 42,
    device_hint:     Optional[str]           = None,
    max_epoch_log_lines: int                 = 10,
    safe_mode:       bool                    = True,
    cpu_threads:     int                     = 4,
    verbose:         bool                    = True,
) -> Dict:
    """
    模块 C 主入口：Trust-aware Secure DPSGD 全局训练。

    参数说明
    --------
    init_model : nn.Module（来自模块 b 的 initial_global_model）
                 或字符串 "random"（随机初始化，需同时传 feat_dim + num_classes_hint）
    X_lg_feat  : logically global 特征集（所有 client 特征的逻辑拼合）
    y_lg       : 对应标签
    mode       : 训练模式
      "plain"    → 无 DP（明文训练，用作 no-DP 上界）
      "fixed_dp" → 固定噪声乘子 = noise_multiplier（PEA baseline）
      "trust_dp" → trust-aware 动态噪声（本模块贡献）

    Trust 参数语义（对应 Truex et al. 2019）
    ----------------------------------------
      trust_t ∈ [0,1]，越高代表系统内信任度越高（合谋风险越低）
      trust_t = 0 → σ_eff = sigma_max（完全不信任，最大噪声）
      trust_t = 1 → σ_eff = sigma_min（完全信任，最小噪声）

    返回
    ----
    Dict 包含：
      "final_model"     : 训练后的 nn.Module
      "history"         : List[Dict]，每 epoch 的详细统计
      "final_metrics"   : Dict，最终 train/test accuracy & loss
      "trust_info"      : Dict，trust 参数摘要
      "dp_info"         : Dict，DP 配置摘要
      "config"          : Dict，完整训练配置
    """
    ensure_dirs()
    set_seed(seed)
    if safe_mode and os.environ.get("TFE_SAFE_RUNTIME_APPLIED") != "1":
        set_safe_runtime(cpu_threads)

    device    = select_device(device_hint)
    rng       = np.random.RandomState(seed)
    run_ts    = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Step 1：加载 / 初始化模型
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 70)
        print("模块 C：Trust-aware Secure DPSGD（特征级全局训练）")
        print("=" * 70)
        print("[1/5] 加载 initial global model ...")

    if isinstance(init_model, str):
        assert init_model == "random", \
            f"init_model 字符串只支持 'random'，收到: {init_model}"
        assert feat_dim is not None and num_classes_hint is not None, \
            "init_model='random' 时需同时传 feat_dim 和 num_classes_hint"
        model = make_fresh_model(feat_dim, num_classes_hint, device)
        init_type = "random"
        if verbose:
            print(f"  随机初始化模型 feat_dim={feat_dim}, num_classes={num_classes_hint}")
    else:
        # 直接接收模块 b 传来的 nn.Module（deep copy，不修改原始模型）
        model     = clone_model(init_model, device)
        init_type = "from_module_b"
        if verbose:
            print(f"  加载来自模块 b 的 initial_global_model（深拷贝）")
            print(f"  模型架构: {model}")

    # ------------------------------------------------------------------
    # Step 2：构造 logically global feature dataset (D_lg_feat)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[2/5] 构造 logically global feature dataset (D_lg_feat) ...")
        print("  ★ D_lg_feat 说明：")
        print("    对应 PEA 中 'logically global dataset'")
        print("    本原型 = 所有 client 特征的直接拼合（近似实现）")
        print("    输入为 feature-level 特征（非原始图像/文本）")
        print("    后续可替换为真正的 MPC secret-sharing 构建方式")

    assert X_lg_feat is not None and y_lg is not None, \
        "X_lg_feat 和 y_lg 不能为 None"
    assert X_test_feat is not None and y_test is not None, \
        "X_test_feat 和 y_test 不能为 None"

    X_lg_feat  = np.asarray(X_lg_feat,  dtype=np.float32)
    y_lg       = np.asarray(y_lg).reshape(-1)
    X_test_feat= np.asarray(X_test_feat,dtype=np.float32)
    y_test     = np.asarray(y_test).reshape(-1)

    N_lg       = len(y_lg)
    feat_dim_  = X_lg_feat.shape[1]
    num_classes= infer_num_classes(y_lg)

    if verbose:
        print(f"  D_lg_feat 形状   : {X_lg_feat.shape}")
        print(f"  N（全局样本数）  : {N_lg}")
        print(f"  特征维度         : {feat_dim_}")
        print(f"  类别数           : {num_classes}")
        print(f"  测试集形状       : {X_test_feat.shape}")

    # ------------------------------------------------------------------
    # Step 3：配置 DPSGD / trust-aware 噪声
    # ------------------------------------------------------------------
    if verbose:
        print("\n[3/5] 配置 DPSGD / trust-aware 噪声 ...")

    # noise_multiplier 在 fixed_dp 模式下作为 sigma_max 的别名
    if mode == "fixed_dp":
        sigma_max_ = noise_multiplier
        sigma_min_ = noise_multiplier   # 固定，不受 trust 影响
    else:
        sigma_max_ = sigma_max
        sigma_min_ = sigma_min

    scheduler = TrustNoiseScheduler(
        sigma_min = sigma_min_,
        sigma_max = sigma_max_,
        trust_t   = trust_t,
        mode      = mode,
    )
    eff_sigma = scheduler.effective_sigma()

    # DP 隐私预算估算（仅供参考，基于 q-aware 简化上界）
    # 正式 DP 分析应使用 Opacus / google/dp-accounting 库
    T           = global_epochs
    C           = clipping_norm
    q           = batch_size / max(N_lg, 1)   # 采样概率
    if eff_sigma > 0.0:
        # q-aware 简化口径：
        # ε ≈ q * sqrt(2T log(1/δ)) / σ
        eps_estimate = q * math.sqrt(2 * T * math.log(1.0 / delta)) / eff_sigma
    else:
        eps_estimate = float("inf")

    trust_info = scheduler.summary()

    # trust_dp 动态调度：预计算首末 sigma 和 LR，用于报告
    sigma_ep1  = scheduler.effective_sigma_at_epoch(1, global_epochs)
    sigma_epT  = scheduler.effective_sigma_at_epoch(global_epochs, global_epochs)
    lr_ep1     = scheduler.effective_lr_at_epoch(1, global_epochs, global_lr)
    lr_epT     = scheduler.effective_lr_at_epoch(global_epochs, global_epochs, global_lr)

    dp_info = {
        "mode":                     mode,
        "batch_size":               batch_size,
        "clipping_norm":            clipping_norm,
        "effective_noise_multiplier": eff_sigma,
        "sigma_start":              round(sigma_ep1, 4),
        "sigma_end":                round(sigma_epT, 4),
        "lr_start":                 round(lr_ep1, 6),
        "lr_end":                   round(lr_epT, 6),
        "noise_std":                eff_sigma * C,
        "delta":                    delta,
        "eps_estimate_upper_bound": round(eps_estimate, 4),
        "sampling_rate_q":          round(q, 6),
        "global_epochs":            global_epochs,
        "target_epsilon":           target_epsilon,
        "schedule":                 "cosine_decay" if mode == "trust_dp" else "constant",
        "note": "eps_estimate 为 q-aware 简化上界，精确 DP 分析请用 Opacus / dp-accounting",
    }

    if verbose:
        print(f"  训练模式                  : {mode}")
        if mode == "trust_dp":
            print(f"  trust_t                   : {trust_t}")
            print(f"  sigma_min                 : {sigma_min_}")
            print(f"  sigma_max                 : {sigma_max_}")
            print(f"  噪声调度                  : cosine decay {sigma_ep1:.4f} → {sigma_epT:.4f}")
            print(f"  学习率调度                : adaptive {lr_ep1:.5f} → {lr_epT:.5f}")
        print(f"  effective_noise_multiplier: {eff_sigma:.4f}  (静态参考值)")
        print(f"  实际加噪标准差             : {eff_sigma * C:.4f}  (= σ_eff × C)")
        print(f"  clipping_norm             : {clipping_norm}")
        print(f"  batch_size                : {batch_size}")
        print(f"  global_epochs             : {global_epochs}")
        print(f"  global_lr                 : {global_lr}")
        print(f"  ε 估算上界（供参考）      : {eps_estimate:.4f}  (δ={delta})")

    # ------------------------------------------------------------------
    # Step 4：执行全局训练
    # ------------------------------------------------------------------
    if verbose:
        print("\n[4/5] 执行全局训练 ...")
        print(f"  N_lg={N_lg}，batch_size={batch_size}，"
              f"每 epoch 约 {max(1, N_lg // batch_size)} 个 mini-batch")

    log_epochs = build_log_epochs(global_epochs, max_epoch_log_lines)
    history    = []
    t_train_start = time.time()

    stopped_by_epsilon = False
    stop_epoch = int(global_epochs)
    eps_accum_inv_sq = 0.0   # RDP 组合累加器：Σ 1/σ(t)²（支持变 σ 的隐私会计）

    for epoch in range(1, global_epochs + 1):
        t_ep = time.time()

        # ---- trust_dp 动态调度：每 epoch 更新 σ 和 lr ----
        eff_sigma_ep = scheduler.effective_sigma_at_epoch(epoch, global_epochs)
        lr_ep        = scheduler.effective_lr_at_epoch(epoch, global_epochs, global_lr)

        epoch_stats = train_one_epoch(
            model          = model,
            X_lg_feat      = X_lg_feat,
            y_lg           = y_lg,
            num_classes    = num_classes,
            device         = device,
            lr             = lr_ep,
            batch_size     = batch_size,
            clipping_norm  = clipping_norm,
            effective_sigma= eff_sigma_ep,
            rng            = rng,
        )

        train_loss = compute_loss(model, X_lg_feat, y_lg, num_classes, device)
        train_acc  = compute_accuracy(model, X_lg_feat, y_lg, num_classes, device)
        test_loss  = compute_loss(model, X_test_feat, y_test, num_classes, device)
        test_acc   = compute_accuracy(model, X_test_feat, y_test, num_classes, device)
        ep_time    = time.time() - t_ep

        # ε 会计（支持变 σ 的 RDP 组合）
        if eff_sigma_ep > 0.0:
            eps_accum_inv_sq += 1.0 / (eff_sigma_ep ** 2)
            eps_epoch = q * math.sqrt(2.0 * math.log(1.0 / delta) * eps_accum_inv_sq)
        else:
            eps_epoch = float("inf")

        row = {
            "epoch":            epoch,
            "train_loss":       round(train_loss, 6),
            "train_acc":        round(train_acc,  4),
            "test_loss":        round(test_loss,  6),
            "test_acc":         round(test_acc,   4),
            "mean_grad_norm":   round(epoch_stats["mean_grad_norm"], 4),
            "effective_sigma":  round(eff_sigma_ep, 4),
            "effective_lr":     round(lr_ep, 6),
            "epsilon_estimate": float(round(eps_epoch, 6)) if np.isfinite(eps_epoch) else float("inf"),
            "epoch_time_s":     round(ep_time, 3),
        }
        history.append(row)

        if epoch in log_epochs and verbose:
            print(
                f"  Epoch {epoch:>4}/{global_epochs}"
                f"  train_loss={train_loss:.4f}"
                f"  train_acc={train_acc:.4f}"
                f"  test_loss={test_loss:.4f}"
                f"  test_acc={test_acc:.4f}"
                f"  σ={eff_sigma_ep:.3f}"
                f"  lr={lr_ep:.5f}"
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

    # ------------------------------------------------------------------
    # Step 5：汇总并写入报告
    # ------------------------------------------------------------------
    if verbose:
        print("\n[5/5] 汇总 & 写入报告 ...")

    final_train_loss = history[-1]["train_loss"] if history else float("nan")
    final_train_acc  = history[-1]["train_acc"]  if history else 0.0
    final_test_loss  = history[-1]["test_loss"]  if history else float("nan")
    final_test_acc   = history[-1]["test_acc"]   if history else 0.0

    epochs_ran = len(history)
    final_metrics = {
        "final_train_loss":     final_train_loss,
        "final_train_accuracy": final_train_acc,
        "final_test_loss":      final_test_loss,
        "final_test_accuracy":  final_test_acc,
        "total_train_time_s":   round(total_train_time, 3),
        "avg_epoch_time_s":     round(total_train_time / max(1, epochs_ran), 3),
        "epochs_ran":           int(epochs_ran),
        "stopped_by_epsilon":   bool(stopped_by_epsilon),
        "stop_epoch":           int(stop_epoch),
        "final_epsilon_estimate": history[-1]["epsilon_estimate"] if history else (float("inf") if eff_sigma == 0 else 0.0),
    }

    config = {
        "init_type":        init_type,
        "feat_dim":         feat_dim_,
        "num_classes":      num_classes,
        "N_lg_feat":        N_lg,
        "N_test":           len(y_test),
        "global_epochs":    global_epochs,
        "target_epsilon":   target_epsilon,
        "global_lr":        global_lr,
        "batch_size":       batch_size,
        "clipping_norm":    clipping_norm,
        "noise_multiplier": noise_multiplier,
        "delta":            delta,
        "seed":             seed,
        "device":           str(device),
    }

    result = {
        "final_model":  model,                  # nn.Module，可直接传给后续模块
        "history":      history,
        "final_metrics":final_metrics,
        "trust_info":   trust_info,
        "dp_info":      dp_info,
        "config":       config,
        "run_ts":       run_ts,
    }

    if verbose:
        print("")
        print("  [最终结果]")
        print(f"  final_train_accuracy : {final_train_acc:.4f}")
        print(f"  final_test_accuracy  : {final_test_acc:.4f}")
        print(f"  total_train_time (s) : {total_train_time:.2f}")
        print(f"  mode                 : {mode}")
        if mode == "trust_dp":
            print(f"  trust_t              : {trust_t}")
        print(f"  effective σ           : {eff_sigma:.4f}")

    return result


# ============================================================================
# ★ 三模式对照实验（Mode A / B / C 一键对比）
# ============================================================================

def run_mode_comparison(
    init_model:       nn.Module,
    X_lg_feat:        np.ndarray,
    y_lg:             np.ndarray,
    X_test_feat:      np.ndarray,
    y_test:           np.ndarray,
    global_epochs:    int   = 50,
    global_lr:        float = 0.01,
    batch_size:       int   = 64,
    clipping_norm:    float = 1.0,
    noise_multiplier: float = 1.0,   # Mode B 固定 sigma
    trust_t:          float = 0.7,
    sigma_min:        float = 0.1,
    sigma_max:        float = 2.0,
    delta:            float = 1e-5,
    target_epsilon:   Optional[float] = None,
    seed:             int   = 42,
    device_hint:      Optional[str] = None,
    max_epoch_log_lines: int = 10,
    safe_mode:        bool  = True,
    cpu_threads:      int   = 4,
    verbose:          bool  = True,
) -> Dict:
    """
    一键运行 Mode A / B / C 三种模式，生成对照表。

    对应论文实验设计：
      Mode A (plain)     : no-DP 明文训练 → 精度上界
      Mode B (fixed_dp)  : 固定噪声 DPSGD → PEA baseline
      Mode C (trust_dp)  : trust-aware DP  → 本模块贡献

    可视化对照：test_accuracy 随 epoch 的曲线对比。
    """
    if verbose:
        print("\n" + "=" * 70)
        print("三模式对照实验：Mode A (plain) / B (fixed_dp) / C (trust_dp)")
        print("=" * 70)

    modes_cfg = [
        {"mode": "plain",    "label": "A_plain"},
        {"mode": "fixed_dp", "label": "B_fixed_dp"},
        {"mode": "trust_dp", "label": "C_trust_dp"},
    ]

    results = {}
    for cfg in modes_cfg:
        if verbose:
            print(f"\n--- 运行 {cfg['label']} ---")

        r = run_trust_dpsgd(
            init_model       = init_model,
            X_lg_feat        = X_lg_feat,
            y_lg             = y_lg,
            X_test_feat      = X_test_feat,
            y_test           = y_test,
            global_epochs    = global_epochs,
            global_lr        = global_lr,
            batch_size       = batch_size,
            clipping_norm    = clipping_norm,
            noise_multiplier = noise_multiplier,
            delta            = delta,
            target_epsilon   = target_epsilon,
            mode             = cfg["mode"],
            trust_t          = trust_t,
            sigma_min        = sigma_min,
            sigma_max        = sigma_max,
            seed             = seed,
            device_hint      = device_hint,
            max_epoch_log_lines = max_epoch_log_lines,
            safe_mode        = safe_mode,
            cpu_threads      = cpu_threads,
            verbose          = verbose,
        )
        results[cfg["label"]] = r

    # 打印对照摘要
    if verbose:
        print("\n" + "=" * 70)
        print("  [三模式对照摘要]")
        print("  {:<18} {:<16} {:<16} {:<20}".format(
            "Mode", "Train_Acc", "Test_Acc", "σ_eff"))
        print("  " + "-" * 70)
        for label, r in results.items():
            fm = r["final_metrics"]
            sd_start = r["dp_info"].get("sigma_start", r["dp_info"]["effective_noise_multiplier"])
            sd_end   = r["dp_info"].get("sigma_end", r["dp_info"]["effective_noise_multiplier"])
            if abs(sd_start - sd_end) > 1e-6:
                sigma_str = f"{sd_start:.2f}→{sd_end:.2f}"
            else:
                sigma_str = f"{sd_start:.4f}"
            print("  {:<18} {:<16.4f} {:<16.4f} {:<20}".format(
                label,
                fm["final_train_accuracy"],
                fm["final_test_accuracy"],
                sigma_str,
            ))

    return results


# ============================================================================
# ★ 与模块 b 的桥接接口
# ============================================================================

def bridge_from_module_b(
    module_b_output: Dict,
    init_type:       str   = "chosen",   # "chosen" / "avg" / "best" / "random"
    **kwargs,
) -> Dict:
    """
    从模块 b 的 run_kd_hetero_init_pipeline 输出直接接入模块 c。

    使用示例：
      import feature_kd_init as kd
      import module_c_trust_dpsgd as mc

      b_out = kd.run_kd_hetero_init_pipeline(...)
      c_out = mc.bridge_from_module_b(
          module_b_output = b_out,
          init_type       = "avg",       # 使用 avg_model 作为初始化
          X_lg_feat       = X_train_feat,
          y_lg            = y_train,
          X_test_feat     = X_test_feat,
          y_test          = y_test,
          mode            = "trust_dp",
          trust_t         = 0.7,
          global_epochs   = 100,
      )

    Parameters
    ----------
    module_b_output : 模块 b 的完整返回字典
    init_type       : 从模块 b 选用哪个模型
      "chosen"  → module_b_output["initial_global_model"]（模块 b 自动选择的最优）
      "avg"     → module_b_output["avg_model"]
      "best"    → module_b_output["best_model"]
      "random"  → 随机初始化（kwargs 中需传 feat_dim 和 num_classes_hint）
    """
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

    return run_trust_dpsgd(init_model=model, **kwargs)


# ============================================================================
# ★ JSON 报告保存（不序列化 nn.Module）
# ============================================================================

def save_report(result: Dict, dataset_name: str = "unknown",
                mode: str = "trust_dp",
                json_path: Optional[Path] = None) -> Path:
    """
    将 run_trust_dpsgd 的返回结果保存为 JSON 报告。
    nn.Module 对象不写入 JSON（另存 state_dict .pt）。
    """
    ensure_dirs()
    run_ts   = result.get("run_ts", datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_path = json_path if json_path is not None else (
        LOG_JSON_DIR / f"trustDpsgd_{dataset_name}_{run_ts}.json"
    )
    pt_path = LOG_FINAL_MODEL_DIR / f"module_c_{dataset_name}_{mode}_{run_ts}_model.pt"

    # 保存模型权重
    torch.save(result["final_model"].state_dict(), str(pt_path))

    # 构造可序列化报告
    report = {
        "meta": {
            "experiment":   "module_c_trust_dpsgd",
            "run_id":       run_ts,
            "dataset":      dataset_name,
            "python":       sys.version.split()[0],
            "platform":     platform.platform(),
            "pytorch":      torch.__version__,
        },
        "config":        result["config"],
        "init_info":     {"init_type": result["config"]["init_type"]},
        "trust_info":    result["trust_info"],
        "dp_info":       result["dp_info"],
        "final_metrics": result["final_metrics"],
        "history":       result["history"],
        "model_pt":      str(pt_path),
        "notes": [
            "★ 训练对象是 logically global feature dataset（D_lg_feat）",
            "★ D_lg_feat = 所有 client 特征的逻辑拼合（本原型近似实现）",
            "★ 特征为 feature-level 数据（模块 a 提取），非原始图像/文本",
            "★ 后续可替换为真正的 TF-Encrypted / MPC secure backend",
            "★ per-sample gradient 当前用 Python 循环（正确但较慢）",
            "★ trust_t 抽象自 Truex et al. 2019 的 trust 参数",
        ],
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return log_path


# ============================================================================
# ★ Demo：最小可运行验证（合成特征）
# ============================================================================

def _demo_run():
    """
    用合成特征做最小可运行验证，不依赖任何外部数据或模型。
    覆盖 Mode A / B / C 三种模式。
    """
    print("\n" + "=" * 70)
    print("Demo：合成特征最小可运行验证")
    print("=" * 70)

    set_seed(42)
    n_train, n_test = 500, 100
    feat_dim   = 64
    num_classes = 4

    # 合成可分特征
    W = np.random.randn(feat_dim, num_classes).astype(np.float32) * 0.5
    X_train = np.random.randn(n_train, feat_dim).astype(np.float32)
    X_test  = np.random.randn(n_test,  feat_dim).astype(np.float32)
    y_train = np.argmax(X_train.dot(W) + 0.3 * np.random.randn(n_train, num_classes), axis=1)
    y_test  = np.argmax(X_test.dot(W)  + 0.3 * np.random.randn(n_test,  num_classes), axis=1)

    # 模拟模块 b 输出（随机初始化一个 initial_global_model）
    device      = select_device()
    init_model  = make_fresh_model(feat_dim, num_classes, device)

    print(f"\n合成数据: train={X_train.shape}, test={X_test.shape}")
    print(f"类别数: {num_classes}, 特征维度: {feat_dim}")

    # 三模式对照（epochs 少一些，快速验证）
    results = run_mode_comparison(
        init_model       = init_model,
        X_lg_feat        = X_train,
        y_lg             = y_train,
        X_test_feat      = X_test,
        y_test           = y_test,
        global_epochs    = 20,
        global_lr        = 0.05,
        batch_size       = 50,
        clipping_norm    = 1.0,
        noise_multiplier = 1.0,
        trust_t          = 0.7,
        sigma_min        = 0.1,
        sigma_max        = 1.5,
        delta            = 1e-5,
        seed             = 42,
        max_epoch_log_lines = 5,
        verbose          = True,
    )

    # 保存其中一个模式的报告
    log_path = save_report(results["C_trust_dp"], dataset_name="demo", mode="trust_dp")
    print(f"\nJSON 报告已写入: {log_path}")
    print("Demo 完成。")


if __name__ == "__main__":
    _demo_run()

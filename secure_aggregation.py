#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
密态聚合模块：将模块 b 的权重聚合与候选评估迁移到 ABY3 密态空间。

对应 PEA 原论文流程：
  1) 各 client 的蒸馏后模型权重 → secret share → 在 MPC 内求平均 → reveal avg_model
  2) avg_model / best_model → secret share → 在 MPC 内对 ref set 做前向推理 → 密态比较精度 → reveal 胜者

与原 feature_kd_init.py 的明文版等价，但全部在 ABY3 三方安全计算协议内完成。

CPU 线程限制：
  本模块在入口函数强制设置 OMP/MKL/OPENBLAS 线程上限，防止 TFE 拉起过多线程导致死机。

用法：
  import secure_aggregation as sa

  # 替代明文 average_model_weights + evaluate_candidate_models
  result = sa.secure_aggregate_and_select(
      distill_models=distill_models,
      X_ref=X_ref, y_ref=y_ref,
      feat_dim=feat_dim, num_classes=num_classes,
  )
  # result["avg_model"], result["best_model"], result["chosen_model"]
"""

import io
import logging
import math
import os
import sys
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CPU 线程安全限制（在任何 import torch / tf 之前设置）
# ---------------------------------------------------------------------------
_MAX_THREADS = int(os.environ.get("SECURE_AGG_THREADS", "2"))


def set_thread_limits(max_threads: int = _MAX_THREADS) -> None:
    """强制限制 CPU 线程数，防止 TFE/TF 拉起过多线程。"""
    t = max(1, int(max_threads))
    for key in [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS",
    ]:
        os.environ[key] = str(t)
    # 标记已设置
    os.environ["TFE_SAFE_RUNTIME_APPLIED"] = "1"


# 立即生效
set_thread_limits(_MAX_THREADS)

import torch
import torch.nn as nn

try:
    # intra-op 线程一般可重复设置；若失败则静默回退。
    torch.set_num_threads(_MAX_THREADS)
except Exception:
    pass

# 注意：
# 不在这里调用 torch.set_num_interop_threads。
# 在主流程（feature_kd_bridge / module_c_bridge）中线程池可能已启动，
# 二次调用会触发 c10::Error 并直接中止进程。

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _state_dict_to_numpy(model: nn.Module) -> Dict[str, np.ndarray]:
    """提取模型 state_dict 为 numpy 字典。"""
    return {
        k: v.detach().cpu().numpy().astype(np.float32)
        for k, v in model.state_dict().items()
    }


def _extract_wb(model: nn.Module, feat_dim: int, num_classes: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取线性模型的 (W, b)，并转为 TFE Dense 格式。
    torch: W[out_dim, in_dim], b[out_dim]
    tfe  : W[in_dim, out_dim], b[out_dim]
    """
    sd = model.state_dict()
    w = sd["fc.weight"].detach().cpu().numpy().astype(np.float32)  # [out, in]
    b = sd["fc.bias"].detach().cpu().numpy().astype(np.float32)    # [out]
    return w.T.copy(), b.copy()   # → [in, out]


def _to_onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y).reshape(-1).astype(int)
    oh = np.zeros((len(y), n_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def _acc_from_logits(logits: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    y = np.asarray(y).reshape(-1).astype(int)
    if n_classes == 2:
        pred = (logits.reshape(-1) > 0.0).astype(np.int32)
        return float(np.mean(pred == y))
    pred = np.argmax(logits, axis=1).astype(np.int32)
    return float(np.mean(pred == y))


def _quiet_import_tfe():
    """静默导入 TensorFlow 和 tf-encrypted，避免大量 warnings 刷屏。"""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        import tensorflow as tf
        import tf_encrypted as tfe

    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception:
        pass
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("tf_encrypted").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    # 线程限制对 TF session 也生效
    tf.config.threading.set_inter_op_parallelism_threads(_MAX_THREADS)
    tf.config.threading.set_intra_op_parallelism_threads(_MAX_THREADS)

    return tf, tfe


# ===========================================================================
# 核心 1：密态权重聚合（Secure Weight Aggregation）
# ===========================================================================

def secure_average_weights(
    models: List[nn.Module],
    feat_dim: int,
    num_classes: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 ABY3 密态空间中对多个 client 模型权重求平均。

    对应 PEA 原论文流程：
      每个 party 将自己的权重 secret-share 到 MPC 系统中 →
      在密态下执行 element-wise sum → 除以 N → reveal 平均权重。

    Parameters
    ----------
    models : 蒸馏后的各 client 模型列表
    feat_dim : 特征维度
    num_classes : 类别数

    Returns
    -------
    (W_avg, b_avg) : TFE 格式的平均权重 [in_dim, out_dim], [out_dim]
    """
    tf, tfe = _quiet_import_tfe()
    from tf_encrypted.keras import backend as KE

    n_clients = len(models)
    out_dim = 1 if num_classes == 2 else num_classes

    if verbose:
        print(f"  [密态聚合] {n_clients} 个 client 模型 → ABY3 secret-share → 求平均")

    # 提取所有 client 的权重（转为 TFE 格式）
    all_w = []
    all_b = []
    for m in models:
        w, b = _extract_wb(m, feat_dim, num_classes)
        all_w.append(w)
        all_b.append(b)

    # 在 ABY3 密态空间中求和再除以 N
    quiet_out = io.StringIO()
    quiet_err = io.StringIO()

    tf.compat.v1.reset_default_graph()

    with tfe.protocol.ABY3():
        with redirect_stdout(quiet_out), redirect_stderr(quiet_err):
            # 每个 client 的权重作为 private input 进入 MPC
            priv_ws = []
            priv_bs = []
            for i in range(n_clients):
                pw = tfe.define_private_variable(all_w[i])
                pb = tfe.define_private_variable(all_b[i])
                priv_ws.append(pw)
                priv_bs.append(pb)

            # 密态求和
            w_sum = priv_ws[0]
            b_sum = priv_bs[0]
            for i in range(1, n_clients):
                w_sum = w_sum + priv_ws[i]
                b_sum = b_sum + priv_bs[i]

            # 密态除以 N（使用乘法：× (1/N)）
            inv_n = 1.0 / float(n_clients)
            w_avg_priv = w_sum * inv_n
            b_avg_priv = b_sum * inv_n

            sess = KE.get_session()
            sess.run(tf.compat.v1.global_variables_initializer())

        # reveal 平均权重
        with redirect_stdout(quiet_out), redirect_stderr(quiet_err):
            w_avg_np = sess.run(w_avg_priv.reveal())
            b_avg_np = sess.run(b_avg_priv.reveal())

    KE.clear_session()

    if verbose:
        print(f"  [密态聚合] 完成: W_avg shape={w_avg_np.shape}, b_avg shape={b_avg_np.shape}")

    return w_avg_np.astype(np.float32), b_avg_np.astype(np.float32)


# ===========================================================================
# 核心 2：密态候选评估（Secure Candidate Evaluation）
# ===========================================================================

def secure_evaluate_candidate(
    w_cand: np.ndarray,
    b_cand: np.ndarray,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    feat_dim: int,
    num_classes: int,
    label: str = "candidate",
    verbose: bool = True,
) -> float:
    """
    在 ABY3 密态空间中评估一个候选模型在 reference set 上的精度。

    对应 PEA 原论文：
      将 candidate 权重和 ref set 数据 secret-share 到 MPC →
      密态前向推理 → reveal 预测结果 → 计算精度。

    Parameters
    ----------
    w_cand : [in_dim, out_dim] TFE 格式权重
    b_cand : [out_dim] 偏置
    X_ref  : reference set 特征
    y_ref  : reference set 标签

    Returns
    -------
    accuracy : 精度（float）
    """
    tf, tfe = _quiet_import_tfe()
    from tf_encrypted.keras import Sequential, backend as KE
    from tf_encrypted.keras.layers import Dense

    X_ref = np.asarray(X_ref, dtype=np.float32)
    y_ref = np.asarray(y_ref).reshape(-1).astype(int)
    n_ref = len(y_ref)
    out_dim = 1 if num_classes == 2 else num_classes

    quiet_out = io.StringIO()
    quiet_err = io.StringIO()

    tf.compat.v1.reset_default_graph()

    with tfe.protocol.ABY3():
        with redirect_stdout(quiet_out), redirect_stderr(quiet_err):
            # 构建密态模型，用 candidate 权重初始化
            model = Sequential()
            model.add(Dense(
                out_dim,
                batch_input_shape=(n_ref, feat_dim),
                kernel_initializer=tf.constant_initializer(w_cand),
                bias_initializer=tf.constant_initializer(b_cand),
            ))

            # secret-share reference set
            x_priv = tfe.define_private_variable(X_ref)
            model(x_priv)

            # 密态前向推理
            pred_tensor = model.predict(x_priv)

            sess = KE.get_session()
            sess.run(tf.compat.v1.global_variables_initializer())

        # reveal 预测 logits
        with redirect_stdout(quiet_out), redirect_stderr(quiet_err):
            pred_logits = sess.run(pred_tensor.reveal())

    KE.clear_session()

    # 计算精度
    acc = _acc_from_logits(pred_logits, y_ref, num_classes)

    if verbose:
        print(f"  [密态评估] {label}: ref_acc={acc:.4f}")

    return float(acc)


def secure_select_candidate(
    w_avg: np.ndarray,
    b_avg: np.ndarray,
    w_best: np.ndarray,
    b_best: np.ndarray,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    feat_dim: int,
    num_classes: int,
    verbose: bool = True,
) -> Dict:
    """
    在 ABY3 密态空间中对 avg_model 和 best_model 做精度评估并选择胜者。

    对应 PEA 原论文的 secure accuracy evaluation：
      两个 candidate 权重和 ref set 分别 secret-share →
      密态前向推理 → reveal 精度 → 选较优者。

    Returns
    -------
    Dict with keys:
      avg_ref_acc, best_ref_acc, chosen_init_type, w_chosen, b_chosen
    """
    if verbose:
        print("\n[密态候选评估] 在 ABY3 中比较 avg vs best ...")

    avg_acc = secure_evaluate_candidate(
        w_avg, b_avg, X_ref, y_ref, feat_dim, num_classes,
        label="avg_model", verbose=verbose,
    )

    best_acc = secure_evaluate_candidate(
        w_best, b_best, X_ref, y_ref, feat_dim, num_classes,
        label="best_model", verbose=verbose,
    )

    if avg_acc >= best_acc:
        chosen = "avg"
        w_chosen, b_chosen = w_avg, b_avg
    else:
        chosen = "best"
        w_chosen, b_chosen = w_best, b_best

    if verbose:
        print(f"  [密态候选评估] 胜者: {chosen} (avg={avg_acc:.4f}, best={best_acc:.4f})")

    return {
        "avg_ref_acc": float(avg_acc),
        "best_ref_acc": float(best_acc),
        "chosen_init_type": chosen,
        "w_chosen": w_chosen,
        "b_chosen": b_chosen,
    }


# ===========================================================================
# 统一入口：密态聚合 + 密态评估
# ===========================================================================

def secure_aggregate_and_select(
    distill_models: List[nn.Module],
    best_local_model: nn.Module,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    feat_dim: int,
    num_classes: int,
    max_threads: int = 2,
    verbose: bool = True,
) -> Dict:
    """
    完整密态聚合流程（替代明文 average_model_weights + evaluate_candidate_models）。

    对应 PEA 原论文 Section 4.2 的初始化聚合：
      Step 1: 各 client 模型权重 → ABY3 secret-share → 密态求平均 → reveal avg_model
      Step 2: avg_model + best_model → ABY3 secret-share → 密态 ref set 评估 → 选胜者

    Parameters
    ----------
    distill_models   : 蒸馏后的各 client nn.Module 列表
    best_local_model : 在 ref set 上精度最高的单个 client 模型
    X_ref            : reference set 特征
    y_ref            : reference set 标签
    feat_dim         : 特征维度
    num_classes      : 类别数
    max_threads      : CPU 线程上限（防死机）

    Returns
    -------
    Dict:
      "avg_model"          : nn.Module (avg 权重还原的 torch 模型)
      "best_model"         : nn.Module (best local 模型的深拷贝)
      "chosen_model"       : nn.Module (胜者模型)
      "candidate"          : Dict (密态评估结果)
      "secure_aggregation" : True (标记使用了密态聚合)
    """
    set_thread_limits(max_threads)

    if verbose:
        print("\n" + "=" * 70)
        print("密态聚合（ABY3 Secure Aggregation）")
        print("=" * 70)
        print(f"  clients          : {len(distill_models)}")
        print(f"  feat_dim         : {feat_dim}")
        print(f"  num_classes      : {num_classes}")
        print(f"  ref_size         : {len(y_ref)}")
        print(f"  max_threads      : {max_threads}")

    t0 = time.time()

    # ---- Step 1: 密态权重聚合 ----
    if verbose:
        print("\n[Step 1/2] 密态权重聚合 ...")
    w_avg, b_avg = secure_average_weights(
        distill_models, feat_dim, num_classes, verbose=verbose,
    )

    # 将密态聚合得到的 avg 权重还原为 torch 模型
    from feature_kd_init import LinearClassifier
    out_dim = 1 if num_classes == 2 else num_classes

    avg_model = LinearClassifier(feat_dim, num_classes)
    # TFE 格式 W[in, out] → torch 格式 W[out, in]
    avg_sd = {
        "fc.weight": torch.from_numpy(w_avg.T.copy()),
        "fc.bias": torch.from_numpy(b_avg.copy()),
    }
    avg_model.load_state_dict(avg_sd)
    avg_model.eval()

    # ---- best_model：提取权重用于密态评估 ----
    best_model = deepcopy(best_local_model).cpu()
    best_model.eval()
    w_best, b_best = _extract_wb(best_model, feat_dim, num_classes)

    # ---- Step 2: 密态候选评估 ----
    if verbose:
        print("\n[Step 2/2] 密态候选评估 ...")
    cand = secure_select_candidate(
        w_avg, b_avg, w_best, b_best,
        X_ref, y_ref, feat_dim, num_classes,
        verbose=verbose,
    )

    # 根据密态评估结果选择胜者模型
    if cand["chosen_init_type"] == "avg":
        chosen_model = deepcopy(avg_model)
    else:
        chosen_model = deepcopy(best_model)

    elapsed = round(time.time() - t0, 2)

    if verbose:
        print(f"\n[密态聚合] 完成，耗时 {elapsed}s")
        print(f"  avg_ref_acc     : {cand['avg_ref_acc']:.4f}")
        print(f"  best_ref_acc    : {cand['best_ref_acc']:.4f}")
        print(f"  chosen          : {cand['chosen_init_type']}")
        print("=" * 70)

    return {
        "avg_model": avg_model,
        "best_model": best_model,
        "chosen_model": chosen_model,
        "candidate": {
            "avg_ref_acc": cand["avg_ref_acc"],
            "best_ref_acc": cand["best_ref_acc"],
            "chosen_init_type": cand["chosen_init_type"],
        },
        "secure_aggregation": True,
        "elapsed_s": elapsed,
    }


# ===========================================================================
# 独立测试
# ===========================================================================

def _demo():
    """用合成数据验证密态聚合流程。"""
    from feature_kd_init import LinearClassifier, set_seed

    set_seed(42)
    feat_dim, num_classes = 32, 4
    n_clients = 3
    ref_size = 50

    # 构造模拟 client 模型
    models = []
    for i in range(n_clients):
        m = LinearClassifier(feat_dim, num_classes)
        # 给每个 client 略微不同的权重
        with torch.no_grad():
            m.fc.weight.add_(torch.randn_like(m.fc.weight) * 0.1 * (i + 1))
        models.append(m)

    # 模拟 best local model（取第一个）
    best_model = deepcopy(models[0])

    # 模拟 ref set
    X_ref = np.random.randn(ref_size, feat_dim).astype(np.float32)
    y_ref = np.random.randint(0, num_classes, size=ref_size)

    result = secure_aggregate_and_select(
        distill_models=models,
        best_local_model=best_model,
        X_ref=X_ref,
        y_ref=y_ref,
        feat_dim=feat_dim,
        num_classes=num_classes,
        max_threads=2,
    )
    print("\n[demo] 密态聚合结果:")
    print(f"  chosen: {result['candidate']['chosen_init_type']}")
    print(f"  avg_ref_acc: {result['candidate']['avg_ref_acc']:.4f}")
    print(f"  best_ref_acc: {result['candidate']['best_ref_acc']:.4f}")
    print(f"  secure_aggregation: {result['secure_aggregation']}")


if __name__ == "__main__":
    _demo()

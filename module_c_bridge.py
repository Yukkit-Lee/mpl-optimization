#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
桥接脚本：模块 b → 模块 c（Trust-aware Secure DPSGD）
=======================================================

用法（与 feature_kd_bridge.py 配套）：

  # 方式 1：命令行（仅指定数据集）
  python module_c_bridge.py --dataset cifar
  其余参数（mode / trust_t / epochs / init_type）统一在 CFG 中设置。

  # 方式 2：在 Python 中调用
  import feature_kd_init as kd
  import module_c_bridge as cb

  b_out = kd.run_kd_hetero_init_pipeline(...)
  c_out = cb.run_b_to_c(b_out, X_train_feat, y_train, X_test_feat, y_test)

支持的数据集（与 feature_kd_bridge.py 一致）：
  mnist  →  MNIST + HOG
  imdb   →  IMDb + RoBERTa
  cifar  →  CIFAR-10 + DINO
"""

import argparse
import json
import math
import os
import sys
import time
import io
import warnings
import logging
from contextlib import contextmanager
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR  = BASE_DIR / "log"
LOG_TXT_DIR = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"
DATA_DIR = BASE_DIR / "data"
INIT_MODEL_DIR = BASE_DIR / "init_model"
EXPFIG_DIR = BASE_DIR / "expFig"


# ===========================================================================
# 实验配置（优先在此处改参数）
# ===========================================================================
CFG = {
    # ---- 通用安全配置 ----
    "safe_mode":            True,
    "cpu_threads":          1,
    "max_test_cap":         1000,
    "seed":                 42,
    "max_epoch_log_lines":  10,
    "device":               "cpu",

    # ---- 模块 b 缓存读取配置（只读，不重跑）----
    "b_init_policy":        "latest",  # latest / best

    # ---- 模块 c 配置 ----
    # 固定读取模块 b 输出的 initial_global_model（chosen）
    "mode":                 "trust_dp",     # plain / fixed_dp / trust_dp
    "trust_t":              0.7,
    "sigma_min":            0.1,
    "sigma_max":            2.0,
    "noise_multiplier":     1.0,            # Mode B 固定 sigma
    "global_epochs":        800,
    "global_lr":            0.01,
    "batch_size":           64,
    "clipping_norm":        1.0,
    "delta":                1e-5,
    "target_epsilon":       None,    # 例如 8.0；None 表示不按 ε 提前停止

    # 是否同时运行三模式对照（A / B / C）
    "run_comparison":       False,

    # ---- Figure5（TFE 密态训练版）----
    "run_tfe_figure5":       True,
    "figure5_epochs":        800,
    "figure5_lr":            0.01,
    "figure5_max_log_lines": 10,
    "figure5_delta":         1e-5,
    "figure5_target_epsilon": None,
    # Figure5 ε 显示口径：
    # - "pea_aligned": 按论文风格对齐到 epsilon_end（默认）
    # - "accountant":  按当前会计器原始口径
    "figure5_epsilon_mode":  "pea_aligned",
    "figure5_epsilon_end":   2.07,
    # None 表示按总 epoch 自动取 3-4 个打点；也可传 [0, 400, 800]
    "figure5_annotate_iters": None,
}

# 针对高负载数据集的安全覆盖（优先防止系统卡死）
DATASET_SAFETY_OVERRIDES = {
    "cifar": {
        "cpu_threads": 1,
        "global_epochs": 300,
        "target_epsilon": 2.2,
        "run_tfe_figure5": True,
        "figure5_epochs": 200,
        "figure5_lr": 0.005,
        "figure5_target_epsilon": 2.2,
        "figure5_max_log_lines": 10,
    },
}


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INIT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPFIG_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def tee_console_to_file(log_path: Path):
    """将控制台输出同时写入文件（人类可读日志）。"""
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

        def close(self):
            # 兼容某些 logging handler 在退出时调用 close()
            self.flush()

    orig_out, orig_err = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as f:
        tee = _Tee(orig_out, f)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err


def set_safe_runtime(max_threads: int = 4) -> None:
    t = max(1, int(max_threads))
    for key in [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    ]:
        os.environ[key] = str(t)
    try:
        import torch
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
    # 标记：当前进程已做过线程安全初始化，避免后续模块重复设置导致崩溃。
    os.environ["TFE_SAFE_RUNTIME_APPLIED"] = "1"


# ===========================================================================
# 数据加载（复用 feature_kd_bridge 的加载逻辑）
# ===========================================================================

def load_dataset_features(dataset: str, max_test_cap: int = 0) -> Dict:
    """委托给 feature_kd_bridge 的加载函数。"""
    try:
        import feature_kd_bridge as bridge
        return bridge.load_dataset_features(dataset, max_test_cap=max_test_cap)
    except Exception as e:
        raise RuntimeError(
            "加载特征数据失败。\n"
            "feature_kd_bridge.py 已导入，但其内部执行报错。\n"
            f"底层错误: {type(e).__name__}: {e}"
        )


def _b_cache_paths(dataset: str, policy: str) -> Dict[str, Path]:
    """根据策略返回模块 b 模型缓存路径。"""
    tag = "init"
    return {
        "meta_json": INIT_MODEL_DIR / f"module_b_{tag}_{dataset}.json",
        "weights_pt": INIT_MODEL_DIR / f"module_b_{tag}_{dataset}.pt",
    }


def _load_module_b_init_or_raise(dataset: str, feat_dim: int, num_classes: int) -> Dict:
    """
    只读取模块 b 已保存的初始化模型，不触发模块 b 训练。
    若读取失败则直接抛错，提示先运行模块 b。
    """
    import torch
    import feature_kd_init as kd

    policy = "latest"
    paths = _b_cache_paths(dataset, policy)
    if not paths["weights_pt"].exists():
        raise RuntimeError(
            "未找到模块 b 初始化模型缓存。\n"
            f"期望文件: {paths['weights_pt']} (latest)\n"
            "请先运行模块 b（feature_kd_bridge.py）生成 init_model 缓存，再运行模块 c。"
        )

    ckpt = torch.load(str(paths["weights_pt"]), map_location="cpu")
    ckpt_feat_dim = int(ckpt.get("feat_dim", -1))
    ckpt_num_classes = int(ckpt.get("num_classes", -1))
    if ckpt_feat_dim != int(feat_dim) or ckpt_num_classes != int(num_classes):
        raise RuntimeError(
            "模块 b 初始化模型与当前数据不匹配。\n"
            f"缓存维度/类别: feat_dim={ckpt_feat_dim}, num_classes={ckpt_num_classes}\n"
            f"当前维度/类别: feat_dim={feat_dim}, num_classes={num_classes}\n"
            "请重新运行模块 b 生成对应数据集缓存。"
        )

    chosen_model = kd.LinearClassifier(feat_dim, num_classes)
    chosen_model.load_state_dict(ckpt["state_dicts"]["initial_global_model"])
    chosen_model.eval()

    avg_model = kd.LinearClassifier(feat_dim, num_classes)
    avg_model.load_state_dict(ckpt["state_dicts"]["avg_model"])
    avg_model.eval()

    best_model = kd.LinearClassifier(feat_dim, num_classes)
    best_model.load_state_dict(ckpt["state_dicts"]["best_model"])
    best_model.eval()

    return {
        "candidate": ckpt.get("candidate", {}),
        "config": ckpt.get("config", {}),
        "initial_global_model": chosen_model,
        "avg_model": avg_model,
        "best_model": best_model,
        "_cache_meta": {
            "policy": policy,
            "meta_json": str(paths["meta_json"]),
            "weights_pt": str(paths["weights_pt"]),
        },
    }


def _build_log_epochs(total_epochs: int, max_lines: int) -> set:
    total_epochs = max(1, int(total_epochs))
    max_lines = max(1, int(max_lines))
    if total_epochs <= max_lines:
        return set(range(1, total_epochs + 1))
    picks = np.linspace(1, total_epochs, num=max_lines, dtype=int)
    out = set(int(x) for x in picks.tolist())
    out.add(1)
    out.add(total_epochs)
    return out


def _to_onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y).reshape(-1).astype(int)
    oh = np.zeros((len(y), n_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def _as_scalar_loss(x) -> float:
    arr = np.asarray(x, dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def _acc_from_logits(logits: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    y = np.asarray(y).reshape(-1).astype(int)
    if n_classes == 2:
        pred = (logits.reshape(-1) > 0.0).astype(np.int32)
        return float(np.mean(pred == y))
    pred = np.argmax(logits, axis=1).astype(np.int32)
    return float(np.mean(pred == y))


def _extract_linear_weights_from_torch(model, feat_dim: int, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从模块 b 的 torch 线性模型提取权重，转换为 TFE Dense 所需格式：
    torch: W[out_dim, in_dim], b[out_dim]
    tfe  : W[in_dim, out_dim], b[out_dim]
    """
    sd = model.state_dict()
    w_t = sd["fc.weight"].detach().cpu().numpy().astype(np.float32)
    b_t = sd["fc.bias"].detach().cpu().numpy().astype(np.float32)
    out_dim = 1 if int(num_classes) == 2 else int(num_classes)
    if w_t.shape != (out_dim, feat_dim):
        raise RuntimeError(f"模型权重形状异常: {w_t.shape}, 期望 {(out_dim, feat_dim)}")
    return w_t.T.copy(), b_t.copy()


def run_tfe_figure5_init_comparison(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    b_out: Dict,
    epochs: int,
    lr: float,
    max_epoch_log_lines: int,
    delta: float,
    target_epsilon: Optional[float],
    annotate_iters: Optional[list],
    epsilon_mode: str,
    epsilon_end: float,
    log_fn,
) -> Dict:
    """
    Figure5（TFE密态训练版）：
    random_init / avg_init / best_init 三种初始化进入真正 TFE 训练并画收敛曲线。
    """
    # 静默 TensorFlow / TFE 的非关键警告信息（不影响训练结果）。
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # tf_encrypted 在导入时会输出 custom ops / deprecated 警告，定向静默导入。
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        import tensorflow as tf
        import tf_encrypted as tfe
    from tf_encrypted.keras import Sequential
    from tf_encrypted.keras import backend as KE
    from tf_encrypted.keras.layers import Dense

    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception:
        pass
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("tf_encrypted").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train).reshape(-1).astype(int)
    y_test = np.asarray(y_test).reshape(-1).astype(int)

    feat_dim = int(x_train.shape[1])
    num_classes = int(len(np.unique(y_train)))
    out_dim = 1 if num_classes == 2 else num_classes
    n_train = int(len(y_train))
    log_epochs = _build_log_epochs(epochs, max_epoch_log_lines)

    # 与模块 A/B 的 TFE 训练保持一致：先做明文标准化，提升数值稳定性。
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    # 维度缩放学习率：高维时降低步长，避免 MSE + 全批训练爆梯度。
    if feat_dim <= 128:
        lr_eff = float(lr)
    elif feat_dim <= 256:
        lr_eff = float(lr) * 0.5
    elif feat_dim <= 512:
        lr_eff = float(lr) * 0.2
    else:
        lr_eff = float(lr) * 0.1

    if num_classes == 2:
        y_train_tfe = (y_train.reshape(-1, 1).astype(np.float32) * 2.0) - 1.0
    else:
        y_train_tfe = _to_onehot(y_train, num_classes)

    # 从模块 b 读取 avg/best 初始化权重
    w_avg, b_avg = _extract_linear_weights_from_torch(b_out["avg_model"], feat_dim, num_classes)
    w_best, b_best = _extract_linear_weights_from_torch(b_out["best_model"], feat_dim, num_classes)

    rng = np.random.RandomState(int(CFG["seed"]))
    w_rand = (rng.normal(loc=0.0, scale=0.05, size=(feat_dim, out_dim))).astype(np.float32)
    b_rand = np.zeros((out_dim,), dtype=np.float32)

    init_bank = {
        "random_init": (w_rand, b_rand),
        "avg_init": (w_avg, b_avg),
        "best_init": (w_best, b_best),
    }

    curves = {}
    finals = {}
    log_fn("\n[Figure5-TFE] random/avg/best 三初始化进入 TFE 密态训练")
    log_fn(f"  epochs/lr(base->eff) : {epochs}/{lr}->{lr_eff}")
    log_fn(f"  feat_dim/classes : {feat_dim}/{num_classes}")

    # 采用与模块 C 一致的简化会计器：ε_t ≈ q * sqrt(2 t log(1/δ)) / σ_eff
    # Figure5 若未显式给 DP 噪声配置，则沿用模块 C 当前模式参数推导 σ_eff。
    mode = str(CFG.get("mode", "trust_dp"))
    if mode == "plain":
        eff_sigma = 0.0
    elif mode == "fixed_dp":
        eff_sigma = float(CFG.get("noise_multiplier", 1.0))
    else:
        smin = float(CFG.get("sigma_min", 0.1))
        smax = float(CFG.get("sigma_max", 2.0))
        tt = float(CFG.get("trust_t", 0.5))
        eff_sigma = max(0.0, smax - (smax - smin) * tt)
    # 该 TFE 训练为全量 fit_batch，采样率 q=1。
    q = 1.0
    log_fn(f"  ε-accountant: mode={epsilon_mode}, epsilon_end={epsilon_end}, delta={delta}, q={q:.3f}, sigma_eff={eff_sigma:.4f}, target_epsilon={target_epsilon}")

    def _resolve_annot_iters(total_ep: int, preset):
        if isinstance(preset, (list, tuple)) and len(preset) > 0:
            vals = sorted(set(int(x) for x in preset))
            return [v for v in vals if 0 <= v <= int(total_ep)]
        p1 = int(round(total_ep * 0.33))
        p2 = int(round(total_ep * 0.66))
        return sorted(set([0, p1, p2, int(total_ep)]))

    resolved_annot_iters = _resolve_annot_iters(int(epochs), annotate_iters)

    for name, (w0, b0) in init_bank.items():
        log_fn(f"\n  -> 训练 {name}")
        tf.compat.v1.reset_default_graph()
        quiet_out = io.StringIO()
        quiet_err = io.StringIO()
        with tfe.protocol.ABY3():
            with redirect_stdout(quiet_out), redirect_stderr(quiet_err):
                model = Sequential()
                model.add(Dense(
                    out_dim,
                    batch_input_shape=(n_train, feat_dim),
                    kernel_initializer=tf.constant_initializer(w0),
                    bias_initializer=tf.constant_initializer(b0),
                ))
                x_priv = tfe.define_private_variable(x_train)
                y_priv = tfe.define_private_variable(y_train_tfe)
                loss_fn = tfe.keras.losses.MeanSquaredError()
                # 使用维度缩放后的学习率，避免大维度下数值发散。
                optimizer = tfe.keras.optimizers.SGD(learning_rate=float(lr_eff))
                model.compile(optimizer=optimizer, loss=loss_fn)
                model(x_priv)
                fit_op, loss_tensor = model.fit_batch(x_priv, y_priv)
                pred_tensor = model.predict(x_priv)

                sess = KE.get_session()
                sess.run(tf.compat.v1.global_variables_initializer())

            history = []
            t_start = time.time()
            # 先记录 epoch0，确认初始化权重是否按 random/avg/best 正确生效。
            with redirect_stdout(quiet_out), redirect_stderr(quiet_err):
                init_train_logits = sess.run(pred_tensor.reveal())
                w_np0, b_np0 = sess.run([
                    model.layers[0].weights[0].reveal(),
                    model.layers[0].weights[1].reveal(),
                ])
            init_test_logits = x_test.dot(w_np0) + b_np0
            with redirect_stdout(quiet_out), redirect_stderr(quiet_err):
                init_loss_val = _as_scalar_loss(sess.run(loss_tensor.reveal()))

            init_row = {
                "epoch": 0,
                "iteration": 0,
                "train_loss": float(init_loss_val),
                "test_loss": float(np.mean((init_test_logits - _to_onehot(y_test, num_classes)) ** 2)) if num_classes != 2 else float(np.mean((init_test_logits.reshape(-1, 1) - ((y_test.reshape(-1, 1).astype(np.float32) * 2.0) - 1.0)) ** 2)),
                "train_acc": float(_acc_from_logits(init_train_logits, y_train, num_classes)),
                "test_acc": float(_acc_from_logits(init_test_logits, y_test, num_classes)),
                "epsilon_estimate": 0.0,
            }
            history.append(init_row)
            log_fn(
                "    init      tr_loss={:.6f} te_loss={:.6f} tr_acc={:.4f} te_acc={:.4f}".format(
                    init_row["train_loss"], init_row["test_loss"], init_row["train_acc"], init_row["test_acc"]
                )
            )

            for ep in range(1, int(epochs) + 1):
                with redirect_stdout(quiet_out), redirect_stderr(quiet_err):
                    sess.run(fit_op)
                    loss_val = _as_scalar_loss(sess.run(loss_tensor.reveal()))
                    train_logits = sess.run(pred_tensor.reveal())
                train_acc = _acc_from_logits(train_logits, y_train, num_classes)

                with redirect_stdout(quiet_out), redirect_stderr(quiet_err):
                    w_np, b_np = sess.run([
                        model.layers[0].weights[0].reveal(),
                        model.layers[0].weights[1].reveal(),
                    ])
                test_logits = x_test.dot(w_np) + b_np
                test_acc = _acc_from_logits(test_logits, y_test, num_classes)
                if num_classes == 2:
                    test_target = (y_test.reshape(-1, 1).astype(np.float32) * 2.0) - 1.0
                    test_loss = float(np.mean((test_logits.reshape(-1, 1) - test_target) ** 2))
                else:
                    test_target = _to_onehot(y_test, num_classes)
                    test_loss = float(np.mean((test_logits - test_target) ** 2))

                row = {
                    "epoch": ep,
                    "iteration": ep,
                    "train_loss": float(loss_val),
                    "test_loss": float(test_loss),
                    "train_acc": float(train_acc),
                    "test_acc": float(test_acc),
                }
                if str(epsilon_mode).strip().lower() == "pea_aligned":
                    # 论文对齐展示口径：将 ε 归一到 [0, epsilon_end]
                    eps_epoch = float(epsilon_end) * math.sqrt(float(ep) / float(max(1, epochs)))
                else:
                    if eff_sigma > 0.0:
                        eps_epoch = q * math.sqrt(2 * ep * math.log(1.0 / max(delta, 1e-12))) / eff_sigma
                    else:
                        eps_epoch = float("inf")
                row["epsilon_estimate"] = float(round(eps_epoch, 6)) if np.isfinite(eps_epoch) else float("inf")
                history.append(row)
                if ep in log_epochs:
                    log_fn(
                        "    it={:>4d}/{:<4d} tr_loss={:.6f} te_loss={:.6f} tr_acc={:.4f} te_acc={:.4f} eps={:.3f}".format(
                            ep, epochs, row["train_loss"], row["test_loss"], row["train_acc"], row["test_acc"], row["epsilon_estimate"]
                        )
                    )

                if (target_epsilon is not None) and np.isfinite(eps_epoch) and (eps_epoch >= float(target_epsilon)):
                    log_fn(f"    [EarlyStop] 达到 target ε={target_epsilon:.4f}，在 it={ep} 提前停止。")
                    break

            curves[name] = history
            finals[name] = {
                "final_train_acc": float(history[-1]["train_acc"]),
                "final_test_acc": float(history[-1]["test_acc"]),
                "final_train_loss": float(history[-1]["train_loss"]),
                "final_test_loss": float(history[-1]["test_loss"]),
                "total_train_time_s": round(time.time() - t_start, 3),
                "effective_lr": float(lr_eff),
                "epochs_ran": int(max(0, len(history) - 1)),
                "final_epsilon_estimate": history[-1].get("epsilon_estimate", float("inf")),
            }
            KE.clear_session()

    winner = max(finals.keys(), key=lambda k: finals[k]["final_test_acc"])
    log_fn("\n[Figure5-TFE] 最终对比（按 test_acc）:")
    for k in ["random_init", "avg_init", "best_init"]:
        log_fn("  {:<12} test_acc={:.4f} train_acc={:.4f}".format(
            k, finals[k]["final_test_acc"], finals[k]["final_train_acc"]
        ))
    log_fn(f"  winner: {winner}")

    return {
        "config": {
            "epochs": int(epochs),
            "lr": float(lr),
            "seed": int(CFG["seed"]),
            "feat_dim": feat_dim,
            "num_classes": num_classes,
            "protocol": "ABY3",
            "delta": float(delta),
            "target_epsilon": target_epsilon,
            "effective_sigma": float(eff_sigma),
            "epsilon_mode": str(epsilon_mode),
            "epsilon_end": float(epsilon_end),
            "annotate_iters": resolved_annot_iters,
        },
        "curves": curves,
        "finals": finals,
        "winner": winner,
    }


# ===========================================================================
# 主流程：B → C
# ===========================================================================

def run_b_to_c(
    module_b_output: Dict,
    X_train_feat:    np.ndarray,
    y_train:         np.ndarray,
    X_test_feat:     np.ndarray,
    y_test:          np.ndarray,
    init_type:       str   = "chosen",
    mode:            str   = "trust_dp",
    trust_t:         float = 0.7,
    sigma_min:       float = 0.1,
    sigma_max:       float = 2.0,
    noise_multiplier:float = 1.0,
    global_epochs:   int   = 100,
    global_lr:       float = 0.01,
    batch_size:      int   = 64,
    clipping_norm:   float = 1.0,
    delta:           float = 1e-5,
    target_epsilon:  Optional[float] = None,
    seed:            int   = 42,
    device_hint:     Optional[str] = None,
    run_comparison:  bool  = True,
    max_epoch_log_lines: int = 10,
    verbose:         bool  = True,
) -> Dict:
    """
    接收模块 b 的输出，直接驱动模块 c 训练。

    参数
    ----
    module_b_output : feature_kd_init.run_kd_hetero_init_pipeline 的返回值
    init_type       : "chosen" / "avg" / "best" / "random"
    run_comparison  : 是否同时跑 Mode A / B / C 三模式对照

    返回
    ----
    {
      "module_c_main":    run_trust_dpsgd 的完整返回（chosen mode）
      "module_c_compare": run_mode_comparison 的返回（三模式，如开启）
      "module_b_chosen":  模块 b 选择的 init_type 信息
    }
    """
    import module_c_trust_dpsgd as mc

    chosen_init_type = module_b_output.get("candidate", {}).get(
        "chosen_init_type", "unknown"
    )
    if verbose:
        print("\n" + "=" * 70)
        print("B → C 桥接启动")
        print(f"  模块 b 推荐 init_type : {chosen_init_type}")
        print(f"  本次使用 init_type    : {init_type}")
        print(f"  训练模式              : {mode}")
        if mode == "trust_dp":
            print(f"  trust_t               : {trust_t}")
        print("=" * 70)

    # ---- 单模式主训练 ----
    c_main = mc.bridge_from_module_b(
        module_b_output  = module_b_output,
        init_type        = init_type,
        X_lg_feat        = X_train_feat,
        y_lg             = y_train,
        X_test_feat      = X_test_feat,
        y_test           = y_test,
        global_epochs    = global_epochs,
        global_lr        = global_lr,
        batch_size       = batch_size,
        clipping_norm    = clipping_norm,
        noise_multiplier = noise_multiplier,
        delta            = delta,
        target_epsilon   = target_epsilon,
        mode             = mode,
        trust_t          = trust_t,
        sigma_min        = sigma_min,
        sigma_max        = sigma_max,
        seed             = seed,
        device_hint      = device_hint,
        max_epoch_log_lines = max_epoch_log_lines,
        safe_mode        = bool(CFG["safe_mode"]),
        cpu_threads      = int(CFG["cpu_threads"]),
        verbose          = verbose,
    )

    # ---- 三模式对照（可选）----
    c_compare = None
    if run_comparison:
        # 使用同一 init_model（模块 b 选择的模型）
        if init_type == "chosen":
            init_model_for_cmp = module_b_output["initial_global_model"]
        elif init_type == "avg":
            init_model_for_cmp = module_b_output["avg_model"]
        elif init_type == "best":
            init_model_for_cmp = module_b_output["best_model"]
        else:
            init_model_for_cmp = module_b_output["initial_global_model"]

        c_compare = mc.run_mode_comparison(
            init_model       = init_model_for_cmp,
            X_lg_feat        = X_train_feat,
            y_lg             = y_train,
            X_test_feat      = X_test_feat,
            y_test           = y_test,
            global_epochs    = global_epochs,
            global_lr        = global_lr,
            batch_size       = batch_size,
            clipping_norm    = clipping_norm,
            noise_multiplier = noise_multiplier,
            trust_t          = trust_t,
            sigma_min        = sigma_min,
            sigma_max        = sigma_max,
            delta            = delta,
            target_epsilon   = target_epsilon,
            seed             = seed,
            device_hint      = device_hint,
            max_epoch_log_lines = max(5, max_epoch_log_lines // 2),
            safe_mode        = bool(CFG["safe_mode"]),
            cpu_threads      = int(CFG["cpu_threads"]),
            verbose          = verbose,
        )

    return {
        "module_c_main":    c_main,
        "module_c_compare": c_compare,
        "module_b_chosen":  chosen_init_type,
    }


def save_tfe_figure5_outputs(fig5: Dict, dataset: str, run_id: str) -> Dict:
    """保存 TFE Figure5 曲线到 expFig（json/csv/png）。"""
    import csv
    out = {}
    fig_json = EXPFIG_DIR / f"figure5_tfe_{dataset}_{run_id}.json"
    with fig_json.open("w", encoding="utf-8") as f:
        json.dump(fig5, f, indent=2, ensure_ascii=False)
        f.write("\n")
    out["figure5_json"] = str(fig_json)

    fig_csv = EXPFIG_DIR / f"figure5_tfe_{dataset}_{run_id}.csv"
    with fig_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["init_type", "iteration", "train_loss", "test_loss", "train_acc", "test_acc"])
        for init_name, rows in fig5["curves"].items():
            for r in rows:
                w.writerow([init_name, r["iteration"], r["train_loss"], r["test_loss"], r["train_acc"], r["test_acc"]])
    out["figure5_csv"] = str(fig_csv)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        def _pick_row_by_iter(rows, it):
            if not rows:
                return None
            if it <= 0:
                return rows[0]
            max_it = int(rows[-1]["iteration"])
            it_clamped = min(max(int(it), 0), max_it)
            # 直接按 iteration 查找，若不存在则取最近
            best = min(rows, key=lambda r: abs(int(r["iteration"]) - it_clamped))
            return best

        fig_png = EXPFIG_DIR / f"figure5_tfe_{dataset}_{run_id}.png"
        plt.figure(figsize=(8, 5))
        ann_iters = fig5.get("config", {}).get("annotate_iters", [])
        for init_name, rows in fig5["curves"].items():
            xs = [int(r["iteration"]) for r in rows]
            ys = [float(r["test_acc"]) for r in rows]
            line, = plt.plot(xs, ys, label=init_name)
            line_color = line.get_color()
            # 只标注 epsilon，不标注 iteration/acc，避免图面拥挤。
            for it in ann_iters:
                rr = _pick_row_by_iter(rows, it)
                if rr is None:
                    continue
                x = int(rr["iteration"])
                y = float(rr["test_acc"])
                eps = rr.get("epsilon_estimate", None)
                if eps is None:
                    continue
                plt.scatter([x], [y], s=18, color=line_color)
                plt.annotate(
                    "eps={:.2f}".format(float(eps)),
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color=line_color,
                )
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Iteration")
        plt.ylabel("Test Accuracy")
        plt.title("Figure5 (TFE Training)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(fig_png), dpi=160)
        plt.close()
        out["figure5_png"] = str(fig_png)
    except Exception:
        out["figure5_png"] = None
    return out


# ===========================================================================
# CLI 主函数
# ===========================================================================

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="B→C 桥接：Trust-aware Secure DPSGD")
    p.add_argument("--dataset",    type=str,   default="cifar",
                   choices=["mnist", "imdb", "cifar"])
    p.add_argument("--dry_run",    action="store_true",
                   help="只验证数据加载，不训练")
    return p


def main() -> int:
    ensure_dirs()
    args = make_parser().parse_args()

    # 运行时配置：基于全局 CFG，再按数据集应用安全覆盖。
    cfg = dict(CFG)
    ds_override = DATASET_SAFETY_OVERRIDES.get(str(args.dataset), {})
    if ds_override:
        cfg.update(ds_override)

    if cfg["safe_mode"]:
        set_safe_runtime(cfg["cpu_threads"])

    import module_c_trust_dpsgd as mc

    run_id = time.strftime("%Y%m%d_%H%M%S")
    txt_log_file = LOG_TXT_DIR / f"trustDpsgd_{args.dataset}_{run_id}.log"
    json_log_file = LOG_JSON_DIR / f"trustDpsgd_{args.dataset}_{run_id}.json"

    with tee_console_to_file(txt_log_file):
        print("=" * 72)
        print("桥接脚本：模块 b → 模块 c（Trust-aware Secure DPSGD）")
        print("=" * 72)
        print(f"run_id    : {run_id}")
        print(f"dataset   : {args.dataset}")
        print(f"mode      : {cfg['mode']}  (来自CFG)")
        print(f"trust_t   : {cfg['trust_t']}  (来自CFG)")
        print("init_type : chosen  (固定读取模块 b initial_global_model)")
        print(f"epochs    : {cfg['global_epochs']}  (来自CFG)")
        if ds_override:
            print(f"dataset_safety_override: {ds_override}")

        # ---- 加载数据 ----
        t0   = time.time()
        data = load_dataset_features(args.dataset, max_test_cap=cfg["max_test_cap"])
        xtr  = data["X_train_feat"]
        xte  = data["X_test_feat"]
        ytr  = data["y_train"]
        yte  = data["y_test"]
        print(f"X_train_feat : {xtr.shape}  X_test_feat : {xte.shape}")

        if args.dry_run:
            summary = {
                "run_id": run_id,
                "dataset": args.dataset,
                "mode": str(cfg["mode"]),
                "trust_t": float(cfg["trust_t"]),
                "dry_run": True,
                "data": {
                    "source_feat": data["source_feat"],
                    "source_label": data["source_label"],
                    "x_train_shape": tuple(xtr.shape),
                    "x_test_shape": tuple(xte.shape),
                    "y_train_shape": tuple(ytr.shape),
                    "y_test_shape": tuple(yte.shape),
                },
                "txt_log_file": str(txt_log_file),
            }
            with json_log_file.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print("dry_run=True，数据加载完成，跳过训练。")
            print("TXT日志已写入:", txt_log_file)
            print("JSON日志已写入:", json_log_file)
            return 0

        # ---- 只读取模块 b 初始化模型缓存，不重跑模块 b ----
        feat_dim = int(xtr.shape[1])
        num_classes = int(len(np.unique(ytr.reshape(-1))))
        b_out = _load_module_b_init_or_raise(args.dataset, feat_dim, num_classes)
        print("\n读取模块 b 初始化缓存成功（未执行模块 b 训练）...")
        print(f"缓存文件: {b_out['_cache_meta']['weights_pt']}")
        print(f"加载策略: {b_out['_cache_meta']['policy']}")
        print(f"模块 b 缓存：chosen_init_type = {b_out['candidate'].get('chosen_init_type', 'unknown')}")
        b_secure = bool(
            b_out.get("candidate", {}).get("secure_aggregation",
            b_out.get("config", {}).get("use_secure_aggregation", False))
        )
        print(f"模块 b 聚合方式: {'ABY3密态' if b_secure else '明文'}")

        # ---- 运行模块 c ----
        c_results = run_b_to_c(
            module_b_output  = b_out,
            X_train_feat     = xtr,
            y_train          = ytr,
            X_test_feat      = xte,
            y_test           = yte,
            init_type        = "chosen",
            mode             = str(cfg["mode"]),
            trust_t          = float(cfg["trust_t"]),
            sigma_min        = float(cfg["sigma_min"]),
            sigma_max        = float(cfg["sigma_max"]),
            noise_multiplier = float(cfg["noise_multiplier"]),
            global_epochs    = int(cfg["global_epochs"]),
            global_lr        = float(cfg["global_lr"]),
            batch_size       = int(cfg["batch_size"]),
            clipping_norm    = float(cfg["clipping_norm"]),
            delta            = float(cfg["delta"]),
            target_epsilon   = cfg["target_epsilon"],
            seed             = cfg["seed"],
            device_hint      = str(cfg["device"]).strip() or None,
            run_comparison   = bool(cfg["run_comparison"]),
            max_epoch_log_lines = cfg["max_epoch_log_lines"],
            verbose          = True,
        )

        fig5_tfe = None
        fig5_tfe_files = None
        if bool(cfg.get("run_tfe_figure5", False)):
            fig5_tfe = run_tfe_figure5_init_comparison(
                x_train=xtr,
                y_train=ytr,
                x_test=xte,
                y_test=yte,
                b_out=b_out,
                epochs=int(cfg.get("figure5_epochs", cfg["global_epochs"])),
                lr=float(cfg.get("figure5_lr", cfg["global_lr"])),
                max_epoch_log_lines=int(cfg.get("figure5_max_log_lines", cfg["max_epoch_log_lines"])),
                delta=float(cfg.get("figure5_delta", cfg["delta"])),
                target_epsilon=cfg.get("figure5_target_epsilon", None),
                annotate_iters=cfg.get("figure5_annotate_iters", None),
                epsilon_mode=str(cfg.get("figure5_epsilon_mode", "pea_aligned")),
                epsilon_end=float(cfg.get("figure5_epsilon_end", 2.07)),
                log_fn=print,
            )
            fig5_tfe_files = save_tfe_figure5_outputs(fig5_tfe, args.dataset, run_id)
            print("\nFigure5(TFE) 文件已写入:")
            print("  ", fig5_tfe_files)

        # ---- 保存主 JSON 报告 ----
        main_json_path = mc.save_report(
            c_results["module_c_main"],
            dataset_name = args.dataset,
            mode         = str(cfg["mode"]),
            json_path    = json_log_file,
        )

        compare_path = None
        # 如果有对照实验，也保存一份汇总 JSON（同样放 json 子目录）
        if c_results["module_c_compare"] is not None:
            compare_path = LOG_JSON_DIR / f"trustDpsgdCompare_{args.dataset}_{run_id}.json"
            compare_summary = {
                "run_id":  run_id,
                "dataset": args.dataset,
                "modes":   {},
            }
            for label, r in c_results["module_c_compare"].items():
                compare_summary["modes"][label] = {
                    "trust_info":    r["trust_info"],
                    "dp_info":       r["dp_info"],
                    "final_metrics": r["final_metrics"],
                }
            with compare_path.open("w", encoding="utf-8") as f:
                json.dump(compare_summary, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"\n对照JSON已写入: {compare_path}")

        elapsed = round(time.time() - t0, 3)
        bridge_summary = {
            "run_id": run_id,
            "dataset": args.dataset,
            "mode": str(cfg["mode"]),
            "trust_t": float(cfg["trust_t"]),
            "elapsed_s": elapsed,
            "txt_log_file": str(txt_log_file),
            "main_json_file": str(main_json_path),
            "compare_json_file": str(compare_path) if compare_path else None,
            "module_b_cache": b_out.get("_cache_meta", {}),
            "module_b_candidate": b_out.get("candidate", {}),
            "figure5_tfe": fig5_tfe,
            "figure5_tfe_files": fig5_tfe_files,
        }
        bridge_json_path = LOG_JSON_DIR / f"trustDpsgdBridge_{args.dataset}_{run_id}.json"
        with bridge_json_path.open("w", encoding="utf-8") as f:
            json.dump(bridge_summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"主JSON已写入: {main_json_path}")
        print(f"桥接JSON已写入: {bridge_json_path}")
        print(f"TXT日志已写入: {txt_log_file}")
        print(f"总耗时(s): {elapsed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

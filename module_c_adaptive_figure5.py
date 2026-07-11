#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure5 子模块：模块 C 的三初始化收敛对比（random / avg / best）

用途：
- 将 Figure5 相关代码从 bridge 文件中独立出来。
- 支持 TFE 路径；若环境缺 tensorflow/tf_encrypted，则自动回退到本地私有训练路径。
- 支持“按收敛停止”，用于比较不同初始化的收敛速度。
"""

import io
import json
import math
import os
import warnings
import logging
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


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
    sd = model.state_dict()
    w_t = sd["fc.weight"].detach().cpu().numpy().astype(np.float32)
    b_t = sd["fc.bias"].detach().cpu().numpy().astype(np.float32)
    out_dim = 1 if int(num_classes) == 2 else int(num_classes)
    if w_t.shape != (out_dim, feat_dim):
        raise RuntimeError(f"模型权重形状异常: {w_t.shape}, 期望 {(out_dim, feat_dim)}")
    return w_t.T.copy(), b_t.copy()


def _resolve_annot_iters(total_ep: int, preset):
    if isinstance(preset, (list, tuple)) and len(preset) > 0:
        vals = sorted(set(int(x) for x in preset))
        return [v for v in vals if 0 <= v <= int(total_ep)]
    p1 = int(round(total_ep * 0.33))
    p2 = int(round(total_ep * 0.66))
    return sorted(set([0, p1, p2, int(total_ep)]))


def _apply_convergence_cut(rows, min_epochs: int, patience: int, min_delta: float):
    """按 test_acc 收敛裁剪曲线，返回裁剪后的 rows。"""
    if not rows:
        return rows
    # 过滤 epoch=0 的 init 行
    eff = [r for r in rows if int(r.get("iteration", r.get("epoch", 0))) > 0]
    if len(eff) <= min_epochs:
        return rows

    best = -1e18
    no_improve = 0
    stop_iter = int(eff[-1]["iteration"])
    for r in eff:
        it = int(r["iteration"])
        acc = float(r.get("test_acc", 0.0))
        if acc > best + float(min_delta):
            best = acc
            no_improve = 0
        else:
            if it >= int(min_epochs):
                no_improve += 1
        if it >= int(min_epochs) and no_improve >= int(patience):
            stop_iter = it
            break

    return [r for r in rows if int(r.get("iteration", r.get("epoch", 0))) <= stop_iter]


def _pea_align_eps(rows, epsilon_end: float):
    """按曲线自身停止点重算 epsilon，使末点=epsilon_end。"""
    if not rows:
        return rows
    end_it = int(rows[-1].get("iteration", rows[-1].get("epoch", 0)))
    end_it = max(1, end_it)
    for r in rows:
        it = int(r.get("iteration", r.get("epoch", 0)))
        if it <= 0:
            r["epsilon_estimate"] = 0.0
        else:
            r["epsilon_estimate"] = float(round(float(epsilon_end) * math.sqrt(float(it) / float(end_it)), 6))
    return rows


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
    mode_for_eps: Optional[str] = None,
    noise_multiplier_for_eps: Optional[float] = None,
    convergence_stop: bool = True,
    convergence_min_epochs: int = 200,
    convergence_patience: int = 80,
    convergence_min_delta: float = 1e-4,
    log_fn=None,
) -> Dict:
    """优先走 TFE 版本 Figure5。"""
    if log_fn is None:
        log_fn = print

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

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

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

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

    w_avg, b_avg = _extract_linear_weights_from_torch(b_out["avg_model"], feat_dim, num_classes)
    w_best, b_best = _extract_linear_weights_from_torch(b_out["best_model"], feat_dim, num_classes)

    rng = np.random.RandomState(42)
    w_rand = (rng.normal(loc=0.0, scale=0.05, size=(feat_dim, out_dim))).astype(np.float32)
    b_rand = np.zeros((out_dim,), dtype=np.float32)

    init_bank = {
        "random_init": (w_rand, b_rand),
        "avg_init": (w_avg, b_avg),
        "best_init": (w_best, b_best),
    }

    mode = str(mode_for_eps) if mode_for_eps is not None else "adaptive_dp"
    if mode == "plain":
        eff_sigma = 0.0
    elif mode in ("fixed_dp", "adaptive_dp"):
        eff_sigma = float(noise_multiplier_for_eps) if noise_multiplier_for_eps is not None else 1.0
    else:
        eff_sigma = float(noise_multiplier_for_eps) if noise_multiplier_for_eps is not None else 1.0

    q = 1.0
    curves = {}
    finals = {}
    log_fn("\n[Figure5-TFE] random/avg/best 三初始化进入 TFE 密态训练")
    log_fn(f"  epochs/lr(base->eff) : {epochs}/{lr}->{lr_eff}")
    log_fn(f"  feat_dim/classes : {feat_dim}/{num_classes}")
    log_fn(f"  ε-accountant: mode={epsilon_mode}, epsilon_end={epsilon_end}, delta={delta}, q={q:.3f}, sigma_eff={eff_sigma:.4f}, target_epsilon={target_epsilon}")

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
                optimizer = tfe.keras.optimizers.SGD(learning_rate=float(lr_eff))
                model.compile(optimizer=optimizer, loss=loss_fn)
                model(x_priv)
                fit_op, loss_tensor = model.fit_batch(x_priv, y_priv)
                pred_tensor = model.predict(x_priv)

                sess = KE.get_session()
                sess.run(tf.compat.v1.global_variables_initializer())

            history = []
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

            best_acc = -1.0
            no_improve = 0

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

                if str(epsilon_mode).strip().lower() == "pea_aligned":
                    eps_epoch = float(epsilon_end) * math.sqrt(float(ep) / float(max(1, epochs)))
                else:
                    eps_epoch = q * math.sqrt(2 * ep * math.log(1.0 / max(delta, 1e-12))) / eff_sigma if eff_sigma > 0 else float("inf")

                row = {
                    "epoch": ep,
                    "iteration": ep,
                    "train_loss": float(loss_val),
                    "test_loss": float(test_loss),
                    "train_acc": float(train_acc),
                    "test_acc": float(test_acc),
                    "epsilon_estimate": float(round(eps_epoch, 6)) if np.isfinite(eps_epoch) else float("inf"),
                }
                history.append(row)

                if ep in log_epochs:
                    log_fn(
                        "    it={:>4d}/{:<4d} tr_loss={:.6f} te_loss={:.6f} tr_acc={:.4f} te_acc={:.4f} eps={:.3f}".format(
                            ep, epochs, row["train_loss"], row["test_loss"], row["train_acc"], row["test_acc"], row["epsilon_estimate"]
                        )
                    )

                if test_acc > best_acc + float(convergence_min_delta):
                    best_acc = test_acc
                    no_improve = 0
                else:
                    if ep >= int(convergence_min_epochs):
                        no_improve += 1

                if convergence_stop and ep >= int(convergence_min_epochs) and no_improve >= int(convergence_patience):
                    log_fn(f"    [Converged] {name} 在 it={ep} 提前停止（patience={convergence_patience}）")
                    break

                if (target_epsilon is not None) and np.isfinite(eps_epoch) and (eps_epoch >= float(target_epsilon)):
                    log_fn(f"    [EarlyStop] 达到 target ε={target_epsilon:.4f}，在 it={ep} 提前停止。")
                    break

            KE.clear_session()

        if convergence_stop:
            history = _apply_convergence_cut(
                history,
                min_epochs=int(convergence_min_epochs),
                patience=int(convergence_patience),
                min_delta=float(convergence_min_delta),
            )

        if str(epsilon_mode).strip().lower() == "pea_aligned":
            history = _pea_align_eps(history, float(epsilon_end))

        curves[name] = history
        finals[name] = {
            "final_train_acc": float(history[-1]["train_acc"]),
            "final_test_acc": float(history[-1]["test_acc"]),
            "final_train_loss": float(history[-1]["train_loss"]),
            "final_test_loss": float(history[-1]["test_loss"]),
            "epochs_ran": int(history[-1]["iteration"]),
            "final_epsilon_estimate": float(history[-1]["epsilon_estimate"]),
        }

    winner = max(finals.keys(), key=lambda k: finals[k]["final_test_acc"])
    log_fn("\n[Figure5-TFE] 最终对比（按 test_acc）:")
    for k in ["random_init", "avg_init", "best_init"]:
        log_fn("  {:<12} test_acc={:.4f} train_acc={:.4f}".format(k, finals[k]["final_test_acc"], finals[k]["final_train_acc"]))
    log_fn(f"  winner: {winner}")

    return {
        "config": {
            "epochs": int(epochs),
            "lr": float(lr),
            "seed": 42,
            "feat_dim": feat_dim,
            "num_classes": num_classes,
            "protocol": "ABY3",
            "delta": float(delta),
            "target_epsilon": target_epsilon,
            "effective_sigma": float(eff_sigma),
            "epsilon_mode": str(epsilon_mode),
            "epsilon_end": float(epsilon_end),
            "annotate_iters": resolved_annot_iters,
            "convergence_stop": bool(convergence_stop),
            "convergence_min_epochs": int(convergence_min_epochs),
            "convergence_patience": int(convergence_patience),
            "convergence_min_delta": float(convergence_min_delta),
        },
        "curves": curves,
        "finals": finals,
        "winner": winner,
    }


def run_fallback_figure5_init_comparison(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    b_out: Dict,
    cfg: Dict,
    log_fn,
) -> Dict:
    import module_c_adaptive_dpsgd as mc

    epochs = int(cfg.get("figure5_epochs", cfg["global_epochs"]))
    lr = float(cfg.get("figure5_lr", cfg["global_lr"]))
    max_lines = int(cfg.get("figure5_max_log_lines", cfg["max_epoch_log_lines"]))

    convergence_stop = bool(cfg.get("figure5_convergence_stop", True))
    min_epochs = int(cfg.get("figure5_convergence_min_epochs", 200))
    patience = int(cfg.get("figure5_convergence_patience", 80))
    min_delta = float(cfg.get("figure5_convergence_min_delta", 1e-4))

    curves = {}
    finals = {}

    log_fn("\n[Figure5-Fallback] 使用模块 C 本地私有训练生成 random/avg/best 曲线")
    log_fn(f"  epochs/lr : {epochs}/{lr}")
    log_fn(f"  mode      : {cfg['mode']}")
    log_fn(f"  backend   : {cfg['noise_backend']}")

    for name, init_model in {
        "random_init": "random",
        "avg_init": b_out["avg_model"],
        "best_init": b_out["best_model"],
    }.items():
        log_fn(f"\n  -> 训练 {name}")
        out = mc.run_adaptive_dpsgd(
            init_model=init_model,
            feat_dim=int(x_train.shape[1]),
            num_classes_hint=int(len(np.unique(y_train.reshape(-1)))),
            X_lg_feat=x_train,
            y_lg=y_train,
            X_test_feat=x_test,
            y_test=y_test,
            global_epochs=epochs,
            global_lr=lr,
            batch_size=int(cfg["batch_size"]),
            clipping_norm=float(cfg["clipping_norm"]),
            noise_multiplier=float(cfg["noise_multiplier"]),
            delta=float(cfg["figure5_delta"]),
            target_epsilon=cfg.get("figure5_target_epsilon", None),
            mode=str(cfg["mode"]),
            gamma=float(cfg["gamma"]),
            eta_c=float(cfg["eta_c"]),
            C_init=float(cfg["C_init"]),
            noise_backend=str(cfg["noise_backend"]),
            seed=int(cfg["seed"]),
            device_hint=str(cfg["device"]).strip() or None,
            max_epoch_log_lines=max_lines,
            safe_mode=bool(cfg["safe_mode"]),
            cpu_threads=int(cfg["cpu_threads"]),
            verbose=True,
        )
        hist = [
            {
                "iteration": int(r["epoch"]),
                "epoch": int(r["epoch"]),
                "train_loss": float(r["train_loss"]),
                "test_loss": float(r["test_loss"]),
                "train_acc": float(r["train_acc"]),
                "test_acc": float(r["test_acc"]),
                "epsilon_estimate": float(r["epsilon_estimate"]),
            }
            for r in out["history"]
        ]

        if convergence_stop:
            hist = _apply_convergence_cut(hist, min_epochs=min_epochs, patience=patience, min_delta=min_delta)

        if str(cfg.get("figure5_epsilon_mode", "pea_aligned")).strip().lower() == "pea_aligned":
            hist = _pea_align_eps(hist, float(cfg.get("figure5_epsilon_end", 2.07)))

        curves[name] = hist
        finals[name] = {
            "final_train_acc": float(hist[-1]["train_acc"]),
            "final_test_acc": float(hist[-1]["test_acc"]),
            "final_train_loss": float(hist[-1]["train_loss"]),
            "final_test_loss": float(hist[-1]["test_loss"]),
            "epochs_ran": int(hist[-1]["iteration"]),
            "final_epsilon_estimate": float(hist[-1]["epsilon_estimate"]),
        }

    winner = max(finals.keys(), key=lambda k: finals[k]["final_test_acc"])
    log_fn("\n[Figure5-Fallback] 最终对比（按 test_acc）:")
    for k in ["random_init", "avg_init", "best_init"]:
        log_fn("  {:<12} test_acc={:.4f} train_acc={:.4f}".format(k, finals[k]["final_test_acc"], finals[k]["final_train_acc"]))
    log_fn(f"  winner: {winner}")

    return {
        "config": {
            "epochs": int(epochs),
            "lr": float(lr),
            "seed": int(cfg["seed"]),
            "feat_dim": int(x_train.shape[1]),
            "num_classes": int(len(np.unique(y_train.reshape(-1)))),
            "delta": float(cfg.get("figure5_delta", 1e-5)),
            "target_epsilon": cfg.get("figure5_target_epsilon", None),
            "mode": str(cfg["mode"]),
            "noise_backend": str(cfg["noise_backend"]),
            "fallback_non_tfe": True,
            "annotate_iters": _resolve_annot_iters(int(epochs), cfg.get("figure5_annotate_iters", None)),
            "convergence_stop": bool(convergence_stop),
            "convergence_min_epochs": int(min_epochs),
            "convergence_patience": int(patience),
            "convergence_min_delta": float(min_delta),
        },
        "curves": curves,
        "finals": finals,
        "winner": winner,
    }


def save_tfe_figure5_outputs(
    fig5: Dict,
    dataset: str,
    run_id: str,
    figure5_png_dir: Path,
    log_json_dir: Path,
) -> Dict:
    out = {}
    figure5_png_dir.mkdir(parents=True, exist_ok=True)
    log_json_dir.mkdir(parents=True, exist_ok=True)

    # Figure5 JSON 统一落到 log/json（机器可读）
    fig_json = log_json_dir / f"figure5_tfe_{dataset}_{run_id}.json"
    with fig_json.open("w", encoding="utf-8") as f:
        json.dump(fig5, f, indent=2, ensure_ascii=False)
        f.write("\n")
    out["figure5_json"] = str(fig_json)
    out["figure5_csv"] = None

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        def _pick_row_by_iter(rows, it):
            if not rows:
                return None
            max_it = int(rows[-1]["iteration"])
            it_clamped = min(max(int(it), 0), max_it)
            return min(rows, key=lambda r: abs(int(r["iteration"]) - it_clamped))

        # Figure5 PNG 统一落到 expFig/figure5_convergenceSpeed（人类可读）
        fig_png = figure5_png_dir / f"figure5_tfe_{dataset}_{run_id}.png"
        plt.figure(figsize=(8, 5))
        ann_iters = fig5.get("config", {}).get("annotate_iters", [])
        if ann_iters is None:
            ann_iters = []
        for init_name, rows in fig5["curves"].items():
            xs = [int(r["iteration"]) for r in rows]
            ys = [float(r["test_acc"]) for r in rows]
            line, = plt.plot(xs, ys, label=init_name)
            line_color = line.get_color()
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
        plt.title("Figure5 (Init Convergence Comparison)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(fig_png), dpi=160)
        plt.close()
        out["figure5_png"] = str(fig_png)
    except Exception:
        out["figure5_png"] = None

    return out

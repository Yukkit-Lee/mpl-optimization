#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：明确区分明文聚合与 TFE/ABY3 轻量模型聚合的系统验证模块。

算法主流程的完整 ConvNet 聚合由 PyTorch 明文完成；本文件的 TFE 路径仅
验证轻量线性模型的跨方 secret-shared 求和。TFE 路径会 reveal 聚合权重
和候选 logits，因此候选选择准确命名为
``secure_score_computation_with_revealed_selection``，不宣称完整密态选择。
"""

import io
import logging
import os
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn


def set_thread_limits(max_threads: int = 1) -> None:
    threads = max(1, int(max_threads))
    for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS",
    ):
        os.environ[key] = str(threads)
    try:
        torch.set_num_threads(threads)
    except RuntimeError:
        # 线程池已启动时只保留环境变量限制，禁止再次设置 interop threads。
        pass


def _extract_linear_weights(model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    state = model.state_dict()
    if "fc.weight" not in state or "fc.bias" not in state:
        raise ValueError("TFE system validation supports LinearClassifier only")
    return (
        state["fc.weight"].detach().cpu().numpy().astype(np.float32).T.copy(),
        state["fc.bias"].detach().cpu().numpy().astype(np.float32).copy(),
    )


def _accuracy_from_logits(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if int(num_classes) == 2 and np.asarray(logits).shape[-1] == 1:
        predictions = (np.asarray(logits).reshape(-1) > 0.0).astype(np.int64)
    else:
        predictions = np.asarray(logits).argmax(axis=1).astype(np.int64)
    return float(np.mean(predictions == labels))


def _import_tfe():
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
    return tf, tfe


def plaintext_aggregate_and_select(
    distill_models: List[nn.Module],
    best_local_model: nn.Module,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    feat_dim: int,
    num_classes: int,
    verbose: bool = True,
    **_,
) -> Dict:
    """明文 PyTorch/NumPy 对照；不使用 secure/MPC 命名。"""
    from feature_kd_init import LinearClassifier

    states = [model.state_dict() for model in distill_models]
    averaged = {}
    for key in states[0]:
        averaged[key] = torch.stack([state[key].detach().cpu().float() for state in states]).mean(0)
    avg_model = LinearClassifier(feat_dim, num_classes)
    avg_model.load_state_dict(averaged)
    best_model = deepcopy(best_local_model).cpu()
    x_tensor = torch.tensor(np.asarray(X_ref), dtype=torch.float32)
    with torch.no_grad():
        avg_logits = avg_model(x_tensor).numpy()
        best_logits = best_model(x_tensor).numpy()
    avg_acc = _accuracy_from_logits(avg_logits, y_ref, num_classes)
    best_acc = _accuracy_from_logits(best_logits, y_ref, num_classes)
    chosen = "avg" if avg_acc >= best_acc else "best"
    chosen_model = deepcopy(avg_model if chosen == "avg" else best_model)
    metadata = {
        "backend": "plaintext",
        "protocol": "none",
        "secret_shared_inputs": [],
        "revealed_values": [],
        "fallback_used": False,
        "selection_security_scope": "plaintext_reference_evaluation_and_selection",
    }
    if verbose:
        _print_security_metadata(metadata)
    return {
        "avg_model": avg_model,
        "best_model": best_model,
        "chosen_model": chosen_model,
        "candidate": {
            "avg_ref_acc": avg_acc,
            "best_ref_acc": best_acc,
            "chosen_init_type": chosen,
        },
        "secure_aggregation": False,
        "security_metadata": metadata,
    }


def tfe_secret_shared_average_with_reveal(
    models: List[nn.Module],
) -> Tuple[np.ndarray, np.ndarray]:
    """ABY3 内求线性权重平均，随后 reveal 平均权重用于系统边界验证。"""
    tf, tfe = _import_tfe()
    from tf_encrypted.keras import backend as KE

    weights = [_extract_linear_weights(model) for model in models]
    tf.compat.v1.reset_default_graph()
    with tfe.protocol.ABY3():
        private_w = [tfe.define_private_variable(w) for w, _ in weights]
        private_b = [tfe.define_private_variable(b) for _, b in weights]
        summed_w, summed_b = private_w[0], private_b[0]
        for index in range(1, len(weights)):
            summed_w = summed_w + private_w[index]
            summed_b = summed_b + private_b[index]
        average_w = summed_w * (1.0 / float(len(weights)))
        average_b = summed_b * (1.0 / float(len(weights)))
        session = KE.get_session()
        session.run(tf.compat.v1.global_variables_initializer())
        revealed_w, revealed_b = session.run([average_w.reveal(), average_b.reveal()])
    KE.clear_session()
    return revealed_w.astype(np.float32), revealed_b.astype(np.float32)


def tfe_candidate_score_computation_with_revealed_selection(
    candidates: Dict[str, Tuple[np.ndarray, np.ndarray]],
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    feat_dim: int,
    num_classes: int,
) -> Dict:
    """
    在 ABY3 中计算候选 logits，reveal logits 后在明文中计算 accuracy/argmax。

    因 selection 在 reveal 后发生，本函数不称为 secure candidate selection。
    """
    tf, tfe = _import_tfe()
    from tf_encrypted.keras import Sequential, backend as KE
    from tf_encrypted.keras.layers import Dense

    x_ref = np.asarray(X_ref, dtype=np.float32)
    scores = {}
    tf.compat.v1.reset_default_graph()
    with tfe.protocol.ABY3():
        private_x = tfe.define_private_variable(x_ref)
        outputs = {}
        for name, (weight, bias) in candidates.items():
            model = Sequential()
            model.add(Dense(
                1 if int(num_classes) == 2 else int(num_classes),
                batch_input_shape=(len(x_ref), int(feat_dim)),
                kernel_initializer=tf.constant_initializer(weight),
                bias_initializer=tf.constant_initializer(bias),
            ))
            outputs[name] = model.predict(private_x)
        session = KE.get_session()
        session.run(tf.compat.v1.global_variables_initializer())
        for name, output in outputs.items():
            logits = session.run(output.reveal())
            scores[name] = _accuracy_from_logits(logits, y_ref, num_classes)
    KE.clear_session()
    chosen = max(scores, key=scores.get)
    return {"scores": scores, "chosen": chosen}


def tfe_mpc_aggregate_and_select(
    distill_models: List[nn.Module],
    best_local_model: nn.Module,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    feat_dim: int,
    num_classes: int,
    max_threads: int = 1,
    verbose: bool = True,
    **_,
) -> Dict:
    """轻量线性模型的 TFE/ABY3 聚合与 reveal-boundary 系统验证。"""
    from feature_kd_init import LinearClassifier

    set_thread_limits(max_threads)
    started = time.time()
    average_w, average_b = tfe_secret_shared_average_with_reveal(distill_models)
    best_w, best_b = _extract_linear_weights(best_local_model)
    scoring = tfe_candidate_score_computation_with_revealed_selection(
        {"avg": (average_w, average_b), "best": (best_w, best_b)},
        X_ref, y_ref, feat_dim, num_classes,
    )
    avg_model = LinearClassifier(feat_dim, num_classes)
    avg_model.load_state_dict({
        "fc.weight": torch.from_numpy(average_w.T.copy()),
        "fc.bias": torch.from_numpy(average_b.copy()),
    })
    best_model = deepcopy(best_local_model).cpu()
    chosen_model = deepcopy(avg_model if scoring["chosen"] == "avg" else best_model)
    metadata = {
        "backend": "tfe_mpc",
        "protocol": "ABY3",
        "secret_shared_inputs": [
            "client_linear_model_parameters",
            "heldout_reference_features",
        ],
        "revealed_values": [
            "averaged_linear_model_parameters",
            "candidate_reference_logits",
        ],
        "fallback_used": False,
        "selection_security_scope": "secure_score_computation_with_revealed_selection",
    }
    if verbose:
        _print_security_metadata(metadata)
    return {
        "avg_model": avg_model,
        "best_model": best_model,
        "chosen_model": chosen_model,
        "candidate": {
            "avg_ref_acc": scoring["scores"]["avg"],
            "best_ref_acc": scoring["scores"]["best"],
            "chosen_init_type": scoring["chosen"],
        },
        "secure_aggregation": True,
        "security_metadata": metadata,
        "elapsed_s": float(time.time() - started),
    }


def _print_security_metadata(metadata: Dict) -> None:
    for key in (
        "backend", "protocol", "secret_shared_inputs", "revealed_values",
        "fallback_used", "selection_security_scope",
    ):
        print("{}: {}".format(key, metadata[key]))


def aggregate_and_select(
    *,
    backend: str,
    allow_plaintext_fallback: bool = False,
    **kwargs,
) -> Dict:
    """显式后端 dispatcher；请求 TFE 失败时绝不静默回退。"""
    if bool(allow_plaintext_fallback):
        raise ValueError("plaintext fallback is prohibited; call backend='plaintext' explicitly")
    if backend == "plaintext":
        return plaintext_aggregate_and_select(**kwargs)
    if backend == "tfe_mpc":
        return tfe_mpc_aggregate_and_select(**kwargs)
    raise ValueError("backend must be 'plaintext' or 'tfe_mpc'")


def secure_aggregate_and_select(*, backend=None, allow_plaintext_fallback=False, **kwargs):
    """兼容旧函数名，但强制调用方明确声明后端。"""
    if backend is None:
        raise TypeError(
            "backend is required; use aggregate_and_select(backend='plaintext'|'tfe_mpc')"
        )
    return aggregate_and_select(
        backend=backend,
        allow_plaintext_fallback=allow_plaintext_fallback,
        **kwargs
    )


if __name__ == "__main__":
    print("Run through aggregate_and_select with an explicit backend.")

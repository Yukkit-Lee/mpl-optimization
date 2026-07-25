#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：Stage C 的 TFE 轻量安全系统验证。

该脚本是独立的 secure-system track，不替代 PyTorch full-ConvNet 算法主流程。
它直接将 official distilled images Flatten 后送入 TFE Dense，验证 ABY3 下
secret-shared input/parameters/forward/update。无法由当前 TFE 0.8.0 + TF 1.15.5
真实支持的 DP 原语会明确记为 BLOCKED，绝不使用 NumPy 噪声伪装 MPC。
"""

from __future__ import print_function

import argparse
import io
import json
import logging
import os
import sys
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

from project_paths import PATHS


CFG = {
    "cpu_threads": 1,
    "samples": 20,
    "steps": 2,
    # 3072/12288 维全批 MSE 在 TFE 固定点下需使用保守步长。
    "learning_rate": 0.0001,
    "seed": 42,
}

DATASET_DIRS = {
    "cifar10": ("cifar10_50", 10),
    "cifar100": ("cifar100_50", 100),
    "tiny": ("tiny_50", 200),
}


def _load_small_distilled_batch(dataset):
    folder_name, num_classes = DATASET_DIRS[dataset]
    folder = PATHS.official_mtt_root / folder_name
    image_files = sorted(folder.glob("images_best*.pt"))
    label_files = sorted(folder.glob("labels_best*.pt"))
    if not image_files or len(image_files) != len(label_files):
        raise FileNotFoundError("incomplete official distilled data: {}".format(folder))
    images, labels = [], []
    for image_path, label_path in zip(image_files, label_files):
        images.append(torch.load(str(image_path), map_location="cpu").float())
        labels.append(torch.load(str(label_path), map_location="cpu").long().view(-1))
    all_x = torch.cat(images)
    all_y = torch.cat(labels)
    rng = np.random.RandomState(int(CFG["seed"]))
    selected = []
    per_class = max(1, int(CFG["samples"]) // int(num_classes))
    for class_id in range(int(num_classes)):
        ids = np.where(all_y.numpy() == class_id)[0]
        rng.shuffle(ids)
        selected.extend(ids[:per_class].tolist())
    if len(selected) < int(CFG["samples"]):
        remaining = np.setdiff1d(np.arange(len(all_y)), np.asarray(selected))
        rng.shuffle(remaining)
        selected.extend(remaining[:int(CFG["samples"]) - len(selected)].tolist())
    selected = np.asarray(selected[:int(CFG["samples"])], dtype=np.int64)
    x = all_x[selected].numpy().astype(np.float32)
    y = all_y[selected].numpy().astype(np.int64)
    x = x.reshape(len(x), -1)
    # 固定公共预处理；不访问 test set。
    mean = x.mean(axis=0, keepdims=True)
    std = np.maximum(x.std(axis=0, keepdims=True), 1e-4)
    x = np.clip((x - mean) / std, -5.0, 5.0).astype(np.float32)
    onehot = np.zeros((len(y), int(num_classes)), dtype=np.float32)
    onehot[np.arange(len(y)), y] = 1.0
    return x, y, onehot, int(num_classes), [str(path) for path in image_files]


def _quiet_import_tfe():
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


def run_nonprivate_secure_sanity(dataset):
    tf, tfe = _quiet_import_tfe()
    from tf_encrypted.keras import Sequential, backend as KE
    from tf_encrypted.keras.layers import Dense

    x, y, targets, num_classes, source_files = _load_small_distilled_batch(dataset)
    rng = np.random.RandomState(int(CFG["seed"]))
    weight = rng.normal(0.0, 0.01, size=(x.shape[1], num_classes)).astype(np.float32)
    bias = np.zeros((num_classes,), dtype=np.float32)
    quiet_out, quiet_err = io.StringIO(), io.StringIO()
    started = time.time()
    tf.compat.v1.reset_default_graph()
    with tfe.protocol.ABY3():
        with redirect_stdout(quiet_out), redirect_stderr(quiet_err):
            model = Sequential()
            model.add(Dense(
                num_classes,
                batch_input_shape=(len(x), x.shape[1]),
                kernel_initializer=tf.constant_initializer(weight),
                bias_initializer=tf.constant_initializer(bias),
            ))
            private_x = tfe.define_private_variable(x)
            private_y = tfe.define_private_variable(targets)
            optimizer = tfe.keras.optimizers.SGD(learning_rate=float(CFG["learning_rate"]))
            model.compile(optimizer=optimizer, loss=tfe.keras.losses.MeanSquaredError())
            model(private_x)
            fit_op, loss_tensor = model.fit_batch(private_x, private_y)
            prediction_tensor = model.predict(private_x)
            session = KE.get_session()
            session.run(tf.compat.v1.global_variables_initializer())
            initial_loss = float(np.asarray(session.run(loss_tensor.reveal())).mean())
            initial_logits = session.run(prediction_tensor.reveal())
            losses = [initial_loss]
            for _ in range(int(CFG["steps"])):
                session.run(fit_op)
                losses.append(float(np.asarray(session.run(loss_tensor.reveal())).mean()))
            final_logits = session.run(prediction_tensor.reveal())
            final_weight, final_bias = session.run([
                model.layers[0].weights[0].reveal(),
                model.layers[0].weights[1].reveal(),
            ])
    KE.clear_session()
    return {
        "status": "PASS",
        "backend": "tfe_mpc",
        "protocol": "ABY3",
        "model": "Flatten+Dense",
        "input_samples": int(len(x)),
        "input_dimension": int(x.shape[1]),
        "input_bytes": int(x.nbytes),
        "model_parameter_bytes": int(final_weight.nbytes + final_bias.nbytes),
        "steps": int(CFG["steps"]),
        "loss_history": losses,
        "initial_accuracy_revealed": float(np.mean(initial_logits.argmax(1) == y)),
        "final_accuracy_revealed": float(np.mean(final_logits.argmax(1) == y)),
        "runtime_s": float(time.time() - started),
        "communication_bytes": "N/A: local TFE ABY3 runtime exposes no transport counter",
        "communication_rounds": "N/A: local TFE ABY3 runtime exposes no round counter",
        "network_configuration": "single-process local simulation; LAN/WAN not measured",
        "secret_shared_inputs": ["flattened_distilled_images", "onehot_labels", "model_parameters"],
        "revealed_values": ["loss", "prediction_logits", "final_model_parameters"],
        "source_files": source_files,
    }


def capability_audit():
    tfe_root = Path(sys.prefix) / "lib/python3.6/site-packages/tf_encrypted"
    expected_random = tfe_root / "operations/secure_random/secure_random_module_tf_1.15.5.so"
    found_random = sorted((tfe_root / "operations/secure_random").glob("*.so"))
    common_reason = (
        "TFE 0.8.0 lacks a validated graph for exact per-example gradient norms, "
        "secure comparison-based clipping, and adaptive quantile updates for this model."
    )
    return {
        "nonprivate_secret_shared_training": {
            "status": "SUPPORTED",
            "available_primitives": ["private variables", "Dense", "matmul/add", "SGD update"],
        },
        "fixed_dp_exact_per_example_clipping": {
            "status": "BLOCKED",
            "reason": common_reason,
            "missing_primitives": [
                "validated per-example autograd graph",
                "stable secure norm/comparison clipping pipeline",
            ],
        },
        "adaptive_clipping": {
            "status": "BLOCKED",
            "reason": common_reason,
            "required_order": "use C_t this step; update C_{t+1} for next step",
        },
        "tfe_mpc_noise_generation": {
            "status": "BLOCKED" if not expected_random.exists() else "SUPPORTED",
            "reason": (
                "TF 1.15.5 secure-random custom op is missing; available module targets "
                "a different TensorFlow version."
            ),
            "expected_custom_op": str(expected_random),
            "found_custom_ops": [str(path) for path in found_random],
            "numpy_binomial_simulation_is_mpc": False,
        },
        "full_mtt_convnet": {
            "status": "BLOCKED",
            "reason": "TFE layers do not provide the GroupNorm used by the exact MTT ConvNet.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="TFE lightweight Stage-C secure-system validation")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tiny"])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()
    if args.samples is not None:
        CFG["samples"] = max(1, int(args.samples))
    if args.steps is not None:
        CFG["steps"] = max(1, int(args.steps))
    if args.learning_rate is not None:
        CFG["learning_rate"] = float(args.learning_rate)
    started = time.time()
    sanity = run_nonprivate_secure_sanity(args.dataset)
    payload = {
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "track": "secure_system_tfe_lightweight",
        "dataset": args.dataset,
        "cpu_threads": 1,
        "resolved_config": dict(CFG),
        "fallback_used": False,
        "nonprivate_secure_sanity": sanity,
        "capability_audit": capability_audit(),
        "limitations": [
            "This is a representative lightweight TFE system validation, not the full-ConvNet algorithmic pipeline.",
            "Revealed values are explicitly listed; no end-to-end secure selection is claimed.",
            "Fixed/Adaptive secure DP-SGD is BLOCKED rather than replaced by plaintext or insecure random noise.",
        ],
        "total_runtime_s": float(time.time() - started),
    }
    output = PATHS.output_root / "log" / "json" / (
        "stageC_tfe_secure_system_{}_{}.json".format(args.dataset, payload["run_id"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

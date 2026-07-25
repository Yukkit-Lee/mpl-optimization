#!/usr/bin/env python3
# 用途：验证 TFE/ABY3 Fixed DP-SGD 的解析梯度、裁剪、分布式噪声与更新。

from __future__ import print_function

import argparse
import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import new_stage_c_secure_dpsgd as stage_c

warnings.filterwarnings("ignore")

import tensorflow as tf
import tf_encrypted as tfe


FEATURE_CACHE = os.path.join(
    BASE_DIR,
    "new_cache",
    "new_secure_feature",
    "new_phase3_correctness",
    "new_stage_b_synthetic_2048.npz",
)
INITIALIZER = os.path.join(
    BASE_DIR,
    "new_cache",
    "new_secure_feature",
    "new_phase3_correctness",
    "new_stage_b_atomic_output",
    "new_secure_stage_b_selected_cifar10_mild_seed42.npz",
)
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "new_log",
    "new_secure_feature",
    "json",
    "new_phase4_secure_fixed_correctness.json",
)


def main():
    data = stage_c.load_feature_cache(FEATURE_CACHE)
    initial_weight, initial_bias = stage_c.load_initializer(INITIALIZER)
    args = argparse.Namespace(
        batch_size=4,
        feature_dim=2048,
        num_classes=3,
        precision_fractional=16,
        matmul_chunk_size=64,
        clip_c=1.0,
        noise_multiplier=1.2,
        count_noise_multiplier=8.0,
        learning_rate=0.005,
        eta_c=0.05,
        gamma=0.5,
        c_min=0.05,
        c_max=4.0,
    )
    graph = stage_c.build_graph(data, initial_weight, initial_bias, args)
    ids = stage_c.deterministic_batch_indices(24, 4, 42, 0)
    x = data["x_encoder_train"][ids]
    y = stage_c.one_hot(data["y_encoder_train"][ids], 3)
    logits_np = np.matmul(x, initial_weight) + initial_bias
    error_np = logits_np - y
    norm_sq_np = np.sum(error_np * error_np, axis=1) * (
        np.sum(x * x, axis=1) + 1.0
    )
    factor_np = np.minimum(1.0, 1.0 / np.sqrt(norm_sq_np + 1e-5))
    clipped_error_np = error_np * factor_np[:, None]
    grad_w_np = np.matmul(x.T, clipped_error_np) / 4.0
    grad_b_np = np.sum(clipped_error_np, axis=0) / 4.0

    with tfe.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        feed = {graph["raw_x"]: x, graph["raw_y"]: y}
        fetched = session.run(
            [
                graph["fixed_update"],
                graph["logits"].reveal(),
                graph["norm_squared"].reveal(),
                graph["factor"].reveal(),
                graph["clipped_grad_w"].reveal(),
                graph["clipped_grad_b"].reveal(),
                graph["standard_noise_w"].reveal(),
                graph["standard_noise_b"].reveal(),
            ],
            feed_dict=feed,
        )
        actual_weight, actual_bias, counters = session.run(
            [
                graph["weight"].reveal(),
                graph["bias"].reveal(),
                graph["provider_counters"],
            ]
        )

    noise_w = np.asarray(fetched[6])
    noise_b = np.asarray(fetched[7])
    expected_weight = initial_weight - 0.005 * (
        grad_w_np + noise_w * 1.2 / 4.0
    )
    expected_bias = initial_bias - 0.005 * (
        grad_b_np + noise_b * 1.2 / 4.0
    )
    errors = {
        "logits": float(np.max(np.abs(np.asarray(fetched[1]) - logits_np))),
        "norm_squared": float(
            np.max(np.abs(np.asarray(fetched[2]) - norm_sq_np))
        ),
        "clip_factor": float(np.max(np.abs(np.asarray(fetched[3]) - factor_np))),
        "clipped_grad_w": float(
            np.max(np.abs(np.asarray(fetched[4]) - grad_w_np))
        ),
        "clipped_grad_b": float(
            np.max(np.abs(np.asarray(fetched[5]) - grad_b_np))
        ),
        "updated_weight": float(np.max(np.abs(actual_weight - expected_weight))),
        "updated_bias": float(np.max(np.abs(actual_bias - expected_bias))),
    }
    status = (
        "PASS"
        if max(errors.values()) <= 0.02
        and [int(value) for value in counters] == [1, 1, 1]
        and 0.8 <= float(np.std(noise_w)) <= 1.2
        else "FAIL"
    )
    payload = {
        "status": status,
        "errors": errors,
        "noise_weight_mean": float(np.mean(noise_w)),
        "noise_weight_std": float(np.std(noise_w)),
        "provider_counters": [int(value) for value in counters],
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

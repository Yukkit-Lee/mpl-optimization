#!/usr/bin/env python3
# 用途：验证 Adaptive DP-SGD 使用当前 C_t 裁剪并只更新下一步 C_{t+1}。

from __future__ import print_function

import argparse
import json
import math
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
    "new_cache/new_secure_feature/new_phase3_correctness/new_stage_b_synthetic_2048.npz",
)
INITIALIZER = os.path.join(
    BASE_DIR,
    "new_cache/new_secure_feature/new_phase3_correctness/"
    "new_stage_b_atomic_output/new_secure_stage_b_selected_cifar10_mild_seed42.npz",
)
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "new_log/new_secure_feature/json/new_phase5_secure_adaptive_correctness.json",
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
        clip_c=0.7,
        clipping_scope="global",
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
    expected_factor = np.minimum(
        1.0, args.clip_c / np.sqrt(norm_sq_np + 1e-5)
    )

    with tfe.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        feed = {graph["raw_x"]: x, graph["raw_y"]: y}
        old_c = float(
            np.asarray(session.run(graph["clip_bound"].reveal())).reshape(-1)[0]
        )
        factor_value, _, noisy_fraction, proposed_c = session.run(
            [
                graph["factor"].reveal(),
                graph["adaptive_update"],
                graph["noisy_fraction"].reveal(),
                graph["new_clip_bound"].reveal(),
            ],
            feed_dict=feed,
        )
        actual_new_c = float(
            np.asarray(session.run(graph["clip_bound"].reveal())).reshape(-1)[0]
        )
        counters = session.run(graph["provider_counters"])

    noisy_fraction = float(np.asarray(noisy_fraction).reshape(-1)[0])
    expected_new_c = old_c * math.exp(
        -args.eta_c * (noisy_fraction - args.gamma)
    )
    expected_new_c = min(args.c_max, max(args.c_min, expected_new_c))
    errors = {
        "old_c": abs(old_c - args.clip_c),
        "clip_factor_uses_old_c": float(
            np.max(np.abs(np.asarray(factor_value) - expected_factor))
        ),
        "proposed_c": abs(
            float(np.asarray(proposed_c).reshape(-1)[0]) - expected_new_c
        ),
        "committed_next_c": abs(actual_new_c - expected_new_c),
    }
    status = (
        "PASS"
        if max(errors.values()) <= 0.01
        and [int(value) for value in counters] == [1, 1, 1]
        and abs(actual_new_c - old_c) > 1e-5
        else "FAIL"
    )
    payload = {
        "status": status,
        "errors": errors,
        "old_c": old_c,
        "noisy_fraction_below": noisy_fraction,
        "expected_next_c": expected_new_c,
        "actual_next_c": actual_new_c,
        "provider_counters": [int(value) for value in counters],
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

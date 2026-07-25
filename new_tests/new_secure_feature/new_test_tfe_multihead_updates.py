#!/usr/bin/env python3
# 用途：验证同一 ABY3 图中多个 2048 维私有线性头是否在更新时发生状态串扰。

from __future__ import print_function

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

import new_stage_b_secure_init as stage_b

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
ATTEMPT_NAME = os.environ.get("NEW_DIAGNOSTIC_ATTEMPT", "attempt0")
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "new_log",
    "new_secure_feature",
    "json",
    "new_phase3_tfe_multihead_diagnostic_{}.json".format(ATTEMPT_NAME),
)


def main():
    data = stage_b.load_feature_cache(FEATURE_CACHE)
    client_indices = stage_b.make_client_splits(
        data["y_encoder_train"], 3, "mild", 42
    )
    initial_weights, initial_biases = stage_b.initialize_heads(
        3, 2048, 3, 42, 0.001
    )
    expected_weights = [value.copy() for value in initial_weights]
    expected_biases = [value.copy() for value in initial_biases]
    y_hot = stage_b.one_hot(data["y_encoder_train"], 3)

    tf.compat.v1.reset_default_graph()
    protocol = tfe.protocol.ABY3()
    with protocol:
        with tf.device(protocol.servers[0].device_name):
            raw_x = tf.compat.v1.placeholder(tf.float32, [4, 2048])
            raw_y = tf.compat.v1.placeholder(tf.float32, [4, 3])
        private_x = tfe.define_private_input("server0", lambda: raw_x)
        private_y = tfe.define_private_input("server0", lambda: raw_y)
        weights = [
            tfe.define_private_variable(value, name="weight_{}".format(i))
            for i, value in enumerate(initial_weights)
        ]
        biases = [
            tfe.define_private_variable(value, name="bias_{}".format(i))
            for i, value in enumerate(initial_biases)
        ]
        updates = []
        logits_nodes = []
        grad_w_nodes = []
        grad_b_nodes = []
        for client_id in range(3):
            logits = stage_b.secure_chunked_matmul(
                tfe, private_x, weights[client_id], 64
            ) + biases[client_id]
            error = logits - private_y
            grad_w = stage_b.secure_chunked_matmul(
                tfe, tfe.transpose(private_x), error, 64
            ) * 0.25
            grad_b = tfe.reduce_sum(error, axis=0) * 0.25
            logits_nodes.append(logits)
            grad_w_nodes.append(grad_w)
            grad_b_nodes.append(grad_b)
            updates.append(
                stage_b.secure_simultaneous_assign(
                    tf,
                    [
                        (weights[client_id], weights[client_id] - 0.005 * grad_w),
                        (biases[client_id], biases[client_id] - 0.005 * grad_b),
                    ],
                    "new_atomic_test_update_{:02d}".format(client_id),
                )
            )

        rows = []
        with tfe.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            for client_id in range(3):
                batch = stage_b.deterministic_batches(
                    client_indices[client_id], 4, 42, 1, 0, client_id
                )[0]
                x_batch = data["x_encoder_train"][batch]
                y_batch = y_hot[batch]
                logits = np.matmul(x_batch, expected_weights[client_id])
                logits += expected_biases[client_id]
                error = logits - y_batch
                expected_grad_w = np.matmul(x_batch.T, error) / 4.0
                expected_grad_b = np.sum(error, axis=0) / 4.0
                actual_logits, actual_grad_w, actual_grad_b = session.run(
                    [
                        logits_nodes[client_id].reveal(),
                        grad_w_nodes[client_id].reveal(),
                        grad_b_nodes[client_id].reveal(),
                    ],
                    feed_dict={raw_x: x_batch, raw_y: y_batch},
                )
                pre_update_errors = {
                    "logits": float(np.max(np.abs(actual_logits - logits))),
                    "grad_w": float(np.max(np.abs(actual_grad_w - expected_grad_w))),
                    "grad_b": float(np.max(np.abs(actual_grad_b - expected_grad_b))),
                }
                expected_weights[client_id] -= 0.005 * expected_grad_w
                expected_biases[client_id] -= 0.005 * expected_grad_b
                session.run(updates[client_id], feed_dict={raw_x: x_batch, raw_y: y_batch})
                actual_weights = session.run([value.reveal() for value in weights])
                errors = [
                    float(np.max(np.abs(actual - expected)))
                    for actual, expected in zip(actual_weights, expected_weights)
                ]
                rows.append(
                    {
                        "updated_client": int(client_id),
                        "pre_update_errors": pre_update_errors,
                        "all_head_weight_errors": errors,
                    }
                )
    maximum_error = max(max(row["all_head_weight_errors"]) for row in rows)
    payload = {
        "status": "PASS" if maximum_error <= 0.01 else "FAIL",
        "maximum_error": maximum_error,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

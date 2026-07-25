#!/usr/bin/env python3
# 用途：隔离验证单个客户端多次 TFE/ABY3 线性更新，排查多客户端图的数值异常。

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

import new_stage_b_secure_init as stage_b

warnings.filterwarnings("ignore")

import tensorflow as tf
import tf_encrypted as tfe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--feature_cache", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data = stage_b.load_feature_cache(args.feature_cache)
    indices = stage_b.make_client_splits(
        data["y_encoder_train"], 3, "mild", 42
    )[args.client_id]
    initial_weights, initial_biases = stage_b.initialize_heads(
        3, 2048, 3, 42, 0.001
    )
    expected_weight = initial_weights[args.client_id].copy()
    expected_bias = initial_biases[args.client_id].copy()
    y_hot = stage_b.one_hot(data["y_encoder_train"], 3)

    tf.compat.v1.reset_default_graph()
    with tfe.protocol.ABY3():
        private_x = tfe.define_private_placeholder([4, 2048])
        private_y = tfe.define_private_placeholder([4, 3])
        private_weight = tfe.define_private_variable(expected_weight)
        private_bias = tfe.define_private_variable(expected_bias)
        logits = stage_b.secure_chunked_matmul(
            tfe, private_x, private_weight, 64
        ) + private_bias
        error = logits - private_y
        grad_w = stage_b.secure_chunked_matmul(
            tfe, tfe.transpose(private_x), error, 64
        ) * 0.25
        grad_b = tfe.reduce_sum(error, axis=0) * 0.25
        update = stage_b.secure_simultaneous_assign(
            tf,
            [
                (private_weight, private_weight - 0.005 * grad_w),
                (private_bias, private_bias - 0.005 * grad_b),
            ],
            "new_atomic_repeated_test_update",
        )
        rows = []
        with tfe.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            step = 0
            for epoch in range(2):
                for batch in stage_b.deterministic_batches(
                    indices, 4, 42, 1, epoch, args.client_id
                ):
                    x_batch = data["x_encoder_train"][batch]
                    y_batch = y_hot[batch]
                    expected_logits = np.matmul(x_batch, expected_weight) + expected_bias
                    expected_error = expected_logits - y_batch
                    expected_weight -= 0.005 * np.matmul(
                        x_batch.T, expected_error
                    ) / 4.0
                    expected_bias -= 0.005 * np.sum(expected_error, axis=0) / 4.0
                    feed = {}
                    feed.update(private_x.feed(x_batch))
                    feed.update(private_y.feed(y_batch))
                    session.run(update, feed_dict=feed)
                    actual_weight, actual_bias = session.run(
                        [private_weight.reveal(), private_bias.reveal()]
                    )
                    step += 1
                    rows.append(
                        {
                            "step": step,
                            "weight_max_abs_error": float(
                                np.max(np.abs(actual_weight - expected_weight))
                            ),
                            "bias_max_abs_error": float(
                                np.max(np.abs(actual_bias - expected_bias))
                            ),
                            "actual_weight_abs_max": float(
                                np.max(np.abs(actual_weight))
                            ),
                        }
                    )
    payload = {
        "status": (
            "PASS"
            if max(row["weight_max_abs_error"] for row in rows) <= 0.01
            else "FAIL"
        ),
        "client_id": int(args.client_id),
        "num_samples": int(len(indices)),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

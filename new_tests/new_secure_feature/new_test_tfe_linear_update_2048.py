#!/usr/bin/env python3
# 用途：逐项验证 TFE/ABY3 2048 维线性头的一次私有梯度更新数值。

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

from new_stage_b_secure_init import one_hot
from new_stage_b_secure_init import secure_chunked_matmul
from new_stage_b_secure_init import secure_simultaneous_assign

warnings.filterwarnings("ignore")

import tensorflow as tf
import tf_encrypted as tfe


OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "new_log",
    "new_secure_feature",
    "json",
    "new_phase3_tfe_linear_update_test.json",
)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.reset_default_graph()
    rng = np.random.RandomState(42)
    x = rng.normal(0.0, 0.01, size=(4, 2048)).astype(np.float32)
    x /= np.maximum(1.0, np.linalg.norm(x, axis=1, keepdims=True))
    labels = np.asarray([0, 1, 2, 0], dtype=np.int64)
    y = one_hot(labels, 3)
    weight = rng.normal(0.0, 0.001, size=(2048, 3)).astype(np.float32)
    bias = np.zeros((3,), dtype=np.float32)
    learning_rate = 0.005

    expected_logits = np.matmul(x, weight) + bias
    expected_error = expected_logits - y
    expected_grad_w = np.matmul(x.T, expected_error) / float(len(x))
    expected_grad_b = np.sum(expected_error, axis=0) / float(len(x))
    expected_weight = weight - learning_rate * expected_grad_w
    expected_bias = bias - learning_rate * expected_grad_b

    with tfe.protocol.ABY3():
        private_x = tfe.define_private_placeholder([4, 2048])
        private_y = tfe.define_private_placeholder([4, 3])
        private_weight = tfe.define_private_variable(weight)
        private_bias = tfe.define_private_variable(bias)
        logits = secure_chunked_matmul(tfe, private_x, private_weight, 64)
        logits = logits + private_bias
        error = logits - private_y
        grad_w = secure_chunked_matmul(tfe, tfe.transpose(private_x), error, 64)
        grad_w = grad_w * 0.25
        grad_b = tfe.reduce_sum(error, axis=0) * 0.25
        update = secure_simultaneous_assign(
            tf,
            [
                (private_weight, private_weight - learning_rate * grad_w),
                (private_bias, private_bias - learning_rate * grad_b),
            ],
            "new_atomic_linear_test_update",
        )
        with tfe.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            feed = {}
            feed.update(private_x.feed(x))
            feed.update(private_y.feed(y))
            logits_value, grad_w_value, grad_b_value = session.run(
                [logits.reveal(), grad_w.reveal(), grad_b.reveal()], feed_dict=feed
            )
            session.run(update, feed_dict=feed)
            weight_value, bias_value = session.run(
                [private_weight.reveal(), private_bias.reveal()]
            )

    payload = {
        "status": "PASS",
        "logits_max_abs_error": float(
            np.max(np.abs(np.asarray(logits_value) - expected_logits))
        ),
        "grad_w_max_abs_error": float(
            np.max(np.abs(np.asarray(grad_w_value) - expected_grad_w))
        ),
        "grad_b_max_abs_error": float(
            np.max(np.abs(np.asarray(grad_b_value) - expected_grad_b))
        ),
        "weight_max_abs_error": float(
            np.max(np.abs(np.asarray(weight_value) - expected_weight))
        ),
        "bias_max_abs_error": float(
            np.max(np.abs(np.asarray(bias_value) - expected_bias))
        ),
    }
    if max(value for key, value in payload.items() if key.endswith("_error")) > 0.01:
        payload["status"] = "FAIL"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

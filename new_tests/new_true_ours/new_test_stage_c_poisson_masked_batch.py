#!/usr/bin/env python3
"""Real TFE check for padded Poisson batches and expected-batch normalization."""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import new_stage_c_secure_dpsgd as stage_c


def main():
    feature_dim = 4
    num_classes = 3
    graph_capacity = 4
    expected_batch = 2
    x = np.asarray(
        [
            [0.2, -0.1, 0.3, 0.4],
            [-0.4, 0.5, 0.1, -0.2],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 2, 0, 0], dtype=np.int64)
    y = stage_c.one_hot(labels, num_classes)
    mask = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    initial_weight = np.asarray(
        [
            [0.01, -0.02, 0.03],
            [-0.01, 0.04, 0.02],
            [0.02, 0.01, -0.03],
            [0.03, -0.01, 0.01],
        ],
        dtype=np.float32,
    )
    initial_bias = np.asarray([0.01, -0.02, 0.0], dtype=np.float32)
    data = {
        "x_reference": x[:2],
        "y_reference": labels[:2],
    }
    args = argparse.Namespace(
        batch_size=graph_capacity,
        expected_batch_size=expected_batch,
        feature_dim=feature_dim,
        num_classes=num_classes,
        precision_fractional=16,
        matmul_chunk_size=4,
        clip_c=1.0,
        clipping_scope="global",
        noise_multiplier=1.2,
        count_noise_multiplier=8.0,
        learning_rate=0.005,
        eta_c=0.05,
        gamma=0.5,
        c_min=0.05,
        c_max=4.0,
    )

    logits = np.matmul(x, initial_weight) + initial_bias
    error = (logits - y) * mask[:, None]
    norm_squared = np.sum(error * error, axis=1) * (
        np.sum(x * x, axis=1) + 1.0
    )
    factors = np.minimum(
        1.0, args.clip_c / np.sqrt(norm_squared + 1e-5)
    )
    clipped = error * factors[:, None]
    expected_grad_w = np.matmul(x.T, clipped) / float(expected_batch)
    expected_grad_b = np.sum(clipped, axis=0) / float(expected_batch)
    expected_fraction = float(
        np.sum((norm_squared <= args.clip_c**2) * mask)
        / float(expected_batch)
    )

    graph = stage_c.build_graph(data, initial_weight, initial_bias, args)
    tf, tfe = graph["tf"], graph["tfe"]
    with tfe.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        feed = {
            graph["raw_x"]: x,
            graph["raw_y"]: y,
            graph["raw_sample_mask"]: mask,
        }
        grad_w, grad_b, fraction, secure_norm = session.run(
            [
                graph["clipped_grad_w"].reveal(),
                graph["clipped_grad_b"].reveal(),
                graph["fraction_below"].reveal(),
                graph["norm_squared"].reveal(),
            ],
            feed_dict=feed,
        )
        empty_grad_w, empty_grad_b, empty_fraction = session.run(
            [
                graph["clipped_grad_w"].reveal(),
                graph["clipped_grad_b"].reveal(),
                graph["fraction_below"].reveal(),
            ],
            feed_dict={
                graph["raw_x"]: np.zeros_like(x),
                graph["raw_y"]: np.zeros_like(y),
                graph["raw_sample_mask"]: np.zeros_like(mask),
            },
        )

    errors = {
        "masked_grad_w": float(
            np.max(np.abs(np.asarray(grad_w) - expected_grad_w))
        ),
        "masked_grad_b": float(
            np.max(np.abs(np.asarray(grad_b) - expected_grad_b))
        ),
        "masked_fraction": abs(
            float(np.asarray(fraction).reshape(-1)[0])
            - expected_fraction
        ),
        "empty_grad_w": float(np.max(np.abs(np.asarray(empty_grad_w)))),
        "empty_grad_b": float(np.max(np.abs(np.asarray(empty_grad_b)))),
        "empty_fraction": abs(
            float(np.asarray(empty_fraction).reshape(-1)[0])
        ),
    }
    values = {
        "expected_fraction": expected_fraction,
        "actual_fraction": float(np.asarray(fraction).reshape(-1)[0]),
        "expected_norm_squared": norm_squared.tolist(),
        "actual_norm_squared": np.asarray(secure_norm).tolist(),
        "expected_grad_w": expected_grad_w.tolist(),
        "actual_grad_w": np.asarray(grad_w).tolist(),
        "expected_grad_b": expected_grad_b.tolist(),
        "actual_grad_b": np.asarray(grad_b).tolist(),
    }
    print(errors)
    print(values)
    return 0 if max(errors.values()) <= 5e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())

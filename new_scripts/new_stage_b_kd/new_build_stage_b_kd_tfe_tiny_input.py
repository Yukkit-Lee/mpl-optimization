#!/usr/bin/env python3
"""Build a small numerical gate for the real TFE/ABY3 CE+KD graph."""

import json
from pathlib import Path

import numpy as np

from new_stage_b_kd_core import (
    LOG_ROOT,
    OUTPUT_ROOT,
    atomic_json,
    atomic_npz,
    one_hot,
    sha256_file,
    softmax,
    source_commit,
)


def local_step(x_value, labels, weight, bias, lr, weight_decay):
    targets = one_hot(labels, bias.shape[0])
    probabilities = softmax(np.matmul(x_value, weight) + bias)
    error = probabilities - targets
    grad_weight = np.matmul(x_value.T, error) / float(len(labels))
    grad_weight += weight_decay * weight
    grad_bias = np.mean(error, axis=0)
    return (
        (weight - lr * grad_weight).astype(np.float32),
        (bias - lr * grad_bias).astype(np.float32),
    )


def main():
    rng = np.random.RandomState(72942)
    num_clients = 2
    feature_dim = 4
    num_classes = 3
    x_clients = [
        rng.normal(0.0, 0.2, size=(6, feature_dim)).astype(np.float32)
        for _ in range(num_clients)
    ]
    y_clients = [
        np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        np.asarray([2, 1, 0, 2, 1, 0], dtype=np.int64),
    ]
    x_anchor = rng.normal(0.0, 0.2, size=(6, feature_dim)).astype(np.float32)
    y_anchor = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    initial_weight = rng.normal(
        0.0, 0.01, size=(feature_dim, num_classes)
    ).astype(np.float32)
    initial_bias = np.zeros(num_classes, dtype=np.float32)
    local_lr = 0.05
    weight_decay = 1e-4
    local_heads = [
        local_step(
            x_clients[index],
            y_clients[index],
            initial_weight.copy(),
            initial_bias.copy(),
            local_lr,
            weight_decay,
        )
        for index in range(num_clients)
    ]
    average_weight = np.mean(
        np.stack([value[0] for value in local_heads]), axis=0
    ).astype(np.float32)
    average_bias = np.mean(
        np.stack([value[1] for value in local_heads]), axis=0
    ).astype(np.float32)
    temperature = 2.0
    teacher = np.mean(
        np.stack(
            [
                softmax(
                    np.matmul(x_anchor, value[0]) + value[1], temperature
                )
                for value in local_heads
            ]
        ),
        axis=0,
    ).astype(np.float32)
    student_probabilities = softmax(
        np.matmul(x_anchor, average_weight) + average_bias
    )
    student_temperature = softmax(
        np.matmul(x_anchor, average_weight) + average_bias, temperature
    )
    error = 0.75 * (
        student_probabilities - one_hot(y_anchor, num_classes)
    ) + 0.25 * temperature * (student_temperature - teacher)
    student_weight = (
        average_weight
        - 0.05 * np.matmul(x_anchor.T, error) / float(len(y_anchor))
    ).astype(np.float32)
    student_bias = (
        average_bias - 0.05 * np.mean(error, axis=0)
    ).astype(np.float32)
    config = {
        "schema": "new_stage_b_kd_tfe_input_config_v1",
        "dataset": "tiny_synthetic",
        "seed": 42,
        "distribution": "synthetic_balanced",
        "num_clients": num_clients,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "anchor_samples_per_class": 2,
        "local_epochs": 1,
        "local_lr": local_lr,
        "weight_decay": weight_decay,
        "reliability_rule": "uniform",
        "teacher_temperature": temperature,
        "weight_temperature": 1.0,
        "maximum_client_weight": 1.0,
        "cap_was_inactive_in_plaintext": True,
        "shrinkage_to_uniform": 0.0,
        "kd_lambda": 0.25,
        "average_regularization": 0.0,
        "student_steps": 1,
        "student_lr": 0.05,
        "fixedpoint_fractional": 16,
        "matmul_chunk_size": 4,
        "checkpoint_interval": 1,
        "source_commit": source_commit(),
        "initializer_source": "official_mtt_only",
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    arrays = {
        "x_anchor": x_anchor,
        "y_anchor": y_anchor,
        "initial_weight": initial_weight,
        "initial_bias": initial_bias,
        "plaintext_average_weight": average_weight,
        "plaintext_average_bias": average_bias,
        "plaintext_teacher_anchor": teacher,
        "plaintext_class_weights": np.full(
            (num_clients, num_classes), 1.0 / num_clients, dtype=np.float32
        ),
        "plaintext_student_weight": student_weight,
        "plaintext_student_bias": student_bias,
        "configuration_json": np.asarray(
            [json.dumps(config, sort_keys=True)]
        ),
    }
    for client_id in range(num_clients):
        arrays["x_client_{}".format(client_id)] = x_clients[client_id]
        arrays["y_client_{}".format(client_id)] = y_clients[client_id]
        arrays["client_ids_{}".format(client_id)] = np.arange(
            len(y_clients[client_id]), dtype=np.int64
        )
    output = (
        OUTPUT_ROOT
        / "tfe_inputs/tiny_synthetic/new_stage_b_kd_tfe_input_tiny_synthetic_seed42.npz"
    )
    atomic_npz(output, **arrays)
    manifest = {
        "schema": "new_stage_b_kd_tfe_tiny_input_v1",
        "status": "COMPLETE",
        "source_commit": source_commit(),
        "input_path": str(output),
        "input_sha256": sha256_file(output),
        "expected": {
            "average_weight": average_weight.tolist(),
            "average_bias": average_bias.tolist(),
            "teacher": teacher.tolist(),
            "student_weight": student_weight.tolist(),
            "student_bias": student_bias.tolist(),
        },
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    manifest_path = LOG_ROOT / "new_stage_b_kd_tfe_tiny_input.json"
    atomic_json(manifest_path, manifest)
    print("COMPLETE input={} manifest={}".format(output, manifest_path))


if __name__ == "__main__":
    main()

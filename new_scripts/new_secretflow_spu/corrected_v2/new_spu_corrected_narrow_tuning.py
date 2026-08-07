#!/usr/bin/env python3
"""Bounded development-only batch/LR screen for corrected SPU Stage B.

The computation is plaintext NumPy and uses only the already frozen training
and development feature caches.  It mirrors the corrected runner's exact
balanced split, phase-1 batch order, shared/independent initialization,
FedAvg schedule, and Ours anchor CE.  It never reads official test or locked
holdout records and is not a formal secure result.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "new_scripts/new_secretflow_spu"
for value in (str(BASE), str(ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from new_spu_pipeline_lib import (  # noqa: E402
    aggregation_weights,
    deterministic_batches,
    make_client_splits,
    mtt_anchor_ids,
    one_hot,
)
from new_spu_run_formal import load_config, load_data  # noqa: E402

OUT = ROOT / "new_log/new_secretflow_spu/corrected_v2/new_spu_corrected_narrow_tuning.json"

CANDIDATES = {
    ("cifar10", "true_ours"): [(8, 0.25)],
    ("cifar100", "true_ours"): [(32, 0.25), (32, 0.50), (32, 1.00)],
    ("tiny", "true_ours"): [(64, 0.20), (64, 0.40), (64, 0.80)],
    ("cifar10", "full_data"): [(64, 0.20), (64, 0.40), (64, 0.80)],
    ("cifar100", "full_data"): [(64, 0.20), (64, 0.40), (64, 0.80)],
    ("tiny", "full_data"): [(64, 0.10), (64, 0.20), (64, 0.40)],
}

FROZEN_FULL_BASELINES = {
    "cifar10": {"batch_size": 16, "local_lr": 0.2, "accuracy": 0.7266},
    "cifar100": {"batch_size": 16, "local_lr": 0.2, "accuracy": 0.4308},
    "tiny": {"batch_size": 16, "local_lr": 0.1, "accuracy": 0.2765},
}


def softmax(values):
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def run_candidate(dataset, method, batch_size, local_lr):
    started = time.monotonic()
    config = load_config(dataset, method)
    data = load_data(dataset, method, 42)
    classes = int(data["num_classes"])
    bcfg = config["stage_b"]
    if config["anchor_ce"]:
        local_ids, anchor_ids = mtt_anchor_ids(data["b_y"], classes)
        train_x = np.asarray(data["b_x"])[local_ids]
        train_y = np.asarray(data["b_y"])[local_ids]
        anchor_x = np.asarray(data["b_x"])[anchor_ids]
        anchor_y = np.asarray(data["b_y"])[anchor_ids]
    else:
        train_x = data["b_x"]
        train_y = np.asarray(data["b_y"])
        anchor_x = None
        anchor_y = None
    client_ids = make_client_splits(train_y, 5, "balanced", 42)
    coeffs = aggregation_weights(train_y, client_ids, bcfg["client_weighting"])
    rng = np.random.RandomState(42)
    weights = [
        rng.normal(0.0, 0.01, size=(train_x.shape[1], classes)).astype(np.float32)
        for _ in range(5)
    ]
    if bool(bcfg["shared_initialization"]):
        weights = [weights[0].copy() for _ in range(5)]
    biases = [np.zeros(classes, np.float32) for _ in range(5)]
    labels_hot = one_hot(train_y, classes)
    for epoch in range(int(bcfg["local_epochs"])):
        for client in range(5):
            for ids in deterministic_batches(client_ids[client], batch_size, 42, 1, epoch, client):
                x = np.asarray(train_x[ids], dtype=np.float32)
                error = x @ weights[client] + biases[client] - labels_hot[ids]
                grad_w = x.T @ error / float(batch_size)
                grad_w += float(bcfg["weight_decay"]) * weights[client]
                weights[client] -= float(local_lr) * grad_w
                biases[client] -= float(local_lr) * np.mean(error, axis=0)
        interval = int(bcfg["fedavg_sync_interval"])
        if interval > 0 and (epoch + 1) % interval == 0:
            avg_w = sum(float(c) * w for c, w in zip(coeffs, weights))
            avg_b = sum(float(c) * b for c, b in zip(coeffs, biases))
            weights = [avg_w.copy() for _ in range(5)]
            biases = [avg_b.copy() for _ in range(5)]
    avg_w = sum(float(c) * w for c, w in zip(coeffs, weights))
    avg_b = sum(float(c) * b for c, b in zip(coeffs, biases))
    if config["anchor_ce"]:
        reference_w, reference_b = avg_w.copy(), avg_b.copy()
        anchor_hot = one_hot(anchor_y, classes)
        for _ in range(int(config["anchor_steps"])):
            error = softmax(anchor_x @ avg_w + avg_b) - anchor_hot
            grad_w = anchor_x.T @ error / float(len(anchor_x))
            grad_b = np.mean(error, axis=0)
            mu = float(config["anchor_mu"])
            grad_w += 2.0 * mu * (avg_w - reference_w)
            grad_b += 2.0 * mu * (avg_b - reference_b)
            avg_w -= float(config["anchor_lr"]) * grad_w
            avg_b -= float(config["anchor_lr"]) * grad_b
    predictions = np.argmax(np.asarray(data["c_x_dev"]) @ avg_w + avg_b, axis=1)
    accuracy = float(np.mean(predictions == np.asarray(data["c_y_dev"])))
    return {
        "dataset": dataset,
        "method": method,
        "batch_size": int(batch_size),
        "local_lr": float(local_lr),
        "development_accuracy_after_stage_b": accuracy,
        "runtime_seconds": time.monotonic() - started,
        "formal_secure_result": False,
        "official_test_read": False,
        "locked_holdout_read": False,
    }


def main():
    rows = []
    for (dataset, method), candidates in CANDIDATES.items():
        for batch_size, local_lr in candidates:
            row = run_candidate(dataset, method, batch_size, local_lr)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    payload = {
        "schema": "new_spu_corrected_narrow_tuning_v1",
        "status": "COMPLETE",
        "scope": "development-only plaintext semantic twin",
        "candidate_policy": "pre-registered batch/LR pairs only",
        "frozen_full_baselines_not_rerun": FROZEN_FULL_BASELINES,
        "results": rows,
        "official_test_read": False,
        "locked_holdout_read": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUT)


if __name__ == "__main__":
    main()

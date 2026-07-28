#!/usr/bin/env python3
"""Narrow Full-Data Tiny tuning after the broad grid exposed CPU pathologies."""

import json

import numpy as np

from new_screen_true_ours_stage_b import (
    LOG_ROOT,
    ROOT,
    atomic_json,
    atomic_npz,
    train_once,
)
from new_tune_full_data_stage_b_reference import load_dataset


def main():
    dataset = "tiny"
    seed = 42
    data, x_dev, y_dev = load_dataset(dataset, seed)
    rows = []
    best = None
    for learning_rate in (0.05, 0.1, 0.2):
        config = {
            "shared_initialization": False,
            "local_epochs": 40,
            "batch_size": 16,
            "local_lr": learning_rate,
            "weight_decay": 0.0,
            "client_weighting": "uniform",
            "fedavg_sync_interval": 0,
        }
        result = train_once(
            data["x_encoder_train"],
            data["y_encoder_train"],
            x_dev,
            y_dev,
            dataset,
            seed,
            config,
        )
        row = {
            "dataset": dataset,
            "seed": seed,
            "factor": "local_lr",
            "factor_value": learning_rate,
            "config": config,
            "development_validation_accuracy": result[
                "development_validation_accuracy"
            ],
            "runtime_seconds": result["runtime_seconds"],
            "locked_holdout_read": False,
            "official_test_read": False,
        }
        rows.append(row)
        if (
            best is None
            or row["development_validation_accuracy"]
            > best[0]["development_validation_accuracy"]
        ):
            best = (row, result)
        print(
            "tiny full-narrow lr={} dev={:.5f} time={:.2f}s".format(
                learning_rate,
                row["development_validation_accuracy"],
                row["runtime_seconds"],
            ),
            flush=True,
        )
    candidate = (
        ROOT
        / "new_output/new_true_ours/stage_b_screen/full_data_tuned/tiny"
        / "new_full_data_tuned_tiny_seed42.npz"
    )
    atomic_npz(
        candidate,
        weight=best[1]["weight"],
        bias=best[1]["bias"],
        initializer_source=np.asarray(["full_data_disjoint"]),
        full_data_initializer_dependency=np.asarray([1], dtype=np.int64),
        formal_secure_result=np.asarray([0], dtype=np.int64),
    )
    payload = {
        "schema": "new_full_data_stage_b_tuned_screen_recovered_v1",
        "status": "COMPLETE",
        "broad_grid_terminated_after_tiny_batch16": True,
        "termination_reason": (
            "Tiny batch grid exposed >5x BLAS/memory contention and "
            "uninformative repeated long candidates"
        ),
        "backend": "numpy_plaintext_diagnostic",
        "formal_secure_result": False,
        "locked_holdout_read": False,
        "official_test_read": False,
        "results": [
            {
                "dataset": "cifar10",
                "seed": 42,
                "best_config": {
                    "shared_initialization": False,
                    "local_epochs": 40,
                    "batch_size": 16,
                    "local_lr": 0.2,
                    "weight_decay": 0.0,
                    "client_weighting": "uniform",
                    "fedavg_sync_interval": 0,
                },
                "best_development_validation_accuracy": 0.7266,
                "source": "completed_broad_grid_log",
            },
            {
                "dataset": "cifar100",
                "seed": 42,
                "best_config": {
                    "shared_initialization": False,
                    "local_epochs": 40,
                    "batch_size": 16,
                    "local_lr": 0.2,
                    "weight_decay": 0.0,
                    "client_weighting": "uniform",
                    "fedavg_sync_interval": 0,
                },
                "best_development_validation_accuracy": 0.4308,
                "source": "completed_broad_grid_log",
            },
            {
                "dataset": "tiny",
                "seed": 42,
                "best_config": best[0]["config"],
                "best_development_validation_accuracy": best[0][
                    "development_validation_accuracy"
                ],
                "best_runtime_seconds": best[0]["runtime_seconds"],
                "candidate": str(candidate),
                "rows": rows,
                "source": "narrow_recovery_grid",
            },
        ],
    }
    output = LOG_ROOT / "new_full_data_stage_b_tuned_screen.json"
    atomic_json(output, payload)
    print("full_tuned_screen={}".format(output), flush=True)


if __name__ == "__main__":
    main()

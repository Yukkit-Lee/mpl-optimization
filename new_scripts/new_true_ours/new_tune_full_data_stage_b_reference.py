#!/usr/bin/env python3
"""Fair development-only one-factor tuning for the disjoint Full-Data reference."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from new_screen_true_ours_stage_b import (
    DATASETS,
    LOG_ROOT,
    ROOT,
    atomic_json,
    atomic_npz,
    train_once,
)

sys.path.insert(0, str(ROOT / "new_scripts"))
from new_full_feature_memmap_loader import load_feature_cache_memmap


BASE_EPOCHS = {"cifar10": 20, "cifar100": 30, "tiny": 20}
OUTPUT_ROOT = (
    ROOT / "new_output/new_true_ours/stage_b_screen/full_data_tuned"
)


def choose(candidates, factor, tolerance=0.001):
    maximum = max(
        row["development_validation_accuracy"] for row in candidates
    )
    eligible = [
        row
        for row in candidates
        if row["development_validation_accuracy"] >= maximum - tolerance
    ]
    if factor == "local_epochs":
        return min(
            eligible,
            key=lambda row: (
                int(row["factor_value"]),
                -row["development_validation_accuracy"],
            ),
        )
    if factor == "batch_size":
        return max(
            eligible,
            key=lambda row: (
                int(row["factor_value"]),
                row["development_validation_accuracy"],
            ),
        )
    return max(
        candidates,
        key=lambda row: row["development_validation_accuracy"],
    )


def load_dataset(dataset, seed):
    metadata_path = (
        ROOT
        / "new_cache/new_true_ours/full_data_stage_b"
        / dataset
        / "new_full_stage_b_{}_seed{}.npz".format(dataset, seed)
    )
    split_path = (
        ROOT
        / "new_cache/new_true_ours/splits"
        / dataset
        / "new_true_ours_split_{}_seed{}.npz".format(dataset, seed)
    )
    data = load_feature_cache_memmap(str(metadata_path))
    with np.load(str(split_path), allow_pickle=False) as split:
        dev_ids = np.asarray(
            split["development_validation"], dtype=np.int64
        )
    full_dir = (
        ROOT
        / "new_cache/new_public_mtt_mainline/new_full_features"
        / dataset
        / "new_seed{}".format(seed)
    )
    x_full = np.load(
        str(full_dir / "new_full_train_features.npy"), mmap_mode="r"
    )
    y_full = np.load(
        str(full_dir / "new_full_train_labels.npy"), mmap_mode="r"
    )
    return (
        data,
        np.asarray(x_full[dev_ids], dtype=np.float32),
        np.asarray(y_full[dev_ids], dtype=np.int64),
    )


def run_dataset(dataset, seed):
    data, x_dev, y_dev = load_dataset(dataset, seed)
    base_epochs = BASE_EPOCHS[dataset]
    current = {
        "shared_initialization": False,
        "local_epochs": base_epochs,
        "batch_size": 32,
        "local_lr": 0.05,
        "weight_decay": 0.0,
        "client_weighting": "uniform",
        "fedavg_sync_interval": 0,
    }
    epoch_values = sorted(set([base_epochs, 40]))
    factors = [
        ("local_epochs", epoch_values),
        ("batch_size", [16, 32, 64]),
        ("local_lr", [0.025, 0.05, 0.1, 0.2]),
        ("weight_decay", [0.0, 1e-4, 1e-3]),
    ]
    rows = []
    for factor, values in factors:
        candidates = []
        for value in values:
            config = dict(current)
            config[factor] = value
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
                "seed": int(seed),
                "factor": factor,
                "factor_value": value,
                "config": config,
                "development_validation_accuracy": result[
                    "development_validation_accuracy"
                ],
                "runtime_seconds": result["runtime_seconds"],
                "locked_holdout_read": False,
                "official_test_read": False,
            }
            rows.append(row)
            candidates.append(row)
            print(
                "{} full-tune factor={} value={} dev={:.5f} time={:.2f}s".format(
                    dataset,
                    factor,
                    value,
                    row["development_validation_accuracy"],
                    row["runtime_seconds"],
                ),
                flush=True,
            )
        selected = choose(candidates, factor)
        current = dict(selected["config"])
        print(
            "{} FULL KEEP {}={} dev={:.5f}".format(
                dataset,
                factor,
                current[factor],
                selected["development_validation_accuracy"],
            ),
            flush=True,
        )
    final = train_once(
        data["x_encoder_train"],
        data["y_encoder_train"],
        x_dev,
        y_dev,
        dataset,
        seed,
        current,
    )
    candidate = (
        OUTPUT_ROOT
        / dataset
        / "new_full_data_tuned_{}_seed{}.npz".format(dataset, seed)
    )
    atomic_npz(
        candidate,
        weight=final["weight"],
        bias=final["bias"],
        initializer_source=np.asarray(["full_data_disjoint"]),
        full_data_initializer_dependency=np.asarray([1], dtype=np.int64),
        formal_secure_result=np.asarray([0], dtype=np.int64),
    )
    return {
        "dataset": dataset,
        "seed": int(seed),
        "best_config": current,
        "best_development_validation_accuracy": final[
            "development_validation_accuracy"
        ],
        "best_runtime_seconds": final["runtime_seconds"],
        "candidate": str(candidate),
        "rows": rows,
        "backend": "numpy_plaintext_diagnostic",
        "formal_secure_result": False,
        "locked_holdout_read": False,
        "official_test_read": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_path = (
        LOG_ROOT / "new_full_data_stage_b_tuned_screen.json"
    )
    existing = []
    if output_path.is_file():
        existing = json.loads(
            output_path.read_text(encoding="utf-8")
        ).get("results", [])
    existing = [
        row
        for row in existing
        if not (
            row["dataset"] in args.datasets
            and int(row["seed"]) == int(args.seed)
        )
    ]
    results = existing + [
        run_dataset(dataset, args.seed) for dataset in args.datasets
    ]
    atomic_json(
        output_path,
        {
            "schema": "new_full_data_stage_b_tuned_screen_v1",
            "status": "COMPLETE",
            "selection_tolerance_accuracy": 0.001,
            "backend": "numpy_plaintext_diagnostic",
            "formal_secure_result": False,
            "locked_holdout_read": False,
            "official_test_read": False,
            "results": results,
        },
    )
    print(
        "full_tuned_screen={} datasets={}".format(
            output_path, len(results)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

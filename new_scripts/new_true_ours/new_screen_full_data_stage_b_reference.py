#!/usr/bin/env python3
"""Development-only plaintext reference for the disjoint Full-Data Stage B."""

import argparse
import json
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

import sys

sys.path.insert(0, str(ROOT / "new_scripts"))
from new_full_feature_memmap_loader import load_feature_cache_memmap


EPOCHS = {"cifar10": 20, "cifar100": 30, "tiny": 20}
OUTPUT_ROOT = ROOT / "new_output/new_true_ours/stage_b_screen/full_data_reference"


def run_dataset(dataset, seed):
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
    x_dev = np.asarray(x_full[dev_ids], dtype=np.float32)
    y_dev = np.asarray(y_full[dev_ids], dtype=np.int64)
    config = {
        "shared_initialization": False,
        "local_epochs": EPOCHS[dataset],
        "batch_size": 32,
        "local_lr": 0.05,
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
    candidate = (
        OUTPUT_ROOT
        / dataset
        / "new_full_data_reference_{}_seed{}.npz".format(dataset, seed)
    )
    atomic_npz(
        candidate,
        weight=result["weight"],
        bias=result["bias"],
        initializer_source=np.asarray(["full_data_disjoint"]),
        full_data_initializer_dependency=np.asarray([1], dtype=np.int64),
        formal_secure_result=np.asarray([0], dtype=np.int64),
    )
    return {
        "dataset": dataset,
        "seed": int(seed),
        "configuration": config,
        "development_validation_accuracy": result[
            "development_validation_accuracy"
        ],
        "runtime_seconds": result["runtime_seconds"],
        "client_counts": result["client_counts"],
        "candidate": str(candidate),
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
        LOG_ROOT / "new_full_data_stage_b_reference_screen.json"
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
            "schema": "new_full_data_stage_b_reference_screen_v1",
            "status": "COMPLETE",
            "purpose": "development-only retention target reference",
            "backend": "numpy_plaintext_diagnostic",
            "formal_secure_result": False,
            "locked_holdout_read": False,
            "official_test_read": False,
            "results": results,
        },
    )
    for row in results:
        print(
            "{} seed={} full-reference dev={:.5f} time={:.2f}s".format(
                row["dataset"],
                row["seed"],
                row["development_validation_accuracy"],
                row["runtime_seconds"],
            )
        )


if __name__ == "__main__":
    main()

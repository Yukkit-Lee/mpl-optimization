#!/usr/bin/env python3
"""Phase-4 development-only rescue screen for MTT-only Stage B."""

import argparse
import json

import numpy as np

from new_screen_true_ours_stage_b import (
    DATASETS,
    LOG_ROOT,
    ROOT,
    atomic_json,
    atomic_npz,
    load_assets,
    train_once,
)


OUTPUT_ROOT = ROOT / "new_output/new_true_ours/stage_b_screen/rescue"


def select_candidate(candidates, factor, tolerance=0.001):
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
    if factor == "fedavg_sync_interval":
        return max(
            eligible,
            key=lambda row: (
                -(
                    0
                    if int(row["factor_value"]) == 0
                    else int(row["config"]["local_epochs"])
                    // int(row["factor_value"])
                ),
                row["development_validation_accuracy"],
            ),
        )
    return max(
        candidates,
        key=lambda row: row["development_validation_accuracy"],
    )


def run_dataset(dataset, seed, base_config):
    _, _, x_train, y_train, x_dev, y_dev = load_assets(dataset, seed)
    current = dict(base_config)
    batch_values = [4, 8, 16] if dataset == "cifar10" else [8]
    lr_values = (
        [0.2, 0.25, 0.3, 0.4]
        if dataset == "cifar10"
        else [0.15, 0.2, 0.25]
    )
    factors = [
        ("local_epochs", [40, 60, 80]),
        ("batch_size", batch_values),
        ("local_lr", lr_values),
        ("weight_decay", [0.0, 1e-4, 1e-3]),
        ("fedavg_sync_interval", [0, 1, 5, 10, 20]),
    ]
    rows = []
    for factor, values in factors:
        candidates = []
        for value in values:
            config = dict(current)
            config[factor] = value
            result = train_once(
                x_train, y_train, x_dev, y_dev, dataset, seed, config
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
                "{} rescue factor={} value={} dev={:.5f} time={:.2f}s".format(
                    dataset,
                    factor,
                    value,
                    row["development_validation_accuracy"],
                    row["runtime_seconds"],
                ),
                flush=True,
            )
        selected = select_candidate(candidates, factor)
        current = dict(selected["config"])
        print(
            "{} RESCUE KEEP {}={} dev={:.5f}".format(
                dataset,
                factor,
                current[factor],
                selected["development_validation_accuracy"],
            ),
            flush=True,
        )
    final = train_once(
        x_train, y_train, x_dev, y_dev, dataset, seed, current
    )
    centralized_config = dict(current)
    centralized_config["num_clients"] = 1
    centralized_config["fedavg_sync_interval"] = 0
    centralized = train_once(
        x_train,
        y_train,
        x_dev,
        y_dev,
        dataset,
        seed,
        centralized_config,
    )
    candidate = (
        OUTPUT_ROOT
        / dataset
        / "new_true_ours_stage_b_rescue_{}_seed{}.npz".format(
            dataset, seed
        )
    )
    atomic_npz(
        candidate,
        weight=final["weight"],
        bias=final["bias"],
        initializer_source=np.asarray(["official_mtt_only"]),
        full_data_initializer_dependency=np.asarray([0], dtype=np.int64),
        formal_secure_result=np.asarray([0], dtype=np.int64),
    )
    return {
        "dataset": dataset,
        "seed": int(seed),
        "base_config": base_config,
        "best_config": current,
        "best_development_validation_accuracy": final[
            "development_validation_accuracy"
        ],
        "centralized_diagnostic_config": centralized_config,
        "centralized_development_validation_accuracy": centralized[
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
    base = json.loads(
        (
            LOG_ROOT / "new_true_ours_stage_b_fedavg_screen.json"
        ).read_text(encoding="utf-8")
    )
    configs = {
        (row["dataset"], int(row["seed"])): row["best_config"]
        for row in base["results"]
    }
    output_path = LOG_ROOT / "new_true_ours_stage_b_rescue_screen.json"
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
        run_dataset(
            dataset,
            args.seed,
            configs[(dataset, int(args.seed))],
        )
        for dataset in args.datasets
    ]
    atomic_json(
        output_path,
        {
            "schema": "new_true_ours_stage_b_rescue_screen_v1",
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
        "rescue_screen={} datasets={}".format(output_path, len(results)),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Follow-up one-factor screen for periodic secure FedAvg synchronization."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from new_screen_true_ours_stage_b import (
    DATASETS,
    ROOT,
    atomic_json,
    atomic_npz,
    load_assets,
    sha256_file,
    train_once,
)


LOG_ROOT = ROOT / "new_log/new_true_ours"
TABLE_ROOT = ROOT / "new_expTab/new_true_ours"
OUTPUT_ROOT = ROOT / "new_output/new_true_ours/stage_b_screen"


def choose_accuracy_time_pareto(candidates, factor, tolerance=0.001):
    best_accuracy = max(
        item["development_validation_accuracy"] for item in candidates
    )
    eligible = [
        item
        for item in candidates
        if item["development_validation_accuracy"] >= best_accuracy - tolerance
    ]
    if factor == "shared_initialization":
        return max(
            candidates,
            key=lambda item: item["development_validation_accuracy"],
        )
    if factor == "fedavg_sync_interval":
        return max(
            eligible,
            key=lambda item: (
                -(
                    0
                    if int(item["factor_value"]) == 0
                    else int(item["config"]["local_epochs"])
                    // int(item["factor_value"])
                ),
                item["development_validation_accuracy"],
            ),
        )
    if factor == "batch_size":
        return max(
            eligible,
            key=lambda item: (
                int(item["factor_value"]),
                item["development_validation_accuracy"],
            ),
        )
    return max(
        candidates,
        key=lambda item: item["development_validation_accuracy"],
    )


def run_dataset(dataset, seed, base_config):
    cache_path, split_path, x_train, y_train, x_dev, y_dev = load_assets(
        dataset, seed
    )
    current = dict(base_config)
    current.setdefault("fedavg_sync_interval", 0)
    factors = [
        ("fedavg_sync_interval", [0, 1, 5, 10]),
        ("shared_initialization", [False, True]),
        ("batch_size", [8, 16, 32]),
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
                "seed": seed,
                "factor": factor,
                "factor_value": value,
                "config": config,
                "development_validation_accuracy": result[
                    "development_validation_accuracy"
                ],
                "calibration_ece": result["calibration_ece"],
                "mean_inter_client_distance": float(
                    np.mean(result["inter_client_distance_to_average"])
                ),
                "runtime_seconds": result["runtime_seconds"],
                "locked_holdout_read": False,
                "official_test_read": False,
            }
            rows.append(row)
            candidates.append(row)
            print(
                "{} fedavg-followup factor={} value={} dev={:.5f} time={:.2f}s".format(
                    dataset,
                    factor,
                    value,
                    row["development_validation_accuracy"],
                    row["runtime_seconds"],
                ),
                flush=True,
            )
        chosen = choose_accuracy_time_pareto(candidates, factor)
        current = dict(chosen["config"])
        print(
            "{} PARETO KEEP {}={} dev={:.5f} time={:.2f}s".format(
                dataset,
                factor,
                current[factor],
                chosen["development_validation_accuracy"],
                chosen["runtime_seconds"],
            ),
            flush=True,
        )
    final = train_once(
        x_train, y_train, x_dev, y_dev, dataset, seed, current
    )
    output = (
        OUTPUT_ROOT / dataset
        / "new_true_ours_stage_b_fedavg_best_{}_seed{}.npz".format(
            dataset, seed
        )
    )
    atomic_npz(
        output,
        weight=final["weight"],
        bias=final["bias"],
        initializer_source=np.asarray(["official_mtt_only"]),
        full_data_initializer_dependency=np.asarray([0], dtype=np.int64),
        formal_secure_result=np.asarray([0], dtype=np.int64),
        feature_cache_sha256=np.asarray([sha256_file(cache_path)]),
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "base_config": base_config,
        "best_config": current,
        "best_development_validation_accuracy": final[
            "development_validation_accuracy"
        ],
        "best_runtime_seconds": final["runtime_seconds"],
        "best_screen_initializer": str(output),
        "selection_rule": (
            "lowest-cost candidate within 0.1pp of factor-local best accuracy"
        ),
        "formal_secure_result": False,
        "locked_holdout_read": False,
        "official_test_read": False,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    base_payload = json.loads(
        (LOG_ROOT / "new_true_ours_stage_b_screen.json").read_text(
            encoding="utf-8"
        )
    )
    base_by_key = {
        (row["dataset"], int(row["seed"])): row["best_config"]
        for row in base_payload["results"]
    }
    output_path = LOG_ROOT / "new_true_ours_stage_b_fedavg_screen.json"
    existing = []
    if output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        existing = [
            row
            for row in payload.get("results", [])
            if not (
                row["dataset"] in args.datasets
                and int(row["seed"]) == int(args.seed)
            )
        ]
    fresh = [
        run_dataset(
            dataset,
            args.seed,
            base_by_key[(dataset, int(args.seed))],
        )
        for dataset in args.datasets
    ]
    results = existing + fresh
    payload = {
        "schema": "new_true_ours_stage_b_fedavg_screen_v1",
        "status": "COMPLETE",
        "backend": "numpy_plaintext_diagnostic",
        "not_formal_secure_result": True,
        "locked_holdout_read": False,
        "official_test_read": False,
        "results": results,
    }
    atomic_json(output_path, payload)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_ROOT / "new_true_ours_stage_b_fedavg_screen.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "dataset",
                "seed",
                "factor",
                "factor_value",
                "development_validation_accuracy",
                "runtime_seconds",
            ]
        )
        for result in results:
            for row in result["rows"]:
                writer.writerow(
                    [
                        row["dataset"],
                        row["seed"],
                        row["factor"],
                        row["factor_value"],
                        "{:.8f}".format(
                            row["development_validation_accuracy"]
                        ),
                        "{:.6f}".format(row["runtime_seconds"]),
                    ]
                )
    print(
        "fedavg_screen={} datasets={} locked_holdout_read=False".format(
            csv_path, len(results)
        )
    )


if __name__ == "__main__":
    main()

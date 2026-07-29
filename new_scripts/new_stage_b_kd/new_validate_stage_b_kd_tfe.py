#!/usr/bin/env python3
"""Validate completed seed42 TFE/ABY3 models against frozen P1 references."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from new_stage_b_kd_core import (
    DATASETS,
    LOG_ROOT,
    OUTPUT_ROOT,
    SEED,
    TABLE_ROOT,
    atomic_json,
    evaluate_head,
    load_dataset_assets,
    sha256_file,
    source_commit,
)


OUTPUT_JSON = LOG_ROOT / "new_stage_b_kd_tfe_validation.json"
OUTPUT_CSV = TABLE_ROOT / "new_stage_b_kd_tfe_validation.csv"


def max_abs(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def validate_one(dataset):
    input_path = (
        OUTPUT_ROOT
        / "tfe_inputs"
        / dataset
        / "new_stage_b_kd_tfe_input_{}_seed42.npz".format(dataset)
    )
    output_dir = OUTPUT_ROOT / "tfe_validation" / dataset / "seed42"
    report_path = output_dir / (
        "new_stage_b_kd_tfe_report_{}_seed42.json".format(dataset)
    )
    model_path = output_dir / (
        "new_stage_b_kd_tfe_model_{}_seed42.npz".format(dataset)
    )
    if not report_path.is_file() or not model_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE":
        return None
    with np.load(str(input_path), allow_pickle=False) as loaded:
        source = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(str(model_path), allow_pickle=False) as loaded:
        model = {name: np.asarray(loaded[name]) for name in loaded.files}
    finite = all(np.isfinite(value).all() for value in model.values() if value.dtype.kind in "fc")
    if dataset == "tiny_synthetic":
        comparisons = {
            "average_weight_max_abs": max_abs(
                model["average_weight"], source["plaintext_average_weight"]
            ),
            "average_bias_max_abs": max_abs(
                model["average_bias"], source["plaintext_average_bias"]
            ),
            "teacher_max_abs": max_abs(
                model["teacher_anchor_probabilities"],
                source["plaintext_teacher_anchor"],
            ),
            "student_weight_max_abs": max_abs(
                model["student_weight"], source["plaintext_student_weight"]
            ),
            "student_bias_max_abs": max_abs(
                model["student_bias"], source["plaintext_student_bias"]
            ),
        }
        correctness_pass = (
            finite
            and comparisons["average_weight_max_abs"] <= 5e-3
            and comparisons["average_bias_max_abs"] <= 5e-3
            and comparisons["teacher_max_abs"] <= 5e-2
            and comparisons["student_weight_max_abs"] <= 1e-2
            and comparisons["student_bias_max_abs"] <= 1e-2
        )
        return {
            "dataset": dataset,
            "seed": SEED,
            "backend": "TFE_ABY3",
            "finite": finite,
            "comparisons": comparisons,
            "tfe_average_accuracy": None,
            "tfe_student_accuracy": None,
            "tfe_gain": None,
            "plaintext_gain": None,
            "gain_retention": None,
            "utility_gate": correctness_pass,
            "correctness_pass": correctness_pass,
            "total_secure_seconds": report["total_secure_seconds"],
            "peak_rss_gib": report["peak_rss_gib"],
            "report_path": str(report_path),
            "report_sha256": sha256_file(report_path),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
        }

    assets = load_dataset_assets(dataset, SEED)
    tfe_average = evaluate_head(
        assets.x_dev,
        assets.y_dev,
        model["average_weight"],
        model["average_bias"],
        assets.num_classes,
    )
    tfe_student = evaluate_head(
        assets.x_dev,
        assets.y_dev,
        model["student_weight"],
        model["student_bias"],
        assets.num_classes,
    )
    p1_average = evaluate_head(
        assets.x_dev,
        assets.y_dev,
        source["plaintext_average_weight"],
        source["plaintext_average_bias"],
        assets.num_classes,
    )
    p1_student = evaluate_head(
        assets.x_dev,
        assets.y_dev,
        source["plaintext_student_weight"],
        source["plaintext_student_bias"],
        assets.num_classes,
    )
    tfe_gain = tfe_student["accuracy"] - tfe_average["accuracy"]
    plaintext_gain = p1_student["accuracy"] - p1_average["accuracy"]
    gain_retention = None if plaintext_gain <= 0 else tfe_gain / plaintext_gain
    comparisons = {
        "average_weight_max_abs": max_abs(
            model["average_weight"], source["plaintext_average_weight"]
        ),
        "average_bias_max_abs": max_abs(
            model["average_bias"], source["plaintext_average_bias"]
        ),
        "teacher_max_abs": max_abs(
            model["teacher_anchor_probabilities"],
            source["plaintext_teacher_anchor"],
        ),
        "class_weights_max_abs": max_abs(
            model["class_weights"], source["plaintext_class_weights"]
        ),
        "student_weight_max_abs": max_abs(
            model["student_weight"], source["plaintext_student_weight"]
        ),
        "student_bias_max_abs": max_abs(
            model["student_bias"], source["plaintext_student_bias"]
        ),
    }
    utility_gate = tfe_gain >= 0.008 or (
        gain_retention is not None and gain_retention >= 0.8
    )
    correctness_pass = (
        finite
        and report["backend"] == "TFE"
        and report["protocol"] == "ABY3"
        and report["no_plaintext_fallback"]
        and report["initializer_source"] == "official_mtt_only"
        and not report["locked_holdout_read"]
        and not report["official_test_read"]
        and utility_gate
    )
    return {
        "dataset": dataset,
        "seed": SEED,
        "backend": "TFE_ABY3",
        "finite": finite,
        "comparisons": comparisons,
        "tfe_average_accuracy": tfe_average["accuracy"],
        "tfe_student_accuracy": tfe_student["accuracy"],
        "tfe_gain": tfe_gain,
        "plaintext_average_accuracy": p1_average["accuracy"],
        "plaintext_student_accuracy": p1_student["accuracy"],
        "plaintext_gain": plaintext_gain,
        "gain_retention": gain_retention,
        "utility_gate": utility_gate,
        "correctness_pass": correctness_pass,
        "total_secure_seconds": report["total_secure_seconds"],
        "local_secure_seconds": report["local_secure_seconds"],
        "student_secure_seconds": report["student_secure_seconds"],
        "peak_rss_gib": report["peak_rss_gib"],
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require",
        choices=["tiny_synthetic", "cifar10", "cifar100", "all"],
        default=None,
    )
    args = parser.parse_args()
    datasets = ("tiny_synthetic",) + DATASETS
    rows = [row for row in (validate_one(ds) for ds in datasets) if row is not None]
    payload = {
        "schema": "new_stage_b_kd_tfe_validation_v1",
        "status": "COMPLETE" if rows else "NO_COMPLETE_ROWS",
        "source_commit": source_commit(),
        "seed_policy": [SEED],
        "rows": rows,
        "locked_holdout_read": False,
        "official_test_read": False,
        "full_data_initializer_loaded": False,
    }
    atomic_json(OUTPUT_JSON, payload)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "seed",
        "backend",
        "finite",
        "tfe_average_accuracy",
        "tfe_student_accuracy",
        "tfe_gain",
        "plaintext_gain",
        "gain_retention",
        "utility_gate",
        "correctness_pass",
        "local_secure_seconds",
        "student_secure_seconds",
        "total_secure_seconds",
        "peak_rss_gib",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})
    if args.require:
        required = set(DATASETS) if args.require == "all" else {args.require}
        available = {
            row["dataset"] for row in rows if row["correctness_pass"]
        }
        missing = required - available
        if missing:
            raise SystemExit(
                "TFE_VALIDATION_FAILED required={} available={}".format(
                    sorted(required), sorted(available)
                )
            )
    print(
        "TFE_VALIDATION rows={} passed={}".format(
            len(rows), [row["dataset"] for row in rows if row["correctness_pass"]]
        )
    )


if __name__ == "__main__":
    main()

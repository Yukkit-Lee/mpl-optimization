#!/usr/bin/env python3
"""Validate frozen C1/C4/C5 TFE outputs against plaintext references."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KD_DIR = ROOT / "new_scripts/new_stage_b_kd"
for path in (HERE, KD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from new_stage_b_kd_core import (  # noqa: E402
    DATASETS,
    SEED,
    atomic_json,
    load_dataset_assets,
    sha256_file,
)
from new_stage_b_anchor_postagg_core import (  # noqa: E402
    classification_metrics,
)


LOG_ROOT = ROOT / "new_log/new_stage_b_anchor_postagg"
TABLE_ROOT = ROOT / "new_expTab/new_stage_b_anchor_postagg"
OUTPUT_ROOT = ROOT / "new_output/new_stage_b_anchor_postagg"
OUTPUT_JSON = LOG_ROOT / "new_stage_b_anchor_postagg_tfe_validation.json"
OUTPUT_CSV = TABLE_ROOT / "new_stage_b_anchor_postagg_tfe_results.csv"


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def validate_one(dataset: str):
    input_path = (
        OUTPUT_ROOT
        / "tfe_inputs"
        / dataset
        / "new_stage_b_anchor_postagg_tfe_input_{}_seed42.npz".format(
            dataset
        )
    )
    output_dir = OUTPUT_ROOT / "tfe_validation" / dataset / "seed42"
    report_path = output_dir / (
        "new_stage_b_anchor_postagg_tfe_report_{}_seed42.json".format(
            dataset
        )
    )
    model_path = output_dir / (
        "new_stage_b_anchor_postagg_tfe_model_{}_seed42.npz".format(
            dataset
        )
    )
    if not report_path.is_file() or not model_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE":
        return None
    with np.load(str(input_path), allow_pickle=False) as loaded:
        frozen = {name: np.asarray(loaded[name]) for name in loaded.files}
    with np.load(str(model_path), allow_pickle=False) as loaded:
        model = {name: np.asarray(loaded[name]) for name in loaded.files}
    assets = load_dataset_assets(dataset, SEED)
    keys = {
        "C1": ("c1_weight", "c1_bias"),
        "C4": ("average_weight", "average_bias"),
        "C5": ("c5_weight", "c5_bias"),
    }
    reference_keys = {
        "C1": ("plaintext_c1_weight", "plaintext_c1_bias"),
        "C4": ("plaintext_average_weight", "plaintext_average_bias"),
        "C5": ("plaintext_c5_weight", "plaintext_c5_bias"),
    }
    tfe_accuracy = {}
    plaintext_accuracy = {}
    parameter_max_abs = {}
    for method in ("C1", "C4", "C5"):
        weight_key, bias_key = keys[method]
        ref_weight, ref_bias = reference_keys[method]
        tfe_accuracy[method] = classification_metrics(
            assets.x_dev,
            assets.y_dev,
            model[weight_key],
            model[bias_key],
            assets.num_classes,
        )["accuracy"]
        plaintext_accuracy[method] = classification_metrics(
            assets.x_dev,
            assets.y_dev,
            frozen[ref_weight],
            frozen[ref_bias],
            assets.num_classes,
        )["accuracy"]
        parameter_max_abs[method] = max(
            max_abs(model[weight_key], frozen[ref_weight]),
            max_abs(model[bias_key], frozen[ref_bias]),
        )
    finite = all(
        np.isfinite(value).all()
        for value in model.values()
        if value.dtype.kind in "fc"
    )
    plaintext_delta = plaintext_accuracy["C5"] - plaintext_accuracy["C1"]
    tfe_delta = tfe_accuracy["C5"] - tfe_accuracy["C1"]
    retention = (
        None if plaintext_delta <= 0.0 else tfe_delta / plaintext_delta
    )
    gaps = {
        method: abs(tfe_accuracy[method] - plaintext_accuracy[method])
        for method in ("C1", "C4", "C5")
    }
    utility_gate = (
        retention is not None
        and retention >= 0.8
        and tfe_accuracy["C5"] > tfe_accuracy["C4"]
    )
    correctness_pass = (
        finite
        and report["backend"] == "TFE"
        and report["protocol"] == "ABY3"
        and report["no_plaintext_fallback"]
        and report["secure_average_executed_under_secret_sharing"]
        and report["anchor_data_and_labels_secret_shared"]
        and report["initializer_source"] == "official_mtt_only"
        and not report["test_read"]
        and not report["locked_holdout_read"]
        and max(gaps.values()) <= 0.005
        and max(parameter_max_abs.values()) <= 0.01
        and utility_gate
    )
    return {
        "dataset": dataset,
        "seed": SEED,
        "distribution": "severe",
        "backend": "TFE_ABY3",
        "finite": finite,
        "plaintext_c1_accuracy": plaintext_accuracy["C1"],
        "plaintext_c4_accuracy": plaintext_accuracy["C4"],
        "plaintext_c5_accuracy": plaintext_accuracy["C5"],
        "tfe_c1_accuracy": tfe_accuracy["C1"],
        "tfe_c4_accuracy": tfe_accuracy["C4"],
        "tfe_c5_accuracy": tfe_accuracy["C5"],
        "plaintext_c5_minus_c1": plaintext_delta,
        "tfe_c5_minus_c1": tfe_delta,
        "gain_retention": retention,
        "maximum_accuracy_gap": max(gaps.values()),
        "maximum_parameter_abs_error": max(parameter_max_abs.values()),
        "utility_gate": utility_gate,
        "correctness_pass": correctness_pass,
        "graph_build_seconds": report["graph_build_seconds"],
        "local_secure_seconds": report["local_secure_seconds"],
        "secure_average_seconds": report[
            "secure_average_initialize_seconds"
        ],
        "c1_anchor_secure_seconds": report["c1_anchor_secure_seconds"],
        "c5_anchor_secure_seconds": report["c5_anchor_secure_seconds"],
        "total_secure_seconds": report["total_secure_seconds"],
        "peak_rss_gib": report["peak_rss_gib"],
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "source_git_commit": source_commit(),
        "official_mtt_only": True,
        "full_data_initializer_dependency": False,
        "test_read": False,
        "locked_holdout_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require", choices=["cifar10", "cifar100", "all"], default=None
    )
    args = parser.parse_args()
    rows = [
        row for row in (validate_one(dataset) for dataset in DATASETS)
        if row is not None
    ]
    status = (
        "COMPLETE"
        if len(rows) == 2 and all(row["correctness_pass"] for row in rows)
        else "PARTIAL_OR_FAILED"
    )
    payload = {
        "schema": "new_stage_b_anchor_postagg_tfe_validation_v1",
        "status": status,
        "source_git_commit": source_commit(),
        "rows": rows,
        "official_mtt_only": True,
        "full_data_initializer_dependency": False,
        "test_read": False,
        "locked_holdout_read": False,
    }
    atomic_json(OUTPUT_JSON, payload)
    fields = [
        "dataset", "seed", "distribution", "backend", "finite",
        "plaintext_c1_accuracy", "plaintext_c4_accuracy",
        "plaintext_c5_accuracy", "tfe_c1_accuracy", "tfe_c4_accuracy",
        "tfe_c5_accuracy", "plaintext_c5_minus_c1",
        "tfe_c5_minus_c1", "gain_retention", "maximum_accuracy_gap",
        "maximum_parameter_abs_error", "utility_gate", "correctness_pass",
        "graph_build_seconds", "local_secure_seconds",
        "secure_average_seconds", "c1_anchor_secure_seconds",
        "c5_anchor_secure_seconds", "total_secure_seconds", "peak_rss_gib",
        "input_sha256", "report_sha256", "model_sha256",
        "source_git_commit", "official_mtt_only",
        "full_data_initializer_dependency", "test_read",
        "locked_holdout_read"
    ]
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    if args.require:
        required = set(DATASETS) if args.require == "all" else {args.require}
        passed = {
            row["dataset"] for row in rows if row["correctness_pass"]
        }
        if not required.issubset(passed):
            raise SystemExit(
                "TFE_VALIDATION_FAILED required={} passed={}".format(
                    sorted(required), sorted(passed)
                )
            )
    print(
        "TFE_VALIDATION status={} passed={}".format(
            status,
            [row["dataset"] for row in rows if row["correctness_pass"]],
        )
    )


if __name__ == "__main__":
    main()

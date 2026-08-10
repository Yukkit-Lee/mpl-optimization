#!/usr/bin/env python3
"""Acceptance audit for one full-trajectory SPU cell (schema v1).

Validates every hard gate from handoff §9. Reads only the exact report,
combined trace, effective config, Stage-B/C traces. Writes a machine-readable
audit JSON next to the report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LOG = ROOT / "new_log/new_secretflow_spu/full_trajectory_v1"
OUT = ROOT / "new_output/new_secretflow_spu/full_trajectory_v1"

DATASETS = ("cifar10", "cifar100", "tiny")
METHODS = ("full_data", "true_ours")
EXPECTED_STAGE_C_STEPS = {"cifar10": 80, "cifar100": 20, "tiny": 180}
EXPECTED_EPOCHS = {
    ("cifar10", "true_ours"): 40, ("cifar10", "full_data"): 40,
    ("cifar100", "true_ours"): 80, ("cifar100", "full_data"): 40,
    ("tiny", "true_ours"): 80, ("tiny", "full_data"): 40,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_cell(dataset, method):
    run_id = "full_trajectory_v1_formal_{}_{}_seed42".format(dataset, method)
    report_path = LOG / "{}_report.json".format(run_id)
    if not report_path.exists():
        return {
            "dataset": dataset, "method": method, "status": "NO_REPORT",
            "passed": False,
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ec = report["effective_config"]
    output_dir = OUT / run_id
    trace_path = output_dir / "new_spu_full_trajectory_trace.csv"
    stage_b_trace = output_dir / "stage_b_trace.csv"
    stage_c_trace = output_dir / "stage_c_trace.csv"

    with trace_path.open(newline="", encoding="utf-8") as handle:
        trace = list(csv.DictReader(handle))
    with stage_b_trace.open(newline="", encoding="utf-8") as handle:
        b_rows = list(csv.DictReader(handle))
    with stage_c_trace.open(newline="", encoding="utf-8") as handle:
        c_rows = list(csv.DictReader(handle))

    expected_b_points = 1 + EXPECTED_EPOCHS[(dataset, method)] // int(
        ec["stage_b_eval_interval_epochs"]
    )
    if ec.get("stage_b_post_anchor", False) or method == "true_ours":
        # True Ours adds pre-anchor + post-anchor.
        pass
    # Determine anchor presence from the effective config base.
    base = report.get("effective_config", {}).get("base", {})
    anchor_ce = bool(base.get("anchor_ce", method == "true_ours"))
    if anchor_ce:
        expected_b_points += 2

    c_steps = [int(r["epoch_or_step"]) for r in c_rows]
    expected_c = EXPECTED_STAGE_C_STEPS[dataset]

    gates = {
        "status_complete": report.get("status") == "COMPLETE",
        "run_kind_formal": report.get("run_kind") == "formal",
        "schema_v1": report.get("schema") == "new_spu_full_trajectory_v1",
        "seed_42": ec.get("seed") == 42,
        "backend": report.get("backend") == "SecretFlow-SPU device API",
        "protocol_aby3_fm128_fxp16": (
            report.get("protocol", {}).get("protocol") == "ABY3"
            and report.get("protocol", {}).get("field") == "FM128"
            and report.get("protocol", {}).get("fxp_fraction_bits") == 16
        ),
        "stage_b_point_count": len(b_rows) == expected_b_points,
        "stage_c_strict_steps": (
            c_steps == list(range(10, expected_c + 1, 10))
            or c_steps == [10 * i for i in range(1, expected_c // 10 + 1)]
        ),
        "stage_b_client_public_reveals_0": (
            report.get("stage_b_client_state_public_reveals") == 0
        ),
        "no_client_local_public_reconstruction": all(
            not (e.get("scope") == "client_local_secret_state" and e.get("public_reconstruction"))
            for e in report["reveal_ledger"]
        ),
        "stage_b_public_only_aggregate_logits": all(
            not e.get("public_reconstruction")
            or e.get("scope") in (
                "aggregate_development_logits", "global_secret_state",
                "conditional_dp_global_state", "provider_public_metadata",
            )
            for e in report["reveal_ledger"]
        ),
        "evaluation_used_for_training_false": all(
            str(r.get("used_for_training", "False")).strip().lower() != "true"
            for r in trace
        ),
        "epsilon_within_target": float(report["conditional_stage_c_epsilon"]) <= 3.000001,
        "delta_1e5": float(report["delta"]) == 1e-5,
        "final_acc_matches_trace": (
            abs(float(report["final_development_accuracy"]) - float(c_rows[-1]["development_accuracy"])) < 1e-6
        ),
        "official_test_unread": report.get("official_test_read") is False,
        "locked_holdout_unread": report.get("locked_holdout_read") is False,
        "all_finite": all(
            np.isfinite(float(v))
            for v in [
                report["final_development_accuracy"],
                report["conditional_stage_c_epsilon"],
                report["online_secure_seconds"],
                report["raw_wall_seconds"],
            ]
        ),
        "trace_monotonic": all(
            float(b) >= float(a)
            for a, b in zip(
                [float(r["cumulative_training_online_secure_seconds"]) for r in trace],
                [float(r["cumulative_training_online_secure_seconds"]) for r in trace][1:],
            )
        ),
    }
    audit = {
        "schema": "new_spu_full_trajectory_audit_v1",
        "dataset": dataset,
        "method": method,
        "passed": all(gates.values()),
        "gates": gates,
        "report_sha256": sha256_file(report_path),
        "trace_sha256": sha256_file(trace_path),
        "stage_b_trace_points": len(b_rows),
        "expected_stage_b_points": expected_b_points,
        "stage_c_trace_points": len(c_rows),
        "report_path": str(report_path),
    }
    audit_path = LOG / "{}_audit.json".format(run_id)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    result = audit_cell(args.dataset, args.method)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

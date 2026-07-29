#!/usr/bin/env python3
"""Validate the user-approved 14-task True Ours correction set."""

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_scripts.new_true_ours.new_run_tiny_full_seed42_correction import (
    correction_paths,
)
from new_scripts.new_true_ours.new_run_true_ours_formal_three_seed import (
    CONFIG_PATH,
    atomic_json,
    file_sha256,
    task_paths,
    validate_stage_b,
    validate_stage_c,
)


POLICY_PATH = (
    ROOT
    / "new_configs/new_true_ours/new_true_ours_minimal_correction_policy.json"
)
OUTPUT_PATH = (
    ROOT / "new_log/new_true_ours/new_true_ours_minimal_validation.json"
)


def expected_tasks(policy):
    return [
        (dataset, method, int(seed))
        for dataset, seeds in policy["expected_seeds"].items()
        for method in policy["methods"]
        for seed in seeds
    ]


def paths_for(dataset, method, seed):
    if (dataset, method, int(seed)) == ("tiny", "full_data", 42):
        return correction_paths()
    return task_paths(dataset, method, int(seed))


def validate_timeline(path, dataset, method, seed):
    timeline = json.loads(Path(path).read_text(encoding="utf-8"))
    points = timeline.get("points", [])
    times = [
        float(point["cumulative_online_secure_seconds"]) for point in points
    ]
    checks = {
        "identity": (
            timeline.get("status") == "COMPLETE_DEVELOPMENT_ONLY"
            and timeline.get("dataset") == dataset
            and timeline.get("method") == method
            and int(timeline.get("seed")) == int(seed)
        ),
        "development_only": (
            timeline.get("locked_holdout_read") is False
            and timeline.get("official_test_read") is False
        ),
        "finite_positive_points": (
            bool(points)
            and all(math.isfinite(value) and value > 0 for value in times)
        ),
        "monotonic_time": all(
            later >= earlier for earlier, later in zip(times, times[1:])
        ),
        "stages_present": (
            any(point.get("stage") == "stage_b" for point in points)
            and any(point.get("stage") == "stage_c" for point in points)
        ),
        "total_positive": (
            float(timeline.get("total_online_secure_seconds", 0)) > 0
        ),
    }
    if not all(checks.values()):
        raise ValueError(
            "timeline validation failed {} {} seed{}: {}".format(
                method, dataset, seed, checks
            )
        )
    return timeline, checks


def prove_old_tiny_full_rejected(config):
    frozen = config["datasets"]["tiny"]["methods"]["full_data"]
    old = task_paths("tiny", "full_data", 42)
    rejected = {}
    try:
        validate_stage_b(
            old["stage_b_report"], "tiny", "full_data", 42, frozen
        )
        rejected["stage_b"] = False
    except Exception as exc:
        rejected["stage_b"] = True
        rejected["stage_b_reason"] = str(exc)
    try:
        validate_stage_c(
            old["stage_c_report"], "tiny", "full_data", 42, frozen
        )
        rejected["stage_c"] = False
    except Exception as exc:
        rejected["stage_c"] = True
        rejected["stage_c_reason"] = str(exc)
    if not rejected["stage_b"] or not rejected["stage_c"]:
        raise ValueError("superseded Tiny Full was not rejected by both gates")
    return rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow_missing_correction", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    tasks = expected_tasks(policy)
    if len(tasks) != int(policy["expected_task_count"]):
        raise ValueError("policy task count mismatch")
    rows = []
    missing = []
    failed = []
    for dataset, method, seed in tasks:
        frozen = config["datasets"][dataset]["methods"][method]
        paths = paths_for(dataset, method, seed)
        required = (
            paths["stage_b_report"],
            paths["stage_c_report"],
            paths["timeline"],
        )
        absent = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
        if absent:
            row = {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "status": "MISSING",
                "missing": absent,
            }
            rows.append(row)
            missing.append(row)
            continue
        try:
            stage_b, stage_b_checks = validate_stage_b(
                paths["stage_b_report"], dataset, method, seed, frozen
            )
            stage_c, stage_c_checks = validate_stage_c(
                paths["stage_c_report"], dataset, method, seed, frozen
            )
            timeline, timeline_checks = validate_timeline(
                paths["timeline"], dataset, method, seed
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "status": "PASS",
                    "stage_b_report": str(
                        paths["stage_b_report"].relative_to(ROOT)
                    ),
                    "stage_b_report_sha256": file_sha256(
                        paths["stage_b_report"]
                    ),
                    "stage_c_report": str(
                        paths["stage_c_report"].relative_to(ROOT)
                    ),
                    "stage_c_report_sha256": file_sha256(
                        paths["stage_c_report"]
                    ),
                    "timeline": str(paths["timeline"].relative_to(ROOT)),
                    "timeline_sha256": file_sha256(paths["timeline"]),
                    "stage_b_checks": stage_b_checks,
                    "stage_c_checks": stage_c_checks,
                    "timeline_checks": timeline_checks,
                    "stage_b_online_secure_seconds": float(
                        stage_b["online_secure_runtime_seconds"]
                    ),
                    "stage_c_online_secure_seconds": float(
                        stage_c["online_secure_runtime_seconds"]
                    ),
                    "total_online_secure_seconds": float(
                        timeline["total_online_secure_seconds"]
                    ),
                    "conditional_stage_c_epsilon": float(
                        stage_c["privacy"][
                            "stage_c_training_feature_record_epsilon"
                        ]
                    ),
                    "final_development_accuracy": float(
                        stage_c["history"][-1][
                            "development_validation_accuracy"
                        ]
                    ),
                }
            )
        except Exception as exc:
            row = {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "status": "FAIL",
                "reason": str(exc),
            }
            rows.append(row)
            failed.append(row)
    passed = sum(row["status"] == "PASS" for row in rows)
    old_rejection = prove_old_tiny_full_rejected(config)
    if failed:
        status = "FAILED"
    elif passed == len(tasks):
        status = "COMPLETE_14_OF_14"
    elif (
        args.allow_missing_correction
        and passed == len(tasks) - 1
        and len(missing) == 1
        and (
            missing[0]["dataset"],
            missing[0]["method"],
            int(missing[0]["seed"]),
        )
        == ("tiny", "full_data", 42)
    ):
        status = "PRE_CORRECTION_READY_13_OF_14"
    else:
        status = "INCOMPLETE"
    payload = {
        "schema": "new_true_ours_minimal_validation_v1",
        "status": status,
        "expected": len(tasks),
        "passed": passed,
        "missing": len(missing),
        "failed": len(failed),
        "seed_policy": policy["expected_seeds"],
        "old_tiny_full_rejection": old_rejection,
        "locked_holdout_read": False,
        "official_test_read": False,
        "rows": rows,
    }
    atomic_json(OUTPUT_PATH, payload)
    print(
        "{} passed={}/{} missing={} failed={}".format(
            status, passed, len(tasks), len(missing), len(failed)
        )
    )
    if status in ("COMPLETE_14_OF_14", "PRE_CORRECTION_READY_13_OF_14"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

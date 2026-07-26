#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve and optionally activate the first persistent Public MTT mainline queue."""

import argparse
import json
import os
from pathlib import Path

from new_public_mtt_mainline_common import ROOT, load_config


PYTHON = os.environ.get(
    "NEW_MODERN_PYTHON", str(ROOT.parent / "new_envs" / "distillation" / "bin" / "python")
)
QUEUE = (
    ROOT
    / "new_configs/new_public_mtt_mainline"
    / "new_gpu0_feature_quality.tsv"
)
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
PLAN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_resolved_mainline_plan.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activate",
        action="store_true",
        help="atomically make this queue active after writing the plan",
    )
    args = parser.parse_args()
    config = load_config()
    tuning_runs = (
        len(config["datasets"])
        * len(config["tuning_seeds"])
        * len(config["linear_probe"]["candidates"])
    )
    formal_runs = len(config["datasets"]) * len(config["formal_seeds"])
    formal_output = (
        ROOT
        / "new_log/new_public_mtt_mainline/json/new_feature_quality_formal.json"
    )
    rows = [
        "# task_id\tcheckpoint\tcommand",
        (
            "public_mainline_prepare_test_free_caches\t{checkpoint}\t"
            "{python} {script}"
        ).format(
            checkpoint=(
                ROOT
                / "new_log/new_public_mtt_mainline/json"
                / "new_test_free_feature_cache_manifest.json"
            ),
            python=PYTHON,
            script=ROOT / "new_prepare_public_mtt_reference_caches.py",
        ),
        (
            "public_mainline_feature_quality\t{checkpoint}\t{python} {script} "
            "--phase all --device cuda"
        ).format(
            checkpoint=formal_output,
            python=PYTHON,
            script=ROOT / "new_public_mtt_feature_quality.py",
        ),
    ]
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "new_public_mtt_mainline_resolved_plan_v1",
        "status": "FEATURE_QUALITY_GATE_READY",
        "dp_mtt_status": "DEFERRED_UNTIL_MAIN_PIPELINE_COMPLETE",
        "datasets": config["datasets"],
        "tuning_seeds": config["tuning_seeds"],
        "formal_seeds": config["formal_seeds"],
        "resolved_runs": {
            "feature_quality_tuning": tuning_runs,
            "feature_quality_formal": formal_runs,
            "first_queue_total": 1 + tuning_runs + formal_runs,
        },
        "resource_policy": {
            "cpu_threads_per_process": 1,
            "num_workers": 0,
            "visible_gpu": 0,
            "max_processes_on_gpu": 1,
            "heartbeat_seconds": 60,
            "checkpoint_interval_steps": 100,
        },
        "test_policy": {
            "tuning_test_loaded": False,
            "formal_test_usage": "single_final_readonly_after_config_freeze",
        },
        "queue": str(QUEUE),
        "expected_outputs": [
            str(
                ROOT
                / "new_log/new_public_mtt_mainline/json"
                / "new_test_free_feature_cache_manifest.json"
            ),
            str(
                ROOT
                / "new_log/new_public_mtt_mainline/json"
                / "new_feature_quality_tuning.json"
            ),
            str(formal_output),
        ],
    }
    PLAN.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.activate:
        ACTIVE.write_text(str(QUEUE) + "\n", encoding="utf-8")
    print("=" * 88)
    print("Public MTT mainline resolved configuration")
    print("phase=feature_quality_gate")
    print("datasets={}".format(config["datasets"]))
    print("cache_prep=1 tuning_runs={} formal_runs={} total={}".format(
        tuning_runs, formal_runs, 1 + tuning_runs + formal_runs
    ))
    print("cpu_threads=1 num_workers=0 visible_gpu=0")
    print("dp_mtt_status=DEFERRED_UNTIL_MAIN_PIPELINE_COMPLETE")
    print("queue={}".format(QUEUE))
    for output in payload["expected_outputs"]:
        print("expected_output={}".format(output))
    print("activated={}".format(args.activate))
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

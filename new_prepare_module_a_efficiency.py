#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the persistent single-GPU Module-A efficiency queue."""

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
    / "new_configs/new_public_mtt_mainline/new_gpu0_module_a_efficiency.tsv"
)
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
PLAN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_module_a_efficiency_plan.json"
)


def main():
    parser = argparse.ArgumentParser(
        description="Resolve the Module-A queue without starting it by default"
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="atomically make this queue active after writing the plan",
    )
    args = parser.parse_args()
    config = load_config()
    quality = json.loads(
        (
            ROOT
            / "new_log/new_public_mtt_mainline/json"
            / "new_feature_quality_formal.json"
        ).read_text(encoding="utf-8")
    )
    if quality["status"] != "QUALITY_GATE_PASS":
        raise RuntimeError("feature quality gate has not passed")
    rows = ["# task_id\tcheckpoint\tcommand"]
    expected = []
    for dataset in config["datasets"]:
        for seed in config["formal_seeds"]:
            manifest = (
                ROOT
                / "new_cache/new_public_mtt_mainline/new_full_features"
                / dataset
                / "new_seed{}".format(seed)
                / "new_full_feature_manifest.json"
            )
            rows.append(
                (
                    "public_mainline_extract_full_{dataset}_seed{seed}\t"
                    "{checkpoint}\t{python} {script} --dataset {dataset} "
                    "--seed {seed} --batch_size 256 --device cuda"
                ).format(
                    dataset=dataset,
                    seed=seed,
                    checkpoint=manifest,
                    python=PYTHON,
                    script=ROOT / "new_extract_public_mtt_full_features.py",
                )
            )
            result = (
                ROOT
                / "new_log/new_public_mtt_mainline/json/new_module_a_runs"
                / dataset
                / "new_module_a_efficiency_{}_seed{}.json".format(
                    dataset, seed
                )
            )
            rows.append(
                (
                    "public_mainline_module_a_{dataset}_seed{seed}\t"
                    "{checkpoint}\t{python} {script} --dataset {dataset} "
                    "--seed {seed} --device cuda"
                ).format(
                    dataset=dataset,
                    seed=seed,
                    checkpoint=result,
                    python=PYTHON,
                    script=ROOT / "new_module_a_efficiency_run.py",
                )
            )
            expected.append(str(result))
    report = (
        ROOT
        / "new_log/new_public_mtt_mainline/json"
        / "new_module_a_efficiency_report.json"
    )
    rows.append(
        "public_mainline_report_module_a\t{output}\t{python} {script}".format(
            output=report,
            python=PYTHON,
            script=ROOT / "new_report_module_a_efficiency.py",
        )
    )
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    payload = {
        "schema": "new_module_a_efficiency_plan_v1",
        "status": "READY",
        "datasets": config["datasets"],
        "seeds": config["formal_seeds"],
        "methods": config["module_a_efficiency"]["sources"],
        "resolved_tasks": len(rows) - 1,
        "logical_method_runs": (
            len(config["datasets"])
            * len(config["formal_seeds"])
            * len(config["module_a_efficiency"]["sources"])
        ),
        "resource_policy": {
            "cpu_threads": 1,
            "num_workers": 0,
            "visible_gpu": 0,
            "checkpoint_resume": True,
        },
        "expected_outputs": expected + [str(report)],
        "queue": str(QUEUE),
    }
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.activate:
        temporary = ACTIVE.with_name(ACTIVE.name + ".new_tmp")
        temporary.write_text(str(QUEUE) + "\n", encoding="utf-8")
        temporary.replace(ACTIVE)
    print("=" * 88)
    print("Module A Public-MTT efficiency resolved configuration")
    print("datasets={} seeds={}".format(config["datasets"], config["formal_seeds"]))
    print("methods={}".format(config["module_a_efficiency"]["sources"]))
    print(
        "extraction_tasks=9 method_runs=27 report_tasks=1 queue_tasks={}".format(
            len(rows) - 1
        )
    )
    print("cpu_threads=1 num_workers=0 visible_gpu=0")
    print("queue={}".format(QUEUE))
    print("report={}".format(report))
    print("activated={}".format(args.activate))
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

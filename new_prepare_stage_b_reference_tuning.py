#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve and optionally activate the persistent Stage-B tuning queue."""

import argparse
import json
import os

from new_public_mtt_mainline_common import ROOT, load_config


PYTHON = os.environ.get(
    "NEW_MODERN_PYTHON", str(ROOT.parent / "new_envs" / "distillation" / "bin" / "python")
)
QUEUE = (
    ROOT
    / "new_configs/new_public_mtt_mainline/new_gpu0_stage_b_tuning_round2.tsv"
)
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
PLAN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_stage_b_tuning_round2_plan.json"
)


def main():
    parser = argparse.ArgumentParser(
        description="Resolve Stage-B tuning; activation must be explicit"
    )
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    config = load_config()
    candidates_by_dataset = {}
    for dataset in config["datasets"]:
        candidates = 1
        for values in config["module_b"]["tuning_grid_by_dataset"][
            dataset
        ].values():
            candidates *= len(values)
        candidates_by_dataset[dataset] = candidates
    rows = ["# task_id\tcheckpoint\tcommand"]
    outputs = []
    manifest = (
        ROOT
        / "new_cache/new_public_mtt_mainline/new_stage_b_npz_round2"
        / "new_stage_b_npz_manifest.json"
    )
    rows.append(
        (
            "public_mainline_stage_b_round2_export\t{manifest}\t"
            "{python} {script} --dataset all --output_tag "
            "new_stage_b_npz_round2 --reference_per_class 5"
        ).format(
            manifest=manifest,
            python=PYTHON,
            script=ROOT / "new_export_public_mtt_stage_b_npz.py",
        )
    )
    for dataset in config["datasets"]:
        output = (
            ROOT
            / "new_log/new_public_mtt_mainline/json/new_stage_b_tuning"
            / "new_stage_b_tuning_round2_{}.json".format(dataset)
        )
        rows.append(
            (
                "public_mainline_stage_b_round2_tune_{dataset}\t{output}\t"
                "{python} {script} --dataset {dataset} --cache_tag "
                "new_stage_b_npz_round2 --round_tag round2"
            ).format(
                dataset=dataset,
                output=output,
                python=PYTHON,
                script=ROOT / "new_stage_b_reference_tuning.py",
            )
        )
        outputs.append(str(output))
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    logical_runs = (
        sum(candidates_by_dataset.values())
        * len(config["tuning_seeds"])
        * len(config["module_b"]["splits"])
    )
    payload = {
        "schema": "new_stage_b_tuning_plan_v1",
        "status": "READY",
        "datasets": config["datasets"],
        "tuning_seeds": config["tuning_seeds"],
        "splits": config["module_b"]["splits"],
        "candidate_count_per_dataset": candidates_by_dataset,
        "logical_reference_only_runs": logical_runs,
        "queue_tasks": len(rows) - 1,
        "test_loaded": False,
        "resource_policy": {
            "cpu_threads": 1,
            "num_workers": 0,
            "visible_gpu": 0,
            "serial": True,
            "checkpoint_resume": True,
        },
        "expected_outputs": outputs,
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
    print("=" * 92)
    print("Stage B reference-only tuning resolved configuration")
    print("datasets={}".format(config["datasets"]))
    print(
        "candidates/dataset={} seeds={} splits={}".format(
            candidates_by_dataset,
            config["tuning_seeds"],
            config["module_b"]["splits"],
        )
    )
    print(
        "logical_runs={} queue_tasks={} test_loaded=False".format(
            logical_runs, len(rows) - 1
        )
    )
    print("cpu_threads=1 num_workers=0 visible_gpu=0 serial=True")
    print("outputs={}".format(outputs))
    print("activated={}".format(args.activate))
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

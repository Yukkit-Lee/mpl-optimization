#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare missing Random-Subset NoKD secure-average initializers."""

import argparse
import json
import os

from new_public_mtt_mainline_common import ROOT, load_config


TFE_PYTHON = os.environ.get(
    "NEW_TFE_PYTHON", str(ROOT.parent / "new_envs" / "tfe" / "bin" / "python")
)
QUEUE = (
    ROOT
    / "new_configs/new_public_mtt_mainline"
    / "new_tfe_end_to_end_baseline_stage_b.tsv"
)
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
PLAN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_end_to_end_baseline_stage_b_plan.json"
)
FEATURE_MANIFEST = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_end_to_end_feature_manifest.json"
)
MODULE_C_REPORT = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_module_c_formal_report.json"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    config = load_config()
    stage_b = config["end_to_end_baseline_stage_b"]
    features = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    if features["status"] != "COMPLETE" or features["test_loaded"]:
        raise ValueError("End-to-End feature assets are not a test-free COMPLETE")
    if config["dp_mtt_status"] != "EXCLUDED_BY_USER":
        raise ValueError("DP-MTT must remain excluded")
    if stage_b["test_loaded"]:
        raise ValueError("baseline Stage B cannot load test")
    if args.activate:
        module_c = json.loads(MODULE_C_REPORT.read_text(encoding="utf-8"))
        if (
            module_c["status"] != "COMPLETE"
            or module_c["runs_completed"] != module_c["runs_expected"]
        ):
            raise ValueError("Module C formal gate is incomplete")

    by_key = {
        (row["dataset"], int(row["seed"])): row
        for row in features["rows"]
    }
    rows = ["# task_id\tcheckpoint\tcommand"]
    tasks = []
    for dataset in config["datasets"]:
        for seed in config["formal_seeds"]:
            asset = by_key[(dataset, seed)]
            feature_cache = asset["random_subset_cache"]
            output_dir = (
                ROOT
                / "new_init_model/new_public_mtt_mainline/end_to_end_baseline"
                / dataset
                / "seed{}".format(seed)
            )
            checkpoint = (
                ROOT
                / "new_checkpoint/new_public_mtt_mainline"
                / "end_to_end_baseline_stage_b"
                / dataset
                / "seed{}/new_stage_b".format(seed)
            )
            expected_initializer = (
                output_dir
                / "new_secure_stage_b_candidate_nokd_{}_iid_seed{}.npz".format(
                    dataset, seed
                )
            )
            task_id = "public_mainline_end_to_end_baseline_stage_b_{}_seed{}".format(
                dataset, seed
            )
            command = (
                "{python} {script} --feature_cache {cache} "
                "--dataset {dataset} --output_dir {output} "
                "--checkpoint_prefix {checkpoint} --seed {seed} "
                "--num_clients {clients} --split iid --batch_size {batch} "
                "--local_epochs {local_epochs} --local_lr {local_lr} "
                "--distill_epochs 0 --distill_lr 0.03 --lambda_kd 0.0 "
                "--reference_supervised_weight 0.0 --weight_temperature 0.005 "
                "--init_scale {init_scale} "
                "--precision_fractional {precision} "
                "--matmul_chunk_size {chunk} "
                "--checkpoint_interval {checkpoint_interval} "
                "--formal_candidate_release"
            ).format(
                python=TFE_PYTHON,
                script=ROOT / "new_stage_b_secure_init.py",
                cache=feature_cache,
                dataset=dataset,
                output=output_dir,
                checkpoint=checkpoint,
                seed=seed,
                clients=stage_b["num_clients"],
                batch=stage_b["batch_size"],
                local_epochs=stage_b["local_epochs"][dataset],
                local_lr=stage_b["local_learning_rate"],
                init_scale=stage_b["init_scale"],
                precision=stage_b["precision_fractional"],
                chunk=stage_b["matmul_chunk_size"],
                checkpoint_interval=stage_b["checkpoint_interval"],
            )
            rows.append("{}\t{}\t{}".format(task_id, checkpoint, command))
            tasks.append(
                {
                    "task_id": task_id,
                    "phase": "end_to_end_baseline_initializer",
                    "dataset": dataset,
                    "seed": seed,
                    "split": "iid",
                    "method": "NoKD-Secure-Average",
                    "privacy": "N/A_MPC_CONFIDENTIALITY_ONLY",
                    "feature_source": "random_subset",
                    "feature_cache": feature_cache,
                    "feature_cache_sha256": asset["random_subset_sha256"],
                    "local_epochs": stage_b["local_epochs"][dataset],
                    "distill_epochs": 0,
                    "checkpoint_interval": stage_b["checkpoint_interval"],
                    "test_loaded": False,
                    "resource": "tfe",
                    "expected_initializer": str(expected_initializer),
                    "checkpoint_prefix": str(checkpoint),
                }
            )
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(
        json.dumps(
            {
                "schema": "new_end_to_end_baseline_stage_b_plan_v1",
                "status": "READY",
                "formal_result": True,
                "test_loaded": False,
                "dp_mtt_status": "EXCLUDED_BY_USER",
                "not_a_rerun": (
                    "new Random-Subset baseline; completed Official-MTT Stage B "
                    "assets are unchanged"
                ),
                "method": "NoKD-Secure-Average",
                "backend": "tfe_mpc",
                "protocol": "ABY3",
                "privacy": "N/A_MPC_CONFIDENTIALITY_ONLY",
                "queue": str(QUEUE),
                "tasks": tasks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.activate:
        ACTIVE.write_text("tfe\t{}\n".format(QUEUE), encoding="utf-8")
    for task in tasks:
        print(
            "phase={phase} dataset={dataset} seed={seed} split={split} "
            "method={method} privacy={privacy} resource={resource} "
            "expected={expected_initializer}".format(**task)
        )
    print("queue={}".format(QUEUE))
    print("plan={}".format(PLAN))
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

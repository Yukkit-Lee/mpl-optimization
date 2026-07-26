#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a single-GPU, test-free DP-MTT trajectory-window tuning queue."""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
PYTHON = os.environ.get(
    "NEW_MODERN_PYTHON", str(ROOT.parent / "new_envs" / "distillation" / "bin" / "python")
)
ROUND2 = (
    ROOT
    / "new_cache/new_secure_feature/new_tuning_phase6_dp_mtt_cifar10_round2"
    / "new_seed42"
)
TRAJECTORY_ROOT = ROUND2 / "new_trajectories"
BASE = (
    ROOT
    / "new_cache/new_secure_feature/new_tuning_phase6_dp_mtt_cifar10_round3"
)
QUEUE = (
    ROOT
    / "new_configs/new_secure_feature/new_gpu0_phase6_dp_trajectory_tuning_round3.tsv"
)
PLAN = (
    ROOT
    / "new_log/new_secure_feature/json/new_phase6_dp_trajectory_tuning_round3_plan.json"
)
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"


CANDIDATES = {
    "late3": {
        "max_start_epoch": 45,
        "expert_epochs": 3,
        "syn_steps": 20,
    },
    "late10": {
        "max_start_epoch": 40,
        "expert_epochs": 10,
        "syn_steps": 30,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Phase-6 DP-MTT trajectory tuning round 3"
    )
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    manifest = (
        TRAJECTORY_ROOT
        / "cifar10/new_stage_a_dp_trajectory_manifest_cifar10.json"
    )
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    rows = ["# task_id\tcheckpoint\tcommand"]
    expected = {}
    for name, config in CANDIDATES.items():
        candidate_root = BASE / ("new_" + name)
        distilled_root = candidate_root / "new_distilled"
        encoder_root = candidate_root / "new_encoder"
        mtt_checkpoint = (
            distilled_root
            / "cifar10/new_dp_mtt_checkpoint_cifar10_seed42.pt"
        )
        encoder_checkpoint = (
            encoder_root
            / "dp_mtt/cifar10/new_shared_encoder_checkpoint_cifar10_dp_mtt_seed42.pt"
        )
        encoder_model = (
            encoder_root
            / "dp_mtt/cifar10/new_shared_encoder_cifar10_dp_mtt_seed42.pt"
        )
        reference_json = (
            ROOT
            / "new_log/new_secure_feature/json"
            / ("new_phase6_dp_round3_{}_public_reference.json".format(name))
        )
        rows.append(
            (
                "tuning_phase6_dp_round3_{name}_mtt\t{checkpoint}\t{python} "
                "{script} --dataset cifar10 --trajectory_root {trajectory_root} "
                "--output_dir {output_dir} --ipc 10 --iterations 2500 "
                "--syn_steps {syn_steps} --expert_epochs {expert_epochs} "
                "--max_start_epoch {max_start_epoch} --lr_img 1000 "
                "--lr_lr 1e-5 --lr_teacher 0.01 --checkpoint_interval 100 "
                "--seed 42 --device cuda"
            ).format(
                name=name,
                checkpoint=mtt_checkpoint,
                python=PYTHON,
                script=ROOT / "new_stage_a_dp_distill.py",
                trajectory_root=TRAJECTORY_ROOT,
                output_dir=distilled_root,
                **config
            )
        )
        rows.append(
            (
                "tuning_phase6_dp_round3_{name}_encoder\t{checkpoint}\t{python} "
                "{script} --dataset cifar10 --track dp_mtt --dp_root {dp_root} "
                "--dp_ipc 10 --output_dir {output_dir} --epochs 300 "
                "--batch_size 128 --eval_batch_size 256 --learning_rate 0.01 "
                "--momentum 0.9 --weight_decay 0.0005 --feature_norm_bound 1.0 "
                "--holdout_per_class 1 --encoder_use_all_synthetic --seed 42 "
                "--device cuda --augmentation"
            ).format(
                name=name,
                checkpoint=encoder_checkpoint,
                python=PYTHON,
                script=ROOT / "new_stage_a_shared_encoder.py",
                dp_root=distilled_root,
                output_dir=encoder_root,
            )
        )
        rows.append(
            (
                "tuning_phase6_dp_round3_{name}_public_reference\t{output}\t"
                "{python} {script} --dataset cifar10 --encoder {encoder} "
                "--output_json {output} --seed 42 --batch_size 256 --device cuda"
            ).format(
                name=name,
                output=reference_json,
                python=PYTHON,
                script=ROOT / "new_eval_stage_a_public_reference.py",
                encoder=encoder_model,
            )
        )
        expected[name] = {
            "configuration": config,
            "distilled_root": str(distilled_root),
            "encoder_root": str(encoder_root),
            "public_reference_json": str(reference_json),
        }

    report = (
        ROOT
        / "new_log/new_secure_feature/json/new_phase6_dp_trajectory_tuning_round3_result.json"
    )
    rows.append(
        (
            "report_phase6_dp_trajectory_tuning_round3\t{output}\t{python} "
            "{script}"
        ).format(
            output=report,
            python=PYTHON,
            script=ROOT / "new_report_phase6_dp_trajectory_tuning_round3.py",
        )
    )
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(
        json.dumps(
            {
                "schema": "new_phase6_dp_trajectory_tuning_round3_plan_v1",
                "dataset": "cifar10",
                "seed": 42,
                "selection_metric": "external_public_mtt_reference_accuracy",
                "baseline_public_reference_accuracy": 0.292,
                "test_loaded": False,
                "raw_private_data_loaded_during_mtt": False,
                "stage_a_epsilon_unchanged": True,
                "coarse_iterations": 2500,
                "candidates": expected,
                "cpu_threads": 1,
                "num_workers": 0,
                "visible_gpu": 0,
                "queue": str(QUEUE),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.activate:
        ACTIVE.write_text(str(QUEUE) + "\n", encoding="utf-8")
    print("Phase 6 DP-MTT trajectory tuning round 3")
    print("candidates={} coarse_iterations=2500".format(sorted(CANDIDATES)))
    print("selection=external_public_mtt_reference test_loaded=False")
    print("cpu_threads=1 num_workers=0 visible_gpu=0")
    print("queue={}".format(QUEUE))
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

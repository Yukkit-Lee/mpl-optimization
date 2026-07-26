#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the single-GPU 5000-step refinement of the selected DP-MTT candidate."""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
PYTHON = os.environ.get(
    "NEW_MODERN_PYTHON", str(ROOT.parent / "new_envs" / "distillation" / "bin" / "python")
)
TRAJECTORY_ROOT = (
    ROOT
    / "new_cache/new_secure_feature/new_tuning_phase6_dp_mtt_cifar10_round2"
    / "new_seed42/new_trajectories"
)
ROUND3_ROOT = (
    ROOT
    / "new_cache/new_secure_feature/new_tuning_phase6_dp_mtt_cifar10_round3"
    / "new_late10"
)
DISTILLED_ROOT = ROUND3_ROOT / "new_distilled"
ENCODER_ROOT = (
    ROOT
    / "new_cache/new_secure_feature/new_tuning_phase6_dp_mtt_cifar10_round4"
    / "new_late10_5000/new_encoder"
)
JSON_ROOT = ROOT / "new_log/new_secure_feature/json"
QUEUE = (
    ROOT
    / "new_configs/new_secure_feature"
    / "new_gpu0_phase6_dp_trajectory_tuning_round4.tsv"
)
PLAN = JSON_ROOT / "new_phase6_dp_trajectory_tuning_round4_plan.json"
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Phase-6 DP-MTT selected-candidate refinement"
    )
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()

    mtt_checkpoint = (
        DISTILLED_ROOT
        / "cifar10/new_dp_mtt_checkpoint_cifar10_seed42.pt"
    )
    if not mtt_checkpoint.exists():
        raise FileNotFoundError(mtt_checkpoint)
    encoder_checkpoint = (
        ENCODER_ROOT
        / "dp_mtt/cifar10/new_shared_encoder_checkpoint_cifar10_dp_mtt_seed42.pt"
    )
    encoder_model = (
        ENCODER_ROOT
        / "dp_mtt/cifar10/new_shared_encoder_cifar10_dp_mtt_seed42.pt"
    )
    reference_json = (
        JSON_ROOT / "new_phase6_dp_round4_late10_5000_public_reference.json"
    )
    report_json = (
        JSON_ROOT / "new_phase6_dp_trajectory_tuning_round4_result.json"
    )
    rows = [
        "# task_id\tcheckpoint\tcommand",
        (
            "tuning_phase6_dp_round4_late10_5000_mtt\t{checkpoint}\t{python} "
            "{script} --dataset cifar10 --trajectory_root {trajectories} "
            "--output_dir {output_dir} --ipc 10 --iterations 5000 "
            "--syn_steps 30 --expert_epochs 10 --max_start_epoch 40 "
            "--lr_img 1000 --lr_lr 1e-5 --lr_teacher 0.01 "
            "--checkpoint_interval 100 --seed 42 --device cuda"
        ).format(
            checkpoint=mtt_checkpoint,
            python=PYTHON,
            script=ROOT / "new_stage_a_dp_distill.py",
            trajectories=TRAJECTORY_ROOT,
            output_dir=DISTILLED_ROOT,
        ),
        (
            "tuning_phase6_dp_round4_late10_5000_encoder\t{checkpoint}\t"
            "{python} {script} --dataset cifar10 --track dp_mtt "
            "--dp_root {dp_root} --dp_ipc 10 --output_dir {output_dir} "
            "--epochs 300 --batch_size 128 --eval_batch_size 256 "
            "--learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 "
            "--feature_norm_bound 1.0 --holdout_per_class 1 "
            "--encoder_use_all_synthetic --seed 42 --device cuda --augmentation"
        ).format(
            checkpoint=encoder_checkpoint,
            python=PYTHON,
            script=ROOT / "new_stage_a_shared_encoder.py",
            dp_root=DISTILLED_ROOT,
            output_dir=ENCODER_ROOT,
        ),
        (
            "tuning_phase6_dp_round4_late10_5000_public_reference\t{output}\t"
            "{python} {script} --dataset cifar10 --encoder {encoder} "
            "--output_json {output} --seed 42 --batch_size 256 --device cuda"
        ).format(
            output=reference_json,
            python=PYTHON,
            script=ROOT / "new_eval_stage_a_public_reference.py",
            encoder=encoder_model,
        ),
        (
            "report_phase6_dp_trajectory_tuning_round4\t{output}\t{python} "
            "{script}"
        ).format(
            output=report_json,
            python=PYTHON,
            script=ROOT / "new_report_phase6_dp_trajectory_tuning_round4.py",
        ),
    ]
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(
        json.dumps(
            {
                "schema": "new_phase6_dp_trajectory_tuning_round4_plan_v1",
                "dataset": "cifar10",
                "seed": 42,
                "selected_from_round3": "late10",
                "selection_metric": "external_public_mtt_reference_accuracy",
                "baseline_public_reference_accuracy": 0.328,
                "resume_iteration": 2500,
                "target_iterations": 5000,
                "configuration": {
                    "max_start_epoch": 40,
                    "expert_epochs": 10,
                    "syn_steps": 30,
                },
                "test_loaded": False,
                "stage_a_epsilon_unchanged": True,
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
    print("Phase 6 DP-MTT round4: late10 2500 -> 5000")
    print("selection=external_public_mtt_reference test_loaded=False")
    print("cpu_threads=1 num_workers=0 visible_gpu=0")
    print("queue={}".format(QUEUE))
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

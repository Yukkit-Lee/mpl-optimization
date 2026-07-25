#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：准备 CIFAR-10 DP-MTT 完整 validation 链路的单 GPU 串行队列。

链路严格为 DP experts（5 parties × 2 experts × 50 epochs）→ 仅 trajectory
后处理的 5000-iteration MTT → reference-only encoder pilot。该队列是正式
配置前的 validation，不读取 test，也不计入三 seed 正式均值。
"""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
PYTHON = os.environ.get(
    "NEW_MODERN_PYTHON", "/home/lixujie/.conda/envs/distillation/bin/python"
)
BASE = ROOT / "new_cache/new_secure_feature/new_tuning_phase6_dp_mtt_cifar10_round2"
TRAJECTORY_ROOT = BASE / "new_seed42/new_trajectories"
DISTILLED_ROOT = BASE / "new_seed42/new_distilled"
ENCODER_ROOT = BASE / "new_seed42/new_encoder_pilot"
QUEUE = ROOT / "new_configs/new_secure_feature/new_gpu0_phase6_dp_validation.tsv"
PLAN = ROOT / "new_log/new_secure_feature/json/new_phase6_dp_validation_plan.json"
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description="准备 Phase 6 DP-MTT validation")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    expert_checkpoint = (
        TRAJECTORY_ROOT
        / "cifar10"
        / "new_dp_expert_checkpoint_cifar10_party00_expert00.pt"
    )
    distill_checkpoint = (
        DISTILLED_ROOT / "cifar10/new_dp_mtt_checkpoint_cifar10_seed42.pt"
    )
    encoder_checkpoint = (
        ENCODER_ROOT
        / "dp_mtt/cifar10/new_shared_encoder_checkpoint_cifar10_dp_mtt_seed42.pt"
    )
    rows = [
        "# task_id\tcheckpoint\tcommand",
        (
            "tuning_phase6_dp_experts_cifar10_full_seed42\t{}\t{} {} "
            "--dataset cifar10 --output_dir {} --num_parties 5 "
            "--num_experts_per_party 2 --party_ids all --epochs 50 "
            "--batch_size 256 --learning_rate 0.05 --momentum 0.0 "
            "--weight_decay 0.0 --noise_multiplier 1.5 --max_grad_norm 1.0 "
            "--delta 1e-5 --seed 42 --device cuda --secure_rng"
        ).format(
            expert_checkpoint,
            PYTHON,
            ROOT / "new_stage_a_dp_experts.py",
            TRAJECTORY_ROOT,
        ),
        (
            "tuning_phase6_dp_mtt_cifar10_full_seed42\t{}\t{} {} "
            "--dataset cifar10 --trajectory_root {} --output_dir {} "
            "--ipc 10 --iterations 5000 --syn_steps 20 --expert_epochs 3 "
            "--max_start_epoch 20 --lr_img 1000 --lr_lr 1e-5 "
            "--lr_teacher 0.01 --checkpoint_interval 100 --seed 42 --device cuda"
        ).format(
            distill_checkpoint,
            PYTHON,
            ROOT / "new_stage_a_dp_distill.py",
            TRAJECTORY_ROOT,
            DISTILLED_ROOT,
        ),
        (
            "tuning_phase6_dp_encoder_cifar10_reference_seed42\t{}\t{} {} "
            "--dataset cifar10 --track dp_mtt --dp_root {} --dp_ipc 10 "
            "--output_dir {} --epochs 500 --batch_size 128 --eval_batch_size 256 "
            "--learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 "
            "--feature_norm_bound 1.0 --holdout_per_class 1 --seed 42 "
            "--device cuda --augmentation"
        ).format(
            encoder_checkpoint,
            PYTHON,
            ROOT / "new_stage_a_shared_encoder.py",
            DISTILLED_ROOT,
            ENCODER_ROOT,
        ),
        (
            "prepare_phase6_dp_validation_final\t{}\t{} {} --activate"
        ).format(
            ROOT / "new_log/new_secure_feature/json/new_phase6_dp_validation_final_plan.json",
            PYTHON,
            ROOT / "new_prepare_phase6_dp_validation_final.py",
        ),
    ]
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    PLAN.write_text(
        json.dumps(
            {
                "schema": "new_phase6_dp_validation_plan_v1",
                "dataset": "cifar10",
                "seed": 42,
                "cpu_threads": 1,
                "num_workers": 0,
                "visible_gpu": 0,
                "test_loaded": False,
                "expert_subruns": 10,
                "expert_epochs": 50,
                "distill_iterations": 5000,
                "ipc": 10,
                "queue": str(QUEUE),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.activate:
        ACTIVE.write_text(str(QUEUE) + "\n", encoding="utf-8")
    print("Phase 6 DP-MTT validation queue")
    print("dataset=cifar10 seed=42 experts=10x50epochs distill_iterations=5000")
    print("cpu_threads=1 num_workers=0 visible_gpu=0 test_loaded=False")
    print("queue={}".format(QUEUE))
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用途：依据 DP-MTT reference pilot 冻结一次 CIFAR-10 readonly test 验证。"""

import argparse
import json
import math
import os
from pathlib import Path

import torch


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
PYTHON = os.environ.get(
    "NEW_MODERN_PYTHON", str(ROOT.parent / "new_envs" / "distillation" / "bin" / "python")
)
BASE = ROOT / "new_cache/new_secure_feature/new_tuning_phase6_dp_mtt_cifar10_round2"
PILOT = (
    BASE
    / "new_seed42/new_encoder_pilot/dp_mtt/cifar10"
    / "new_shared_encoder_checkpoint_cifar10_dp_mtt_seed42.pt"
)
DISTILLED_ROOT = BASE / "new_seed42/new_distilled"
FORMAL_ROOT = BASE / "new_seed42/new_encoder_frozen"
QUEUE = ROOT / "new_configs/new_secure_feature/new_gpu0_phase6_dp_validation_final.tsv"
PLAN = ROOT / "new_log/new_secure_feature/json/new_phase6_dp_validation_final_plan.json"
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
STATUS = (
    ROOT
    / "new_log/new_secure_feature/status"
    / "new_tuning_phase6_dp_encoder_cifar10_reference_seed42.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结 DP-MTT validation encoder")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE":
        raise RuntimeError("DP encoder pilot 尚未完成")
    payload = torch.load(str(PILOT), map_location="cpu", weights_only=False)
    history = payload["history"]
    best = -1.0
    stagnant = 0
    plateau = int(history[-1]["epoch"])
    for row in history:
        score = float(row["reference_accuracy"])
        if score - best >= 0.002:
            best = score
            stagnant = 0
        else:
            stagnant += 1
        if int(row["epoch"]) >= 240 and stagnant >= 5:
            plateau = int(row["epoch"])
            break
    formal_epochs = max(300, min(500, int(math.ceil(plateau * 1.25 / 10.0) * 10)))
    checkpoint = (
        FORMAL_ROOT
        / "dp_mtt/cifar10/new_shared_encoder_checkpoint_cifar10_dp_mtt_seed42.pt"
    )
    report = ROOT / "new_log/new_secure_feature/json/new_phase6_dp_validation_result.json"
    rows = [
        "# task_id\tcheckpoint\tcommand",
        (
            "tuning_phase6_dp_encoder_cifar10_frozen_seed42\t{}\t{} {} "
            "--dataset cifar10 --track dp_mtt --dp_root {} --dp_ipc 10 "
            "--output_dir {} --epochs {} --batch_size 128 --eval_batch_size 256 "
            "--learning_rate 0.01 --momentum 0.9 --weight_decay 0.0005 "
            "--feature_norm_bound 1.0 --holdout_per_class 1 "
            "--encoder_use_all_synthetic --seed 42 --device cuda "
            "--augmentation --include_test"
        ).format(
            checkpoint,
            PYTHON,
            ROOT / "new_stage_a_shared_encoder.py",
            DISTILLED_ROOT,
            FORMAL_ROOT,
            formal_epochs,
        ),
        (
            "report_phase6_dp_validation\t{}\t{} {}"
        ).format(
            report,
            PYTHON,
            ROOT / "new_report_phase6_dp_validation.py",
        ),
    ]
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    PLAN.write_text(
        json.dumps(
            {
                "schema": "new_phase6_dp_validation_final_plan_v1",
                "selection_source": "synthetic_reference_only",
                "test_tuning": False,
                "detected_plateau_epoch": plateau,
                "formal_epochs": formal_epochs,
                "queue": str(QUEUE),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.activate:
        ACTIVE.write_text(str(QUEUE) + "\n", encoding="utf-8")
    print(
        "DP-MTT frozen validation: plateau={} formal_epochs={} test_tuning=False".format(
            plateau, formal_epochs
        )
    )
    print("queue={}".format(QUEUE))
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

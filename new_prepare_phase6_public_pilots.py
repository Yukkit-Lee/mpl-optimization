#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：生成 Phase 6 缺失的 official MTT validation pilot 单 GPU 队列。

CIFAR-10 已有通过门槛的三个正式 seed，因此不会重跑。本脚本只生成
CIFAR-100 和 Tiny ImageNet 的 reference-only pilot；pilot 不读取 test，
用于依据平台期冻结各数据集正式 epoch，不能计入正式均值。
"""

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
PYTHON = os.environ.get(
    "NEW_MODERN_PYTHON", "/home/lixujie/.conda/envs/distillation/bin/python"
)
QUEUE = ROOT / "new_configs/new_secure_feature/new_gpu0_phase6_public_pilots.tsv"
PLAN = ROOT / "new_log/new_secure_feature/json/new_phase6_public_pilot_plan.json"
CACHE = ROOT / "new_cache/new_secure_feature/new_tuning_phase6_public_pilots"

PILOTS = {
    "cifar100": {
        "epochs": 500,
        "batch_size": 256,
        "eval_batch_size": 512,
        "learning_rate": 0.01,
    },
    "tiny": {
        "epochs": 700,
        "batch_size": 128,
        "eval_batch_size": 256,
        "learning_rate": 0.01,
    },
}


def main() -> int:
    rows = ["# task_id\tcheckpoint\tcommand"]
    resolved = []
    for dataset, cfg in PILOTS.items():
        task_id = "tuning_phase6_public_{}_plateau_seed42".format(dataset)
        output = CACHE / "new_{}_seed42".format(dataset)
        checkpoint = (
            output
            / "public_mtt_benchmark"
            / dataset
            / "new_shared_encoder_checkpoint_{}_public_mtt_benchmark_seed42.pt".format(
                dataset
            )
        )
        command = [
            PYTHON,
            str(ROOT / "new_stage_a_shared_encoder.py"),
            "--dataset",
            dataset,
            "--track",
            "public_mtt_benchmark",
            "--output_dir",
            str(output),
            "--epochs",
            str(cfg["epochs"]),
            "--batch_size",
            str(cfg["batch_size"]),
            "--eval_batch_size",
            str(cfg["eval_batch_size"]),
            "--learning_rate",
            str(cfg["learning_rate"]),
            "--momentum",
            "0.9",
            "--weight_decay",
            "0.0005",
            "--feature_norm_bound",
            "1.0",
            "--holdout_per_class",
            "1",
            "--seed",
            "42",
            "--device",
            "cuda",
            "--augmentation",
        ]
        rows.append("{}\t{}\t{}".format(task_id, checkpoint, " ".join(command)))
        resolved.append(
            {
                "task_id": task_id,
                "dataset": dataset,
                "seed": 42,
                "purpose": "reference_only_plateau_detection",
                "test_loaded": False,
                "checkpoint": str(checkpoint),
                "configuration": cfg,
            }
        )
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    PLAN.write_text(
        json.dumps(
            {
                "schema": "new_phase6_public_pilot_plan_v1",
                "cpu_threads": 1,
                "num_workers": 0,
                "visible_gpu": 0,
                "tasks": resolved,
                "queue": str(QUEUE),
                "already_complete_and_skipped": {
                    "dataset": "cifar10",
                    "formal_round": 3,
                    "seeds": [42, 43, 44],
                    "mean_test_accuracy": 0.6782666666666667,
                    "quality_gate": 0.6684,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Phase 6 public validation pilot")
    print("cpu_threads=1 num_workers=0 visible_gpu=0")
    print("tasks={} (CIFAR-10 formal Round 3 skipped)".format(len(resolved)))
    print("queue={}".format(QUEUE))
    print("plan={}".format(PLAN))
    for item in resolved:
        print(
            "  {dataset}: seed={seed} max_epochs={configuration[epochs]} "
            "test_loaded={test_loaded}".format(**item)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

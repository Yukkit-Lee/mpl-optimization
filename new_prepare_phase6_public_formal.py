#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：根据 reference-only pilot 冻结 CIFAR-100/Tiny public MTT 正式 epoch。

本脚本不读取 test 指标。正式运行随后使用全部 official synthetic images 训练
共享 encoder，并只在冻结配置完成后进行一次 readonly test evaluation。
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
PYTHON = os.environ.get(
    "NEW_MODERN_PYTHON", "/home/lixujie/.conda/envs/distillation/bin/python"
)
PILOT_ROOT = (
    ROOT / "new_cache/new_secure_feature/new_tuning_phase6_public_pilots"
)
FORMAL_ROOT = ROOT / "new_cache/new_secure_feature/new_formal_phase6/new_public_mtt_final"
QUEUE = ROOT / "new_configs/new_secure_feature/new_gpu0_phase6_public_formal.tsv"
PLAN = ROOT / "new_log/new_secure_feature/json/new_phase6_public_formal_plan.json"
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
STATUS_ROOT = ROOT / "new_log/new_secure_feature/status"

DATASETS = {
    "cifar100": {"pilot_epochs": 500, "batch_size": 256, "eval_batch_size": 512},
    "tiny": {"pilot_epochs": 700, "batch_size": 128, "eval_batch_size": 256},
}
OFFICIAL_REFERENCE = {"cifar10": 0.6884, "cifar100": 0.4587, "tiny": 0.2558}


def require_complete(task_id: str) -> None:
    path = STATUS_ROOT / "new_{}.json".format(task_id)
    if not path.exists():
        raise RuntimeError("pilot 状态不存在: {}".format(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise RuntimeError("pilot 尚未完成: {} status={}".format(task_id, payload.get("status")))


def plateau_epoch(history, minimum_epoch=240, intervals=5, threshold=0.002):
    """按 reference running-best 连续五个 interval 无显著提升检测平台期。"""

    best = -1.0
    stagnant = 0
    plateau = int(history[-1]["epoch"])
    for row in history:
        epoch = int(row["epoch"])
        score = float(row["reference_accuracy"])
        improvement = score - best
        if improvement >= threshold:
            best = score
            stagnant = 0
        else:
            stagnant += 1
        if epoch >= minimum_epoch and stagnant >= intervals:
            plateau = epoch
            break
    return plateau


def freeze_epochs(dataset: str, history) -> int:
    plateau = plateau_epoch(history)
    proposed = int(math.ceil(plateau * 1.25 / 10.0) * 10)
    lower = 300
    upper = DATASETS[dataset]["pilot_epochs"]
    return max(lower, min(upper, proposed))


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结并生成 Phase 6 public formal 队列")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    rows = ["# task_id\tcheckpoint\tcommand"]
    decisions = []
    for dataset, cfg in DATASETS.items():
        pilot_task = "tuning_phase6_public_{}_plateau_seed42".format(dataset)
        require_complete(pilot_task)
        pilot_checkpoint = (
            PILOT_ROOT
            / "new_{}_seed42".format(dataset)
            / "public_mtt_benchmark"
            / dataset
            / "new_shared_encoder_checkpoint_{}_public_mtt_benchmark_seed42.pt".format(
                dataset
            )
        )
        payload = torch.load(str(pilot_checkpoint), map_location="cpu", weights_only=False)
        history = payload["history"]
        formal_epochs = freeze_epochs(dataset, history)
        decision = {
            "dataset": dataset,
            "pilot_task": pilot_task,
            "pilot_test_loaded": False,
            "pilot_reference_best": max(
                float(row["reference_accuracy"]) for row in history
            ),
            "detected_plateau_epoch": plateau_epoch(history),
            "formal_epochs": formal_epochs,
            "official_reference_accuracy": OFFICIAL_REFERENCE[dataset],
            "strict_quality_gate": max(
                0.95 * OFFICIAL_REFERENCE[dataset],
                OFFICIAL_REFERENCE[dataset] - 0.02,
            ),
        }
        decisions.append(decision)
        for seed in (42, 43, 44):
            task_id = "formal_phase6_public_{}_seed{}".format(dataset, seed)
            output = FORMAL_ROOT / "new_{}_seed{}".format(dataset, seed)
            checkpoint = (
                output
                / "public_mtt_benchmark"
                / dataset
                / "new_shared_encoder_checkpoint_{}_public_mtt_benchmark_seed{}.pt".format(
                    dataset, seed
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
                str(formal_epochs),
                "--batch_size",
                str(cfg["batch_size"]),
                "--eval_batch_size",
                str(cfg["eval_batch_size"]),
                "--learning_rate",
                "0.01",
                "--momentum",
                "0.9",
                "--weight_decay",
                "0.0005",
                "--feature_norm_bound",
                "1.0",
                "--holdout_per_class",
                "1",
                "--encoder_use_all_synthetic",
                "--seed",
                str(seed),
                "--device",
                "cuda",
                "--augmentation",
                "--include_test",
            ]
            rows.append("{}\t{}\t{}".format(task_id, checkpoint, " ".join(command)))
    report_path = ROOT / "new_log/new_secure_feature/json/new_phase6_public_mtt_results.json"
    rows.append(
        "report_phase6_public_mtt\t{}\t{} {}".format(
            report_path,
            PYTHON,
            ROOT / "new_report_phase6_public.py",
        )
    )
    dp_plan = ROOT / "new_log/new_secure_feature/json/new_phase6_dp_validation_plan.json"
    rows.append(
        "prepare_phase6_dp_validation_queue\t{}\t{} {} --activate".format(
            dp_plan,
            PYTHON,
            ROOT / "new_prepare_phase6_dp_validation.py",
        )
    )
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    PLAN.write_text(
        json.dumps(
            {
                "schema": "new_phase6_public_formal_plan_v1",
                "selection_source": "reference_only_pilot",
                "test_tuning": False,
                "cpu_threads": 1,
                "num_workers": 0,
                "visible_gpu": 0,
                "already_complete": {
                    "cifar10": {
                        "seeds": [42, 43, 44],
                        "formal_round": 3,
                        "formal_epochs": 300,
                        "mean_test_accuracy": 0.6782666666666667,
                    }
                },
                "decisions": decisions,
                "formal_task_count": 8,
                "queue": str(QUEUE),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.activate:
        ACTIVE.write_text(str(QUEUE) + "\n", encoding="utf-8")
    print("Phase 6 public formal freeze")
    print("cpu_threads=1 num_workers=0 visible_gpu=0 formal_tasks=6")
    for item in decisions:
        print(
            "  {dataset}: plateau={detected_plateau_epoch} formal_epochs={formal_epochs} "
            "ref_best={pilot_reference_best:.4f} gate={strict_quality_gate:.4f}".format(
                **item
            )
        )
    print("queue={}".format(QUEUE))
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

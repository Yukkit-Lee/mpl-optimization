#!/usr/bin/env python3
"""只调优Adaptive-DP，不运行或修改Fixed-DP。

运行方式（仓库根目录，建议放入tmux）：

    python new_scripts/new_paper_adaptive/new_tune_adaptive_only_latest_initializer.py \
        --workers 24

参数网格读取：

    new_configs/new_paper_adaptive/new_adaptive_independent_tuning_freeze.json

输出：

    new_log/new_paper_adaptive/new_adaptive_independent_screen.json
    new_expTab/new_paper_adaptive/new_adaptive_independent_screen.csv

本脚本只读development-validation。Frozen Fixed仅作为元数据边界，不参与候选
选择；Adaptive每个候选按自己的steps、batch和count releases重新校准epsilon。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_scripts.new_paper_adaptive.new_screen_latest_initializer_fixed_vs_adaptive import (  # noqa: E402
    load_inputs,
    run_candidate,
)


FREEZE = ROOT / "new_configs/new_paper_adaptive/new_adaptive_independent_tuning_freeze.json"
OUT_JSON = ROOT / "new_log/new_paper_adaptive/new_adaptive_independent_screen.json"
OUT_CSV = ROOT / "new_expTab/new_paper_adaptive/new_adaptive_independent_screen.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose(rows):
    """使用冻结规则选唯一最佳点，绝不参考Fixed最终准确率。"""
    return max(
        rows,
        key=lambda row: (
            row["final_development_validation_accuracy"],
            row["early_auc"],
            -row["runtime_seconds"],
        ),
    )


def run_parallel(configs, weight, bias, data, workers):
    """并行运行独立候选；workers只影响速度，不改变候选参数。"""
    payloads = [(config, weight, bias, data) for config in configs]
    with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
        return list(executor.map(run_candidate, payloads))


def flatten(stage_name, row):
    """将候选的完整可调参数和结果写入CSV，便于人工筛选/复核。"""
    config = row["config"]
    return {
        "stage": stage_name,
        "steps": config["steps"],
        "learning_rate": config["learning_rate"],
        "expected_batch_size": config["expected_batch_size"],
        "initial_c": config["clip_c"],
        "gamma": config["gamma"],
        "eta_c": config["eta_c"],
        "warmup_steps": config["warmup_steps"],
        "update_interval": config["update_interval"],
        "count_noise_multiplier": config["count_noise_multiplier"],
        "final_accuracy": row["final_development_validation_accuracy"],
        "best_accuracy": row["best_development_validation_accuracy"],
        "early_auc": row["early_auc"],
        "final_c": row["final_c"],
        "gradient_noise_multiplier": row["noise_multiplier"],
        "accounted_epsilon": row["accounted_epsilon"],
        "count_releases": row["count_releases"],
        "runtime_seconds": row["runtime_seconds"],
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptive-only独立参数筛选")
    parser.add_argument(
        "--workers", type=int, default=24,
        help="并行候选数；当前32 CPU配额建议24，未限制BLAS为单线程",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers必须大于0")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_ADAPTIVE_ONLY_SCREEN":
        raise ValueError("Adaptive-only配置未在筛选前冻结")
    if freeze["official_test_read"] or freeze["locked_holdout_read"]:
        raise ValueError("筛选配置越过了test/holdout边界")
    model_path, cache_path, weight, bias, data = load_inputs(freeze)

    base = dict(freeze["adaptive_defaults"])
    base.update(freeze["shared_privacy_only"])
    base["rng_seed"] = int(freeze["screen_rng_seed"])
    stages = {}

    # 第一阶段：Adaptive独立选择训练预算、学习率和期望batch。
    configs = []
    stage = freeze["stages"]["adaptive_budget_lr_batch"]
    for steps in stage["steps"]:
        for learning_rate in stage["learning_rate"]:
            for expected_batch_size in stage["expected_batch_size"]:
                configs.append(
                    dict(
                        base,
                        steps=int(steps),
                        learning_rate=float(learning_rate),
                        expected_batch_size=int(expected_batch_size),
                    )
                )
    stages["adaptive_budget_lr_batch"] = run_parallel(
        configs, weight, bias, data, args.workers
    )
    budget_best = choose(stages["adaptive_budget_lr_batch"])

    # 第二阶段：固定第一阶段胜者，只搜索Adaptive控制器本身。
    configs = []
    stage = freeze["stages"]["adaptive_controller"]
    for initial_c in stage["initial_c"]:
        for gamma in stage["gamma"]:
            for eta_c in stage["eta_c"]:
                configs.append(
                    dict(
                        budget_best["config"],
                        clip_c=float(initial_c),
                        gamma=float(gamma),
                        eta_c=float(eta_c),
                    )
                )
    stages["adaptive_controller"] = run_parallel(
        configs, weight, bias, data, args.workers
    )
    controller_best = choose(stages["adaptive_controller"])

    # 第三阶段：选择C更新频率与count机制噪声；它们会改变隐私组成。
    configs = []
    stage = freeze["stages"]["adaptive_update_schedule"]
    for warmup_steps in stage["warmup_steps"]:
        for update_interval in stage["update_interval"]:
            for count_noise_multiplier in stage["count_noise_multiplier"]:
                configs.append(
                    dict(
                        controller_best["config"],
                        warmup_steps=int(warmup_steps),
                        update_interval=int(update_interval),
                        count_noise_multiplier=float(count_noise_multiplier),
                    )
                )
    stages["adaptive_update_schedule"] = run_parallel(
        configs, weight, bias, data, args.workers
    )
    best = choose(stages["adaptive_update_schedule"])

    all_rows = [
        flatten(stage_name, row)
        for stage_name, rows in stages.items()
        for row in rows
    ]
    payload = {
        "schema": "new_adaptive_independent_screen_v1",
        "status": "COMPLETE",
        "freeze": str(FREEZE.relative_to(ROOT)),
        "freeze_sha256": sha256(FREEZE),
        "initializer": str(model_path.relative_to(ROOT)),
        "initializer_sha256": sha256(model_path),
        "feature_cache": str(cache_path.relative_to(ROOT)),
        "feature_cache_sha256": sha256(cache_path),
        "fixed_immutable": freeze["fixed_immutable"],
        "selection_uses_fixed_result": False,
        "candidate_count": len(all_rows),
        "best_adaptive": best,
        "stages": stages,
        "official_test_read": False,
        "locked_holdout_read": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    print(
        json.dumps(
            {
                "candidate_count": len(all_rows),
                "best_adaptive": flatten("winner", best),
                "output": str(OUT_JSON.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

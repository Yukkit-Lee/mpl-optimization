#!/usr/bin/env python3
"""运行用户自行配置的 Fixed DP 与 Adaptive DP Figure 6 消融。

本脚本不冻结任何一方的超参数，也不复用旧 Fixed 结果。两种方法都会按照
``new_user_editable_figure6_config.py`` 中的参数重新运行，并针对各自的 steps、
batch 和 Adaptive count releases 单独校准梯度噪声，使最终 epsilon 相同。

运行方式（仓库根目录）：

    tmux new -s figure6_user_tune
    python new_scripts/new_paper_adaptive/new_run_user_configurable_fixed_vs_adaptive.py \
        --workers 6

``--workers`` 是同时运行的任务数。默认最多并行六个 mode-seed 任务；它只影响
速度，不改变实验参数。正式运行前请先检查 CPU、内存、磁盘，并保留 tmux 日志。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


# 将仓库根目录加入模块搜索路径，保证从任意当前目录执行都能导入训练器。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_scripts.new_paper_adaptive.new_screen_latest_initializer_fixed_vs_adaptive import (  # noqa: E402
    load_inputs,
    run_candidate,
)
from new_scripts.new_paper_adaptive.new_user_editable_figure6_config import (  # noqa: E402
    ADAPTIVE_CONFIG,
    FEATURE_CACHE,
    FIXED_CONFIG,
    INITIALIZER,
    OUTPUT_CSV,
    OUTPUT_JSON,
    RNG_SEEDS,
)


# 训练器真正需要的全部字段。缺字段时立即报错，避免跑几小时后才发现配置不完整。
REQUIRED_KEYS = {
    "mode", "steps", "learning_rate", "expected_batch_size", "clip_c",
    "gamma", "eta_c", "warmup_steps", "update_interval",
    "count_noise_multiplier", "c_min", "c_max", "target_epsilon", "delta",
}


def validate_config(name: str, config: dict, expected_mode: str) -> None:
    """检查用户参数的基本范围；不限制 Fixed 与 Adaptive 使用相同超参数。"""
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"{name} 缺少字段: {sorted(missing)}")
    if config["mode"] != expected_mode:
        raise ValueError(f"{name}.mode 必须是 {expected_mode}")
    if int(config["steps"]) <= 0 or int(config["expected_batch_size"]) <= 0:
        raise ValueError(f"{name} 的 steps 和 expected_batch_size 必须大于0")
    if float(config["learning_rate"]) <= 0 or float(config["clip_c"]) <= 0:
        raise ValueError(f"{name} 的 learning_rate 和 clip_c 必须大于0")
    if not 0.0 < float(config["delta"]) < 1.0:
        raise ValueError(f"{name}.delta 必须位于(0, 1)")
    if expected_mode == "adaptive_dp":
        if not 0.0 <= float(config["gamma"]) <= 1.0:
            raise ValueError("ADAPTIVE_CONFIG.gamma 必须位于[0, 1]")
        if int(config["update_interval"]) <= 0:
            raise ValueError("ADAPTIVE_CONFIG.update_interval 必须大于0")
        if float(config["c_min"]) <= 0 or float(config["c_max"]) < float(config["c_min"]):
            raise ValueError("Adaptive 的 C 范围无效")


def make_input_descriptor() -> dict:
    """构造 load_inputs 所需的只读输入描述，并明确禁止 official test。"""
    return {
        "initializer": INITIALIZER,
        "feature_cache": FEATURE_CACHE,
    }


def make_tasks(weight, bias, data):
    """为每个 seed 分别创建 Fixed 和 Adaptive 任务，两边参数互不覆盖。"""
    tasks = []
    for seed in RNG_SEEDS:
        for base in (FIXED_CONFIG, ADAPTIVE_CONFIG):
            config = dict(base)       # 复制，避免训练器意外修改用户配置对象。
            config["rng_seed"] = int(seed)
            tasks.append((config, weight, bias, data))
    return tasks


def result_row(config: dict, result: dict) -> dict:
    """提取适合写入 CSV 和汇总比较的标量指标。"""
    return {
        "seed": int(config["rng_seed"]),
        "mode": config["mode"],
        "steps": int(config["steps"]),
        "learning_rate": float(config["learning_rate"]),
        "expected_batch_size": int(config["expected_batch_size"]),
        "initial_c": float(config["clip_c"]),
        "final_accuracy": result["final_development_validation_accuracy"],
        "best_accuracy": result["best_development_validation_accuracy"],
        "early_auc": result["early_auc"],
        "final_c": result["final_c"],
        "noise_multiplier": result["noise_multiplier"],
        "accounted_epsilon": result["accounted_epsilon"],
        "count_releases": result["count_releases"],
        "runtime_seconds": result["runtime_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="用户可配置的 Figure 6 Fixed/Adaptive 正式实验")
    parser.add_argument(
        "--workers", type=int, default=6,
        help="并行任务数；三seed×两方法最多6个任务，单seed建议设2",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="只检查参数并打印任务摘要，不读取数据、不启动训练",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers 必须大于0")

    # 这里只要求相同隐私预算；steps、LR、batch、C 等均允许不同。
    validate_config("FIXED_CONFIG", FIXED_CONFIG, "fixed_dp")
    validate_config("ADAPTIVE_CONFIG", ADAPTIVE_CONFIG, "adaptive_dp")
    if float(FIXED_CONFIG["target_epsilon"]) != float(ADAPTIVE_CONFIG["target_epsilon"]):
        raise ValueError("公平比较要求 Fixed 与 Adaptive 的 target_epsilon 相同")
    if float(FIXED_CONFIG["delta"]) != float(ADAPTIVE_CONFIG["delta"]):
        raise ValueError("公平比较要求 Fixed 与 Adaptive 的 delta 相同")
    if not RNG_SEEDS or len(set(RNG_SEEDS)) != len(RNG_SEEDS):
        raise ValueError("RNG_SEEDS 不能为空或包含重复值")

    if args.check_only:
        print(json.dumps({
            "status": "CONFIG_VALID",
            "rng_seeds": RNG_SEEDS,
            "fixed_config": FIXED_CONFIG,
            "adaptive_config": ADAPTIVE_CONFIG,
            "task_count": 2 * len(RNG_SEEDS),
            "training_started": False,
        }, ensure_ascii=False, indent=2))
        return

    model_path, cache_path, weight, bias, data = load_inputs(make_input_descriptor())
    tasks = make_tasks(weight, bias, data)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        results = list(executor.map(run_candidate, tasks))

    rows = []
    detailed = []
    for (config, _, _, _), result in zip(tasks, results):
        row = result_row(config, result)
        # run_candidate 会对每套参数重新会计；这里再核验实际 epsilon 没有漂移。
        target = float(config["target_epsilon"])
        if abs(float(row["accounted_epsilon"]) - target) > 1e-8:
            raise ValueError(f"{config['mode']} seed={config['rng_seed']} 的 epsilon 校准失败")
        rows.append(row)
        detailed.append({"row": row, "config": config, "history": result["history"]})

    # 同 seed 比较可以减少随机性造成的干扰，但不要求两种方法参数相同。
    by_seed = {}
    for seed in RNG_SEEDS:
        fixed = next(row for row in rows if row["seed"] == seed and row["mode"] == "fixed_dp")
        adaptive = next(row for row in rows if row["seed"] == seed and row["mode"] == "adaptive_dp")
        by_seed[str(seed)] = {
            "fixed": fixed,
            "adaptive": adaptive,
            "adaptive_minus_fixed_pp": 100.0 * (
                adaptive["final_accuracy"] - fixed["final_accuracy"]
            ),
            "adaptive_minus_fixed_early_auc_pp": 100.0 * (
                adaptive["early_auc"] - fixed["early_auc"]
            ),
        }

    gains = [value["adaptive_minus_fixed_pp"] for value in by_seed.values()]
    auc_gains = [value["adaptive_minus_fixed_early_auc_pp"] for value in by_seed.values()]
    payload = {
        "schema": "new_user_configurable_fixed_vs_adaptive_v1",
        "status": "COMPLETE",
        "comparison_scope": "same epsilon/delta/data/initializer/evaluation; all optimizer and compute parameters independently user-configurable",
        "initializer": str(model_path.relative_to(ROOT)),
        "feature_cache": str(cache_path.relative_to(ROOT)),
        "rng_seeds": list(RNG_SEEDS),
        "fixed_config": dict(FIXED_CONFIG),
        "adaptive_config": dict(ADAPTIVE_CONFIG),
        "by_seed": by_seed,
        "mean_adaptive_minus_fixed_pp": float(np.mean(gains)),
        # n=1 诊断没有可定义的样本标准差，因此明确写 null，不伪造误差条。
        "sample_std_adaptive_minus_fixed_pp": (
            float(np.std(gains, ddof=1)) if len(gains) >= 2 else None
        ),
        "mean_adaptive_minus_fixed_early_auc_pp": float(np.mean(auc_gains)),
        "detailed": detailed,
        "official_test_read": False,
        "locked_holdout_read": False,
    }

    out_json = ROOT / OUTPUT_JSON
    out_csv = ROOT / OUTPUT_CSV
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "by_seed": by_seed,
        "mean_adaptive_minus_fixed_pp": payload["mean_adaptive_minus_fixed_pp"],
        "sample_std_adaptive_minus_fixed_pp": payload["sample_std_adaptive_minus_fixed_pp"],
        "output_json": OUTPUT_JSON,
        "output_csv": OUTPUT_CSV,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

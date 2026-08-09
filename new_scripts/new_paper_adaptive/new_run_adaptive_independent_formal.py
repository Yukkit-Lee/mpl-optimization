#!/usr/bin/env python3
"""运行独立调优后的Adaptive三seed正式复核，并复用冻结Fixed结果。

运行方式（必须先生成并提交formal freeze）：

    python new_scripts/new_paper_adaptive/new_run_adaptive_independent_formal.py \
        --workers 3

本脚本不会重新运行Fixed-DP。它逐项核验旧Fixed配置仍为steps=320、LR=0.001、
C=2.0，再只运行Adaptive seeds 42/43/44。输出使用独立文件，不覆盖旧matched实验。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_scripts.new_paper_adaptive.new_screen_latest_initializer_fixed_vs_adaptive import (  # noqa: E402
    load_inputs,
    run_candidate,
)


FREEZE = ROOT / "new_configs/new_paper_adaptive/new_adaptive_independent_formal_freeze.json"
INPUT_FREEZE = ROOT / "new_configs/new_paper_adaptive/new_adaptive_independent_tuning_freeze.json"
OUT_JSON = ROOT / "new_log/new_paper_adaptive/new_adaptive_independent_formal.json"
OUT_CSV = ROOT / "new_expTab/new_paper_adaptive/new_adaptive_independent_formal.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_and_copy_fixed(source_payload, freeze):
    """复制旧Fixed证据并硬拒绝任何Fixed参数变化。"""
    expected = freeze["fixed_immutable"]
    fixed_items = [
        item for item in source_payload["detailed"]
        if item["row"]["mode"] == "fixed_dp"
    ]
    if len(fixed_items) != 3:
        raise ValueError("旧正式结果中Fixed不是严格3个seed")
    for item in fixed_items:
        config = item["config"]
        if int(config["rng_seed"]) not in expected["rng_seeds"]:
            raise ValueError("Fixed seed发生变化")
        for key in ("steps", "learning_rate", "clip_c"):
            if float(config[key]) != float(expected[key]):
                raise ValueError("Fixed参数发生变化: {}".format(key))
        if abs(float(item["row"]["accounted_epsilon"]) - 3.0) > 1e-8:
            raise ValueError("旧Fixed epsilon不是3")
    return sorted(fixed_items, key=lambda item: item["row"]["seed"])


def main():
    parser = argparse.ArgumentParser(description="Adaptive独立最优配置三seed复核")
    parser.add_argument("--workers", type=int, default=3, help="三个formal seed并行")
    args = parser.parse_args()
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_ADAPTIVE_FORMAL":
        raise ValueError("formal配置未在运行前冻结")
    screen_path = ROOT / freeze["source_screen"]
    fixed_path = ROOT / freeze["fixed_result_source"]
    if sha256(screen_path) != freeze["source_screen_sha256"]:
        raise ValueError("Adaptive screen来源hash不匹配")
    if sha256(fixed_path) != freeze["fixed_result_source_sha256"]:
        raise ValueError("Frozen Fixed来源hash不匹配")
    input_freeze = json.loads(INPUT_FREEZE.read_text(encoding="utf-8"))
    model_path, cache_path, weight, bias, data = load_inputs(input_freeze)
    fixed_source = json.loads(fixed_path.read_text(encoding="utf-8"))
    fixed_items = validate_and_copy_fixed(fixed_source, freeze)

    configs = []
    for seed in freeze["formal_rng_seeds"]:
        config = dict(freeze["adaptive_config"])
        config["rng_seed"] = int(seed)
        configs.append(config)
    payloads = [(config, weight, bias, data) for config in configs]
    with ProcessPoolExecutor(max_workers=min(args.workers, 3)) as executor:
        adaptive_results = list(executor.map(run_candidate, payloads))

    adaptive_items = []
    rows = []
    for config, result in zip(configs, adaptive_results):
        row = {
            "seed": config["rng_seed"],
            "mode": "adaptive_dp",
            "final_accuracy": result["final_development_validation_accuracy"],
            "best_accuracy": result["best_development_validation_accuracy"],
            "early_auc": result["early_auc"],
            "final_c": result["final_c"],
            "noise_multiplier": result["noise_multiplier"],
            "accounted_epsilon": result["accounted_epsilon"],
            "count_releases": result["count_releases"],
            "runtime_seconds": result["runtime_seconds"],
        }
        if abs(float(row["accounted_epsilon"]) - 3.0) > 1e-8:
            raise ValueError("Adaptive epsilon没有校准到3")
        rows.append(row)
        adaptive_items.append({"row": row, "config": config, "history": result["history"]})

    fixed_rows = [item["row"] for item in fixed_items]
    by_seed = {}
    for seed in freeze["formal_rng_seeds"]:
        fixed = next(row for row in fixed_rows if int(row["seed"]) == int(seed))
        adaptive = next(row for row in rows if int(row["seed"]) == int(seed))
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
    gains = [item["adaptive_minus_fixed_pp"] for item in by_seed.values()]
    auc_gains = [
        item["adaptive_minus_fixed_early_auc_pp"] for item in by_seed.values()
    ]
    output = {
        "schema": "new_adaptive_independent_formal_v1",
        "status": "COMPLETE",
        "comparison_scope": "same epsilon/data/initializer/evaluation; independently tuned optimizer and compute parameters",
        "freeze": str(FREEZE.relative_to(ROOT)),
        "freeze_sha256": sha256(FREEZE),
        "fixed_result_source": str(fixed_path.relative_to(ROOT)),
        "fixed_rerun": False,
        "fixed_immutable_verified": True,
        "adaptive_screen_source": str(screen_path.relative_to(ROOT)),
        "initializer": str(model_path.relative_to(ROOT)),
        "feature_cache": str(cache_path.relative_to(ROOT)),
        "by_seed": by_seed,
        "mean_adaptive_minus_fixed_pp": float(np.mean(gains)),
        "sample_std_adaptive_minus_fixed_pp": float(np.std(gains, ddof=1)),
        "mean_adaptive_minus_fixed_early_auc_pp": float(np.mean(auc_gains)),
        "detailed": fixed_items + adaptive_items,
        "official_test_read": False,
        "locked_holdout_read": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_rows = fixed_rows + rows
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({
        "by_seed": by_seed,
        "mean_adaptive_minus_fixed_pp": output["mean_adaptive_minus_fixed_pp"],
        "sample_std_adaptive_minus_fixed_pp": output["sample_std_adaptive_minus_fixed_pp"],
        "output": str(OUT_JSON.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：解析并打印 new secure feature 正式实验配置、运行总量和输出路径。

本文件不启动训练。其输出是所有正式 tmux 队列的冻结输入，确保在正式运行前
明确数据集、seed、隐私点、资源限制和预期产物目录。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
OUTPUT_ROOT = Path(os.environ.get("NEW_OUTPUT_ROOT", str(PROJECT_ROOT)))
DEFAULT_CONFIG = PROJECT_ROOT / "new_configs" / "new_secure_feature" / "new_formal_config.json"
PLAN_JSON = OUTPUT_ROOT / "new_log" / "new_secure_feature" / "json" / "new_formal_resolved_plan.json"
PLAN_TXT = OUTPUT_ROOT / "new_log" / "new_secure_feature" / "txt" / "new_formal_resolved_plan.txt"


def calculate_run_counts(config: Dict) -> Dict[str, int]:
    datasets = len(config["datasets"])
    seeds = len(config["formal_seeds"])
    parties = int(config["stage_a_dp_mtt"]["num_parties"])
    experts = int(config["stage_a_dp_mtt"]["num_experts_per_party"])
    epsilon_points = len(config["stage_c_target_epsilons"])
    counts = {
        "stage_a_dp_experts": datasets * seeds * parties * experts,
        "stage_a_dp_mtt": datasets * seeds,
        "module_a_comparison": datasets * seeds * 4,
        # 18 个初始化收敛运行 + 81 个 Non-IID 方法运行。
        "module_b_formal": datasets * seeds * 2 + datasets * seeds * 3 * 3,
        # 每数据集/seed 一次 Plain，三个 epsilon 下各跑 Fixed/Adaptive，
        # 另加一次 Adaptive block-wise 消融。
        "module_c_formal": datasets * seeds * (1 + epsilon_points * 2 + 1),
        "end_to_end": datasets * seeds * 6,
        "tfe_system_scales": datasets * 3,
    }
    counts["minimum_total"] = sum(counts.values())
    return counts


def expected_paths() -> Dict[str, str]:
    return {
        "json": str(OUTPUT_ROOT / "new_log" / "new_secure_feature" / "json"),
        "txt": str(OUTPUT_ROOT / "new_log" / "new_secure_feature" / "txt"),
        "tables": str(OUTPUT_ROOT / "new_expTab" / "new_secure_feature"),
        "figures": str(OUTPUT_ROOT / "new_expFig" / "new_secure_feature"),
        "models": str(OUTPUT_ROOT / "new_init_model" / "new_secure_feature"),
        "cache": str(OUTPUT_ROOT / "new_cache" / "new_secure_feature"),
        "secret_sidecars": os.environ.get(
            "NEW_SECRET_ROOT",
            str(Path(os.environ.get("NEW_DATA_ROOT", "/home/lixujie/tmpWork"))
                / "new_secure_feature_secrets"),
        ),
    }


def render(payload: Dict) -> str:
    config = payload["config"]
    lines = [
        "=" * 96,
        "New Secure Feature Pipeline：正式实验冻结配置",
        "=" * 96,
        "datasets             : {}".format(config["datasets"]),
        "formal_seeds         : {}".format(config["formal_seeds"]),
        "tuning_seeds         : {}".format(config["tuning_seeds"]),
        "cpu_threads/process  : {}".format(config["cpu_threads"]),
        "DataLoader workers   : {}".format(config["num_workers"]),
        "GPU policy           : {}".format(config["gpu_policy"]),
        "feature_dim/norm     : {}/{}".format(
            config["feature_dim"], config["feature_l2_bound"]
        ),
        "Stage-C epsilon      : {}".format(config["stage_c_target_epsilons"]),
        "Stage-C delta        : {}".format(config["stage_c_delta"]),
        "secure RNG backend   : {}".format(
            config["stage_a_dp_mtt"]["secure_rng_backend"]
        ),
        "",
        "[正式运行数量；validation tuning trials 另计且不得删除]",
    ]
    for name, count in payload["run_counts"].items():
        lines.append("  {:24s}: {:4d}".format(name, count))
    lines.extend(["", "[预期输出路径]"])
    for name, path in payload["expected_paths"].items():
        lines.append("  {:24s}: {}".format(name, path))
    lines.extend(
        [
            "",
            "[执行约束]",
            "  正式任务仅通过 tmux + resource lock + heartbeat + checkpoint 启动。",
            "  数据集串行；仅使用 GPU 0 且同时最多一个训练进程；TFE 串行。",
            "  validation/reference 冻结超参数前禁止读取 test 指标。",
            "=" * 96,
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="打印正式实验冻结配置")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["cpu_threads"] != 1 or config["num_workers"] != 0:
        raise RuntimeError("正式配置必须 cpu_threads=1 且 num_workers=0")
    if len(set(config["formal_seeds"])) < 3:
        raise RuntimeError("正式实验至少需要 3 个不同 seed")
    if config.get("gpu_policy") != {
        "visible_devices": [0],
        "max_training_processes": 1,
    }:
        raise RuntimeError("本轮正式配置必须仅使用 GPU 0 且最多一个训练进程")
    payload = {
        "schema": "new_formal_resolved_plan_v1",
        "config_path": str(args.config.resolve()),
        "config": config,
        "run_counts": calculate_run_counts(config),
        "expected_paths": expected_paths(),
        "test_tuning_allowed": False,
    }
    text = render(payload)
    for path in (PLAN_JSON, PLAN_TXT):
        path.parent.mkdir(parents=True, exist_ok=True)
    PLAN_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    PLAN_TXT.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""只补齐 Tiny ImageNet End-to-End 缺失的 Full Data seed43/44。

当前 True Ours seed43/44 的 Stage B、Stage C 和 timeline 已经完成；缺失单元只有
Full Data seed43/44。本脚本会先验证已有 True Ours 单元，再顺序执行两个 Full
Data 单元。它不会重跑 seed42，也不会启动 CIFAR-10/CIFAR-100。

在仓库根目录运行：

    # 只做输入/冻结配置检查，不训练
    /root/autodl-tmp/new_envs/distillation/bin/python \
      new_scripts/new_true_ours/new_run_tiny_end2end_extra_seeds.py \
      --preflight-only

    # 正式续跑（请放入 tmux；脚本可按已有报告自动续跑/跳过）
    /root/autodl-tmp/new_envs/distillation/bin/python \
      new_scripts/new_true_ours/new_run_tiny_end2end_extra_seeds.py \
      --seeds 43 44

TFE 子进程固定使用项目已恢复的 ``new_envs/tfe/bin/python``。两个 Full Data
任务顺序执行，避免同时构建两个大型 TF1/TFE 图造成内存峰值叠加。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_scripts.new_true_ours.new_run_true_ours_formal_three_seed import (  # noqa: E402
    CONFIG_PATH,
    TFE_PYTHON,
    atomic_json,
    file_sha256,
    run_logged,
    task_paths,
    validate_stage_b,
    validate_stage_c,
    write_timeline,
)


DATASET = "tiny"
METHOD = "full_data"
ALLOWED_SEEDS = (43, 44)
NAMESPACE = "full_data_formal_corrected_v2"
STATUS_PATH = ROOT / "new_log/new_true_ours/new_tiny_end2end_extra_seeds_status.json"
FROZEN_CODE_COMMIT = "6973d2528af611c98b60176538e9f6139f7477ab"
FROZEN_STAGE_C_RELATIVE = "new_stage_c_secure_dpsgd.py"
FROZEN_RUNTIME_ROOT = ROOT.parent / "new_runtime/new_true_ours/frozen_6973"
FROZEN_ACCOUNTANT_SHA256 = (
    "c3459f375aeccbd931981261c61530a5178d18c04e981f62ddf8db1c2214e40b"
)


def frozen_stage_c_code(config: dict, materialize: bool = False) -> Path | None:
    """从Git对象读取seed42使用的Stage-C及其直接依赖，避免当前代码漂移。"""
    expected_by_file = {
        FROZEN_STAGE_C_RELATIVE: config["exact_code_sha256"][FROZEN_STAGE_C_RELATIVE],
        "new_stage_b_secure_init.py": config["exact_code_sha256"][
            "new_stage_b_secure_init.py"
        ],
        "new_privacy_accountant.py": FROZEN_ACCOUNTANT_SHA256,
    }
    contents = {}
    for relative, expected in expected_by_file.items():
        content = subprocess.check_output(
            ["git", "show", f"{FROZEN_CODE_COMMIT}:{relative}"], cwd=str(ROOT)
        )
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise ValueError(f"Git冻结代码SHA不匹配 {relative}: {actual} != {expected}")
        contents[relative] = content
    if not materialize:
        return None
    # 运行副本在仓库外的runtime目录，不纳入Git，也不改变当前开发文件。
    FROZEN_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    for relative, content in contents.items():
        path = FROZEN_RUNTIME_ROOT / relative
        path.write_bytes(content)
        if file_sha256(path) != expected_by_file[relative]:
            raise ValueError(f"落盘后的冻结代码SHA不匹配: {relative}")
    return FROZEN_RUNTIME_ROOT / FROZEN_STAGE_C_RELATIVE


def paths_for(seed: int) -> dict:
    """为每个额外seed建立独立输出/checkpoint/log命名空间。"""
    stage_b_output = (
        ROOT / "new_output/new_true_ours/stage_b" / NAMESPACE
        / DATASET / f"seed{seed}"
    )
    stage_c_output = (
        ROOT / "new_output/new_true_ours/stage_c" / NAMESPACE
        / DATASET / f"seed{seed}"
    )
    stage_b_checkpoint = (
        ROOT / "new_checkpoints/new_true_ours/stage_b" / NAMESPACE
        / DATASET / f"seed{seed}/new_stage_b"
    )
    stage_c_checkpoint = (
        ROOT / "new_checkpoints/new_true_ours/stage_c" / NAMESPACE
        / DATASET / f"seed{seed}/new_stage_c"
    )
    return {
        "lineage": "full_data_disjoint",
        "cache_mode": "memmap",
        "stage_b_cache": (
            ROOT / "new_cache/new_true_ours/full_data_stage_b/tiny"
            / f"new_full_stage_b_tiny_seed{seed}.npz"
        ),
        "stage_c_cache": (
            ROOT / "new_cache/new_true_ours/full_data_stage_c/tiny"
            / f"new_full_stage_c_tiny_seed{seed}.npz"
        ),
        "stage_b_output": stage_b_output,
        "stage_c_output": stage_c_output,
        "stage_b_checkpoint": stage_b_checkpoint,
        "stage_c_checkpoint": stage_c_checkpoint,
        "stage_b_report": (
            stage_b_output / f"new_secure_stage_b_report_tiny_balanced_seed{seed}.json"
        ),
        "initializer": (
            stage_b_output
            / f"new_secure_stage_b_candidate_nokd_tiny_balanced_seed{seed}.npz"
        ),
        "stage_c_report": (
            stage_c_output
            / f"new_secure_stage_c_report_tiny_adaptive_dp_seed{seed}.json"
        ),
        "stage_b_log": (
            ROOT / "new_log/new_true_ours/txt/formal_corrected_v2"
            / f"new_formal_stage_b_full_data_tiny_seed{seed}.log"
        ),
        "stage_c_log": (
            ROOT / "new_log/new_true_ours/txt/formal_corrected_v2"
            / f"new_formal_stage_c_full_data_tiny_seed{seed}.log"
        ),
        "timeline": (
            ROOT / "new_log/new_true_ours/formal_timelines/tiny"
            / NAMESPACE / f"new_true_ours_timeline_seed{seed}.json"
        ),
    }


def stage_b_command(paths: dict, frozen: dict, seed: int) -> list[str]:
    """构造冻结的 Tiny Full Data Stage-B TFE/ABY3 命令。"""
    config = frozen["stage_b"]
    command = [
        str(TFE_PYTHON), str(ROOT / "new_stage_b_secure_init.py"),
        "--feature_cache", str(paths["stage_b_cache"]),
        "--feature_cache_mode", paths["cache_mode"],
        "--dataset", DATASET,
        "--output_dir", str(paths["stage_b_output"]),
        "--checkpoint_prefix", str(paths["stage_b_checkpoint"]),
        "--seed", str(seed),
        "--num_clients", str(config["num_clients"]),
        "--split", config["split"],
        "--batch_size", str(config["batch_size"]),
        "--local_epochs", str(config["local_epochs"]),
        "--local_lr", str(config["local_lr"]),
        "--weight_decay", str(config["weight_decay"]),
        "--client_weighting", config["client_weighting"],
        "--fedavg_sync_interval", str(config["fedavg_sync_interval"]),
        "--distill_epochs", "0", "--lambda_kd", "0",
        "--reference_supervised_weight", "0",
        "--precision_fractional", str(config["precision_fractional"]),
        "--matmul_chunk_size", str(config["matmul_chunk_size"]),
        "--checkpoint_interval", str(config["checkpoint_interval_epochs"]),
        "--formal_candidate_release", "--secure_average_only",
        "--development_checkpoint_evaluation",
        "--input_lineage", frozen["initializer_lineage"],
    ]
    if config["shared_initialization"]:
        command.append("--shared_initialization")
    # 与corrected seed42一致：4小时超时、失败后冷却并最多重试一次。
    return ["timeout", "--foreground", "-k", "60", "14400"] + command


def stage_c_command(
    paths: dict, frozen: dict, seed: int, stage_c_script: Path
) -> list[str]:
    """构造冻结的 Tiny Full Data Adaptive Stage-C TFE/ABY3 命令。"""
    config = frozen["stage_c"]
    return [
        "env",
        f"NEW_MODERN_PYTHON={ROOT.parent / 'new_envs/distillation/bin/python'}",
        f"PYTHONPATH={ROOT}",
        str(TFE_PYTHON), str(stage_c_script),
        "--feature_cache", str(paths["stage_c_cache"]),
        "--feature_cache_mode", paths["cache_mode"],
        "--initializer", str(paths["initializer"]),
        "--dataset", DATASET, "--mode", "adaptive_dp",
        "--output_dir", str(paths["stage_c_output"]),
        "--checkpoint_prefix", str(paths["stage_c_checkpoint"]),
        "--seed", str(seed),
        "--batch_size", str(config["graph_batch_capacity"]),
        "--expected_batch_size", str(config["expected_batch_size"]),
        "--sampling_mode", "poisson",
        "--steps", str(config["steps"]),
        "--learning_rate", str(config["learning_rate"]),
        "--noise_multiplier", str(config["noise_multiplier"]),
        "--count_noise_multiplier", str(config["count_noise_multiplier"]),
        "--clip_c", str(config["clip_c"]),
        "--clipping_scope", config["clipping_scope"],
        "--gamma", str(config["gamma"]), "--eta_c", str(config["eta_c"]),
        "--c_min", str(config["c_min"]), "--c_max", str(config["c_max"]),
        "--warmup_steps", str(config["warmup_steps"]),
        "--update_interval", str(config["update_interval"]),
        "--delta", str(config["delta"]),
        "--precision_fractional", str(config["precision_fractional"]),
        "--matmul_chunk_size", str(config["matmul_chunk_size"]),
        "--checkpoint_interval", str(config["checkpoint_interval_steps"]),
        "--initializer_privacy_status", config["initializer_privacy_status"],
        "--input_lineage", frozen["initializer_lineage"],
        "--evaluation_split", "development_validation",
    ]


def preflight(config: dict, seeds: list[int]) -> dict:
    """检查冻结代码、split、缓存以及已有True Ours seed43/44。"""
    for relative, expected_sha in config["exact_code_sha256"].items():
        if relative == FROZEN_STAGE_C_RELATIVE:
            continue
        if file_sha256(ROOT / relative) != expected_sha:
            raise ValueError(f"冻结代码SHA不匹配: {relative}")
    frozen_stage_c_code(config, materialize=False)
    full_frozen = config["datasets"][DATASET]["methods"][METHOD]
    checks = []
    for seed in seeds:
        paths = paths_for(seed)
        for key in ("stage_b_cache", "stage_c_cache"):
            if not paths[key].is_file():
                raise FileNotFoundError(str(paths[key]))
        split = config["datasets"][DATASET]["splits"][str(seed)]
        if file_sha256(split["split_npz"]) != split["split_npz_sha256"]:
            raise ValueError(f"Tiny seed{seed} split SHA不匹配")

        # True Ours seed43/44已经完成：这里只验证，不重复运行。
        ours_paths = task_paths(DATASET, "true_ours", seed)
        ours_frozen = config["datasets"][DATASET]["methods"]["true_ours"]
        validate_stage_b(
            ours_paths["stage_b_report"], DATASET, "true_ours", seed, ours_frozen
        )
        validate_stage_c(
            ours_paths["stage_c_report"], DATASET, "true_ours", seed, ours_frozen
        )
        checks.append({
            "seed": seed,
            "true_ours_existing_valid": True,
            "full_data_stage_b_cache": str(paths["stage_b_cache"].relative_to(ROOT)),
            "full_data_stage_c_cache": str(paths["stage_c_cache"].relative_to(ROOT)),
        })
    return {"passed": True, "cells_to_run": len(seeds), "checks": checks,
            "frozen_full_data_config": full_frozen}


def main() -> int:
    parser = argparse.ArgumentParser(description="补齐Tiny End-to-End Full Data seeds")
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=ALLOWED_SEEDS,
        default=list(ALLOWED_SEEDS), help="允许43和/或44；默认两者顺序执行",
    )
    parser.add_argument("--preflight-only", action="store_true", help="只检查，不训练")
    args = parser.parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    checks = preflight(config, seeds)
    status = {
        "schema": "new_tiny_end2end_extra_seeds_status_v1",
        "status": "PREFLIGHT_PASS",
        "dataset": DATASET,
        "method_run": METHOD,
        "seeds": seeds,
        "namespace": NAMESPACE,
        "frozen_code_commit": FROZEN_CODE_COMMIT,
        "frozen_stage_c_sha256": config["exact_code_sha256"][
            FROZEN_STAGE_C_RELATIVE
        ],
        "frozen_accountant_sha256": FROZEN_ACCOUNTANT_SHA256,
        "preflight": checks,
        "rows": [],
        "official_test_read": False,
        "locked_holdout_read": False,
    }
    atomic_json(STATUS_PATH, status)
    print(f"PREFLIGHT_PASS Tiny Full Data seeds={seeds}", flush=True)
    if args.preflight_only:
        return 0

    frozen = config["datasets"][DATASET]["methods"][METHOD]
    stage_c_script = frozen_stage_c_code(config, materialize=True)
    assert stage_c_script is not None
    for seed in seeds:
        paths = paths_for(seed)
        for key in ("stage_b_output", "stage_c_output"):
            paths[key].mkdir(parents=True, exist_ok=True)
        for key in ("stage_b_checkpoint", "stage_c_checkpoint"):
            paths[key].parent.mkdir(parents=True, exist_ok=True)

        status["status"] = f"SEED{seed}_STAGE_B_RUNNING"
        atomic_json(STATUS_PATH, status)
        if not paths["stage_b_report"].is_file():
            run_logged(stage_b_command(paths, frozen, seed), paths["stage_b_log"],
                       max_attempts=2)
        stage_b, stage_b_checks = validate_stage_b(
            paths["stage_b_report"], DATASET, METHOD, seed, frozen
        )
        if not paths["initializer"].is_file():
            raise FileNotFoundError(str(paths["initializer"]))

        status["status"] = f"SEED{seed}_STAGE_C_RUNNING"
        atomic_json(STATUS_PATH, status)
        if not paths["stage_c_report"].is_file():
            run_logged(
                stage_c_command(paths, frozen, seed, stage_c_script),
                paths["stage_c_log"],
            )
        stage_c, stage_c_checks = validate_stage_c(
            paths["stage_c_report"], DATASET, METHOD, seed, frozen
        )
        timeline = write_timeline(paths, DATASET, METHOD, seed, stage_b, stage_c)
        row = {
            "seed": seed,
            "stage_b_report": str(paths["stage_b_report"].relative_to(ROOT)),
            "stage_b_report_sha256": file_sha256(paths["stage_b_report"]),
            "stage_c_report": str(paths["stage_c_report"].relative_to(ROOT)),
            "stage_c_report_sha256": file_sha256(paths["stage_c_report"]),
            "timeline": str(paths["timeline"].relative_to(ROOT)),
            "timeline_sha256": file_sha256(paths["timeline"]),
            "total_online_secure_seconds": timeline["total_online_secure_seconds"],
            "stage_b_checks": stage_b_checks,
            "stage_c_checks": stage_c_checks,
        }
        status["rows"] = [r for r in status["rows"] if r["seed"] != seed] + [row]
        status["status"] = f"SEED{seed}_COMPLETE"
        atomic_json(STATUS_PATH, status)
        print(f"COMPLETE Tiny Full Data seed{seed}", flush=True)

    status["status"] = "COMPLETE"
    atomic_json(STATUS_PATH, status)
    print(f"COMPLETE status={STATUS_PATH.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

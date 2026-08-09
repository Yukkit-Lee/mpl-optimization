#!/usr/bin/env python3
"""把用户已完成的CIFAR-10独立配置消融扩展到CIFAR-100和Tiny。

本脚本不会重跑或改写CIFAR-10，而是读取其已归档三seed JSON，并保持其中
Fixed/Adaptive参数完全不变，新增CIFAR-100与Tiny各2方法×3seed。每个任务
根据自己的数据规模、steps、batch和Adaptive count releases重新校准噪声，
最终epsilon仍为3。

在仓库根目录通过tmux运行：

    /root/autodl-tmp/new_envs/distillation/bin/python \
        new_scripts/new_paper_adaptive/new_run_user_configurable_multidataset_extension.py \
        --workers 12

输出不会覆盖CIFAR-10原结果：

    new_log/new_paper_adaptive/new_user_tuned_fixed_vs_adaptive_multidataset_formal.json
    new_expTab/new_paper_adaptive/new_user_tuned_fixed_vs_adaptive_multidataset_formal.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_scripts.new_paper_adaptive.new_screen_latest_initializer_fixed_vs_adaptive import (  # noqa: E402
    run_candidate,
)
from new_scripts.new_paper_adaptive.new_user_editable_figure6_config import (  # noqa: E402
    ADAPTIVE_CONFIG,
    FIXED_CONFIG,
    RNG_SEEDS,
)


MODERN_PYTHON = ROOT.parent / "new_envs/distillation/bin/python"
CIFAR10_SOURCE = (
    ROOT / "new_log/new_paper_adaptive/"
    "new_user_tuned_fixed_vs_adaptive_formal.json"
)
OUT_JSON = (
    ROOT / "new_log/new_paper_adaptive/"
    "new_user_tuned_fixed_vs_adaptive_multidataset_formal.json"
)
OUT_CSV = (
    ROOT / "new_expTab/new_paper_adaptive/"
    "new_user_tuned_fixed_vs_adaptive_multidataset_formal.csv"
)

# CIFAR-100存在与CIFAR-10相同类型的Anchor-post-aggregation C5模型。
# Tiny没有该C5正式资产，因此使用其Official-MTT-only True-Ours Stage-B模型；
# 该差异会写入结果provenance和论文图注，绝不伪装成同一个初始化器。
DATASET_SPECS = {
    "cifar100": {
        "initializer": (
            "new_output/new_stage_b_anchor_postagg/tfe_validation/cifar100/seed42/"
            "new_stage_b_anchor_postagg_tfe_model_cifar100_seed42.npz"
        ),
        "weight_key": "c5_weight",
        "bias_key": "c5_bias",
        "feature_cache": (
            "new_cache/new_true_ours/official_mtt/stage_c/cifar100/"
            "new_true_ours_stage_c_cifar100_seed42.npz"
        ),
        "initializer_variant": "anchor_postagg_c5",
    },
    "tiny": {
        "initializer": (
            "new_output/new_true_ours/stage_b/true_ours_formal/tiny/seed42/"
            "new_secure_stage_b_selected_tiny_balanced_seed42.npz"
        ),
        "weight_key": "weight",
        "bias_key": "bias",
        "feature_cache": (
            "new_cache/new_true_ours/official_mtt/stage_c/tiny/"
            "new_true_ours_stage_c_tiny_seed42.npz"
        ),
        "initializer_variant": "true_ours_stage_b_selected",
    },
}


def sha256(path: Path) -> str:
    """计算输入和结果文件SHA256，用于复现实验来源。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dependencies() -> None:
    """在创建worker前检查Opacus，避免再次误用base Python。"""
    if importlib.util.find_spec("opacus") is None:
        raise ModuleNotFoundError(
            f"当前解释器{sys.executable}缺少opacus；请使用{MODERN_PYTHON}"
        )


def load_dataset_inputs(dataset: str):
    """加载冻结初始化器与development缓存，并拒绝test或full-data依赖。"""
    spec = DATASET_SPECS[dataset]
    model_path = ROOT / spec["initializer"]
    cache_path = ROOT / spec["feature_cache"]
    with np.load(model_path, allow_pickle=False) as model:
        if str(model["initializer_source"].reshape(-1)[0]) != "official_mtt_only":
            raise ValueError(f"{dataset} initializer不是official_mtt_only")
        if int(model["full_data_initializer_dependency"].reshape(-1)[0]) != 0:
            raise ValueError(f"{dataset} initializer含full-data依赖")
        weight = np.asarray(model[spec["weight_key"]], dtype=np.float32)
        bias = np.asarray(model[spec["bias_key"]], dtype=np.float32)
    with np.load(cache_path, allow_pickle=False) as cache:
        if int(cache["test_loaded"].reshape(-1)[0]) != 0:
            raise ValueError(f"{dataset} cache读取了official test")
        data = {
            "x_train": np.asarray(cache["x_encoder_train"], dtype=np.float32),
            "y_train": np.asarray(cache["y_encoder_train"], dtype=np.int64),
            "x_dev": np.asarray(cache["x_reference"], dtype=np.float32),
            "y_dev": np.asarray(cache["y_reference"], dtype=np.int64),
        }
    if weight.shape != (data["x_train"].shape[1], bias.shape[0]):
        raise ValueError(f"{dataset} initializer/cache维度不匹配")
    return model_path, cache_path, weight, bias, data


def make_row(dataset: str, config: dict, result: dict) -> dict:
    """把单任务结果转换为CSV和跨seed汇总使用的标量行。"""
    return {
        "dataset": dataset,
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


def summarize_dataset(rows: list[dict]) -> dict:
    """按数据集计算逐seed差值、均值和样本标准差。"""
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
        }
    gains = [item["adaptive_minus_fixed_pp"] for item in by_seed.values()]
    return {
        "by_seed": by_seed,
        "mean_adaptive_minus_fixed_pp": float(np.mean(gains)),
        "sample_std_adaptive_minus_fixed_pp": float(np.std(gains, ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="扩展用户配置Figure 6到C100/Tiny")
    parser.add_argument("--workers", type=int, default=12, help="12个新增任务的并行数")
    parser.add_argument(
        "--check-only", action="store_true",
        help="只检查依赖、参数和输入资产，不启动训练",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers必须大于0")
    ensure_dependencies()
    if RNG_SEEDS != [42, 43, 44]:
        raise ValueError("正式扩展要求RNG_SEEDS=[42,43,44]")
    if FIXED_CONFIG["target_epsilon"] != ADAPTIVE_CONFIG["target_epsilon"]:
        raise ValueError("Fixed/Adaptive epsilon不一致")

    cifar10 = json.loads(CIFAR10_SOURCE.read_text(encoding="utf-8"))
    if cifar10["status"] != "COMPLETE" or cifar10["rng_seeds"] != RNG_SEEDS:
        raise ValueError("CIFAR-10来源不完整")
    if cifar10["fixed_config"] != FIXED_CONFIG or cifar10["adaptive_config"] != ADAPTIVE_CONFIG:
        raise ValueError("当前参数与CIFAR-10归档参数不一致")
    if cifar10["official_test_read"] or cifar10["locked_holdout_read"]:
        raise ValueError("CIFAR-10来源越过评价边界")

    loaded = {dataset: load_dataset_inputs(dataset) for dataset in DATASET_SPECS}
    if args.check_only:
        print(json.dumps({
            "status": "PREFLIGHT_PASS",
            "training_started": False,
            "cifar10_source_sha256": sha256(CIFAR10_SOURCE),
            "fixed_config": FIXED_CONFIG,
            "adaptive_config": ADAPTIVE_CONFIG,
            "new_task_count": 12,
            "inputs": {
                dataset: {
                    "initializer": str(item[0].relative_to(ROOT)),
                    "initializer_sha256": sha256(item[0]),
                    "feature_cache": str(item[1].relative_to(ROOT)),
                    "feature_cache_sha256": sha256(item[1]),
                    "initializer_variant": DATASET_SPECS[dataset]["initializer_variant"],
                }
                for dataset, item in loaded.items()
            },
        }, ensure_ascii=False, indent=2))
        return

    tasks = []
    for dataset, (_, _, weight, bias, data) in loaded.items():
        for seed in RNG_SEEDS:
            for base in (FIXED_CONFIG, ADAPTIVE_CONFIG):
                config = dict(base, rng_seed=int(seed))
                tasks.append((dataset, config, weight, bias, data))
    payloads = [(config, weight, bias, data) for _, config, weight, bias, data in tasks]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        results = list(executor.map(run_candidate, payloads))

    detailed = []
    rows = []
    # 先复制CIFAR-10既有正式证据，不重新运行。
    for item in cifar10["detailed"]:
        copied = deepcopy(item)
        copied["row"]["dataset"] = "cifar10"
        copied["config"]["dataset"] = "cifar10"
        detailed.append(copied)
        rows.append({"dataset": "cifar10", **copied["row"]})

    for (dataset, config, _, _, _), result in zip(tasks, results):
        row = make_row(dataset, config, result)
        if abs(row["accounted_epsilon"] - float(config["target_epsilon"])) > 1e-8:
            raise ValueError(f"{dataset}/{config['mode']}/seed{config['rng_seed']} epsilon失败")
        rows.append(row)
        detailed.append({
            "row": row,
            "config": {"dataset": dataset, **config},
            "history": result["history"],
        })

    output = {
        "schema": "new_user_configurable_fixed_vs_adaptive_multidataset_v1",
        "status": "COMPLETE",
        "comparison_scope": "same epsilon and independently configured methods; matched initializer/data within each dataset",
        "fixed_config": FIXED_CONFIG,
        "adaptive_config": ADAPTIVE_CONFIG,
        "rng_seeds": RNG_SEEDS,
        "cifar10_reused_not_rerun": True,
        "cifar10_source": str(CIFAR10_SOURCE.relative_to(ROOT)),
        "cifar10_source_sha256": sha256(CIFAR10_SOURCE),
        "dataset_inputs": {
            dataset: {
                "initializer": str(item[0].relative_to(ROOT)),
                "initializer_sha256": sha256(item[0]),
                "feature_cache": str(item[1].relative_to(ROOT)),
                "feature_cache_sha256": sha256(item[1]),
                "initializer_variant": DATASET_SPECS[dataset]["initializer_variant"],
            }
            for dataset, item in loaded.items()
        },
        "by_dataset": {
            dataset: summarize_dataset([row for row in rows if row["dataset"] == dataset])
            for dataset in ("cifar10", "cifar100", "tiny")
        },
        "detailed": detailed,
        "official_test_read": False,
        "locked_holdout_read": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "status": output["status"],
        "cifar10_reused_not_rerun": True,
        "new_task_count": len(tasks),
        "by_dataset": output["by_dataset"],
        "output_json": str(OUT_JSON.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

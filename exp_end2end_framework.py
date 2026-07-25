#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：直接图像、完整 ConvNet 的 A+B+C 端到端主流程。

Stage A 提供 official MTT distilled images；Stage B 在完整 ConvNet 上执行
本地训练、KD 和候选选择；Stage C 从磁盘重新加载 chosen full-ConvNet
checkpoint，并继续训练同一个模型。test set 仅用于固定间隔只读报告。
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

from project_paths import PATHS
import official_distilled_convnet_pipeline as pipe
import exp_module_b_kd_noniid_official as stage_b_entry


CFG = {
    "cpu_threads": 1,
    "stage_c_steps": 60,
    "stage_c_batch_size": 4,
    "stage_c_lr": 0.005,
    "fixed_clip_c": 1.0,
    "noise_multiplier": 1.2,
    "gamma": 0.5,
    "eta_c": 0.15,
    "eval_interval": 10,
}

JSON_DIR = PATHS.output_root / "log" / "json"
CSV_DIR = PATHS.output_root / "expTab"


def run_end2end(dataset: str, seed: int, quick: bool = False):
    """运行直接图像的完整 A+B+C，并验证跨阶段参数没有被重置。"""
    pipe.set_safe_runtime(seed, int(CFG["cpu_threads"]))
    device = pipe.select_device("auto")

    stage_b_out = stage_b_entry.run_stage_b_experiment(dataset, seed, quick)
    stage_b = stage_b_out["stage_b"]
    checkpoint = stage_b_out["checkpoint"]
    bundle = stage_b_out["bundle"]

    chosen_model, checkpoint_meta = pipe.load_stage_b_checkpoint(checkpoint, device)
    before_signature = stage_b["chosen_signature"]
    after_signature = pipe.state_signature(chosen_model)
    if before_signature != after_signature:
        raise RuntimeError("Stage B -> Stage C checkpoint state signature mismatch")
    if pipe.state_schema(chosen_model) != stage_b["model_schema"]:
        raise RuntimeError("Stage B -> Stage C full state-dict schema mismatch")

    _, initial_ref_accuracy = pipe.evaluate_model(
        chosen_model, bundle["x_ref"], bundle["y_ref"], device
    )
    # 收敛目标只由 heldout reference set 构造，禁止使用 test accuracy。
    target_ref_accuracy = min(0.999, initial_ref_accuracy + 0.02)
    steps = 2 if quick else int(CFG["stage_c_steps"])
    batch_size = 2 if quick else int(CFG["stage_c_batch_size"])
    eval_interval = 1 if quick else int(CFG["eval_interval"])

    results = {}
    for mode in ("plain", "fixed_dp", "adaptive_dp"):
        result = pipe.train_stage_c_convnet(
            model=chosen_model,
            bundle=bundle,
            mode=mode,
            steps=steps,
            batch_size=batch_size,
            lr=float(CFG["stage_c_lr"]),
            clip_c=float(CFG["fixed_clip_c"]),
            noise_multiplier=0.0 if mode == "plain" else float(CFG["noise_multiplier"]),
            gamma=float(CFG["gamma"]),
            eta_c=float(CFG["eta_c"]),
            seed=seed + 3000,
            device=device,
            eval_interval=eval_interval,
            target_reference_accuracy=target_ref_accuracy,
        )
        if result["initialization_signature"] != after_signature:
            raise RuntimeError("Stage C did not start from the loaded Stage B checkpoint")
        results[mode] = result

    return {
        "dataset": dataset,
        "seed": seed,
        "pipeline": "algorithmic_pytorch_convnet",
        "backend": "local_plaintext_pytorch",
        "stage_a": {
            "input": "official_mtt_distilled_nchw_images",
            "dp_protected": False,
            "feature_extraction_used": False,
        },
        "stage_b": pipe.json_safe_stage_b(stage_b),
        "stage_b_checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_meta["checkpoint_sha256"],
        "checkpoint_state_signature_before": before_signature,
        "checkpoint_state_signature_after": after_signature,
        "checkpoint_signature_verified": True,
        "target_source": "heldout_reference_set",
        "target_reference_accuracy": target_ref_accuracy,
        "test_usage": "fixed_interval_readonly_reporting",
        "privacy_scope": "stage_c_only",
        "stage_c": {
            mode: {k: v for k, v in result.items() if k != "model"}
            for mode, result in results.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct-image full-ConvNet A+B+C pipeline")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tiny"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    started = time.time()
    payload = run_end2end(args.dataset, args.seed, args.quick)
    payload["run_id"] = time.strftime("%Y%m%d_%H%M%S")
    payload["total_time_s"] = time.time() - started

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    json_path = JSON_DIR / "crossStageConvNet_end2end_{}_{}.json".format(
        args.dataset, payload["run_id"]
    )
    csv_path = CSV_DIR / "crossStageConvNet_end2end_{}_{}.csv".format(
        args.dataset, payload["run_id"]
    )
    pipe.write_json(json_path, payload)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Dataset", "Mode", "FinalTestAcc", "FinalRefAcc", "StepAtRefTarget"])
        for mode, result in payload["stage_c"].items():
            writer.writerow([
                args.dataset,
                mode,
                result["final_test_accuracy"],
                result["final_reference_accuracy"],
                result["step_at_target_reference_accuracy"],
            ])
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("json:", json_path)
    print("csv :", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

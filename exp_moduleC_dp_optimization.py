#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：完整 ConvNet 的 Stage C 算法主实验。

输入为 Stage B chosen full-ConvNet checkpoint，Plain、Fixed-DP 与 Adaptive-DP
均继续训练同一个 ConvNet。当前 backend 是 local_plaintext_pytorch，不是 MPC。
"""

import argparse
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

JSON_DIR = PATHS.output_root / "log" / "json"

CFG = {
    "cpu_threads": 1,
    "steps": 60,
    # 逐样本完整 ConvNet 梯度成本较高，默认使用安全的小 batch。
    "batch_size": 4,
    "lr": 0.005,
    "fixed_clip_c": 1.0,
    "noise_multiplier": 1.2,
    "gamma": 0.5,
    "eta_c": 0.15,
    "eval_interval": 10,
}


def run_stage_c_experiment(dataset: str, seed: int, quick: bool = False):
    pipe.set_safe_runtime(seed, int(CFG["cpu_threads"]))
    device = pipe.select_device("auto")
    stage_b_out = stage_b_entry.run_stage_b_experiment(dataset, seed, quick)
    bundle = stage_b_out["bundle"]
    loaded_model, metadata = pipe.load_stage_b_checkpoint(stage_b_out["checkpoint"], device)
    assert pipe.state_signature(loaded_model) == stage_b_out["stage_b"]["chosen_signature"]
    _, initial_ref_acc = pipe.evaluate_model(loaded_model, bundle["x_ref"], bundle["y_ref"], device)
    # 目标只依赖 heldout reference set，不读取 test accuracy。
    target_reference_accuracy = min(0.999, initial_ref_acc + 0.02)
    steps = 2 if quick else int(CFG["steps"])
    batch_size = 2 if quick else int(CFG["batch_size"])
    eval_interval = 1 if quick else int(CFG["eval_interval"])
    results = {}
    for mode in ("plain", "fixed_dp", "adaptive_dp"):
        results[mode] = pipe.train_stage_c_convnet(
            model=loaded_model,
            bundle=bundle,
            mode=mode,
            steps=steps,
            batch_size=batch_size,
            lr=float(CFG["lr"]),
            clip_c=float(CFG["fixed_clip_c"]),
            noise_multiplier=0.0 if mode == "plain" else float(CFG["noise_multiplier"]),
            gamma=float(CFG["gamma"]),
            eta_c=float(CFG["eta_c"]),
            seed=seed + 2000,
            device=device,
            eval_interval=eval_interval,
            target_reference_accuracy=target_reference_accuracy,
        )
        if results[mode]["initialization_signature"] != metadata["state_signature"]:
            raise RuntimeError("Stage B checkpoint was not used as the real Stage C initialization")
    return stage_b_out, results, target_reference_accuracy


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage C full-ConvNet adaptive DP optimization")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tiny"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    started = time.time()
    stage_b_out, results, target_ref = run_stage_c_experiment(args.dataset, args.seed, args.quick)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "run_id": run_id,
        "dataset": args.dataset,
        "pipeline": "algorithmic_pytorch_convnet",
        "backend": "local_plaintext_pytorch",
        "input_type": "nchw_images",
        "feature_extraction_used": False,
        "model_class": stage_b_out["stage_b"]["model_class"],
        "initialization_source": str(stage_b_out["checkpoint"]),
        "initialization_signature_verified": True,
        "heldout_reference_target": target_ref,
        "test_usage": "fixed_interval_readonly_reporting",
        "privacy_scope": "stage_c_only",
        "stage_a_dp_protected": False,
        "stage_b_dp_protected": False,
        "stage_c_epsilon_estimate": "PENDING_PHASE_4_ACCOUNTANT",
        "results": {
            mode: {key: value for key, value in result.items() if key != "model"}
            for mode, result in results.items()
        },
        "time_s": time.time() - started,
    }
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    path = JSON_DIR / "crossStageConvNet_stageC_{}_{}.json".format(args.dataset, run_id)
    pipe.write_json(path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("output:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

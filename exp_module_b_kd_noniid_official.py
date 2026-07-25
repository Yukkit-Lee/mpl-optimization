#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：Stage B 完整 ConvNet 的 Non-IID KD 实验。

主流程直接接收蒸馏图像，本地训练、KD、Avg、ReferenceWeighted 与 Best
都作用于完整 ConvNet state_dict；不提取 2048 维特征。
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

import torch

from project_paths import PATHS
import official_distilled_convnet_pipeline as pipe

CFG = {
    "cpu_threads": 1,
    "num_clients": 5,
    "classes_per_client": None,
    "reference_per_class": 5,
    "max_test_samples": 3000,
    "local_steps": 45,
    "kd_steps": 20,
    "batch_size": 32,
    "local_lr": 0.01,
    "kd_lr": 0.005,
    "tau": 2.0,
    "lambda_kd": 3.0,
    "ref_weight_temperature": 0.08,
}

LOG_DIR = PATHS.output_root / "log" / "json"


def prepare_image_bundle(dataset: str, seed: int):
    bundle = pipe.load_image_bundle(
        dataset,
        reference_per_class=int(CFG["reference_per_class"]),
        max_test_samples=int(CFG["max_test_samples"]),
        seed=seed,
    )
    pipe.assert_split_boundaries(bundle)
    return bundle


def run_stage_b_experiment(dataset: str, seed: int, quick: bool = False):
    pipe.set_safe_runtime(seed, int(CFG["cpu_threads"]))
    device = pipe.select_device("auto")
    bundle = prepare_image_bundle(dataset, seed)
    local_steps = 2 if quick else int(CFG["local_steps"])
    kd_steps = 2 if quick else int(CFG["kd_steps"])
    clients = 2 if quick else int(CFG["num_clients"])
    stage_b = pipe.run_stage_b(
        dataset=dataset,
        bundle=bundle,
        seed=seed,
        num_clients=clients,
        classes_per_client=CFG["classes_per_client"],
        local_steps=local_steps,
        kd_steps=kd_steps,
        batch_size=int(CFG["batch_size"]),
        local_lr=float(CFG["local_lr"]),
        kd_lr=float(CFG["kd_lr"]),
        tau=float(CFG["tau"]),
        lambda_kd=float(CFG["lambda_kd"]),
        weight_temperature=float(CFG["ref_weight_temperature"]),
        device=device,
    )
    if not all(stage_b["kd_parameter_changed"]):
        raise RuntimeError("KD did not update every local full ConvNet")
    expected_schema = pipe.state_schema(stage_b["chosen_model"])
    for name, candidate in stage_b["candidates"].items():
        if pipe.state_schema(candidate) != expected_schema:
            raise RuntimeError("{} is not a full-state ConvNet candidate".format(name))
    checkpoint = pipe.save_stage_b_checkpoint(dataset, stage_b, seed)
    loaded, metadata = pipe.load_stage_b_checkpoint(checkpoint, device)
    assert pipe.state_signature(loaded) == stage_b["chosen_signature"]
    return {
        "bundle": bundle,
        "stage_b": stage_b,
        "checkpoint": checkpoint,
        "checkpoint_metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage B full-ConvNet KD initialization")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tiny"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    started = time.time()
    out = run_stage_b_experiment(args.dataset, args.seed, args.quick)
    stage_b = out["stage_b"]
    payload = {
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "dataset": args.dataset,
        "seed": args.seed,
        "pipeline": "algorithmic_pytorch_convnet",
        "backend": "local_plaintext_pytorch",
        "input_type": "nchw_images",
        "feature_extraction_used": False,
        "reference_set": "heldout_reference_set",
        "reference_private_dependency": True,
        "stage_b": pipe.json_safe_stage_b(stage_b),
        "checkpoint": str(out["checkpoint"]),
        "checkpoint_signature_verified": True,
        "time_s": time.time() - started,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "crossStageConvNet_stageB_{}_{}.json".format(args.dataset, payload["run_id"])
    pipe.write_json(path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("output:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

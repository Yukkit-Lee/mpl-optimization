#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：仅使用已发布 DP expert trajectories 执行 MTT 参数轨迹匹配。

本文件不导入 torchvision.datasets、不接受 raw-data 路径，也不加载原始训练集。
synthetic images 从随机噪声初始化；因此其输出是 DP trajectories 的后处理，
Stage-A epsilon 与输入 trajectory manifest 完全一致，不额外消耗记录级预算。
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from new_secure_feature_models import (
    DATASET_SPECS,
    atomic_torch_save,
    build_mtt_convnet,
    capture_rng_state,
    configure_single_thread_runtime,
    restore_rng_state,
    sha256_file,
)


PROJECT_ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
MTT_ROOT = Path(os.environ.get("NEW_MTT_ROOT", PROJECT_ROOT.parent / "mtt-distillation"))
OUTPUT_ROOT = Path(os.environ.get("NEW_OUTPUT_ROOT", str(PROJECT_ROOT)))
DEFAULT_TRAJECTORY_ROOT = OUTPUT_ROOT / "new_cache" / "new_secure_feature" / "new_stage_a_dp_trajectories"
DEFAULT_OUTPUT = OUTPUT_ROOT / "new_cache" / "new_secure_feature" / "new_stage_a_dp_distilled"


def assert_no_raw_data_contract(args) -> None:
    """运行时再次断言蒸馏 CLI 没有 raw data 参数。"""

    forbidden = ("data_path", "raw_data", "train_dataset", "pix_init_real")
    for name in forbidden:
        if hasattr(args, name):
            raise RuntimeError("DP-MTT 蒸馏阶段禁止参数: {}".format(name))


def load_dp_trajectories(manifest_path: Path) -> tuple:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("original_record_privacy_scope") != "stage_a_dp_trajectory":
        raise RuntimeError("trajectory manifest 缺少 Stage-A DP 隐私范围")
    trajectories: List[List[List[torch.Tensor]]] = []
    for row in manifest["trajectories"]:
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError("trajectory SHA256 不匹配: {}".format(path))
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("dp_trajectory") is not True:
            raise RuntimeError("拒绝 non-DP trajectory: {}".format(path))
        trajectories.append(payload["trajectory"])
    if not trajectories:
        raise RuntimeError("manifest 中没有 DP trajectories")
    return manifest, trajectories


def import_reparam_module():
    """只导入纯参数化模型工具；不得导入 MTT 的数据工具模块。"""

    if str(MTT_ROOT) not in sys.path:
        sys.path.insert(0, str(MTT_ROOT))
    from reparam_module import ReparamModule

    return ReparamModule


def build_log_steps(steps: int, max_logs: int = 10) -> set:
    points = np.linspace(1, steps, min(max_logs, steps), dtype=int)
    return set(int(value) for value in points)


def main() -> int:
    parser = argparse.ArgumentParser(description="仅由 DP trajectories 生成 DP distilled images")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="cifar10")
    parser.add_argument("--trajectory_root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--syn_steps", type=int, default=20)
    parser.add_argument("--expert_epochs", type=int, default=3)
    parser.add_argument("--max_start_epoch", type=int, default=20)
    parser.add_argument("--lr_img", type=float, default=1000.0)
    parser.add_argument("--lr_lr", type=float, default=1e-5)
    parser.add_argument("--lr_teacher", type=float, default=0.01)
    parser.add_argument("--batch_syn", type=int, default=0)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    assert_no_raw_data_contract(args)
    configure_single_thread_runtime(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境未检测到 GPU")

    manifest_path = (
        args.trajectory_root
        / args.dataset
        / "new_stage_a_dp_trajectory_manifest_{}.json".format(args.dataset)
    )
    manifest, trajectories = load_dp_trajectories(manifest_path)
    spec = DATASET_SPECS[args.dataset]
    labels = torch.arange(spec["num_classes"], dtype=torch.long).repeat_interleave(args.ipc).to(device)
    images = torch.randn(
        len(labels),
        spec["channels"],
        spec["image_size"][0],
        spec["image_size"][1],
        device=device,
        requires_grad=True,
    )
    synthetic_lr = torch.tensor(args.lr_teacher, device=device, requires_grad=True)
    optimizer_img = torch.optim.SGD([images], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([synthetic_lr], lr=args.lr_lr, momentum=0.5)
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "new_dp_mtt_checkpoint_{}_seed{}.pt".format(args.dataset, args.seed)
    start_iteration = 0
    history: List[Dict] = []
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        images.data.copy_(checkpoint["images"].to(device))
        synthetic_lr.data.copy_(checkpoint["synthetic_lr"].to(device))
        optimizer_img.load_state_dict(checkpoint["optimizer_img"])
        optimizer_lr.load_state_dict(checkpoint["optimizer_lr"])
        start_iteration = int(checkpoint["next_iteration"])
        history = checkpoint["history"]
        restore_rng_state(checkpoint["rng_state"])
        print("从 iteration {} 恢复 DP-MTT".format(start_iteration), flush=True)

    ReparamModule = import_reparam_module()
    criterion = nn.CrossEntropyLoss()
    log_steps = build_log_steps(args.iterations)
    started = time.time()
    print("=" * 88, flush=True)
    print("Stage A Track B：DP trajectories -> MTT post-processing", flush=True)
    print("dataset={} trajectories={} ipc={} iterations={} syn_steps={}".format(
        args.dataset, len(trajectories), args.ipc, args.iterations, args.syn_steps
    ), flush=True)
    print("raw_data_access=False pix_init=noise cpu_threads=1 num_workers=0", flush=True)
    print("stage_a_epsilon={} delta={}".format(
        manifest["stage_a_epsilon"], manifest["stage_a_delta"]
    ), flush=True)
    print("output: {}".format(output_dir), flush=True)
    print("=" * 88, flush=True)

    for iteration in range(start_iteration, args.iterations):
        trajectory = trajectories[np.random.randint(0, len(trajectories))]
        max_start = min(args.max_start_epoch, len(trajectory) - args.expert_epochs - 1)
        if max_start < 0:
            raise RuntimeError("expert trajectory 长度不足")
        start_epoch = np.random.randint(0, max_start + 1)
        start_tensors = trajectory[start_epoch]
        target_tensors = trajectory[start_epoch + args.expert_epochs]
        base_model = build_mtt_convnet(args.dataset, args.seed + iteration, device)
        student = ReparamModule(base_model).to(device)
        start_flat = torch.cat([tensor.to(device).reshape(-1) for tensor in start_tensors])
        target_flat = torch.cat([tensor.to(device).reshape(-1) for tensor in target_tensors])
        student_params = [start_flat.detach().clone().requires_grad_(True)]
        indices = torch.randperm(len(images), device=device)
        if args.batch_syn > 0:
            indices = indices[: min(args.batch_syn, len(indices))]
        batch_images = images[indices]
        batch_labels = labels[indices]
        for _ in range(args.syn_steps):
            logits = student(batch_images, flat_param=student_params[-1])
            loss = criterion(logits, batch_labels)
            gradient = torch.autograd.grad(loss, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - synthetic_lr * gradient)
        numerator = nn.functional.mse_loss(student_params[-1], target_flat, reduction="sum")
        denominator = nn.functional.mse_loss(start_flat, target_flat, reduction="sum").clamp_min(1e-12)
        trajectory_loss = numerator / denominator
        optimizer_img.zero_grad(set_to_none=True)
        optimizer_lr.zero_grad(set_to_none=True)
        trajectory_loss.backward()
        optimizer_img.step()
        optimizer_lr.step()
        record = {
            "iteration": iteration + 1,
            "trajectory_loss": float(trajectory_loss.detach()),
            "synthetic_lr": float(synthetic_lr.detach()),
            "start_epoch": int(start_epoch),
        }
        history.append(record)
        if (iteration + 1) % args.checkpoint_interval == 0 or iteration + 1 == args.iterations:
            atomic_torch_save(
                {
                    "schema": "new_stage_a_dp_mtt_checkpoint_v1",
                    "dataset": args.dataset,
                    "next_iteration": iteration + 1,
                    "images": images.detach().cpu(),
                    "labels": labels.detach().cpu(),
                    "synthetic_lr": synthetic_lr.detach().cpu(),
                    "optimizer_img": optimizer_img.state_dict(),
                    "optimizer_lr": optimizer_lr.state_dict(),
                    "history": history,
                    "rng_state": capture_rng_state(),
                    "raw_data_access": False,
                },
                checkpoint_path,
            )
        if iteration + 1 in log_steps:
            print(
                "  iteration={:5d}/{:5d} trajectory_loss={:.6f} synthetic_lr={:.6f}".format(
                    iteration + 1,
                    args.iterations,
                    record["trajectory_loss"],
                    record["synthetic_lr"],
                ),
                flush=True,
            )

    result_path = output_dir / "new_dp_mtt_images_{}_ipc{}_seed{}.pt".format(
        args.dataset, args.ipc, args.seed
    )
    atomic_torch_save(
        {
            "schema": "new_stage_a_dp_mtt_distilled_v1",
            "dataset": args.dataset,
            "images": images.detach().cpu(),
            "labels": labels.detach().cpu(),
            "ipc": args.ipc,
            "stage_a_track": "dp_mtt",
            "stage_a_epsilon": manifest["stage_a_epsilon"],
            "stage_a_delta": manifest["stage_a_delta"],
            "accountant_type": manifest["accountant_type"],
            "num_expert_trajectories": manifest["num_expert_trajectories"],
            "original_record_privacy_scope": "stage_a_dp_trajectory",
            "raw_data_access_during_distillation": False,
            "image_initialization": "random_noise",
            "post_processing_only": True,
            "trajectory_manifest": str(manifest_path),
            "trajectory_manifest_sha256": sha256_file(manifest_path),
            "iterations": args.iterations,
            "runtime_seconds": time.time() - started,
            "history": history,
        },
        result_path,
    )
    metadata_path = output_dir / "new_dp_mtt_metadata_{}_ipc{}_seed{}.json".format(
        args.dataset, args.ipc, args.seed
    )
    metadata_path.write_text(
        json.dumps(
            {
                "result_path": str(result_path),
                "result_sha256": sha256_file(result_path),
                "stage_a_track": "dp_mtt",
                "stage_a_epsilon": manifest["stage_a_epsilon"],
                "stage_a_delta": manifest["stage_a_delta"],
                "accountant_type": manifest["accountant_type"],
                "num_expert_trajectories": manifest["num_expert_trajectories"],
                "raw_data_access_during_distillation": False,
                "status": "COMPLETE",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("result: {}".format(result_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

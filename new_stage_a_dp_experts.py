#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：使用 Opacus DP-SGD 生成可正式会计的 Stage-A expert trajectories。

只有本文件允许读取原始私有训练集。数据先按类别分层、互斥划分给多个机构；
同一机构的多个 expert 使用 RDP 串行组合，不同机构因记录互斥使用 parallel
composition，最终 Stage-A epsilon 取各机构 composed epsilon 的最大值。
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from new_secure_feature_models import (
    DATASET_SPECS,
    atomic_torch_save,
    build_mtt_convnet,
    capture_rng_state,
    configure_single_thread_runtime,
    model_schema_hash,
    parameter_snapshot,
    restore_rng_state,
    sha256_file,
    unwrap_private_model,
)


PROJECT_ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = Path(os.environ.get("NEW_DATA_ROOT", "/home/lixujie/tmpWork"))
OUTPUT_ROOT = Path(os.environ.get("NEW_OUTPUT_ROOT", str(PROJECT_ROOT)))
DEFAULT_OUTPUT = OUTPUT_ROOT / "new_cache" / "new_secure_feature" / "new_stage_a_dp_trajectories"

RAW_DATA_PATHS = {
    "cifar10": DATA_ROOT / "mtt_cifar10_work" / "data",
    "cifar100": DATA_ROOT / "mtt_cifar100_work" / "data",
    "tiny": DATA_ROOT / "mtt_tiny_work" / "data" / "tiny-imagenet-200",
}

NORMALIZATION = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    "cifar100": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    "tiny": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def load_private_training_dataset(dataset: str):
    """本函数是整个 DP-MTT 路线唯一允许访问 raw training data 的边界。"""

    mean, std = NORMALIZATION[dataset]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    root = RAW_DATA_PATHS[dataset]
    if dataset == "cifar10":
        return datasets.CIFAR10(str(root), train=True, download=False, transform=transform)
    if dataset == "cifar100":
        return datasets.CIFAR100(str(root), train=True, download=False, transform=transform)
    train_root = root / "train"
    if not train_root.exists():
        raise FileNotFoundError("Tiny ImageNet train 目录不存在: {}".format(train_root))
    return datasets.ImageFolder(str(train_root), transform=transform)


def dataset_labels(dataset_obj) -> np.ndarray:
    if hasattr(dataset_obj, "targets"):
        return np.asarray(dataset_obj.targets, dtype=np.int64)
    return np.asarray([label for _, label in dataset_obj.samples], dtype=np.int64)


def make_disjoint_party_splits(labels: np.ndarray, num_parties: int, seed: int) -> List[np.ndarray]:
    """按类分层后均匀切分，保证机构之间样本索引严格互斥。"""

    rng = np.random.RandomState(seed)
    parties: List[List[int]] = [[] for _ in range(num_parties)]
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        for party_id, chunk in enumerate(np.array_split(indices, num_parties)):
            parties[party_id].extend(chunk.tolist())
    arrays = []
    seen = set()
    for party in parties:
        rng.shuffle(party)
        array = np.asarray(party, dtype=np.int64)
        if seen.intersection(array.tolist()):
            raise AssertionError("机构数据划分出现重叠")
        seen.update(array.tolist())
        arrays.append(array)
    if len(seen) != len(labels):
        raise AssertionError("机构数据划分未覆盖全部训练样本")
    return arrays


def compose_rdp_histories(histories: Sequence[Sequence], delta: float) -> float:
    accountant = RDPAccountant()
    accountant.history = []
    for history in histories:
        for noise_multiplier, sample_rate, steps in history:
            accountant.history.append((float(noise_multiplier), float(sample_rate), int(steps)))
    return float(accountant.get_epsilon(delta=delta))


def build_log_epochs(epochs: int, max_logs: int = 10) -> set:
    points = np.linspace(1, epochs, min(max_logs, epochs), dtype=int)
    return set(int(value) for value in points)


def train_one_expert(
    dataset: str,
    party_id: int,
    expert_id: int,
    party_indices: np.ndarray,
    dataset_obj,
    output_dir: Path,
    args,
    device: torch.device,
) -> Path:
    task_seed = args.seed + party_id * 1000 + expert_id
    configure_single_thread_runtime(task_seed)
    trajectory_path = output_dir / (
        "new_dp_trajectory_{}_party{:02d}_expert{:02d}.pt".format(dataset, party_id, expert_id)
    )
    checkpoint_path = output_dir / (
        "new_dp_expert_checkpoint_{}_party{:02d}_expert{:02d}.pt".format(dataset, party_id, expert_id)
    )
    if trajectory_path.exists() and not args.force:
        print("  跳过已完成轨迹: {}".format(trajectory_path.name), flush=True)
        return trajectory_path

    subset = Subset(dataset_obj, party_indices.tolist())
    loader_generator = torch.Generator()
    loader_generator.manual_seed(task_seed)
    loader = DataLoader(
        subset,
        batch_size=min(args.batch_size, len(subset)),
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        generator=loader_generator,
    )
    model = build_mtt_convnet(dataset, task_seed, device)
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        raise RuntimeError("模型不满足 Opacus per-example gradient 要求: {}".format(errors))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    try:
        privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=args.secure_rng)
    except Exception as exc:
        if args.secure_rng:
            raise RuntimeError(
                "正式 DP expert 要求 secure RNG，但 torchcsprng 与当前 PyTorch/CUDA 不兼容；"
                "禁止静默回退。原始错误: {}".format(exc)
            ) from exc
        raise
    model, optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        poisson_sampling=True,
        grad_sample_mode="hooks",
    )
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    trajectory = [parameter_snapshot(model)]
    optimizer_steps = 0
    if checkpoint_path.exists() and not args.force:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        unwrap_private_model(model).load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        privacy_engine.accountant.load_state_dict(checkpoint["accountant_state"])
        trajectory = checkpoint["trajectory"]
        start_epoch = int(checkpoint["next_epoch"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        restore_rng_state(checkpoint["rng_state"])
        print("  从 epoch {} 恢复 expert".format(start_epoch), flush=True)

    log_epochs = build_log_epochs(args.epochs)
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        for images, labels in private_loader:
            images = images.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.detach()) * labels.numel()
            correct += int((logits.argmax(1) == labels).sum().item())
            count += int(labels.numel())
        trajectory.append(parameter_snapshot(model))
        epsilon = float(privacy_engine.accountant.get_epsilon(delta=args.delta))
        checkpoint = {
            "schema": "new_stage_a_dp_expert_checkpoint_v1",
            "dataset": dataset,
            "party_id": party_id,
            "expert_id": expert_id,
            "next_epoch": epoch + 1,
            "optimizer_steps": optimizer_steps,
            "model_state": unwrap_private_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "accountant_state": privacy_engine.accountant.state_dict(),
            "trajectory": trajectory,
            "rng_state": capture_rng_state(),
        }
        atomic_torch_save(checkpoint, checkpoint_path)
        if epoch + 1 in log_epochs:
            print(
                "  party={:02d} expert={:02d} epoch={:4d}/{:4d} "
                "loss={:.4f} train_acc={:.4f} epsilon={:.4f}".format(
                    party_id,
                    expert_id,
                    epoch + 1,
                    args.epochs,
                    total_loss / max(count, 1),
                    correct / max(count, 1),
                    epsilon,
                ),
                flush=True,
            )

    payload = {
        "schema": "new_stage_a_dp_trajectory_v1",
        "dp_trajectory": True,
        "dataset": dataset,
        "party_id": party_id,
        "expert_id": expert_id,
        "seed": task_seed,
        "num_party_records": len(party_indices),
        "trajectory": trajectory,
        "trajectory_epochs": args.epochs,
        "optimizer_steps": optimizer_steps,
        "noise_multiplier": args.noise_multiplier,
        "max_grad_norm": args.max_grad_norm,
        "sample_rate": 1.0 / len(private_loader),
        "delta": args.delta,
        "epsilon": float(privacy_engine.accountant.get_epsilon(delta=args.delta)),
        "accountant_type": "opacus_rdp",
        "accountant_history": privacy_engine.accountant.state_dict()["history"],
        "model_schema_sha256": model_schema_hash(model),
        "runtime_seconds": time.time() - started,
        "raw_data_access": "expert_training_only",
        "secure_rng": args.secure_rng,
    }
    atomic_torch_save(payload, trajectory_path)
    return trajectory_path


def write_manifest(dataset: str, output_dir: Path, args, party_splits: List[np.ndarray]) -> Path:
    trajectory_paths = sorted(output_dir.glob("new_dp_trajectory_{}_party*_expert*.pt".format(dataset)))
    party_histories: Dict[int, List[Sequence]] = {party_id: [] for party_id in range(args.num_parties)}
    rows = []
    for path in trajectory_paths:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("dp_trajectory") is not True:
            raise RuntimeError("检测到非 DP trajectory: {}".format(path))
        party_histories[int(payload["party_id"])].append(payload["accountant_history"])
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "party_id": payload["party_id"],
                "expert_id": payload["expert_id"],
                "epsilon_individual": payload["epsilon"],
                "steps": payload["optimizer_steps"],
            }
        )
    party_epsilons = {
        str(party_id): compose_rdp_histories(histories, args.delta)
        for party_id, histories in party_histories.items()
        if histories
    }
    manifest = {
        "schema": "new_stage_a_dp_trajectory_manifest_v1",
        "dataset": dataset,
        "original_record_privacy_scope": "stage_a_dp_trajectory",
        "stage_a_epsilon": max(party_epsilons.values()) if party_epsilons else None,
        "stage_a_delta": args.delta,
        "accountant_type": "opacus_rdp",
        "composition": {
            "within_party": "sequential_rdp",
            "across_disjoint_parties": "parallel_max_epsilon",
        },
        "num_parties": args.num_parties,
        "selected_party_ids": args.selected_party_ids,
        "coverage_status": (
            "COMPLETE"
            if args.selected_party_ids == list(range(args.num_parties))
            else "DIAGNOSTIC_PARTIAL"
        ),
        "num_experts_per_party": args.num_experts_per_party,
        "num_expert_trajectories": len(rows),
        "party_record_counts": [len(indices) for indices in party_splits],
        "party_composed_epsilons": party_epsilons,
        "noise_multiplier": args.noise_multiplier,
        "max_grad_norm": args.max_grad_norm,
        "epochs_per_expert": args.epochs,
        "delta": args.delta,
        "secure_rng": args.secure_rng,
        "privacy_assumptions": [
            "蒸馏优化不得重新访问 raw training data",
            "发布的 synthetic images 只能由此 manifest 中的 DP trajectories 生成",
            "任何 non-DP trajectory 不得放入本输出目录",
        ],
        "trajectories": rows,
    }
    path = output_dir / "new_stage_a_dp_trajectory_manifest_{}.json".format(dataset)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成可会计的 DP expert trajectories")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="cifar10")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_parties", type=int, default=5)
    parser.add_argument("--num_experts_per_party", type=int, default=2)
    parser.add_argument(
        "--party_ids",
        type=str,
        default="all",
        help="诊断时可指定逗号分隔机构编号；正式运行必须为 all",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--noise_multiplier", type=float, default=1.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--secure_rng",
        action="store_true",
        help="正式运行启用 Opacus secure RNG；不可用时硬失败，不允许回退",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    configure_single_thread_runtime(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境未检测到 GPU")
    device = torch.device(args.device)
    dataset_obj = load_private_training_dataset(args.dataset)
    labels = dataset_labels(dataset_obj)
    splits = make_disjoint_party_splits(labels, args.num_parties, args.seed)
    if args.party_ids == "all":
        args.selected_party_ids = list(range(args.num_parties))
    else:
        args.selected_party_ids = sorted({int(value) for value in args.party_ids.split(",")})
        if not args.selected_party_ids or min(args.selected_party_ids) < 0:
            raise ValueError("party_ids 不能为空或为负数")
        if max(args.selected_party_ids) >= args.num_parties:
            raise ValueError("party_ids 超出 num_parties")
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 88, flush=True)
    print("Stage A Track B：DP-SGD experts -> DP trajectories", flush=True)
    print("dataset={} records={} parties={} experts/party={} epochs={}".format(
        args.dataset, len(dataset_obj), args.num_parties, args.num_experts_per_party, args.epochs
    ), flush=True)
    print("device={} cpu_threads=1 num_workers=0 noise={} clip={} delta={} secure_rng={}".format(
        device, args.noise_multiplier, args.max_grad_norm, args.delta, args.secure_rng
    ), flush=True)
    print("selected_party_ids={}".format(args.selected_party_ids), flush=True)
    print("预计总 expert runs: {}".format(
        len(args.selected_party_ids) * args.num_experts_per_party
    ), flush=True)
    print("output: {}".format(output_dir), flush=True)
    print("=" * 88, flush=True)
    for party_id in args.selected_party_ids:
        indices = splits[party_id]
        for expert_id in range(args.num_experts_per_party):
            train_one_expert(
                args.dataset,
                party_id,
                expert_id,
                indices,
                dataset_obj,
                output_dir,
                args,
                device,
            )
    manifest_path = write_manifest(args.dataset, output_dir, args, splits)
    print("manifest: {}".format(manifest_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

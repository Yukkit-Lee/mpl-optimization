#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：由 public MTT 或 DP-MTT 蒸馏图像训练唯一共享 ConvNet，并导出 2048 维特征。

共享 encoder 只在 synthetic encoder-train split 上训练，checkpoint 仅由 synthetic
reference split 选择；Stage-B selection split 独立保留。scaler 只拟合训练特征，
随后将每条特征裁剪到固定 L2 上界。test 默认不加载，只有最终正式导出时显式
传入 --include_test 才执行只读编码，且 test 不参与训练、scaler 或模型选择。
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from new_secure_feature_models import (
    DATASET_SPECS,
    atomic_torch_save,
    build_mtt_convnet,
    capture_rng_state,
    configure_single_thread_runtime,
    model_schema_hash,
    restore_rng_state,
    sha256_file,
)


PROJECT_ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_ROOT = Path(os.environ.get("NEW_DATA_ROOT", "/home/lixujie/tmpWork"))
OFFICIAL_ROOT = Path(
    os.environ.get(
        "NEW_OFFICIAL_MTT_ROOT",
        "/home/lixujie/tmpWork/mtt_data/georgecazenavette",
    )
)
OUTPUT_ROOT = Path(os.environ.get("NEW_OUTPUT_ROOT", str(PROJECT_ROOT)))
DEFAULT_DP_ROOT = OUTPUT_ROOT / "new_cache" / "new_secure_feature" / "new_stage_a_dp_distilled"
DEFAULT_OUTPUT = OUTPUT_ROOT / "new_cache" / "new_secure_feature" / "new_shared_encoder"

RAW_TEST_PATHS = {
    "cifar10": DATA_ROOT / "mtt_cifar10_work" / "data",
    "cifar100": DATA_ROOT / "mtt_cifar100_work" / "data",
    "tiny": DATA_ROOT / "mtt_tiny_work" / "data" / "tiny-imagenet-200",
}
NORMALIZATION = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    "cifar100": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    "tiny": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def load_public_mtt(dataset: str) -> Dict:
    folder = OFFICIAL_ROOT / "{}_50".format(dataset)
    image_paths = sorted(folder.glob("images_best*.pt"))
    label_paths = sorted(folder.glob("labels_best*.pt"))
    if not image_paths or not label_paths:
        raise FileNotFoundError("official MTT ipc=50 文件不完整: {}".format(folder))
    images = torch.cat(
        [torch.load(str(path), map_location="cpu", weights_only=False).float() for path in image_paths]
    )
    labels = torch.cat(
        [torch.load(str(path), map_location="cpu", weights_only=False).long() for path in label_paths]
    )
    return {
        "images": images,
        "labels": labels,
        "stage_a_track": "public_mtt_benchmark",
        "stage_a_epsilon": None,
        "stage_a_delta": None,
        "source_paths": [str(path) for path in image_paths + label_paths],
        "source_sha256": {str(path): sha256_file(path) for path in image_paths + label_paths},
    }


def load_dp_mtt(dataset: str, dp_root: Path, seed: int, ipc: int) -> Dict:
    path = dp_root / dataset / "new_dp_mtt_images_{}_ipc{}_seed{}.pt".format(dataset, ipc, seed)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != "new_stage_a_dp_mtt_distilled_v1":
        raise RuntimeError("拒绝未知 DP-MTT 文件格式: {}".format(path))
    if payload.get("raw_data_access_during_distillation") is not False:
        raise RuntimeError("DP-MTT 文件未证明 raw-data 隔离")
    return {
        "images": payload["images"].float(),
        "labels": payload["labels"].long(),
        "stage_a_track": "dp_mtt",
        "stage_a_epsilon": payload["stage_a_epsilon"],
        "stage_a_delta": payload["stage_a_delta"],
        "source_paths": [str(path)],
        "source_sha256": {str(path): sha256_file(path)},
    }


def stratified_synthetic_splits(labels: torch.Tensor, seed: int) -> Dict[str, torch.Tensor]:
    """逐类切分，保证 encoder/reference/selection 三者互斥且均覆盖全部类别。"""

    rng = np.random.RandomState(seed)
    result: Dict[str, List[int]] = {"encoder_train": [], "reference": [], "selection": []}
    for class_id in sorted(torch.unique(labels).tolist()):
        indices = torch.nonzero(labels == int(class_id), as_tuple=False).flatten().numpy()
        rng.shuffle(indices)
        count = len(indices)
        if count < 3:
            raise ValueError("每类至少需要 3 个 synthetic samples，当前 class {} 只有 {}".format(class_id, count))
        reference_count = max(1, int(round(count * 0.1)))
        selection_count = max(1, int(round(count * 0.1)))
        if count - reference_count - selection_count < 1:
            reference_count = 1
            selection_count = 1
        result["reference"].extend(indices[:reference_count].tolist())
        result["selection"].extend(indices[reference_count : reference_count + selection_count].tolist())
        result["encoder_train"].extend(indices[reference_count + selection_count :].tolist())
    tensors = {
        name: torch.tensor(sorted(indices), dtype=torch.long)
        for name, indices in result.items()
    }
    sets = [set(value.tolist()) for value in tensors.values()]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("synthetic split 存在重叠")
    if sum(len(value) for value in sets) != len(labels):
        raise AssertionError("synthetic split 未覆盖全部样本")
    return tensors


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """GPU 上执行轻量随机平移和翻转，不创建 DataLoader worker。"""

    batch, _, height, width = images.shape
    padded = nn.functional.pad(images, (4, 4, 4, 4), mode="reflect")
    output = torch.empty_like(images)
    offsets_y = torch.randint(0, 9, (batch,), device=images.device)
    offsets_x = torch.randint(0, 9, (batch,), device=images.device)
    flips = torch.rand(batch, device=images.device) < 0.5
    for index in range(batch):
        crop = padded[index, :, offsets_y[index] : offsets_y[index] + height, offsets_x[index] : offsets_x[index] + width]
        output[index] = torch.flip(crop, dims=(2,)) if flips[index] else crop
    return output


@torch.no_grad()
def evaluate_tensor(model, images: torch.Tensor, labels: torch.Tensor, device: torch.device, batch_size: int) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(labels), batch_size):
        x = images[start : start + batch_size].to(device)
        y = labels[start : start + batch_size].to(device)
        correct += int((model(x).argmax(1) == y).sum().item())
    return correct / max(len(labels), 1)


def train_shared_encoder(source: Dict, dataset: str, splits: Dict[str, torch.Tensor], args, device: torch.device, output_dir: Path):
    images = source["images"]
    labels = source["labels"]
    x_train = images[splits["encoder_train"]]
    y_train = labels[splits["encoder_train"]]
    x_reference = images[splits["reference"]]
    y_reference = labels[splits["reference"]]
    model = build_mtt_convnet(dataset, args.seed, device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = output_dir / "new_shared_encoder_checkpoint_{}_{}_seed{}.pt".format(
        dataset, source["stage_a_track"], args.seed
    )
    start_epoch = 0
    history = []
    best_reference = -1.0
    best_state = copy.deepcopy(model.state_dict())
    if checkpoint_path.exists() and not args.force:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["next_epoch"])
        history = checkpoint["history"]
        best_reference = float(checkpoint["best_reference_accuracy"])
        best_state = checkpoint["best_state"]
        restore_rng_state(checkpoint["rng_state"])
        print("从 epoch {} 恢复共享 encoder".format(start_epoch), flush=True)

    log_epochs = set(np.linspace(1, args.epochs, min(10, args.epochs), dtype=int).tolist())
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        permutation = torch.randperm(len(y_train), generator=generator)
        total_loss = 0.0
        correct = 0
        count = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            x = x_train[indices].to(device)
            y = y_train[indices].to(device)
            if args.augmentation:
                x = augment_batch(x)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            count += len(y)
        scheduler.step()
        reference_accuracy = evaluate_tensor(
            model, x_reference, y_reference, device, args.eval_batch_size
        )
        if reference_accuracy > best_reference:
            best_reference = reference_accuracy
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(count, 1),
                "train_accuracy": correct / max(count, 1),
                "reference_accuracy": reference_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        atomic_torch_save(
            {
                "schema": "new_shared_encoder_training_checkpoint_v1",
                "dataset": dataset,
                "stage_a_track": source["stage_a_track"],
                "next_epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_reference_accuracy": best_reference,
                "best_state": best_state,
                "history": history,
                "rng_state": capture_rng_state(),
            },
            checkpoint_path,
        )
        if epoch + 1 in log_epochs:
            row = history[-1]
            print(
                "  epoch={:4d}/{:4d} train_loss={:.4f} train_acc={:.4f} ref_acc={:.4f}".format(
                    epoch + 1,
                    args.epochs,
                    row["train_loss"],
                    row["train_accuracy"],
                    row["reference_accuracy"],
                ),
                flush=True,
            )
    model.load_state_dict(best_state)
    return model, history, best_reference, time.time() - started


@torch.no_grad()
def extract_features(model, images: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, len(images), batch_size):
        chunks.append(model.encode(images[start : start + batch_size].to(device)).cpu())
    features = torch.cat(chunks)
    if features.shape[1] != 2048:
        raise AssertionError("共享特征维度不是 2048")
    return features


def fit_and_apply_scaler(
    train_features: torch.Tensor,
    feature_groups: Dict[str, torch.Tensor],
    norm_bound: float,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    transformed = {}
    for name, features in feature_groups.items():
        normalized = (features - mean) / std
        norms = normalized.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(norm_bound / norms, max=1.0)
        transformed[name] = normalized * scale
        if float(transformed[name].norm(p=2, dim=1).max()) > norm_bound + 1e-5:
            raise AssertionError("特征 L2 上界失效: {}".format(name))
    return transformed, {"mean": mean, "std": std, "norm_bound": norm_bound}


def load_test_dataset(dataset: str):
    mean, std = NORMALIZATION[dataset]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    root = RAW_TEST_PATHS[dataset]
    if dataset == "cifar10":
        return datasets.CIFAR10(str(root), train=False, download=False, transform=transform)
    if dataset == "cifar100":
        return datasets.CIFAR100(str(root), train=False, download=False, transform=transform)
    return datasets.ImageFolder(str(root / "val" / "images"), transform=transform)


def materialize_dataset(dataset_obj) -> Tuple[torch.Tensor, torch.Tensor]:
    images, labels = [], []
    for index in range(len(dataset_obj)):
        image, label = dataset_obj[index]
        images.append(image)
        labels.append(int(label))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def main() -> int:
    parser = argparse.ArgumentParser(description="训练共享 MTT ConvNet 并导出 2048-D 特征")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="cifar10")
    parser.add_argument("--track", choices=("public_mtt_benchmark", "dp_mtt"), default="public_mtt_benchmark")
    parser.add_argument("--dp_root", type=Path, default=DEFAULT_DP_ROOT)
    parser.add_argument("--dp_ipc", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--feature_norm_bound", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--include_test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configure_single_thread_runtime(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境未检测到 GPU")
    source = (
        load_public_mtt(args.dataset)
        if args.track == "public_mtt_benchmark"
        else load_dp_mtt(args.dataset, args.dp_root, args.seed, args.dp_ipc)
    )
    splits = stratified_synthetic_splits(source["labels"], args.seed)
    output_dir = args.output_dir / args.track / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 92, flush=True)
    print("Stage A Phase 2：唯一共享 ConvNet -> 冻结 2048-D representation", flush=True)
    print("dataset={} track={} samples={} epochs={} device={}".format(
        args.dataset, args.track, len(source["labels"]), args.epochs, device
    ), flush=True)
    print("split_sizes={} cpu_threads=1 num_workers=0 include_test={}".format(
        {name: len(value) for name, value in splits.items()}, args.include_test
    ), flush=True)
    print("output: {}".format(output_dir), flush=True)
    print("=" * 92, flush=True)
    model, history, best_reference, runtime_seconds = train_shared_encoder(
        source, args.dataset, splits, args, device, output_dir
    )
    encoder_path = output_dir / "new_shared_encoder_{}_{}_seed{}.pt".format(
        args.dataset, args.track, args.seed
    )
    atomic_torch_save(
        {
            "schema": "new_shared_encoder_v1",
            "dataset": args.dataset,
            "stage_a_track": args.track,
            "model_state": model.state_dict(),
            "feature_dim": 2048,
            "model_schema_sha256": model_schema_hash(model),
            "best_reference_accuracy": best_reference,
            "encoder_training_history": history,
            "encoder_training_seconds": runtime_seconds,
            "stage_a_epsilon": source["stage_a_epsilon"],
            "stage_a_delta": source["stage_a_delta"],
            "source_sha256": source["source_sha256"],
        },
        encoder_path,
    )
    encoder_sha256 = sha256_file(encoder_path)

    images = source["images"]
    labels = source["labels"]
    raw_features = {
        name: extract_features(model, images[indices], device, args.eval_batch_size)
        for name, indices in splits.items()
    }
    test_labels = None
    if args.include_test:
        test_images, test_labels = materialize_dataset(load_test_dataset(args.dataset))
        raw_features["test"] = extract_features(model, test_images, device, args.eval_batch_size)
    transformed, scaler = fit_and_apply_scaler(
        raw_features["encoder_train"], raw_features, args.feature_norm_bound
    )
    scaler_path = output_dir / "new_feature_scaler_{}_{}_seed{}.pt".format(
        args.dataset, args.track, args.seed
    )
    atomic_torch_save(scaler, scaler_path)
    cache_path = output_dir / "new_feature_cache_{}_{}_seed{}.pt".format(
        args.dataset, args.track, args.seed
    )
    cache = {
        "schema": "new_shared_2048_feature_cache_v1",
        "dataset": args.dataset,
        "stage_a_track": args.track,
        "feature_dim": 2048,
        "encoder_sha256": encoder_sha256,
        "scaler_sha256": sha256_file(scaler_path),
        "input_norm_bound": args.feature_norm_bound,
        "split_indices": splits,
        "x_encoder_train": transformed["encoder_train"],
        "y_encoder_train": labels[splits["encoder_train"]],
        "x_reference": transformed["reference"],
        "y_reference": labels[splits["reference"]],
        "x_selection": transformed["selection"],
        "y_selection": labels[splits["selection"]],
        "x_test": transformed.get("test"),
        "y_test": test_labels,
        "test_loaded": args.include_test,
        "scaler_fit_scope": "synthetic_encoder_train_only",
        "encoder_selection_scope": "synthetic_reference_only",
        "test_usage": "final_readonly_encoding" if args.include_test else "not_loaded",
        "stage_a_epsilon": source["stage_a_epsilon"],
        "stage_a_delta": source["stage_a_delta"],
    }
    atomic_torch_save(cache, cache_path)
    legacy_npz_path = output_dir / "new_feature_cache_{}_{}_seed{}.npz".format(
        args.dataset, args.track, args.seed
    )
    np.savez_compressed(
        str(legacy_npz_path),
        x_encoder_train=transformed["encoder_train"].numpy().astype(np.float32),
        y_encoder_train=labels[splits["encoder_train"]].numpy().astype(np.int64),
        x_reference=transformed["reference"].numpy().astype(np.float32),
        y_reference=labels[splits["reference"]].numpy().astype(np.int64),
        x_selection=transformed["selection"].numpy().astype(np.float32),
        y_selection=labels[splits["selection"]].numpy().astype(np.int64),
        x_test=(
            transformed["test"].numpy().astype(np.float32)
            if args.include_test
            else np.empty((0, 2048), dtype=np.float32)
        ),
        y_test=(
            test_labels.numpy().astype(np.int64)
            if args.include_test
            else np.empty((0,), dtype=np.int64)
        ),
        feature_dim=np.asarray([2048], dtype=np.int64),
        input_norm_bound=np.asarray([args.feature_norm_bound], dtype=np.float32),
        test_loaded=np.asarray([int(args.include_test)], dtype=np.int8),
    )
    metadata = {
        "dataset": args.dataset,
        "stage_a_track": args.track,
        "feature_dim": 2048,
        "encoder_path": str(encoder_path),
        "encoder_sha256": encoder_sha256,
        "scaler_path": str(scaler_path),
        "scaler_sha256": sha256_file(scaler_path),
        "feature_cache_path": str(cache_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "legacy_npz_path": str(legacy_npz_path),
        "legacy_npz_sha256": sha256_file(legacy_npz_path),
        "input_norm_bound": args.feature_norm_bound,
        "best_reference_accuracy": best_reference,
        "test_loaded": args.include_test,
        "status": "COMPLETE",
    }
    metadata_path = output_dir / "new_shared_feature_metadata_{}_{}_seed{}.json".format(
        args.dataset, args.track, args.seed
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("encoder_sha256: {}".format(encoder_sha256), flush=True)
    print("feature_cache: {}".format(cache_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

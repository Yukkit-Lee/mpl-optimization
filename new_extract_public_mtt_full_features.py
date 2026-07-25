#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpointable raw-train feature extraction with the frozen Public-MTT encoder."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torchvision import datasets, transforms

from new_public_mtt_mainline_common import (
    ROOT,
    feature_cache_path,
    sha256_file,
)
from new_secure_feature_models import (
    build_mtt_convnet,
    configure_single_thread_runtime,
)
from new_stage_a_shared_encoder import NORMALIZATION, RAW_TEST_PATHS


def atomic_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new_tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def load_train_dataset(dataset):
    mean, std = NORMALIZATION[dataset]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    root = RAW_TEST_PATHS[dataset]
    if dataset == "cifar10":
        return datasets.CIFAR10(
            str(root), train=True, download=False, transform=transform
        )
    if dataset == "cifar100":
        return datasets.CIFAR100(
            str(root), train=True, download=False, transform=transform
        )
    return datasets.ImageFolder(str(root / "train"), transform=transform)


def main():
    parser = argparse.ArgumentParser(
        description="Extract full training features with frozen Public-MTT encoder"
    )
    parser.add_argument(
        "--dataset", choices=("cifar10", "cifar100", "tiny"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    configure_single_thread_runtime(args.seed)
    device = torch.device(args.device)
    source_cache = feature_cache_path(args.dataset, args.seed)
    directory = source_cache.parent
    encoder_path = directory / (
        "new_shared_encoder_{}_public_mtt_benchmark_seed{}.pt".format(
            args.dataset, args.seed
        )
    )
    scaler_path = directory / (
        "new_feature_scaler_{}_public_mtt_benchmark_seed{}.pt".format(
            args.dataset, args.seed
        )
    )
    encoder = torch.load(
        str(encoder_path), map_location=device, weights_only=False
    )
    scaler = torch.load(
        str(scaler_path), map_location="cpu", weights_only=False
    )
    model = build_mtt_convnet(args.dataset, args.seed, device)
    model.load_state_dict(encoder["model_state"])
    model.eval()
    mean = scaler["mean"].float()
    std = scaler["std"].float()
    norm_bound = float(scaler["norm_bound"])
    dataset = load_train_dataset(args.dataset)
    output_dir = (
        ROOT
        / "new_cache/new_public_mtt_mainline/new_full_features"
        / args.dataset
        / "new_seed{}".format(args.seed)
    )
    x_path = output_dir / "new_full_train_features.npy"
    y_path = output_dir / "new_full_train_labels.npy"
    state_path = output_dir / "new_full_feature_extraction_state.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if x_path.exists() and y_path.exists():
        x_out = open_memmap(str(x_path), mode="r+")
        y_out = open_memmap(str(y_path), mode="r+")
    else:
        x_out = open_memmap(
            str(x_path), mode="w+", dtype="float32", shape=(len(dataset), 2048)
        )
        y_out = open_memmap(
            str(y_path), mode="w+", dtype="int64", shape=(len(dataset),)
        )
    start = 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start = int(state.get("next_index", 0))
    print("=" * 88, flush=True)
    print("Public MTT shared encoder -> Full Images 2048-D features", flush=True)
    print(
        "dataset={} seed={} samples={} start={} batch={} device={}".format(
            args.dataset, args.seed, len(dataset), start, args.batch_size, device
        ),
        flush=True,
    )
    print("cpu_threads=1 num_workers=0 test_loaded=False", flush=True)
    print("=" * 88, flush=True)
    log_points = set(
        np.linspace(1, len(dataset), 10, dtype=np.int64).tolist()
    )
    with torch.no_grad():
        for offset in range(start, len(dataset), args.batch_size):
            end = min(offset + args.batch_size, len(dataset))
            items = [dataset[index] for index in range(offset, end)]
            images = torch.stack([item[0] for item in items]).to(device)
            labels = np.asarray([int(item[1]) for item in items], dtype=np.int64)
            features = model.encode(images).cpu()
            normalized = (features - mean) / std
            norms = normalized.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            transformed = normalized * torch.clamp(
                norm_bound / norms, max=1.0
            )
            x_out[offset:end] = transformed.numpy()
            y_out[offset:end] = labels
            x_out.flush()
            y_out.flush()
            atomic_json(
                {
                    "schema": "new_full_feature_extraction_state_v1",
                    "status": "RUNNING" if end < len(dataset) else "COMPLETE",
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "next_index": end,
                    "total_samples": len(dataset),
                    "feature_dim": 2048,
                    "encoder_sha256": sha256_file(encoder_path),
                    "scaler_sha256": sha256_file(scaler_path),
                    "test_loaded": False,
                    "cpu_threads": 1,
                    "num_workers": 0,
                },
                state_path,
            )
            if end in log_points or end == len(dataset):
                print(
                    "  extracted={}/{}".format(end, len(dataset)), flush=True
                )
    manifest = {
        "schema": "new_full_feature_cache_v1",
        "status": "COMPLETE",
        "dataset": args.dataset,
        "seed": args.seed,
        "num_samples": len(dataset),
        "feature_dim": 2048,
        "feature_path": str(x_path),
        "label_path": str(y_path),
        "feature_sha256": sha256_file(x_path),
        "label_sha256": sha256_file(y_path),
        "encoder_sha256": sha256_file(encoder_path),
        "scaler_sha256": sha256_file(scaler_path),
        "test_loaded": False,
    }
    atomic_json(manifest, output_dir / "new_full_feature_manifest.json")
    print("manifest={}".format(output_dir / "new_full_feature_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

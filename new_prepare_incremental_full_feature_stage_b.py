#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3: Generate Full Data NPZ for Stage B, plus memmap adapter.

This script handles TWO modes:

1. --mode npz: Build a full NPZ from metadata + memmapped .npy files.
   This is backward-compatible with the existing Stage B entry point
   but duplicates the feature data (~270-550 MiB per file after compression).

2. --mode memmap-adapter: Generate a small adapter NPZ that Stage B can load,
   where x_encoder_train is a memmap-backed lazy array. This avoids
   duplicating the full feature data.

3. --mode benchmark: Compare NPZ vs memmap loading time and memory.
"""

import argparse
import json
import os
import sys
import time
import hashlib

import numpy as np


PROJECT_ROOT = os.environ.get(
    "NEW_PROJECT_ROOT", "/root/autodl-tmp/tfe_experiments"
)

METADATA_DIR = os.path.join(
    PROJECT_ROOT, "new_cache", "new_incremental_comparison",
    "new_full_feature_metadata"
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "new_cache", "new_incremental_comparison",
)
MANIFEST_PATH = os.path.join(
    PROJECT_ROOT, "new_log", "new_incremental_comparison", "json",
    "new_incremental_full_feature_asset_manifest.json",
)


def get_asset_paths(dataset: str, seed: int):
    """Resolve paths for a given asset from the manifest."""
    dirname = "tiny" if dataset == "tiny" else dataset
    meta_npz = os.path.join(
        METADATA_DIR, dirname, f"new_seed{seed}",
        f"new_full_feature_metadata_{dataset}_seed{seed}.npz",
    )
    full_npy = os.path.join(
        PROJECT_ROOT, "new_cache", "new_public_mtt_mainline",
        "new_full_features", dirname, f"new_seed{seed}",
        f"new_full_train_features.npy",
    )
    full_lbl = os.path.join(
        PROJECT_ROOT, "new_cache", "new_public_mtt_mainline",
        "new_full_features", dirname, f"new_seed{seed}",
        f"new_full_train_labels.npy",
    )
    return meta_npz, full_npy, full_lbl


def build_npz(dataset: str, seed: int, output_dir: str):
    """Build a full training NPZ for Stage B from metadata + .npy files."""
    meta_npz, full_npy, full_lbl = get_asset_paths(dataset, seed)

    print(f"Loading metadata: {meta_npz}")
    meta = np.load(meta_npz, allow_pickle=False)
    train_indices = np.asarray(meta["train_indices"])
    y_train = np.asarray(meta["y_encoder_train"])

    print(f"mmap features: {full_npy}")
    t0 = time.time()
    x_full = np.load(full_npy, mmap_mode="r")
    x_train = np.array(x_full[train_indices], dtype=np.float32)
    load_time = time.time() - t0
    print(f"  Loaded {x_train.shape[0]} samples in {load_time:.1f}s")

    dirname = "tiny" if dataset == "tiny" else dataset
    out_path = os.path.join(
        output_dir, "new_full_feature_npz", dirname, f"new_seed{seed}",
        f"new_full_feature_{dataset}_seed{seed}.npz",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    t0 = time.time()
    arrays = {
        "x_encoder_train": x_train,
        "y_encoder_train": y_train,
        "x_reference": np.asarray(meta["x_reference"]),
        "y_reference": np.asarray(meta["y_reference"]),
        "x_selection": np.asarray(meta["x_selection"]),
        "y_selection": np.asarray(meta["y_selection"]),
        "feature_dim": np.asarray(meta["feature_dim"]),
        "encoder_sha256": np.asarray(meta["encoder_sha256"]),
        "scaler_sha256": np.asarray(meta["scaler_sha256"]),
        "test_loaded": np.asarray(meta["test_loaded"]),
    }
    temporary = out_path + ".new_tmp"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, out_path)
    save_time = time.time() - t0
    size_mb = os.path.getsize(out_path) / (1024 * 1024)

    sha = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)

    print(f"  Saved: {out_path} ({size_mb:.0f} MiB, {save_time:.1f}s)")
    print(f"  SHA256: {sha.hexdigest()}")
    return out_path


def build_memmap_adapter(dataset: str, seed: int, output_dir: str):
    """Build a memmap adapter NPZ pointing to the existing .npy files.

    This NPZ is small (~100 KiB) and contains only metadata + path references.
    The actual features stay in the .npy files.
    """
    meta_npz, full_npy, full_lbl = get_asset_paths(dataset, seed)
    meta = np.load(meta_npz, allow_pickle=False)

    dirname = "tiny" if dataset == "tiny" else dataset
    out_path = os.path.join(
        output_dir, "new_full_feature_memmap", dirname, f"new_seed{seed}",
        f"new_full_feature_memmap_adapter_{dataset}_seed{seed}.npz",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    arrays = {
        "train_indices": np.asarray(meta["train_indices"]),
        "y_encoder_train": np.asarray(meta["y_encoder_train"]),
        "x_reference": np.asarray(meta["x_reference"]),
        "y_reference": np.asarray(meta["y_reference"]),
        "x_selection": np.asarray(meta["x_selection"]),
        "y_selection": np.asarray(meta["y_selection"]),
        "feature_dim": np.asarray(meta["feature_dim"]),
        "encoder_sha256": np.asarray(meta["encoder_sha256"]),
        "scaler_sha256": np.asarray(meta["scaler_sha256"]),
        "test_loaded": np.asarray(meta["test_loaded"]),
        "feature_npy_path": np.array([full_npy.encode("utf-8")]),
        "label_npy_path": np.array([full_lbl.encode("utf-8")]),
    }
    np.savez_compressed(out_path, **arrays)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Memmap adapter saved: {out_path} ({size_kb:.0f} KiB)")


def benchmark(dataset: str, seed: int):
    """Benchmark NPZ loading vs memmap loading time and memory."""
    import resource

    meta_npz, full_npy, full_lbl = get_asset_paths(dataset, seed)
    meta = np.load(meta_npz, allow_pickle=False)
    train_indices = np.asarray(meta["train_indices"])

    print(f"\n=== Benchmark {dataset} seed{seed} ===")
    print(f"  train samples: {len(train_indices)}")

    # NPZ write benchmark
    print(f"\n  [NPZ build from mmap]")
    t0 = time.time()
    x_full = np.load(full_npy, mmap_mode="r")
    x_train_copy = np.array(x_full[train_indices], dtype=np.float32)
    copy_time = time.time() - t0
    mem_after_copy = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"    memmap slice copy: {copy_time:.1f}s, x_train={x_train_copy.shape}")
    print(f"    size in memory: {x_train_copy.nbytes / (1024*1024):.0f} MiB")

    # NPZ save benchmark
    tmp = "/tmp/bench_full_feature.npz"
    t0 = time.time()
    np.savez_compressed(tmp, x_encoder_train=x_train_copy)
    save_time = time.time() - t0
    size_mb = os.path.getsize(tmp) / (1024*1024)
    os.remove(tmp)
    print(f"    np.savez_compressed: {save_time:.1f}s, size={size_mb:.0f} MiB")

    # NPZ load benchmark
    # (use the freshly built one from above)
    out_path = os.path.join(
        OUTPUT_DIR, "new_full_feature_npz", "tiny" if dataset == "tiny" else dataset,
        f"new_seed{seed}", f"new_full_feature_{dataset}_seed{seed}.npz",
    )
    if os.path.exists(out_path):
        t0 = time.time()
        loaded = np.load(out_path, allow_pickle=False)
        load_time = time.time() - t0
        mem_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"    np.load NPZ: {load_time:.1f}s")

    # Memmap direct load benchmark
    print(f"\n  [Memmap direct load]")
    t0 = time.time()
    x_mmap = np.load(full_npy, mmap_mode="r")
    mmap_time = time.time() - t0
    print(f"    np.load mmap: {mmap_time:.3f}s")
    print(f"    shape: {x_mmap.shape}, dtype: {x_mmap.dtype}")

    # Batch access benchmark with indices
    print(f"\n  [Batch access via indices]")
    batch_size = 16
    n_batches = min(100, len(train_indices) // batch_size)
    t0 = time.time()
    for i in range(n_batches):
        start = i * batch_size
        batch_idx = train_indices[start : start + batch_size]
        _ = np.array(x_mmap[batch_idx], dtype=np.float32)
    batch_time = time.time() - t0
    print(f"    {n_batches} batches of {batch_size}: {batch_time:.3f}s")
    print(f"    per batch: {batch_time/n_batches*1000:.1f} ms")

    print(f"\n  Conclusion: NPZ save={save_time:.1f}s, size={size_mb:.0f} MiB | "
          f"Memmap batch access={batch_time/n_batches*1000:.1f} ms/batch")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Generate assets + benchmark")
    parser.add_argument("--mode", choices=["npz", "memmap-adapter", "benchmark"],
                        default="npz")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true",
                        help="Process all 9 assets")
    args = parser.parse_args()

    datasets = ["cifar10", "cifar100", "tiny"] if args.all else [args.dataset]
    seeds = [42, 43, 44] if args.all else [args.seed]

    if args.mode == "benchmark":
        benchmark(args.dataset, args.seed)
        return 0

    for ds in datasets:
        for s in seeds:
            if args.mode == "npz":
                build_npz(ds, s, OUTPUT_DIR)
            elif args.mode == "memmap-adapter":
                build_memmap_adapter(ds, s, OUTPUT_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

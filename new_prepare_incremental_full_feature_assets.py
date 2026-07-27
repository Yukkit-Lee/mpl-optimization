#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2: Prepare Full Data Feature assets for incremental comparison.

Performs:
1. Reconstruct shared reference/selection indices matching the old end-to-end cache
2. Verify exact match with old shared reference/selection
3. Generate a small metadata NPZ (indices + labels only, not the full 4.6 GiB features)
4. Produce a memmap manifest that lets Stage B load features from existing .npy files
5. Optionally generate full NPZ for correctness verification (--generate-npz)

The metadata NPZ contains:
  - train_indices: int64 array of training sample indices into the full .npy
  - y_encoder_train: int64 labels for training samples
  - x_reference, y_reference, x_selection, y_selection: shared holdout arrays
  - feature_dim, encoder_sha256, scaler_sha256, test_loaded: metadata scalars

The memmap manifest records how to resolve actual feature data from existing .npy files.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# Resolve project root
PROJECT_ROOT = os.environ.get(
    "NEW_PROJECT_ROOT", "/root/autodl-tmp/tfe_experiments"
)

FULL_FEATURES_DIR = os.path.join(
    PROJECT_ROOT, "new_cache", "new_public_mtt_mainline", "new_full_features"
)
OLD_E2E_DIR = os.path.join(
    PROJECT_ROOT, "new_cache", "new_public_mtt_mainline", "new_end_to_end_features"
)
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "new_cache", "new_incremental_comparison"
)
MANIFEST_DIR = os.path.join(
    PROJECT_ROOT, "new_log", "new_incremental_comparison", "json"
)

DATASETS = ["cifar10", "cifar100", "tiny"]
SEEDS = [42, 43, 44]


def sha256_bytes(data: np.ndarray) -> str:
    return hashlib.sha256(data.tobytes()).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_full_path(dataset: str, seed: int, kind: str) -> str:
    dirname = "tiny" if dataset == "tiny" else dataset
    fname = f"new_full_train_{kind}.npy"
    return os.path.join(FULL_FEATURES_DIR, dirname, f"new_seed{seed}", fname)


def resolve_old_official_stage_b_path(dataset: str, seed: int) -> str:
    """Path to the Official-MTT Stage-B NPZ used by old prepare script."""
    dirname = "tiny" if dataset == "tiny" else dataset
    return os.path.join(
        PROJECT_ROOT, "new_cache", "new_public_mtt_mainline",
        "new_stage_b_npz_round2", dirname,
        f"new_stage_b_features_{dataset}_seed{seed}.npz",
    )


def resolve_old_random_e2e_path(dataset: str, seed: int) -> str:
    """Path to the old end-to-end random_subset cache (for verification)."""
    return os.path.join(
        OLD_E2E_DIR, "random_subset", dataset,
        f"new_end_to_end_random_{dataset}_seed{seed}.npz",
    )


def stratified_indices(y_full: np.ndarray, official: dict, seed: int) -> dict:
    """Reconstruct the same stratified split as new_prepare_end_to_end_assets.py.

    Uses the Official-MTT Stage-B NPZ (round2) to determine per-class counts,
    ensuring the reference/selection indices match the existing end-to-end
    cache exactly.

    For Full Data, all candidates NOT allocated to reference or selection
    become training samples.

    Returns dict with keys: train_indices, reference_indices, selection_indices
    where each is an int64 array of positions into x_full/y_full.
    """
    off_y = official["y_encoder_train"]
    off_ref_y = official["y_reference"]
    off_sel_y = official["y_selection"]

    train_indices_list = []
    reference_indices_list = []
    selection_indices_list = []

    classes = np.unique(off_y)
    for class_id in classes:
        train_count = int(np.sum(off_y == class_id))
        ref_count = int(np.sum(off_ref_y == class_id))
        sel_count = int(np.sum(off_sel_y == class_id))
        needed = train_count + ref_count + sel_count

        candidates = np.flatnonzero(y_full == class_id)
        if needed > len(candidates):
            raise ValueError(
                f"insufficient full features for class {class_id}: "
                f"need {needed}, have {len(candidates)}"
            )
        rng = np.random.RandomState(int(seed) + int(class_id) * 1009 + 88001)
        chosen = candidates[rng.permutation(len(candidates))[:needed]]

        train_ids = chosen[:train_count]
        ref_ids = chosen[train_count : train_count + ref_count]
        sel_ids = chosen[train_count + ref_count : train_count + ref_count + sel_count]

        reference_indices_list.append(ref_ids)
        selection_indices_list.append(sel_ids)

        # Full Data training: ALL candidates minus reference+selection
        holdout_set = set(ref_ids) | set(sel_ids)
        train_for_class = np.array(
            [i for i in candidates if i not in holdout_set], dtype=np.int64
        )
        train_indices_list.append(train_for_class)

    train_indices = np.concatenate(train_indices_list)
    reference_indices = np.concatenate(reference_indices_list)
    selection_indices = np.concatenate(selection_indices_list)

    # Train indices are sorted for deterministic iteration order
    # Reference/selection indices are kept in per-class concatenation order
    # (matching old end-to-end cache ordering for verification)
    return {
        "train_indices": np.sort(train_indices),
        "reference_indices": np.concatenate(reference_indices_list),
        "selection_indices": np.concatenate(selection_indices_list),
    }


def build_metadata_npz(
    dataset: str, seed: int, x_full_path: str, y_full_path: str,
    indices: dict, manifest_data: dict, output_path: str,
):
    """Build a compact metadata NPZ.

    The metadata NPZ is SMALL: it stores only index arrays (int64) and small
    array-valued metadata. It does NOT duplicate the 4.6 GiB of features.
    The actual features remain in the .npy files for mmap access.
    """
    y_full = np.load(y_full_path, mmap_mode="r")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read small holdout arrays from mmap
    x_ref = np.array(
        np.load(x_full_path, mmap_mode="r")[indices["reference_indices"]],
        dtype=np.float32,
    )
    x_sel = np.array(
        np.load(x_full_path, mmap_mode="r")[indices["selection_indices"]],
        dtype=np.float32,
    )

    arrays = {
        "train_indices": indices["train_indices"].astype(np.int64),
        "y_encoder_train": y_full[indices["train_indices"]].astype(np.int64),
        "x_reference": x_ref,
        "y_reference": y_full[indices["reference_indices"]].astype(np.int64),
        "x_selection": x_sel,
        "y_selection": y_full[indices["selection_indices"]].astype(np.int64),
        "feature_dim": np.asarray([2048], dtype=np.int64),
        "encoder_sha256": np.array([manifest_data["encoder_sha256"]]),
        "scaler_sha256": np.array([manifest_data["scaler_sha256"]]),
        "test_loaded": np.asarray([0], dtype=np.int64),
    }

    temporary = output_path + ".new_tmp"
    with open(temporary, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, output_path)

    return {
        "num_train": len(indices["train_indices"]),
        "num_reference": len(indices["reference_indices"]),
        "num_selection": len(indices["selection_indices"]),
    }


def build_full_npz(
    dataset: str, seed: int, x_full_path: str, y_full_path: str,
    indices: dict, manifest_data: dict, output_path: str,
):
    """Build a full NPZ (duplicates features, 400-800 MiB per file).

    This is mainly for correctness verification against the memmap approach.
    """
    x_full = np.load(x_full_path, mmap_mode="r")
    y_full = np.load(y_full_path, mmap_mode="r")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    arrays = {
        "x_encoder_train": np.array(
            x_full[indices["train_indices"]], dtype=np.float32
        ),
        "y_encoder_train": y_full[indices["train_indices"]].astype(np.int64),
        "x_reference": np.array(
            x_full[indices["reference_indices"]], dtype=np.float32
        ),
        "y_reference": y_full[indices["reference_indices"]].astype(np.int64),
        "x_selection": np.array(
            x_full[indices["selection_indices"]], dtype=np.float32
        ),
        "y_selection": y_full[indices["selection_indices"]].astype(np.int64),
        "feature_dim": np.asarray([2048], dtype=np.int64),
        "encoder_sha256": np.array([manifest_data["encoder_sha256"]]),
        "scaler_sha256": np.array([manifest_data["scaler_sha256"]]),
        "test_loaded": np.asarray([0], dtype=np.int64),
    }

    temporary = output_path + ".new_tmp"
    with open(temporary, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Prepare Full Data Feature assets"
    )
    parser.add_argument(
        "--generate-npz", action="store_true",
        help="Also generate full NPZ files (one seed per dataset only, for verification)",
    )
    parser.add_argument(
        "--npz-dataset", default=None,
        help="Dataset for NPZ generation (default: cifar10 first seed only)",
    )
    parser.add_argument(
        "--npz-seed", type=int, default=42,
        help="Seed for NPZ generation",
    )
    args = parser.parse_args()

    manifest_rows = []
    all_verified = True

    for dataset in DATASETS:
        dirname = "tiny" if dataset == "tiny" else dataset
        for seed in SEEDS:
            key = f"{dataset}_seed{seed}"

            # Paths
            full_manifest_path = os.path.join(
                FULL_FEATURES_DIR, dirname, f"new_seed{seed}",
                "new_full_feature_manifest.json",
            )
            with open(full_manifest_path) as f:
                full_manifest = json.load(f)

            x_full_path = resolve_full_path(dataset, seed, "features")
            y_full_path = resolve_full_path(dataset, seed, "labels")

            # Load Official-MTT Stage-B NPZ (round2) for per-class count template
            official_path = resolve_old_official_stage_b_path(dataset, seed)
            official_npz = np.load(official_path, allow_pickle=False)
            official = {
                "y_encoder_train": np.asarray(official_npz["y_encoder_train"]),
                "y_reference": np.asarray(official_npz["y_reference"]),
                "y_selection": np.asarray(official_npz["y_selection"]),
            }

            if bool(np.asarray(official_npz.get("test_loaded", [1])).reshape(-1)[0]):
                raise ValueError(f"{official_path} is not test-free")

            y_full = np.load(y_full_path, mmap_mode="r")

            # Reconstruct indices using the same logic as the old script
            indices = stratified_indices(y_full, official, seed)

            # Load old end-to-end random_subset cache for verification
            old_random_path = resolve_old_random_e2e_path(dataset, seed)
            old_npz = np.load(old_random_path, allow_pickle=False)

            # --- Verification ---
            x_full = np.load(x_full_path, mmap_mode="r")

            # Verify reference match
            new_ref_x = np.array(
                x_full[indices["reference_indices"]], dtype=np.float32
            )
            new_sel_x = np.array(
                x_full[indices["selection_indices"]], dtype=np.float32
            )
            old_ref_x = np.asarray(old_npz["x_reference"])
            old_sel_x = np.asarray(old_npz["x_selection"])

            ref_match = np.allclose(new_ref_x, old_ref_x)
            sel_match = np.allclose(new_sel_x, old_sel_x)
            ref_label_match = np.array_equal(
                y_full[indices["reference_indices"]],
                np.asarray(old_npz["y_reference"]),
            )
            sel_label_match = np.array_equal(
                y_full[indices["selection_indices"]],
                np.asarray(old_npz["y_selection"]),
            )

            print(
                f"{key}: ref_x_match={ref_match}, sel_x_match={sel_match}, "
                f"ref_y_match={ref_label_match}, sel_y_match={sel_label_match}"
            )

            if not all([ref_match, sel_match, ref_label_match, sel_label_match]):
                all_verified = False
                print(f"  ERROR: {key} does not match old end-to-end cache!")
                continue

            # --- Build metadata NPZ ---
            meta_path = os.path.join(
                OUTPUT_DIR, "new_full_feature_metadata",
                dirname, f"new_seed{seed}",
                f"new_full_feature_metadata_{dataset}_seed{seed}.npz",
            )
            sizes = build_metadata_npz(
                dataset, seed, x_full_path, y_full_path,
                indices, full_manifest, meta_path,
            )

            # Store hashes of the metadata NPZ
            meta_sha = sha256_file(meta_path)

            manifest_rows.append({
                "dataset": dataset,
                "seed": seed,
                "num_train": sizes["num_train"],
                "num_reference": sizes["num_reference"],
                "num_selection": sizes["num_selection"],
                "feature_dim": 2048,
                "metadata_npz_path": meta_path,
                "metadata_npz_sha256": meta_sha,
                "feature_npy_path": x_full_path,
                "feature_npy_sha256": full_manifest["feature_sha256"],
                "label_npy_path": y_full_path,
                "label_npy_sha256": full_manifest["label_sha256"],
                "encoder_sha256": full_manifest["encoder_sha256"],
                "scaler_sha256": full_manifest["scaler_sha256"],
                "train_indices_sha256": sha256_bytes(indices["train_indices"]),
                "reference_indices_sha256": sha256_bytes(indices["reference_indices"]),
                "selection_indices_sha256": sha256_bytes(indices["selection_indices"]),
                "reference_match": True,
                "selection_match": True,
                "test_loaded": False,
                "memmap_compatible": True,
            })

            print(
                f"{key}: train={sizes['num_train']}, ref={sizes['num_reference']}, "
                f"sel={sizes['num_selection']}"
            )

            # --- Optionally generate full NPZ ---
            if args.generate_npz:
                if args.npz_dataset is None or dataset == args.npz_dataset:
                    if seed == args.npz_seed:
                        npz_path = os.path.join(
                            OUTPUT_DIR, "new_full_feature_npz",
                            dirname, f"new_seed{seed}",
                            f"new_full_feature_{dataset}_seed{seed}.npz",
                        )
                        build_full_npz(
                            dataset, seed, x_full_path, y_full_path,
                            indices, full_manifest, npz_path,
                        )
                        npz_sha = sha256_file(npz_path)
                        print(f"{key}: full NPZ saved ({npz_sha[:16]}...)")

    # --- Write manifest ---
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    manifest = {
        "schema": "new_incremental_full_feature_asset_manifest_v1",
        "status": "COMPLETE" if all_verified else "VERIFICATION_FAILED",
        "test_loaded": False,
        "shared_reference_protocol": True,
        "old_e2e_cache_verified": all_verified,
        "memmap_primary": True,
        "rows": manifest_rows,
    }
    manifest_path = os.path.join(
        MANIFEST_DIR, "new_incremental_full_feature_asset_manifest.json"
    )
    temporary = manifest_path + ".new_tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(temporary, manifest_path)

    print(f"\nManifest saved to: {manifest_path}")
    print(f"All 9/9 verified: {all_verified}")
    print(f"Total assets: {len(manifest_rows)}")
    total_train = sum(r["num_train"] for r in manifest_rows)
    print(f"Total train samples across all: {total_train}")

    if not all_verified:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

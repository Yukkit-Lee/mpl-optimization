#!/usr/bin/env python3
"""Audit seed42 feature/split assets for the isolated SecretFlow-SPU backend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "new_log/new_secretflow_spu"
JSON_PATH = OUT / "new_spu_feature_reuse_audit.json"
MD_PATH = OUT / "new_spu_feature_conversion_audit.md"
DATASETS = ("cifar10", "cifar100", "tiny")
CLASSES = {"cifar10": 10, "cifar100": 100, "tiny": 200}
FULL_COUNTS = {"cifar10": 50000, "cifar100": 50000, "tiny": 100000}
MTT_COUNTS = {"cifar10": 480, "cifar100": 4800, "tiny": 9600}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def npz_scalar(cache: Any, name: str) -> Any:
    return cache[name].reshape(-1)[0].item()


def require_finite_in_chunks(value: np.ndarray, chunk: int = 4096) -> None:
    for start in range(0, len(value), chunk):
        if not np.isfinite(value[start : start + chunk]).all():
            raise ValueError("non-finite feature value")


def mtt_anchor_ids(labels: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
    local_parts = []
    anchor_parts = []
    for class_id in range(classes):
        ids = np.flatnonzero(labels == class_id)
        shuffled = np.random.RandomState(42 * 1000003 + class_id * 1009 + 729).permutation(ids)
        anchor_parts.append(shuffled[:8])
        local_parts.append(shuffled[8:])
    return (
        np.sort(np.concatenate(local_parts).astype(np.int64)),
        np.sort(np.concatenate(anchor_parts).astype(np.int64)),
    )


def audit_dataset(dataset: str) -> dict[str, Any]:
    classes = CLASSES[dataset]
    split_path = ROOT / f"new_cache/new_true_ours/splits/{dataset}/new_true_ours_split_{dataset}_seed42.npz"
    full_b_path = ROOT / f"new_cache/new_true_ours/full_data_stage_b/{dataset}/new_full_stage_b_{dataset}_seed42.npz"
    full_c_path = ROOT / f"new_cache/new_true_ours/full_data_stage_c/{dataset}/new_full_stage_c_{dataset}_seed42.npz"
    ours_b_path = ROOT / f"new_cache/new_true_ours/official_mtt/stage_b/{dataset}/new_true_ours_stage_b_{dataset}_seed42.npz"
    ours_c_path = ROOT / f"new_cache/new_true_ours/official_mtt/stage_c/{dataset}/new_true_ours_stage_c_{dataset}_seed42.npz"

    with np.load(split_path, allow_pickle=False) as split:
        split_arrays = {
            name: np.asarray(split[name], dtype=np.int64)
            for name in (
                "stage_b_train",
                "stage_c_train",
                "reference_internal",
                "selection_internal",
                "development_validation",
            )
        }
        # Deliberately do not access the locked_holdout member.
    names = list(split_arrays)
    for left_id, left in enumerate(names):
        for right in names[left_id + 1 :]:
            if np.intersect1d(split_arrays[left], split_arrays[right]).size:
                raise ValueError(f"split overlap: {dataset} {left}/{right}")

    with np.load(full_b_path, allow_pickle=False) as full_b:
        feature_path = Path(bytes(full_b["feature_npy_path"][0]).decode("utf-8"))
        full_b_ids = np.asarray(full_b["train_indices"], dtype=np.int64)
        full_b_y = np.asarray(full_b["y_encoder_train"], dtype=np.int64)
        if npz_scalar(full_b, "test_loaded") != 0:
            raise ValueError("Full Stage-B cache reports test access")
        if npz_scalar(full_b, "initializer_source") != "full_data_disjoint":
            raise ValueError("Full Stage-B lineage mismatch")
    label_path = feature_path.with_name("new_full_train_labels.npy")
    x_full = np.load(feature_path, mmap_mode="r")
    y_full = np.load(label_path, mmap_mode="r")
    if x_full.shape != (FULL_COUNTS[dataset], 2048) or x_full.dtype != np.float32:
        raise ValueError(f"Full feature shape/dtype mismatch: {dataset}")
    if not np.array_equal(full_b_ids, split_arrays["stage_b_train"]):
        raise ValueError(f"Full Stage-B IDs mismatch: {dataset}")
    if not np.array_equal(full_b_y, y_full[full_b_ids]):
        raise ValueError(f"Full Stage-B labels mismatch: {dataset}")
    require_finite_in_chunks(x_full)

    with np.load(full_c_path, allow_pickle=False) as full_c:
        full_c_x = np.asarray(full_c["x_reference"], dtype=np.float32)
        full_c_y = np.asarray(full_c["y_reference"], dtype=np.int64)
        full_selection_x = np.asarray(full_c["x_selection"], dtype=np.float32)
        full_selection_y = np.asarray(full_c["y_selection"], dtype=np.int64)
        if not np.array_equal(np.asarray(full_c["train_indices"]), split_arrays["stage_c_train"]):
            raise ValueError(f"Full Stage-C IDs mismatch: {dataset}")
        if npz_scalar(full_c, "test_loaded") != 0:
            raise ValueError("Full Stage-C cache reports test access")
    # In Stage C, x/y_encoder_train are the protected training pool while
    # x/y_reference are the disjoint development-validation evaluation pool.
    if not np.array_equal(full_c_y, y_full[split_arrays["development_validation"]]):
        raise ValueError(f"Full Stage-C development labels mismatch: {dataset}")
    if not np.array_equal(full_c_x, x_full[split_arrays["development_validation"]]):
        raise ValueError(f"Full Stage-C development feature values mismatch: {dataset}")
    if not np.array_equal(full_selection_x, x_full[split_arrays["selection_internal"]]):
        raise ValueError(f"selection feature values mismatch: {dataset}")
    if not np.array_equal(full_selection_y, y_full[split_arrays["selection_internal"]]):
        raise ValueError(f"selection labels mismatch: {dataset}")

    with np.load(ours_b_path, allow_pickle=False) as ours_b:
        ours_b_x = np.asarray(ours_b["x_encoder_train"], dtype=np.float32)
        ours_b_y = np.asarray(ours_b["y_encoder_train"], dtype=np.int64)
        encoder_sha = str(npz_scalar(ours_b, "encoder_sha256"))
        scaler_sha = str(npz_scalar(ours_b, "scaler_sha256"))
        upstream_path = str(npz_scalar(ours_b, "upstream_asset_path"))
        gates = (
            npz_scalar(ours_b, "initializer_source") == "official_mtt_only"
            and npz_scalar(ours_b, "full_data_initializer_dependency") == 0
            and npz_scalar(ours_b, "test_loaded") == 0
        )
    if not gates or ours_b_x.shape != (MTT_COUNTS[dataset], 2048):
        raise ValueError(f"Ours Stage-B lineage/shape mismatch: {dataset}")
    histogram = np.bincount(ours_b_y, minlength=classes)
    if not np.all(histogram == 48) or ours_b_x.dtype != np.float32:
        raise ValueError(f"Official-MTT balance/dtype mismatch: {dataset}")
    require_finite_in_chunks(ours_b_x)

    with np.load(ours_c_path, allow_pickle=False) as ours_c:
        if not np.array_equal(ours_b_x, np.asarray(ours_c["x_encoder_train"], dtype=np.float32)):
            raise ValueError(f"Stage-B/C MTT values mismatch: {dataset}")
        if not np.array_equal(ours_b_y, np.asarray(ours_c["y_encoder_train"], dtype=np.int64)):
            raise ValueError(f"Stage-B/C MTT labels mismatch: {dataset}")
        if not np.array_equal(full_c_x, np.asarray(ours_c["x_reference"], dtype=np.float32)):
            raise ValueError(f"Full/Ours development features mismatch: {dataset}")
        if not np.array_equal(full_c_y, np.asarray(ours_c["y_reference"], dtype=np.int64)):
            raise ValueError(f"Full/Ours development labels mismatch: {dataset}")
        if npz_scalar(ours_c, "test_loaded") != 0:
            raise ValueError("Ours Stage-C cache reports test access")

    local_ids, anchor_ids = mtt_anchor_ids(ours_b_y, classes)
    if np.intersect1d(local_ids, anchor_ids).size:
        raise ValueError(f"local/anchor overlap: {dataset}")
    dev_ids = split_arrays["development_validation"]
    require_finite_in_chunks(x_full[dev_ids])
    target_freeze = ROOT / "new_configs/new_true_ours/new_true_ours_target_freeze.json"

    def file_row(path: Path, shape: list[int], dtype: str, count: int, role: str) -> dict[str, Any]:
        return {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "shape": shape,
            "dtype": dtype,
            "sample_count": count,
            "feature_dimension": 2048,
            "role": role,
        }

    return {
        "dataset": dataset,
        "seed": 42,
        "num_classes": classes,
        "split": {
            "path": str(split_path.relative_to(ROOT)),
            "sha256": sha256_file(split_path),
            "audited_members": {name: len(value) for name, value in split_arrays.items()},
            "audited_members_pairwise_disjoint": True,
            "locked_holdout_member_read": False,
            "development_ids_sha256": sha256_array(dev_ids),
        },
        "full": {
            "stage_b_cache": file_row(full_b_path, [len(full_b_ids), 2048], "float32", len(full_b_ids), "indexed Full Stage-B training"),
            "feature_store": file_row(feature_path, list(x_full.shape), str(x_full.dtype), len(x_full), "read-only memory-mapped full feature store"),
            "label_store_path": str(label_path.relative_to(ROOT)),
            "label_store_sha256": sha256_file(label_path),
            "stage_c_cache": file_row(full_c_path, [len(split_arrays["stage_c_train"]), 2048], "float32", len(split_arrays["stage_c_train"]), "indexed Full Stage-C protected training"),
            "initializer_source": "full_data_disjoint",
        },
        "ours": {
            "stage_b_cache": file_row(ours_b_path, list(ours_b_x.shape), str(ours_b_x.dtype), len(ours_b_x), "Official-MTT compact Stage-B"),
            "stage_c_cache": file_row(ours_c_path, list(ours_b_x.shape), str(ours_b_x.dtype), len(ours_b_x), "Official-MTT compact Stage-C protected training"),
            "upstream_asset_path": upstream_path,
            "encoder_sha256": encoder_sha,
            "scaler_sha256": scaler_sha,
            "class_histogram": histogram.astype(int).tolist(),
            "local_samples": len(local_ids),
            "anchor_samples": len(anchor_ids),
            "local_per_class": 40,
            "anchor_per_class": 8,
            "local_ids_sha256": sha256_array(local_ids),
            "anchor_ids_sha256": sha256_array(anchor_ids),
            "initializer_source": "official_mtt_only",
            "full_data_initializer_dependency": False,
        },
        "matched_development_values": True,
        "stage_c_training_sets_intentionally_differ": True,
        "development_shape": [len(dev_ids), 2048],
        "target_freeze_path": str(target_freeze.relative_to(ROOT)),
        "target_freeze_sha256": sha256_file(target_freeze),
        "jax_spu_host_ingest": "direct float32 ndarray/memmap chunks; no conversion required",
        "test_read": False,
        "locked_holdout_read": False,
    }


def main() -> None:
    rows = [audit_dataset(dataset) for dataset in DATASETS]
    payload = {
        "schema": "new_spu_feature_reuse_audit_v1",
        "status": "PASS",
        "backend_target": "SecretFlow-SPU",
        "seed_policy": [42],
        "feature_extraction_runs": 0,
        "stage_a_distillation_runs": 0,
        "conversion_performed": False,
        "conversion_reason": "All formal inputs are already float32 NumPy arrays or indexed float32 NumPy memmaps consumable by JAX/SPU.",
        "datasets": rows,
        "anchor_config_evidence": {
            "cifar10": "validated freeze: 40 steps, lr=0.1, mu=0.001",
            "cifar100": "validated freeze: 40 steps, lr=0.2, mu=0.001",
            "tiny": "not covered by prior Anchor CE ablation; preregister CIFAR-100 setting as the bounded pre-formal starting point",
        },
        "test_read": False,
        "locked_holdout_read": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# SecretFlow-SPU feature conversion audit",
        "",
        "Status: `PASS`",
        "",
        "No feature conversion was needed. Full uses the existing read-only float32",
        "2048-D NumPy memmap plus frozen row indices; Ours uses the existing float32",
        "Official-MTT NPZ arrays directly. Stage A and feature extraction runs are zero.",
        "",
        "| Dataset | Full Stage-B rows | Ours rows | Stage-C rows | Dev rows | Feature dim |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['full']['stage_b_cache']['sample_count']} | "
            f"{row['ours']['stage_b_cache']['sample_count']} | {row['full']['stage_c_cache']['sample_count']} | "
            f"{row['development_shape'][0]} | 2048 |"
        )
    lines.extend(
        [
            "",
            "All audited training/reference/selection/development members are pairwise",
            "disjoint. The locked-holdout member and official test were not read.",
            "",
            "CIFAR-10/100 Anchor CE parameters come from the completed positive freeze.",
            "Tiny-ImageNet was excluded from that ablation, so its identical method",
            "semantics use the CIFAR-100 parameters only as a registered smoke/tuning",
            "starting point; this is not mislabeled as prior Tiny validation.",
        ]
    )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "datasets": DATASETS}))


if __name__ == "__main__":
    main()

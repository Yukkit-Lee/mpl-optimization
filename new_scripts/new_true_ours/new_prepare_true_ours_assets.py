#!/usr/bin/env python3
"""Prepare disjoint Stage-B/Stage-C evaluation assets for True Ours.

The official MTT tensors are treated as a fixed public upstream asset.  All
original-training-set splits created here are mutually disjoint and are never
mixed back into the Stage-B or Stage-C training pools.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATASETS = ("cifar10", "cifar100", "tiny")
SEEDS = (42, 43, 44)
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "tiny": 200}
EXPECTED_FULL = {"cifar10": 50000, "cifar100": 50000, "tiny": 100000}
EXPECTED_MTT = {"cifar10": 480, "cifar100": 4800, "tiny": 9600}
OFFICIAL_TEST = {"cifar10": 10000, "cifar100": 10000, "tiny": 10000}

CACHE_ROOT = ROOT / "new_cache/new_true_ours"
LOG_ROOT = ROOT / "new_log/new_true_ours"
TABLE_ROOT = ROOT / "new_expTab/new_true_ours"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def class_histogram(labels, num_classes):
    values = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    return {str(i): int(values[i]) for i in range(num_classes)}


def build_splits(labels, dataset, seed):
    labels = np.asarray(labels, dtype=np.int64)
    num_classes = NUM_CLASSES[dataset]
    buckets = {
        "stage_b_train": [],
        "stage_c_train": [],
        "reference_internal": [],
        "selection_internal": [],
        "development_validation": [],
        "locked_holdout": [],
    }
    for class_id in range(num_classes):
        ids = np.flatnonzero(labels == class_id)
        if len(ids) == 0:
            raise ValueError("{} seed{} missing class {}".format(dataset, seed, class_id))
        rng = np.random.RandomState(seed * 1000003 + class_id * 1009 + 280728)
        ids = rng.permutation(ids)
        n_dev = int(round(0.10 * len(ids)))
        n_hold = int(round(0.10 * len(ids)))
        n_stage_c = int(round(0.10 * len(ids)))
        n_reference = 5
        n_selection = 5
        cursor = 0
        allocations = (
            ("development_validation", n_dev),
            ("locked_holdout", n_hold),
            ("reference_internal", n_reference),
            ("selection_internal", n_selection),
            ("stage_c_train", n_stage_c),
        )
        for name, count in allocations:
            buckets[name].append(ids[cursor : cursor + count])
            cursor += count
        buckets["stage_b_train"].append(ids[cursor:])

    result = {}
    for name, parts in buckets.items():
        value = np.sort(np.concatenate(parts).astype(np.int64))
        result[name] = value
    return result


def assert_disjoint_and_complete(splits, total):
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = np.intersect1d(
                splits[left], splits[right], assume_unique=True
            )
            if len(overlap):
                raise ValueError("{} overlaps {} by {} IDs".format(
                    left, right, len(overlap)
                ))
    combined = np.concatenate([splits[name] for name in names])
    if len(combined) != total or len(np.unique(combined)) != total:
        raise ValueError("split union does not cover exactly {} rows".format(total))


def load_official_mtt(dataset, seed):
    path = (
        ROOT
        / "new_cache/new_public_mtt_mainline/new_test_free_features"
        / dataset
        / "new_reference_feature_cache_{}_seed{}.pt".format(dataset, seed)
    )
    source = torch.load(str(path), map_location="cpu", weights_only=False)
    if bool(source.get("test_loaded", False)):
        raise ValueError("Official MTT asset is not test-free: {}".format(path))
    payload = {
        "encoder_sha256": np.asarray([source["encoder_sha256"]]),
        "scaler_sha256": np.asarray([source["scaler_sha256"]]),
    }
    x_all = source["x_encoder_train"].detach().cpu().numpy().astype(np.float32)
    y_all = source["y_encoder_train"].detach().cpu().numpy().astype(np.int64)
    return path, payload, x_all, y_all


def write_memmap_metadata(path, feature_path, labels, indices, reference, selection):
    atomic_npz(
        path,
        feature_npy_path=np.asarray([str(feature_path).encode("utf-8")]),
        train_indices=indices.astype(np.int64),
        y_encoder_train=np.asarray(labels[indices], dtype=np.int64),
        x_reference=np.asarray(reference[0], dtype=np.float32),
        y_reference=np.asarray(reference[1], dtype=np.int64),
        x_selection=np.asarray(selection[0], dtype=np.float32),
        y_selection=np.asarray(selection[1], dtype=np.int64),
        feature_dim=np.asarray([2048], dtype=np.int64),
        test_loaded=np.asarray([0], dtype=np.int64),
    )


def main():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    split_rows = []
    mtt_rows = []
    split_npz_paths = []

    for dataset in DATASETS:
        for seed in SEEDS:
            full_dir = (
                ROOT
                / "new_cache/new_public_mtt_mainline/new_full_features"
                / dataset
                / "new_seed{}".format(seed)
            )
            feature_path = full_dir / "new_full_train_features.npy"
            label_path = full_dir / "new_full_train_labels.npy"
            manifest_path = full_dir / "new_full_feature_manifest.json"
            x_full = np.load(str(feature_path), mmap_mode="r")
            y_full = np.load(str(label_path), mmap_mode="r")
            full_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if len(y_full) != EXPECTED_FULL[dataset]:
                raise ValueError("{} full count mismatch".format(dataset))

            splits = build_splits(y_full, dataset, seed)
            assert_disjoint_and_complete(splits, len(y_full))
            split_path = (
                CACHE_ROOT / "splits" / dataset
                / "new_true_ours_split_{}_seed{}.npz".format(dataset, seed)
            )
            atomic_npz(split_path, **splits)
            split_npz_paths.append(split_path)

            ref_ids = splits["reference_internal"]
            sel_ids = splits["selection_internal"]
            dev_ids = splits["development_validation"]
            ref = (x_full[ref_ids], y_full[ref_ids])
            sel = (x_full[sel_ids], y_full[sel_ids])
            dev = (x_full[dev_ids], y_full[dev_ids])

            full_b_path = (
                CACHE_ROOT / "full_data_stage_b" / dataset
                / "new_full_stage_b_{}_seed{}.npz".format(dataset, seed)
            )
            write_memmap_metadata(
                full_b_path,
                feature_path,
                y_full,
                splits["stage_b_train"],
                ref,
                sel,
            )
            full_c_path = (
                CACHE_ROOT / "full_data_stage_c" / dataset
                / "new_full_stage_c_{}_seed{}.npz".format(dataset, seed)
            )
            write_memmap_metadata(
                full_c_path,
                feature_path,
                y_full,
                splits["stage_c_train"],
                dev,
                sel,
            )

            official_path, official, mtt_x, mtt_y = load_official_mtt(dataset, seed)
            if len(mtt_y) != EXPECTED_MTT[dataset]:
                raise ValueError(
                    "{} seed{} MTT count {} != {}".format(
                        dataset, seed, len(mtt_y), EXPECTED_MTT[dataset]
                    )
                )
            num_classes = NUM_CLASSES[dataset]
            histogram = np.bincount(mtt_y, minlength=num_classes)
            if len(np.unique(histogram)) != 1:
                raise ValueError("{} seed{} MTT is not class balanced".format(
                    dataset, seed
                ))

            common = {
                "x_encoder_train": mtt_x,
                "y_encoder_train": mtt_y,
                "feature_dim": np.asarray([2048], dtype=np.int64),
                "encoder_sha256": official["encoder_sha256"],
                "scaler_sha256": official["scaler_sha256"],
                "test_loaded": np.asarray([0], dtype=np.int64),
                "initializer_source": np.asarray(["official_mtt_only"]),
                "full_data_initializer_dependency": np.asarray([0], dtype=np.int64),
                "upstream_asset_path": np.asarray([str(official_path)]),
                "split_manifest_path": np.asarray([str(split_path)]),
                "split_manifest_sha256": np.asarray([sha256_file(split_path)]),
            }
            true_b_path = (
                CACHE_ROOT / "official_mtt" / "stage_b" / dataset
                / "new_true_ours_stage_b_{}_seed{}.npz".format(dataset, seed)
            )
            atomic_npz(
                true_b_path,
                x_reference=np.asarray(ref[0], dtype=np.float32),
                y_reference=np.asarray(ref[1], dtype=np.int64),
                x_selection=np.asarray(sel[0], dtype=np.float32),
                y_selection=np.asarray(sel[1], dtype=np.int64),
                **common
            )
            true_c_path = (
                CACHE_ROOT / "official_mtt" / "stage_c" / dataset
                / "new_true_ours_stage_c_{}_seed{}.npz".format(dataset, seed)
            )
            atomic_npz(
                true_c_path,
                x_reference=np.asarray(dev[0], dtype=np.float32),
                y_reference=np.asarray(dev[1], dtype=np.int64),
                x_selection=np.asarray(sel[0], dtype=np.float32),
                y_selection=np.asarray(sel[1], dtype=np.int64),
                **common
            )

            split_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "split_npz": str(split_path),
                    "split_npz_sha256": sha256_file(split_path),
                    "full_feature_path": str(feature_path),
                    "full_feature_sha256": full_manifest["feature_sha256"],
                    "full_label_path": str(label_path),
                    "encoder_sha256": full_manifest["encoder_sha256"],
                    "scaler_sha256": full_manifest["scaler_sha256"],
                    "splits": {
                        name: {
                            "count": int(len(ids)),
                            "indices_sha256": sha256_array(ids),
                            "class_histogram": class_histogram(
                                y_full[ids], num_classes
                            ),
                        }
                        for name, ids in splits.items()
                    },
                    "pairwise_disjoint": True,
                    "union_covers_full_train": True,
                    "official_test_count": OFFICIAL_TEST[dataset],
                    "official_test_previously_read_for_historical_experiments": True,
                    "full_stage_b_cache": str(full_b_path),
                    "full_stage_c_cache": str(full_c_path),
                    "true_ours_stage_b_cache": str(true_b_path),
                    "true_ours_stage_c_cache": str(true_c_path),
                }
            )
            mtt_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "source_path": str(official_path),
                    "source_sha256": sha256_file(official_path),
                    "true_ours_stage_b_cache": str(true_b_path),
                    "true_ours_stage_b_sha256": sha256_file(true_b_path),
                    "true_ours_stage_c_cache": str(true_c_path),
                    "true_ours_stage_c_sha256": sha256_file(true_c_path),
                    "samples": int(len(mtt_y)),
                    "compression_ratio": float(len(mtt_y))
                    / float(EXPECTED_FULL[dataset]),
                    "samples_per_class": int(histogram[0]),
                    "class_balanced": True,
                    "feature_dim": int(mtt_x.shape[1]),
                    "finite": bool(np.isfinite(mtt_x).all()),
                    "encoder_sha256": str(
                        official["encoder_sha256"].reshape(-1)[0]
                    ),
                    "scaler_sha256": str(
                        official["scaler_sha256"].reshape(-1)[0]
                    ),
                    "initializer_source": "official_mtt_only",
                    "full_data_initializer_dependency": False,
                    "stage_a_upstream_scope": (
                        "fixed public MTT asset created before this Stage-B/C split"
                    ),
                }
            )

    split_manifest = {
        "schema": "new_true_ours_split_manifest_v1",
        "status": "FROZEN_PRE_TUNING",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "stratified": True,
        "pairwise_disjoint_original_train_splits": True,
        "development_validation_used_for_tuning": True,
        "locked_holdout_read": False,
        "official_test_read_in_this_rebuild": False,
        "split_rule": {
            "development_validation_per_class": "10%",
            "locked_holdout_per_class": "10%",
            "reference_internal_per_class": 5,
            "selection_internal_per_class": 5,
            "stage_c_train_per_class": "10%",
            "stage_b_train_per_class": "remainder",
        },
        "rows": split_rows,
    }
    split_manifest_path = LOG_ROOT / "new_true_ours_split_manifest.json"
    atomic_json(split_manifest_path, split_manifest)

    audit_lines = [
        "# True Ours disjoint split audit",
        "",
        "- Status: FROZEN_PRE_TUNING",
        "- All original-training-set splits are stratified, pairwise disjoint, and exhaustive.",
        "- Locked holdout was not read for metrics.",
        "- Official test was not read by this rebuild.",
        "- Official MTT is a fixed public upstream Stage-A asset; disjointness claims apply from Stage B onward.",
        "",
        "| Dataset | Seed | Stage B | Stage C | Ref | Sel | Dev | Locked | Disjoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in split_rows:
        counts = row["splits"]
        audit_lines.append(
            "| {dataset} | {seed} | {stage_b_train} | {stage_c_train} | "
            "{reference_internal} | {selection_internal} | "
            "{development_validation} | {locked_holdout} | YES |".format(
                dataset=row["dataset"],
                seed=row["seed"],
                **{name: item["count"] for name, item in counts.items()}
            )
        )
    (LOG_ROOT / "new_true_ours_split_audit.md").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )

    mtt_audit = {
        "schema": "new_mtt_asset_audit_v1",
        "status": "COMPLETE",
        "expected_samples": EXPECTED_MTT,
        "all_class_balanced": all(row["class_balanced"] for row in mtt_rows),
        "all_finite": all(row["finite"] for row in mtt_rows),
        "all_official_mtt_only": all(
            not row["full_data_initializer_dependency"] for row in mtt_rows
        ),
        "rows": mtt_rows,
    }
    atomic_json(LOG_ROOT / "new_mtt_asset_audit.json", mtt_audit)
    csv_path = TABLE_ROOT / "new_mtt_class_distribution.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "dataset",
                "seed",
                "samples",
                "samples_per_class",
                "compression_ratio",
                "class_balanced",
                "feature_dim",
                "finite",
            ]
        )
        for row in mtt_rows:
            writer.writerow(
                [
                    row["dataset"],
                    row["seed"],
                    row["samples"],
                    row["samples_per_class"],
                    "{:.6f}".format(row["compression_ratio"]),
                    row["class_balanced"],
                    row["feature_dim"],
                    row["finite"],
                ]
            )

    print("split_manifest={}".format(split_manifest_path))
    print("split_assets={}".format(len(split_npz_paths)))
    print("mtt_assets={} status=COMPLETE locked_holdout_read=False".format(
        len(mtt_rows)
    ))


if __name__ == "__main__":
    main()

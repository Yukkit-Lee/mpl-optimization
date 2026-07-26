#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build equal-size Random/Official MTT caches with one shared test-free reference."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from new_public_mtt_mainline_common import ROOT, load_config


OUTPUT_ROOT = (
    ROOT / "new_cache/new_public_mtt_mainline/new_end_to_end_features"
)
MANIFEST = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_end_to_end_feature_manifest.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".new_tmp_{}".format(os.getpid()))
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def stratified_parts(x_full, y_full, official, seed):
    train_x, train_y = [], []
    reference_x, reference_y = [], []
    selection_x, selection_y = [], []
    classes = np.unique(official["y_encoder_train"])
    for class_id in classes:
        train_count = int(np.sum(official["y_encoder_train"] == class_id))
        reference_count = int(np.sum(official["y_reference"] == class_id))
        selection_count = int(np.sum(official["y_selection"] == class_id))
        needed = train_count + reference_count + selection_count
        candidates = np.flatnonzero(y_full == class_id)
        if needed > len(candidates):
            raise ValueError("insufficient full features for class {}".format(class_id))
        rng = np.random.RandomState(int(seed) + int(class_id) * 1009 + 88001)
        chosen = candidates[rng.permutation(len(candidates))[:needed]]
        train_ids = chosen[:train_count]
        reference_ids = chosen[train_count : train_count + reference_count]
        selection_ids = chosen[train_count + reference_count :]
        train_x.append(np.asarray(x_full[train_ids], dtype=np.float32))
        train_y.append(np.asarray(y_full[train_ids], dtype=np.int64))
        reference_x.append(np.asarray(x_full[reference_ids], dtype=np.float32))
        reference_y.append(np.asarray(y_full[reference_ids], dtype=np.int64))
        selection_x.append(np.asarray(x_full[selection_ids], dtype=np.float32))
        selection_y.append(np.asarray(y_full[selection_ids], dtype=np.int64))
    return {
        "x_encoder_train": np.concatenate(train_x, axis=0),
        "y_encoder_train": np.concatenate(train_y, axis=0),
        "x_reference": np.concatenate(reference_x, axis=0),
        "y_reference": np.concatenate(reference_y, axis=0),
        "x_selection": np.concatenate(selection_x, axis=0),
        "y_selection": np.concatenate(selection_y, axis=0),
    }


def main():
    config = load_config()
    rows = []
    for dataset in config["datasets"]:
        for seed in config["formal_seeds"]:
            official_path = (
                ROOT
                / "new_cache/new_public_mtt_mainline/new_stage_b_npz_round2"
                / dataset
                / "new_stage_b_features_{}_seed{}.npz".format(dataset, seed)
            )
            full_dir = (
                ROOT
                / "new_cache/new_public_mtt_mainline/new_full_features"
                / dataset
                / "new_seed{}".format(seed)
            )
            full_manifest = json.loads(
                (full_dir / "new_full_feature_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            x_full = np.load(
                full_dir / "new_full_train_features.npy", mmap_mode="r"
            )
            y_full = np.load(
                full_dir / "new_full_train_labels.npy", mmap_mode="r"
            )
            with np.load(str(official_path), allow_pickle=False) as source:
                official = {name: np.asarray(source[name]) for name in source.files}
            if bool(official["test_loaded"].reshape(-1)[0]):
                raise ValueError("{} is not test-free".format(official_path))
            random_parts = stratified_parts(x_full, y_full, official, seed)
            common = {
                "feature_dim": np.asarray([2048], dtype=np.int64),
                "encoder_sha256": np.asarray([full_manifest["encoder_sha256"]]),
                "scaler_sha256": np.asarray([full_manifest["scaler_sha256"]]),
                "test_loaded": np.asarray([0], dtype=np.int64),
            }
            random_path = (
                OUTPUT_ROOT
                / "random_subset"
                / dataset
                / "new_end_to_end_random_{}_seed{}.npz".format(dataset, seed)
            )
            official_eval_path = (
                OUTPUT_ROOT
                / "official_mtt"
                / dataset
                / "new_end_to_end_official_{}_seed{}.npz".format(dataset, seed)
            )
            atomic_npz(random_path, **dict(random_parts, **common))
            official_eval = {
                "x_encoder_train": official["x_encoder_train"].astype(np.float32),
                "y_encoder_train": official["y_encoder_train"].astype(np.int64),
                "x_reference": random_parts["x_reference"],
                "y_reference": random_parts["y_reference"],
                "x_selection": random_parts["x_selection"],
                "y_selection": random_parts["y_selection"],
            }
            atomic_npz(official_eval_path, **dict(official_eval, **common))
            if len(random_parts["y_encoder_train"]) != len(
                official["y_encoder_train"]
            ):
                raise ValueError("Random/MTT train size mismatch")
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "random_subset_cache": str(random_path),
                    "random_subset_sha256": sha256_file(random_path),
                    "official_mtt_cache": str(official_eval_path),
                    "official_mtt_sha256": sha256_file(official_eval_path),
                    "num_train_each": int(len(random_parts["y_encoder_train"])),
                    "num_shared_reference": int(
                        len(random_parts["y_reference"])
                    ),
                    "num_shared_selection": int(
                        len(random_parts["y_selection"])
                    ),
                    "feature_dim": 2048,
                    "encoder_sha256": full_manifest["encoder_sha256"],
                    "scaler_sha256": full_manifest["scaler_sha256"],
                    "reference_equal": True,
                    "selection_equal": True,
                    "test_loaded": False,
                }
            )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_name(MANIFEST.name + ".new_tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema": "new_end_to_end_feature_manifest_v1",
                "status": "COMPLETE",
                "test_loaded": False,
                "dp_mtt_status": "EXCLUDED_BY_USER",
                "random_and_mtt_sample_counts_equal": True,
                "shared_reference_protocol": True,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(MANIFEST))
    print(
        "end_to_end_features={} test_loaded=False manifest={}".format(
            len(rows), MANIFEST
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

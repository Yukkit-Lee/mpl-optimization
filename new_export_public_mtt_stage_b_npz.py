#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export physically test-free Public-MTT features for the Python-3.6 TFE runtime."""

import argparse
import json

import numpy as np

from new_public_mtt_mainline_common import ROOT, load_config, load_feature_cache
from new_secure_feature_models import configure_single_thread_runtime


OUTPUT_ROOT = ROOT / "new_cache/new_public_mtt_mainline"


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new_tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new_tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    temporary.replace(path)


def expand_holdouts(cache, target_per_class, seed):
    x_train = cache["x_encoder_train"].numpy()
    y_train = cache["y_encoder_train"].numpy()
    x_reference = [cache["x_reference"].numpy()]
    y_reference = [cache["y_reference"].numpy()]
    x_selection = [cache["x_selection"].numpy()]
    y_selection = [cache["y_selection"].numpy()]
    keep = np.ones(len(y_train), dtype=bool)
    classes = np.unique(y_train)
    existing_reference = len(y_reference[0]) // len(classes)
    existing_selection = len(y_selection[0]) // len(classes)
    for class_id in classes:
        class_indices = np.flatnonzero(y_train == class_id)
        rng = np.random.RandomState(seed + int(class_id) * 1009)
        rng.shuffle(class_indices)
        reference_extra = max(0, target_per_class - existing_reference)
        selection_extra = max(0, target_per_class - existing_selection)
        required = reference_extra + selection_extra
        if required >= len(class_indices):
            raise RuntimeError("holdout request leaves no local samples")
        reference_ids = class_indices[:reference_extra]
        selection_ids = class_indices[reference_extra:required]
        keep[reference_ids] = False
        keep[selection_ids] = False
        x_reference.append(x_train[reference_ids])
        y_reference.append(y_train[reference_ids])
        x_selection.append(x_train[selection_ids])
        y_selection.append(y_train[selection_ids])
    return {
        "x_encoder_train": x_train[keep],
        "y_encoder_train": y_train[keep],
        "x_reference": np.concatenate(x_reference, axis=0),
        "y_reference": np.concatenate(y_reference, axis=0),
        "x_selection": np.concatenate(x_selection, axis=0),
        "y_selection": np.concatenate(y_selection, axis=0),
    }


def export_one(dataset, seed, output_tag, reference_per_class):
    cache = load_feature_cache(dataset, seed, allow_test=False)
    arrays = expand_holdouts(cache, reference_per_class, seed)
    output = (
        OUTPUT_ROOT
        / output_tag
        / dataset
        / "new_stage_b_features_{}_seed{}.npz".format(dataset, seed)
    )
    atomic_npz(
        output,
        x_encoder_train=arrays["x_encoder_train"].astype(np.float32),
        y_encoder_train=arrays["y_encoder_train"].astype(np.int64),
        x_reference=arrays["x_reference"].astype(np.float32),
        y_reference=arrays["y_reference"].astype(np.int64),
        x_selection=arrays["x_selection"].astype(np.float32),
        y_selection=arrays["y_selection"].astype(np.int64),
        feature_dim=np.asarray([2048], dtype=np.int64),
        encoder_sha256=np.asarray([cache["encoder_sha256"]]),
        scaler_sha256=np.asarray([cache["scaler_sha256"]]),
        test_loaded=np.asarray([0], dtype=np.int64),
    )
    return {
        "dataset": dataset,
        "seed": int(seed),
        "path": str(output),
        "num_train": int(len(arrays["y_encoder_train"])),
        "num_reference": int(len(arrays["y_reference"])),
        "num_selection": int(len(arrays["y_selection"])),
        "reference_per_class": int(reference_per_class),
        "feature_dim": 2048,
        "encoder_sha256": cache["encoder_sha256"],
        "scaler_sha256": cache["scaler_sha256"],
        "test_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export test-free Public-MTT feature caches to NPZ"
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "cifar10", "cifar100", "tiny"),
        default="all",
    )
    parser.add_argument("--output_tag", default="new_stage_b_npz_round2")
    parser.add_argument("--reference_per_class", type=int, default=5)
    args = parser.parse_args()
    configure_single_thread_runtime(42)
    config = load_config()
    datasets = config["datasets"] if args.dataset == "all" else [args.dataset]
    rows = [
        export_one(
            dataset,
            seed,
            args.output_tag,
            args.reference_per_class,
        )
        for dataset in datasets
        for seed in config["formal_seeds"]
    ]
    manifest = {
        "schema": "new_public_mtt_stage_b_npz_manifest_v1",
        "status": "COMPLETE",
        "test_loaded": False,
        "cpu_threads": 1,
        "num_workers": 0,
        "rows": rows,
    }
    manifest_path = (
        OUTPUT_ROOT / args.output_tag / "new_stage_b_npz_manifest.json"
    )
    atomic_json(manifest, manifest_path)
    print("exported={} manifest={}".format(len(rows), manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

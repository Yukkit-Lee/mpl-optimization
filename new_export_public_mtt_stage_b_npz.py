#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export physically test-free Public-MTT features for the Python-3.6 TFE runtime."""

import argparse
import json

import numpy as np

from new_public_mtt_mainline_common import ROOT, load_config, load_feature_cache
from new_secure_feature_models import configure_single_thread_runtime


OUTPUT_ROOT = ROOT / "new_cache/new_public_mtt_mainline/new_stage_b_npz"


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


def export_one(dataset, seed):
    cache = load_feature_cache(dataset, seed, allow_test=False)
    output = (
        OUTPUT_ROOT
        / dataset
        / "new_stage_b_features_{}_seed{}.npz".format(dataset, seed)
    )
    atomic_npz(
        output,
        x_encoder_train=cache["x_encoder_train"].numpy().astype(np.float32),
        y_encoder_train=cache["y_encoder_train"].numpy().astype(np.int64),
        x_reference=cache["x_reference"].numpy().astype(np.float32),
        y_reference=cache["y_reference"].numpy().astype(np.int64),
        x_selection=cache["x_selection"].numpy().astype(np.float32),
        y_selection=cache["y_selection"].numpy().astype(np.int64),
        feature_dim=np.asarray([2048], dtype=np.int64),
        encoder_sha256=np.asarray([cache["encoder_sha256"]]),
        scaler_sha256=np.asarray([cache["scaler_sha256"]]),
        test_loaded=np.asarray([0], dtype=np.int64),
    )
    return {
        "dataset": dataset,
        "seed": int(seed),
        "path": str(output),
        "num_train": int(len(cache["y_encoder_train"])),
        "num_reference": int(len(cache["y_reference"])),
        "num_selection": int(len(cache["y_selection"])),
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
    args = parser.parse_args()
    configure_single_thread_runtime(42)
    config = load_config()
    datasets = config["datasets"] if args.dataset == "all" else [args.dataset]
    rows = [
        export_one(dataset, seed)
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
    manifest_path = OUTPUT_ROOT / "new_stage_b_npz_manifest.json"
    atomic_json(manifest, manifest_path)
    print("exported={} manifest={}".format(len(rows), manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

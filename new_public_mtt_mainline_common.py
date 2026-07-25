#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared paths and validation for the Public MTT formal mainline."""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict

import torch

from new_secure_feature_models import configure_single_thread_runtime


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
CONFIG_PATH = (
    ROOT / "new_configs/new_public_mtt_mainline/new_mainline_config.json"
)


def load_config() -> Dict:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload["dp_mtt_status"] != "DEFERRED_UNTIL_MAIN_PIPELINE_COMPLETE":
        raise RuntimeError("DP-MTT must not block the Public MTT mainline")
    if int(payload["cpu_threads_per_process"]) != 1:
        raise RuntimeError("formal CPU thread count must be one")
    if int(payload["num_workers"]) != 0:
        raise RuntimeError("formal DataLoader workers must be zero")
    return payload


def feature_cache_path(dataset: str, seed: int) -> Path:
    if dataset == "cifar10":
        base = (
            ROOT
            / "new_cache/new_secure_feature/new_formal_phase6"
            / "new_public_mtt_round3"
            / ("new_seed{}".format(seed))
        )
    else:
        base = (
            ROOT
            / "new_cache/new_secure_feature/new_formal_phase6"
            / "new_public_mtt_final"
            / ("new_{}_seed{}".format(dataset, seed))
        )
    return (
        base
        / "public_mtt_benchmark"
        / dataset
        / "new_feature_cache_{}_public_mtt_benchmark_seed{}.pt".format(
            dataset, seed
        )
    )


def reference_cache_path(dataset: str, seed: int) -> Path:
    return (
        ROOT
        / "new_cache/new_public_mtt_mainline/new_test_free_features"
        / dataset
        / "new_reference_feature_cache_{}_seed{}.pt".format(dataset, seed)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_cache(dataset: str, seed: int, allow_test: bool) -> Dict:
    configure_single_thread_runtime(seed)
    path = (
        feature_cache_path(dataset, seed)
        if allow_test
        else reference_cache_path(dataset, seed)
    )
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    required = {
        "x_encoder_train",
        "y_encoder_train",
        "x_reference",
        "y_reference",
        "x_selection",
        "y_selection",
        "feature_dim",
        "encoder_sha256",
        "scaler_sha256",
    }
    missing = required - set(payload)
    if missing:
        raise RuntimeError("feature cache missing keys: {}".format(sorted(missing)))
    if not allow_test and (
        "x_test" in payload or "y_test" in payload or payload.get("test_loaded")
    ):
        raise RuntimeError("test-free tuning cache contains test data")
    if int(payload["feature_dim"]) != 2048:
        raise RuntimeError("feature dimension is not 2048")
    for key in ("x_encoder_train", "x_reference", "x_selection"):
        maximum = float(payload[key].norm(p=2, dim=1).max())
        if maximum > 1.00001:
            raise RuntimeError("{} violates L2 bound: {}".format(key, maximum))
    result = {
        key: payload[key]
        for key in (
            "x_encoder_train",
            "y_encoder_train",
            "x_reference",
            "y_reference",
            "x_selection",
            "y_selection",
        )
    }
    result.update(
        {
            "dataset": dataset,
            "seed": int(seed),
            "feature_dim": 2048,
            "encoder_sha256": payload["encoder_sha256"],
            "scaler_sha256": payload["scaler_sha256"],
            "feature_cache_path": str(path),
            "feature_cache_sha256": sha256_file(path),
            "stage_a_track": "public_mtt_benchmark",
            "stage_a_epsilon": "N/A",
            "stage_a_dp_protected": False,
        }
    )
    if allow_test:
        if payload.get("x_test") is None or payload.get("y_test") is None:
            raise RuntimeError("formal test tensors are absent")
        result["x_test"] = payload["x_test"]
        result["y_test"] = payload["y_test"]
        result["test_loaded"] = True
    else:
        result["test_loaded"] = False
    return result

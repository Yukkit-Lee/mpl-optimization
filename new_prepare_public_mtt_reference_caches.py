#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create immutable test-free feature caches before any mainline tuning."""

import json
from pathlib import Path

import torch

from new_public_mtt_mainline_common import (
    ROOT,
    feature_cache_path,
    load_config,
    reference_cache_path,
    sha256_file,
)
from new_secure_feature_models import atomic_torch_save, configure_single_thread_runtime


OUTPUT = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_test_free_feature_cache_manifest.json"
)


def main() -> int:
    config = load_config()
    rows = []
    for dataset in config["datasets"]:
        for seed in config["formal_seeds"]:
            configure_single_thread_runtime(seed)
            source = feature_cache_path(dataset, seed)
            payload = torch.load(
                str(source), map_location="cpu", weights_only=False
            )
            output = reference_cache_path(dataset, seed)
            test_free = {
                key: payload[key]
                for key in (
                    "dataset",
                    "feature_dim",
                    "encoder_sha256",
                    "scaler_sha256",
                    "input_norm_bound",
                    "x_encoder_train",
                    "y_encoder_train",
                    "x_reference",
                    "y_reference",
                    "x_selection",
                    "y_selection",
                )
            }
            test_free.update(
                {
                    "schema": "new_test_free_2048_feature_cache_v1",
                    "stage_a_track": "public_mtt_benchmark",
                    "stage_a_epsilon": "N/A",
                    "stage_a_dp_protected": False,
                    "test_loaded": False,
                    "source_feature_cache_sha256": sha256_file(source),
                    "purpose": "reference_only_tuning_no_test_payload",
                }
            )
            atomic_torch_save(test_free, output)
            verification = torch.load(
                str(output), map_location="cpu", weights_only=False
            )
            if "x_test" in verification or "y_test" in verification:
                raise RuntimeError("test tensor leaked into {}".format(output))
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "source": str(source),
                    "source_sha256": sha256_file(source),
                    "test_free_cache": str(output),
                    "test_free_sha256": sha256_file(output),
                    "keys": sorted(verification),
                    "test_loaded": False,
                }
            )
            print(
                "test-free cache: {} seed={} samples={}".format(
                    dataset, seed, len(verification["y_encoder_train"])
                ),
                flush=True,
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + ".new_tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema": "new_test_free_feature_cache_manifest_v1",
                "status": "COMPLETE",
                "test_tensors_present": False,
                "entries": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)
    print("manifest={}".format(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

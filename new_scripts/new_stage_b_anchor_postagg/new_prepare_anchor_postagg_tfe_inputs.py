#!/usr/bin/env python3
"""Prepare frozen Severe seed42 C1/C4/C5 inputs for TFE/ABY3."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KD_DIR = ROOT / "new_scripts/new_stage_b_kd"
for path in (HERE, KD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from new_run_stage_b_kd_plaintext import (  # noqa: E402
    build_or_load_local_heads,
)
from new_stage_b_kd_core import (  # noqa: E402
    DATASETS,
    SEED,
    atomic_json,
    atomic_npz,
    average_head,
    initialize_shared_head,
    load_dataset_assets,
    sha256_array,
    sha256_file,
)
from new_stage_b_anchor_postagg_core import (  # noqa: E402
    classification_metrics,
    train_anchor_ce_trajectory,
)


LOG_ROOT = ROOT / "new_log/new_stage_b_anchor_postagg"
OUTPUT_ROOT = ROOT / "new_output/new_stage_b_anchor_postagg"
FREEZE_PATH = LOG_ROOT / "new_stage_b_anchor_postagg_config_freeze.json"
MANIFEST_PATH = LOG_ROOT / "new_stage_b_anchor_postagg_tfe_input_manifest.json"


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def quantize(value: np.ndarray, fractional: int = 16) -> np.ndarray:
    scale = float(2**int(fractional))
    return (
        np.round(np.asarray(value, dtype=np.float32) * scale) / scale
    ).astype(np.float32)


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if not freeze["frozen"] or freeze["status"] != "FROZEN_AFTER_SMOKE_SUCCESS":
        raise ValueError("TFE input preparation requires a successful freeze")
    if freeze["formal_distributions"] != ["severe"]:
        raise ValueError("formal scope drifted from frozen Severe-only protocol")
    source = source_commit()
    freeze_sha = sha256_file(FREEZE_PATH)
    rows = []
    for dataset in DATASETS:
        assets = load_dataset_assets(dataset, SEED)
        local = freeze["local_configs"][dataset]
        anchor = freeze["selected_anchor_configs"][dataset]
        bundle = build_or_load_local_heads(
            assets,
            "severe",
            freeze["anchor_per_class"],
            local["epochs"],
            local["learning_rate"],
            local["weight_decay"],
        )
        random_weight, random_bias = initialize_shared_head(
            assets.x_mtt.shape[1], assets.num_classes, SEED
        )
        average_weight, average_bias = average_head(
            bundle["weights"], bundle["biases"]
        )
        c1_weight, c1_bias, _ = train_anchor_ce_trajectory(
            bundle["x_anchor"],
            bundle["y_anchor"],
            assets.x_dev,
            assets.y_dev,
            random_weight,
            random_bias,
            anchor["steps"],
            anchor["learning_rate"],
            anchor["mu"],
        )
        c5_weight, c5_bias, _ = train_anchor_ce_trajectory(
            bundle["x_anchor"],
            bundle["y_anchor"],
            assets.x_dev,
            assets.y_dev,
            average_weight,
            average_bias,
            anchor["steps"],
            anchor["learning_rate"],
            anchor["mu"],
        )
        configuration = {
            "schema": "new_stage_b_anchor_postagg_tfe_input_config_v1",
            "dataset": dataset,
            "seed": SEED,
            "distribution": "severe",
            "num_clients": 5,
            "feature_dim": int(assets.x_mtt.shape[1]),
            "num_classes": assets.num_classes,
            "local_epochs": local["epochs"],
            "local_lr": local["learning_rate"],
            "weight_decay": local["weight_decay"],
            "anchor_steps": anchor["steps"],
            "anchor_lr": anchor["learning_rate"],
            "anchor_mu": anchor["mu"],
            "anchor_samples_per_class": freeze["anchor_per_class"],
            "fixedpoint_fractional": freeze["tfe_fixed_point_precision"],
            "matmul_chunk_size": freeze["tfe_matmul_chunk_size"],
            "checkpoint_interval": 5 if dataset == "cifar10" else 10,
            "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
            "freeze_sha256": freeze_sha,
            "freeze_experiment_source": freeze[
                "experiment_source_git_commit"
            ],
            "input_builder_source": source,
            "initializer_source": "official_mtt_only",
            "full_data_initializer_dependency": False,
            "test_read": False,
            "locked_holdout_read": False,
        }
        arrays = {
            "x_anchor": bundle["x_anchor"].astype(np.float32),
            "y_anchor": bundle["y_anchor"].astype(np.int64),
            "initial_weight": random_weight,
            "initial_bias": random_bias,
            "plaintext_average_weight": average_weight,
            "plaintext_average_bias": average_bias,
            "plaintext_c1_weight": c1_weight,
            "plaintext_c1_bias": c1_bias,
            "plaintext_c5_weight": c5_weight,
            "plaintext_c5_bias": c5_bias,
            "p2_average_weight": quantize(
                average_weight, configuration["fixedpoint_fractional"]
            ),
            "p2_average_bias": quantize(
                average_bias, configuration["fixedpoint_fractional"]
            ),
            "p2_c1_weight": quantize(
                c1_weight, configuration["fixedpoint_fractional"]
            ),
            "p2_c1_bias": quantize(
                c1_bias, configuration["fixedpoint_fractional"]
            ),
            "p2_c5_weight": quantize(
                c5_weight, configuration["fixedpoint_fractional"]
            ),
            "p2_c5_bias": quantize(
                c5_bias, configuration["fixedpoint_fractional"]
            ),
            "configuration_json": np.asarray(
                [json.dumps(configuration, sort_keys=True)]
            ),
            "mtt_cache_sha256": np.asarray([assets.mtt_cache_sha256]),
            "original_split_sha256": np.asarray(
                [assets.original_split_sha256]
            ),
        }
        for client_id, indices in enumerate(bundle["partitions"]):
            arrays["x_client_{}".format(client_id)] = bundle["x_local"][
                indices
            ].astype(np.float32)
            arrays["y_client_{}".format(client_id)] = bundle["y_local"][
                indices
            ].astype(np.int64)
            arrays["client_ids_{}".format(client_id)] = indices.astype(
                np.int64
            )
        output = (
            OUTPUT_ROOT
            / "tfe_inputs"
            / dataset
            / "new_stage_b_anchor_postagg_tfe_input_{}_seed42.npz".format(
                dataset
            )
        )
        atomic_npz(output, **arrays)

        p1 = {}
        p2 = {}
        for name, weight, bias in (
            ("C1", c1_weight, c1_bias),
            ("C4", average_weight, average_bias),
            ("C5", c5_weight, c5_bias),
        ):
            p1[name] = classification_metrics(
                assets.x_dev,
                assets.y_dev,
                weight,
                bias,
                assets.num_classes,
            )["accuracy"]
        for name, weight_key, bias_key in (
            ("C1", "p2_c1_weight", "p2_c1_bias"),
            ("C4", "p2_average_weight", "p2_average_bias"),
            ("C5", "p2_c5_weight", "p2_c5_bias"),
        ):
            p2[name] = classification_metrics(
                assets.x_dev,
                assets.y_dev,
                arrays[weight_key],
                arrays[bias_key],
                assets.num_classes,
            )["accuracy"]
        rows.append(
            {
                "dataset": dataset,
                "seed": SEED,
                "distribution": "severe",
                "input_path": str(output),
                "input_sha256": sha256_file(output),
                "configuration": configuration,
                "client_counts": [
                    int(len(bundle["partitions"][index]))
                    for index in range(5)
                ],
                "client_partition_hashes": [
                    sha256_array(value) for value in bundle["partitions"]
                ],
                "p1_plaintext_accuracy": p1,
                "p1_c5_minus_c1": p1["C5"] - p1["C1"],
                "p2_endpoint_accuracy": p2,
                "p2_c5_minus_c1": p2["C5"] - p2["C1"],
                "p2_limitation": (
                    "precision16 endpoint quantization sensitivity only; "
                    "full trajectory correctness is tested by TFE"
                ),
                "test_read": False,
                "locked_holdout_read": False,
            }
        )
    atomic_json(
        MANIFEST_PATH,
        {
            "schema": "new_stage_b_anchor_postagg_tfe_input_manifest_v1",
            "status": "COMPLETE",
            "source_git_commit": source,
            "freeze_sha256": freeze_sha,
            "rows": rows,
            "official_mtt_only": True,
            "full_data_initializer_dependency": False,
            "test_read": False,
            "locked_holdout_read": False,
        },
    )
    print("COMPLETE TFE inputs={}".format(len(rows)))
    for row in rows:
        print(
            "{} P1 C1/C4/C5={} delta={:+.4f} P2 delta={:+.4f}".format(
                row["dataset"],
                row["p1_plaintext_accuracy"],
                row["p1_c5_minus_c1"],
                row["p2_c5_minus_c1"],
            )
        )


if __name__ == "__main__":
    main()

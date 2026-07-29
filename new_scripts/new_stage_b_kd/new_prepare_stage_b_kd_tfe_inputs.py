#!/usr/bin/env python3
"""Freeze seed42 selected plaintext inputs/references for TFE validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from new_run_stage_b_kd_plaintext import (
    build_or_load_local_heads,
    evaluate_local_bundle,
    teacher_candidate,
)
from new_stage_b_kd_core import (
    DATASETS,
    LOG_ROOT,
    OUTPUT_ROOT,
    SEED,
    atomic_json,
    atomic_npz,
    evaluate_head,
    initialize_shared_head,
    load_dataset_assets,
    sha256_array,
    sha256_file,
    source_commit,
    train_student,
)


SUMMARY_PATH = LOG_ROOT / "new_stage_b_kd_smoke_summary.json"
INPUT_ROOT = OUTPUT_ROOT / "tfe_inputs"
MANIFEST_PATH = LOG_ROOT / "new_stage_b_kd_tfe_input_manifest.json"


def quantize(value: np.ndarray, fractional: int = 16) -> np.ndarray:
    scale = float(2**fractional)
    return (np.round(np.asarray(value) * scale) / scale).astype(np.float32)


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["status"] != "COMPLETE" or summary["seed_policy"] != [SEED]:
        raise ValueError("plaintext summary is not COMPLETE seed42-only")
    source = source_commit()
    rows = []
    for smoke in summary["smoke_results"]:
        dataset = smoke["dataset"]
        assets = load_dataset_assets(dataset, SEED)
        local = smoke["selected_local_config"]
        teacher_config = smoke["selected_teacher_config"]
        student_config = smoke["selected_student_config"]
        bundle = build_or_load_local_heads(
            assets,
            "severe",
            local["anchor_per_class"],
            local["local_epochs"],
            local["local_lr"],
            local["weight_decay"],
        )
        evaluation = evaluate_local_bundle(assets, bundle)
        teacher = teacher_candidate(
            assets,
            bundle,
            teacher_config["rule"],
            teacher_config["teacher_temperature"],
            teacher_config["weight_temperature"],
            teacher_config["maximum_client_weight"],
            teacher_config["shrinkage_to_uniform"],
        )
        student_weight, student_bias, _ = train_student(
            bundle["x_anchor"],
            bundle["y_anchor"],
            teacher["teacher_anchor_probabilities"],
            evaluation["average_weight"],
            evaluation["average_bias"],
            teacher["teacher_temperature"],
            student_config["kd_lambda"],
            student_config["average_regularization"],
            student_config["student_steps"],
            student_config["student_lr"],
        )
        initial_weight, initial_bias = initialize_shared_head(
            assets.x_mtt.shape[1], assets.num_classes, SEED
        )
        configuration = {
            "schema": "new_stage_b_kd_tfe_input_config_v1",
            "dataset": dataset,
            "seed": SEED,
            "distribution": "severe",
            "num_clients": 5,
            "feature_dim": int(assets.x_mtt.shape[1]),
            "num_classes": assets.num_classes,
            "anchor_samples_per_class": local["anchor_per_class"],
            "local_epochs": local["local_epochs"],
            "local_lr": local["local_lr"],
            "weight_decay": local["weight_decay"],
            "reliability_rule": teacher_config["rule"],
            "teacher_temperature": teacher_config["teacher_temperature"],
            "weight_temperature": teacher_config["weight_temperature"],
            "maximum_client_weight": teacher_config["maximum_client_weight"],
            "cap_was_inactive_in_plaintext": (
                float(np.max(np.asarray(teacher["class_weights"])))
                < teacher_config["maximum_client_weight"]
            ),
            "shrinkage_to_uniform": teacher_config["shrinkage_to_uniform"],
            "kd_lambda": student_config["kd_lambda"],
            "average_regularization": student_config["average_regularization"],
            "student_steps": student_config["student_steps"],
            "student_lr": student_config["student_lr"],
            "fixedpoint_fractional": 16,
            "matmul_chunk_size": 64,
            "checkpoint_interval": 5 if dataset == "cifar10" else 10,
            "source_commit": source,
            "initializer_source": "official_mtt_only",
            "full_data_initializer_dependency": False,
            "locked_holdout_read": False,
            "official_test_read": False,
        }
        arrays = {
            "x_anchor": bundle["x_anchor"].astype(np.float32),
            "y_anchor": bundle["y_anchor"].astype(np.int64),
            "initial_weight": initial_weight,
            "initial_bias": initial_bias,
            "plaintext_average_weight": evaluation["average_weight"],
            "plaintext_average_bias": evaluation["average_bias"],
            "plaintext_teacher_anchor": teacher[
                "teacher_anchor_probabilities"
            ].astype(np.float32),
            "plaintext_class_weights": np.asarray(
                teacher["class_weights"], dtype=np.float32
            ),
            "plaintext_student_weight": student_weight,
            "plaintext_student_bias": student_bias,
            "p2_quantized_average_weight": quantize(
                evaluation["average_weight"]
            ),
            "p2_quantized_average_bias": quantize(
                evaluation["average_bias"]
            ),
            "p2_quantized_student_weight": quantize(student_weight),
            "p2_quantized_student_bias": quantize(student_bias),
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
        output_path = (
            INPUT_ROOT
            / dataset
            / "new_stage_b_kd_tfe_input_{}_seed42.npz".format(dataset)
        )
        atomic_npz(output_path, **arrays)
        p1_average = evaluate_head(
            assets.x_dev,
            assets.y_dev,
            evaluation["average_weight"],
            evaluation["average_bias"],
            assets.num_classes,
        )
        p1_student = evaluate_head(
            assets.x_dev,
            assets.y_dev,
            student_weight,
            student_bias,
            assets.num_classes,
        )
        p2_average = evaluate_head(
            assets.x_dev,
            assets.y_dev,
            arrays["p2_quantized_average_weight"],
            arrays["p2_quantized_average_bias"],
            assets.num_classes,
        )
        p2_student = evaluate_head(
            assets.x_dev,
            assets.y_dev,
            arrays["p2_quantized_student_weight"],
            arrays["p2_quantized_student_bias"],
            assets.num_classes,
        )
        rows.append(
            {
                "dataset": dataset,
                "seed": SEED,
                "input_path": str(output_path),
                "input_sha256": sha256_file(output_path),
                "configuration": configuration,
                "client_counts": [
                    int(len(bundle["partitions"][client_id]))
                    for client_id in range(5)
                ],
                "client_partition_hashes": [
                    sha256_array(value) for value in bundle["partitions"]
                ],
                "p1_plaintext": {
                    "average_accuracy": p1_average["accuracy"],
                    "student_accuracy": p1_student["accuracy"],
                    "gain": p1_student["accuracy"] - p1_average["accuracy"],
                },
                "p2_endpoint_fixedpoint_quantization": {
                    "scope": (
                        "precision16 endpoint quantization sensitivity; "
                        "not a full secure trajectory emulator"
                    ),
                    "average_accuracy": p2_average["accuracy"],
                    "student_accuracy": p2_student["accuracy"],
                    "gain": p2_student["accuracy"] - p2_average["accuracy"],
                    "student_delta_vs_p1": (
                        p2_student["accuracy"] - p1_student["accuracy"]
                    ),
                },
                "locked_holdout_read": False,
                "official_test_read": False,
            }
        )
    payload = {
        "schema": "new_stage_b_kd_tfe_input_manifest_v1",
        "status": "COMPLETE",
        "source_commit": source,
        "seed_policy": [SEED],
        "rows": rows,
        "p2_limitation_disclosed": True,
        "locked_holdout_read": False,
        "official_test_read": False,
        "full_data_initializer_loaded": False,
    }
    atomic_json(MANIFEST_PATH, payload)
    print("COMPLETE manifest={} inputs={}".format(MANIFEST_PATH, len(rows)))
    for row in rows:
        print(
            "{dataset} P1_gain={p1:+.4f} P2_gain={p2:+.4f}".format(
                dataset=row["dataset"],
                p1=row["p1_plaintext"]["gain"],
                p2=row["p2_endpoint_fixedpoint_quantization"]["gain"],
            )
        )


if __name__ == "__main__":
    main()

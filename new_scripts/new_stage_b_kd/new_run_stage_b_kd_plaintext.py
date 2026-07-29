#!/usr/bin/env python3
"""Resumable single-seed plaintext search for MTT-anchor Consensus KD.

The search order exactly follows S1--S6 from the frozen plan.  Candidate rows
are appended atomically so a tmux/session interruption can resume without
repeating completed local-head training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from new_stage_b_kd_core import (
    CHECKPOINT_ROOT,
    DATASETS,
    LOG_ROOT,
    NUM_CLIENTS,
    OUTPUT_ROOT,
    SEED,
    TABLE_ROOT,
    DatasetAssets,
    atomic_json,
    atomic_npz,
    average_head,
    classification_metrics,
    client_partitions,
    class_aware_teacher,
    evaluate_head,
    gain_retention,
    head_probabilities,
    load_dataset_assets,
    mtt_anchor_split,
    normalize_reliability,
    pairwise_client_diversity,
    partition_summary,
    reliability_matrix,
    sha256_array,
    sha256_file,
    softmax,
    source_commit,
    train_local_heads,
    train_student,
)


REGISTRY_JSON = LOG_ROOT / "new_stage_b_kd_smoke_registry.json"
SUMMARY_JSON = LOG_ROOT / "new_stage_b_kd_smoke_summary.json"
REGISTRY_CSV = TABLE_ROOT / "new_stage_b_kd_smoke_registry.csv"
CONFIRMATION_CSV = TABLE_ROOT / "new_stage_b_kd_plaintext_confirmation.csv"
RUNTIME_CSV = TABLE_ROOT / "new_stage_b_kd_runtime_breakdown.csv"
RESOURCE_CSV = TABLE_ROOT / "new_stage_b_kd_resource_summary.csv"

RELIABILITY_RULES = (
    "uniform",
    "per_class_accuracy",
    "confidence",
    "negative_entropy",
    "calibrated_confidence",
    "class_coverage_aware",
)

DEFAULTS = {
    "local_epochs": 20,
    "local_lr": 0.1,
    "weight_decay": 1e-4,
    "anchor_per_class": 8,
    "teacher_temperature": 2.0,
    "weight_temperature": 1.0,
    "maximum_client_weight": 1.0,
    "shrinkage_to_uniform": 0.0,
    "reliability_rule": "uniform",
    "kd_lambda": 0.5,
    "average_regularization": 1e-4,
    "student_steps": 50,
    "student_lr": 0.1,
}


def stable_key(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def current_peak_rss_gib() -> float:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return float("nan")


def load_registry() -> Dict[str, Any]:
    if REGISTRY_JSON.is_file():
        payload = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
        if payload.get("seed_policy") != [SEED]:
            raise ValueError("registry seed policy mismatch")
        return payload
    return {
        "schema": "new_stage_b_kd_smoke_registry_v1",
        "status": "RUNNING",
        "source_commit": source_commit(),
        "seed_policy": [SEED],
        "backend": "numpy_plaintext_float",
        "runs": [],
        "locked_holdout_read": False,
        "official_test_read": False,
        "full_data_initializer_loaded": False,
    }


def save_registry(registry: Dict[str, Any]) -> None:
    registry["updated_at_unix"] = time.time()
    atomic_json(REGISTRY_JSON, registry)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_key",
        "dataset",
        "seed",
        "stage",
        "distribution",
        "method",
        "factor",
        "factor_value",
        "baseline_accuracy",
        "teacher_accuracy",
        "student_accuracy",
        "baseline_worst_class_accuracy",
        "teacher_worst_class_accuracy",
        "student_worst_class_accuracy",
        "teacher_gain",
        "student_gain",
        "student_gain_retention",
        "iid_safety_delta",
        "runtime_seconds",
        "peak_rss_gib",
        "source_commit",
    ]
    with REGISTRY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in registry["runs"]:
            writer.writerow({name: row.get(name, "") for name in fields})


def find_run(registry: Dict[str, Any], key: str) -> Dict[str, Any] | None:
    for row in registry["runs"]:
        if row["run_key"] == key and row.get("status") == "COMPLETE":
            return row
    return None


def checkpoint_path(dataset: str, key: str) -> Path:
    return CHECKPOINT_ROOT / "plaintext" / dataset / (key + ".npz")


def local_config_payload(
    dataset: str,
    distribution: str,
    anchor_per_class: int,
    local_epochs: int,
    local_lr: float,
    weight_decay: float,
) -> Dict[str, Any]:
    return {
        "type": "local_heads",
        "dataset": dataset,
        "seed": SEED,
        "distribution": distribution,
        "anchor_per_class": int(anchor_per_class),
        "local_epochs": int(local_epochs),
        "local_lr": float(local_lr),
        "weight_decay": float(weight_decay),
        "num_clients": NUM_CLIENTS,
        "final_synchronization": "disabled",
        "optimizer": "full_batch_gradient_descent_cross_entropy",
        "shared_initialization": True,
    }


def build_or_load_local_heads(
    assets: DatasetAssets,
    distribution: str,
    anchor_per_class: int,
    local_epochs: int,
    local_lr: float,
    weight_decay: float,
) -> Dict[str, Any]:
    config = local_config_payload(
        assets.dataset,
        distribution,
        anchor_per_class,
        local_epochs,
        local_lr,
        weight_decay,
    )
    key = stable_key(config)
    path = checkpoint_path(assets.dataset, key)
    local_ids, anchor_ids = mtt_anchor_split(
        assets.y_mtt, assets.num_classes, anchor_per_class, SEED
    )
    x_local = assets.x_mtt[local_ids]
    y_local = assets.y_mtt[local_ids]
    partitions = client_partitions(y_local, distribution, SEED)
    if path.is_file():
        with np.load(str(path), allow_pickle=False) as saved:
            weights = [
                np.asarray(saved["weights"][index], dtype=np.float32)
                for index in range(NUM_CLIENTS)
            ]
            biases = [
                np.asarray(saved["biases"][index], dtype=np.float32)
                for index in range(NUM_CLIENTS)
            ]
            cached_local_ids = np.asarray(saved["local_ids"], dtype=np.int64)
            cached_anchor_ids = np.asarray(saved["anchor_ids"], dtype=np.int64)
            cached_partition_hashes = [
                str(value)
                for value in saved["partition_hashes"].reshape(-1).tolist()
            ]
            local_training_seconds = float(
                saved["local_training_seconds"].reshape(-1)[0]
            )
        expected_partition_hashes = [sha256_array(value) for value in partitions]
        if (
            not np.array_equal(cached_local_ids, local_ids)
            or not np.array_equal(cached_anchor_ids, anchor_ids)
            or cached_partition_hashes != expected_partition_hashes
        ):
            raise ValueError("plaintext checkpoint partition mismatch: {}".format(path))
        cache_hit = True
    else:
        weights, biases, training = train_local_heads(
            x_local,
            y_local,
            partitions,
            assets.num_classes,
            SEED,
            local_epochs,
            local_lr,
            weight_decay,
        )
        local_training_seconds = training["local_training_seconds"]
        atomic_npz(
            path,
            weights=np.stack(weights),
            biases=np.stack(biases),
            local_ids=local_ids,
            anchor_ids=anchor_ids,
            partition_hashes=np.asarray(
                [sha256_array(value) for value in partitions]
            ),
            local_training_seconds=np.asarray([local_training_seconds]),
            initializer_source=np.asarray(["official_mtt_only"]),
            full_data_initializer_dependency=np.asarray([0], dtype=np.int64),
            seed=np.asarray([SEED], dtype=np.int64),
        )
        cache_hit = False
    return {
        "key": key,
        "config": config,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "cache_hit": cache_hit,
        "weights": weights,
        "biases": biases,
        "local_ids": local_ids,
        "anchor_ids": anchor_ids,
        "x_local": x_local,
        "y_local": y_local,
        "x_anchor": assets.x_mtt[anchor_ids],
        "y_anchor": assets.y_mtt[anchor_ids],
        "partitions": partitions,
        "partition_summary": partition_summary(
            y_local, partitions, assets.num_classes
        ),
        "local_training_seconds": local_training_seconds,
    }


def evaluate_local_bundle(
    assets: DatasetAssets, bundle: Dict[str, Any]
) -> Dict[str, Any]:
    inference_started = time.perf_counter()
    weights = bundle["weights"]
    biases = bundle["biases"]
    average_weight, average_bias = average_head(weights, biases)
    average_metrics = evaluate_head(
        assets.x_dev,
        assets.y_dev,
        average_weight,
        average_bias,
        assets.num_classes,
    )
    development_client_probabilities = head_probabilities(
        assets.x_dev, weights, biases, 1.0
    )
    anchor_client_probabilities = head_probabilities(
        bundle["x_anchor"], weights, biases, 1.0
    )
    local_metrics = [
        classification_metrics(
            development_client_probabilities[index],
            assets.y_dev,
            assets.num_classes,
        )
        for index in range(NUM_CLIENTS)
    ]
    uniform_weights = np.full(
        (NUM_CLIENTS, assets.num_classes), 1.0 / NUM_CLIENTS, dtype=np.float32
    )
    uniform_development = class_aware_teacher(
        development_client_probabilities, uniform_weights
    )
    uniform_anchor = class_aware_teacher(
        anchor_client_probabilities, uniform_weights
    )
    uniform_metrics = classification_metrics(
        uniform_development, assets.y_dev, assets.num_classes
    )
    diversity = pairwise_client_diversity(
        weights, biases, development_client_probabilities
    )

    oracle_reliability = np.asarray(
        [
            metrics["per_class_accuracy"]
            for metrics in local_metrics
        ],
        dtype=np.float64,
    )
    oracle_weights = np.zeros_like(oracle_reliability, dtype=np.float32)
    oracle_weights[
        np.argmax(oracle_reliability, axis=0), np.arange(assets.num_classes)
    ] = 1.0
    oracle_metrics = classification_metrics(
        class_aware_teacher(development_client_probabilities, oracle_weights),
        assets.y_dev,
        assets.num_classes,
    )
    return {
        "average_weight": average_weight,
        "average_bias": average_bias,
        "average_metrics": average_metrics,
        "local_metrics": local_metrics,
        "uniform_teacher_metrics": uniform_metrics,
        "uniform_anchor_probabilities": uniform_anchor,
        "diversity": diversity,
        "oracle_classwise_teacher_metrics": oracle_metrics,
        "anchor_inference_seconds": time.perf_counter() - inference_started,
    }


def teacher_candidate(
    assets: DatasetAssets,
    bundle: Dict[str, Any],
    rule: str,
    teacher_temperature: float,
    weight_temperature: float,
    maximum_client_weight: float,
    shrinkage_to_uniform: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    anchor_probabilities = head_probabilities(
        bundle["x_anchor"],
        bundle["weights"],
        bundle["biases"],
        teacher_temperature,
    )
    development_probabilities = head_probabilities(
        assets.x_dev,
        bundle["weights"],
        bundle["biases"],
        teacher_temperature,
    )
    reliability = reliability_matrix(
        rule,
        anchor_probabilities,
        bundle["y_anchor"],
        bundle["y_local"],
        bundle["partitions"],
        assets.num_classes,
    )
    weights = normalize_reliability(
        reliability,
        weight_temperature,
        maximum_client_weight,
        shrinkage_to_uniform,
    )
    teacher_anchor = class_aware_teacher(anchor_probabilities, weights)
    teacher_development = class_aware_teacher(development_probabilities, weights)
    return {
        "rule": rule,
        "teacher_temperature": float(teacher_temperature),
        "weight_temperature": float(weight_temperature),
        "maximum_client_weight": float(maximum_client_weight),
        "shrinkage_to_uniform": float(shrinkage_to_uniform),
        "reliability": reliability.tolist(),
        "class_weights": weights.tolist(),
        "teacher_anchor_probabilities": teacher_anchor,
        "metrics": classification_metrics(
            teacher_development, assets.y_dev, assets.num_classes
        ),
        "teacher_aggregation_seconds": time.perf_counter() - started,
    }


def append_local_run(
    registry: Dict[str, Any],
    assets: DatasetAssets,
    stage: str,
    factor: str,
    factor_value: Any,
    bundle: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    key = stable_key(
        {
            "stage": stage,
            "factor": factor,
            "factor_value": factor_value,
            "local_key": bundle["key"],
        }
    )
    existing = find_run(registry, key)
    if existing is not None:
        return existing
    baseline = evaluation["average_metrics"]
    teacher = evaluation["uniform_teacher_metrics"]
    row = {
        "run_key": key,
        "status": "COMPLETE",
        "dataset": assets.dataset,
        "seed": SEED,
        "stage": stage,
        "distribution": bundle["config"]["distribution"],
        "method": "uniform_logit_teacher",
        "factor": factor,
        "factor_value": factor_value,
        "config": bundle["config"],
        "local_checkpoint": bundle["checkpoint"],
        "local_checkpoint_sha256": bundle["checkpoint_sha256"],
        "baseline_accuracy": baseline["accuracy"],
        "teacher_accuracy": teacher["accuracy"],
        "student_accuracy": None,
        "baseline_worst_class_accuracy": baseline["worst_class_accuracy"],
        "teacher_worst_class_accuracy": teacher["worst_class_accuracy"],
        "student_worst_class_accuracy": None,
        "baseline_macro_f1": baseline["macro_f1"],
        "teacher_macro_f1": teacher["macro_f1"],
        "teacher_gain": teacher["accuracy"] - baseline["accuracy"],
        "student_gain": None,
        "student_gain_retention": None,
        "iid_safety_delta": None,
        "local_metrics": evaluation["local_metrics"],
        "oracle_classwise_teacher_metrics": evaluation[
            "oracle_classwise_teacher_metrics"
        ],
        "diversity": evaluation["diversity"],
        "partition_summary": bundle["partition_summary"],
        "runtime_breakdown": {
            "local_training_seconds": bundle["local_training_seconds"],
            "anchor_inference_seconds": evaluation["anchor_inference_seconds"],
            "teacher_aggregation_seconds": 0.0,
            "student_kd_seconds": 0.0,
        },
        "runtime_seconds": bundle["local_training_seconds"]
        + evaluation["anchor_inference_seconds"],
        "peak_rss_gib": current_peak_rss_gib(),
        "source_commit": registry["source_commit"],
        "initializer_source": "official_mtt_only",
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
        "backend": "numpy_plaintext_float",
    }
    registry["runs"].append(row)
    save_registry(registry)
    print(
        "{ds} {stage} {factor}={value} avg={avg:.4f} uniform={teacher:.4f} "
        "gain={gain:+.4f} worst={worst:.4f} disagreement={dis:.4f}".format(
            ds=assets.dataset,
            stage=stage,
            factor=factor,
            value=factor_value,
            avg=row["baseline_accuracy"],
            teacher=row["teacher_accuracy"],
            gain=row["teacher_gain"],
            worst=row["teacher_worst_class_accuracy"],
            dis=row["diversity"]["mean_prediction_disagreement"],
        ),
        flush=True,
    )
    return row


def append_teacher_run(
    registry: Dict[str, Any],
    assets: DatasetAssets,
    stage: str,
    factor: str,
    factor_value: Any,
    bundle: Dict[str, Any],
    local_evaluation: Dict[str, Any],
    teacher: Dict[str, Any],
) -> Dict[str, Any]:
    config = {
        "local": bundle["config"],
        "rule": teacher["rule"],
        "teacher_temperature": teacher["teacher_temperature"],
        "weight_temperature": teacher["weight_temperature"],
        "maximum_client_weight": teacher["maximum_client_weight"],
        "shrinkage_to_uniform": teacher["shrinkage_to_uniform"],
    }
    key = stable_key({"stage": stage, "factor": factor, "config": config})
    existing = find_run(registry, key)
    if existing is not None:
        return existing
    baseline = local_evaluation["average_metrics"]
    metrics = teacher["metrics"]
    row = {
        "run_key": key,
        "status": "COMPLETE",
        "dataset": assets.dataset,
        "seed": SEED,
        "stage": stage,
        "distribution": bundle["config"]["distribution"],
        "method": "class_aware_consensus_teacher",
        "factor": factor,
        "factor_value": factor_value,
        "config": config,
        "baseline_accuracy": baseline["accuracy"],
        "teacher_accuracy": metrics["accuracy"],
        "student_accuracy": None,
        "baseline_worst_class_accuracy": baseline["worst_class_accuracy"],
        "teacher_worst_class_accuracy": metrics["worst_class_accuracy"],
        "student_worst_class_accuracy": None,
        "baseline_macro_f1": baseline["macro_f1"],
        "teacher_macro_f1": metrics["macro_f1"],
        "teacher_gain": metrics["accuracy"] - baseline["accuracy"],
        "student_gain": None,
        "student_gain_retention": None,
        "iid_safety_delta": None,
        "diversity": local_evaluation["diversity"],
        "class_weights": teacher["class_weights"],
        "runtime_breakdown": {
            "local_training_seconds": bundle["local_training_seconds"],
            "anchor_inference_seconds": local_evaluation[
                "anchor_inference_seconds"
            ],
            "teacher_aggregation_seconds": teacher[
                "teacher_aggregation_seconds"
            ],
            "student_kd_seconds": 0.0,
        },
        "runtime_seconds": bundle["local_training_seconds"]
        + local_evaluation["anchor_inference_seconds"]
        + teacher["teacher_aggregation_seconds"],
        "peak_rss_gib": current_peak_rss_gib(),
        "source_commit": registry["source_commit"],
        "initializer_source": "official_mtt_only",
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
        "backend": "numpy_plaintext_float",
    }
    registry["runs"].append(row)
    save_registry(registry)
    print(
        "{ds} {stage} {factor}={value} avg={avg:.4f} teacher={teacher:.4f} "
        "gain={gain:+.4f} worst={worst:.4f}".format(
            ds=assets.dataset,
            stage=stage,
            factor=factor,
            value=factor_value,
            avg=row["baseline_accuracy"],
            teacher=row["teacher_accuracy"],
            gain=row["teacher_gain"],
            worst=row["teacher_worst_class_accuracy"],
        ),
        flush=True,
    )
    return row


def append_student_run(
    registry: Dict[str, Any],
    assets: DatasetAssets,
    stage: str,
    factor: str,
    factor_value: Any,
    bundle: Dict[str, Any],
    local_evaluation: Dict[str, Any],
    teacher: Dict[str, Any],
    student_config: Dict[str, Any],
) -> Dict[str, Any]:
    config = {
        "local": bundle["config"],
        "teacher": {
            name: teacher[name]
            for name in (
                "rule",
                "teacher_temperature",
                "weight_temperature",
                "maximum_client_weight",
                "shrinkage_to_uniform",
            )
        },
        "student": student_config,
    }
    key = stable_key({"stage": stage, "factor": factor, "config": config})
    existing = find_run(registry, key)
    if existing is not None:
        return existing
    weight, bias, training = train_student(
        bundle["x_anchor"],
        bundle["y_anchor"],
        teacher["teacher_anchor_probabilities"],
        local_evaluation["average_weight"],
        local_evaluation["average_bias"],
        teacher["teacher_temperature"],
        student_config["kd_lambda"],
        student_config["average_regularization"],
        student_config["student_steps"],
        student_config["student_lr"],
    )
    student_metrics = evaluate_head(
        assets.x_dev, assets.y_dev, weight, bias, assets.num_classes
    )
    baseline = local_evaluation["average_metrics"]
    teacher_metrics = teacher["metrics"]
    output_path = (
        OUTPUT_ROOT
        / "plaintext_candidates"
        / assets.dataset
        / (key + ".npz")
    )
    atomic_npz(
        output_path,
        weight=weight,
        bias=bias,
        initializer_source=np.asarray(["official_mtt_only"]),
        full_data_initializer_dependency=np.asarray([0], dtype=np.int64),
        seed=np.asarray([SEED], dtype=np.int64),
    )
    retention = gain_retention(
        baseline["accuracy"], teacher_metrics["accuracy"], student_metrics["accuracy"]
    )
    row = {
        "run_key": key,
        "status": "COMPLETE",
        "dataset": assets.dataset,
        "seed": SEED,
        "stage": stage,
        "distribution": bundle["config"]["distribution"],
        "method": "class_aware_consensus_kd_student",
        "factor": factor,
        "factor_value": factor_value,
        "config": config,
        "output_model": str(output_path),
        "output_model_sha256": sha256_file(output_path),
        "baseline_accuracy": baseline["accuracy"],
        "teacher_accuracy": teacher_metrics["accuracy"],
        "student_accuracy": student_metrics["accuracy"],
        "baseline_worst_class_accuracy": baseline["worst_class_accuracy"],
        "teacher_worst_class_accuracy": teacher_metrics["worst_class_accuracy"],
        "student_worst_class_accuracy": student_metrics["worst_class_accuracy"],
        "baseline_macro_f1": baseline["macro_f1"],
        "teacher_macro_f1": teacher_metrics["macro_f1"],
        "student_macro_f1": student_metrics["macro_f1"],
        "teacher_gain": teacher_metrics["accuracy"] - baseline["accuracy"],
        "student_gain": student_metrics["accuracy"] - baseline["accuracy"],
        "student_gain_retention": retention,
        "distance_from_average": training["distance_from_average"],
        "iid_safety_delta": None,
        "diversity": local_evaluation["diversity"],
        "runtime_breakdown": {
            "local_training_seconds": bundle["local_training_seconds"],
            "anchor_inference_seconds": local_evaluation[
                "anchor_inference_seconds"
            ],
            "teacher_aggregation_seconds": teacher[
                "teacher_aggregation_seconds"
            ],
            "student_kd_seconds": training["student_kd_seconds"],
        },
        "runtime_seconds": bundle["local_training_seconds"]
        + local_evaluation["anchor_inference_seconds"]
        + teacher["teacher_aggregation_seconds"]
        + training["student_kd_seconds"],
        "peak_rss_gib": current_peak_rss_gib(),
        "source_commit": registry["source_commit"],
        "initializer_source": "official_mtt_only",
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
        "backend": "numpy_plaintext_float",
    }
    registry["runs"].append(row)
    save_registry(registry)
    print(
        "{ds} {stage} {factor}={value} avg={avg:.4f} teacher={teacher:.4f} "
        "student={student:.4f} gain={gain:+.4f} retention={ret}".format(
            ds=assets.dataset,
            stage=stage,
            factor=factor,
            value=factor_value,
            avg=row["baseline_accuracy"],
            teacher=row["teacher_accuracy"],
            student=row["student_accuracy"],
            gain=row["student_gain"],
            ret="N/A" if retention is None else "{:.3f}".format(retention),
        ),
        flush=True,
    )
    return row


def best_teacher_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["teacher_accuracy"],
            row["teacher_worst_class_accuracy"],
            row["teacher_macro_f1"],
        ),
    )


def best_student_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["student_accuracy"],
            row["student_worst_class_accuracy"],
            row["student_macro_f1"],
        ),
    )


def run_dataset_smoke(
    registry: Dict[str, Any], assets: DatasetAssets
) -> Dict[str, Any]:
    # S1: local diversity, sequential epochs then LR multiplier.
    s1_rows: List[Dict[str, Any]] = []
    epoch_rows = []
    for epochs in (5, 10, 20, 40):
        bundle = build_or_load_local_heads(
            assets, "severe", 8, epochs, DEFAULTS["local_lr"], DEFAULTS["weight_decay"]
        )
        evaluation = evaluate_local_bundle(assets, bundle)
        epoch_rows.append(
            append_local_run(
                registry, assets, "S1", "local_epochs", epochs, bundle, evaluation
            )
        )
    best_epoch = best_teacher_row(epoch_rows)["config"]["local_epochs"]
    s1_rows.extend(epoch_rows)
    for multiplier in (0.5, 1.0, 2.0):
        lr = DEFAULTS["local_lr"] * multiplier
        bundle = build_or_load_local_heads(
            assets, "severe", 8, best_epoch, lr, DEFAULTS["weight_decay"]
        )
        evaluation = evaluate_local_bundle(assets, bundle)
        row = append_local_run(
            registry, assets, "S1", "local_lr_multiplier", multiplier, bundle, evaluation
        )
        if row["run_key"] not in {value["run_key"] for value in s1_rows}:
            s1_rows.append(row)
    top_two = sorted(
        s1_rows,
        key=lambda row: (
            row["teacher_accuracy"],
            row["teacher_worst_class_accuracy"],
            row["diversity"]["mean_prediction_disagreement"],
        ),
        reverse=True,
    )[:2]

    # S2: anchor sizes for at most two S1 settings.
    s2_rows = []
    for local_row in top_two:
        config = local_row["config"]
        for anchor_size in (4, 8, 12):
            bundle = build_or_load_local_heads(
                assets,
                "severe",
                anchor_size,
                config["local_epochs"],
                config["local_lr"],
                config["weight_decay"],
            )
            evaluation = evaluate_local_bundle(assets, bundle)
            s2_rows.append(
                append_local_run(
                    registry,
                    assets,
                    "S2",
                    "anchor_samples_per_class",
                    anchor_size,
                    bundle,
                    evaluation,
                )
            )
    s2_best = best_teacher_row(s2_rows)
    local_config = s2_best["config"]
    bundle = build_or_load_local_heads(
        assets,
        "severe",
        local_config["anchor_per_class"],
        local_config["local_epochs"],
        local_config["local_lr"],
        local_config["weight_decay"],
    )
    local_evaluation = evaluate_local_bundle(assets, bundle)

    # S3: reliability -> teacher temperature -> weight temperature.
    reliability_rows = []
    for rule in RELIABILITY_RULES:
        teacher = teacher_candidate(
            assets, bundle, rule, 2.0, 1.0, 1.0, 0.0
        )
        reliability_rows.append(
            append_teacher_run(
                registry, assets, "S3", "reliability_rule", rule, bundle,
                local_evaluation, teacher
            )
        )
    best_rule = best_teacher_row(reliability_rows)["config"]["rule"]
    temperature_rows = []
    for temperature in (1.0, 2.0, 4.0, 8.0):
        teacher = teacher_candidate(
            assets, bundle, best_rule, temperature, 1.0, 1.0, 0.0
        )
        temperature_rows.append(
            append_teacher_run(
                registry, assets, "S3", "teacher_temperature", temperature,
                bundle, local_evaluation, teacher
            )
        )
    best_temperature = best_teacher_row(temperature_rows)["config"][
        "teacher_temperature"
    ]
    weight_temperature_rows = []
    for weight_temperature in (0.5, 1.0, 2.0, 4.0):
        teacher = teacher_candidate(
            assets,
            bundle,
            best_rule,
            best_temperature,
            weight_temperature,
            1.0,
            0.0,
        )
        weight_temperature_rows.append(
            append_teacher_run(
                registry, assets, "S3", "weight_temperature",
                weight_temperature, bundle, local_evaluation, teacher
            )
        )
    best_weight_temperature = best_teacher_row(weight_temperature_rows)["config"][
        "weight_temperature"
    ]

    # S4: cap then shrinkage.
    cap_rows = []
    for cap in (0.30, 0.40, 0.50, 1.00):
        teacher = teacher_candidate(
            assets,
            bundle,
            best_rule,
            best_temperature,
            best_weight_temperature,
            cap,
            0.0,
        )
        cap_rows.append(
            append_teacher_run(
                registry, assets, "S4", "maximum_client_weight", cap,
                bundle, local_evaluation, teacher
            )
        )
    best_cap = best_teacher_row(cap_rows)["config"]["maximum_client_weight"]
    shrinkage_rows = []
    for shrinkage in (0.0, 0.25, 0.50):
        teacher = teacher_candidate(
            assets,
            bundle,
            best_rule,
            best_temperature,
            best_weight_temperature,
            best_cap,
            shrinkage,
        )
        shrinkage_rows.append(
            append_teacher_run(
                registry, assets, "S4", "shrinkage_to_uniform", shrinkage,
                bundle, local_evaluation, teacher
            )
        )
    selected_teacher_row = best_teacher_row(shrinkage_rows)
    teacher_config = selected_teacher_row["config"]
    selected_teacher = teacher_candidate(
        assets,
        bundle,
        teacher_config["rule"],
        teacher_config["teacher_temperature"],
        teacher_config["weight_temperature"],
        teacher_config["maximum_client_weight"],
        teacher_config["shrinkage_to_uniform"],
    )

    student_rows: List[Dict[str, Any]] = []
    selected_student_config = {
        "kd_lambda": DEFAULTS["kd_lambda"],
        "average_regularization": DEFAULTS["average_regularization"],
        "student_steps": DEFAULTS["student_steps"],
        "student_lr": DEFAULTS["student_lr"],
    }
    if selected_teacher_row["teacher_gain"] > 0.0:
        for factor, values in (
            ("kd_lambda", (0.25, 0.50, 0.75)),
            ("average_regularization", (0.0, 1e-4, 1e-3)),
            ("student_steps", (20, 50, 100)),
            ("student_lr", (0.05, 0.10, 0.20)),
        ):
            factor_rows = []
            for value in values:
                config = dict(selected_student_config)
                config[factor] = value
                factor_rows.append(
                    append_student_run(
                        registry,
                        assets,
                        "S5",
                        factor,
                        value,
                        bundle,
                        local_evaluation,
                        selected_teacher,
                        config,
                    )
                )
            selected = best_student_row(factor_rows)
            selected_student_config = dict(selected["config"]["student"])
            student_rows.extend(factor_rows)
        selected_student_row = best_student_row(
            [
                row
                for row in student_rows
                if row["config"]["student"] == selected_student_config
            ]
            or student_rows
        )
    else:
        selected_student_row = None

    result = {
        "dataset": assets.dataset,
        "seed": SEED,
        "selected_local_config": local_config,
        "selected_teacher_config": {
            name: teacher_config[name]
            for name in (
                "rule",
                "teacher_temperature",
                "weight_temperature",
                "maximum_client_weight",
                "shrinkage_to_uniform",
            )
        },
        "selected_student_config": selected_student_config,
        "baseline_severe_accuracy": local_evaluation["average_metrics"]["accuracy"],
        "teacher_severe_accuracy": selected_teacher["metrics"]["accuracy"],
        "teacher_severe_gain": selected_teacher["metrics"]["accuracy"]
        - local_evaluation["average_metrics"]["accuracy"],
        "student_severe_accuracy": (
            None if selected_student_row is None else selected_student_row["student_accuracy"]
        ),
        "student_severe_gain": (
            None if selected_student_row is None else selected_student_row["student_gain"]
        ),
        "student_gain_retention": (
            None
            if selected_student_row is None
            else selected_student_row["student_gain_retention"]
        ),
        "diversity": local_evaluation["diversity"],
        "teacher_positive": selected_teacher_row["teacher_gain"] > 0.0,
        "student_available": selected_student_row is not None,
    }
    return result


def confirmation_method_rows(
    registry: Dict[str, Any],
    assets: DatasetAssets,
    smoke: Dict[str, Any],
    distribution: str,
) -> List[Dict[str, Any]]:
    local = smoke["selected_local_config"]
    bundle = build_or_load_local_heads(
        assets,
        distribution,
        local["anchor_per_class"],
        local["local_epochs"],
        local["local_lr"],
        local["weight_decay"],
    )
    evaluation = evaluate_local_bundle(assets, bundle)
    teacher_config = smoke["selected_teacher_config"]
    class_teacher = teacher_candidate(
        assets,
        bundle,
        teacher_config["rule"],
        teacher_config["teacher_temperature"],
        teacher_config["weight_temperature"],
        teacher_config["maximum_client_weight"],
        teacher_config["shrinkage_to_uniform"],
    )
    uniform_teacher = teacher_candidate(
        assets,
        bundle,
        "uniform",
        teacher_config["teacher_temperature"],
        1.0,
        1.0,
        0.0,
    )
    student_config = smoke["selected_student_config"]
    rows = []
    baseline = evaluation["average_metrics"]
    rows.append(
        {
            "dataset": assets.dataset,
            "seed": SEED,
            "distribution": distribution,
            "method": "secure_average_equivalent_plaintext",
            "initializer_accuracy": baseline["accuracy"],
            "macro_f1": baseline["macro_f1"],
            "worst_class_accuracy": baseline["worst_class_accuracy"],
            "teacher_accuracy": None,
            "teacher_gain": None,
            "student_gain_retention": None,
            "stage_b_runtime_seconds": bundle["local_training_seconds"],
            "source_commit": registry["source_commit"],
        }
    )
    for method, teacher in (
        ("uniform_logit_kd", uniform_teacher),
        ("class_aware_consensus_kd", class_teacher),
    ):
        weight, bias, training = train_student(
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
        metrics = evaluate_head(
            assets.x_dev, assets.y_dev, weight, bias, assets.num_classes
        )
        retention = gain_retention(
            baseline["accuracy"], teacher["metrics"]["accuracy"], metrics["accuracy"]
        )
        rows.append(
            {
                "dataset": assets.dataset,
                "seed": SEED,
                "distribution": distribution,
                "method": method,
                "initializer_accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "worst_class_accuracy": metrics["worst_class_accuracy"],
                "teacher_accuracy": teacher["metrics"]["accuracy"],
                "teacher_gain": teacher["metrics"]["accuracy"]
                - baseline["accuracy"],
                "student_gain_retention": retention,
                "stage_b_runtime_seconds": bundle["local_training_seconds"]
                + evaluation["anchor_inference_seconds"]
                + teacher["teacher_aggregation_seconds"]
                + training["student_kd_seconds"],
                "source_commit": registry["source_commit"],
            }
        )
    return rows


def write_confirmation(rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "seed",
        "distribution",
        "method",
        "initializer_accuracy",
        "macro_f1",
        "worst_class_accuracy",
        "teacher_accuracy",
        "teacher_gain",
        "student_gain_retention",
        "stage_b_runtime_seconds",
        "source_commit",
    ]
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    with CONFIRMATION_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_runtime_and_resources(registry: Dict[str, Any]) -> None:
    runtime_fields = [
        "run_key",
        "dataset",
        "stage",
        "distribution",
        "method",
        "local_training_seconds",
        "anchor_inference_seconds",
        "teacher_aggregation_seconds",
        "student_kd_seconds",
        "total_seconds",
    ]
    with RUNTIME_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=runtime_fields, lineterminator="\n")
        writer.writeheader()
        for row in registry["runs"]:
            breakdown = row["runtime_breakdown"]
            writer.writerow(
                {
                    "run_key": row["run_key"],
                    "dataset": row["dataset"],
                    "stage": row["stage"],
                    "distribution": row["distribution"],
                    "method": row["method"],
                    **breakdown,
                    "total_seconds": row["runtime_seconds"],
                }
            )
    resource_fields = [
        "run_key",
        "dataset",
        "stage",
        "method",
        "peak_rss_gib",
        "output_model_sha256",
        "communication_bytes",
        "communication_rounds",
    ]
    with RESOURCE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resource_fields, lineterminator="\n")
        writer.writeheader()
        for row in registry["runs"]:
            writer.writerow(
                {
                    "run_key": row["run_key"],
                    "dataset": row["dataset"],
                    "stage": row["stage"],
                    "method": row["method"],
                    "peak_rss_gib": row["peak_rss_gib"],
                    "output_model_sha256": row.get("output_model_sha256", ""),
                    "communication_bytes": "N/A",
                    "communication_rounds": "N/A",
                }
            )


def build_summary(
    registry: Dict[str, Any],
    smoke_results: Sequence[Dict[str, Any]],
    confirmation_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    decisions = []
    for smoke in smoke_results:
        dataset = smoke["dataset"]
        rows = [
            row for row in confirmation_rows if row["dataset"] == dataset
        ]
        severe = {row["method"]: row for row in rows if row["distribution"] == "severe"}
        iid = {row["method"]: row for row in rows if row["distribution"] == "iid"}
        baseline_severe = severe["secure_average_equivalent_plaintext"]
        candidate_severe = severe["class_aware_consensus_kd"]
        baseline_iid = iid["secure_average_equivalent_plaintext"]
        candidate_iid = iid["class_aware_consensus_kd"]
        accuracy_gain = (
            candidate_severe["initializer_accuracy"]
            - baseline_severe["initializer_accuracy"]
        )
        worst_gain = (
            candidate_severe["worst_class_accuracy"]
            - baseline_severe["worst_class_accuracy"]
        )
        iid_delta = (
            candidate_iid["initializer_accuracy"]
            - baseline_iid["initializer_accuracy"]
        )
        teacher_positive = candidate_severe["teacher_gain"] > 0.0
        retention = candidate_severe["student_gain_retention"]
        diversity = smoke["diversity"]
        gates = {
            "primary_effect": accuracy_gain >= 0.01 or worst_gain >= 0.02,
            "iid_safety": iid_delta >= -0.005,
            "teacher_positive": teacher_positive,
            "student_retention": retention is not None and retention >= 0.8,
            "diversity_valid": (
                diversity["mean_prediction_disagreement"] > 1e-6
                and diversity["mean_js_divergence"] > 1e-9
            ),
            "lineage": True,
        }
        decisions.append(
            {
                "dataset": dataset,
                "seed": SEED,
                "severe_accuracy_gain": accuracy_gain,
                "severe_worst_class_gain": worst_gain,
                "iid_accuracy_delta": iid_delta,
                "teacher_gain": candidate_severe["teacher_gain"],
                "student_gain_retention": retention,
                "gates": gates,
                "plaintext_confirmation_pass": all(gates.values()),
            }
        )
    return {
        "schema": "new_stage_b_kd_smoke_summary_v1",
        "status": "COMPLETE",
        "source_commit": registry["source_commit"],
        "seed_policy": [SEED],
        "cross_seed_stability_claimed": False,
        "datasets": list(DATASETS),
        "search_stages_completed": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "smoke_results": list(smoke_results),
        "decisions": decisions,
        "any_plaintext_confirmation_pass": any(
            decision["plaintext_confirmation_pass"] for decision in decisions
        ),
        "all_required_datasets_pass": all(
            decision["plaintext_confirmation_pass"] for decision in decisions
        ),
        "locked_holdout_read": False,
        "official_test_read": False,
        "full_data_initializer_loaded": False,
        "next_phase": (
            "single_seed_tfe_validation_for_passing_datasets"
            if any(decision["plaintext_confirmation_pass"] for decision in decisions)
            else "negative_report_no_tfe"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=SEED, choices=[SEED])
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
    registry = load_registry()
    print(
        "Stage-B KD plaintext start source_commit={} seed=42 nproc={} "
        "locked_holdout_read=False official_test_read=False".format(
            registry["source_commit"], os.cpu_count()
        ),
        flush=True,
    )
    assets_by_dataset = {
        dataset: load_dataset_assets(dataset, SEED) for dataset in args.datasets
    }
    smoke_results = [
        run_dataset_smoke(registry, assets_by_dataset[dataset])
        for dataset in args.datasets
    ]
    confirmation_rows = []
    for smoke in smoke_results:
        assets = assets_by_dataset[smoke["dataset"]]
        for distribution in ("iid", "mild", "severe"):
            confirmation_rows.extend(
                confirmation_method_rows(
                    registry, assets, smoke, distribution
                )
            )
    write_confirmation(confirmation_rows)
    write_runtime_and_resources(registry)
    summary = build_summary(registry, smoke_results, confirmation_rows)
    atomic_json(SUMMARY_JSON, summary)
    registry["status"] = "COMPLETE"
    registry["summary_path"] = str(SUMMARY_JSON)
    save_registry(registry)
    print(
        "COMPLETE runs={} passing_datasets={} next_phase={}".format(
            len(registry["runs"]),
            [
                row["dataset"]
                for row in summary["decisions"]
                if row["plaintext_confirmation_pass"]
            ],
            summary["next_phase"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

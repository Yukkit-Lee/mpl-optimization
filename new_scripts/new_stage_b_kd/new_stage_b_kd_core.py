#!/usr/bin/env python3
"""Core utilities for single-seed MTT-anchor Stage-B experiments.

This module only loads audited Official-MTT features and the disjoint
development-validation split.  It never loads a Full-Data initializer, locked
holdout, official test, or any Stage-C model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "new_log/new_stage_b_kd"
TABLE_ROOT = ROOT / "new_expTab/new_stage_b_kd"
OUTPUT_ROOT = ROOT / "new_output/new_stage_b_kd"
CHECKPOINT_ROOT = ROOT / "new_checkpoints/new_stage_b_kd"
CONFIG_ROOT = ROOT / "new_configs/new_stage_b_kd"

DATASETS = ("cifar10", "cifar100")
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
SEED = 42
NUM_CLIENTS = 5
ANCHOR_SIZES = (4, 8, 12)


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float32) / float(temperature)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / np.sum(values, axis=1, keepdims=True)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    result = np.zeros((len(labels), num_classes), dtype=np.float32)
    result[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
    return result


def cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    selected = probabilities[np.arange(len(labels)), labels.astype(np.int64)]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def macro_f1(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    scores = []
    for class_id in range(num_classes):
        true_positive = int(np.sum((predictions == class_id) & (labels == class_id)))
        false_positive = int(np.sum((predictions == class_id) & (labels != class_id)))
        false_negative = int(np.sum((predictions != class_id) & (labels == class_id)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


def classification_metrics(
    probabilities: np.ndarray, labels: np.ndarray, num_classes: int
) -> Dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1)
    per_class = []
    for class_id in range(num_classes):
        selected = labels == class_id
        per_class.append(float(np.mean(predictions[selected] == labels[selected])))
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": macro_f1(predictions, labels, num_classes),
        "worst_class_accuracy": float(np.min(per_class)),
        "per_class_accuracy": per_class,
        "cross_entropy": cross_entropy(probabilities, labels),
    }


@dataclass
class DatasetAssets:
    dataset: str
    seed: int
    num_classes: int
    x_mtt: np.ndarray
    y_mtt: np.ndarray
    x_dev: np.ndarray
    y_dev: np.ndarray
    mtt_cache_path: Path
    mtt_cache_sha256: str
    original_split_path: Path
    original_split_sha256: str
    development_ids_sha256: str
    encoder_sha256: str
    scaler_sha256: str


def load_dataset_assets(dataset: str, seed: int = SEED) -> DatasetAssets:
    if dataset not in DATASETS:
        raise ValueError("unsupported dataset: {}".format(dataset))
    cache_path = (
        ROOT
        / "new_cache/new_true_ours/official_mtt/stage_b"
        / dataset
        / "new_true_ours_stage_b_{}_seed{}.npz".format(dataset, seed)
    )
    original_split_path = (
        ROOT
        / "new_cache/new_true_ours/splits"
        / dataset
        / "new_true_ours_split_{}_seed{}.npz".format(dataset, seed)
    )
    with np.load(str(cache_path), allow_pickle=False) as cache:
        source = str(cache["initializer_source"].reshape(-1)[0])
        dependency = int(cache["full_data_initializer_dependency"].reshape(-1)[0])
        test_loaded = int(cache["test_loaded"].reshape(-1)[0])
        x_mtt = np.asarray(cache["x_encoder_train"], dtype=np.float32)
        y_mtt = np.asarray(cache["y_encoder_train"], dtype=np.int64)
        encoder_sha = str(cache["encoder_sha256"].reshape(-1)[0])
        scaler_sha = str(cache["scaler_sha256"].reshape(-1)[0])
    if source != "official_mtt_only" or dependency != 0 or test_loaded != 0:
        raise ValueError("Official-MTT lineage gate failed for {}".format(cache_path))

    with np.load(str(original_split_path), allow_pickle=False) as split:
        development_ids = np.asarray(
            split["development_validation"], dtype=np.int64
        )
        # Deliberately do not access split["locked_holdout"].
    full_dir = (
        ROOT
        / "new_cache/new_public_mtt_mainline/new_full_features"
        / dataset
        / "new_seed{}".format(seed)
    )
    x_full = np.load(str(full_dir / "new_full_train_features.npy"), mmap_mode="r")
    y_full = np.load(str(full_dir / "new_full_train_labels.npy"), mmap_mode="r")
    x_dev = np.asarray(x_full[development_ids], dtype=np.float32)
    y_dev = np.asarray(y_full[development_ids], dtype=np.int64)

    num_classes = NUM_CLASSES[dataset]
    histogram = np.bincount(y_mtt, minlength=num_classes)
    if len(y_mtt) != 48 * num_classes or not np.all(histogram == 48):
        raise ValueError("{} Official MTT is not 48 samples/class".format(dataset))
    if not np.isfinite(x_mtt).all() or not np.isfinite(x_dev).all():
        raise ValueError("{} features contain non-finite values".format(dataset))
    return DatasetAssets(
        dataset=dataset,
        seed=seed,
        num_classes=num_classes,
        x_mtt=x_mtt,
        y_mtt=y_mtt,
        x_dev=x_dev,
        y_dev=y_dev,
        mtt_cache_path=cache_path,
        mtt_cache_sha256=sha256_file(cache_path),
        original_split_path=original_split_path,
        original_split_sha256=sha256_file(original_split_path),
        development_ids_sha256=sha256_array(development_ids),
        encoder_sha256=encoder_sha,
        scaler_sha256=scaler_sha,
    )


def mtt_anchor_split(
    labels: np.ndarray, num_classes: int, anchor_per_class: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    local_parts: List[np.ndarray] = []
    anchor_parts: List[np.ndarray] = []
    for class_id in range(num_classes):
        ids = np.flatnonzero(labels == class_id)
        rng = np.random.RandomState(seed * 1000003 + class_id * 1009 + 729)
        shuffled = rng.permutation(ids)
        anchor_parts.append(shuffled[:anchor_per_class])
        local_parts.append(shuffled[anchor_per_class:])
    local_ids = np.sort(np.concatenate(local_parts).astype(np.int64))
    anchor_ids = np.sort(np.concatenate(anchor_parts).astype(np.int64))
    if len(np.intersect1d(local_ids, anchor_ids)) != 0:
        raise ValueError("local and anchor pools overlap")
    if len(np.unique(np.concatenate([local_ids, anchor_ids]))) != len(labels):
        raise ValueError("local and anchor pools do not cover Official MTT")
    return local_ids, anchor_ids


def client_partitions(
    labels: np.ndarray,
    distribution: str,
    seed: int,
    num_clients: int = NUM_CLIENTS,
) -> List[np.ndarray]:
    buckets: List[List[int]] = [[] for _ in range(num_clients)]
    class_ids = sorted(int(value) for value in np.unique(labels))
    if distribution == "iid":
        for class_id in class_ids:
            ids = np.flatnonzero(labels == class_id)
            rng = np.random.RandomState(seed + class_id * 1009 + 1701)
            for client_id, part in enumerate(np.array_split(rng.permutation(ids), num_clients)):
                buckets[client_id].extend(int(value) for value in part)
    elif distribution in ("mild", "severe"):
        alpha = 0.5 if distribution == "mild" else 0.1
        for class_id in class_ids:
            ids = np.flatnonzero(labels == class_id)
            rng = np.random.RandomState(seed + class_id * 1009 + (500 if alpha == 0.5 else 100))
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            counts = rng.multinomial(len(ids), proportions)
            shuffled = rng.permutation(ids)
            cursor = 0
            for client_id, count in enumerate(counts):
                buckets[client_id].extend(
                    int(value) for value in shuffled[cursor : cursor + count]
                )
                cursor += count
    else:
        raise ValueError("unknown distribution: {}".format(distribution))

    if any(len(bucket) == 0 for bucket in buckets):
        # Deterministic repair: move one sample from the largest client.  This
        # keeps every sample unique while preserving the intended heterogeneity.
        for empty_id in [index for index, bucket in enumerate(buckets) if not bucket]:
            donor = max(range(num_clients), key=lambda index: len(buckets[index]))
            buckets[empty_id].append(buckets[donor].pop())
    result = [
        np.asarray(
            np.random.RandomState(seed + client_id * 7919 + 314).permutation(bucket),
            dtype=np.int64,
        )
        for client_id, bucket in enumerate(buckets)
    ]
    combined = np.concatenate(result)
    if len(combined) != len(labels) or len(np.unique(combined)) != len(labels):
        raise ValueError("client partition is not an exact partition")
    return result


def partition_summary(
    labels: np.ndarray, partitions: Sequence[np.ndarray], num_classes: int
) -> Dict[str, Any]:
    rows = []
    for client_id, ids in enumerate(partitions):
        histogram = np.bincount(labels[ids], minlength=num_classes)
        nonzero = histogram[histogram > 0]
        rows.append(
            {
                "client_id": client_id,
                "samples": int(len(ids)),
                "classes_present": int(np.sum(histogram > 0)),
                "empty_classes": int(np.sum(histogram == 0)),
                "minimum_nonzero_class_count": int(np.min(nonzero)),
                "maximum_class_count": int(np.max(histogram)),
                "class_histogram": histogram.astype(int).tolist(),
                "indices_sha256": sha256_array(ids),
            }
        )
    return {
        "clients": rows,
        "minimum_client_samples": min(row["samples"] for row in rows),
        "maximum_client_samples": max(row["samples"] for row in rows),
        "minimum_class_coverage": min(row["classes_present"] for row in rows),
        "maximum_class_coverage": max(row["classes_present"] for row in rows),
        "total_empty_classes": sum(row["empty_classes"] for row in rows),
    }


def initialize_shared_head(
    feature_dim: int, num_classes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed + 991)
    weight = rng.normal(0.0, 0.01, size=(feature_dim, num_classes)).astype(
        np.float32
    )
    bias = np.zeros(num_classes, dtype=np.float32)
    return weight, bias


def train_local_heads(
    x_local: np.ndarray,
    y_local: np.ndarray,
    partitions: Sequence[np.ndarray],
    num_classes: int,
    seed: int,
    local_epochs: int,
    local_lr: float,
    weight_decay: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    started = time.perf_counter()
    initial_weight, initial_bias = initialize_shared_head(
        x_local.shape[1], num_classes, seed
    )
    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    histories = []
    for client_id, ids in enumerate(partitions):
        weight = initial_weight.copy()
        bias = initial_bias.copy()
        labels = y_local[ids]
        targets = one_hot(labels, num_classes)
        x_value = x_local[ids]
        history = []
        for epoch in range(local_epochs):
            logits = np.matmul(x_value, weight) + bias
            probabilities = softmax(logits)
            error = probabilities - targets
            grad_weight = np.matmul(x_value.T, error) / float(len(ids))
            grad_weight += float(weight_decay) * weight
            grad_bias = np.mean(error, axis=0)
            weight -= float(local_lr) * grad_weight.astype(np.float32)
            bias -= float(local_lr) * grad_bias.astype(np.float32)
            if epoch in (0, local_epochs - 1):
                history.append(
                    {
                        "epoch": epoch + 1,
                        "cross_entropy": cross_entropy(probabilities, labels),
                    }
                )
        weights.append(weight.astype(np.float32))
        biases.append(bias.astype(np.float32))
        histories.append({"client_id": client_id, "history": history})
    return weights, biases, {
        "local_training_seconds": time.perf_counter() - started,
        "histories": histories,
    }


def average_head(
    weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.mean(np.stack(weights, axis=0), axis=0).astype(np.float32),
        np.mean(np.stack(biases, axis=0), axis=0).astype(np.float32),
    )


def head_probabilities(
    x_value: np.ndarray,
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    temperature: float = 1.0,
) -> np.ndarray:
    return np.stack(
        [
            softmax(np.matmul(x_value, weight) + bias, temperature)
            for weight, bias in zip(weights, biases)
        ],
        axis=0,
    )


def pairwise_client_diversity(
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    development_probabilities: np.ndarray,
) -> Dict[str, Any]:
    parameter_distances = []
    disagreements = []
    js_values = []
    client_count = len(weights)
    predictions = np.argmax(development_probabilities, axis=2)
    for left in range(client_count):
        left_vector = np.concatenate([weights[left].ravel(), biases[left].ravel()])
        for right in range(left + 1, client_count):
            right_vector = np.concatenate(
                [weights[right].ravel(), biases[right].ravel()]
            )
            denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
            cosine = 0.0 if denominator == 0.0 else float(
                np.dot(left_vector, right_vector) / denominator
            )
            parameter_distances.append(1.0 - cosine)
            disagreements.append(
                float(np.mean(predictions[left] != predictions[right]))
            )
            p = np.clip(development_probabilities[left], 1e-12, 1.0)
            q = np.clip(development_probabilities[right], 1e-12, 1.0)
            midpoint = 0.5 * (p + q)
            js = 0.5 * np.sum(p * np.log(p / midpoint), axis=1)
            js += 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
            js_values.append(float(np.mean(js)))
    return {
        "pairwise_parameter_cosine_distance": parameter_distances,
        "mean_parameter_cosine_distance": float(np.mean(parameter_distances)),
        "pairwise_prediction_disagreement": disagreements,
        "mean_prediction_disagreement": float(np.mean(disagreements)),
        "pairwise_js_divergence": js_values,
        "mean_js_divergence": float(np.mean(js_values)),
    }


def reliability_matrix(
    rule: str,
    anchor_probabilities: np.ndarray,
    anchor_labels: np.ndarray,
    local_labels: np.ndarray,
    partitions: Sequence[np.ndarray],
    num_classes: int,
) -> np.ndarray:
    client_count = anchor_probabilities.shape[0]
    result = np.zeros((client_count, num_classes), dtype=np.float64)
    for client_id in range(client_count):
        probabilities = np.clip(anchor_probabilities[client_id], 1e-12, 1.0)
        predictions = np.argmax(probabilities, axis=1)
        coverage = np.bincount(
            local_labels[partitions[client_id]], minlength=num_classes
        ).astype(np.float64)
        coverage /= max(1.0, float(np.max(coverage)))
        for class_id in range(num_classes):
            selected = anchor_labels == class_id
            if rule == "uniform":
                value = 0.0
            elif rule == "per_class_accuracy":
                value = float(np.mean(predictions[selected] == class_id))
            elif rule == "confidence":
                value = float(np.mean(probabilities[selected, class_id]))
            elif rule == "negative_entropy":
                entropy = -np.sum(
                    probabilities[selected] * np.log(probabilities[selected]), axis=1
                )
                value = -float(np.mean(entropy))
            elif rule == "calibrated_confidence":
                confidence = float(np.mean(probabilities[selected, class_id]))
                accuracy = float(np.mean(predictions[selected] == class_id))
                value = confidence - abs(confidence - accuracy)
            elif rule == "class_coverage_aware":
                confidence = float(np.mean(probabilities[selected, class_id]))
                value = confidence + 0.25 * float(coverage[class_id])
            else:
                raise ValueError("unknown reliability rule: {}".format(rule))
            result[client_id, class_id] = value
    return result


def normalize_reliability(
    reliability: np.ndarray,
    weight_temperature: float,
    maximum_client_weight: float = 1.0,
    shrinkage_to_uniform: float = 0.0,
) -> np.ndarray:
    scaled = reliability / float(weight_temperature)
    scaled = scaled - np.max(scaled, axis=0, keepdims=True)
    weights = np.exp(scaled)
    weights /= np.sum(weights, axis=0, keepdims=True)
    cap = float(maximum_client_weight)
    if cap < 1.0:
        for class_id in range(weights.shape[1]):
            values = weights[:, class_id].copy()
            result = np.zeros_like(values)
            remaining = np.ones(len(values), dtype=bool)
            mass = 1.0
            while np.any(remaining):
                proposal = values[remaining] / np.sum(values[remaining]) * mass
                over = proposal > cap
                ids = np.flatnonzero(remaining)
                if not np.any(over):
                    result[ids] = proposal
                    break
                result[ids[over]] = cap
                remaining[ids[over]] = False
                mass = 1.0 - np.sum(result)
            weights[:, class_id] = result / np.sum(result)
    shrinkage = float(shrinkage_to_uniform)
    if shrinkage > 0.0:
        uniform = np.full_like(weights, 1.0 / weights.shape[0])
        weights = (1.0 - shrinkage) * weights + shrinkage * uniform
    weights /= np.sum(weights, axis=0, keepdims=True)
    return weights.astype(np.float32)


def class_aware_teacher(
    client_probabilities: np.ndarray, class_weights: np.ndarray
) -> np.ndarray:
    # weights[i, c] scales client i's support for output class c.  This does
    # not use the unknown evaluation label and is deployable at inference.
    result = np.sum(
        client_probabilities * class_weights[:, None, :], axis=0
    )
    result = np.clip(result, 1e-12, None)
    result /= np.sum(result, axis=1, keepdims=True)
    return result.astype(np.float32)


def train_student(
    x_anchor: np.ndarray,
    y_anchor: np.ndarray,
    teacher_probabilities: np.ndarray,
    average_weight: np.ndarray,
    average_bias: np.ndarray,
    temperature: float,
    kd_lambda: float,
    average_regularization: float,
    student_steps: int,
    student_lr: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    started = time.perf_counter()
    weight = average_weight.copy()
    bias = average_bias.copy()
    targets = one_hot(y_anchor, average_bias.shape[0])
    history = []
    for step in range(student_steps):
        logits = np.matmul(x_anchor, weight) + bias
        probabilities = softmax(logits)
        probabilities_temperature = softmax(logits, temperature)
        ce_error = probabilities - targets
        kd_error = float(temperature) * (
            probabilities_temperature - teacher_probabilities
        )
        combined = (1.0 - float(kd_lambda)) * ce_error
        combined += float(kd_lambda) * kd_error
        grad_weight = np.matmul(x_anchor.T, combined) / float(len(x_anchor))
        grad_bias = np.mean(combined, axis=0)
        if average_regularization > 0.0:
            grad_weight += (
                2.0
                * float(average_regularization)
                * (weight - average_weight)
            )
            grad_bias += (
                2.0
                * float(average_regularization)
                * (bias - average_bias)
            )
        weight -= float(student_lr) * grad_weight.astype(np.float32)
        bias -= float(student_lr) * grad_bias.astype(np.float32)
        if step in (0, student_steps - 1):
            history.append(
                {
                    "step": step + 1,
                    "anchor_cross_entropy": cross_entropy(
                        probabilities, y_anchor
                    ),
                }
            )
    return weight.astype(np.float32), bias.astype(np.float32), {
        "student_kd_seconds": time.perf_counter() - started,
        "history": history,
        "distance_from_average": float(
            math.sqrt(
                float(np.sum((weight - average_weight) ** 2))
                + float(np.sum((bias - average_bias) ** 2))
            )
        ),
    }


def evaluate_head(
    x_value: np.ndarray,
    y_value: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    return classification_metrics(
        softmax(np.matmul(x_value, weight) + bias), y_value, num_classes
    )


def gain_retention(
    baseline_accuracy: float, teacher_accuracy: float, student_accuracy: float
) -> float | None:
    gain = teacher_accuracy - baseline_accuracy
    if gain <= 0.0:
        return None
    return float((student_accuracy - baseline_accuracy) / gain)

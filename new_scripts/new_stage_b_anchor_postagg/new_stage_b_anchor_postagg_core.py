#!/usr/bin/env python3
"""Core math for MTT-anchor post-aggregation initialization ablations."""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np


def stable_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_sha256(weight: np.ndarray, bias: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (weight, bias):
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float32)
    shifted = shifted - np.max(shifted, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    result = np.zeros((len(labels), num_classes), dtype=np.float32)
    result[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
    return result


def classification_metrics(
    x_value: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    probabilities = softmax(np.matmul(x_value, weight) + bias)
    predictions = np.argmax(probabilities, axis=1)
    per_class: List[float] = []
    f1_values: List[float] = []
    for class_id in range(num_classes):
        selected = labels == class_id
        per_class.append(float(np.mean(predictions[selected] == labels[selected])))
        true_positive = int(np.sum((predictions == class_id) & selected))
        false_positive = int(np.sum((predictions == class_id) & ~selected))
        false_negative = int(np.sum((predictions != class_id) & selected))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(
            0.0 if denominator == 0 else 2.0 * true_positive / denominator
        )
    chosen = probabilities[
        np.arange(len(labels)), labels.astype(np.int64)
    ]
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(np.mean(f1_values)),
        "worst_class_accuracy": float(np.min(per_class)),
        "per_class_accuracy": per_class,
        "cross_entropy": float(
            -np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))
        ),
    }


def train_anchor_ce_trajectory(
    x_anchor: np.ndarray,
    y_anchor: np.ndarray,
    x_development: np.ndarray,
    y_development: np.ndarray,
    initial_weight: np.ndarray,
    initial_bias: np.ndarray,
    steps: int,
    learning_rate: float,
    mu: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Full-batch CE with identical budgets and a complete Stage-B trajectory."""
    weight = np.asarray(initial_weight, dtype=np.float32).copy()
    bias = np.asarray(initial_bias, dtype=np.float32).copy()
    reference_weight = weight.copy()
    reference_bias = bias.copy()
    targets = one_hot(y_anchor, bias.shape[0])
    history: List[Dict[str, Any]] = []
    forward_seconds = 0.0
    backward_update_seconds = 0.0
    development_evaluation_seconds = 0.0
    started = time.perf_counter()

    def record(step: int, anchor_ce: float) -> None:
        evaluation_started = time.perf_counter()
        metrics = classification_metrics(
            x_development,
            y_development,
            weight,
            bias,
            bias.shape[0],
        )
        nonlocal development_evaluation_seconds
        development_evaluation_seconds += (
            time.perf_counter() - evaluation_started
        )
        distance = math.sqrt(
            float(np.sum((weight - reference_weight) ** 2))
            + float(np.sum((bias - reference_bias) ** 2))
        )
        history.append(
            {
                "step": int(step),
                "development_accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "worst_class_accuracy": metrics["worst_class_accuracy"],
                "anchor_cross_entropy": float(anchor_ce),
                "distance_from_initialization": float(distance),
                "elapsed_refinement_seconds": (
                    forward_seconds + backward_update_seconds
                ),
            }
        )

    initial_probabilities = softmax(np.matmul(x_anchor, weight) + bias)
    initial_chosen = initial_probabilities[
        np.arange(len(y_anchor)), y_anchor.astype(np.int64)
    ]
    record(
        0,
        float(-np.mean(np.log(np.clip(initial_chosen, 1e-12, 1.0)))),
    )

    for step in range(1, int(steps) + 1):
        forward_started = time.perf_counter()
        probabilities = softmax(np.matmul(x_anchor, weight) + bias)
        chosen = probabilities[
            np.arange(len(y_anchor)), y_anchor.astype(np.int64)
        ]
        anchor_ce = float(
            -np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))
        )
        forward_seconds += time.perf_counter() - forward_started

        update_started = time.perf_counter()
        error = probabilities - targets
        grad_weight = np.matmul(x_anchor.T, error) / float(len(x_anchor))
        grad_bias = np.mean(error, axis=0)
        if float(mu) > 0.0:
            grad_weight += (
                2.0 * float(mu) * (weight - reference_weight)
            )
            grad_bias += 2.0 * float(mu) * (bias - reference_bias)
        weight -= float(learning_rate) * grad_weight.astype(np.float32)
        bias -= float(learning_rate) * grad_bias.astype(np.float32)
        backward_update_seconds += time.perf_counter() - update_started
        record(step, anchor_ce)

    final_metrics = classification_metrics(
        x_development,
        y_development,
        weight,
        bias,
        bias.shape[0],
    )
    return weight, bias, {
        "history": history,
        "final_metrics": final_metrics,
        "anchor_forward_seconds": float(forward_seconds),
        "anchor_backward_update_seconds": float(backward_update_seconds),
        "development_evaluation_seconds": float(
            development_evaluation_seconds
        ),
        "refinement_seconds": float(
            forward_seconds + backward_update_seconds
        ),
        "wall_seconds_with_development_tracing": float(
            time.perf_counter() - started
        ),
        "distance_from_initialization": history[-1][
            "distance_from_initialization"
        ],
    }


def step_efficiency(
    history: List[Dict[str, Any]], c5_final_accuracy: float
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for ratio in (0.8, 0.9, 0.95):
        label = str(int(ratio * 100))
        target = float(c5_final_accuracy) * ratio
        reached = next(
            (
                row
                for row in history
                if row["development_accuracy"] >= target
            ),
            None,
        )
        result["target_{}".format(label)] = target
        result["steps_at_{}".format(label)] = (
            None if reached is None else reached["step"]
        )
        result["seconds_at_{}".format(label)] = (
            None
            if reached is None
            else reached["elapsed_refinement_seconds"]
        )
    return result


def select_registered_local_client() -> int:
    """Development-independent pre-registered local-head rule."""
    return 0

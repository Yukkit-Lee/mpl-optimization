#!/usr/bin/env python3
"""Core math for the seed42 Stage-B secure-anchor-refinement study."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
KD_SCRIPT_DIR = ROOT / "new_scripts/new_stage_b_kd"
if str(KD_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(KD_SCRIPT_DIR))

from new_stage_b_kd_core import (  # noqa: E402
    NUM_CLIENTS,
    SEED,
    classification_metrics,
    evaluate_head,
    head_probabilities,
    one_hot,
    sha256_array,
    sha256_file,
    softmax,
    source_commit,
)


LOG_ROOT = ROOT / "new_log/new_stage_b_anchor_refinement"
TABLE_ROOT = ROOT / "new_expTab/new_stage_b_anchor_refinement"
OUTPUT_ROOT = ROOT / "new_output/new_stage_b_anchor_refinement"
CHECKPOINT_ROOT = ROOT / "new_checkpoints/new_stage_b_anchor_refinement"
CONFIG_ROOT = ROOT / "new_configs/new_stage_b_anchor_refinement"


def disagreement_scores(client_probabilities: np.ndarray) -> Dict[str, np.ndarray]:
    """Return D1 probability variance, D2 mean JS, and D3 vote entropy."""
    probabilities = np.clip(
        np.asarray(client_probabilities, dtype=np.float64), 1e-12, 1.0
    )
    mean_probabilities = np.mean(probabilities, axis=0)
    variance = np.mean(np.var(probabilities, axis=0), axis=1)
    js = np.mean(
        np.sum(
            probabilities
            * np.log(probabilities / mean_probabilities[None, :, :]),
            axis=2,
        ),
        axis=0,
    )
    votes = np.argmax(probabilities, axis=2)
    entropy = np.zeros(probabilities.shape[1], dtype=np.float64)
    for sample_id in range(probabilities.shape[1]):
        counts = np.bincount(
            votes[:, sample_id], minlength=probabilities.shape[2]
        ).astype(np.float64)
        fractions = counts[counts > 0] / float(probabilities.shape[0])
        entropy[sample_id] = -np.sum(fractions * np.log(fractions))
    entropy /= math.log(float(probabilities.shape[0]))
    return {
        "d1_variance": variance.astype(np.float32),
        "d2_mean_js": js.astype(np.float32),
        "d3_vote_entropy": entropy.astype(np.float32),
    }


def rank_auroc(scores: np.ndarray, binary_labels: np.ndarray) -> float:
    """AUROC with average ranks for ties; returns NaN for a single class."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(binary_labels, dtype=np.int64)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    rank_sum = float(np.sum(ranks[labels == 1]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (
        positive * negative
    )


def pearson_safe(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def diagnostic_metrics(
    scores: np.ndarray,
    average_probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    development_per_class_accuracy: Sequence[float],
) -> Dict[str, Any]:
    predictions = np.argmax(average_probabilities, axis=1)
    errors = (predictions != labels).astype(np.int64)
    order = np.argsort(-scores, kind="mergesort")
    precision = {}
    for ratio in (0.1, 0.2, 0.3):
        count = max(1, int(math.ceil(len(scores) * ratio)))
        precision["top_{:02d}".format(int(ratio * 100))] = float(
            np.mean(errors[order[:count]])
        )
    fifth = max(1, int(math.ceil(len(scores) * 0.2)))
    class_disagreement = []
    for class_id in range(num_classes):
        selected = labels == class_id
        class_disagreement.append(float(np.mean(scores[selected])))
    return {
        "auroc_average_error": rank_auroc(scores, errors),
        "pearson_sample_disagreement_vs_error": pearson_safe(scores, errors),
        "precision_at_top_disagreement": precision,
        "low_20_error_rate": float(np.mean(errors[order[-fifth:]])),
        "high_20_error_rate": float(np.mean(errors[order[:fifth]])),
        "class_disagreement": class_disagreement,
        "pearson_class_disagreement_vs_weakness": pearson_safe(
            np.asarray(class_disagreement),
            1.0 - np.asarray(development_per_class_accuracy),
        ),
    }


def normalized_linear(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    low = float(np.min(scores))
    high = float(np.max(scores))
    if high - low <= 1e-15:
        return np.ones_like(scores, dtype=np.float32)
    return ((scores - low) / (high - low)).astype(np.float32)


def build_anchor_weights(
    scores: np.ndarray,
    labels: np.ndarray,
    rule: str,
    top_ratio: float = 0.3,
    soft_scale: float = 0.15,
    cap_multiplier: float | None = None,
) -> np.ndarray:
    base = normalized_linear(scores)
    threshold = float(np.quantile(scores, 1.0 - float(top_ratio)))
    if rule == "uniform":
        weights = np.ones_like(base)
    elif rule == "linear":
        weights = base
    elif rule == "threshold":
        weights = (scores >= threshold).astype(np.float32)
    elif rule == "soft_threshold":
        spread = max(float(np.std(scores)) * float(soft_scale), 1e-8)
        weights = 1.0 / (1.0 + np.exp(-(scores - threshold) / spread))
    elif rule == "class_balanced":
        weights = base.copy()
        counts = np.bincount(labels.astype(np.int64))
        weights /= counts[labels.astype(np.int64)].astype(np.float32)
        weights *= float(len(labels)) / max(float(np.sum(weights)), 1e-12)
    else:
        raise ValueError("unknown anchor weight rule: {}".format(rule))
    if cap_multiplier is not None:
        positive = weights[weights > 0.0]
        if len(positive):
            weights = np.minimum(
                weights, float(cap_multiplier) * float(np.median(positive))
            )
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("anchor weights have zero total mass")
    return np.asarray(weights, dtype=np.float32)


def train_weighted_anchor_ce(
    x_anchor: np.ndarray,
    y_anchor: np.ndarray,
    sample_weights: np.ndarray,
    average_weight: np.ndarray,
    average_bias: np.ndarray,
    steps: int,
    learning_rate: float,
    mu: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    started = time.perf_counter()
    weight = average_weight.copy()
    bias = average_bias.copy()
    targets = one_hot(y_anchor, average_bias.shape[0])
    sample_weights = np.asarray(sample_weights, dtype=np.float32)
    denominator = max(float(np.sum(sample_weights)), 1e-12)
    history = []
    for step in range(int(steps)):
        probabilities = softmax(np.matmul(x_anchor, weight) + bias)
        error = (probabilities - targets) * sample_weights[:, None]
        grad_weight = np.matmul(x_anchor.T, error) / denominator
        grad_bias = np.sum(error, axis=0) / denominator
        if mu > 0.0:
            grad_weight += 2.0 * float(mu) * (weight - average_weight)
            grad_bias += 2.0 * float(mu) * (bias - average_bias)
        weight -= float(learning_rate) * grad_weight.astype(np.float32)
        bias -= float(learning_rate) * grad_bias.astype(np.float32)
        if step in (0, int(steps) - 1):
            chosen = probabilities[
                np.arange(len(y_anchor)), y_anchor.astype(np.int64)
            ]
            history.append(
                {
                    "step": step + 1,
                    "weighted_anchor_ce": float(
                        -np.sum(sample_weights * np.log(np.clip(chosen, 1e-12, 1.0)))
                        / denominator
                    ),
                }
            )
    distance = math.sqrt(
        float(np.sum((weight - average_weight) ** 2))
        + float(np.sum((bias - average_bias) ** 2))
    )
    return weight.astype(np.float32), bias.astype(np.float32), {
        "refinement_seconds": time.perf_counter() - started,
        "history": history,
        "distance_from_average": float(distance),
        "active_anchor_ratio": float(np.mean(sample_weights > 1e-8)),
        "weighted_anchor_mass": float(np.sum(sample_weights)),
        "effective_anchor_count": int(np.sum(sample_weights > 1e-8)),
    }


#!/usr/bin/env python3
"""Core utilities for secure heterogeneity-triggered Stage-B routing."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
KD_DIR = ROOT / "new_scripts/new_stage_b_kd"
REFINE_DIR = ROOT / "new_scripts/new_stage_b_anchor_refinement"
for value in (KD_DIR, REFINE_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from new_stage_b_kd_core import (  # noqa: E402
    classification_metrics,
    sha256_array,
    sha256_file,
    softmax,
    source_commit,
)


LOG_ROOT = ROOT / "new_log/new_stage_b_heterogeneity_trigger"
TABLE_ROOT = ROOT / "new_expTab/new_stage_b_heterogeneity_trigger"
FIG_ROOT = ROOT / "new_expFig/new_stage_b_heterogeneity_trigger"
OUTPUT_ROOT = ROOT / "new_output/new_stage_b_heterogeneity_trigger"
CHECKPOINT_ROOT = ROOT / "new_checkpoints/new_stage_b_heterogeneity_trigger"
CONFIG_ROOT = ROOT / "new_configs/new_stage_b_heterogeneity_trigger"

DATASETS = ("cifar10", "cifar100")
DISTRIBUTIONS = ("iid", "mild", "severe")
METRICS = (
    "H1_mean_tv",
    "H2_entropy_deficit",
    "H3_missing_class_ratio",
)


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def class_count_matrix(
    local_labels: np.ndarray,
    partitions: Sequence[np.ndarray],
    num_classes: int,
) -> np.ndarray:
    counts = np.stack(
        [
            np.bincount(
                local_labels[np.asarray(partition, dtype=np.int64)],
                minlength=num_classes,
            )
            for partition in partitions
        ],
        axis=0,
    ).astype(np.int64)
    if np.any(np.sum(counts, axis=1) == 0):
        raise ValueError("empty client partition")
    return counts


def heterogeneity_metrics(counts: np.ndarray) -> Dict[str, float]:
    """H1-H4 from private client-by-class counts."""
    counts = np.asarray(counts, dtype=np.float64)
    totals = np.sum(counts, axis=1, keepdims=True)
    q = counts / totals
    global_q = np.sum(counts, axis=0) / float(np.sum(counts))
    tv_per_client = 0.5 * np.sum(np.abs(q - global_q[None, :]), axis=1)
    num_classes = counts.shape[1]
    entropy = -np.sum(q * np.log(np.clip(q, 1e-12, 1.0)), axis=1)
    return {
        "H1_mean_tv": float(np.mean(tv_per_client)),
        "H2_entropy_deficit": float(
            1.0 - np.mean(entropy) / math.log(float(num_classes))
        ),
        "H3_missing_class_ratio": float(np.mean(counts == 0.0)),
        "H4_max_tv": float(np.max(tv_per_client)),
    }


def route_accuracy(
    score: float, threshold: float, b0_accuracy: float, b1_accuracy: float
) -> Dict[str, Any]:
    trigger = int(float(score) >= float(threshold))
    routed = float(b1_accuracy if trigger else b0_accuracy)
    oracle = float(max(b0_accuracy, b1_accuracy))
    return {
        "trigger_bit": trigger,
        "b2_accuracy": routed,
        "oracle_accuracy": oracle,
        "oracle_regret": oracle - routed,
    }


def midpoint_thresholds(
    metric_rows: Sequence[Dict[str, Any]], metric: str
) -> List[Tuple[str, float]]:
    candidates: Dict[str, float] = {}
    for dataset in DATASETS:
        values = {
            row["distribution"]: float(row[metric])
            for row in metric_rows
            if row["dataset"] == dataset
        }
        candidates["{}_iid_mild".format(dataset)] = 0.5 * (
            values["iid"] + values["mild"]
        )
        candidates["{}_mild_severe".format(dataset)] = 0.5 * (
            values["mild"] + values["severe"]
        )
        candidates["{}_iid_severe".format(dataset)] = 0.5 * (
            values["iid"] + values["severe"]
        )
    if metric == "H3_missing_class_ratio":
        for value in (0.10, 0.25, 0.50):
            candidates["registered_{:.2f}".format(value)] = value
    unique: Dict[str, Tuple[str, float]] = {}
    for name, value in candidates.items():
        key = "{:.12g}".format(value)
        if key not in unique:
            unique[key] = (name, float(value))
    return sorted(unique.values(), key=lambda item: (item[1], item[0]))


def evaluate_single_trigger(
    metric_rows: Sequence[Dict[str, Any]],
    metric: str,
    threshold: float,
) -> Dict[str, Any]:
    routed = []
    for row in metric_rows:
        decision = route_accuracy(
            float(row[metric]),
            threshold,
            float(row["b0_accuracy"]),
            float(row["b1_accuracy"]),
        )
        routed.append(
            {
                "dataset": row["dataset"],
                "distribution": row["distribution"],
                "score": float(row[metric]),
                **decision,
            }
        )
    regrets = [row["oracle_regret"] for row in routed]
    extreme = all(
        row["trigger_bit"] == (0 if row["distribution"] == "iid" else 1)
        for row in routed
        if row["distribution"] in ("iid", "severe")
    )
    severe_recovery = all(
        row["b2_accuracy"]
        >= next(
            float(source["b0_accuracy"])
            for source in metric_rows
            if source["dataset"] == row["dataset"]
            and source["distribution"] == "severe"
        )
        + 0.05
        for row in routed
        if row["distribution"] == "severe"
    )
    iid_safety = all(
        row["b2_accuracy"]
        >= next(
            float(source["b0_accuracy"])
            for source in metric_rows
            if source["dataset"] == row["dataset"]
            and source["distribution"] == "iid"
        )
        - 0.003
        for row in routed
        if row["distribution"] == "iid"
    )
    return {
        "rule_type": "single_metric",
        "metric": metric,
        "threshold": float(threshold),
        "routed": routed,
        "mean_oracle_regret": float(np.mean(regrets)),
        "max_oracle_regret": float(np.max(regrets)),
        "gate_extreme_routing": extreme,
        "gate_oracle_regret": (
            float(np.mean(regrets)) <= 0.005
            and float(np.max(regrets)) <= 0.01
        ),
        "gate_severe_recovery": severe_recovery,
        "gate_iid_safety": iid_safety,
        "gate_cost_avoidance": all(
            row["trigger_bit"] == 0
            for row in routed
            if row["distribution"] == "iid"
        ),
        "all_mandatory_gates": (
            extreme
            and float(np.mean(regrets)) <= 0.005
            and float(np.max(regrets)) <= 0.01
            and severe_recovery
            and iid_safety
        ),
    }


def evaluate_two_metric_trigger(
    metric_rows: Sequence[Dict[str, Any]],
    tv_threshold: float,
    miss_threshold: float,
    operator: str,
) -> Dict[str, Any]:
    routed = []
    for row in metric_rows:
        left = float(row["H3_missing_class_ratio"]) >= float(miss_threshold)
        right = float(row["H1_mean_tv"]) >= float(tv_threshold)
        trigger = int(left or right) if operator == "OR" else int(left and right)
        b2 = float(row["b1_accuracy"] if trigger else row["b0_accuracy"])
        oracle = float(max(row["b0_accuracy"], row["b1_accuracy"]))
        routed.append(
            {
                "dataset": row["dataset"],
                "distribution": row["distribution"],
                "trigger_bit": trigger,
                "b2_accuracy": b2,
                "oracle_accuracy": oracle,
                "oracle_regret": oracle - b2,
            }
        )
    regrets = [row["oracle_regret"] for row in routed]
    extreme = all(
        row["trigger_bit"] == (0 if row["distribution"] == "iid" else 1)
        for row in routed
        if row["distribution"] in ("iid", "severe")
    )
    severe = all(
        row["b2_accuracy"]
        >= next(
            float(source["b0_accuracy"])
            for source in metric_rows
            if source["dataset"] == row["dataset"]
            and source["distribution"] == "severe"
        )
        + 0.05
        for row in routed
        if row["distribution"] == "severe"
    )
    iid = all(
        row["b2_accuracy"]
        >= next(
            float(source["b0_accuracy"])
            for source in metric_rows
            if source["dataset"] == row["dataset"]
            and source["distribution"] == "iid"
        )
        - 0.003
        for row in routed
        if row["distribution"] == "iid"
    )
    return {
        "rule_type": "two_metric_{}".format(operator),
        "metric": "H3_missing_class_ratio_{}_H1_mean_tv".format(operator),
        "threshold": None,
        "tv_threshold": float(tv_threshold),
        "miss_threshold": float(miss_threshold),
        "routed": routed,
        "mean_oracle_regret": float(np.mean(regrets)),
        "max_oracle_regret": float(np.max(regrets)),
        "gate_extreme_routing": extreme,
        "gate_oracle_regret": (
            float(np.mean(regrets)) <= 0.005
            and float(np.max(regrets)) <= 0.01
        ),
        "gate_severe_recovery": severe,
        "gate_iid_safety": iid,
        "gate_cost_avoidance": all(
            row["trigger_bit"] == 0
            for row in routed
            if row["distribution"] == "iid"
        ),
        "all_mandatory_gates": (
            extreme
            and float(np.mean(regrets)) <= 0.005
            and float(np.max(regrets)) <= 0.01
            and severe
            and iid
        ),
    }


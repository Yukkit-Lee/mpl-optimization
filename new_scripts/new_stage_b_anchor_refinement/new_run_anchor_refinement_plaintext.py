#!/usr/bin/env python3
"""Registered sequential plaintext search for secure anchor refinement."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KD_DIR = ROOT / "new_scripts/new_stage_b_kd"
for value in (HERE, KD_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from new_run_stage_b_kd_plaintext import (  # noqa: E402
    build_or_load_local_heads,
    evaluate_local_bundle,
)
from new_stage_b_kd_core import (  # noqa: E402
    DATASETS,
    SEED,
    atomic_json,
    head_probabilities,
    load_dataset_assets,
    sha256_array,
    sha256_file,
    softmax,
)
from new_stage_b_anchor_refinement_core import (  # noqa: E402
    CONFIG_ROOT,
    LOG_ROOT,
    TABLE_ROOT,
    build_anchor_weights,
    diagnostic_metrics,
    disagreement_scores,
    source_commit,
    train_weighted_anchor_ce,
)


REGISTRY_JSON = LOG_ROOT / "new_stage_b_anchor_refinement_smoke_summary.json"
REGISTRY_CSV = TABLE_ROOT / "new_stage_b_anchor_refinement_smoke_registry.csv"
DIAGNOSTIC_JSON = LOG_ROOT / "new_stage_b_anchor_refinement_diagnostics.json"
SPLIT_JSON = LOG_ROOT / "new_stage_b_anchor_refinement_split_manifest.json"

LOCAL_CONFIGS = {
    "cifar10": {"local_epochs": 5, "local_lr": 0.1, "weight_decay": 1e-4},
    "cifar100": {"local_epochs": 40, "local_lr": 0.2, "weight_decay": 1e-4},
}
BASE_REFINEMENT_LR = {"cifar10": 0.1, "cifar100": 0.2}


def peak_rss_gib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return float("nan")


def near_best_lowest_cost(rows: List[Dict[str, Any]], tolerance: float = 0.002):
    best = max(row["initializer_accuracy"] for row in rows)
    eligible = [
        row for row in rows if row["initializer_accuracy"] >= best - tolerance
    ]
    return min(
        eligible,
        key=lambda row: (
            row["steps"],
            row["effective_anchor_count"],
            row["refinement_seconds"],
        ),
    )


def candidate(
    assets,
    bundle,
    evaluation,
    distribution: str,
    method: str,
    metric: str,
    weight_rule: str,
    steps: int,
    lr: float,
    mu: float,
    top_ratio: float = 1.0,
    soft_scale: float = 0.15,
    cap_multiplier=None,
) -> Dict[str, Any]:
    client_anchor_probabilities = head_probabilities(
        bundle["x_anchor"], bundle["weights"], bundle["biases"]
    )
    scores = disagreement_scores(client_anchor_probabilities)[metric]
    weights = build_anchor_weights(
        scores,
        bundle["y_anchor"],
        weight_rule,
        top_ratio=top_ratio,
        soft_scale=soft_scale,
        cap_multiplier=cap_multiplier,
    )
    weight, bias, timing = train_weighted_anchor_ce(
        bundle["x_anchor"],
        bundle["y_anchor"],
        weights,
        evaluation["average_weight"],
        evaluation["average_bias"],
        steps,
        lr,
        mu,
    )
    probabilities = softmax(np.matmul(assets.x_dev, weight) + bias)
    from new_stage_b_kd_core import classification_metrics

    metrics = classification_metrics(
        probabilities, assets.y_dev, assets.num_classes
    )
    baseline = evaluation["average_metrics"]
    return {
        "dataset": assets.dataset,
        "seed": SEED,
        "backend": "numpy_plaintext_float",
        "distribution": distribution,
        "method": method,
        "disagreement_metric": metric,
        "weight_rule": weight_rule,
        "steps": int(steps),
        "learning_rate": float(lr),
        "mu": float(mu),
        "top_ratio": float(top_ratio),
        "soft_scale": float(soft_scale),
        "cap_multiplier": cap_multiplier,
        "b0_accuracy": baseline["accuracy"],
        "b0_macro_f1": baseline["macro_f1"],
        "b0_worst_class_accuracy": baseline["worst_class_accuracy"],
        "initializer_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "worst_class_accuracy": metrics["worst_class_accuracy"],
        "gain_over_b0": metrics["accuracy"] - baseline["accuracy"],
        "worst_class_gain_over_b0": (
            metrics["worst_class_accuracy"]
            - baseline["worst_class_accuracy"]
        ),
        "per_class_accuracy": metrics["per_class_accuracy"],
        "active_anchor_ratio": timing["active_anchor_ratio"],
        "weighted_anchor_mass": timing["weighted_anchor_mass"],
        "effective_anchor_count": timing["effective_anchor_count"],
        "total_anchor_count": len(bundle["y_anchor"]),
        "distance_from_average": timing["distance_from_average"],
        "refinement_seconds": timing["refinement_seconds"],
        "history": timing["history"],
        "source_commit": source_commit(),
        "initializer_source": "official_mtt_only",
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
    }


def diagnose(assets, bundle, evaluation):
    client_anchor_probabilities = head_probabilities(
        bundle["x_anchor"], bundle["weights"], bundle["biases"]
    )
    client_dev_probabilities = head_probabilities(
        assets.x_dev, bundle["weights"], bundle["biases"]
    )
    average_anchor_probabilities = softmax(
        np.matmul(bundle["x_anchor"], evaluation["average_weight"])
        + evaluation["average_bias"]
    )
    metrics = disagreement_scores(client_anchor_probabilities)
    return {
        "dataset": assets.dataset,
        "seed": SEED,
        "distribution": bundle["config"]["distribution"],
        "client_development_metrics": [
            {
                "client_id": index,
                "accuracy": float(
                    np.mean(
                        np.argmax(client_dev_probabilities[index], axis=1)
                        == assets.y_dev
                    )
                ),
            }
            for index in range(len(bundle["weights"]))
        ],
        "secure_average_development_metrics": evaluation["average_metrics"],
        "metrics": {
            name: diagnostic_metrics(
                scores,
                average_anchor_probabilities,
                bundle["y_anchor"],
                assets.num_classes,
                evaluation["average_metrics"]["per_class_accuracy"],
            )
            for name, scores in metrics.items()
        },
        "anchor_prediction_hash": sha256_array(client_anchor_probabilities),
        "source_commit": source_commit(),
        "locked_holdout_read": False,
        "official_test_read": False,
    }


def run_dataset(dataset: str):
    assets = load_dataset_assets(dataset, SEED)
    local = LOCAL_CONFIGS[dataset]
    bundle = build_or_load_local_heads(
        assets,
        "severe",
        8,
        local["local_epochs"],
        local["local_lr"],
        local["weight_decay"],
    )
    evaluation = evaluate_local_bundle(assets, bundle)
    diagnostic = diagnose(assets, bundle, evaluation)
    rows = []
    base_lr = BASE_REFINEMENT_LR[dataset]
    common = dict(
        assets=assets,
        bundle=bundle,
        evaluation=evaluation,
        distribution="severe",
        metric="d2_mean_js",
        weight_rule="uniform",
        lr=base_lr,
        mu=0.0,
    )

    step_rows = []
    for steps in (5, 10, 20, 40):
        row = candidate(method="B1_uniform_s1_steps", steps=steps, **common)
        rows.append(row)
        step_rows.append(row)
    selected_steps = near_best_lowest_cost(step_rows)["steps"]

    lr_rows = []
    for multiplier in (0.25, 0.5, 1.0):
        row = candidate(
            method="B1_uniform_s1_lr",
            steps=selected_steps,
            **{**common, "lr": base_lr * multiplier},
        )
        rows.append(row)
        lr_rows.append(row)
    selected_lr = max(
        lr_rows,
        key=lambda row: (
            row["initializer_accuracy"],
            row["worst_class_accuracy"],
            -row["learning_rate"],
        ),
    )["learning_rate"]

    mu_rows = []
    for mu in (0.0, 1e-4, 1e-3):
        row = candidate(
            method="B1_uniform_s1_mu",
            steps=selected_steps,
            **{**common, "lr": selected_lr, "mu": mu},
        )
        rows.append(row)
        mu_rows.append(row)
    b1 = near_best_lowest_cost(mu_rows)
    selected_mu = b1["mu"]

    metric_rows = []
    for metric in ("d1_variance", "d2_mean_js", "d3_vote_entropy"):
        row = candidate(
            assets,
            bundle,
            evaluation,
            "severe",
            "B2_s2_metric",
            metric,
            "linear",
            selected_steps,
            selected_lr,
            selected_mu,
        )
        rows.append(row)
        metric_rows.append(row)
    best_metric = max(
        metric_rows,
        key=lambda row: (
            row["initializer_accuracy"],
            row["worst_class_accuracy"],
            row["macro_f1"],
        ),
    )["disagreement_metric"]

    weight_rows = []
    for rule in ("linear", "class_balanced"):
        row = candidate(
            assets,
            bundle,
            evaluation,
            "severe",
            "B2_s3_weight",
            best_metric,
            rule,
            selected_steps,
            selected_lr,
            selected_mu,
        )
        rows.append(row)
        weight_rows.append(row)
    for rule in ("threshold", "soft_threshold"):
        for ratio in (0.1, 0.2, 0.3, 0.5):
            row = candidate(
                assets,
                bundle,
                evaluation,
                "severe",
                "B3_s3_weight",
                best_metric,
                rule,
                selected_steps,
                selected_lr,
                selected_mu,
                top_ratio=ratio,
            )
            rows.append(row)
            weight_rows.append(row)
    selected_weight = max(
        weight_rows,
        key=lambda row: (
            row["initializer_accuracy"],
            row["worst_class_accuracy"],
            row["macro_f1"],
            -row["effective_anchor_count"],
        ),
    )

    cost_rows = []
    for ratio in (1.0, 0.5, 0.3, 0.2, 0.1):
        rule = selected_weight["weight_rule"]
        if ratio < 1.0:
            rule = "threshold"
        row = candidate(
            assets,
            bundle,
            evaluation,
            "severe",
            "B3_s4_cost",
            best_metric,
            rule,
            selected_steps,
            selected_lr,
            selected_mu,
            top_ratio=ratio,
        )
        rows.append(row)
        cost_rows.append(row)

    # Required diagnostic rescue families are bounded and only run if needed.
    best_initial = max(weight_rows + cost_rows, key=lambda row: row["initializer_accuracy"])
    if best_initial["initializer_accuracy"] < b1["initializer_accuracy"] + 0.01:
        for cap in (2.0, 4.0, 8.0):
            for lr_mult in (0.25, 0.5):
                for mu in (1e-4, 1e-3, 1e-2):
                    row = candidate(
                        assets,
                        bundle,
                        evaluation,
                        "severe",
                        "diagnostic_capped_weight",
                        best_metric,
                        selected_weight["weight_rule"],
                        selected_steps,
                        base_lr * lr_mult,
                        mu,
                        top_ratio=selected_weight["top_ratio"],
                        cap_multiplier=cap,
                    )
                    rows.append(row)

    all_method_rows = [
        row for row in rows if row["method"] != "B1_uniform_s1_steps"
        and row["weight_rule"] != "uniform"
    ]
    best_method = max(
        all_method_rows,
        key=lambda row: (
            row["initializer_accuracy"],
            row["worst_class_accuracy"],
            row["macro_f1"],
            -row["effective_anchor_count"],
        ),
    )
    cost_eligible = [
        row
        for row in cost_rows
        if row["initializer_accuracy"] >= b1["initializer_accuracy"] - 0.002
        and row["effective_anchor_count"] <= 0.7 * b1["effective_anchor_count"]
    ]
    best_cost = (
        max(
            cost_eligible,
            key=lambda row: (
                -row["effective_anchor_count"],
                row["initializer_accuracy"],
            ),
        )
        if cost_eligible
        else None
    )

    # IID safety uses only the selected severe method, with no retuning.
    iid_bundle = build_or_load_local_heads(
        assets,
        "iid",
        8,
        local["local_epochs"],
        local["local_lr"],
        local["weight_decay"],
    )
    iid_evaluation = evaluate_local_bundle(assets, iid_bundle)
    iid_b1 = candidate(
        assets,
        iid_bundle,
        iid_evaluation,
        "iid",
        "B1_iid_safety",
        "d2_mean_js",
        "uniform",
        b1["steps"],
        b1["learning_rate"],
        b1["mu"],
    )
    iid_best = candidate(
        assets,
        iid_bundle,
        iid_evaluation,
        "iid",
        "B2B3_iid_safety",
        best_method["disagreement_metric"],
        best_method["weight_rule"],
        best_method["steps"],
        best_method["learning_rate"],
        best_method["mu"],
        top_ratio=best_method["top_ratio"],
        soft_scale=best_method["soft_scale"],
        cap_multiplier=best_method["cap_multiplier"],
    )
    rows.extend([iid_b1, iid_best])

    accuracy_gain = best_method["initializer_accuracy"] - b1["initializer_accuracy"]
    worst_gain = (
        best_method["worst_class_accuracy"] - b1["worst_class_accuracy"]
    )
    utility_pass = accuracy_gain >= 0.01 or worst_gain >= 0.02
    cost_pass = best_cost is not None
    iid_delta = iid_best["initializer_accuracy"] - iid_b1["initializer_accuracy"]
    auroc_best = max(
        item["auroc_average_error"] for item in diagnostic["metrics"].values()
    )
    return {
        "dataset": dataset,
        "local_config": bundle["config"],
        "local_checkpoint_path": str(bundle["checkpoint"]),
        "local_checkpoint_sha256": bundle["checkpoint_sha256"],
        "local_ids_sha256": sha256_array(bundle["local_ids"]),
        "anchor_ids_sha256": sha256_array(bundle["anchor_ids"]),
        "local_class_histogram": np.bincount(
            bundle["y_local"], minlength=assets.num_classes
        ).tolist(),
        "anchor_class_histogram": np.bincount(
            bundle["y_anchor"], minlength=assets.num_classes
        ).tolist(),
        "local_anchor_intersection": int(
            len(np.intersect1d(bundle["local_ids"], bundle["anchor_ids"]))
        ),
        "local_anchor_union": int(
            len(np.union1d(bundle["local_ids"], bundle["anchor_ids"]))
        ),
        "diagnostic": diagnostic,
        "rows": rows,
        "selected": {
            "b0": evaluation["average_metrics"],
            "b1": b1,
            "best_utility": best_method,
            "best_cost": best_cost,
            "iid_b1": iid_b1,
            "iid_best": iid_best,
        },
        "gates": {
            "accuracy_gain_vs_b1": accuracy_gain,
            "worst_class_gain_vs_b1": worst_gain,
            "utility_pass": utility_pass,
            "cost_pass": cost_pass,
            "best_disagreement_auroc": auroc_best,
            "disagreement_predictive_preferred": auroc_best > 0.60,
            "iid_delta_vs_b1": iid_delta,
            "iid_safety_pass": iid_delta >= -0.005,
            "lineage_pass": True,
            "plaintext_pass": (utility_pass or cost_pass)
            and iid_delta >= -0.005,
        },
    }


def write_csv(results):
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "seed", "backend", "distribution", "method",
        "disagreement_metric", "weight_rule", "steps", "learning_rate", "mu",
        "top_ratio", "soft_scale", "cap_multiplier", "b0_accuracy",
        "initializer_accuracy", "macro_f1", "worst_class_accuracy",
        "gain_over_b0", "worst_class_gain_over_b0", "active_anchor_ratio",
        "weighted_anchor_mass", "effective_anchor_count", "total_anchor_count",
        "distance_from_average", "refinement_seconds", "source_commit",
        "initializer_source", "full_data_initializer_dependency",
        "locked_holdout_read", "official_test_read",
    ]
    with REGISTRY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            for row in result["rows"]:
                writer.writerow({key: row.get(key, "") for key in fields})


def main():
    started = time.time()
    results = [run_dataset(dataset) for dataset in DATASETS]
    source = source_commit()
    diagnostics = {
        "schema": "new_stage_b_anchor_refinement_diagnostics_v1",
        "status": "COMPLETE",
        "source_commit": source,
        "datasets": [result["diagnostic"] for result in results],
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    split_manifest = {
        "schema": "new_stage_b_anchor_refinement_split_manifest_v1",
        "source_commit": source,
        "seed_policy": [SEED],
        "anchor_per_class": 8,
        "local_per_class": 40,
        "dependencies": [
            {
                "dataset": result["dataset"],
                "local_config": result["local_config"],
                "local_checkpoint_path": result["local_checkpoint_path"],
                "local_checkpoint_sha256": result["local_checkpoint_sha256"],
                "local_ids_sha256": result["local_ids_sha256"],
                "anchor_ids_sha256": result["anchor_ids_sha256"],
                "local_class_histogram": result["local_class_histogram"],
                "anchor_class_histogram": result["anchor_class_histogram"],
                "local_anchor_intersection": result["local_anchor_intersection"],
                "local_anchor_union": result["local_anchor_union"],
            }
            for result in results
        ],
        "initializer_source": "official_mtt_only",
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    payload = {
        "schema": "new_stage_b_anchor_refinement_smoke_summary_v1",
        "status": "COMPLETE",
        "source_commit": source,
        "seed_policy": [SEED],
        "datasets": ["cifar10", "cifar100"],
        "search_order": ["S1_steps", "S1_lr", "S1_mu", "S2_metric", "S3_weight", "S4_cost", "S5_iid"],
        "results": results,
        "all_plaintext_gates_pass": all(
            result["gates"]["plaintext_pass"] for result in results
        ),
        "runtime_seconds": time.time() - started,
        "peak_rss_gib": peak_rss_gib(),
        "initializer_source": "official_mtt_only",
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
        "next_phase": "TFE_ABY3" if all(
            result["gates"]["plaintext_pass"] for result in results
        ) else "NEGATIVE_RESULT_STOP_BEFORE_TFE",
    }
    atomic_json(DIAGNOSTIC_JSON, diagnostics)
    atomic_json(SPLIT_JSON, split_manifest)
    atomic_json(REGISTRY_JSON, payload)
    write_csv(results)
    print(json.dumps({
        result["dataset"]: {
            "b0": result["selected"]["b0"]["accuracy"],
            "b1": result["selected"]["b1"]["initializer_accuracy"],
            "best": result["selected"]["best_utility"]["initializer_accuracy"],
            "gates": result["gates"],
        }
        for result in results
    }, indent=2))


if __name__ == "__main__":
    main()

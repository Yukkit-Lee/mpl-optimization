#!/usr/bin/env python3
"""S0 crossover, H1-H4 diagnosis, and registered trigger screening."""

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
REFINE_DIR = ROOT / "new_scripts/new_stage_b_anchor_refinement"
for value in (HERE, KD_DIR, REFINE_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from new_run_stage_b_kd_plaintext import (  # noqa: E402
    build_or_load_local_heads,
    evaluate_local_bundle,
)
from new_stage_b_kd_core import (  # noqa: E402
    atomic_json,
    classification_metrics,
    load_dataset_assets,
    sha256_array,
    sha256_file,
    softmax,
    source_commit,
)
from new_stage_b_anchor_refinement_core import (  # noqa: E402
    train_weighted_anchor_ce,
)
from new_stage_b_heterogeneity_trigger_core import (  # noqa: E402
    CONFIG_ROOT,
    DATASETS,
    DISTRIBUTIONS,
    LOG_ROOT,
    METRICS,
    TABLE_ROOT,
    class_count_matrix,
    evaluate_single_trigger,
    evaluate_two_metric_trigger,
    heterogeneity_metrics,
    midpoint_thresholds,
    sha256_json,
)


OLD_SUMMARY_PATH = (
    ROOT
    / "new_log/new_stage_b_anchor_refinement/"
    "new_stage_b_anchor_refinement_smoke_summary.json"
)
OLD_SPLIT_PATH = (
    ROOT
    / "new_log/new_stage_b_anchor_refinement/"
    "new_stage_b_anchor_refinement_split_manifest.json"
)
POLICY_PATH = (
    CONFIG_ROOT / "new_stage_b_heterogeneity_trigger_policy.json"
)
SPLIT_PATH = LOG_ROOT / "new_stage_b_heterogeneity_trigger_split_manifest.json"
REUSE_PATH = LOG_ROOT / "new_stage_b_heterogeneity_trigger_reuse_audit.json"
SUMMARY_PATH = LOG_ROOT / "new_stage_b_heterogeneity_smoke_summary.json"
METRIC_CSV = TABLE_ROOT / "new_stage_b_heterogeneity_metric_table.csv"
THRESHOLD_CSV = TABLE_ROOT / "new_stage_b_heterogeneity_threshold_screen.csv"


LOCAL_CONFIGS = {
    "cifar10": {"local_epochs": 5, "local_lr": 0.1, "weight_decay": 1e-4},
    "cifar100": {"local_epochs": 40, "local_lr": 0.2, "weight_decay": 1e-4},
}


def peak_rss_gib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return float("nan")


def write_csv(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def old_rows(old_summary: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    result = next(
        row for row in old_summary["results"] if row["dataset"] == dataset
    )
    return {
        "result": result,
        "severe": result["selected"]["b1"],
        "iid": result["selected"]["iid_b1"],
    }


def refine_mild(assets, bundle, evaluation, config):
    sample_weights = np.ones(len(bundle["y_anchor"]), dtype=np.float32)
    weight, bias, timing = train_weighted_anchor_ce(
        bundle["x_anchor"],
        bundle["y_anchor"],
        sample_weights,
        evaluation["average_weight"],
        evaluation["average_bias"],
        config["steps"],
        config["learning_rate"],
        config["mu"],
    )
    metrics = classification_metrics(
        softmax(np.matmul(assets.x_dev, weight) + bias),
        assets.y_dev,
        assets.num_classes,
    )
    return metrics, timing


def main() -> None:
    started = time.time()
    source = source_commit()
    old_summary = json.loads(OLD_SUMMARY_PATH.read_text(encoding="utf-8"))
    old_split = json.loads(OLD_SPLIT_PATH.read_text(encoding="utf-8"))
    if old_summary["status"] != "COMPLETE":
        raise ValueError("old exact-valid summary is incomplete")
    metric_rows: List[Dict[str, Any]] = []
    reuse_rows: List[Dict[str, Any]] = []
    split_dependencies: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        assets = load_dataset_assets(dataset, 42)
        prior = old_rows(old_summary, dataset)
        selected_config = prior["severe"]
        b1_config = {
            "steps": selected_config["steps"],
            "learning_rate": selected_config["learning_rate"],
            "mu": selected_config["mu"],
        }
        dataset_results = []
        for distribution in DISTRIBUTIONS:
            local = LOCAL_CONFIGS[dataset]
            bundle = build_or_load_local_heads(
                assets,
                distribution,
                8,
                local["local_epochs"],
                local["local_lr"],
                local["weight_decay"],
            )
            evaluation = evaluate_local_bundle(assets, bundle)
            b0 = evaluation["average_metrics"]
            counts = class_count_matrix(
                bundle["y_local"], bundle["partitions"], assets.num_classes
            )
            metric_started = time.perf_counter()
            h_values = heterogeneity_metrics(counts)
            metric_seconds = time.perf_counter() - metric_started
            if distribution in ("iid", "severe"):
                reused = prior[distribution]
                if abs(float(reused["b0_accuracy"]) - b0["accuracy"]) > 1e-12:
                    raise ValueError(
                        "{} {} B0 reuse mismatch".format(dataset, distribution)
                    )
                b1 = {
                    "accuracy": reused["initializer_accuracy"],
                    "macro_f1": reused["macro_f1"],
                    "worst_class_accuracy": reused["worst_class_accuracy"],
                    "per_class_accuracy": reused["per_class_accuracy"],
                }
                refinement_seconds = reused["refinement_seconds"]
                reuse_status = "REUSED_EXACT_VALID"
                source_artifact = str(OLD_SUMMARY_PATH.relative_to(ROOT))
            else:
                b1, timing = refine_mild(
                    assets, bundle, evaluation, b1_config
                )
                refinement_seconds = timing["refinement_seconds"]
                reuse_status = (
                    "COMPUTED_MISSING_MILD_B1_WITH_CACHED_LOCAL_HEAD"
                )
                source_artifact = ""
            model_config = {
                "dataset": dataset,
                "distribution": distribution,
                "local_config": bundle["config"],
                "b1_config": b1_config,
            }
            row = {
                "dataset": dataset,
                "seed": 42,
                "distribution": distribution,
                **h_values,
                "b0_accuracy": b0["accuracy"],
                "b1_accuracy": b1["accuracy"],
                "b1_minus_b0_accuracy": b1["accuracy"] - b0["accuracy"],
                "b0_macro_f1": b0["macro_f1"],
                "b1_macro_f1": b1["macro_f1"],
                "b0_worst_class_accuracy": b0["worst_class_accuracy"],
                "b1_worst_class_accuracy": b1["worst_class_accuracy"],
                "b1_minus_b0_worst_class": (
                    b1["worst_class_accuracy"]
                    - b0["worst_class_accuracy"]
                ),
                "b0_per_class_accuracy": b0["per_class_accuracy"],
                "b1_per_class_accuracy": b1["per_class_accuracy"],
                "local_training_seconds": bundle["local_training_seconds"],
                "secure_average_proxy_seconds": evaluation[
                    "anchor_inference_seconds"
                ],
                "anchor_refinement_seconds": refinement_seconds,
                "heterogeneity_plaintext_seconds": metric_seconds,
                "peak_rss_gib": peak_rss_gib(),
                "local_checkpoint_cache_hit": bundle["cache_hit"],
                "local_checkpoint_path": bundle["checkpoint"],
                "model_sha256": bundle["checkpoint_sha256"],
                "model_config_sha256": sha256_json(model_config),
                "partition_hashes_sha256": sha256_json(
                    [sha256_array(partition) for partition in bundle["partitions"]]
                ),
                "private_histogram_sha256": sha256_array(counts),
                "raw_histogram_revealed": False,
                "reuse_status": reuse_status,
                "source_artifact": source_artifact,
                "source_git_commit": source,
                "official_mtt_only": True,
                "full_data_initializer_dependency": False,
                "locked_holdout_read": False,
                "official_test_read": False,
            }
            metric_rows.append(row)
            dataset_results.append(row)
            reuse_rows.append(
                {
                    "dataset": dataset,
                    "distribution": distribution,
                    "reuse_status": reuse_status,
                    "source_artifact": source_artifact,
                    "semantic_match": True,
                    "model_sha256": bundle["checkpoint_sha256"],
                    "model_config_sha256": row["model_config_sha256"],
                }
            )
            split_dependencies.append(
                {
                    "dataset": dataset,
                    "distribution": distribution,
                    "local_ids_sha256": sha256_array(bundle["local_ids"]),
                    "anchor_ids_sha256": sha256_array(bundle["anchor_ids"]),
                    "partition_hashes_sha256": row[
                        "partition_hashes_sha256"
                    ],
                    "local_checkpoint_path": bundle["checkpoint"],
                    "local_checkpoint_sha256": bundle[
                        "checkpoint_sha256"
                    ],
                    "local_anchor_intersection": int(
                        len(
                            np.intersect1d(
                                bundle["local_ids"], bundle["anchor_ids"]
                            )
                        )
                    ),
                    "local_anchor_union": int(
                        len(
                            np.union1d(
                                bundle["local_ids"], bundle["anchor_ids"]
                            )
                        )
                    ),
                }
            )
        detailed_results.append(
            {"dataset": dataset, "rows": dataset_results, "b1_config": b1_config}
        )

    split_payload = {
        "schema": "new_stage_b_heterogeneity_trigger_split_manifest_v1",
        "source_git_commit": source,
        "seed": 42,
        "anchor_per_class": 8,
        "local_per_class": 40,
        "dependencies": split_dependencies,
        "upstream_split_manifest_path": str(OLD_SPLIT_PATH.relative_to(ROOT)),
        "upstream_split_manifest_sha256": sha256_file(OLD_SPLIT_PATH),
        "official_mtt_only": True,
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    atomic_json(SPLIT_PATH, split_payload)
    split_sha = sha256_file(SPLIT_PATH)
    common_lineage = {
        "source_git_commit": source,
        "split_manifest_path": str(SPLIT_PATH.relative_to(ROOT)),
        "split_manifest_sha256": split_sha,
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": sha256_file(POLICY_PATH),
        "official_mtt_only": True,
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    for row in metric_rows:
        row.update(common_lineage)

    threshold_results = []
    for metric in METRICS:
        for name, threshold in midpoint_thresholds(metric_rows, metric):
            result = evaluate_single_trigger(metric_rows, metric, threshold)
            result["candidate_name"] = name
            threshold_results.append(result)
    if not any(row["all_mandatory_gates"] for row in threshold_results):
        tv_threshold = float(
            np.mean(
                [
                    0.5
                    * (
                        next(
                            row["H1_mean_tv"]
                            for row in metric_rows
                            if row["dataset"] == dataset
                            and row["distribution"] == "iid"
                        )
                        + next(
                            row["H1_mean_tv"]
                            for row in metric_rows
                            if row["dataset"] == dataset
                            and row["distribution"] == "severe"
                        )
                    )
                    for dataset in DATASETS
                ]
            )
        )
        miss_threshold = float(
            np.mean(
                [
                    0.5
                    * (
                        next(
                            row["H3_missing_class_ratio"]
                            for row in metric_rows
                            if row["dataset"] == dataset
                            and row["distribution"] == "iid"
                        )
                        + next(
                            row["H3_missing_class_ratio"]
                            for row in metric_rows
                            if row["dataset"] == dataset
                            and row["distribution"] == "severe"
                        )
                    )
                    for dataset in DATASETS
                ]
            )
        )
        for operator in ("OR", "AND"):
            result = evaluate_two_metric_trigger(
                metric_rows, tv_threshold, miss_threshold, operator
            )
            result["candidate_name"] = "registered_two_metric_{}".format(
                operator
            )
            threshold_results.append(result)

    best = min(
        threshold_results,
        key=lambda row: (
            not row["all_mandatory_gates"],
            row["max_oracle_regret"],
            row["mean_oracle_regret"],
            not row["gate_extreme_routing"],
            row["rule_type"],
        ),
    )
    crossover = {
        dataset: {
            distribution: next(
                {
                    "b0_accuracy": row["b0_accuracy"],
                    "b1_accuracy": row["b1_accuracy"],
                    "b1_minus_b0_accuracy": row[
                        "b1_minus_b0_accuracy"
                    ],
                }
                for row in metric_rows
                if row["dataset"] == dataset
                and row["distribution"] == distribution
            )
            for distribution in DISTRIBUTIONS
        }
        for dataset in DATASETS
    }
    crossover_pass = all(
        crossover[dataset]["iid"]["b0_accuracy"]
        >= crossover[dataset]["iid"]["b1_accuracy"]
        and crossover[dataset]["severe"]["b1_accuracy"]
        > crossover[dataset]["severe"]["b0_accuracy"]
        for dataset in DATASETS
    )
    monotonic = {
        metric: all(
            next(
                row[metric]
                for row in metric_rows
                if row["dataset"] == dataset
                and row["distribution"] == "iid"
            )
            <= next(
                row[metric]
                for row in metric_rows
                if row["dataset"] == dataset
                and row["distribution"] == "mild"
            )
            <= next(
                row[metric]
                for row in metric_rows
                if row["dataset"] == dataset
                and row["distribution"] == "severe"
            )
            for dataset in DATASETS
        )
        for metric in METRICS
    }
    smoke_pass = crossover_pass and any(
        row["all_mandatory_gates"] for row in threshold_results
    )
    reuse_payload = {
        "schema": "new_stage_b_heterogeneity_trigger_reuse_audit_v1",
        "status": "COMPLETE",
        "rows": reuse_rows,
        "exact_valid_reused": sum(
            row["reuse_status"] == "REUSED_EXACT_VALID"
            for row in reuse_rows
        ),
        "new_missing_cases_computed": sum(
            row["reuse_status"]
            != "REUSED_EXACT_VALID"
            for row in reuse_rows
        ),
        **common_lineage,
    }
    atomic_json(REUSE_PATH, reuse_payload)
    summary = {
        "schema": "new_stage_b_heterogeneity_smoke_summary_v1",
        "status": "COMPLETE",
        "crossover": crossover,
        "crossover_pass": crossover_pass,
        "metric_monotonicity": monotonic,
        "metric_rows": metric_rows,
        "threshold_results": threshold_results,
        "selected_candidate": best,
        "smoke_pass": smoke_pass,
        "partition_confirmation_status": (
            "PENDING_AFTER_SMOKE_PASS"
            if smoke_pass
            else "SKIPPED_SMOKE_GATE"
        ),
        "tfe_status": (
            "PENDING_AFTER_FREEZE" if smoke_pass else "SKIPPED_SMOKE_GATE"
        ),
        "next_phase": (
            "PARTITION_CONFIRMATION_AND_FREEZE"
            if smoke_pass
            else "NEGATIVE_STOP_S0_OR_TRIGGER_GATE"
        ),
        "runtime_seconds": time.time() - started,
        "peak_rss_gib": peak_rss_gib(),
        "raw_histograms_revealed": False,
        "trigger_bit_revealed_in_plaintext_diagnosis": True,
        **common_lineage,
    }
    atomic_json(SUMMARY_PATH, summary)

    metric_fields = [
        "dataset",
        "seed",
        "distribution",
        "H1_mean_tv",
        "H2_entropy_deficit",
        "H3_missing_class_ratio",
        "H4_max_tv",
        "b0_accuracy",
        "b1_accuracy",
        "b1_minus_b0_accuracy",
        "b0_worst_class_accuracy",
        "b1_worst_class_accuracy",
        "b1_minus_b0_worst_class",
        "reuse_status",
        "model_sha256",
        "model_config_sha256",
        "source_git_commit",
        "split_manifest_path",
        "split_manifest_sha256",
        "official_mtt_only",
        "full_data_initializer_dependency",
        "locked_holdout_read",
        "official_test_read",
    ]
    write_csv(METRIC_CSV, metric_fields, metric_rows)
    screen_rows = []
    for result in threshold_results:
        for routed in result["routed"]:
            screen_rows.append(
                {
                    "candidate_name": result["candidate_name"],
                    "rule_type": result["rule_type"],
                    "metric": result["metric"],
                    "threshold": result.get("threshold"),
                    "tv_threshold": result.get("tv_threshold"),
                    "miss_threshold": result.get("miss_threshold"),
                    **routed,
                    "mean_oracle_regret": result[
                        "mean_oracle_regret"
                    ],
                    "max_oracle_regret": result["max_oracle_regret"],
                    "gate_extreme_routing": result[
                        "gate_extreme_routing"
                    ],
                    "gate_oracle_regret": result["gate_oracle_regret"],
                    "gate_severe_recovery": result[
                        "gate_severe_recovery"
                    ],
                    "gate_iid_safety": result["gate_iid_safety"],
                    "all_mandatory_gates": result[
                        "all_mandatory_gates"
                    ],
                    **common_lineage,
                }
            )
    threshold_fields = list(screen_rows[0])
    write_csv(THRESHOLD_CSV, threshold_fields, screen_rows)
    print(
        json.dumps(
            {
                "crossover": crossover,
                "monotonicity": monotonic,
                "selected": {
                    key: best.get(key)
                    for key in (
                        "candidate_name",
                        "metric",
                        "threshold",
                        "mean_oracle_regret",
                        "max_oracle_regret",
                        "gate_extreme_routing",
                        "all_mandatory_gates",
                    )
                },
                "smoke_pass": smoke_pass,
                "next_phase": summary["next_phase"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


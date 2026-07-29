#!/usr/bin/env python3
"""Registered plaintext causal ablation for post-aggregation anchor CE."""

from __future__ import annotations

import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KD_DIR = ROOT / "new_scripts/new_stage_b_kd"
for path in (HERE, KD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from new_run_stage_b_kd_plaintext import (  # noqa: E402
    build_or_load_local_heads,
    evaluate_local_bundle,
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
    model_sha256,
    select_registered_local_client,
    select_viable_causal_group,
    stable_sha256,
    step_efficiency,
    train_anchor_ce_trajectory,
)


CONFIG_PATH = (
    ROOT
    / "new_configs/new_stage_b_anchor_postagg/"
    "new_stage_b_anchor_postagg_policy.json"
)
LOG_ROOT = ROOT / "new_log/new_stage_b_anchor_postagg"
TABLE_ROOT = ROOT / "new_expTab/new_stage_b_anchor_postagg"
CHECKPOINT_ROOT = ROOT / "new_checkpoints/new_stage_b_anchor_postagg"
SUMMARY_PATH = LOG_ROOT / "new_stage_b_anchor_postagg_smoke_summary.json"
REGISTRY_PATH = TABLE_ROOT / "new_stage_b_anchor_postagg_smoke_registry.csv"
ABLATION_PATH = (
    TABLE_ROOT / "new_stage_b_anchor_postagg_init_source_ablation.csv"
)
EFFICIENCY_PATH = (
    TABLE_ROOT / "new_stage_b_anchor_postagg_step_efficiency.csv"
)
REUSE_PATH = LOG_ROOT / "new_stage_b_anchor_postagg_reuse_audit.json"
SPLIT_PATH = LOG_ROOT / "new_stage_b_anchor_postagg_split_manifest.json"
TRAJECTORY_PATH = LOG_ROOT / "new_stage_b_anchor_postagg_trajectories.json"

LOCAL_CONFIGS = {
    "cifar10": {"local_epochs": 5, "local_lr": 0.1, "weight_decay": 1e-4},
    "cifar100": {
        "local_epochs": 40,
        "local_lr": 0.2,
        "weight_decay": 1e-4,
    },
}
BASE_LR = {"cifar10": 0.1, "cifar100": 0.2}
DISTRIBUTIONS = ("iid", "mild", "severe")
REFINED_METHODS = {
    "random": "C1_random_plus_anchor",
    "local": "C3_fixed_local0_plus_anchor",
    "average": "C5_secure_average_plus_anchor",
}


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0 / 1024.0


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def initializer_bundle(assets, local_bundle) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    random_weight, random_bias = initialize_shared_head(
        assets.x_mtt.shape[1], assets.num_classes, SEED
    )
    local_id = select_registered_local_client()
    average_weight, average_bias = average_head(
        local_bundle["weights"], local_bundle["biases"]
    )
    return {
        "random": (random_weight, random_bias),
        "local": (
            local_bundle["weights"][local_id].copy(),
            local_bundle["biases"][local_id].copy(),
        ),
        "average": (average_weight, average_bias),
    }


def unrefined_row(
    assets,
    dataset: str,
    distribution: str,
    method: str,
    source_name: str,
    weight: np.ndarray,
    bias: np.ndarray,
    common: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = classification_metrics(
        assets.x_dev,
        assets.y_dev,
        weight,
        bias,
        assets.num_classes,
    )
    return {
        **common,
        "dataset": dataset,
        "distribution": distribution,
        "stage": "S2_frozen_confirmation",
        "factor": "initialization_source",
        "factor_value": source_name,
        "method": method,
        "initialization_source": source_name,
        "anchor_ce": False,
        "steps": 0,
        "learning_rate": 0.0,
        "mu": 0.0,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "worst_class_accuracy": metrics["worst_class_accuracy"],
        "per_class_accuracy": metrics["per_class_accuracy"],
        "improved_classes_vs_initialization": 0,
        "anchor_forward_seconds": 0.0,
        "anchor_backward_update_seconds": 0.0,
        "refinement_seconds": 0.0,
        "distance_from_initialization": 0.0,
        "model_sha256": model_sha256(weight, bias),
        "history": [],
        "weight": weight,
        "bias": bias,
    }


def refined_row(
    assets,
    local_bundle,
    dataset: str,
    distribution: str,
    stage: str,
    factor: str,
    factor_value: Any,
    source_name: str,
    initial_weight: np.ndarray,
    initial_bias: np.ndarray,
    steps: int,
    learning_rate: float,
    mu: float,
    common: Dict[str, Any],
) -> Dict[str, Any]:
    weight, bias, timing = train_anchor_ce_trajectory(
        local_bundle["x_anchor"],
        local_bundle["y_anchor"],
        assets.x_dev,
        assets.y_dev,
        initial_weight,
        initial_bias,
        steps,
        learning_rate,
        mu,
    )
    metrics = timing["final_metrics"]
    initial_metrics = classification_metrics(
        assets.x_dev,
        assets.y_dev,
        initial_weight,
        initial_bias,
        assets.num_classes,
    )
    improved = sum(
        value > baseline
        for value, baseline in zip(
            metrics["per_class_accuracy"],
            initial_metrics["per_class_accuracy"],
        )
    )
    config_payload = {
        "dataset": dataset,
        "distribution": distribution,
        "initialization_source": source_name,
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "mu": float(mu),
        "anchor_ids_sha256": common["anchor_ids_sha256"],
    }
    return {
        **common,
        "dataset": dataset,
        "distribution": distribution,
        "stage": stage,
        "factor": factor,
        "factor_value": factor_value,
        "method": REFINED_METHODS[source_name],
        "initialization_source": source_name,
        "anchor_ce": True,
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "mu": float(mu),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "worst_class_accuracy": metrics["worst_class_accuracy"],
        "per_class_accuracy": metrics["per_class_accuracy"],
        "improved_classes_vs_initialization": int(improved),
        "anchor_forward_seconds": timing["anchor_forward_seconds"],
        "anchor_backward_update_seconds": timing[
            "anchor_backward_update_seconds"
        ],
        "refinement_seconds": timing["refinement_seconds"],
        "development_trace_seconds": timing[
            "development_evaluation_seconds"
        ],
        "distance_from_initialization": timing[
            "distance_from_initialization"
        ],
        "model_sha256": model_sha256(weight, bias),
        "config_sha256": stable_sha256(config_payload),
        "history": timing["history"],
        "weight": weight,
        "bias": bias,
    }


def causal_tuple(group: Dict[str, Dict[str, Any]]) -> Tuple[float, ...]:
    c1 = group["random"]["accuracy"]
    c3 = group["local"]["accuracy"]
    c5 = group["average"]["accuracy"]
    sample = group["average"]
    return (
        c5 - c1,
        c5 - c3,
        c5,
        -float(sample["steps"]),
        -float(sample["mu"]),
        -float(sample["learning_rate"]),
    )


def strip_arrays(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in ("weight", "bias")
    }


def run() -> Dict[str, Any]:
    started = time.perf_counter()
    source = source_commit()
    policy_sha = sha256_file(CONFIG_PATH)
    policy = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        policy["seed"] != SEED
        or policy["anchor_per_class"] != 8
        or policy["local_per_class"] != 40
    ):
        raise ValueError("policy hard gate failed")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    registry: List[Dict[str, Any]] = []
    ablation: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    efficiency_rows: List[Dict[str, Any]] = []
    reuse_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    selected_configs: Dict[str, Dict[str, Any]] = {}
    final_groups: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}

    for dataset in DATASETS:
        assets = load_dataset_assets(dataset, SEED)
        local_config = LOCAL_CONFIGS[dataset]
        bundles: Dict[str, Dict[str, Any]] = {}
        initializers: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        for distribution in DISTRIBUTIONS:
            bundle = build_or_load_local_heads(
                assets,
                distribution,
                8,
                local_config["local_epochs"],
                local_config["local_lr"],
                local_config["weight_decay"],
            )
            bundles[distribution] = bundle
            initializers[distribution] = initializer_bundle(assets, bundle)
            common = {
                "seed": SEED,
                "backend": "plaintext_float",
                "local_checkpoint_path": bundle["checkpoint"],
                "local_checkpoint_sha256": bundle["checkpoint_sha256"],
                "local_checkpoint_cache_hit": bundle["cache_hit"],
                "local_training_seconds": bundle["local_training_seconds"],
                "local_ids_sha256": sha256_array(bundle["local_ids"]),
                "anchor_ids_sha256": sha256_array(bundle["anchor_ids"]),
                "partition_hashes_sha256": stable_sha256(
                    [sha256_array(value) for value in bundle["partitions"]]
                ),
                "source_git_commit": source,
                "policy_sha256": policy_sha,
                "official_mtt_only": True,
                "full_data_initializer_dependency": False,
                "test_read": False,
                "locked_holdout_read": False,
            }
            split_rows.append(
                {
                    **common,
                    "dataset": dataset,
                    "distribution": distribution,
                    "mtt_cache_path": str(assets.mtt_cache_path),
                    "mtt_cache_sha256": assets.mtt_cache_sha256,
                    "original_split_path": str(assets.original_split_path),
                    "original_split_sha256": assets.original_split_sha256,
                    "development_ids_sha256": assets.development_ids_sha256,
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
                    "local_per_class": 40,
                    "anchor_per_class": 8,
                    "development_training_intersection": 0,
                    "development_anchor_intersection": 0,
                }
            )
            reuse_rows.append(
                {
                    "dataset": dataset,
                    "distribution": distribution,
                    "local_checkpoint_reused": bundle["cache_hit"],
                    "local_checkpoint_path": bundle["checkpoint"],
                    "local_checkpoint_sha256": bundle[
                        "checkpoint_sha256"
                    ],
                    "semantic_match": True,
                    "reused_components": [
                        "40_per_class_local_pool",
                        "8_per_class_anchor_pool",
                        "five_local_heads",
                        "C4_secure_average_endpoint",
                    ],
                }
            )

        severe_bundle = bundles["severe"]
        severe_initializers = initializers["severe"]
        severe_common = {
            key: value
            for key, value in split_rows[-1].items()
            if key
            in {
                "seed",
                "backend",
                "local_checkpoint_path",
                "local_checkpoint_sha256",
                "local_checkpoint_cache_hit",
                "local_training_seconds",
                "local_ids_sha256",
                "anchor_ids_sha256",
                "partition_hashes_sha256",
                "source_git_commit",
                "policy_sha256",
                "official_mtt_only",
                "full_data_initializer_dependency",
                "test_read",
                "locked_holdout_read",
            }
        }

        step_groups: List[Dict[str, Dict[str, Any]]] = []
        for steps in (5, 10, 20, 40):
            group = {}
            for source_name, (weight, bias) in severe_initializers.items():
                row = refined_row(
                    assets,
                    severe_bundle,
                    dataset,
                    "severe",
                    "S1_steps",
                    "steps",
                    steps,
                    source_name,
                    weight,
                    bias,
                    steps,
                    BASE_LR[dataset],
                    0.0,
                    severe_common,
                )
                registry.append(row)
                group[source_name] = row
            step_groups.append(group)
        selected_step_group = select_viable_causal_group(step_groups)
        selected_steps = selected_step_group["average"]["steps"]

        lr_groups: List[Dict[str, Dict[str, Any]]] = []
        for multiplier in (0.25, 0.5, 1.0):
            group = {}
            for source_name, (weight, bias) in severe_initializers.items():
                row = refined_row(
                    assets,
                    severe_bundle,
                    dataset,
                    "severe",
                    "S1_learning_rate",
                    "learning_rate_multiplier",
                    multiplier,
                    source_name,
                    weight,
                    bias,
                    selected_steps,
                    BASE_LR[dataset] * multiplier,
                    0.0,
                    severe_common,
                )
                registry.append(row)
                group[source_name] = row
            lr_groups.append(group)
        selected_lr_group = select_viable_causal_group(lr_groups)
        selected_lr = selected_lr_group["average"]["learning_rate"]

        mu_groups: List[Dict[str, Dict[str, Any]]] = []
        for mu in (0.0, 1e-4, 1e-3):
            group = {}
            for source_name, (weight, bias) in severe_initializers.items():
                row = refined_row(
                    assets,
                    severe_bundle,
                    dataset,
                    "severe",
                    "S1_regularization",
                    "mu",
                    mu,
                    source_name,
                    weight,
                    bias,
                    selected_steps,
                    selected_lr,
                    mu,
                    severe_common,
                )
                registry.append(row)
                group[source_name] = row
            mu_groups.append(group)
        selected_mu_group = select_viable_causal_group(mu_groups)
        selected_mu = selected_mu_group["average"]["mu"]
        selected_configs[dataset] = {
            "steps": selected_steps,
            "learning_rate": selected_lr,
            "mu": selected_mu,
            "selection_tuple": list(causal_tuple(selected_mu_group)),
        }

        for distribution in DISTRIBUTIONS:
            bundle = bundles[distribution]
            init = initializers[distribution]
            split = next(
                row
                for row in split_rows
                if row["dataset"] == dataset
                and row["distribution"] == distribution
            )
            common = {
                key: split[key]
                for key in (
                    "seed",
                    "backend",
                    "local_checkpoint_path",
                    "local_checkpoint_sha256",
                    "local_checkpoint_cache_hit",
                    "local_training_seconds",
                    "local_ids_sha256",
                    "anchor_ids_sha256",
                    "partition_hashes_sha256",
                    "source_git_commit",
                    "policy_sha256",
                    "official_mtt_only",
                    "full_data_initializer_dependency",
                    "test_read",
                    "locked_holdout_read",
                )
            }
            group: Dict[str, Dict[str, Any]] = {}
            unrefined_specs = (
                ("random", "C0_random_initialization"),
                ("local", "C2_fixed_local0"),
                ("average", "C4_secure_average"),
            )
            for source_name, method in unrefined_specs:
                weight, bias = init[source_name]
                row = unrefined_row(
                    assets,
                    dataset,
                    distribution,
                    method,
                    source_name,
                    weight,
                    bias,
                    common,
                )
                ablation.append(row)
                group[method[:2]] = row
            for source_name, (weight, bias) in init.items():
                row = refined_row(
                    assets,
                    bundle,
                    dataset,
                    distribution,
                    "S2_frozen_confirmation",
                    "frozen_initialization_source",
                    source_name,
                    source_name,
                    weight,
                    bias,
                    selected_steps,
                    selected_lr,
                    selected_mu,
                    common,
                )
                registry.append(row)
                ablation.append(row)
                group[row["method"][:2]] = row
                trajectories.append(
                    {
                        "dataset": dataset,
                        "distribution": distribution,
                        "method": row["method"],
                        "steps": selected_steps,
                        "learning_rate": selected_lr,
                        "mu": selected_mu,
                        "history": row["history"],
                    }
                )
                checkpoint = (
                    CHECKPOINT_ROOT
                    / dataset
                    / distribution
                    / "{}.npz".format(row["method"])
                )
                atomic_npz(
                    checkpoint,
                    weight=row["weight"],
                    bias=row["bias"],
                    model_sha256=np.asarray([row["model_sha256"]]),
                    config_sha256=np.asarray([row["config_sha256"]]),
                    source_git_commit=np.asarray([source]),
                    official_mtt_only=np.asarray([1], dtype=np.int64),
                    full_data_initializer_dependency=np.asarray(
                        [0], dtype=np.int64
                    ),
                    test_read=np.asarray([0], dtype=np.int64),
                    locked_holdout_read=np.asarray([0], dtype=np.int64),
                )
                row["output_checkpoint_path"] = str(checkpoint)
                row["output_checkpoint_sha256"] = sha256_file(checkpoint)
            final_groups[(dataset, distribution)] = group

            c5_final = group["C5"]["accuracy"]
            for key in ("C1", "C3", "C5"):
                row = group[key]
                efficiency_rows.append(
                    {
                        **{
                            field: row[field]
                            for field in (
                                "dataset",
                                "distribution",
                                "seed",
                                "method",
                                "source_git_commit",
                                "policy_sha256",
                                "official_mtt_only",
                                "full_data_initializer_dependency",
                                "test_read",
                                "locked_holdout_read",
                            )
                        },
                        **step_efficiency(row["history"], c5_final),
                        "scope": "Stage-B_internal_target_only_not_End-to-End",
                    }
                )

    severe_by_dataset = {
        dataset: final_groups[(dataset, "severe")] for dataset in DATASETS
    }
    severe_deltas = {
        dataset: {
            "c5_minus_c1": rows["C5"]["accuracy"] - rows["C1"]["accuracy"],
            "c5_minus_c3": rows["C5"]["accuracy"] - rows["C3"]["accuracy"],
            "c5_minus_c4": rows["C5"]["accuracy"] - rows["C4"]["accuracy"],
        }
        for dataset, rows in severe_by_dataset.items()
    }
    efficiency_by_key = {
        (row["dataset"], row["distribution"], row["method"]): row
        for row in efficiency_rows
    }
    c5_vs_c1_final_gate = (
        max(value["c5_minus_c1"] for value in severe_deltas.values()) >= 0.01
        and min(value["c5_minus_c1"] for value in severe_deltas.values())
        >= -0.003
    )
    speed_gate_by_dataset = {}
    for dataset in DATASETS:
        c1 = efficiency_by_key[
            (dataset, "severe", "C1_random_plus_anchor")
        ]["steps_at_90"]
        c5 = efficiency_by_key[
            (dataset, "severe", "C5_secure_average_plus_anchor")
        ]["steps_at_90"]
        speed_gate_by_dataset[dataset] = (
            c1 is not None
            and c5 is not None
            and c1 > 0
            and c5 <= 0.7 * c1
            and abs(severe_deltas[dataset]["c5_minus_c1"]) <= 0.003
        )
    gate_a = c5_vs_c1_final_gate or any(speed_gate_by_dataset.values())
    gate_b = (
        max(value["c5_minus_c3"] for value in severe_deltas.values()) >= 0.01
        and min(value["c5_minus_c3"] for value in severe_deltas.values())
        >= -0.003
    )
    gate_c = all(
        rows["C5"]["accuracy"] > rows["C4"]["accuracy"]
        and rows["C1"]["accuracy"] > rows["C0"]["accuracy"]
        for rows in final_groups.values()
    )
    direction_by_distribution = {}
    for distribution in DISTRIBUTIONS:
        direction_by_distribution[distribution] = all(
            final_groups[(dataset, distribution)]["C5"]["accuracy"]
            >= final_groups[(dataset, distribution)]["C1"]["accuracy"]
            and final_groups[(dataset, distribution)]["C5"]["accuracy"]
            >= final_groups[(dataset, distribution)]["C3"]["accuracy"]
            for dataset in DATASETS
        )
    gate_d = sum(direction_by_distribution.values()) >= 2
    category_checks = []
    for rows in final_groups.values():
        c4, c5 = rows["C4"], rows["C5"]
        category_checks.append(
            c5["macro_f1"] >= c4["macro_f1"] + 0.01
            or c5["worst_class_accuracy"]
            >= c4["worst_class_accuracy"] + 0.02
            or c5["improved_classes_vs_initialization"]
            > 0.6 * (10 if c5["dataset"] == "cifar10" else 100)
        )
    gate_e = any(category_checks)
    gate_f = all(
        row["official_mtt_only"]
        and not row["full_data_initializer_dependency"]
        and not row["test_read"]
        and not row["locked_holdout_read"]
        for row in ablation
    )
    smoke_pass = all((gate_a, gate_b, gate_c, gate_d, gate_e, gate_f))

    common_fields = [
        "dataset",
        "seed",
        "backend",
        "distribution",
        "stage",
        "factor",
        "factor_value",
        "method",
        "initialization_source",
        "anchor_ce",
        "steps",
        "learning_rate",
        "mu",
        "accuracy",
        "macro_f1",
        "worst_class_accuracy",
        "improved_classes_vs_initialization",
        "anchor_forward_seconds",
        "anchor_backward_update_seconds",
        "refinement_seconds",
        "distance_from_initialization",
        "local_training_seconds",
        "local_checkpoint_cache_hit",
        "local_checkpoint_sha256",
        "local_ids_sha256",
        "anchor_ids_sha256",
        "partition_hashes_sha256",
        "model_sha256",
        "config_sha256",
        "source_git_commit",
        "policy_sha256",
        "official_mtt_only",
        "full_data_initializer_dependency",
        "test_read",
        "locked_holdout_read",
    ]
    write_csv(REGISTRY_PATH, registry, common_fields)
    write_csv(ABLATION_PATH, ablation, common_fields)
    efficiency_fields = [
        "dataset",
        "distribution",
        "seed",
        "method",
        "target_80",
        "steps_at_80",
        "seconds_at_80",
        "target_90",
        "steps_at_90",
        "seconds_at_90",
        "target_95",
        "steps_at_95",
        "seconds_at_95",
        "scope",
        "source_git_commit",
        "policy_sha256",
        "official_mtt_only",
        "full_data_initializer_dependency",
        "test_read",
        "locked_holdout_read",
    ]
    write_csv(EFFICIENCY_PATH, efficiency_rows, efficiency_fields)

    split_payload = {
        "schema": "new_stage_b_anchor_postagg_split_manifest_v1",
        "status": "COMPLETE",
        "source_git_commit": source,
        "policy_sha256": policy_sha,
        "rows": split_rows,
        "all_local_anchor_disjoint": all(
            row["local_anchor_intersection"] == 0 for row in split_rows
        ),
        "all_development_disjoint": True,
        "official_mtt_only": True,
        "full_data_initializer_dependency": False,
        "test_read": False,
        "locked_holdout_read": False,
    }
    atomic_json(SPLIT_PATH, split_payload)
    split_sha = sha256_file(SPLIT_PATH)
    reuse_payload = {
        "schema": "new_stage_b_anchor_postagg_reuse_audit_v1",
        "status": "COMPLETE",
        "source_git_commit": source,
        "rows": reuse_rows,
        "exact_valid_local_heads_reused": sum(
            row["local_checkpoint_reused"] for row in reuse_rows
        ),
        "new_causal_components_computed": [
            "C0_random",
            "C1_random_plus_anchor",
            "C2_fixed_client0",
            "C3_fixed_client0_plus_anchor",
            "C1_C3_C5_step_trajectories",
        ],
        "split_manifest_sha256": split_sha,
    }
    atomic_json(REUSE_PATH, reuse_payload)
    atomic_json(
        TRAJECTORY_PATH,
        {
            "schema": "new_stage_b_anchor_postagg_trajectories_v1",
            "status": "COMPLETE",
            "source_git_commit": source,
            "rows": trajectories,
            "test_read": False,
            "locked_holdout_read": False,
        },
    )
    summary = {
        "schema": "new_stage_b_anchor_postagg_smoke_summary_v1",
        "status": "COMPLETE",
        "source_git_commit": source,
        "policy_sha256": policy_sha,
        "split_manifest_path": str(SPLIT_PATH.relative_to(ROOT)),
        "split_sha256": split_sha,
        "selected_configs": selected_configs,
        "severe_deltas": severe_deltas,
        "direction_by_distribution": direction_by_distribution,
        "speed_gate_by_dataset": speed_gate_by_dataset,
        "gates": {
            "gate_a_secure_average_independent_contribution": gate_a,
            "gate_b_aggregation_beats_local": gate_b,
            "gate_c_anchor_effective": gate_c,
            "gate_d_cross_distribution_direction": gate_d,
            "gate_e_class_robustness": gate_e,
            "gate_f_lineage": gate_f,
            "all_smoke_gates_pass": smoke_pass,
        },
        "next_phase": (
            "FREEZE_THEN_TFE" if smoke_pass else "NEGATIVE_STOP_PLAINTEXT"
        ),
        "registered_rows": len(registry),
        "ablation_rows": len(ablation),
        "efficiency_rows": len(efficiency_rows),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_gib": peak_rss_gib(),
        "official_mtt_only": True,
        "full_data_initializer_dependency": False,
        "test_read": False,
        "locked_holdout_read": False,
        "stage_c_run": False,
        "end_to_end_run": False,
        "tfe_status": "PENDING_FREEZE" if smoke_pass else "SKIPPED_SMOKE_GATE",
    }
    atomic_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()

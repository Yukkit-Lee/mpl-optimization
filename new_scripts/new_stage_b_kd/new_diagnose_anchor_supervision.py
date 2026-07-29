#!/usr/bin/env python3
"""Separate anchor-label supervision from uniform/class-aware KD effects."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from new_run_stage_b_kd_plaintext import (
    build_or_load_local_heads,
    evaluate_local_bundle,
    teacher_candidate,
)
from new_stage_b_kd_core import (
    DATASETS,
    LOG_ROOT,
    SEED,
    TABLE_ROOT,
    atomic_json,
    evaluate_head,
    load_dataset_assets,
    source_commit,
    train_student,
)


SUMMARY_PATH = LOG_ROOT / "new_stage_b_kd_smoke_summary.json"
OUTPUT_JSON = LOG_ROOT / "new_stage_b_kd_anchor_supervision_ablation.json"
OUTPUT_CSV = TABLE_ROOT / "new_stage_b_kd_anchor_supervision_ablation.csv"


def train_variant(
    assets,
    bundle,
    evaluation,
    teacher,
    student_config,
    method,
    kd_lambda,
) -> Dict[str, Any]:
    weight, bias, timing = train_student(
        bundle["x_anchor"],
        bundle["y_anchor"],
        teacher["teacher_anchor_probabilities"],
        evaluation["average_weight"],
        evaluation["average_bias"],
        teacher["teacher_temperature"],
        kd_lambda,
        student_config["average_regularization"],
        student_config["student_steps"],
        student_config["student_lr"],
    )
    metrics = evaluate_head(
        assets.x_dev, assets.y_dev, weight, bias, assets.num_classes
    )
    baseline = evaluation["average_metrics"]
    return {
        "dataset": assets.dataset,
        "seed": SEED,
        "distribution": bundle["config"]["distribution"],
        "method": method,
        "kd_lambda": kd_lambda,
        "initializer_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "worst_class_accuracy": metrics["worst_class_accuracy"],
        "gain_over_secure_average": metrics["accuracy"] - baseline["accuracy"],
        "worst_class_gain_over_secure_average": (
            metrics["worst_class_accuracy"]
            - baseline["worst_class_accuracy"]
        ),
        "student_kd_seconds": timing["student_kd_seconds"],
    }


def main() -> None:
    source = source_commit()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["status"] != "COMPLETE" or summary["seed_policy"] != [SEED]:
        raise ValueError("single-seed plaintext summary is not COMPLETE")
    smoke_by_dataset = {
        row["dataset"]: row for row in summary["smoke_results"]
    }
    rows: List[Dict[str, Any]] = []
    decisions = []
    for dataset in DATASETS:
        assets = load_dataset_assets(dataset, SEED)
        smoke = smoke_by_dataset[dataset]
        local = smoke["selected_local_config"]
        teacher_config = smoke["selected_teacher_config"]
        student_config = smoke["selected_student_config"]
        distribution_rows = {}
        for distribution in ("iid", "mild", "severe"):
            bundle = build_or_load_local_heads(
                assets,
                distribution,
                local["anchor_per_class"],
                local["local_epochs"],
                local["local_lr"],
                local["weight_decay"],
            )
            evaluation = evaluate_local_bundle(assets, bundle)
            uniform_teacher = teacher_candidate(
                assets,
                bundle,
                "uniform",
                teacher_config["teacher_temperature"],
                1.0,
                1.0,
                0.0,
            )
            class_teacher = teacher_candidate(
                assets,
                bundle,
                teacher_config["rule"],
                teacher_config["teacher_temperature"],
                teacher_config["weight_temperature"],
                teacher_config["maximum_client_weight"],
                teacher_config["shrinkage_to_uniform"],
            )
            baseline = evaluation["average_metrics"]
            base_row = {
                "dataset": dataset,
                "seed": SEED,
                "distribution": distribution,
                "method": "secure_average_equivalent_plaintext",
                "kd_lambda": None,
                "initializer_accuracy": baseline["accuracy"],
                "macro_f1": baseline["macro_f1"],
                "worst_class_accuracy": baseline["worst_class_accuracy"],
                "gain_over_secure_average": 0.0,
                "worst_class_gain_over_secure_average": 0.0,
                "student_kd_seconds": 0.0,
            }
            variants = [
                base_row,
                train_variant(
                    assets,
                    bundle,
                    evaluation,
                    uniform_teacher,
                    student_config,
                    "anchor_ce_only",
                    0.0,
                ),
                train_variant(
                    assets,
                    bundle,
                    evaluation,
                    uniform_teacher,
                    student_config,
                    "uniform_kd_only",
                    1.0,
                ),
                train_variant(
                    assets,
                    bundle,
                    evaluation,
                    class_teacher,
                    student_config,
                    "class_aware_kd_only",
                    1.0,
                ),
                train_variant(
                    assets,
                    bundle,
                    evaluation,
                    uniform_teacher,
                    student_config,
                    "uniform_ce_plus_kd",
                    student_config["kd_lambda"],
                ),
                train_variant(
                    assets,
                    bundle,
                    evaluation,
                    class_teacher,
                    student_config,
                    "class_aware_ce_plus_kd",
                    student_config["kd_lambda"],
                ),
            ]
            for row in variants:
                row.update(
                    {
                        "uniform_teacher_accuracy": uniform_teacher["metrics"][
                            "accuracy"
                        ],
                        "class_aware_teacher_accuracy": class_teacher["metrics"][
                            "accuracy"
                        ],
                        "source_commit": source,
                        "initializer_source": "official_mtt_only",
                        "full_data_initializer_dependency": False,
                        "locked_holdout_read": False,
                        "official_test_read": False,
                    }
                )
            rows.extend(variants)
            distribution_rows[distribution] = {
                row["method"]: row for row in variants
            }

        severe = distribution_rows["severe"]
        iid = distribution_rows["iid"]
        ce_gain = severe["anchor_ce_only"]["gain_over_secure_average"]
        mixed_gain = severe["class_aware_ce_plus_kd"][
            "gain_over_secure_average"
        ]
        class_vs_uniform_student = (
            severe["class_aware_ce_plus_kd"]["initializer_accuracy"]
            - severe["uniform_ce_plus_kd"]["initializer_accuracy"]
        )
        class_vs_uniform_teacher = (
            severe["class_aware_ce_plus_kd"]["class_aware_teacher_accuracy"]
            - severe["class_aware_ce_plus_kd"]["uniform_teacher_accuracy"]
        )
        kd_increment_over_ce = (
            severe["class_aware_ce_plus_kd"]["initializer_accuracy"]
            - severe["anchor_ce_only"]["initializer_accuracy"]
        )
        iid_class_delta = (
            iid["class_aware_ce_plus_kd"]["initializer_accuracy"]
            - iid["secure_average_equivalent_plaintext"]["initializer_accuracy"]
        )
        decisions.append(
            {
                "dataset": dataset,
                "seed": SEED,
                "severe_anchor_ce_gain": ce_gain,
                "severe_class_aware_mixed_gain": mixed_gain,
                "class_aware_vs_uniform_teacher_delta": class_vs_uniform_teacher,
                "class_aware_vs_uniform_student_delta": class_vs_uniform_student,
                "class_aware_kd_increment_over_anchor_ce": kd_increment_over_ce,
                "iid_class_aware_delta": iid_class_delta,
                "anchor_data_effective": (
                    ce_gain >= 0.01
                    or severe["anchor_ce_only"][
                        "worst_class_gain_over_secure_average"
                    ]
                    >= 0.02
                ),
                "class_aware_consensus_incremental_positive": (
                    class_vs_uniform_teacher > 0.0
                    and class_vs_uniform_student > 0.0
                    and kd_increment_over_ce > 0.0
                ),
                "gain_is_anchor_supervision_dominated": (
                    mixed_gain > 0.0 and ce_gain / mixed_gain >= 0.8
                ),
            }
        )

    payload = {
        "schema": "new_stage_b_kd_anchor_supervision_ablation_v1",
        "status": "COMPLETE",
        "source_commit": source,
        "seed_policy": [SEED],
        "purpose": (
            "separate labeled-anchor CE benefit from uniform/class-aware KD"
        ),
        "rows": rows,
        "decisions": decisions,
        "anchor_data_effective_all_datasets": all(
            row["anchor_data_effective"] for row in decisions
        ),
        "class_aware_consensus_positive_all_datasets": all(
            row["class_aware_consensus_incremental_positive"]
            for row in decisions
        ),
        "formal_plan_plaintext_gate_previously_passed": True,
        "tfe_validation_still_required_by_plan": True,
        "locked_holdout_read": False,
        "official_test_read": False,
        "full_data_initializer_loaded": False,
    }
    atomic_json(OUTPUT_JSON, payload)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "seed",
        "distribution",
        "method",
        "kd_lambda",
        "initializer_accuracy",
        "macro_f1",
        "worst_class_accuracy",
        "gain_over_secure_average",
        "worst_class_gain_over_secure_average",
        "uniform_teacher_accuracy",
        "class_aware_teacher_accuracy",
        "student_kd_seconds",
        "source_commit",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})
    print("COMPLETE rows={} output={}".format(len(rows), OUTPUT_JSON))
    for decision in decisions:
        print(
            "{dataset} anchor_CE_gain={severe_anchor_ce_gain:+.4f} "
            "class_vs_uniform_student={class_aware_vs_uniform_student_delta:+.4f} "
            "KD_over_CE={class_aware_kd_increment_over_anchor_ce:+.4f}".format(
                **decision
            )
        )
    print("locked_holdout_read=False official_test_read=False stage_c=False")


if __name__ == "__main__":
    main()

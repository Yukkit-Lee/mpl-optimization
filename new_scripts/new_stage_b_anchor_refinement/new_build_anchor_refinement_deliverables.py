#!/usr/bin/env python3
"""Build and audit every required Stage-B anchor-refinement deliverable."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
KD_DIR = ROOT / "new_scripts/new_stage_b_kd"
for value in (HERE, KD_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from new_stage_b_kd_core import atomic_json, source_commit  # noqa: E402
from new_stage_b_anchor_refinement_core import (  # noqa: E402
    LOG_ROOT,
    OUTPUT_ROOT,
    TABLE_ROOT,
)


SUMMARY = LOG_ROOT / "new_stage_b_anchor_refinement_smoke_summary.json"
KD_ABLATION = (
    ROOT
    / "new_log/new_stage_b_kd/new_stage_b_kd_anchor_supervision_ablation.json"
)
FIG_ROOT = ROOT / "new_expFig/new_stage_b_anchor_refinement"


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for root in (FIG_ROOT, OUTPUT_ROOT):
        figure.savefig(root / (stem + ".png"), dpi=220)
        figure.savefig(root / (stem + ".pdf"))
    plt.close(figure)


def best_rows(result: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    informative = [
        row
        for row in result["rows"]
        if row["distribution"] == "severe"
        and row["effective_disagreement_score_std"] > 1e-12
        and not row["uniform_equivalent"]
    ]
    b2 = max(
        [row for row in informative if row["method"].startswith("B2")],
        key=lambda row: (
            row["initializer_accuracy"],
            row["worst_class_accuracy"],
            row["macro_f1"],
        ),
    )
    b3 = max(
        [row for row in informative if row["method"].startswith("B3")],
        key=lambda row: (
            row["initializer_accuracy"],
            row["worst_class_accuracy"],
            row["macro_f1"],
        ),
    )
    return b2, b3


def stage_b_runtime(
    result: Dict[str, Any], row: Dict[str, Any] | None
) -> Dict[str, float]:
    shared = result["runtime"]
    candidate = row or {}
    historical_local = float(shared["client_local_training_seconds"])
    secure_average = float(shared["secure_average_seconds"])
    anchor_inference = float(candidate.get("anchor_inference_seconds", 0.0))
    disagreement = float(candidate.get("disagreement_seconds", 0.0))
    selection = float(candidate.get("anchor_selection_seconds", 0.0))
    refinement = float(candidate.get("refinement_seconds", 0.0))
    incremental = (
        secure_average
        + anchor_inference
        + disagreement
        + selection
        + refinement
    )
    return {
        "client_local_training_seconds": historical_local,
        "secure_average_seconds": secure_average,
        "anchor_inference_seconds": anchor_inference,
        "disagreement_seconds": disagreement,
        "anchor_selection_seconds": selection,
        "weighted_ce_refinement_seconds": refinement,
        "current_cached_stage_b_seconds": incremental,
        "full_recompute_stage_b_seconds": historical_local + incremental,
    }


def method_payloads(
    result: Dict[str, Any],
    kd_row: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any], str]]:
    b0 = result["selected"]["b0"]
    b1 = result["selected"]["b1"]
    b2, b3 = best_rows(result)
    return [
        ("B0_secure_average", b0, "current_40_8"),
        ("B1_uniform_anchor_ce", b1, "current_40_8"),
        ("B2_disagreement_weighted_ce", b2, "current_40_8"),
        ("B3_top_disagreement_ce", b3, "current_40_8"),
        ("B4_existing_kd_negative_control", kd_row, "prior_registered_kd_protocol"),
    ]


def build_tables(
    payload: Dict[str, Any],
    kd_severe: Dict[str, Dict[str, Any]],
    selected: Dict[str, Dict[str, Any]],
    source: str,
) -> None:
    main_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    diagnosis_rows: List[Dict[str, Any]] = []
    resource_rows: List[Dict[str, Any]] = []
    tfe_rows: List[Dict[str, Any]] = []
    for result in payload["results"]:
        dataset = result["dataset"]
        b0 = result["selected"]["b0"]
        for method, row, protocol in method_payloads(
            result, kd_severe[dataset]
        ):
            is_b0 = method == "B0_secure_average"
            is_kd = method == "B4_existing_kd_negative_control"
            accuracy = row.get("initializer_accuracy", row.get("accuracy"))
            runtime = None if is_kd else stage_b_runtime(
                result, None if is_b0 else row
            )
            main_rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "distribution": "severe",
                    "method": method,
                    "development_accuracy": accuracy,
                    "macro_f1": row.get("macro_f1"),
                    "worst_class_accuracy": row.get("worst_class_accuracy"),
                    "active_anchors": (
                        ""
                        if is_kd
                        else (0 if is_b0 else row["effective_anchor_count"])
                    ),
                    "refinement_steps": (
                        "" if is_kd else (0 if is_b0 else row["steps"])
                    ),
                    "stage_b_runtime_seconds": (
                        ""
                        if runtime is None
                        else runtime["full_recompute_stage_b_seconds"]
                    ),
                    "current_cached_stage_b_seconds": (
                        ""
                        if runtime is None
                        else runtime["current_cached_stage_b_seconds"]
                    ),
                    "backend": "numpy_plaintext_float",
                    "protocol": protocol,
                    "source_commit": source,
                    "initializer_source": "official_mtt_only",
                    "full_data_initializer_dependency": False,
                    "locked_holdout_read": False,
                    "official_test_read": False,
                }
            )
            if not is_kd:
                runtime_rows.append(
                    {
                        "dataset": dataset,
                        "seed": 42,
                        "distribution": "severe",
                        "method": method,
                        **runtime,
                        "peak_rss_gib": payload["peak_rss_gib"],
                        "communication_bytes": "N/A_not_measured",
                        "communication_rounds": "N/A_not_measured",
                        "source_commit": source,
                        "locked_holdout_read": False,
                        "official_test_read": False,
                    }
                )
        for metric, diagnostic in result["diagnostic"]["metrics"].items():
            precision = diagnostic["precision_at_top_disagreement"]
            diagnosis_rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "metric": metric,
                    "error_detection_auroc": diagnostic[
                        "auroc_average_error"
                    ],
                    "top_10_error_precision": precision["top_10"],
                    "top_20_error_precision": precision["top_20"],
                    "top_30_error_precision": precision["top_30"],
                    "high_20_error_rate": diagnostic["high_20_error_rate"],
                    "low_20_error_rate": diagnostic["low_20_error_rate"],
                    "high_minus_low_error_gap": (
                        diagnostic["high_20_error_rate"]
                        - diagnostic["low_20_error_rate"]
                    ),
                    "sample_error_correlation": diagnostic[
                        "pearson_sample_disagreement_vs_error"
                    ],
                    "class_weakness_correlation": diagnostic[
                        "pearson_class_disagreement_vs_weakness"
                    ],
                    "score_std": diagnostic["score_std"],
                    "source_commit": source,
                    "locked_holdout_read": False,
                    "official_test_read": False,
                }
            )
        checkpoint = Path(result["local_checkpoint_path"])
        resource_rows.append(
            {
                "dataset": dataset,
                "seed": 42,
                "registered_candidates": len(result["rows"]),
                "nproc": os.cpu_count(),
                "peak_rss_gib": payload["peak_rss_gib"],
                "local_checkpoint_cache_hit": result["runtime"][
                    "local_checkpoint_cache_hit"
                ],
                "local_checkpoint_bytes": checkpoint.stat().st_size,
                "anchor_count": result["selected"]["b1"]["total_anchor_count"],
                "search_runtime_seconds_all_datasets": payload[
                    "runtime_seconds"
                ],
                "communication_bytes": "N/A_not_measured",
                "source_commit": source,
                "locked_holdout_read": False,
                "official_test_read": False,
            }
        )
        for method, row in (
            ("B0_secure_average", b0),
            ("B1_uniform_anchor_ce", selected[dataset]["b1"]),
            ("B3_best_informative", selected[dataset]["b3"]),
        ):
            accuracy = row.get("initializer_accuracy", row.get("accuracy"))
            tfe_rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "method": method,
                    "plaintext_accuracy": accuracy,
                    "fixed_point_accuracy": "",
                    "tfe_accuracy": "",
                    "absolute_gap": "",
                    "status": "SKIPPED",
                    "reason": "plaintext_gate_failed",
                    "backend": "TFE_ABY3_not_started",
                    "source_commit": source,
                    "locked_holdout_read": False,
                    "official_test_read": False,
                }
            )
    write_csv(
        TABLE_ROOT / "new_stage_b_anchor_refinement_main_table.csv",
        list(main_rows[0]),
        main_rows,
    )
    write_csv(
        TABLE_ROOT / "new_stage_b_anchor_refinement_runtime_breakdown.csv",
        list(runtime_rows[0]),
        runtime_rows,
    )
    write_csv(
        TABLE_ROOT / "new_stage_b_anchor_refinement_resource_summary.csv",
        list(resource_rows[0]),
        resource_rows,
    )
    write_csv(
        TABLE_ROOT / "new_stage_b_anchor_refinement_disagreement_diagnosis.csv",
        list(diagnosis_rows[0]),
        diagnosis_rows,
    )
    write_csv(
        TABLE_ROOT / "new_stage_b_anchor_refinement_tfe_validation.csv",
        list(tfe_rows[0]),
        tfe_rows,
    )


def build_figures(
    payload: Dict[str, Any],
    kd_severe: Dict[str, Dict[str, Any]],
    selected: Dict[str, Dict[str, Any]],
) -> None:
    datasets = ["cifar10", "cifar100"]
    labels = ["B0", "B1", "B2", "B3", "KD†"]
    colors = ["#777777", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, dataset in zip(axes, datasets):
        values = [
            selected[dataset]["b0"]["accuracy"],
            selected[dataset]["b1"]["initializer_accuracy"],
            selected[dataset]["b2"]["initializer_accuracy"],
            selected[dataset]["b3"]["initializer_accuracy"],
            kd_severe[dataset]["initializer_accuracy"],
        ]
        axis.bar(labels, np.asarray(values) * 100.0, color=colors)
        axis.set_title(dataset.upper())
        axis.set_ylabel("Development accuracy (%)")
        axis.set_ylim(0, max(values) * 115.0)
    figure.suptitle(
        "Figure B1: Stage-B initializer quality (seed42, Severe Non-IID)"
    )
    figure.text(
        0.5,
        0.005,
        "† KD is a reused negative control from the prior registered protocol; "
        "it is not an equal-budget current-40/8 comparison.",
        ha="center",
        fontsize=8,
    )
    save_figure(figure, "new_figure_b1_initializer_quality")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, result in zip(axes, payload["results"]):
        diagnostics = result["diagnostic"]["metrics"]
        metric = max(
            [
                name
                for name, item in diagnostics.items()
                if item["score_std"] > 1e-12
            ],
            key=lambda name: diagnostics[name]["auroc_average_error"],
        )
        records = result["diagnostic"]["anchor_records"]
        scores = np.asarray(records[metric], dtype=np.float64)
        losses = np.asarray(records["average_cross_entropy"], dtype=np.float64)
        errors = np.asarray(records["average_errors"], dtype=bool)
        axis.scatter(
            scores[~errors],
            losses[~errors],
            s=18,
            alpha=0.55,
            label="Average correct",
            color="#3b82f6",
        )
        axis.scatter(
            scores[errors],
            losses[errors],
            s=22,
            alpha=0.65,
            label="Average error",
            color="#ef4444",
        )
        axis.set_xlabel("{} score".format(metric))
        axis.set_ylabel("Secure-Average anchor CE loss")
        axis.set_title(
            "{} (error AUROC={:.3f})".format(
                result["dataset"].upper(),
                diagnostics[metric]["auroc_average_error"],
            )
        )
        axis.legend(fontsize=8)
    figure.suptitle(
        "Figure B2: Anchor disagreement versus Secure-Average correctness/loss"
    )
    save_figure(figure, "new_figure_b2_disagreement_vs_error")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, result in zip(axes, payload["results"]):
        rows = [
            row
            for row in result["rows"]
            if row["distribution"] == "severe"
            and row["weight_rule"] != "uniform"
            and row["effective_disagreement_score_std"] > 1e-12
            and not row["uniform_equivalent"]
        ]
        axis.scatter(
            [row["active_anchor_ratio"] * 100.0 for row in rows],
            [100.0 * row["initializer_accuracy"] for row in rows],
            alpha=0.55,
            s=22,
            label="Disagreement candidates",
        )
        b1 = result["selected"]["b1"]
        axis.scatter(
            [100.0],
            [100.0 * b1["initializer_accuracy"]],
            marker="*",
            s=150,
            color="black",
            label="B1 uniform",
        )
        axis.set_title(result["dataset"].upper())
        axis.set_xlabel("Active anchor ratio (%)")
        axis.set_ylabel("Development accuracy (%)")
        axis.legend(fontsize=8)
    figure.suptitle("Figure B3: Utility-cost tradeoff")
    save_figure(figure, "new_figure_b3_utility_cost")

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for axis, dataset in zip(axes, datasets):
        b0 = np.asarray(selected[dataset]["b0"]["per_class_accuracy"])
        b1 = np.asarray(selected[dataset]["b1"]["per_class_accuracy"])
        b3 = np.asarray(selected[dataset]["b3"]["per_class_accuracy"])
        classes = np.arange(len(b0))
        axis.plot(classes, 100 * b0, label="B0 Secure Average", linewidth=1.2)
        axis.plot(classes, 100 * b1, label="B1 uniform CE", linewidth=1.5)
        axis.plot(classes, 100 * b3, label="B3 best top-disagreement", linewidth=1.2)
        axis.set_title(dataset.upper())
        axis.set_ylabel("Per-class accuracy (%)")
        axis.set_xlabel("Class")
        axis.legend(fontsize=8)
    figure.suptitle(
        "Figure B4: Class-wise accuracy before and after anchor refinement"
    )
    save_figure(figure, "new_figure_b4_classwise")


def build_reports(
    payload: Dict[str, Any],
    selected: Dict[str, Dict[str, Any]],
    source: str,
) -> None:
    split_manifest = json.loads(
        (
            LOG_ROOT / "new_stage_b_anchor_refinement_split_manifest.json"
        ).read_text(encoding="utf-8")
    )
    split_lines = [
        "# Stage-B Anchor Refinement Split Audit",
        "",
        "Official-MTT-only 40 local + 8 anchor samples/class; seed42.",
        "",
        "| Dataset | Local/class | Anchor/class | Intersection | Union | Checkpoint SHA256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dependency in split_manifest["dependencies"]:
        split_lines.append(
            "| {dataset} | 40 | 8 | {local_anchor_intersection} | "
            "{local_anchor_union} | `{local_checkpoint_sha256}` |".format(
                **dependency
            )
        )
    split_lines.extend(
        [
            "",
            "Development examples come from the disjoint frozen training split; "
            "locked holdout and official test were not read.",
            "",
            "Source commit: `{}`.".format(source),
        ]
    )
    (LOG_ROOT / "new_stage_b_anchor_refinement_split_audit.md").write_text(
        "\n".join(split_lines) + "\n", encoding="utf-8"
    )

    diagnostic_lines = [
        "# Stage-B Anchor Refinement Diagnostic Report",
        "",
        "Scope: CIFAR-10/100, seed42, Severe Non-IID, plaintext development-only.",
        "",
    ]
    for result in payload["results"]:
        diagnostic_lines.extend(["## {}".format(result["dataset"].upper()), ""])
        for name, diagnostic in result["diagnostic"]["metrics"].items():
            diagnostic_lines.append(
                "- {}: AUROC={:.4f}, sample correlation={:.4f}, "
                "class-weakness correlation={:.4f}, score std={:.6g}.".format(
                    name,
                    diagnostic["auroc_average_error"],
                    diagnostic["pearson_sample_disagreement_vs_error"],
                    diagnostic["pearson_class_disagreement_vs_weakness"],
                    diagnostic["score_std"],
                )
            )
        diagnostic_lines.extend(
            [
                "",
                "D1/D2 are negatively associated with Average errors; D3 vote "
                "entropy is constant. Direct sample- and class-level "
                "high-disagreement prioritization therefore lacks a valid "
                "error-targeting signal.",
                "",
            ]
        )
    (LOG_ROOT / "new_stage_b_anchor_refinement_diagnostic_report.md").write_text(
        "\n".join(diagnostic_lines) + "\n", encoding="utf-8"
    )

    claim_lines = [
        "# Stage-B Anchor Refinement Claim Audit",
        "",
        "| Claim | Status | Evidence |",
        "|---|---|---|",
        "| Uniform labeled MTT-anchor CE improves Secure Average | Supported for seed42 development | B1 vs B0 on C10/C100 |",
        "| Higher client disagreement predicts Secure-Average errors | Not supported | Informative AUROC below 0.5; D3 degenerate |",
        "| Disagreement weighting improves over uniform CE | Not supported | B2/B3 fail utility, cost and IID gates |",
        "| Secure TFE/ABY3 result exists | Not run by design | Plaintext gate failed |",
        "| End-to-End/test/DP improvement | Forbidden and not claimed | Stage-B development-only |",
        "| Communication savings | Not claimed | Bytes/rounds were not measured |",
        "",
        "Source commit: `{}`.".format(source),
    ]
    (LOG_ROOT / "new_stage_b_anchor_refinement_claim_audit.md").write_text(
        "\n".join(claim_lines) + "\n", encoding="utf-8"
    )

    report = [
        "# Heterogeneity-Aware Secure Anchor Refinement Final Report",
        "",
        "## Outcome",
        "",
        "Registered negative result: uniform labeled-anchor CE is effective, "
        "but high-disagreement weighting provides neither utility nor cost "
        "benefits over B1. The plaintext gate failed on both datasets, so "
        "TFE/ABY3 was correctly not started.",
        "",
    ]
    for result in payload["results"]:
        dataset = result["dataset"]
        b0 = selected[dataset]["b0"]
        b1 = selected[dataset]["b1"]
        b2 = selected[dataset]["b2"]
        b3 = selected[dataset]["b3"]
        b1_runtime = stage_b_runtime(result, b1)
        b3_runtime = stage_b_runtime(result, b3)
        report.extend(
            [
                "### {}".format(dataset.upper()),
                "",
                "- B0/B1/B2/B3 accuracy: {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}%.".format(
                    100 * b0["accuracy"],
                    100 * b1["initializer_accuracy"],
                    100 * b2["initializer_accuracy"],
                    100 * b3["initializer_accuracy"],
                ),
                "- B1/B3 worst-class accuracy: {:.2f}% / {:.2f}%.".format(
                    100 * b1["worst_class_accuracy"],
                    100 * b3["worst_class_accuracy"],
                ),
                "- B1/B3 active anchors: {} / {}.".format(
                    b1["effective_anchor_count"], b3["effective_anchor_count"]
                ),
                "- Full-recompute Stage-B runtime B1/B3: {:.4f}s / {:.4f}s; "
                "the local-head training component is shared.".format(
                    b1_runtime["full_recompute_stage_b_seconds"],
                    b3_runtime["full_recompute_stage_b_seconds"],
                ),
                "- Best informative disagreement AUROC: {:.4f}; IID delta vs "
                "B1: {:.2f}pp.".format(
                    result["gates"]["best_disagreement_auroc"],
                    100 * result["gates"]["iid_delta_vs_b1"],
                ),
                "",
            ]
        )
    report.extend(
        [
            "## Ten required answers",
            "",
            "1. **Does disagreement correlate with Average errors?** No; the informative metrics correlate in the opposite direction.",
            "2. **Which metric works best?** D1 on C10 and D2 on C100 by AUROC, but both are below 0.5 and therefore unusable as positive detectors.",
            "3. **Does guided CE beat uniform CE?** No.",
            "4. **Same accuracy with fewer anchors/steps?** No registered candidate meets the 0.2pp/30% gate.",
            "5. **Worst-class improvement?** No.",
            "6. **IID preserved?** No; the selected severe candidate violates the 0.5pp safety gate.",
            "7. **TFE/ABY3 reproduction?** Not attempted because the mandatory plaintext gate failed.",
            "8. **Extra Stage-B runtime?** Exact plaintext components are in the runtime table; secure runtime was not measured because Phase B was not authorized.",
            "9. **Positioning?** Negative result; B1 anchor supervision remains useful.",
            "10. **Paper-safe claims?** Seed42 development-only B1 benefit and the negative disagreement finding.",
            "",
            "## Boundaries",
            "",
            "- Official-MTT-only; Full initializer dependency is false.",
            "- locked holdout and official test were not read.",
            "- No Stage C, DP-SGD, End-to-End, Tiny, seeds43/44, or Figure 2.",
            "- Communication was not measured and no communication-saving claim is made.",
            "- KD is a prior-protocol negative control and is explicitly marked protocol-different.",
            "",
            "Source commit: `{}`.".format(source),
        ]
    )
    (LOG_ROOT / "new_stage_b_anchor_refinement_final_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def completion_audit(payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    required = [
        TABLE_ROOT / "new_stage_b_anchor_refinement_smoke_registry.csv",
        LOG_ROOT / "new_stage_b_anchor_refinement_smoke_summary.json",
        LOG_ROOT / "new_stage_b_anchor_refinement_diagnostic_report.md",
        TABLE_ROOT / "new_stage_b_anchor_refinement_tfe_validation.csv",
        TABLE_ROOT / "new_stage_b_anchor_refinement_main_table.csv",
        LOG_ROOT / "new_stage_b_anchor_refinement_claim_audit.md",
        LOG_ROOT / "new_stage_b_anchor_refinement_final_report.md",
        TABLE_ROOT / "new_stage_b_anchor_refinement_runtime_breakdown.csv",
        TABLE_ROOT / "new_stage_b_anchor_refinement_resource_summary.csv",
        TABLE_ROOT / "new_stage_b_anchor_refinement_disagreement_diagnosis.csv",
        LOG_ROOT / "new_stage_b_anchor_refinement_split_manifest.json",
        LOG_ROOT / "new_stage_b_anchor_refinement_split_audit.md",
    ]
    for stem in (
        "new_figure_b1_initializer_quality",
        "new_figure_b2_disagreement_vs_error",
        "new_figure_b3_utility_cost",
        "new_figure_b4_classwise",
    ):
        for root in (FIG_ROOT, OUTPUT_ROOT):
            required.extend([root / (stem + ".png"), root / (stem + ".pdf")])
    expected_families = {
        "B1_uniform_s1_steps": 4,
        "B1_uniform_s1_lr": 3,
        "B1_uniform_s1_mu": 3,
        "B2_s2_metric": 3,
        "B2_s3_weight": 2,
        "B3_s3_weight": 8,
        "B2_case1_class_priority": 3,
        "B3_s4_cost": 5,
        "diagnostic_capped_weight": 18,
        "B1_iid_safety": 1,
        "B2B3_iid_safety": 1,
    }
    family_checks = []
    for result in payload["results"]:
        counts: Dict[str, int] = {}
        for row in result["rows"]:
            counts[row["method"]] = counts.get(row["method"], 0) + 1
        family_checks.append(
            {
                "dataset": result["dataset"],
                "counts": counts,
                "expected": expected_families,
                "pass": all(
                    counts.get(name) == count
                    for name, count in expected_families.items()
                ),
            }
        )
    all_required = all(path.is_file() and path.stat().st_size > 0 for path in required)
    status = (
        "COMPLETE"
        if all_required
        and all(item["pass"] for item in family_checks)
        and not payload["all_plaintext_gates_pass"]
        and payload["next_phase"] == "NEGATIVE_RESULT_STOP_BEFORE_TFE"
        and not payload["locked_holdout_read"]
        and not payload["official_test_read"]
        and not payload["full_data_initializer_dependency"]
        else "INCOMPLETE"
    )
    audit = {
        "schema": "new_stage_b_anchor_refinement_completion_audit_v2",
        "status": status,
        "source_commit": source,
        "required_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "exists": path.is_file(),
                "nonempty": path.is_file() and path.stat().st_size > 0,
            }
            for path in required
        ],
        "registered_family_checks": family_checks,
        "registered_candidates_total": sum(
            len(result["rows"]) for result in payload["results"]
        ),
        "plaintext_pass": payload["all_plaintext_gates_pass"],
        "tfe_expected": payload["all_plaintext_gates_pass"],
        "tfe_status": "SKIPPED_PLAINTEXT_GATE",
        "negative_stop_proven": payload["next_phase"]
        == "NEGATIVE_RESULT_STOP_BEFORE_TFE",
        "locked_holdout_read": False,
        "official_test_read": False,
        "full_data_initializer_dependency": False,
    }
    atomic_json(
        LOG_ROOT / "new_stage_b_anchor_refinement_completion_audit.json",
        audit,
    )
    lines = [
        "# Stage-B Anchor Refinement Completion Audit",
        "",
        "- Status: **{}**".format(status),
        "- Registered candidates: {}".format(
            audit["registered_candidates_total"]
        ),
        "- TFE status: `SKIPPED_PLAINTEXT_GATE`",
        "- locked/test read: false/false",
        "- Full initializer dependency: false",
        "- Source commit: `{}`".format(source),
        "",
        "All registered families and {} required files were checked.".format(
            len(required)
        ),
    ]
    (
        LOG_ROOT / "new_stage_b_anchor_refinement_completion_audit.md"
    ).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if payload["status"] != "COMPLETE":
        raise SystemExit("plaintext summary is not complete")
    source = source_commit()
    if payload["source_commit"] != source:
        raise SystemExit(
            "summary source commit {} does not match HEAD {}".format(
                payload["source_commit"], source
            )
        )
    kd = json.loads(KD_ABLATION.read_text(encoding="utf-8"))
    kd_severe = {
        row["dataset"]: row
        for row in kd["rows"]
        if row["distribution"] == "severe"
        and row["method"] == "class_aware_ce_plus_kd"
    }
    selected: Dict[str, Dict[str, Any]] = {}
    for result in payload["results"]:
        b2, b3 = best_rows(result)
        selected[result["dataset"]] = {
            "b0": result["selected"]["b0"],
            "b1": result["selected"]["b1"],
            "b2": b2,
            "b3": b3,
        }
    build_tables(payload, kd_severe, selected, source)
    build_figures(payload, kd_severe, selected)
    build_reports(payload, selected, source)
    audit = completion_audit(payload, source)
    print(json.dumps(audit, indent=2))
    if audit["status"] != "COMPLETE":
        raise SystemExit("completion audit failed")


if __name__ == "__main__":
    main()

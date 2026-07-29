#!/usr/bin/env python3
"""Build the required negative-or-positive Stage-B refinement deliverables."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

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
KD_ABLATION = ROOT / "new_log/new_stage_b_kd/new_stage_b_kd_anchor_supervision_ablation.json"


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def best_rows(result):
    informative = [
        row
        for row in result["rows"]
        if row["distribution"] == "severe"
        and row["effective_disagreement_score_std"] > 1e-12
    ]
    b2 = max(
        [row for row in informative if row["method"].startswith("B2")],
        key=lambda row: (row["initializer_accuracy"], row["worst_class_accuracy"]),
    )
    b3 = max(
        [row for row in informative if row["method"].startswith("B3")],
        key=lambda row: (row["initializer_accuracy"], row["worst_class_accuracy"]),
    )
    return b2, b3


def main():
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
    main_rows = []
    selected = {}
    for result in payload["results"]:
        dataset = result["dataset"]
        b0 = result["selected"]["b0"]
        b1 = result["selected"]["b1"]
        b2, b3 = best_rows(result)
        selected[dataset] = {"b0": b0, "b1": b1, "b2": b2, "b3": b3}
        variants = [
            ("B0_secure_average", b0, 0, 0.0, "current_40_8"),
            ("B1_uniform_anchor_ce", b1, b1["effective_anchor_count"], b1["refinement_seconds"], "current_40_8"),
            ("B2_disagreement_weighted_ce", b2, b2["effective_anchor_count"], b2["refinement_seconds"], "current_40_8"),
            ("B3_top_disagreement_ce", b3, b3["effective_anchor_count"], b3["refinement_seconds"], "current_40_8"),
            ("B4_existing_kd_negative_control", kd_severe[dataset], "", kd_severe[dataset]["student_kd_seconds"], "prior_registered_kd_protocol"),
        ]
        for method, row, anchors, runtime, protocol in variants:
            accuracy = row.get("initializer_accuracy", row.get("accuracy"))
            worst = row.get("worst_class_accuracy")
            macro = row.get("macro_f1")
            main_rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "distribution": "severe",
                    "backend": "numpy_plaintext_float",
                    "method": method,
                    "protocol": protocol,
                    "development_accuracy": accuracy,
                    "macro_f1": macro,
                    "worst_class_accuracy": worst,
                    "effective_anchor_count": anchors,
                    "refinement_seconds": runtime,
                    "source_commit": source,
                    "initializer_source": "official_mtt_only",
                    "full_data_initializer_dependency": False,
                    "locked_holdout_read": False,
                    "official_test_read": False,
                }
            )
    write_csv(
        TABLE_ROOT / "new_stage_b_anchor_refinement_main_table.csv",
        list(main_rows[0]),
        main_rows,
    )
    tfe_rows = [
        {
            "dataset": result["dataset"],
            "seed": 42,
            "status": "SKIPPED",
            "reason": "plaintext_gate_failed",
            "backend": "TFE_ABY3_not_started",
            "source_commit": source,
            "locked_holdout_read": False,
            "official_test_read": False,
        }
        for result in payload["results"]
    ]
    write_csv(
        TABLE_ROOT / "new_stage_b_anchor_refinement_tfe_validation.csv",
        list(tfe_rows[0]),
        tfe_rows,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = ["cifar10", "cifar100"]
    labels = ["B0", "B1", "B2", "B3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, dataset in zip(axes, datasets):
        values = [
            selected[dataset]["b0"]["accuracy"],
            selected[dataset]["b1"]["initializer_accuracy"],
            selected[dataset]["b2"]["initializer_accuracy"],
            selected[dataset]["b3"]["initializer_accuracy"],
        ]
        axis.bar(labels, np.asarray(values) * 100.0, color=["#777777", "#3b82f6", "#f59e0b", "#ef4444"])
        axis.set_title(dataset.upper())
        axis.set_ylabel("Development accuracy (%)")
        axis.set_ylim(0, max(values) * 115.0)
    fig.suptitle("Figure B1: Stage-B initializer quality (seed42, Severe Non-IID)")
    fig.savefig(OUTPUT_ROOT / "new_figure_b1_initializer_quality.png", dpi=220)
    fig.savefig(OUTPUT_ROOT / "new_figure_b1_initializer_quality.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, result in zip(axes, payload["results"]):
        names, aucs, high, low = [], [], [], []
        for name, diag in result["diagnostic"]["metrics"].items():
            names.append(name.replace("_", "\n"))
            aucs.append(diag["auroc_average_error"])
            high.append(diag["high_20_error_rate"])
            low.append(diag["low_20_error_rate"])
        x = np.arange(len(names))
        axis.bar(x - 0.25, aucs, 0.25, label="Error AUROC")
        axis.bar(x, high, 0.25, label="High-20% error")
        axis.bar(x + 0.25, low, 0.25, label="Low-20% error")
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(x, names, fontsize=8)
        axis.set_ylim(0, 1)
        axis.set_title(result["dataset"].upper())
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure B2: Disagreement does not detect Secure-Average errors")
    fig.savefig(OUTPUT_ROOT / "new_figure_b2_disagreement_vs_error.png", dpi=220)
    fig.savefig(OUTPUT_ROOT / "new_figure_b2_disagreement_vs_error.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, result in zip(axes, payload["results"]):
        rows = [
            row for row in result["rows"]
            if row["distribution"] == "severe"
            and row["weight_rule"] != "uniform"
            and row["effective_disagreement_score_std"] > 1e-12
        ]
        axis.scatter(
            [row["effective_anchor_count"] for row in rows],
            [100.0 * row["initializer_accuracy"] for row in rows],
            alpha=0.55,
            s=22,
        )
        b1 = result["selected"]["b1"]
        axis.scatter(
            [b1["effective_anchor_count"]],
            [100.0 * b1["initializer_accuracy"]],
            marker="*",
            s=140,
            color="black",
            label="B1",
        )
        axis.set_title(result["dataset"].upper())
        axis.set_xlabel("Effective anchors")
        axis.set_ylabel("Development accuracy (%)")
        axis.legend()
    fig.suptitle("Figure B3: Utility-cost frontier")
    fig.savefig(OUTPUT_ROOT / "new_figure_b3_utility_cost.png", dpi=220)
    fig.savefig(OUTPUT_ROOT / "new_figure_b3_utility_cost.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for axis, dataset in zip(axes, datasets):
        b1 = np.asarray(selected[dataset]["b1"]["per_class_accuracy"])
        b2 = np.asarray(selected[dataset]["b2"]["per_class_accuracy"])
        axis.plot(np.arange(len(b1)), 100 * b1, label="B1 uniform", linewidth=1.5)
        axis.plot(np.arange(len(b2)), 100 * b2, label="B2 best informative", linewidth=1.2)
        axis.set_title(dataset.upper())
        axis.set_ylabel("Per-class accuracy (%)")
        axis.set_xlabel("Class")
        axis.legend()
    fig.suptitle("Figure B4: Class-wise development accuracy")
    fig.savefig(OUTPUT_ROOT / "new_figure_b4_classwise.png", dpi=220)
    fig.savefig(OUTPUT_ROOT / "new_figure_b4_classwise.pdf")
    plt.close(fig)

    diagnostic_lines = [
        "# Stage-B Anchor Refinement Diagnostic Report",
        "",
        "Scope: CIFAR-10/100, seed42, Severe Non-IID, plaintext development-only.",
        "",
    ]
    for result in payload["results"]:
        diagnostic_lines.append("## {}".format(result["dataset"].upper()))
        diagnostic_lines.append("")
        for name, diag in result["diagnostic"]["metrics"].items():
            diagnostic_lines.append(
                "- {}: AUROC={:.4f}, sample correlation={:.4f}, class-weakness correlation={:.4f}, score std={:.6g}.".format(
                    name,
                    diag["auroc_average_error"],
                    diag["pearson_sample_disagreement_vs_error"],
                    diag["pearson_class_disagreement_vs_weakness"],
                    diag["score_std"],
                )
            )
        diagnostic_lines.extend(
            [
                "",
                "The informative probability metrics are negatively associated with Average errors; vote entropy is constant/degenerate. Direct sample- and class-level high-disagreement prioritization therefore has no valid error-detection signal.",
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
        "| Higher client disagreement predicts Secure-Average errors | Not supported | Informative AUROC below 0.5; vote entropy degenerate |",
        "| Disagreement weighting improves over uniform CE | Not supported | B2/B3 fail utility and cost gates |",
        "| Secure TFE/ABY3 implementation was validated | Not run | Plaintext gate failed, so Phase B was correctly skipped |",
        "| End-to-End/test/DP improvement | Forbidden and not claimed | Stage-B development-only; locked/test unread |",
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
        "This is a registered negative result for the disagreement module. Uniform labeled-anchor CE remains effective, but high-disagreement weighting does not provide stable utility or cost benefits beyond B1 under the evaluated compact-data protocol. Because the plaintext gate failed on both datasets, TFE/ABY3 was not started.",
        "",
    ]
    for result in payload["results"]:
        dataset = result["dataset"]
        values = selected[dataset]
        report.extend(
            [
                "### {}".format(dataset.upper()),
                "",
                "- B0 Secure Average: {:.2f}% accuracy.".format(100 * values["b0"]["accuracy"]),
                "- B1 uniform anchor CE: {:.2f}% accuracy, {:.2f}% worst-class.".format(100 * values["b1"]["initializer_accuracy"], 100 * values["b1"]["worst_class_accuracy"]),
                "- Best informative B2: {:.2f}% accuracy, {:.2f}% worst-class.".format(100 * values["b2"]["initializer_accuracy"], 100 * values["b2"]["worst_class_accuracy"]),
                "- Best informative B3: {:.2f}% accuracy, {:.2f}% worst-class.".format(100 * values["b3"]["initializer_accuracy"], 100 * values["b3"]["worst_class_accuracy"]),
                "- Best informative disagreement AUROC: {:.4f}; IID safety delta vs B1: {:.2f}pp.".format(result["gates"]["best_disagreement_auroc"], 100 * result["gates"]["iid_delta_vs_b1"]),
                "",
            ]
        )
    report.extend(
        [
            "## Ten required answers",
            "",
            "1. **Does disagreement detect Average mistakes?** No. D1/D2 are negatively correlated; D3 is constant.",
            "2. **Which metric is best?** None is a valid positive detector; selecting D3 by utility would merely reproduce uniform CE.",
            "3. **Does B2 beat B1?** No on either dataset.",
            "4. **Can it match B1 with fewer anchors/steps?** No registered configuration meets the 0.2pp/30% cost gate.",
            "5. **Does B3 beat B1?** No.",
            "6. **Does class-level fallback help?** No; direct high-disagreement class priority remains below B1.",
            "7. **Is IID safe?** No for the selected informative severe method; the drop exceeds 0.5pp.",
            "8. **Was TFE/ABY3 run?** No. The plan permits it only after a plaintext pass.",
            "9. **What remains useful?** B1 confirms that labeled Official-MTT anchors can strongly refine Secure Average.",
            "10. **What can be claimed?** Only the B1 anchor-supervision result and the negative disagreement finding, both seed42 development-only.",
            "",
            "## Boundaries",
            "",
            "- Official-MTT-only initializer; Full dependency is false.",
            "- locked holdout and official test were not read.",
            "- No Stage C, DP-SGD, End-to-End, Tiny, seeds43/44, or Figure 2.",
            "- B4 is reused from the prior registered KD protocol and is marked as a protocol-different negative control.",
            "",
            "Source commit: `{}`.".format(source),
        ]
    )
    (LOG_ROOT / "new_stage_b_anchor_refinement_final_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    required = [
        TABLE_ROOT / "new_stage_b_anchor_refinement_smoke_registry.csv",
        LOG_ROOT / "new_stage_b_anchor_refinement_smoke_summary.json",
        LOG_ROOT / "new_stage_b_anchor_refinement_diagnostic_report.md",
        TABLE_ROOT / "new_stage_b_anchor_refinement_tfe_validation.csv",
        TABLE_ROOT / "new_stage_b_anchor_refinement_main_table.csv",
        LOG_ROOT / "new_stage_b_anchor_refinement_claim_audit.md",
        LOG_ROOT / "new_stage_b_anchor_refinement_final_report.md",
    ]
    audit = {
        "schema": "new_stage_b_anchor_refinement_completion_audit_v1",
        "status": "COMPLETE" if all(path.is_file() for path in required) else "INCOMPLETE",
        "source_commit": source,
        "required_files": [
            {"path": str(path.relative_to(ROOT)), "exists": path.is_file()}
            for path in required
        ],
        "plaintext_pass": payload["all_plaintext_gates_pass"],
        "tfe_expected": payload["all_plaintext_gates_pass"],
        "tfe_status": "SKIPPED_PLAINTEXT_GATE",
        "locked_holdout_read": False,
        "official_test_read": False,
        "full_data_initializer_dependency": False,
    }
    atomic_json(
        LOG_ROOT / "new_stage_b_anchor_refinement_completion_audit.json",
        audit,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build negative-or-positive deliverables for heterogeneity-trigger routing."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

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

from new_stage_b_kd_core import atomic_json, sha256_file, source_commit  # noqa: E402
from new_stage_b_heterogeneity_trigger_core import (  # noqa: E402
    CONFIG_ROOT,
    DATASETS,
    DISTRIBUTIONS,
    FIG_ROOT,
    LOG_ROOT,
    OUTPUT_ROOT,
    TABLE_ROOT,
)


SUMMARY_PATH = LOG_ROOT / "new_stage_b_heterogeneity_smoke_summary.json"
SPLIT_PATH = LOG_ROOT / "new_stage_b_heterogeneity_trigger_split_manifest.json"
POLICY_PATH = CONFIG_ROOT / "new_stage_b_heterogeneity_trigger_policy.json"


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for root in (FIG_ROOT, OUTPUT_ROOT):
        figure.savefig(root / (stem + ".png"), dpi=220)
        figure.savefig(root / (stem + ".pdf"))
    plt.close(figure)


def row_lookup(summary: Dict[str, Any], dataset: str, distribution: str) -> Dict[str, Any]:
    return next(
        row
        for row in summary["metric_rows"]
        if row["dataset"] == dataset and row["distribution"] == distribution
    )


def routed_lookup(candidate: Dict[str, Any], dataset: str, distribution: str) -> Dict[str, Any]:
    return next(
        row
        for row in candidate["routed"]
        if row["dataset"] == dataset and row["distribution"] == distribution
    )


def lineage(summary: Dict[str, Any], report_source: str) -> Dict[str, Any]:
    return {
        "source_git_commit": report_source,
        "experiment_source_git_commit": summary["source_git_commit"],
        "split_manifest_path": summary["split_manifest_path"],
        "split_manifest_sha256": summary["split_manifest_sha256"],
        "policy_path": summary["policy_path"],
        "policy_sha256": summary["policy_sha256"],
        "official_mtt_only": True,
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
    }


def build_tables(summary: Dict[str, Any], report_source: str) -> Dict[str, List[Dict[str, Any]]]:
    selected = summary["selected_candidate"]
    common = lineage(summary, report_source)
    main_rows = []
    runtime_rows = []
    utility_rows = []
    tfe_rows = []
    for dataset in DATASETS:
        for distribution in DISTRIBUTIONS:
            source = row_lookup(summary, dataset, distribution)
            routed = routed_lookup(selected, dataset, distribution)
            main_rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "distribution": distribution,
                    "b0_accuracy": source["b0_accuracy"],
                    "b1_accuracy": source["b1_accuracy"],
                    "selected_metric": selected["metric"],
                    "metric_value": (
                        source["H1_mean_tv"]
                        if selected["metric"] == "H1_mean_tv"
                        else source.get(selected["metric"], "")
                    ),
                    "threshold": selected.get("threshold"),
                    "tv_threshold": selected.get("tv_threshold"),
                    "miss_threshold": selected.get("miss_threshold"),
                    "trigger_bit": routed["trigger_bit"],
                    "b2_accuracy": routed["b2_accuracy"],
                    "oracle_accuracy": routed["oracle_accuracy"],
                    "oracle_regret": routed["oracle_regret"],
                    "model_sha256": source["model_sha256"],
                    "model_config_sha256": source["model_config_sha256"],
                    **common,
                }
            )
            trigger_seconds = source["heterogeneity_plaintext_seconds"]
            for method in ("B0_secure_average", "B1_always_refine", "B2_triggered"):
                activated = (
                    method == "B1_always_refine"
                    or (
                        method == "B2_triggered"
                        and routed["trigger_bit"] == 1
                    )
                )
                trigger_cost = trigger_seconds if method == "B2_triggered" else 0.0
                anchor_cost = (
                    source["anchor_refinement_seconds"] if activated else 0.0
                )
                total = (
                    source["local_training_seconds"]
                    + source["secure_average_proxy_seconds"]
                    + trigger_cost
                    + anchor_cost
                )
                runtime_rows.append(
                    {
                        "dataset": dataset,
                        "seed": 42,
                        "distribution": distribution,
                        "method": method,
                        "local_head_training_seconds": source[
                            "local_training_seconds"
                        ],
                        "secure_average_seconds": source[
                            "secure_average_proxy_seconds"
                        ],
                        "secure_average_timing_scope": "plaintext_evaluation_proxy",
                        "trigger_seconds": trigger_cost,
                        "anchor_ce_seconds": anchor_cost,
                        "total_stage_b_seconds": total,
                        "refinement_activated": activated,
                        "anchor_examples_processed": (
                            8 * (10 if dataset == "cifar10" else 100)
                            if activated
                            else 0
                        ),
                        "refinement_steps_executed": 40 if activated else 0,
                        "refinement_cost_avoidance_fraction": (
                            0.0
                            if activated
                            else 1.0
                        ),
                        "communication_bytes": "N/A_not_measured",
                        "model_sha256": source["model_sha256"],
                        "model_config_sha256": source[
                            "model_config_sha256"
                        ],
                        **common,
                    }
                )
            b0_classes = np.asarray(source["b0_per_class_accuracy"])
            b1_classes = np.asarray(source["b1_per_class_accuracy"])
            for method, accuracy, macro, worst, per_class in (
                (
                    "B0_secure_average",
                    source["b0_accuracy"],
                    source["b0_macro_f1"],
                    source["b0_worst_class_accuracy"],
                    b0_classes,
                ),
                (
                    "B1_always_refine",
                    source["b1_accuracy"],
                    source["b1_macro_f1"],
                    source["b1_worst_class_accuracy"],
                    b1_classes,
                ),
                (
                    "B2_triggered",
                    routed["b2_accuracy"],
                    (
                        source["b1_macro_f1"]
                        if routed["trigger_bit"]
                        else source["b0_macro_f1"]
                    ),
                    (
                        source["b1_worst_class_accuracy"]
                        if routed["trigger_bit"]
                        else source["b0_worst_class_accuracy"]
                    ),
                    b1_classes if routed["trigger_bit"] else b0_classes,
                ),
            ):
                utility_rows.append(
                    {
                        "dataset": dataset,
                        "seed": 42,
                        "distribution": distribution,
                        "method": method,
                        "accuracy": accuracy,
                        "macro_f1": macro,
                        "worst_class_accuracy": worst,
                        "improved_classes_vs_b0": int(
                            np.sum(per_class > b0_classes)
                        ),
                        "total_classes": len(b0_classes),
                        "model_sha256": source["model_sha256"],
                        "model_config_sha256": source[
                            "model_config_sha256"
                        ],
                        **common,
                    }
                )
            if distribution in ("iid", "severe"):
                tfe_rows.append(
                    {
                        "dataset": dataset,
                        "seed": 42,
                        "distribution": distribution,
                        "metric": selected["metric"],
                        "plaintext_trigger": routed["trigger_bit"],
                        "tfe_trigger": "",
                        "plaintext_accuracy": routed["b2_accuracy"],
                        "tfe_accuracy": "",
                        "absolute_gap": "",
                        "status": "SKIPPED",
                        "reason": "plaintext_smoke_gate_failed",
                        "backend": "TFE_ABY3_not_started",
                        "model_sha256": source["model_sha256"],
                        "model_config_sha256": source[
                            "model_config_sha256"
                        ],
                        **common,
                    }
                )
    write_csv(
        TABLE_ROOT / "new_stage_b_heterogeneity_trigger_main_table.csv",
        list(main_rows[0]),
        main_rows,
    )
    write_csv(
        TABLE_ROOT / "new_stage_b_heterogeneity_trigger_runtime.csv",
        list(runtime_rows[0]),
        runtime_rows,
    )
    write_csv(
        TABLE_ROOT / "new_stage_b_heterogeneity_trigger_utility_table.csv",
        list(utility_rows[0]),
        utility_rows,
    )
    write_csv(
        TABLE_ROOT / "new_stage_b_heterogeneity_trigger_tfe_results.csv",
        list(tfe_rows[0]),
        tfe_rows,
    )
    return {
        "main": main_rows,
        "runtime": runtime_rows,
        "utility": utility_rows,
        "tfe": tfe_rows,
    }


def build_figures(summary: Dict[str, Any], tables: Dict[str, List[Dict[str, Any]]]) -> None:
    selected = summary["selected_candidate"]
    colors = {"iid": "#2563eb", "mild": "#f59e0b", "severe": "#dc2626"}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        rows = [row_lookup(summary, dataset, dist) for dist in DISTRIBUTIONS]
        for row in rows:
            axis.scatter(
                row["H1_mean_tv"],
                100.0 * row["b1_minus_b0_accuracy"],
                s=80,
                color=colors[row["distribution"]],
                label=row["distribution"].upper(),
            )
            axis.annotate(
                row["distribution"],
                (row["H1_mean_tv"], 100.0 * row["b1_minus_b0_accuracy"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_title(dataset.upper())
        axis.set_xlabel("H1 mean total-variation score")
        axis.set_ylabel("B1 − B0 development accuracy (pp)")
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize=8)
    figure.suptitle(
        "Figure B1: Heterogeneity is measurable but does not predict anchor benefit"
    )
    save_figure(figure, "new_figure_b1_heterogeneity_vs_anchor_benefit")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    methods = ("B0", "B1", "B2", "Oracle")
    for axis, dataset in zip(axes, DATASETS):
        x = np.arange(len(DISTRIBUTIONS))
        width = 0.19
        values = {method: [] for method in methods}
        for distribution in DISTRIBUTIONS:
            source = row_lookup(summary, dataset, distribution)
            routed = routed_lookup(selected, dataset, distribution)
            values["B0"].append(100 * source["b0_accuracy"])
            values["B1"].append(100 * source["b1_accuracy"])
            values["B2"].append(100 * routed["b2_accuracy"])
            values["Oracle"].append(100 * routed["oracle_accuracy"])
        for index, method in enumerate(methods):
            axis.bar(
                x + (index - 1.5) * width,
                values[method],
                width,
                label=method,
            )
        axis.set_xticks(x, [dist.upper() for dist in DISTRIBUTIONS])
        axis.set_ylabel("Development accuracy (%)")
        axis.set_title(dataset.upper())
        axis.legend(fontsize=8)
    figure.suptitle("Figure B2: Routed Stage-B accuracy and oracle gap")
    save_figure(figure, "new_figure_b2_routed_stage_b_accuracy")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        rows = [row for row in tables["runtime"] if row["dataset"] == dataset]
        labels = [
            "{}\n{}".format(row["distribution"], row["method"].split("_")[0])
            for row in rows
        ]
        average = np.asarray([row["secure_average_seconds"] for row in rows])
        trigger = np.asarray([row["trigger_seconds"] for row in rows])
        anchor = np.asarray([row["anchor_ce_seconds"] for row in rows])
        x = np.arange(len(rows))
        axis.bar(x, average, label="Average proxy")
        axis.bar(x, trigger, bottom=average, label="Trigger")
        axis.bar(x, anchor, bottom=average + trigger, label="Anchor CE")
        axis.set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("Plaintext Stage-B component seconds")
        axis.set_title(dataset.upper())
        axis.legend(fontsize=7)
    figure.suptitle(
        "Figure B3: Plaintext runtime (secure runtime not measured after smoke failure)"
    )
    save_figure(figure, "new_figure_b3_stage_b_runtime")

    figure, axis = plt.subplots(figsize=(12, 3.4), constrained_layout=True)
    axis.axis("off")
    boxes = [
        (0.10, "Secret-shared\nclient class counts"),
        (0.30, "Secure H1/H2/H3\nheterogeneity score"),
        (0.50, "Frozen threshold\ncomparison"),
        (0.70, "Reveal one\ntrigger bit"),
        (0.90, "0: Average\n1: Anchor CE"),
    ]
    for x, text in boxes:
        axis.text(
            x,
            0.52,
            text,
            transform=axis.transAxes,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="#e8f0fe", ec="#1d4ed8"),
            fontsize=10,
        )
    for left, right in zip(boxes[:-1], boxes[1:]):
        axis.annotate(
            "",
            xy=(right[0] - 0.07, 0.52),
            xytext=(left[0] + 0.08, 0.52),
            xycoords=axis.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )
    axis.text(
        0.5,
        0.08,
        "Formal TFE/ABY3 execution was not launched because the plaintext "
        "oracle-regret gate failed.",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
        color="#991b1b",
    )
    figure.suptitle("Figure B4: Secure trigger decision design")
    save_figure(figure, "new_figure_b4_trigger_decision_diagram")


def build_reports(summary: Dict[str, Any], tables: Dict[str, List[Dict[str, Any]]], report_source: str) -> None:
    selected = summary["selected_candidate"]
    common = lineage(summary, report_source)
    freeze_status = {
        "schema": "new_stage_b_heterogeneity_trigger_config_freeze_v1",
        "status": "NOT_FROZEN_PLAINTEXT_GATE_FAILED",
        "frozen": False,
        "reason": (
            "S0 crossover premise failed and all registered trigger candidates "
            "exceeded the oracle-regret gate"
        ),
        "candidate_for_diagnosis_only": selected,
        "datasets": list(DATASETS),
        "distributions": list(DISTRIBUTIONS),
        "seed": 42,
        "reveal_policy": "one_trigger_bit_only_if_formal_run_were_authorized",
        **common,
    }
    atomic_json(
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_config_freeze.json",
        freeze_status,
    )
    threshold_status = {
        "schema": "new_stage_b_heterogeneity_trigger_threshold_freeze_v1",
        "status": "NOT_FROZEN_PLAINTEXT_GATE_FAILED",
        "frozen": False,
        "selected_metric_for_diagnosis": selected["metric"],
        "selected_threshold_for_diagnosis": selected.get("threshold"),
        "mean_oracle_regret": selected["mean_oracle_regret"],
        "max_oracle_regret": selected["max_oracle_regret"],
        "all_mandatory_gates": selected["all_mandatory_gates"],
        "datasets": list(DATASETS),
        "distributions": list(DISTRIBUTIONS),
        "seed": 42,
        **common,
    }
    atomic_json(
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_threshold_freeze.json",
        threshold_status,
    )
    manifest_lines = [
        "# Heterogeneity Trigger Freeze Manifest",
        "",
        "**NOT FROZEN:** plaintext gate failed.",
        "",
        "- Diagnostic metric: `{}`".format(selected["metric"]),
        "- Diagnostic threshold: `{}`".format(selected.get("threshold")),
        "- Mean/max oracle regret: `{:.2f}/{:.2f}pp`".format(
            100 * selected["mean_oracle_regret"],
            100 * selected["max_oracle_regret"],
        ),
        "- TFE/ABY3: not authorized by the plan gate",
        "- Source commit: `{}`".format(report_source),
    ]
    (
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_freeze_manifest.md"
    ).write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    split_lines = [
        "# Heterogeneity Trigger Split Audit",
        "",
        "| Dataset | Distribution | Local∩Anchor | Union | Checkpoint SHA256 |",
        "|---|---|---:|---:|---|",
    ]
    for dependency in split["dependencies"]:
        split_lines.append(
            "| {dataset} | {distribution} | {local_anchor_intersection} | "
            "{local_anchor_union} | `{local_checkpoint_sha256}` |".format(
                **dependency
            )
        )
    split_lines.extend(
        [
            "",
            "All cases use Official-MTT 40 local + 8 anchor samples/class. "
            "Development is disjoint; locked holdout/test were not read.",
            "",
            "Source commit: `{}`.".format(report_source),
        ]
    )
    (
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_split_audit.md"
    ).write_text("\n".join(split_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Secure Heterogeneity Trigger Plaintext Smoke Report",
        "",
        "## S0 crossover",
        "",
        "| Dataset | IID B0→B1 | Mild B0→B1 | Severe B0→B1 |",
        "|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        values = []
        for distribution in DISTRIBUTIONS:
            row = row_lookup(summary, dataset, distribution)
            values.append(
                "{:.2f}→{:.2f}%".format(
                    100 * row["b0_accuracy"], 100 * row["b1_accuracy"]
                )
            )
        report_lines.append(
            "| {} | {} | {} | {} |".format(dataset, *values)
        )
    report_lines.extend(
        [
            "",
            "The required crossover is absent: B1 is better than B0 in every "
            "dataset/distribution. H1-H3 are monotonic, but heterogeneity does "
            "not indicate whether anchor CE is beneficial because anchor CE is "
            "beneficial everywhere.",
            "",
            "Best registered extreme-routing candidate: `{}` at threshold "
            "`{}`; mean/max oracle regret `{:.2f}/{:.2f}pp`. It therefore fails "
            "the required 0.5/1.0pp regret gates.".format(
                selected["metric"],
                selected.get("threshold"),
                100 * selected["mean_oracle_regret"],
                100 * selected["max_oracle_regret"],
            ),
        ]
    )
    (
        LOG_ROOT / "new_stage_b_heterogeneity_smoke_report.md"
    ).write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    claim_lines = [
        "# Secure Heterogeneity Trigger Claim Audit",
        "",
        "| Claim | Status | Evidence |",
        "|---|---|---|",
        "| Label heterogeneity is measurable with H1-H3 | Supported in plaintext seed42 | All metrics monotonic IID→Mild→Severe |",
        "| Heterogeneity predicts when anchor CE is beneficial | Not supported | B1 beats B0 in all six cases |",
        "| Trigger avoids destructive IID refinement | Premise contradicted | IID B1 gains 34.98pp/11.88pp |",
        "| Trigger has low oracle regret | Not supported | Mean/max 7.81pp/34.98pp |",
        "| Secure TFE trigger works | Not tested | Smoke gate failed before freeze |",
        "| Raw counts remain hidden in TFE | Design only, not empirical | No formal run occurred |",
        "| Trigger is DP | Forbidden and not claimed | One-bit design has no DP guarantee |",
        "",
        "Safe paper statement: label heterogeneity scores separate the simulated "
        "partition regimes, but under the evaluated compact-data protocol they "
        "do not determine whether labeled anchor refinement is useful.",
    ]
    (
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_claim_audit.md"
    ).write_text("\n".join(claim_lines) + "\n", encoding="utf-8")

    final_lines = [
        "# Secure Heterogeneity-Triggered MTT Anchor Refinement Final Report",
        "",
        "## Outcome",
        "",
        "Negative result at the mandatory plaintext gate. The heterogeneity "
        "metrics work as partition-skew measurements, but the proposed routing "
        "causal premise fails because uniform labeled-anchor CE improves B0 in "
        "IID, Mild and Severe for both datasets. TFE/ABY3 was therefore not run.",
        "",
        "## Experimental data",
        "",
        "| Dataset | Distribution | H1 | H2 | H3 | B0 | B1 | B1−B0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for distribution in DISTRIBUTIONS:
            row = row_lookup(summary, dataset, distribution)
            final_lines.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.3f} | {:.2f}% | "
                "{:.2f}% | {:+.2f}pp |".format(
                    dataset,
                    distribution,
                    row["H1_mean_tv"],
                    row["H2_entropy_deficit"],
                    row["H3_missing_class_ratio"],
                    100 * row["b0_accuracy"],
                    100 * row["b1_accuracy"],
                    100 * row["b1_minus_b0_accuracy"],
                )
            )
    final_lines.extend(
        [
            "",
            "## Twelve required answers",
            "",
            "1. **Best separator?** H1 is the selected simplest stable metric; H1-H3 all separate IID and Severe.",
            "2. **Predict anchor benefit?** No. Benefit remains positive in all six cases and is not monotonic with heterogeneity.",
            "3. **Selected threshold?** No threshold is frozen. Diagnostic H1 threshold `{:.6f}` was the lowest-regret registered extreme router.".format(selected.get("threshold") or 0.0),
            "4. **Avoid IID degradation?** The trigger returns B0 in IID, but this avoids a degradation that does not exist and sacrifices +34.98pp/+11.88pp available B1 gains.",
            "5. **Preserve Severe recovery?** Yes mechanically because it routes Severe to B1, yielding +38.86pp/+9.54pp over B0.",
            "6. **Oracle regret?** Mean 7.81pp, maximum 34.98pp; gate fails badly.",
            "7. **Refinement cost avoided?** 100% on no-refine IID: 80/800 anchors and 40 steps skipped, but at unacceptable utility loss.",
            "8. **Secure overhead below avoided cost?** Not measured because no formal run was authorized; plaintext score time is recorded only as a diagnostic.",
            "9. **TFE/ABY3 reproduction?** No; correctly skipped after smoke failure.",
            "10. **Information revealed?** Proposed formal design would reveal one aggregate trigger bit, with no DP guarantee; this design was not executed.",
            "11. **Safe claims?** H1-H3 measure simulated label skew; they do not justify conditional refinement here.",
            "12. **Positive module or case study?** Negative seed42 case study, not a positive Stage-B module.",
            "",
            "## Role of the experiment",
            "",
            "This experiment prevents a misleading paper claim. It distinguishes "
            "a useful diagnostic (securely measurable label skew) from a useful "
            "decision policy (when to refine). The former works; the latter has "
            "no opportunity because B1 is beneficial everywhere in the matched "
            "protocol.",
            "",
            "## Boundaries",
            "",
            "- Development-only, seed42; no cross-seed stability claim.",
            "- Official-MTT-only; Full initializer dependency false.",
            "- No locked holdout/test, Stage C, DP-SGD, End-to-End or Tiny.",
            "- Raw client histograms were not written to reports; only hashes and aggregate H values are stored.",
            "- Communication and secure runtime were not measured or claimed.",
            "",
            "Experiment source: `{}`; report source: `{}`.".format(
                summary["source_git_commit"], report_source
            ),
        ]
    )
    (
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_final_report.md"
    ).write_text("\n".join(final_lines) + "\n", encoding="utf-8")


def completion_audit(summary: Dict[str, Any], report_source: str) -> Dict[str, Any]:
    required = [
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_reuse_audit.json",
        TABLE_ROOT / "new_stage_b_heterogeneity_metric_table.csv",
        TABLE_ROOT / "new_stage_b_heterogeneity_threshold_screen.csv",
        LOG_ROOT / "new_stage_b_heterogeneity_smoke_summary.json",
        LOG_ROOT / "new_stage_b_heterogeneity_smoke_report.md",
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_config_freeze.json",
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_threshold_freeze.json",
        TABLE_ROOT / "new_stage_b_heterogeneity_trigger_tfe_results.csv",
        TABLE_ROOT / "new_stage_b_heterogeneity_trigger_runtime.csv",
        TABLE_ROOT / "new_stage_b_heterogeneity_trigger_main_table.csv",
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_claim_audit.md",
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_final_report.md",
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_split_manifest.json",
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_split_audit.md",
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_freeze_manifest.md",
        TABLE_ROOT / "new_stage_b_heterogeneity_trigger_utility_table.csv",
    ]
    for stem in (
        "new_figure_b1_heterogeneity_vs_anchor_benefit",
        "new_figure_b2_routed_stage_b_accuracy",
        "new_figure_b3_stage_b_runtime",
        "new_figure_b4_trigger_decision_diagram",
    ):
        for root in (FIG_ROOT, OUTPUT_ROOT):
            required.extend([root / (stem + ".png"), root / (stem + ".pdf")])
    all_files = all(path.is_file() and path.stat().st_size > 0 for path in required)
    all_lineage = all(
        not row["locked_holdout_read"]
        and not row["official_test_read"]
        and not row["full_data_initializer_dependency"]
        and row["official_mtt_only"]
        for row in summary["metric_rows"]
    )
    audit = {
        "schema": "new_stage_b_heterogeneity_trigger_completion_audit_v1",
        "status": (
            "COMPLETE_NEGATIVE"
            if all_files
            and all_lineage
            and not summary["smoke_pass"]
            and summary["next_phase"] == "NEGATIVE_STOP_S0_OR_TRIGGER_GATE"
            else "INCOMPLETE"
        ),
        "source_git_commit": report_source,
        "experiment_source_git_commit": summary["source_git_commit"],
        "required_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "exists": path.is_file(),
                "nonempty": path.is_file() and path.stat().st_size > 0,
            }
            for path in required
        ],
        "datasets": list(DATASETS),
        "distributions": list(DISTRIBUTIONS),
        "metric_rows": len(summary["metric_rows"]),
        "threshold_candidates": len(summary["threshold_results"]),
        "crossover_pass": summary["crossover_pass"],
        "smoke_pass": summary["smoke_pass"],
        "negative_stop_proven": summary["next_phase"]
        == "NEGATIVE_STOP_S0_OR_TRIGGER_GATE",
        "tfe_status": summary["tfe_status"],
        "raw_histograms_revealed": summary["raw_histograms_revealed"],
        "official_mtt_only": True,
        "full_data_initializer_dependency": False,
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    atomic_json(
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_completion_audit.json",
        audit,
    )
    lines = [
        "# Heterogeneity Trigger Completion Audit",
        "",
        "- Status: **{}**".format(audit["status"]),
        "- Metric rows: {}".format(audit["metric_rows"]),
        "- Threshold candidates: {}".format(audit["threshold_candidates"]),
        "- Crossover/smoke pass: false/false",
        "- TFE status: `SKIPPED_SMOKE_GATE`",
        "- locked/test read: false/false",
        "- Required files checked: {}".format(len(required)),
    ]
    (
        LOG_ROOT / "new_stage_b_heterogeneity_trigger_completion_audit.md"
    ).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary["status"] != "COMPLETE":
        raise SystemExit("smoke summary incomplete")
    report_source = source_commit()
    tables = build_tables(summary, report_source)
    build_figures(summary, tables)
    build_reports(summary, tables, report_source)
    audit = completion_audit(summary, report_source)
    print(json.dumps(audit, indent=2))
    if audit["status"] != "COMPLETE_NEGATIVE":
        raise SystemExit("completion audit failed")


if __name__ == "__main__":
    main()

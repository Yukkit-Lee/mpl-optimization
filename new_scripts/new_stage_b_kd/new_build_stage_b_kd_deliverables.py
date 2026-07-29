#!/usr/bin/env python3
"""Build audited Stage-B KD tables, figures, reports, and remaining-seed plan.

This entry point intentionally refuses to run until the complete seed42
tiny->CIFAR-10->CIFAR-100 TFE ladder passes.  It never reads a locked holdout
or official test asset.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from new_stage_b_kd_core import evaluate_head, load_dataset_assets


ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "new_log/new_stage_b_kd"
TABLE = ROOT / "new_expTab/new_stage_b_kd"
FIGURE = ROOT / "new_expFig/new_stage_b_kd"
REPORT = ROOT / "reportToMe"
PLAN = ROOT / "plan"

DATASETS = ("cifar10", "cifar100")
DISPLAY = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
DIST_ORDER = ("iid", "mild", "severe")
DIST_DISPLAY = {"iid": "IID", "mild": "Mild", "severe": "Severe"}
METHOD_DISPLAY = {
    "secure_average_equivalent_plaintext": "Secure Average",
    "uniform_logit_kd": "Uniform Logit KD",
    "class_aware_consensus_kd": "Class-Aware Consensus KD",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def as_float(value):
    if value in (None, ""):
        return None
    return float(value)


def pct(value):
    return "{:.2f}%".format(100.0 * value)


def pp(value):
    return "{:+.2f}pp".format(100.0 * value)


def require_complete_tfe():
    path = LOG / "new_stage_b_kd_tfe_validation.json"
    payload = load_json(path)
    rows = {row["dataset"]: row for row in payload["rows"]}
    required = {"tiny_synthetic", "cifar10", "cifar100"}
    missing = required - set(rows)
    failed = {
        dataset
        for dataset in required & set(rows)
        if not rows[dataset].get("correctness_pass")
    }
    if missing or failed:
        raise SystemExit(
            "REFUSE_PARTIAL_TFE missing={} failed={}".format(
                sorted(missing), sorted(failed)
            )
        )
    if payload.get("seed_policy") != [42]:
        raise SystemExit("unexpected seed policy")
    if (
        payload.get("locked_holdout_read")
        or payload.get("official_test_read")
        or payload.get("full_data_initializer_loaded")
    ):
        raise SystemExit("forbidden lineage/read flag in TFE validation")
    return payload, rows


def index_rows(rows, *keys):
    return {tuple(row[key] for key in keys): row for row in rows}


def build_main_table(confirmation):
    fields = [
        "dataset",
        "seed",
        "distribution",
        "method",
        "initializer_accuracy",
        "macro_f1",
        "worst_class_accuracy",
        "stage_b_runtime_seconds",
        "evaluation_scope",
    ]
    rows = []
    for row in confirmation:
        rows.append(
            {
                **{field: row.get(field, "") for field in fields},
                "evaluation_scope": "development_only_single_seed",
            }
        )
    write_csv(TABLE / "new_stage_b_kd_main_ablation.csv", fields, rows)


def augment_runtime_resource(runtime, tfe_rows):
    runtime_fields = [
        "run_key",
        "dataset",
        "stage",
        "distribution",
        "method",
        "local_training_seconds",
        "anchor_inference_seconds",
        "teacher_aggregation_seconds",
        "student_kd_seconds",
        "secure_graph_build_seconds",
        "total_seconds",
    ]
    combined_runtime = [
        {field: row.get(field, "") for field in runtime_fields}
        for row in runtime
        if row.get("stage") != "TFE"
    ]
    for dataset in ("tiny_synthetic",) + DATASETS:
        validation = tfe_rows[dataset]
        report = load_json(Path(validation["report_path"]))
        combined_runtime.append(
            {
                "run_key": "tfe_seed42_" + dataset,
                "dataset": dataset,
                "stage": "TFE",
                "distribution": (
                    "synthetic_balanced" if dataset == "tiny_synthetic" else "severe"
                ),
                "method": "selected_anchor_ce_plus_kd",
                "local_training_seconds": report.get("local_secure_seconds", ""),
                "anchor_inference_seconds": "N/A",
                "teacher_aggregation_seconds": "N/A",
                "student_kd_seconds": report.get("student_secure_seconds", ""),
                "secure_graph_build_seconds": report.get("graph_build_seconds", ""),
                "total_seconds": report["total_secure_seconds"],
            }
        )
    write_csv(
        TABLE / "new_stage_b_kd_runtime_breakdown.csv",
        runtime_fields,
        combined_runtime,
    )

    resource_fields = [
        "run_key",
        "dataset",
        "stage",
        "method",
        "peak_rss_gib",
        "output_model_sha256",
        "communication_bytes",
        "communication_rounds",
    ]
    resources = load_csv(TABLE / "new_stage_b_kd_resource_summary.csv")
    combined_resources = [
        {field: row.get(field, "") for field in resource_fields}
        for row in resources
        if row.get("stage") != "TFE"
    ]
    for dataset in ("tiny_synthetic",) + DATASETS:
        row = tfe_rows[dataset]
        combined_resources.append(
            {
                "run_key": "tfe_seed42_" + dataset,
                "dataset": dataset,
                "stage": "TFE",
                "method": "selected_anchor_ce_plus_kd",
                "peak_rss_gib": row["peak_rss_gib"],
                "output_model_sha256": row["model_sha256"],
                "communication_bytes": "N/A",
                "communication_rounds": "N/A",
            }
        )
    write_csv(
        TABLE / "new_stage_b_kd_resource_summary.csv",
        resource_fields,
        combined_resources,
    )


def save_figure(fig, stem):
    FIGURE.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE / (stem + ".png"), dpi=240, bbox_inches="tight")
    fig.savefig(FIGURE / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def figure_b1(confirmation):
    lookup = index_rows(confirmation, "dataset", "distribution", "method")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=False)
    styles = [
        ("secure_average_equivalent_plaintext", "o", "#4C78A8"),
        ("uniform_logit_kd", "s", "#F58518"),
        ("class_aware_consensus_kd", "^", "#54A24B"),
    ]
    x = np.arange(len(DIST_ORDER))
    for axis, dataset in zip(axes, DATASETS):
        for method, marker, color in styles:
            values = [
                as_float(lookup[(dataset, distribution, method)]["initializer_accuracy"])
                * 100.0
                for distribution in DIST_ORDER
            ]
            axis.plot(
                x,
                values,
                marker=marker,
                linewidth=2.0,
                markersize=6,
                color=color,
                label=METHOD_DISPLAY[method],
            )
        axis.set_title(DISPLAY[dataset] + " (seed 42)")
        axis.set_xticks(x, [DIST_DISPLAY[item] for item in DIST_ORDER])
        axis.set_ylabel("Stage-B development accuracy (%)")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Figure B1 — Non-IID initializer quality (single seed)")
    save_figure(fig, "new_figureB1_noniid_initializer_quality")


def figure_b2(smoke, causal):
    decisions = {row["dataset"]: row for row in causal["decisions"]}
    selected = {row["dataset"]: row for row in smoke["smoke_results"]}
    labels = [DISPLAY[item] for item in DATASETS]
    disagreement = [
        selected[item]["diversity"]["mean_prediction_disagreement"]
        for item in DATASETS
    ]
    js = [selected[item]["diversity"]["mean_js_divergence"] for item in DATASETS]
    teacher_gain = [
        selected[item]["teacher_severe_gain"] * 100.0 for item in DATASETS
    ]
    ce_gain = [
        decisions[item]["severe_anchor_ce_gain"] * 100.0 for item in DATASETS
    ]
    kd_increment = [
        decisions[item]["class_aware_kd_increment_over_anchor_ce"] * 100.0
        for item in DATASETS
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes[0, 0].bar(labels, disagreement, color="#4C78A8")
    axes[0, 0].set_title("Prediction disagreement")
    axes[0, 0].set_ylim(0, 1.08)
    axes[0, 1].bar(labels, js, color="#72B7B2")
    axes[0, 1].set_title("Mean Jensen–Shannon divergence")
    axes[1, 0].bar(labels, teacher_gain, color="#F58518")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Teacher gain over Average (pp)")
    x = np.arange(len(labels))
    width = 0.35
    axes[1, 1].bar(
        x - width / 2, ce_gain, width, label="Anchor CE over Average", color="#54A24B"
    )
    axes[1, 1].bar(
        x + width / 2,
        kd_increment,
        width,
        label="KD increment over CE",
        color="#E45756",
    )
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Causal gain decomposition (pp)")
    axes[1, 1].legend(fontsize=8, frameon=False)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Figure B2 — Diversity, teacher gain, and causal retention")
    fig.tight_layout()
    save_figure(fig, "new_figureB2_client_diversity_teacher_gain")


def selected_runtime_rows(registry, runtime):
    runtime_by_key = {row["run_key"]: row for row in runtime}
    chosen = {}
    targets = {"cifar10": 0.5514, "cifar100": 0.2344}
    for dataset, target in targets.items():
        candidates = [
            row
            for row in registry["runs"]
            if row["dataset"] == dataset
            and row["stage"] == "S5"
            and row["factor"] == "student_lr"
            and row["student_accuracy"] is not None
            and abs(float(row["student_accuracy"]) - target) < 1e-9
        ]
        if len(candidates) != 1:
            raise SystemExit("cannot resolve selected runtime for " + dataset)
        chosen[dataset] = runtime_by_key[candidates[0]["run_key"]]
    return chosen


def figure_b3(registry, runtime, tfe_rows):
    selected = selected_runtime_rows(registry, runtime)
    categories = [
        "local_training_seconds",
        "anchor_inference_seconds",
        "teacher_aggregation_seconds",
        "student_kd_seconds",
    ]
    category_labels = ["Local training", "Anchor inference", "Teacher aggregation", "Student KD"]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for dataset in DATASETS:
        bottom = 0.0
        for category, label, color in zip(categories, category_labels, colors):
            value = float(selected[dataset][category])
            axes[0].bar(DISPLAY[dataset], value, bottom=bottom, label=label, color=color)
            bottom += value
    axes[0].set_title("Plaintext selected configuration")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].legend(fontsize=7.5, frameon=False)
    for dataset in DATASETS:
        row = tfe_rows[dataset]
        local = float(row["local_secure_seconds"])
        student = float(row["student_secure_seconds"])
        axes[1].bar(DISPLAY[dataset], local, label="Secure local training", color="#4C78A8")
        axes[1].bar(
            DISPLAY[dataset],
            student,
            bottom=local,
            label="Secure student/teacher graph",
            color="#E45756",
        )
    axes[1].set_title("TFE/ABY3 measured secure session")
    axes[1].set_ylabel("Secure runtime (s)")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[:2], labels[:2], fontsize=7.5, frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Figure B3 — Stage-B runtime breakdown (seed 42)")
    fig.tight_layout()
    save_figure(fig, "new_figureB3_runtime_breakdown")


def supplementary_figures(registry, tfe_rows):
    runs = registry["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, dataset in zip(axes, DATASETS):
        rows = [
            row
            for row in runs
            if row["dataset"] == dataset
            and row["stage"] == "S2"
            and row["factor"] == "anchor_per_class"
        ]
        rows.sort(key=lambda row: float(row["factor_value"]))
        axis.plot(
            [float(row["factor_value"]) for row in rows],
            [100.0 * float(row["teacher_accuracy"]) for row in rows],
            marker="o",
        )
        axis.set_title(DISPLAY[dataset])
        axis.set_xlabel("Anchor samples/class")
        axis.set_ylabel("Teacher dev accuracy (%)")
        axis.grid(alpha=0.25)
    fig.suptitle("Supplementary — Anchor-size sweep")
    save_figure(fig, "new_figureS1_anchor_size_sweep")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, dataset in zip(axes, DATASETS):
        for factor, marker in (("teacher_temperature", "o"), ("weight_temperature", "s")):
            rows = [
                row
                for row in runs
                if row["dataset"] == dataset
                and row["stage"] == "S3"
                and row["factor"] == factor
            ]
            rows.sort(key=lambda row: float(row["factor_value"]))
            axis.plot(
                [float(row["factor_value"]) for row in rows],
                [100.0 * float(row["teacher_accuracy"]) for row in rows],
                marker=marker,
                label=factor.replace("_", " "),
            )
        axis.set_xscale("log", base=2)
        axis.set_title(DISPLAY[dataset])
        axis.set_xlabel("Temperature")
        axis.set_ylabel("Teacher dev accuracy (%)")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Supplementary — Temperature sweeps")
    save_figure(fig, "new_figureS2_temperature_sweeps")

    comparison_names = [
        "average_weight_max_abs",
        "average_bias_max_abs",
        "teacher_max_abs",
        "class_weights_max_abs",
        "student_weight_max_abs",
        "student_bias_max_abs",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for axis, dataset in zip(axes, DATASETS):
        comparison = tfe_rows[dataset]["comparisons"]
        names = [name for name in comparison_names if name in comparison]
        values = [max(float(comparison[name]), 1e-12) for name in names]
        axis.bar(np.arange(len(names)), values, color="#4C78A8")
        axis.set_yscale("log")
        axis.set_xticks(
            np.arange(len(names)),
            [name.replace("_max_abs", "").replace("_", "\n") for name in names],
            fontsize=7,
        )
        axis.set_title(DISPLAY[dataset])
        axis.set_ylabel("Max absolute difference")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Supplementary — Plaintext P1 vs TFE numerical differences")
    save_figure(fig, "new_figureS3_plaintext_tfe_numerics")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, dataset in zip(axes, DATASETS):
        model = np.load(
            ROOT
            / (
                "new_output/new_stage_b_kd/tfe_validation/{}/seed42/"
                "new_stage_b_kd_tfe_model_{}_seed42.npz"
            ).format(dataset, dataset),
            allow_pickle=False,
        )
        weights = np.asarray(model["class_weights"])
        axis.boxplot([weights[index] for index in range(weights.shape[0])])
        axis.axhline(1.0 / weights.shape[0], color="black", linestyle="--", linewidth=1)
        axis.set_title(DISPLAY[dataset])
        axis.set_xlabel("Client")
        axis.set_ylabel("Class-aware weight")
        model.close()
    fig.suptitle("Supplementary — TFE class-weight distributions")
    save_figure(fig, "new_figureS4_class_weight_distributions")


def build_classwise_accuracy():
    rows = []
    figure_values = {}
    for dataset in DATASETS:
        assets = load_dataset_assets(dataset, 42)
        model_path = ROOT / (
            "new_output/new_stage_b_kd/tfe_validation/{}/seed42/"
            "new_stage_b_kd_tfe_model_{}_seed42.npz"
        ).format(dataset, dataset)
        with np.load(model_path, allow_pickle=False) as model:
            average = evaluate_head(
                assets.x_dev,
                assets.y_dev,
                model["average_weight"],
                model["average_bias"],
                assets.num_classes,
            )
            student = evaluate_head(
                assets.x_dev,
                assets.y_dev,
                model["student_weight"],
                model["student_bias"],
                assets.num_classes,
            )
        average_values = average["per_class_accuracy"]
        student_values = student["per_class_accuracy"]
        delta = np.asarray(student_values) - np.asarray(average_values)
        figure_values[dataset] = delta
        for class_id, (left, right, difference) in enumerate(
            zip(average_values, student_values, delta)
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "class_id": class_id,
                    "tfe_average_accuracy": left,
                    "tfe_student_accuracy": right,
                    "student_minus_average": difference,
                    "evaluation_scope": "development_only",
                }
            )
    fields = [
        "dataset",
        "seed",
        "class_id",
        "tfe_average_accuracy",
        "tfe_student_accuracy",
        "student_minus_average",
        "evaluation_scope",
    ]
    write_csv(TABLE / "new_stage_b_kd_classwise_accuracy.csv", fields, rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for axis, dataset in zip(axes, DATASETS):
        delta = 100.0 * figure_values[dataset]
        axis.bar(np.arange(len(delta)), delta, color=np.where(delta >= 0, "#54A24B", "#E45756"))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(
            "{}: {}/{} classes improved".format(
                DISPLAY[dataset], int(np.sum(delta > 0)), len(delta)
            )
        )
        axis.set_xlabel("Class ID")
        axis.set_ylabel("Student − Average (pp)")
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Supplementary — TFE class-wise initializer change")
    save_figure(fig, "new_figureS5_classwise_accuracy_delta")


def build_reports(smoke, causal, tfe_rows):
    selected = {row["dataset"]: row for row in smoke["smoke_results"]}
    decisions = {row["dataset"]: row for row in causal["decisions"]}
    diagnostic = [
        "# Stage-B MTT-Anchor KD/Consensus Diagnostic Report",
        "",
        "Scope: seed42 only; development-only; Official-MTT-only; Stage B only.",
        "",
        "## Search and diagnosis",
        "",
        "- S1--S6 completed with 89 registered plaintext candidates.",
        "- Client heads showed non-zero parameter/JS diversity and maximal hard prediction disagreement.",
        "- The mechanical teacher/student gates passed, but student gain was much larger than teacher gain.",
        "- A pre-method causal decomposition therefore compared anchor CE-only, KD-only, and CE+KD under the frozen selected budget.",
        "",
        "## Selected plaintext results",
        "",
        "| Dataset | Average | Teacher | CE+KD student | Teacher gain | Student gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        row = selected[dataset]
        diagnostic.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                DISPLAY[dataset],
                pct(row["baseline_severe_accuracy"]),
                pct(row["teacher_severe_accuracy"]),
                pct(row["student_severe_accuracy"]),
                pp(row["teacher_severe_gain"]),
                pp(row["student_severe_gain"]),
            )
        )
    diagnostic += [
        "",
        "## Causal finding",
        "",
        "| Dataset | CE-only gain | KD increment over CE | Class-aware vs uniform student | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in DATASETS:
        row = decisions[dataset]
        diagnostic.append(
            "| {} | {} | {} | {} | anchor supervision dominated |".format(
                DISPLAY[dataset],
                pp(row["severe_anchor_ce_gain"]),
                pp(row["class_aware_kd_increment_over_anchor_ce"]),
                pp(row["class_aware_vs_uniform_student_delta"]),
            )
        )
    diagnostic += [
        "",
        "The registered evidence supports labeled anchor supervision, not an independent",
        "class-aware Consensus/KD contribution. KD-only never exceeded Average.",
        "",
        "## Security and lineage",
        "",
        "- TFE/ABY3, fixed-point precision 16, custom ops required, no plaintext fallback.",
        "- CIFAR-10/100 secure rows passed finite, direction/utility, lineage, and unread-test gates.",
        "- Locked holdout and official test were not read; no Full-Data initializer was loaded.",
    ]
    write_text(LOG / "new_stage_b_kd_diagnostic_report.md", "\n".join(diagnostic))

    claim_rows = [
        ("Clients contain measurable diversity", "SUPPORTED (seed42)", "non-zero cosine/JS diversity"),
        ("Labeled MTT anchors improve the initializer", "SUPPORTED (seed42)", "CE-only gains on both datasets"),
        ("Class-aware teacher beats uniform teacher", "UNSUPPORTED", "0.00/-0.02pp student delta"),
        ("KD is the main source of the student gain", "UNSUPPORTED", "KD-over-CE +0.44/-3.56pp"),
        ("The secure implementation reproduces the selected direction", "SUPPORTED (seed42)", "C10/C100 TFE gates passed"),
        ("The effect is stable across seeds", "UNSUPPORTED", "only seed42 was run"),
        ("Stage C or end-to-end convergence improves", "UNSUPPORTED/NOT TESTED", "forbidden by task scope"),
        ("Released model has end-to-end record DP", "UNSUPPORTED/NOT TESTED", "Stage B is non-DP"),
        ("Communication cost is measured", "UNSUPPORTED", "bytes/rounds remain N/A"),
    ]
    audit = [
        "# Stage-B KD Claim Audit",
        "",
        "| Claim | Status | Evidence boundary |",
        "|---|---|---|",
    ]
    audit += ["| {} | {} | {} |".format(*row) for row in claim_rows]
    audit += [
        "",
        "All claims are single-seed development results. No mean/std, paired-seed",
        "stability, final-test, Stage-C, end-to-end, or DP-convergence claim is made.",
    ]
    write_text(LOG / "new_stage_b_kd_claim_audit.md", "\n".join(audit))

    final = [
        "# Final Report: Stage-B MTT-Anchor KD/Consensus (seed42)",
        "",
        "Status: COMPLETE_SINGLE_SEED_STAGE_B_ONLY",
        "",
        "## Direct answers required by the experiment plan",
        "",
        "1. **Complementary knowledge?** Yes, client heads have measurable non-zero diversity.",
        "2. **Teacher better than Secure Average?** Slightly: +0.16pp on CIFAR-10 and +1.24pp on CIFAR-100 severe Non-IID.",
        "3. **Student preserved teacher gain?** Mechanically yes, but causally the large gain came from labeled anchor CE rather than KD.",
        "4. **Best reliability rule?** CIFAR-10 selected uniform; CIFAR-100 selected confidence, but class-aware output was 0.02pp below uniform.",
        "5. **Help Severe Non-IID?** Anchor-supervised student did; CE-only gained +35.46pp/+18.92pp over Average.",
        "6. **Harm IID?** No for the anchor-supervised student; IID gains were positive on both datasets.",
        "7. **Stable across seeds?** Unknown. Only seed42 was authorized and run.",
        "8. **Reproduced in TFE/ABY3?** Yes for the frozen selected CE+KD path on CIFAR-10/100; this is secure reproduction, not proof of class-aware causality.",
        "9. **Extra Stage-B time?** See the TFE table below; teacher construction is embedded in the student graph and is not separately timed.",
        "10. **Supported claims?** MTT anchor supervision works; real TFE/ABY3 execution reproduces the selected direction.",
        "11. **Unsupported claims?** Independent class-aware Consensus benefit, cross-seed stability, Stage-C/end-to-end speedup, final-test gain, and end-to-end DP.",
        "12. **Paper position?** Keep labeled anchor supervision as a positive optional Stage-B extension; report class-aware Consensus/KD as a negative result, not a positive module.",
        "",
        "## TFE/ABY3 result",
        "",
        "| Dataset | TFE Average | TFE CE+KD student | Gain | Plaintext gain retained | Secure local | Secure student | Total | Peak RSS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        row = tfe_rows[dataset]
        final.append(
            "| {} | {} | {} | {} | {:.1f}% | {:.2f}s | {:.2f}s | {:.2f}s | {:.2f}GiB |".format(
                DISPLAY[dataset],
                pct(row["tfe_average_accuracy"]),
                pct(row["tfe_student_accuracy"]),
                pp(row["tfe_gain"]),
                100.0 * row["gain_retention"],
                row["local_secure_seconds"],
                row["student_secure_seconds"],
                row["total_secure_seconds"],
                row["peak_rss_gib"],
            )
        )
    final += [
        "",
        "## Evidence boundary",
        "",
        "- seed42 only; no cross-seed mean/std or stability claim;",
        "- development evaluation only; locked holdout and official test unread;",
        "- Official-MTT-only, Full-Data initializer dependency false;",
        "- Stage B only; no Stage C, DP-SGD, end-to-end, Figure 2, or T@Target run;",
        "- communication bytes/rounds were not measured and remain N/A.",
        "",
        "## Figures",
        "",
        "- `new_expFig/new_stage_b_kd/new_figureB1_noniid_initializer_quality.{png,pdf}`",
        "- `new_expFig/new_stage_b_kd/new_figureB2_client_diversity_teacher_gain.{png,pdf}`",
        "- `new_expFig/new_stage_b_kd/new_figureB3_runtime_breakdown.{png,pdf}`",
        "- Supplementary S1--S5 and the class-wise CSV in the same result namespace.",
    ]
    write_text(REPORT / "new_stage_b_kd_final_report.md", "\n".join(final))


def build_remaining_seed_plan():
    text = """# Remaining-Seed Plan: Stage-B MTT-Anchor Supervision and KD Audit

## Authorization boundary

This document is generated after the complete seed42 task showed that labeled
Official-MTT anchor data is effective. It is a plan only. It does **not**
authorize or launch seeds 43/44.

## Purpose

Test whether the seed42 causal conclusion is stable:

1. anchor CE-only improves over Secure Average;
2. KD has little or negative incremental value over anchor CE;
3. class-aware weighting does not beat uniform weighting;
4. the selected paths remain numerically reproducible in TFE/ABY3.

## Frozen scope

- datasets: CIFAR-10 and CIFAR-100;
- seeds: 43 and 44 only;
- distributions: IID, Mild, Severe;
- methods: Secure Average, anchor CE-only, uniform KD-only,
  class-aware KD-only, uniform CE+KD, class-aware CE+KD;
- use the seed42-selected hyperparameters without new tuning;
- Official-MTT-only, five clients, disjoint local/anchor/development sets;
- development-only evaluation.

## Execution phases

### Phase R1 — Split manifests

Generate seed43/44 split manifests and verify intersection=0, exhaustive MTT
coverage, independent development data, and no locked/test reads.

### Phase R2 — Plaintext causal confirmation

Run the six frozen methods for IID/Mild/Severe. Primary comparisons are
`CE-only - Average`, `CE+KD - CE-only`, and `Class-aware - Uniform`.
Do not use `Student - Average` as evidence that KD itself works.

### Phase R3 — Cross-seed decision

Report per-seed values, mean, sample standard deviation, and paired deltas over
seeds 42/43/44. Anchor supervision is stable only if at least 2/3 seeds improve
under Severe Non-IID without >0.5pp IID harm. Consensus is positive only if its
increment over both CE-only and uniform is positive on at least 2/3 seeds and
the mean increment is >=1.0pp.

### Phase R4 — TFE validation

Only after the frozen plaintext gate passes, run the minimal TFE/ABY3 finalists
for seeds43/44. Require no fallback, finite numerics, matching lineage/hashes,
and >=80% retention of the corresponding plaintext gain.

### Phase R5 — Reporting

Update the main/causal tables, figures, runtime/resource summary, claim audit,
and final report. Keep communication N/A unless measured.

## Forbidden

No Stage C, Adaptive/Fixed DP-SGD, end-to-end runs, Figure 2, T@Target, final
test, locked holdout, Full-Data initializer, new hyperparameter search, or
claim of end-to-end DP.
"""
    write_text(PLAN / "0729_new_stage_b_kd_remaining_seeds_plan.md", text)


def main():
    _, tfe_rows = require_complete_tfe()
    confirmation = load_csv(TABLE / "new_stage_b_kd_plaintext_confirmation.csv")
    runtime = load_csv(TABLE / "new_stage_b_kd_runtime_breakdown.csv")
    smoke = load_json(LOG / "new_stage_b_kd_smoke_summary.json")
    registry = load_json(LOG / "new_stage_b_kd_smoke_registry.json")
    causal = load_json(LOG / "new_stage_b_kd_anchor_supervision_ablation.json")
    if causal.get("seed_policy") != [42] or not causal.get(
        "anchor_data_effective_all_datasets"
    ):
        raise SystemExit("anchor data effectiveness gate not passed")
    if causal.get("class_aware_consensus_positive_all_datasets"):
        raise SystemExit("unexpected causal conclusion")
    build_main_table(confirmation)
    figure_b1(confirmation)
    figure_b2(smoke, causal)
    figure_b3(registry, runtime, tfe_rows)
    supplementary_figures(registry, tfe_rows)
    build_classwise_accuracy()
    augment_runtime_resource(runtime, tfe_rows)
    build_reports(smoke, causal, tfe_rows)
    build_remaining_seed_plan()
    print("COMPLETE Stage-B KD single-seed deliverables and remaining-seed plan")


if __name__ == "__main__":
    main()

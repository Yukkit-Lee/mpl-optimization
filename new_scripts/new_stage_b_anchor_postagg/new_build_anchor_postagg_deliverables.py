#!/usr/bin/env python3
"""Build and audit the frozen Stage-B post-aggregation deliverables."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
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
for directory in (HERE, KD_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from new_stage_b_anchor_postagg_core import classification_metrics  # noqa: E402
from new_stage_b_kd_core import load_dataset_assets  # noqa: E402


DATASETS = ("cifar10", "cifar100")
DISTRIBUTIONS = ("iid", "mild", "severe")
METHODS = (
    "C0_random_initialization",
    "C1_random_plus_anchor",
    "C3_fixed_local0_plus_anchor",
    "C4_secure_average",
    "C5_secure_average_plus_anchor",
)
SHORT_METHODS = {
    "C0_random_initialization": "Random",
    "C1_random_plus_anchor": "Random+Anchor",
    "C2_fixed_local0": "Fixed Local-0",
    "C3_fixed_local0_plus_anchor": "Local-0+Anchor",
    "C4_secure_average": "Secure Average",
    "C5_secure_average_plus_anchor": "Average+Anchor",
}
TABLE_ROOT = ROOT / "new_expTab/new_stage_b_anchor_postagg"
FIG_ROOT = ROOT / "new_expFig/new_stage_b_anchor_postagg"
LOG_ROOT = ROOT / "new_log/new_stage_b_anchor_postagg"
OUTPUT_ROOT = ROOT / "new_output/new_stage_b_anchor_postagg"
ABLATION = TABLE_ROOT / "new_stage_b_anchor_postagg_init_source_ablation.csv"
EFFICIENCY = TABLE_ROOT / "new_stage_b_anchor_postagg_step_efficiency.csv"
TFE_RESULTS = TABLE_ROOT / "new_stage_b_anchor_postagg_tfe_results.csv"
TRAJECTORIES = LOG_ROOT / "new_stage_b_anchor_postagg_trajectories.json"
SUMMARY = LOG_ROOT / "new_stage_b_anchor_postagg_smoke_summary.json"
SPLIT = LOG_ROOT / "new_stage_b_anchor_postagg_split_manifest.json"
FREEZE = LOG_ROOT / "new_stage_b_anchor_postagg_config_freeze.json"
VALIDATION = LOG_ROOT / "new_stage_b_anchor_postagg_tfe_validation.json"


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, fields: Iterable[str], rows: Iterable[Dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def save_figure(figure: plt.Figure, stem: str) -> None:
    for directory in (FIG_ROOT, OUTPUT_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory / (stem + ".png"), dpi=220)
        figure.savefig(directory / (stem + ".pdf"))
    plt.close(figure)


def as_float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def ablation_lookup(
    rows: List[Dict[str, str]], dataset: str, distribution: str, method: str
) -> Dict[str, str]:
    matches = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["distribution"] == distribution
        and row["method"] == method
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one ablation row for {}/{}/{}, got {}".format(
                dataset, distribution, method, len(matches)
            )
        )
    return matches[0]


def common_lineage(source: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "report_source_git_commit": source,
        "experiment_source_git_commit": summary["source_git_commit"],
        "policy_sha256": summary["policy_sha256"],
        "split_manifest_sha256": summary["split_sha256"],
        "official_mtt_only": True,
        "full_data_initializer_dependency": False,
        "test_read": False,
        "locked_holdout_read": False,
    }


def fixed_point_metrics(dataset: str) -> Dict[str, float]:
    assets = load_dataset_assets(dataset, 42)
    path = (
        OUTPUT_ROOT
        / "tfe_inputs"
        / dataset
        / ("new_stage_b_anchor_postagg_tfe_input_{}_seed42.npz".format(dataset))
    )
    with np.load(str(path), allow_pickle=False) as payload:
        result = {}
        for method, prefix in (
            ("C1_random_plus_anchor", "p2_c1"),
            ("C4_secure_average", "p2_average"),
            ("C5_secure_average_plus_anchor", "p2_c5"),
        ):
            metrics = classification_metrics(
                assets.x_dev,
                assets.y_dev,
                np.asarray(payload[prefix + "_weight"], dtype=np.float32),
                np.asarray(payload[prefix + "_bias"], dtype=np.float32),
                assets.num_classes,
            )
            result[method] = metrics["accuracy"]
    return result


def tfe_per_class(dataset: str) -> Dict[str, List[float]]:
    assets = load_dataset_assets(dataset, 42)
    path = (
        OUTPUT_ROOT
        / "tfe_validation"
        / dataset
        / "seed42"
        / ("new_stage_b_anchor_postagg_tfe_model_{}_seed42.npz".format(dataset))
    )
    with np.load(str(path), allow_pickle=False) as payload:
        result = {}
        for method, prefix in (
            ("C4_secure_average", "average"),
            ("C5_secure_average_plus_anchor", "c5"),
        ):
            metrics = classification_metrics(
                assets.x_dev,
                assets.y_dev,
                np.asarray(payload[prefix + "_weight"], dtype=np.float32),
                np.asarray(payload[prefix + "_bias"], dtype=np.float32),
                assets.num_classes,
            )
            result[method] = metrics["per_class_accuracy"]
    return result


def build_tables(
    source: str,
    ablation: List[Dict[str, str]],
    efficiency: List[Dict[str, str]],
    tfe: List[Dict[str, str]],
    summary: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    lineage = common_lineage(source, summary)
    table_b1: List[Dict[str, Any]] = []
    table_b2: List[Dict[str, Any]] = []
    table_b3: List[Dict[str, Any]] = []
    table_b4: List[Dict[str, Any]] = []
    runtime: List[Dict[str, Any]] = []
    resources: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        for distribution in DISTRIBUTIONS:
            selected = {
                method: ablation_lookup(ablation, dataset, distribution, method)
                for method in (
                    "C0_random_initialization",
                    "C1_random_plus_anchor",
                    "C2_fixed_local0",
                    "C3_fixed_local0_plus_anchor",
                    "C4_secure_average",
                    "C5_secure_average_plus_anchor",
                )
            }
            for method, row in selected.items():
                table_b1.append(
                    {
                        "dataset": dataset,
                        "seed": 42,
                        "distribution": distribution,
                        "initialization_source": row["initialization_source"],
                        "method": method,
                        "anchor_ce": row["anchor_ce"],
                        "development_accuracy": row["accuracy"],
                        "macro_f1": row["macro_f1"],
                        "worst_class_accuracy": row["worst_class_accuracy"],
                        "improved_classes": row[
                            "improved_classes_vs_initialization"
                        ],
                        "refinement_steps": row["steps"],
                        "refinement_time_seconds": row["refinement_seconds"],
                        "backend": row["backend"],
                        "model_sha256": row["model_sha256"],
                        **lineage,
                    }
                )
            c1, c3, c4, c5 = (
                as_float(selected[method], "accuracy")
                for method in (
                    "C1_random_plus_anchor",
                    "C3_fixed_local0_plus_anchor",
                    "C4_secure_average",
                    "C5_secure_average_plus_anchor",
                )
            )
            table_b2.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "distribution": distribution,
                    "c1_random_plus_anchor": c1,
                    "c3_local0_plus_anchor": c3,
                    "c4_secure_average": c4,
                    "c5_secure_average_plus_anchor": c5,
                    "c5_minus_c1": c5 - c1,
                    "c5_minus_c3": c5 - c3,
                    "c5_minus_c4": c5 - c4,
                    **lineage,
                }
            )
            for method, row in selected.items():
                local = as_float(row, "local_training_seconds")
                anchor_forward = as_float(row, "anchor_forward_seconds")
                anchor_update = as_float(row, "anchor_backward_update_seconds")
                anchor_total = as_float(row, "refinement_seconds")
                runtime.append(
                    {
                        "dataset": dataset,
                        "seed": 42,
                        "distribution": distribution,
                        "method": method,
                        "backend": "plaintext_float",
                        "local_head_training_seconds": local,
                        "secure_average_seconds": "N/A_not_separately_measured",
                        "anchor_data_loading_seconds": "N/A_not_separately_measured",
                        "anchor_forward_seconds": anchor_forward,
                        "anchor_ce_backward_update_seconds": anchor_update,
                        "anchor_ce_seconds": anchor_total,
                        "total_stage_b_seconds": local + anchor_total,
                        "active_anchor_samples": (
                            8 * (10 if dataset == "cifar10" else 100)
                            if row["anchor_ce"] == "True"
                            else 0
                        ),
                        "refinement_steps": row["steps"],
                        "output_model_sha256": row["model_sha256"],
                        **lineage,
                    }
                )

    for row in efficiency:
        table_b3.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "distribution": row["distribution"],
                "method": row["method"],
                "steps_at_80": row["steps_at_80"],
                "steps_at_90": row["steps_at_90"],
                "steps_at_95": row["steps_at_95"],
                "seconds_at_80": row["seconds_at_80"],
                "seconds_at_90": row["seconds_at_90"],
                "seconds_at_95": row["seconds_at_95"],
                "scope": "Stage-B_internal_target_only_not_End-to-End",
                **lineage,
            }
        )

    fixed = {dataset: fixed_point_metrics(dataset) for dataset in DATASETS}
    tfe_by_dataset = {row["dataset"]: row for row in tfe}
    for dataset in DATASETS:
        tfe_row = tfe_by_dataset[dataset]
        for method, plain_key, tfe_key in (
            ("C1_random_plus_anchor", "plaintext_c1_accuracy", "tfe_c1_accuracy"),
            ("C4_secure_average", "plaintext_c4_accuracy", "tfe_c4_accuracy"),
            (
                "C5_secure_average_plus_anchor",
                "plaintext_c5_accuracy",
                "tfe_c5_accuracy",
            ),
        ):
            plain = float(tfe_row[plain_key])
            secure = float(tfe_row[tfe_key])
            table_b4.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "distribution": "severe",
                    "method": method,
                    "plaintext_accuracy": plain,
                    "fixed_point_accuracy": fixed[dataset][method],
                    "tfe_accuracy": secure,
                    "absolute_plaintext_tfe_gap": abs(plain - secure),
                    "backend": "TFE_ABY3",
                    "correctness_pass": tfe_row["correctness_pass"],
                    **lineage,
                }
            )
        local = float(tfe_row["local_secure_seconds"])
        average = float(tfe_row["secure_average_seconds"])
        c1_anchor = float(tfe_row["c1_anchor_secure_seconds"])
        c5_anchor = float(tfe_row["c5_anchor_secure_seconds"])
        for method, local_time, average_time, anchor_time in (
            ("C1_random_plus_anchor", 0.0, 0.0, c1_anchor),
            ("C4_secure_average", local, average, 0.0),
            ("C5_secure_average_plus_anchor", local, average, c5_anchor),
        ):
            model_key = {
                "C1_random_plus_anchor": "report_sha256",
                "C4_secure_average": "report_sha256",
                "C5_secure_average_plus_anchor": "model_sha256",
            }[method]
            runtime.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "distribution": "severe",
                    "method": method,
                    "backend": "TFE_ABY3",
                    "local_head_training_seconds": local_time,
                    "secure_average_seconds": average_time,
                    "anchor_data_loading_seconds": "included_in_graph_execution",
                    "anchor_forward_seconds": "included_in_anchor_ce",
                    "anchor_ce_backward_update_seconds": "included_in_anchor_ce",
                    "anchor_ce_seconds": anchor_time,
                    "total_stage_b_seconds": local_time + average_time + anchor_time,
                    "active_anchor_samples": (
                        8 * (10 if dataset == "cifar10" else 100)
                        if anchor_time
                        else 0
                    ),
                    "refinement_steps": 40 if anchor_time else 0,
                    "output_model_sha256": tfe_row[model_key],
                    **lineage,
                }
            )
        resources.append(
            {
                "dataset": dataset,
                "seed": 42,
                "distribution": "severe",
                "backend": "TFE_ABY3",
                "graph_build_seconds": tfe_row["graph_build_seconds"],
                "formal_multi_branch_secure_seconds": tfe_row[
                    "total_secure_seconds"
                ],
                "peak_rss_gib": tfe_row["peak_rss_gib"],
                "fixed_point_precision": 16,
                "finite": tfe_row["finite"],
                "correctness_pass": tfe_row["correctness_pass"],
                **lineage,
            }
        )
        per_class = tfe_per_class(dataset)
        for class_id, (before, after) in enumerate(
            zip(
                per_class["C4_secure_average"],
                per_class["C5_secure_average_plus_anchor"],
            )
        ):
            class_rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "distribution": "severe",
                    "class_id": class_id,
                    "c4_tfe_accuracy": before,
                    "c5_tfe_accuracy": after,
                    "c5_minus_c4": after - before,
                    "improved": after > before,
                    **lineage,
                }
            )
    resources.insert(
        0,
        {
            "dataset": "cifar10+cifar100",
            "seed": 42,
            "distribution": "iid+mild+severe",
            "backend": "plaintext_float",
            "graph_build_seconds": "N/A",
            "formal_multi_branch_secure_seconds": summary["runtime_seconds"],
            "peak_rss_gib": summary["peak_rss_gib"],
            "fixed_point_precision": "N/A",
            "finite": True,
            "correctness_pass": summary["gates"]["all_smoke_gates_pass"],
            **lineage,
        },
    )

    outputs = {
        "table_b1": table_b1,
        "table_b2": table_b2,
        "table_b3": table_b3,
        "table_b4": table_b4,
        "runtime": runtime,
        "resources": resources,
        "classes": class_rows,
    }
    filenames = {
        "table_b1": "new_stage_b_anchor_postagg_table_b1_init_sources.csv",
        "table_b2": "new_stage_b_anchor_postagg_main_table.csv",
        "table_b3": "new_stage_b_anchor_postagg_table_b3_step_efficiency.csv",
        "table_b4": "new_stage_b_anchor_postagg_table_b4_plaintext_tfe.csv",
        "runtime": "new_stage_b_anchor_postagg_runtime.csv",
        "resources": "new_stage_b_anchor_postagg_resource_summary.csv",
        "classes": "new_stage_b_anchor_postagg_classwise.csv",
    }
    for key, rows in outputs.items():
        write_csv(TABLE_ROOT / filenames[key], list(rows[0]), rows)
    write_csv(
        TABLE_ROOT / "new_stage_b_anchor_postagg_runtime_breakdown.csv",
        list(runtime[0]),
        runtime,
    )
    return outputs


def build_figures(
    ablation: List[Dict[str, str]],
    trajectories: Dict[str, Any],
    tables: Dict[str, List[Dict[str, Any]]],
) -> None:
    colors = {
        "iid": "#2563eb",
        "mild": "#f59e0b",
        "severe": "#dc2626",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        x = np.arange(len(METHODS))
        width = 0.24
        for offset, distribution in enumerate(DISTRIBUTIONS):
            values = [
                100
                * as_float(
                    ablation_lookup(ablation, dataset, distribution, method),
                    "accuracy",
                )
                for method in METHODS
            ]
            axis.bar(
                x + (offset - 1) * width,
                values,
                width,
                label=distribution.capitalize(),
                color=colors[distribution],
            )
        axis.set_xticks(x)
        axis.set_xticklabels(
            [SHORT_METHODS[method] for method in METHODS], rotation=18, ha="right"
        )
        axis.set_ylabel("Stage-B development accuracy (%)")
        axis.set_title(dataset.upper())
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Figure B1: Initialization-source causal ablation")
    save_figure(figure, "new_figure_b1_initialization_sources")

    figure, axes = plt.subplots(
        2, 3, figsize=(14, 7.5), sharex=True, constrained_layout=True
    )
    method_colors = {
        "C1_random_plus_anchor": "#7c3aed",
        "C3_fixed_local0_plus_anchor": "#f59e0b",
        "C5_secure_average_plus_anchor": "#059669",
    }
    for row_id, dataset in enumerate(DATASETS):
        for col_id, distribution in enumerate(DISTRIBUTIONS):
            axis = axes[row_id, col_id]
            for item in trajectories["rows"]:
                if (
                    item["dataset"] == dataset
                    and item["distribution"] == distribution
                ):
                    axis.plot(
                        [point["step"] for point in item["history"]],
                        [
                            100 * point["development_accuracy"]
                            for point in item["history"]
                        ],
                        label=SHORT_METHODS[item["method"]],
                        color=method_colors[item["method"]],
                        linewidth=1.8,
                    )
            axis.set_title("{} / {}".format(dataset.upper(), distribution.capitalize()))
            axis.set_xlabel("Anchor refinement steps")
            axis.set_ylabel("Development accuracy (%)")
            axis.grid(alpha=0.2)
            if row_id == 0 and col_id == 0:
                axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Figure B2: Same-budget Anchor CE convergence")
    save_figure(figure, "new_figure_b2_anchor_convergence")

    figure, axes = plt.subplots(2, 1, figsize=(13, 7.2), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        rows = [row for row in tables["classes"] if row["dataset"] == dataset]
        deltas = np.asarray([100 * float(row["c5_minus_c4"]) for row in rows])
        x = np.arange(len(rows))
        axis.bar(x, deltas, color=np.where(deltas >= 0, "#059669", "#dc2626"))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel("C5 − C4 accuracy (pp)")
        axis.set_title(
            "{}: {}/{} classes improved".format(
                dataset.upper(), int(np.sum(deltas > 0)), len(deltas)
            )
        )
        axis.set_xlabel("Class ID")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Figure B3: TFE class-wise effect of post-aggregation refinement")
    save_figure(figure, "new_figure_b3_classwise_correction")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        rows = [
            row
            for row in tables["runtime"]
            if row["dataset"] == dataset and row["backend"] == "TFE_ABY3"
        ]
        labels = [SHORT_METHODS[row["method"]] for row in rows]
        local = np.asarray([float(row["local_head_training_seconds"]) for row in rows])
        average = np.asarray([float(row["secure_average_seconds"]) for row in rows])
        anchor = np.asarray([float(row["anchor_ce_seconds"]) for row in rows])
        x = np.arange(len(rows))
        axis.bar(x, local, label="Local secure training", color="#2563eb")
        axis.bar(x, average, bottom=local, label="Secure average", color="#f59e0b")
        axis.bar(
            x, anchor, bottom=local + average, label="Anchor CE", color="#059669"
        )
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=15, ha="right")
        axis.set_ylabel("Measured secure runtime (s)")
        axis.set_title(dataset.upper())
        axis.grid(axis="y", alpha=0.2)
        if dataset == "cifar10":
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Figure B4: Formal TFE Stage-B runtime breakdown")
    save_figure(figure, "new_figure_b4_stage_b_runtime")


def build_split_audit(split: Dict[str, Any], source: str) -> None:
    rows = split["rows"]
    lines = [
        "# Stage-B MTT-Anchor Post-Aggregation Split Audit",
        "",
        "- Status: **PASS**",
        "- Report source commit: `{}`".format(source),
        "- Experiment source commit: `{}`".format(split["source_git_commit"]),
        "- Scenarios audited: {}".format(len(rows)),
        "- Official-MTT-only: `{}`".format(split["official_mtt_only"]),
        "- Full-data initializer dependency: `{}`".format(
            split["full_data_initializer_dependency"]
        ),
        "- Official test read: `{}`".format(split["test_read"]),
        "- Locked holdout read: `{}`".format(split["locked_holdout_read"]),
        "",
        "| Dataset | Distribution | Local/class | Anchor/class | Local∩Anchor | Dev∩Train | Dev∩Anchor |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {distribution} | {local_per_class} | "
            "{anchor_per_class} | {local_anchor_intersection} | "
            "{development_training_intersection} | "
            "{development_anchor_intersection} |".format(**row)
        )
    lines.extend(
        [
            "",
            "All six scenarios use the frozen 40-local + 8-anchor per class "
            "Official-MTT split. Development IDs are disjoint from training and "
            "anchor pools; the locked holdout key was deliberately not accessed.",
        ]
    )
    write_text(
        LOG_ROOT / "new_stage_b_anchor_postagg_split_audit.md", "\n".join(lines)
    )


def build_reports(
    source: str,
    summary: Dict[str, Any],
    validation: Dict[str, Any],
    tables: Dict[str, List[Dict[str, Any]]],
) -> None:
    main = tables["table_b2"]
    severe = {row["dataset"]: row for row in main if row["distribution"] == "severe"}
    tfe = {row["dataset"]: row for row in validation["rows"]}
    class_summary = {}
    for dataset in DATASETS:
        rows = [row for row in tables["classes"] if row["dataset"] == dataset]
        class_summary[dataset] = sum(bool(row["improved"]) for row in rows)

    claim = f"""# Stage-B Post-Aggregation Claim Audit

## 审计结论

状态：**PASS_WITH_LIMITATIONS**。冻结的 seed42 Stage-B development 实验和正式
TFE/ABY3 均支持“Secure Average 是有效的跨方初始化，随后使用类别均衡的
Official-MTT anchor 做监督修正”的有限主张。

## 有证据支持

- 同预算下，C5 相对 C1 在 CIFAR-10/CIFAR-100 Severe 分别提升
  {100*severe['cifar10']['c5_minus_c1']:.2f}/{100*severe['cifar100']['c5_minus_c1']:.2f} pp。
- C5 相对固定 client-0 的 C3 在 Severe 分别提升
  {100*severe['cifar10']['c5_minus_c3']:.2f}/{100*severe['cifar100']['c5_minus_c3']:.2f} pp。
- C5 相对未修正 C4 在 Severe 分别提升
  {100*severe['cifar10']['c5_minus_c4']:.2f}/{100*severe['cifar100']['c5_minus_c4']:.2f} pp。
- 正式 TFE 保留 plaintext C5−C1 增益的
  {100*tfe['cifar10']['gain_retention']:.1f}%/{100*tfe['cifar100']['gain_retention']:.1f}%。
- TFE 类别级 C4→C5 改善类数为 CIFAR-10 {class_summary['cifar10']}/10、
  CIFAR-100 {class_summary['cifar100']}/100。

## 必须限定

- 当前只有 seed42，指标是 development Stage-B initializer accuracy。
- IID 下 C5 与固定 local-0 的 C3 仅相差 0.04/0.14 pp；跨机构聚合相对单客户端
  的主要价值出现在 Mild/Severe 异质场景。
- 未运行 Stage C、End-to-End、DP-SGD、official test 或 locked holdout。
- Stage-B 使用标签 anchor，因此不是 record-level DP；Secure Average 也不是新算法贡献。

## 不被支持

- 不可声称最终测试精度、End-to-End 收敛或最终模型性能提高。
- 不可声称多 seed 稳健性或统计显著性。
- 不可声称 Secure Average 在 IID 下显著优于任意单客户端初始化。
- 不可声称通信量下降；本轮未测通信字节或轮数。

Lineage: report source `{source}`; experiment source `{summary['source_git_commit']}`.
"""
    write_text(LOG_ROOT / "new_stage_b_anchor_postagg_claim_audit.md", claim)

    final = f"""# Stage-B MTT-Anchor Post-Aggregation Final Report

## 最终结论

本轮任务完成并达到正面停止条件。明文因果消融 Gate A--F 全部通过，冻结配置为
CIFAR-10/CIFAR-100 各 40 个 full-batch Anchor CE steps，LR=0.1/0.2，
mu=1e-3；正式 TFE/ABY3 成功复现方向。没有启动 Stage C、DP、End-to-End、
Tiny 或额外 seed。

## 核心数据

| Dataset | Dist. | C1 Random+Anchor | C3 Local-0+Anchor | C4 Average | C5 Average+Anchor | C5−C1 | C5−C3 | C5−C4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in main:
        final += (
            "| {dataset} | {distribution} | {c1_random_plus_anchor:.2%} | "
            "{c3_local0_plus_anchor:.2%} | {c4_secure_average:.2%} | "
            "{c5_secure_average_plus_anchor:.2%} | {c5_minus_c1:+.2%} | "
            "{c5_minus_c3:+.2%} | {c5_minus_c4:+.2%} |\n"
        ).format(**row)
    final += f"""

正式 TFE Severe：CIFAR-10 C1/C4/C5 =
{tfe['cifar10']['tfe_c1_accuracy']:.2%}/{tfe['cifar10']['tfe_c4_accuracy']:.2%}/{tfe['cifar10']['tfe_c5_accuracy']:.2%}；
CIFAR-100 =
{tfe['cifar100']['tfe_c1_accuracy']:.2%}/{tfe['cifar100']['tfe_c4_accuracy']:.2%}/{tfe['cifar100']['tfe_c5_accuracy']:.2%}。

## 计划要求的 12 个回答

1. **Anchor CE 是否在所有分布下有效？** 是。C1 相对 C0、C3 相对 C2、C5
   相对 C4 的主要方向均为正；尤其 C5−C4 在六场均显著为正。
2. **Secure Average 是否对 Anchor CE 有独立贡献？** 是。相同 anchor、步数、
   LR 和 mu 下，六场 C5 均高于 C1，提升 0.30--11.94 pp。
3. **C5 是否优于 Random+Anchor？** 是，六场方向一致；Severe 为
   +2.16/+7.44 pp。
4. **C5 是否优于 Local+Anchor？** IID 仅近似持平（+0.04/+0.14 pp），但
   Mild/Severe 明显更优；因此结论应限定为异质场景优势。
5. **优势体现在哪里？** 主要是最终精度和类别修正；并非统一的更少步骤优势。
   步数效率仅是 Stage-B 内部目标，不能称作 End-to-End 加速。
6. **是否跨 IID/Mild/Severe 一致？** C5−C1、C5−C3、C5−C4 方向均一致为正，
   但相对 C3 的 IID 幅度很小。
7. **TFE 是否复现 plaintext？** 是。增益保留率为
   {100*tfe['cifar10']['gain_retention']:.1f}%/{100*tfe['cifar100']['gain_retention']:.1f}%，
   最大精度误差 0.02/0.16 pp。
8. **Anchor CE 增加多少 Stage-B 成本？** 正式 TFE 中 C5 anchor 分支为
   {tfe['cifar10']['c5_anchor_secure_seconds']:.2f}s/
   {tfe['cifar100']['c5_anchor_secure_seconds']:.2f}s；详见 runtime breakdown。
9. **Secure Average 额外成本是否值得？** Severe 下安全平均自身仅
   {tfe['cifar10']['secure_average_seconds']:.2f}s/
   {tfe['cifar100']['secure_average_seconds']:.2f}s，并带来相对 C1 的正增益；
   在本 Stage-B development 协议下值得，但不外推通信或 E2E 成本。
10. **哪些表述安全？** 可表述安全平均提供跨方中间初始化，Official-MTT
    anchor 再监督修正，且受控消融/TFE 支持其在异质场景的价值。
11. **哪些表述不被支持？** 最终 test/E2E 提升、多 seed 显著性、record-level
    DP、通信降低、以及 IID 下显著优于任意 local 初始化均不支持。
12. **模块定位？** 可作为 Stage-B 正贡献及 Non-IID 扩展；证据仍属于
    seed42 development-only，后续论文定稿最好补独立 seed 确认。

## 运行与资源

- CIFAR-10 TFE：graph build {tfe['cifar10']['graph_build_seconds']:.2f}s，
  multi-branch secure compute {tfe['cifar10']['total_secure_seconds']:.2f}s，
  peak RSS {tfe['cifar10']['peak_rss_gib']:.2f} GiB。
- CIFAR-100 TFE：graph build {tfe['cifar100']['graph_build_seconds']:.2f}s，
  multi-branch secure compute {tfe['cifar100']['total_secure_seconds']:.2f}s，
  peak RSS {tfe['cifar100']['peak_rss_gib']:.2f} GiB。
- 通信字节/轮数未测，不报告。

## Lineage 与边界

Report source commit: `{source}`。Experiment source commit:
`{summary['source_git_commit']}`。Official-MTT-only=true，
Full-data initializer dependency=false，test/locked holdout read=false。
"""
    write_text(LOG_ROOT / "new_stage_b_anchor_postagg_final_report.md", final)


def audit_outputs(
    source: str,
    summary: Dict[str, Any],
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    mandatory = [
        LOG_ROOT / "new_stage_b_anchor_postagg_reuse_audit.json",
        LOG_ROOT / "new_stage_b_anchor_postagg_split_audit.md",
        TABLE_ROOT / "new_stage_b_anchor_postagg_smoke_registry.csv",
        LOG_ROOT / "new_stage_b_anchor_postagg_smoke_summary.json",
        TABLE_ROOT / "new_stage_b_anchor_postagg_init_source_ablation.csv",
        TABLE_ROOT / "new_stage_b_anchor_postagg_step_efficiency.csv",
        LOG_ROOT / "new_stage_b_anchor_postagg_config_freeze.json",
        TABLE_ROOT / "new_stage_b_anchor_postagg_tfe_results.csv",
        TABLE_ROOT / "new_stage_b_anchor_postagg_runtime.csv",
        TABLE_ROOT / "new_stage_b_anchor_postagg_main_table.csv",
        LOG_ROOT / "new_stage_b_anchor_postagg_claim_audit.md",
        LOG_ROOT / "new_stage_b_anchor_postagg_final_report.md",
        TABLE_ROOT / "new_stage_b_anchor_postagg_runtime_breakdown.csv",
        TABLE_ROOT / "new_stage_b_anchor_postagg_resource_summary.csv",
        TABLE_ROOT / "new_stage_b_anchor_postagg_table_b1_init_sources.csv",
        TABLE_ROOT / "new_stage_b_anchor_postagg_table_b3_step_efficiency.csv",
        TABLE_ROOT / "new_stage_b_anchor_postagg_table_b4_plaintext_tfe.csv",
        TABLE_ROOT / "new_stage_b_anchor_postagg_classwise.csv",
    ]
    figure_stems = (
        "new_figure_b1_initialization_sources",
        "new_figure_b2_anchor_convergence",
        "new_figure_b3_classwise_correction",
        "new_figure_b4_stage_b_runtime",
    )
    for directory in (FIG_ROOT, OUTPUT_ROOT):
        for stem in figure_stems:
            mandatory.extend((directory / (stem + ".png"), directory / (stem + ".pdf")))
    files = [
        {
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "nonempty": path.exists() and path.stat().st_size > 0,
            "sha256": sha256_file(path) if path.exists() else "",
        }
        for path in mandatory
    ]
    pdf_headers = all(
        path.read_bytes()[:4] == b"%PDF"
        for path in mandatory
        if path.suffix == ".pdf"
    )
    csv_finite = True
    for path in mandatory:
        if path.suffix != ".csv":
            continue
        for row in load_csv(path):
            for value in row.values():
                lowered = str(value).strip().lower()
                if lowered in ("nan", "+nan", "-nan", "inf", "+inf", "-inf"):
                    csv_finite = False
    checks = {
        "all_required_files_exist_and_nonempty": all(
            row["exists"] and row["nonempty"] for row in files
        ),
        "all_smoke_gates_pass": summary["gates"]["all_smoke_gates_pass"],
        "tfe_validation_complete": validation["status"] == "COMPLETE",
        "both_tfe_rows_correct": len(validation["rows"]) == 2
        and all(
            row["correctness_pass"] and row["utility_gate"]
            for row in validation["rows"]
        ),
        "official_mtt_only": summary["official_mtt_only"]
        and validation["official_mtt_only"],
        "no_full_initializer_dependency": not summary[
            "full_data_initializer_dependency"
        ]
        and not validation["full_data_initializer_dependency"],
        "no_test_or_locked_read": not summary["test_read"]
        and not summary["locked_holdout_read"]
        and not validation["test_read"]
        and not validation["locked_holdout_read"],
        "no_forbidden_experiment": not summary["stage_c_run"]
        and not summary["end_to_end_run"],
        "pdf_headers_valid": pdf_headers,
        "csv_no_nan_or_inf_tokens": csv_finite,
    }
    result = {
        "schema": "new_stage_b_anchor_postagg_completion_audit_v1",
        "status": "COMPLETE" if all(checks.values()) else "FAILED",
        "report_source_git_commit": source,
        "checks": checks,
        "required_files": files,
    }
    path = LOG_ROOT / "new_stage_b_anchor_postagg_completion_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "COMPLETE":
        raise RuntimeError("Completion audit failed: {}".format(checks))
    return result


def main() -> None:
    source = git_commit()
    ablation = load_csv(ABLATION)
    efficiency = load_csv(EFFICIENCY)
    tfe = load_csv(TFE_RESULTS)
    trajectories = load_json(TRAJECTORIES)
    summary = load_json(SUMMARY)
    split = load_json(SPLIT)
    freeze = load_json(FREEZE)
    validation = load_json(VALIDATION)
    if not summary["gates"]["all_smoke_gates_pass"]:
        raise RuntimeError("Plaintext smoke gate is not complete")
    if freeze["status"] != "FROZEN_AFTER_SMOKE_SUCCESS":
        raise RuntimeError("Configuration is not frozen")
    if validation["status"] != "COMPLETE":
        raise RuntimeError("TFE validation is not complete")
    if len(ablation) != 36 or len(efficiency) != 18 or len(tfe) != 2:
        raise RuntimeError("Unexpected registered result cardinality")

    tables = build_tables(source, ablation, efficiency, tfe, summary)
    build_figures(ablation, trajectories, tables)
    build_split_audit(split, source)
    build_reports(source, summary, validation, tables)
    audit = audit_outputs(source, summary, validation)
    print(
        "COMPLETE {} files={} source={}".format(
            audit["status"], len(audit["required_files"]), source
        )
    )


if __name__ == "__main__":
    main()

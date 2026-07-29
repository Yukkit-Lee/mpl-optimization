#!/usr/bin/env python3
"""Build the minimal 14-task master results, T@Target table, and Figure 2."""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_scripts.new_true_ours.new_run_true_ours_formal_three_seed import (
    atomic_json,
)


VALIDATION_PATH = (
    ROOT / "new_log/new_true_ours/new_true_ours_minimal_validation.json"
)
POLICY_PATH = (
    ROOT
    / "new_configs/new_true_ours/new_true_ours_minimal_correction_policy.json"
)
TARGET_PATH = (
    ROOT / "new_configs/new_true_ours/new_true_ours_target_freeze.json"
)
MASTER_PATH = (
    ROOT / "new_log/new_true_ours/new_true_ours_master_results.json"
)
TABLE_PATH = (
    ROOT
    / "new_expTab/new_true_ours/new_figure2_time_to_target_table.csv"
)
REPORT_PATH = (
    ROOT / "reportToMe/new_true_ours_minimal_correction_report.md"
)
FIGURE_STEM = (
    ROOT
    / "new_expFig/new_true_ours/new_figure2_pea_style_secure_time"
)

DATASETS = ("cifar10", "cifar100", "tiny")
METHODS = ("full_data", "true_ours")
METHOD_LABELS = {"full_data": "Full Data", "true_ours": "True Ours"}
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny": "Tiny-ImageNet",
}
COLORS = {"full_data": "#2C7BB6", "true_ours": "#D7191C"}
TARGET_LABELS = (
    ("80_percent", "T@80"),
    ("90_percent", "T@90"),
    ("95_percent", "T@95"),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_std(values):
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def time_to_target(points, target):
    for point in points:
        if float(point["development_validation_accuracy"]) >= float(target):
            return float(point["cumulative_online_secure_seconds"])
    return None


def load_tasks(validation):
    tasks = {}
    for row in validation["rows"]:
        if row["status"] != "PASS":
            raise ValueError("non-PASS row after complete gate: {}".format(row))
        key = (row["dataset"], row["method"], int(row["seed"]))
        stage_b = read_json(ROOT / row["stage_b_report"])
        stage_c = read_json(ROOT / row["stage_c_report"])
        timeline = read_json(ROOT / row["timeline"])
        client_samples = [
            int(client["num_samples"]) for client in stage_b["client_splits"]
        ]
        tasks[key] = {
            "dataset": row["dataset"],
            "method": row["method"],
            "seed": int(row["seed"]),
            "stage_b_report": row["stage_b_report"],
            "stage_b_report_sha256": row["stage_b_report_sha256"],
            "stage_c_report": row["stage_c_report"],
            "stage_c_report_sha256": row["stage_c_report_sha256"],
            "timeline": row["timeline"],
            "timeline_sha256": row["timeline_sha256"],
            "secure_training_feature_count": int(sum(client_samples)),
            "client_feature_counts": client_samples,
            "stage_b_online_secure_seconds": float(
                timeline["stage_b_online_secure_seconds"]
            ),
            "stage_c_online_secure_seconds": float(
                timeline["stage_c_online_secure_seconds"]
            ),
            "total_online_secure_seconds": float(
                timeline["total_online_secure_seconds"]
            ),
            "conditional_stage_c_epsilon": float(
                stage_c["privacy"][
                    "stage_c_training_feature_record_epsilon"
                ]
            ),
            "end_to_end_record_dp_claimed": bool(
                stage_c["privacy"]["end_to_end_record_dp_claimed"]
            ),
            "final_development_accuracy": float(
                timeline["points"][-1]["development_validation_accuracy"]
            ),
            "points": timeline["points"],
        }
    return tasks


def aggregate_curve(task_rows):
    times = [
        np.asarray(
            [
                float(point["cumulative_online_secure_seconds"])
                for point in task["points"]
            ],
            dtype=np.float64,
        )
        for task in task_rows
    ]
    accuracies = [
        np.asarray(
            [
                float(point["development_validation_accuracy"])
                for point in task["points"]
            ],
            dtype=np.float64,
        )
        for task in task_rows
    ]
    if len(task_rows) == 1:
        return {
            "time_seconds": times[0].tolist(),
            "accuracy_mean": accuracies[0].tolist(),
            "accuracy_sample_std": None,
            "alignment": "native_timeline_n1",
        }
    start = max(float(values[0]) for values in times)
    stop = min(float(values[-1]) for values in times)
    if not (start > 0 and stop > start):
        raise ValueError("no common positive time interval for seed curves")
    grid = np.geomspace(start, stop, num=160)
    aligned = np.vstack(
        [
            np.interp(grid, seed_times, seed_accuracy)
            for seed_times, seed_accuracy in zip(times, accuracies)
        ]
    )
    return {
        "time_seconds": grid.tolist(),
        "accuracy_mean": np.mean(aligned, axis=0).tolist(),
        "accuracy_sample_std": np.std(aligned, axis=0, ddof=1).tolist(),
        "alignment": (
            "linear_interpolation_on_common_log_time_grid_no_extrapolation"
        ),
    }


def build_master(validation, policy, targets, tasks):
    datasets = {}
    table_rows = []
    for dataset in DATASETS:
        datasets[dataset] = {"targets": {}, "methods": {}}
        dataset_targets = targets["datasets"][dataset]["targets"]
        for target_key, target_label in TARGET_LABELS:
            datasets[dataset]["targets"][target_label] = float(
                dataset_targets[target_key]
            )
        for method in METHODS:
            seeds = [int(seed) for seed in policy["expected_seeds"][dataset]]
            task_rows = [tasks[(dataset, method, seed)] for seed in seeds]
            per_seed = []
            for task in task_rows:
                target_times = {}
                for target_key, target_label in TARGET_LABELS:
                    value = time_to_target(
                        task["points"], dataset_targets[target_key]
                    )
                    target_times[target_label] = value
                    table_rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "row_type": "per_seed",
                            "seed": task["seed"],
                            "n": 1,
                            "target": target_label,
                            "target_accuracy": dataset_targets[target_key],
                            "time_seconds": (
                                value if value is not None else "Not Reached"
                            ),
                            "sample_std_seconds": "N/A",
                        }
                    )
                task_copy = dict(task)
                task_copy["time_to_target_seconds"] = target_times
                task_copy.pop("points")
                per_seed.append(task_copy)
            t_summary = {}
            for target_key, target_label in TARGET_LABELS:
                values = [
                    time_to_target(task["points"], dataset_targets[target_key])
                    for task in task_rows
                ]
                reached = [value for value in values if value is not None]
                all_reached = len(reached) == len(values)
                t_summary[target_label] = {
                    "target_accuracy": float(dataset_targets[target_key]),
                    "per_seed_seconds": values,
                    "all_seeds_reached": all_reached,
                    "mean_seconds": (
                        float(np.mean(reached)) if all_reached else None
                    ),
                    "sample_std_seconds": (
                        sample_std(reached) if all_reached else None
                    ),
                    "display": (
                        "{:.2f}".format(float(np.mean(reached)))
                        if all_reached
                        else "Not Reached"
                    ),
                }
                table_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "row_type": "aggregate",
                        "seed": "",
                        "n": len(values),
                        "target": target_label,
                        "target_accuracy": dataset_targets[target_key],
                        "time_seconds": (
                            float(np.mean(reached))
                            if all_reached
                            else "Not Reached"
                        ),
                        "sample_std_seconds": (
                            sample_std(reached)
                            if all_reached and len(reached) > 1
                            else "N/A"
                        ),
                    }
                )
            final_acc = [
                float(task["final_development_accuracy"]) for task in task_rows
            ]
            total_time = [
                float(task["total_online_secure_seconds"]) for task in task_rows
            ]
            counts = [
                int(task["secure_training_feature_count"]) for task in task_rows
            ]
            epsilons = [
                float(task["conditional_stage_c_epsilon"])
                for task in task_rows
            ]
            datasets[dataset]["methods"][method] = {
                "n": len(task_rows),
                "seeds": seeds,
                "per_seed": per_seed,
                "secure_training_feature_count": (
                    counts[0] if len(set(counts)) == 1 else counts
                ),
                "final_development_accuracy_mean": float(np.mean(final_acc)),
                "final_development_accuracy_sample_std": sample_std(final_acc),
                "total_online_secure_seconds_mean": float(np.mean(total_time)),
                "total_online_secure_seconds_sample_std": sample_std(total_time),
                "conditional_stage_c_epsilon_mean": float(np.mean(epsilons)),
                "time_to_target": t_summary,
                "curve": aggregate_curve(task_rows),
                "stage_transition_seconds": [
                    float(task["stage_b_online_secure_seconds"])
                    for task in task_rows
                ],
            }
    return {
        "schema": "new_true_ours_master_results_v1",
        "status": "COMPLETE_DEVELOPMENT_ONLY_14_OF_14",
        "validation_path": str(VALIDATION_PATH.relative_to(ROOT)),
        "validation_sha256": sha256_file(VALIDATION_PATH),
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": sha256_file(POLICY_PATH),
        "target_path": str(TARGET_PATH.relative_to(ROOT)),
        "target_sha256": sha256_file(TARGET_PATH),
        "seed_policy": policy["expected_seeds"],
        "timing_boundary": (
            "Stage-B secure graph/session start through Stage-C final model"
        ),
        "timing_metric": "cumulative_online_secure_seconds",
        "curve_alignment": (
            "C10/C100: linear interpolation on a common log-time grid "
            "within the shared observed interval; no extrapolation. "
            "Tiny: native seed42 timeline."
        ),
        "accuracy_scope": "development_validation_only",
        "privacy_scope": (
            "conditional Stage-C mechanisms only; initializer is "
            "data-dependent non-DP"
        ),
        "locked_holdout_read": False,
        "official_test_read": False,
        "datasets": datasets,
        "table_rows": table_rows,
    }


def write_table(rows):
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "method",
        "row_type",
        "seed",
        "n",
        "target",
        "target_accuracy",
        "time_seconds",
        "sample_std_seconds",
    ]
    with TABLE_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def display_time(summary):
    value = summary["mean_seconds"]
    if value is None:
        return "NR"
    if summary["sample_std_seconds"] is None:
        return "{:.0f}s".format(value)
    return "{:.0f}±{:.0f}s".format(value, summary["sample_std_seconds"])


def draw_figure(master):
    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.6), sharey=False)
    for axis, dataset in zip(axes, DATASETS):
        block = master["datasets"][dataset]
        for method in METHODS:
            result = block["methods"][method]
            curve = result["curve"]
            x = np.asarray(curve["time_seconds"], dtype=np.float64)
            y = np.asarray(curve["accuracy_mean"], dtype=np.float64)
            color = COLORS[method]
            axis.plot(
                x,
                y,
                color=color,
                linewidth=2.2,
                label="{} (n={})".format(
                    METHOD_LABELS[method], result["n"]
                ),
            )
            std = curve["accuracy_sample_std"]
            if std is not None:
                std_values = np.asarray(std, dtype=np.float64)
                axis.fill_between(
                    x,
                    y - std_values,
                    y + std_values,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
            transitions = result["stage_transition_seconds"]
            transition = float(np.median(transitions))
            axis.axvline(
                transition,
                color=color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.7,
            )
            endpoint_std = result[
                "final_development_accuracy_sample_std"
            ]
            acc_text = "{:.1%}".format(
                result["final_development_accuracy_mean"]
            )
            if endpoint_std is not None:
                acc_text += "±{:.1%}".format(endpoint_std)
            annotation = (
                "{}\nAcc {} | features={:,}\nT@90={} | conditional ε={:.3f}"
            ).format(
                METHOD_LABELS[method],
                acc_text,
                result["secure_training_feature_count"],
                display_time(result["time_to_target"]["T@90"]),
                result["conditional_stage_c_epsilon_mean"],
            )
            axis.text(
                0.02 if method == "full_data" else 0.98,
                0.97,
                annotation,
                transform=axis.transAxes,
                ha="left" if method == "full_data" else "right",
                va="top",
                fontsize=7.0,
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.82,
                },
            )
        for target_label, value in block["targets"].items():
            axis.axhline(
                value,
                color="#777777",
                linestyle="--",
                linewidth=0.7,
                alpha=0.45,
            )
            axis.text(
                0.015,
                value,
                target_label,
                transform=axis.get_yaxis_transform(),
                fontsize=7,
                color="#666666",
                va="bottom",
            )
        axis.set_xscale("log")
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_title(DATASET_LABELS[dataset], fontweight="bold")
        axis.set_xlabel("Cumulative online secure runtime (seconds, log scale)")
        axis.grid(True, which="both", alpha=0.18)
        axis.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Development validation accuracy")
    fig.suptitle(
        "Development accuracy versus cumulative online secure runtime",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        (
            "Dotted vertical lines: Stage B→C transition. Shading: sample "
            "standard deviation (CIFAR only). ε is conditional Stage C, "
            "not end-to-end DP."
        ),
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 220
        fig.savefig(str(FIGURE_STEM) + "." + suffix, **kwargs)
    plt.close(fig)
    svg_path = Path(str(FIGURE_STEM) + ".svg")
    normalized_svg = "\n".join(
        line.rstrip()
        for line in svg_path.read_text(encoding="utf-8").splitlines()
    )
    svg_path.write_text(normalized_svg + "\n", encoding="utf-8")


def format_mean_std(mean, std, percent=False):
    if percent:
        if std is None:
            return "{:.2%} (seed42 only; n=1, std N/A)".format(mean)
        return "{:.2%} ± {:.2%}".format(mean, std)
    if std is None:
        return "{:.2f} (seed42 only; n=1, std N/A)".format(mean)
    return "{:.2f} ± {:.2f}".format(mean, std)


def write_report(master):
    lines = [
        "# True Ours minimal correction — development-only formal results",
        "",
        "Status: 14/14 validated after replacing only Tiny Full Data seed42.",
        "",
        "CIFAR-10 and CIFAR-100 use seeds 42/43/44. Tiny-ImageNet uses "
        "seed42 only by user-approved compute policy.",
        "",
        "No locked holdout or official test was read in this correction. "
        "All accuracies below are development-validation accuracies.",
        "",
        "The reported epsilon is conditional on the Stage-C mechanisms. "
        "The Stage-B initializer is data-dependent non-DP, so this is not "
        "an end-to-end record-DP claim.",
        "",
        "## Summary",
        "",
        "| Dataset | Method | N | Final development accuracy | "
        "Online secure time (s) | T@90 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in METHODS:
            result = master["datasets"][dataset]["methods"][method]
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    DATASET_LABELS[dataset],
                    METHOD_LABELS[method],
                    result["n"],
                    format_mean_std(
                        result["final_development_accuracy_mean"],
                        result["final_development_accuracy_sample_std"],
                        percent=True,
                    ),
                    format_mean_std(
                        result["total_online_secure_seconds_mean"],
                        result["total_online_secure_seconds_sample_std"],
                    ),
                    display_time(result["time_to_target"]["T@90"]),
                )
            )
    lines.extend(
        [
            "",
            "## Artifact scope",
            "",
            "- Figure 2 uses cumulative online secure time from Stage B start "
            "through Stage C final model.",
            "- T@Target is the first observed timeline point at or above the "
            "frozen target; otherwise it is `Not Reached`.",
            "- The superseded Tiny Full Data run is excluded.",
            "- Source-unverified direct/final-test accuracy is not used.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    validation = read_json(VALIDATION_PATH)
    if validation.get("status") != "COMPLETE_14_OF_14":
        raise ValueError(
            "14-task gate is not complete: {}".format(validation.get("status"))
        )
    if validation.get("locked_holdout_read") is not False:
        raise ValueError("locked holdout flag is not false")
    if validation.get("official_test_read") is not False:
        raise ValueError("official test flag is not false")
    policy = read_json(POLICY_PATH)
    targets = read_json(TARGET_PATH)
    tasks = load_tasks(validation)
    master = build_master(validation, policy, targets, tasks)
    atomic_json(MASTER_PATH, master)
    write_table(master["table_rows"])
    draw_figure(master)
    write_report(master)
    print("MASTER={}".format(MASTER_PATH))
    print("TABLE={}".format(TABLE_PATH))
    print("FIGURE={}.[png|pdf|svg]".format(FIGURE_STEM))
    print("REPORT={}".format(REPORT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

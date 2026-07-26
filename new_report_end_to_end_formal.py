#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the two-curve End-to-End formal experiment."""

import csv
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from new_public_mtt_mainline_common import ROOT, load_config


SOURCE = ROOT / "new_output/new_public_mtt_mainline/end_to_end_formal"
JSON_OUTPUT = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_end_to_end_formal_report.json"
)
TXT_OUTPUT = (
    ROOT
    / "new_log/new_public_mtt_mainline/txt/new_end_to_end_formal_report.txt"
)
CSV_OUTPUT = (
    ROOT
    / "new_expTab/new_public_mtt_mainline/new_end_to_end_formal_summary.csv"
)
FIGURE_ROOT = ROOT / "new_expFig/new_public_mtt_mainline"
METHODS = [
    "baseline_random_secure_average_fixed",
    "ours_official_secure_consensus_adaptive",
]


def mean_std(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def early_auc(history):
    points = [
        (int(row["step"]), float(row["reference_accuracy"]))
        for row in history
        if "reference_accuracy" in row
    ]
    if len(points) == 1:
        return points[0][1]
    area = sum(
        (right[0] - left[0]) * (right[1] + left[1]) * 0.5
        for left, right in zip(points[:-1], points[1:])
    )
    return area / float(points[-1][0] - points[0][0])


def step_at_target(history, target):
    for row in history:
        if (
            "reference_accuracy" in row
            and float(row["reference_accuracy"]) >= float(target)
        ):
            return int(row["step"])
    return None


def main():
    config = load_config()
    formal = config["end_to_end_formal"]
    targets = config["module_b_convergence_formal"]["target_accuracy"]
    runs = []
    histories = {}
    for dataset in config["datasets"]:
        for method in METHODS:
            for seed in formal["seeds"]:
                directory = SOURCE / dataset / method / "seed{}".format(seed)
                matches = list(directory.glob("new_secure_stage_c_report_*.json"))
                if len(matches) != 1:
                    raise FileNotFoundError(
                        "expected one report in {}; got {}".format(
                            directory, len(matches)
                        )
                    )
                report = json.loads(matches[0].read_text(encoding="utf-8"))
                privacy = report["privacy"]
                if privacy["released_model_record_dp_valid"]:
                    raise ValueError("{} incorrectly claims release DP".format(matches[0]))
                if privacy["epsilon_interpretation"] != (
                    "conditional_gradient_mechanism_only_not_released_model_dp"
                ):
                    raise ValueError("invalid epsilon interpretation")
                achieved = float(
                    privacy["accounted_gradient_mechanism_epsilon"]
                )
                if abs(achieved - float(formal["target_epsilon"])) > 1e-4:
                    raise ValueError("{} misses target accountant epsilon".format(matches[0]))
                logged = [
                    row for row in report["history"] if "reference_accuracy" in row
                ]
                target_step = step_at_target(report["history"], targets[dataset])
                histories.setdefault((dataset, method), []).append(
                    report["history"]
                )
                runs.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "steps": int(privacy["steps"]),
                        "target_accuracy": targets[dataset],
                        "step_at_target": target_step,
                        "time_at_target_seconds": (
                            None
                            if target_step is None
                            else float(report["runtime_seconds"])
                            * target_step
                            / float(privacy["steps"])
                        ),
                        "first_reference_accuracy": float(
                            logged[0]["reference_accuracy"]
                        ),
                        "final_reference_accuracy": float(
                            logged[-1]["reference_accuracy"]
                        ),
                        "early_auc": early_auc(report["history"]),
                        "runtime_seconds": float(report["runtime_seconds"]),
                        "final_c": float(report["final_c"]),
                        "accounted_epsilon": achieved,
                        "released_model_record_dp_valid": False,
                        "test_loaded": False,
                        "source_report": str(matches[0]),
                    }
                )
    groups = []
    for dataset in config["datasets"]:
        for method in METHODS:
            selected = [
                row
                for row in runs
                if row["dataset"] == dataset and row["method"] == method
            ]
            reached = [row for row in selected if row["step_at_target"] is not None]
            groups.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seeds": formal["seeds"],
                    "target_accuracy": targets[dataset],
                    "target_reached_seeds": len(reached),
                    "step_at_target": (
                        mean_std([row["step_at_target"] for row in reached])
                        if reached
                        else "N/A"
                    ),
                    "time_at_target_seconds": (
                        mean_std(
                            [row["time_at_target_seconds"] for row in reached]
                        )
                        if reached
                        else "N/A"
                    ),
                    "final_reference_accuracy": mean_std(
                        [row["final_reference_accuracy"] for row in selected]
                    ),
                    "early_auc": mean_std(
                        [row["early_auc"] for row in selected]
                    ),
                    "runtime_seconds": mean_std(
                        [row["runtime_seconds"] for row in selected]
                    ),
                    "final_c": mean_std([row["final_c"] for row in selected]),
                    "communication_bytes": "N/A",
                    "communication_rounds": "N/A",
                }
            )
    quality = {}
    for dataset in config["datasets"]:
        baseline = next(
            row
            for row in groups
            if row["dataset"] == dataset and row["method"] == METHODS[0]
        )
        ours = next(
            row
            for row in groups
            if row["dataset"] == dataset and row["method"] == METHODS[1]
        )
        accuracy_gain = (
            ours["final_reference_accuracy"]["mean"]
            - baseline["final_reference_accuracy"]["mean"]
        )
        speed_pass = False
        if (
            baseline["step_at_target"] != "N/A"
            and ours["step_at_target"] != "N/A"
        ):
            speed_pass = (
                ours["step_at_target"]["mean"]
                <= 0.8 * baseline["step_at_target"]["mean"]
            )
        quality[dataset] = {
            "ours_minus_baseline_final_accuracy": accuracy_gain,
            "accuracy_gain_target": 0.02,
            "accuracy_gate_passed": accuracy_gain >= 0.02,
            "step_ratio_gate_passed": speed_pass,
            "passed": accuracy_gain >= 0.02 or speed_pass,
        }

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    colors = {METHODS[0]: "#4477AA", METHODS[1]: "#CC6677"}
    labels = {METHODS[0]: "Baseline", METHODS[1]: "Ours"}
    for axis, dataset in zip(axes, config["datasets"]):
        for method in METHODS:
            hs = histories[(dataset, method)]
            points = [
                [row for row in history if "reference_accuracy" in row]
                for history in hs
            ]
            steps = np.asarray([row["step"] for row in points[0]])
            values = np.asarray(
                [
                    [float(row["reference_accuracy"]) for row in history]
                    for history in points
                ]
            )
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            axis.plot(steps, mean, color=colors[method], label=labels[method])
            axis.fill_between(steps, mean - std, mean + std, alpha=0.2)
        axis.set_title(dataset)
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Reference accuracy")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    png = FIGURE_ROOT / "new_figureE_end_to_end_two_curve.png"
    pdf = FIGURE_ROOT / "new_figureE_end_to_end_two_curve.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for row in groups:
        csv_rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "target_reached_seeds": row["target_reached_seeds"],
                "final_accuracy_mean": row["final_reference_accuracy"]["mean"],
                "final_accuracy_std": row["final_reference_accuracy"]["std"],
                "early_auc_mean": row["early_auc"]["mean"],
                "early_auc_std": row["early_auc"]["std"],
                "runtime_mean_seconds": row["runtime_seconds"]["mean"],
                "runtime_std_seconds": row["runtime_seconds"]["std"],
                "communication_bytes": "N/A",
                "communication_rounds": "N/A",
            }
        )
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(csv_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    payload = {
        "schema": "new_end_to_end_formal_report_v1",
        "status": "COMPLETE",
        "formal_result": True,
        "runs_expected": 18,
        "runs_completed": len(runs),
        "test_loaded": False,
        "dp_mtt_status": "EXCLUDED_BY_USER",
        "shared_reference_protocol": True,
        "target_accountant_epsilon": formal["target_epsilon"],
        "privacy_interpretation": (
            "conditional_gradient_mechanism_only_not_released_model_dp"
        ),
        "released_model_record_dp_valid": False,
        "groups": groups,
        "quality_gate": quality,
        "runs": runs,
        "artifacts": {
            "csv": str(CSV_OUTPUT),
            "figure_png": str(png),
            "figure_pdf": str(pdf),
        },
        "limitations": [
            "The frozen portable selected initializer is Secure Consensus Selected, not mislabeled standalone KD-ReferenceWeighted.",
            "Both initializers are data-dependent; epsilon is conditional mechanism accounting only.",
            "Final test remains unread; curves use one shared test-free reference protocol.",
            "Single-machine TFE communication bytes/rounds are N/A.",
        ],
    }
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "End-to-End formal two-curve report",
        "runs=18/18 test_loaded=False shared_reference=True",
    ]
    for row in groups:
        lines.append(
            "{dataset} {method} final={mean:.6f}±{std:.6f}".format(
                mean=row["final_reference_accuracy"]["mean"],
                std=row["final_reference_accuracy"]["std"],
                **row
            )
        )
    for dataset, gate in quality.items():
        lines.append(
            "{} gain={:.6f} passed={}".format(
                dataset,
                gate["ours_minus_baseline_final_accuracy"],
                gate["passed"],
            )
        )
    TXT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TXT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("end_to_end status=COMPLETE runs=18")
    for dataset, gate in quality.items():
        print(
            "{} ours_minus_baseline={:.6f} passed={}".format(
                dataset,
                gate["ours_minus_baseline_final_accuracy"],
                gate["passed"],
            )
        )
    print("json={}".format(JSON_OUTPUT))
    print("figure={}".format(png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

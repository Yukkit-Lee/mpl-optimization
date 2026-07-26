#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the complete formal Module C grid and render paper artifacts."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from new_public_mtt_mainline_common import ROOT, load_config


SOURCE = ROOT / "new_output/new_public_mtt_mainline/module_c_formal"
JSON_OUTPUT = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_module_c_formal_report.json"
)
TXT_OUTPUT = (
    ROOT
    / "new_log/new_public_mtt_mainline/txt/new_module_c_formal_report.txt"
)
CSV_OUTPUT = (
    ROOT
    / "new_expTab/new_public_mtt_mainline/new_module_c_formal_summary.csv"
)
FIGURE_ROOT = ROOT / "new_expFig/new_public_mtt_mainline"


def epsilon_tag(value):
    return "eps{}".format(str(float(value)).replace(".", "p"))


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


def main():
    config = load_config()
    formal = config["module_c_formal"]
    runs = []
    histories = {}
    for dataset in config["datasets"]:
        for mode in formal["modes"]:
            for epsilon in formal["target_epsilons"]:
                tag = epsilon_tag(epsilon)
                for seed in formal["seeds"]:
                    directory = SOURCE / dataset / mode / tag / "seed{}".format(seed)
                    matches = list(
                        directory.glob("new_secure_stage_c_report_*.json")
                    )
                    if len(matches) != 1:
                        raise FileNotFoundError(
                            "expected one report in {}; got {}".format(
                                directory, len(matches)
                            )
                        )
                    report = json.loads(matches[0].read_text(encoding="utf-8"))
                    privacy = report["privacy"]
                    achieved = float(
                        privacy["accounted_gradient_mechanism_epsilon"]
                    )
                    if abs(achieved - float(epsilon)) > 1e-4:
                        raise ValueError("{} misses target epsilon".format(matches[0]))
                    if not privacy["released_model_record_dp_valid"]:
                        raise ValueError("{} has invalid release DP".format(matches[0]))
                    logged = [
                        row
                        for row in report["history"]
                        if "reference_accuracy" in row
                    ]
                    key = (dataset, mode, float(epsilon))
                    histories.setdefault(key, []).append(report["history"])
                    runs.append(
                        {
                            "dataset": dataset,
                            "mode": mode,
                            "target_epsilon": float(epsilon),
                            "achieved_epsilon": achieved,
                            "seed": seed,
                            "steps": int(privacy["steps"]),
                            "count_mechanism_steps": int(
                                privacy["clipping_count_mechanism_steps"]
                            ),
                            "gradient_noise_multiplier": float(
                                privacy["gradient_noise_multiplier"]
                            ),
                            "first_reference_accuracy": float(
                                logged[0]["reference_accuracy"]
                            ),
                            "final_reference_accuracy": float(
                                logged[-1]["reference_accuracy"]
                            ),
                            "early_auc": early_auc(report["history"]),
                            "final_c": float(report["final_c"]),
                            "runtime_seconds": float(report["runtime_seconds"]),
                            "released_model_record_dp_valid": True,
                            "test_loaded": False,
                            "source_report": str(matches[0]),
                        }
                    )
    groups = []
    for dataset in config["datasets"]:
        for mode in formal["modes"]:
            for epsilon in formal["target_epsilons"]:
                selected = [
                    row
                    for row in runs
                    if row["dataset"] == dataset
                    and row["mode"] == mode
                    and row["target_epsilon"] == float(epsilon)
                ]
                groups.append(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "target_epsilon": float(epsilon),
                        "seeds": formal["seeds"],
                        "steps": selected[0]["steps"],
                        "count_mechanism_steps": selected[0][
                            "count_mechanism_steps"
                        ],
                        "gradient_noise_multiplier": selected[0][
                            "gradient_noise_multiplier"
                        ],
                        "achieved_epsilon": mean_std(
                            [row["achieved_epsilon"] for row in selected]
                        ),
                        "final_reference_accuracy": mean_std(
                            [row["final_reference_accuracy"] for row in selected]
                        ),
                        "early_auc": mean_std(
                            [row["early_auc"] for row in selected]
                        ),
                        "final_c": mean_std(
                            [row["final_c"] for row in selected]
                        ),
                        "runtime_seconds": mean_std(
                            [row["runtime_seconds"] for row in selected]
                        ),
                        "communication_bytes": "N/A",
                        "communication_rounds": "N/A",
                    }
                )
    gains = {}
    for dataset in config["datasets"]:
        fixed = next(
            row
            for row in groups
            if row["dataset"] == dataset
            and row["mode"] == "fixed_dp"
            and row["target_epsilon"] == 3.0
        )
        adaptive = next(
            row
            for row in groups
            if row["dataset"] == dataset
            and row["mode"] == "adaptive_dp"
            and row["target_epsilon"] == 3.0
        )
        gain = (
            adaptive["final_reference_accuracy"]["mean"]
            - fixed["final_reference_accuracy"]["mean"]
        )
        gains[dataset] = {
            "adaptive_minus_fixed": gain,
            "target": 0.01,
            "passed": gain >= 0.01,
        }

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    colors = {"fixed_dp": "#4477AA", "adaptive_dp": "#CC6677"}
    labels = {"fixed_dp": "Secure Fixed", "adaptive_dp": "Secure Adaptive"}
    for axis, dataset in zip(axes, config["datasets"]):
        for mode in formal["modes"]:
            selected = [
                row
                for row in groups
                if row["dataset"] == dataset and row["mode"] == mode
            ]
            selected.sort(key=lambda row: row["target_epsilon"])
            axis.errorbar(
                [row["target_epsilon"] for row in selected],
                [row["final_reference_accuracy"]["mean"] for row in selected],
                yerr=[row["final_reference_accuracy"]["std"] for row in selected],
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=colors[mode],
                label=labels[mode],
            )
        axis.set_title(dataset)
        axis.set_xlabel(r"$\epsilon_C$")
        axis.set_ylabel("Final reference accuracy")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    privacy_png = FIGURE_ROOT / "new_figureC_stage_c_privacy_utility.png"
    privacy_pdf = FIGURE_ROOT / "new_figureC_stage_c_privacy_utility.pdf"
    figure.savefig(privacy_png, dpi=220)
    figure.savefig(privacy_pdf)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    for axis, dataset in zip(axes, config["datasets"]):
        selected_histories = histories[(dataset, "adaptive_dp", 3.0)]
        steps = np.asarray(
            [row["step"] for row in selected_histories[0]], dtype=np.int64
        )
        values = np.asarray(
            [
                [float(row["clipping_bound_t"]) for row in history]
                for history in selected_histories
            ],
            dtype=np.float64,
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        axis.plot(steps, mean, color=colors["adaptive_dp"], linewidth=1.8)
        axis.fill_between(steps, mean - std, mean + std, alpha=0.2)
        axis.set_title(dataset)
        axis.set_xlabel("Iteration")
        axis.set_ylabel(r"$C_t$")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    clipping_png = FIGURE_ROOT / "new_figureC_stage_c_clipping_trajectory.png"
    clipping_pdf = FIGURE_ROOT / "new_figureC_stage_c_clipping_trajectory.pdf"
    figure.savefig(clipping_png, dpi=220)
    figure.savefig(clipping_pdf)
    plt.close(figure)

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for row in groups:
        csv_rows.append(
            {
                "dataset": row["dataset"],
                "mode": row["mode"],
                "target_epsilon": row["target_epsilon"],
                "steps": row["steps"],
                "count_mechanism_steps": row["count_mechanism_steps"],
                "gradient_noise_multiplier": row["gradient_noise_multiplier"],
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
        "schema": "new_module_c_formal_report_v1",
        "status": "COMPLETE",
        "formal_result": True,
        "runs_expected": 54,
        "runs_completed": len(runs),
        "test_loaded": False,
        "dp_mtt_status": "EXCLUDED_BY_USER",
        "privacy_scope": "stage_c_distilled_feature_records",
        "initializer_privacy_status": "data_independent",
        "released_model_record_dp_valid": True,
        "groups": groups,
        "adaptive_quality_gate_at_epsilon3": gains,
        "runs": runs,
        "artifacts": {
            "csv": str(CSV_OUTPUT),
            "privacy_utility_png": str(privacy_png),
            "privacy_utility_pdf": str(privacy_pdf),
            "clipping_trajectory_png": str(clipping_png),
            "clipping_trajectory_pdf": str(clipping_pdf),
        },
        "limitations": [
            "Accuracy uses the frozen test-free reference split; final test remains unread.",
            "Single-machine TFE does not expose communication byte/round counters, so both are N/A.",
            "A failed +1pp adaptive gate is reported as a negative result and is not hidden.",
        ],
    }
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    TXT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Module C formal report",
        "runs=54/54 test_loaded=False released_model_record_dp_valid=True",
    ]
    for row in groups:
        lines.append(
            "{dataset} {mode} eps={target_epsilon} final={mean:.6f}±{std:.6f}".format(
                mean=row["final_reference_accuracy"]["mean"],
                std=row["final_reference_accuracy"]["std"],
                **row
            )
        )
    for dataset, gate in gains.items():
        lines.append(
            "{} adaptive_minus_fixed_eps3={:.6f} passed={}".format(
                dataset, gate["adaptive_minus_fixed"], gate["passed"]
            )
        )
    TXT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("module_c_formal status=COMPLETE runs=54")
    for dataset, gate in gains.items():
        print(
            "{} adaptive_gain_eps3={:.6f} passed={}".format(
                dataset, gate["adaptive_minus_fixed"], gate["passed"]
            )
        )
    print("json={}".format(JSON_OUTPUT))
    print("csv={}".format(CSV_OUTPUT))
    print("figures={},{}".format(privacy_png, clipping_png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

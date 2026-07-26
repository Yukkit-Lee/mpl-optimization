#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate all formal Module B convergence runs without selecting seeds."""

import json
import statistics

from new_public_mtt_mainline_common import ROOT, load_config


SOURCE = ROOT / "new_output/new_public_mtt_mainline/module_b_convergence_formal"
OUTPUT = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_module_b_convergence_formal_report.json"
)


def early_auc(history):
    points = [
        (int(row["step"]), float(row["reference_accuracy"]))
        for row in history
        if "reference_accuracy" in row
    ]
    if len(points) == 1:
        return points[0][1]
    area = 0.0
    for left, right in zip(points[:-1], points[1:]):
        area += (right[0] - left[0]) * (right[1] + left[1]) * 0.5
    return area / float(points[-1][0] - points[0][0])


def step_at_target(history, target):
    for row in history:
        if (
            "reference_accuracy" in row
            and float(row["reference_accuracy"]) >= float(target)
        ):
            return int(row["step"])
    return None


def mean_std(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    config = load_config()
    formal = config["module_b_convergence_formal"]
    runs = []
    for dataset in config["datasets"]:
        target = formal["target_accuracy"][dataset]
        for method in formal["methods"]:
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
                if report["status"] != "COMPLETE":
                    raise ValueError("{} is not complete".format(matches[0]))
                history = report["history"]
                logged = [
                    row for row in history if "reference_accuracy" in row
                ]
                step_target = step_at_target(history, target)
                runtime = float(report["runtime_seconds"])
                total_steps = int(report["privacy"]["steps"])
                runs.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "target_accuracy": target,
                        "step_at_target": step_target,
                        "time_at_target_seconds": (
                            None
                            if step_target is None
                            else runtime * step_target / float(total_steps)
                        ),
                        "early_auc": early_auc(history),
                        "final_reference_accuracy": float(
                            logged[-1]["reference_accuracy"]
                        ),
                        "runtime_seconds": runtime,
                        "accounted_gradient_mechanism_epsilon": report[
                            "privacy"
                        ]["accounted_gradient_mechanism_epsilon"],
                        "released_model_record_dp_valid": report["privacy"][
                            "released_model_record_dp_valid"
                        ],
                        "epsilon_interpretation": report["privacy"][
                            "epsilon_interpretation"
                        ],
                        "test_loaded": False,
                        "source_report": str(matches[0]),
                    }
                )
    groups = []
    for dataset in config["datasets"]:
        for method in formal["methods"]:
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
                    "target_accuracy": formal["target_accuracy"][dataset],
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
                    "early_auc": mean_std([row["early_auc"] for row in selected]),
                    "final_reference_accuracy": mean_std(
                        [row["final_reference_accuracy"] for row in selected]
                    ),
                    "runtime_seconds": mean_std(
                        [row["runtime_seconds"] for row in selected]
                    ),
                    "released_model_record_dp_valid": all(
                        row["released_model_record_dp_valid"] for row in selected
                    ),
                    "epsilon_interpretation": selected[0][
                        "epsilon_interpretation"
                    ],
                }
            )
    payload = {
        "schema": "new_module_b_convergence_formal_report_v1",
        "status": "COMPLETE",
        "formal_result": True,
        "test_loaded": False,
        "dp_mtt_status": "EXCLUDED_BY_USER",
        "runs_expected": 18,
        "runs_completed": len(runs),
        "metric_split": "reference_only",
        "groups": groups,
        "runs": runs,
        "limitations": [
            "Formal final test evaluation remains a separate one-time read-only step.",
            "Secure Consensus initialization is data-dependent and non-DP; its epsilon is conditional mechanism accounting only.",
            "Single-machine TFE communication bytes and rounds are N/A.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("module_b_convergence status=COMPLETE runs={}".format(len(runs)))
    for group in groups:
        print(
            "{dataset} {method} final={final_reference_accuracy} "
            "early_auc={early_auc} reached={target_reached_seeds}/3".format(
                **group
            )
        )
    print("output={}".format(OUTPUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

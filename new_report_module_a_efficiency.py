#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Module-A formal runs and generate Figure A1/A2."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from new_public_mtt_mainline_common import ROOT, load_config


JSON_ROOT = ROOT / "new_log/new_public_mtt_mainline/json"
FIG_ROOT = ROOT / "new_expFig/new_public_mtt_mainline"
TAB_ROOT = ROOT / "new_expTab/new_public_mtt_mainline"


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1))


def nullable_mean_std(values):
    numeric = np.asarray(
        [float(value) for value in values if value is not None],
        dtype=np.float64,
    )
    if len(numeric) == 0:
        return None, None
    if len(numeric) == 1:
        return float(numeric[0]), 0.0
    return float(numeric.mean()), float(numeric.std(ddof=1))


def main():
    config = load_config()
    runs = []
    for dataset in config["datasets"]:
        for seed in config["formal_seeds"]:
            path = (
                JSON_ROOT
                / "new_module_a_runs"
                / dataset
                / "new_module_a_efficiency_{}_seed{}.json".format(
                    dataset, seed
                )
            )
            runs.append(json.loads(path.read_text(encoding="utf-8")))
    rows = []
    for run in runs:
        for result in run["results"]:
            rows.append(
                {
                    "Dataset": run["dataset"],
                    "Seed": run["seed"],
                    "Method": result["source"],
                    "Samples": result["num_train_samples"],
                    "CompressionRatio": result["compression_ratio"],
                    "FinalAccuracy": result["final_accuracy"],
                    "StepAtTarget": result["step_at_target"],
                    "TimeAtTargetS": result["time_at_target_seconds"],
                    "ProcessedExamplesAtTarget": result[
                        "processed_examples_at_target"
                    ],
                    "EarlyAUC": result["early_auc"],
                    "RuntimeS": result["runtime_seconds"],
                }
            )
    summary = []
    for dataset in config["datasets"]:
        for method in config["module_a_efficiency"]["sources"]:
            selected = [
                row
                for row in rows
                if row["Dataset"] == dataset and row["Method"] == method
            ]
            final_mean, final_std = mean_std(
                [row["FinalAccuracy"] for row in selected]
            )
            auc_mean, auc_std = mean_std(
                [row["EarlyAUC"] for row in selected]
            )
            step_mean, step_std = nullable_mean_std(
                [row["StepAtTarget"] for row in selected]
            )
            target_time_mean, target_time_std = nullable_mean_std(
                [row["TimeAtTargetS"] for row in selected]
            )
            summary.append(
                {
                    "Dataset": dataset,
                    "Method": method,
                    "Samples": selected[0]["Samples"],
                    "CompressionRatio": selected[0]["CompressionRatio"],
                    "FinalAccuracyMean": final_mean,
                    "FinalAccuracyStd": final_std,
                    "EarlyAUCMean": auc_mean,
                    "EarlyAUCStd": auc_std,
                    "StepAtTargetMean": step_mean,
                    "StepAtTargetStd": step_std,
                    "TimeAtTargetMeanS": target_time_mean,
                    "TimeAtTargetStdS": target_time_std,
                    "TargetReachedSeeds": sum(
                        row["StepAtTarget"] is not None for row in selected
                    ),
                }
            )
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TAB_ROOT.mkdir(parents=True, exist_ok=True)
    colors = {
        "full_images": "#315f8c",
        "random_subset": "#d08b36",
        "official_mtt": "#26866b",
    }
    labels = {
        "full_images": "Full Images",
        "random_subset": "Random Subset",
        "official_mtt": "Official MTT",
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for axis, dataset in zip(axes, config["datasets"]):
        for method in config["module_a_efficiency"]["sources"]:
            histories = [
                result["history"]
                for run in runs
                if run["dataset"] == dataset
                for result in run["results"]
                if result["source"] == method
            ]
            steps = np.asarray(
                [row["step"] for row in histories[0]], dtype=np.float64
            )
            values = np.asarray(
                [
                    [row["test_accuracy"] for row in history]
                    for history in histories
                ]
            )
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            axis.plot(steps, mean, color=colors[method], label=labels[method])
            axis.fill_between(
                steps,
                np.maximum(0, mean - std),
                np.minimum(1, mean + std),
                color=colors[method],
                alpha=0.16,
            )
        axis.set_title(
            {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tiny": "Tiny ImageNet"}[
                dataset
            ]
        )
        axis.set_xlabel("Optimizer Steps")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test Accuracy")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        FIG_ROOT / "new_figureA1_accuracy_vs_steps.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        FIG_ROOT / "new_figureA1_accuracy_vs_steps.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    datasets = config["datasets"]
    methods = config["module_a_efficiency"]["sources"]
    dataset_labels = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "tiny": "Tiny ImageNet",
    }
    grouped = {
        (row["Dataset"], row["Method"]): row for row in summary
    }
    x = np.arange(len(datasets), dtype=np.float64)
    width = 0.24
    metric_specs = (
        ("Samples", None, "Training Samples", True),
        ("StepAtTargetMean", "StepAtTargetStd", "Step @ Target Acc", False),
        (
            "TimeAtTargetMeanS",
            "TimeAtTargetStdS",
            "Time @ Target Acc (s)",
            False,
        ),
    )
    for axis, (value_key, error_key, ylabel, use_log) in zip(
        axes, metric_specs
    ):
        for position, method in enumerate(methods):
            selected = [grouped[(dataset, method)] for dataset in datasets]
            values = [
                np.nan if row[value_key] is None else row[value_key]
                for row in selected
            ]
            errors = (
                None
                if error_key is None
                else [
                    0.0 if row[error_key] is None else row[error_key]
                    for row in selected
                ]
            )
            axis.bar(
                x + (position - 1) * width,
                values,
                width,
                yerr=errors,
                label=labels[method],
                color=colors[method],
                capsize=3,
            )
        axis.set_xticks(x)
        axis.set_xticklabels([dataset_labels[item] for item in datasets])
        axis.set_ylabel(ylabel)
        if use_log:
            axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(
        FIG_ROOT / "new_figureA2_sample_efficiency.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        FIG_ROOT / "new_figureA2_sample_efficiency.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    for name, data in (
        ("new_module_a_efficiency_runs.csv", rows),
        ("new_module_a_efficiency_summary.csv", summary),
    ):
        with (TAB_ROOT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    report = {
        "schema": "new_module_a_efficiency_report_v1",
        "status": "COMPLETE",
        "stage_a_epsilon": "N/A",
        "runs": rows,
        "summary": summary,
        "figures": [
            str(FIG_ROOT / "new_figureA1_accuracy_vs_steps.png"),
            str(FIG_ROOT / "new_figureA2_sample_efficiency.png"),
        ],
    }
    (JSON_ROOT / "new_module_a_efficiency_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

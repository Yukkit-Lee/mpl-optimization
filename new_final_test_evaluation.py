#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-time read-only final test evaluation after every formal model is frozen."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from new_public_mtt_mainline_common import ROOT, load_config, load_feature_cache


MODULE_REPORT = (
    ROOT / "new_log/new_public_mtt_mainline/json/new_module_c_formal_report.json"
)
END_TO_END_REPORT = (
    ROOT / "new_log/new_public_mtt_mainline/json/new_end_to_end_formal_report.json"
)
FEATURE_MANIFEST = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_end_to_end_feature_manifest.json"
)
JSON_OUTPUT = (
    ROOT / "new_log/new_public_mtt_mainline/json/new_final_test_evaluation.json"
)
TXT_OUTPUT = (
    ROOT / "new_log/new_public_mtt_mainline/txt/new_final_test_evaluation.txt"
)
CSV_OUTPUT = (
    ROOT / "new_expTab/new_public_mtt_mainline/new_final_test_evaluation.csv"
)
FIGURE_ROOT = ROOT / "new_expFig/new_public_mtt_mainline"
END_TO_END_METHODS = {
    "baseline_random_secure_average_fixed": "fixed_dp",
    "ours_official_secure_consensus_adaptive": "adaptive_dp",
}


def read_complete_report(path: Path, expected_runs: int):
    if not path.is_file():
        raise FileNotFoundError(
            "final test gate is closed until formal report exists: {}".format(path)
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise RuntimeError("formal report is not COMPLETE: {}".format(path))
    if int(payload.get("runs_completed", -1)) != expected_runs:
        raise RuntimeError("formal report has incomplete run count: {}".format(path))
    if payload.get("test_loaded") is not False:
        raise RuntimeError("formal tuning report unexpectedly used test data")
    return payload


def unique_model(directory: Path):
    matches = list(directory.glob("new_secure_stage_c_model_*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected one frozen model in {}; got {}".format(directory, len(matches))
        )
    return matches[0]


def accuracy(model_path: Path, x_test, y_test):
    with np.load(str(model_path), allow_pickle=False) as model:
        weight = np.asarray(model["weight"], dtype=np.float32)
        bias = np.asarray(model["bias"], dtype=np.float32)
    x = x_test.detach().cpu().numpy().astype(np.float32, copy=False)
    y = y_test.detach().cpu().numpy().astype(np.int64, copy=False)
    if x.ndim != 2 or x.shape[1] != weight.shape[0]:
        raise ValueError("test/model feature shape mismatch: {}".format(model_path))
    predicted = np.argmax(x @ weight + bias, axis=1)
    return float(np.mean(predicted == y)), int(y.shape[0])


def mean_std(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main():
    config = load_config()

    # These gates and all model existence checks intentionally precede any test read.
    module_report = read_complete_report(MODULE_REPORT, 54)
    end_to_end_report = read_complete_report(END_TO_END_REPORT, 18)
    manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("test_loaded") is not False:
        raise RuntimeError("invalid test-free End-to-End feature manifest")
    signatures = {
        (row["dataset"], int(row["seed"])): (
            row["encoder_sha256"],
            row["scaler_sha256"],
        )
        for row in manifest["rows"]
    }

    plans = {}
    for dataset in config["datasets"]:
        for seed in config["formal_seeds"]:
            models = []
            for mode in config["module_c_formal"]["modes"]:
                for epsilon in config["module_c_formal"]["target_epsilons"]:
                    tag = "eps{}".format(str(float(epsilon)).replace(".", "p"))
                    models.append(
                        {
                            "family": "module_c",
                            "method": mode,
                            "target_epsilon": float(epsilon),
                            "released_model_record_dp_valid": True,
                            "model": unique_model(
                                ROOT
                                / "new_output/new_public_mtt_mainline/module_c_formal"
                                / dataset
                                / mode
                                / tag
                                / "seed{}".format(seed)
                            ),
                        }
                    )
            for method, mode in END_TO_END_METHODS.items():
                models.append(
                    {
                        "family": "end_to_end",
                        "method": method,
                        "target_epsilon": float(
                            config["end_to_end_formal"]["target_epsilon"]
                        ),
                        "released_model_record_dp_valid": False,
                        "model": unique_model(
                            ROOT
                            / "new_output/new_public_mtt_mainline/end_to_end_formal"
                            / dataset
                            / method
                            / "seed{}".format(seed)
                        ),
                    }
                )
            plans[(dataset, seed)] = models

    runs = []
    test_reads = []
    for dataset in config["datasets"]:
        for seed in config["formal_seeds"]:
            cache = load_feature_cache(dataset, seed, allow_test=True)
            expected = signatures[(dataset, seed)]
            actual = (cache["encoder_sha256"], cache["scaler_sha256"])
            if actual != expected:
                raise RuntimeError(
                    "encoder/scaler signature mismatch for {} seed{}".format(
                        dataset, seed
                    )
                )
            test_reads.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "test_examples": int(cache["y_test"].shape[0]),
                    "feature_cache_path": cache["feature_cache_path"],
                    "feature_cache_sha256": cache["feature_cache_sha256"],
                    "read_count_in_this_evaluator": 1,
                }
            )
            for plan in plans[(dataset, seed)]:
                value, examples = accuracy(
                    plan["model"], cache["x_test"], cache["y_test"]
                )
                runs.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "family": plan["family"],
                        "method": plan["method"],
                        "target_epsilon": plan["target_epsilon"],
                        "test_accuracy": value,
                        "test_examples": examples,
                        "released_model_record_dp_valid": plan[
                            "released_model_record_dp_valid"
                        ],
                        "model_path": str(plan["model"]),
                    }
                )

    groups = []
    group_keys = sorted(
        {
            (
                row["dataset"],
                row["family"],
                row["method"],
                row["target_epsilon"],
            )
            for row in runs
        }
    )
    for dataset, family, method, epsilon in group_keys:
        selected = [
            row
            for row in runs
            if (
                row["dataset"],
                row["family"],
                row["method"],
                row["target_epsilon"],
            )
            == (dataset, family, method, epsilon)
        ]
        groups.append(
            {
                "dataset": dataset,
                "family": family,
                "method": method,
                "target_epsilon": epsilon,
                "seeds": [row["seed"] for row in selected],
                "test_accuracy": mean_std(
                    [row["test_accuracy"] for row in selected]
                ),
                "released_model_record_dp_valid": selected[0][
                    "released_model_record_dp_valid"
                ],
            }
        )

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        rows = [
            {
                "dataset": row["dataset"],
                "family": row["family"],
                "method": row["method"],
                "target_epsilon": row["target_epsilon"],
                "test_accuracy_mean": row["test_accuracy"]["mean"],
                "test_accuracy_std": row["test_accuracy"]["std"],
                "released_model_record_dp_valid": row[
                    "released_model_record_dp_valid"
                ],
            }
            for row in groups
        ]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, dataset in zip(axes, config["datasets"]):
        selected = [
            row
            for row in groups
            if row["dataset"] == dataset and row["family"] == "module_c"
        ]
        for mode, color in (("fixed_dp", "#4477AA"), ("adaptive_dp", "#CC6677")):
            rows = sorted(
                [row for row in selected if row["method"] == mode],
                key=lambda row: row["target_epsilon"],
            )
            axis.errorbar(
                [row["target_epsilon"] for row in rows],
                [row["test_accuracy"]["mean"] for row in rows],
                yerr=[row["test_accuracy"]["std"] for row in rows],
                marker="o",
                capsize=3,
                color=color,
                label=mode.replace("_dp", "").title(),
            )
        axis.set_title(dataset)
        axis.set_xlabel(r"$\epsilon_C$")
        axis.set_ylabel("Final test accuracy")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    png = FIGURE_ROOT / "new_figureF_final_test_privacy_utility.png"
    pdf = FIGURE_ROOT / "new_figureF_final_test_privacy_utility.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)

    payload = {
        "schema": "new_final_test_evaluation_v1",
        "status": "COMPLETE",
        "formal_models_evaluated": len(runs),
        "test_cache_reads": len(test_reads),
        "test_usage": "single_readonly_after_all_config_and_models_frozen",
        "test_feedback_to_tuning_or_quality_gates": False,
        "known_prefinal_metadata_only_reads": [
            {
                "dataset": "cifar10",
                "seed": 42,
                "purpose": "manual shape and encoder/scaler signature audit",
                "accuracy_or_prediction_computed": False,
                "feedback_to_configuration_or_model_selection": False,
            }
        ],
        "dp_mtt_status": "EXCLUDED_BY_USER",
        "module_c_report_gate": str(MODULE_REPORT),
        "end_to_end_report_gate": str(END_TO_END_REPORT),
        "module_c_release_dp_valid": module_report[
            "released_model_record_dp_valid"
        ],
        "end_to_end_release_dp_valid": end_to_end_report[
            "released_model_record_dp_valid"
        ],
        "test_reads": test_reads,
        "groups": groups,
        "runs": runs,
        "artifacts": {
            "csv": str(CSV_OUTPUT),
            "figure_png": str(png),
            "figure_pdf": str(pdf),
        },
    }
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "Final one-time read-only test evaluation",
        "models={} cache_reads={} feedback_to_tuning=False".format(
            len(runs), len(test_reads)
        ),
    ]
    for row in groups:
        lines.append(
            "{dataset} {family} {method} eps={target_epsilon} "
            "test={mean:.6f}±{std:.6f}".format(
                mean=row["test_accuracy"]["mean"],
                std=row["test_accuracy"]["std"],
                **row
            )
        )
    TXT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TXT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "final_test status=COMPLETE models={} cache_reads={}".format(
            len(runs), len(test_reads)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

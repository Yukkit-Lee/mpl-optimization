#!/usr/bin/env python3
"""Report frozen Stage-B candidate initialization quality.

All candidate construction and selection is already complete before this
script opens benchmark test features. Test accuracy is read-only reporting and
is never fed back into hyperparameter selection.
"""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from new_public_mtt_mainline_common import ROOT, load_feature_cache


DATASETS = ("cifar10", "cifar100", "tiny")
SPLITS = ("iid", "mild", "severe")
METHODS = ("NoKD", "KD-Avg", "KD-ReferenceWeighted")
SEEDS = (42, 43, 44)
INIT_ROOT = ROOT / "new_init_model/new_public_mtt_mainline/stage_b"
JSON_PATH = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_stage_b_initialization_report.json"
)
TXT_PATH = (
    ROOT
    / "new_log/new_public_mtt_mainline/txt"
    / "new_stage_b_initialization_report.txt"
)
RUN_CSV = (
    ROOT
    / "new_expTab/new_public_mtt_mainline"
    / "new_stage_b_initialization_runs.csv"
)
SUMMARY_CSV = (
    ROOT
    / "new_expTab/new_public_mtt_mainline"
    / "new_stage_b_initialization_summary.csv"
)
FIGURE_ROOT = ROOT / "new_expFig/new_public_mtt_mainline"


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_accuracy(path, x_test, y_test):
    payload = np.load(str(path), allow_pickle=False)
    weight = np.asarray(payload["weight"], dtype=np.float32)
    bias = np.asarray(payload["bias"], dtype=np.float32)
    logits = np.matmul(x_test, weight) + bias
    return float(np.mean(np.argmax(logits, axis=1) == y_test))


def expected_report(dataset, seed, split):
    return (
        INIT_ROOT
        / dataset
        / "seed{}".format(seed)
        / split
        / "new_secure_stage_b_report_{}_{}_seed{}.json".format(
            dataset, split, seed
        )
    )


def collect_rows():
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            data = load_feature_cache(dataset, seed, allow_test=True)
            x_test = data["x_test"].cpu().numpy().astype(np.float32, copy=False)
            y_test = data["y_test"].cpu().numpy().astype(np.int64, copy=False)
            for split in SPLITS:
                report_path = expected_report(dataset, seed, split)
                if not report_path.exists():
                    raise RuntimeError(
                        "formal Stage-B output missing: {}".format(report_path)
                    )
                report = load_json(report_path)
                if report.get("status") != "COMPLETE":
                    raise RuntimeError(
                        "formal Stage-B output incomplete: {}".format(report_path)
                    )
                if not report["security"]["formal_candidate_release"]:
                    raise RuntimeError("formal candidate boundary was not enabled")
                candidates = report["formal_candidate_models"]
                for method in METHODS:
                    candidate_path = Path(candidates[method]["path"])
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "split": split,
                            "method": method,
                            "initial_test_accuracy": candidate_accuracy(
                                candidate_path, x_test, y_test
                            ),
                            "tfe_runtime_seconds": float(
                                report["runtime_seconds"]
                            ),
                            "training_elapsed_seconds": float(
                                report["training_elapsed_seconds"]
                            ),
                            "history_epochs": len(report["history"]),
                            "communication_bytes": "N/A",
                            "communication_rounds": "N/A",
                            "test_role": "readonly_final_reporting",
                        }
                    )
    return rows


def summarize(rows):
    summary = []
    for dataset in DATASETS:
        for split in SPLITS:
            for method in METHODS:
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["split"] == split
                    and row["method"] == method
                ]
                values = [row["initial_test_accuracy"] for row in selected]
                runtimes = [row["tfe_runtime_seconds"] for row in selected]
                summary.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "method": method,
                        "initial_accuracy_mean": float(np.mean(values)),
                        "initial_accuracy_std": float(np.std(values, ddof=1)),
                        "tfe_runtime_mean_seconds": float(np.mean(runtimes)),
                        "tfe_runtime_std_seconds": float(
                            np.std(runtimes, ddof=1)
                        ),
                        "seeds": len(values),
                    }
                )
    for dataset in DATASETS:
        for method in METHODS:
            iid = next(
                row
                for row in summary
                if row["dataset"] == dataset
                and row["split"] == "iid"
                and row["method"] == method
            )
            severe = next(
                row
                for row in summary
                if row["dataset"] == dataset
                and row["split"] == "severe"
                and row["method"] == method
            )
            severe["accuracy_retention_from_iid"] = (
                severe["initial_accuracy_mean"]
                / max(iid["initial_accuracy_mean"], 1e-12)
            )
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(summary):
    lines = [
        "| Dataset | Split | NoKD | Secure KD-Avg | Secure KD-RW |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for split in SPLITS:
            cells = []
            for method in METHODS:
                row = next(
                    item
                    for item in summary
                    if item["dataset"] == dataset
                    and item["split"] == split
                    and item["method"] == method
                )
                cells.append(
                    "{:.4f} +/- {:.4f}".format(
                        row["initial_accuracy_mean"],
                        row["initial_accuracy_std"],
                    )
                )
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    dataset, split, cells[0], cells[1], cells[2]
                )
            )
    return "\n".join(lines)


def plot(summary):
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    colors = {
        "NoKD": "#7A7168",
        "KD-Avg": "#E07A3F",
        "KD-ReferenceWeighted": "#187A72",
    }
    labels = {
        "NoKD": "NoKD",
        "KD-Avg": "Secure KD-Avg",
        "KD-ReferenceWeighted": "Secure KD-ReferenceWeighted",
    }
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=False)
    x = np.arange(len(SPLITS), dtype=np.float64)
    for axis, dataset in zip(axes, DATASETS):
        for method in METHODS:
            selected = [
                row
                for row in summary
                if row["dataset"] == dataset and row["method"] == method
            ]
            selected.sort(key=lambda row: SPLITS.index(row["split"]))
            mean = np.asarray(
                [row["initial_accuracy_mean"] for row in selected]
            )
            std = np.asarray(
                [row["initial_accuracy_std"] for row in selected]
            )
            axis.errorbar(
                x,
                mean,
                yerr=std,
                color=colors[method],
                marker="o",
                linewidth=2.0,
                capsize=3,
                label=labels[method],
            )
        axis.set_xticks(x)
        axis.set_xticklabels(["IID", "Mild", "Severe"])
        axis.set_xlabel("Data Heterogeneity")
        axis.set_ylabel("Initial Global Model Test Accuracy")
        axis.set_title(
            {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tiny": "Tiny ImageNet"}[
                dataset
            ]
        )
        axis.grid(alpha=0.2, linestyle="--")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    png = FIGURE_ROOT / "new_figureB2_stage_b_noniid_initialization.png"
    pdf = FIGURE_ROOT / "new_figureB2_stage_b_noniid_initialization.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [str(png), str(pdf)]


def main():
    rows = collect_rows()
    if len(rows) != 81:
        raise RuntimeError("expected 81 candidate rows, found {}".format(len(rows)))
    summary = summarize(rows)
    write_csv(RUN_CSV, rows)
    write_csv(SUMMARY_CSV, summary)
    figures = plot(summary)
    table = markdown_table(summary)
    payload = {
        "schema": "new_stage_b_initialization_report_v1",
        "status": "COMPLETE",
        "test_role": "readonly_final_reporting_after_selection_freeze",
        "test_used_for_tuning": False,
        "backend": "tfe_mpc",
        "protocol": "ABY3",
        "rows": rows,
        "summary": summary,
        "figures": figures,
        "limitations": [
            "This report covers candidate initialization accuracy only.",
            "Step@TargetAcc and EarlyAUC are reported by the subsequent secure Stage-C convergence experiment.",
            "Single-machine TFE does not expose communication byte/round counters; these fields are N/A.",
        ],
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TXT_PATH.write_text(table + "\n", encoding="utf-8")
    print(table)
    print("json={}".format(JSON_PATH))
    print("runs_csv={}".format(RUN_CSV))
    print("summary_csv={}".format(SUMMARY_CSV))
    print("figures={}".format(figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

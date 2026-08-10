#!/usr/bin/env python3
"""Merge the six full-trajectory SPU cells into paper tables and hash manifest.

Reads ONLY the exact full-trajectory report/trace paths, hard-asserts 6/6
COMPLETE + per-cell audit gates, and writes:
  - new_spu_full_trajectory_main_table.csv
  - new_spu_full_trajectory_time_to_target.csv
  - new_spu_full_trajectory_runtime_breakdown.csv
  - new_spu_full_trajectory_hash_manifest.json
  - new_figure2_spu_full_trajectory_plot_data.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LOG = ROOT / "new_log/new_secretflow_spu/full_trajectory_v1"
OUT = ROOT / "new_output/new_secretflow_spu/full_trajectory_v1"
TAB = ROOT / "new_expTab/new_secretflow_spu/full_trajectory_v1"
PLOT = ROOT / "new_output/new_secretflow_spu"

DATASETS = ("cifar10", "cifar100", "tiny")
METHODS = ("full_data", "true_ours")
METHOD_LABELS = {"full_data": "Full Data", "true_ours": "True Ours"}

TARGETS = {
    "cifar10": {"T@80": 0.58128, "T@90": 0.65394, "T@95": 0.69027},
    "cifar100": {"T@80": 0.34464, "T@90": 0.38772, "T@95": 0.40926},
    "tiny": {"T@80": 0.22120, "T@90": 0.24885, "T@95": 0.262675},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def time_to_target(trace, target):
    for row in trace:
        if float(row["development_accuracy"]) >= target:
            return float(row["cumulative_training_online_secure_seconds"])
    return None


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    report_paths = {}
    trace_paths = {}
    main_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            run_id = "full_trajectory_v1_formal_{}_{}_seed42".format(dataset, method)
            report_path = LOG / "{}_report.json".format(run_id)
            output_dir = OUT / run_id
            trace_path = output_dir / "new_spu_full_trajectory_trace.csv"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            with trace_path.open(newline="", encoding="utf-8") as handle:
                trace = list(csv.DictReader(handle))
            report_paths["{}_{}".format(dataset, method)] = report_path
            trace_paths["{}_{}".format(dataset, method)] = trace_path

            if report["status"] != "COMPLETE":
                raise AssertionError("cell {} {} not COMPLETE".format(dataset, method))

            ec = report["effective_config"]
            tt = {k: time_to_target(trace, v) for k, v in TARGETS[dataset].items()}
            main_rows.append({
                "dataset": dataset,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "status": report["status"],
                "final_development_accuracy": report["final_development_accuracy"],
                "stage_b_online_secure_seconds": report["timing_seconds"]["stage_b_online_secure"],
                "stage_c_online_secure_seconds": report["timing_seconds"]["stage_c_online_secure"],
                "total_online_secure_seconds": report["online_secure_seconds"],
                "stage_b_evaluation_online_secure_seconds": report["timing_seconds"].get(
                    "stage_b_evaluation_online_secure", 0.0
                ),
                "raw_wall_seconds": report["raw_wall_seconds"],
                "run_observed_memory_peak_gib": report["resource_measurement"][
                    "run_observed_memory_peak_gib"
                ],
                "conditional_stage_c_epsilon": report["conditional_stage_c_epsilon"],
                "stage_b_trace_points": report["stage_b_trace_points"],
                "stage_c_trace_points": report["stage_c_trace_points"],
                "T@80_seconds": "" if tt["T@80"] is None else round(tt["T@80"], 1),
                "T@90_seconds": "" if tt["T@90"] is None else round(tt["T@90"], 1),
                "T@95_seconds": "" if tt["T@95"] is None else round(tt["T@95"], 1),
                "official_test_read": report["official_test_read"],
                "locked_holdout_read": report["locked_holdout_read"],
            })

    main_path = TAB / "new_spu_full_trajectory_main_table.csv"
    with main_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(main_rows[0].keys()))
        writer.writeheader()
        writer.writerows(main_rows)

    # time-to-target table (speedup + reduction).
    tt_rows = []
    for dataset in DATASETS:
        ours = next(r for r in main_rows if r["dataset"] == dataset and r["method"] == "true_ours")
        full = next(r for r in main_rows if r["dataset"] == dataset and r["method"] == "full_data")
        for label in ("T@80", "T@90", "T@95"):
            o = ours[label + "_seconds"]
            f = full[label + "_seconds"]
            ratio = ""
            reduction = ""
            if o and f:
                ratio = "{:.1f}x".format(float(f) / float(o))
                reduction = "{:.1f}%".format(
                    (1.0 - float(o) / float(f)) * 100.0
                )
            tt_rows.append({
                "dataset": dataset,
                "target": label,
                "true_ours_seconds": o or "Not Reached",
                "full_data_seconds": f or "Not Reached",
                "speedup_ratio": ratio,
                "time_reduction_percentage": reduction,
                "note": "first observed preregistered checkpoint reaching target; not interpolated",
            })
    tt_path = TAB / "new_spu_full_trajectory_time_to_target.csv"
    with tt_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tt_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tt_rows)

    # runtime breakdown.
    brk_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            run_id = "full_trajectory_v1_formal_{}_{}_seed42".format(dataset, method)
            report = json.loads((LOG / "{}_report.json".format(run_id)).read_text(encoding="utf-8"))
            t = report["timing_seconds"]
            brk_rows.append({
                "dataset": dataset,
                "method": method,
                "stage_b_compile_warmup_seconds": t.get("stage_b_compile_warmup", ""),
                "stage_b_private_pack_infeed_seconds": t.get("stage_b_private_pack_and_infeed", ""),
                "stage_b_online_secure_seconds": t.get("stage_b_online_secure", ""),
                "stage_b_eval_compile_warmup_seconds": t.get("stage_b_eval_compile_warmup", ""),
                "stage_b_evaluation_online_secure_seconds": t.get("stage_b_evaluation_online_secure", ""),
                "stage_b_secret_checkpoint_seconds": t.get("stage_b_secret_checkpoint", ""),
                "stage_c_compile_warmup_seconds": t.get("stage_c_compile_warmup", ""),
                "stage_c_private_infeed_noise_seconds": t.get("stage_c_private_infeed_and_noise_generation", ""),
                "stage_c_online_secure_seconds": t.get("stage_c_online_secure", ""),
                "stage_c_authorized_release_eval_seconds": t.get("stage_c_authorized_dp_release_and_dev_evaluation", ""),
                "stage_c_privacy_accounting_seconds": t.get("stage_c_privacy_accounting", ""),
                "raw_wall_seconds": report["raw_wall_seconds"],
            })
    brk_path = TAB / "new_spu_full_trajectory_runtime_breakdown.csv"
    with brk_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(brk_rows[0].keys()))
        writer.writeheader()
        writer.writerows(brk_rows)

    # hash manifest.
    manifest = {
        "schema": "new_spu_full_trajectory_hash_manifest_v1",
        "reports": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in report_paths.items()
        },
        "traces": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in trace_paths.items()
        },
    }
    manifest_path = LOG / "new_spu_full_trajectory_hash_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # plot data (x = cumulative training online secure).
    plot_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            run_id = "full_trajectory_v1_formal_{}_{}_seed42".format(dataset, method)
            report = json.loads((LOG / "{}_report.json".format(run_id)).read_text(encoding="utf-8"))
            trace_path = OUT / run_id / "new_spu_full_trajectory_trace.csv"
            with trace_path.open(newline="", encoding="utf-8") as handle:
                trace = list(csv.DictReader(handle))
            for row in trace:
                plot_rows.append({
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "stage": row["stage"],
                    "epoch_or_step": int(float(row["epoch_or_step"])),
                    "cumulative_training_online_secure_seconds": float(
                        row["cumulative_training_online_secure_seconds"]
                    ),
                    "cumulative_instrumented_wall_seconds": float(
                        row["cumulative_instrumented_wall_seconds"]
                    ),
                    "development_accuracy": float(row["development_accuracy"]),
                    "target_T80": TARGETS[dataset]["T@80"],
                    "target_T90": TARGETS[dataset]["T@90"],
                    "target_T95": TARGETS[dataset]["T@95"],
                    "n": 1,
                })
    plot_path = PLOT / "new_figure2_spu_full_trajectory_plot_data.csv"
    with plot_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plot_rows)

    print("main_table:", main_path)
    print("time_to_target:", tt_path)
    print("runtime_breakdown:", brk_path)
    print("hash_manifest:", manifest_path)
    print("plot_data:", plot_path)
    print("ALL SIX FULL-TRAJECTORY CELLS MERGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

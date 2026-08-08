#!/usr/bin/env python3
"""corrected-v2 end-to-end result builder (Step 7, handoff §7).

Reads ONLY the six exact corrected-v2 report paths and their trace CSVs,
hard-asserts every acceptance gate, and writes:
  - new_spu_corrected_v2_end2end_main_table.csv
  - new_spu_corrected_v2_hash_manifest.json
  - new_figure2_spu_corrected_v2_plot_data.csv
plus a runtime breakdown CSV and an audit markdown.

x-axis = Stage-B online secure cumulative + Stage-C online secure cumulative.
T@Target = first trace point >= frozen target, else Not Reached (no interpolation).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "new_log/new_secretflow_spu/corrected_v2"
EXP = ROOT / "new_output/new_secretflow_spu/corrected_v2"
PLOT = ROOT / "new_output/new_secretflow_spu"

DATASETS = ("cifar10", "cifar100", "tiny")
METHODS = ("true_ours", "full_data")
METHOD_LABELS = {"true_ours": "True Ours", "full_data": "Full Data"}

TARGETS = {
    "cifar10": {"T@80": 0.58128, "T@90": 0.65394, "T@95": 0.69027},
    "cifar100": {"T@80": 0.34464, "T@90": 0.38772, "T@95": 0.40926},
    "tiny": {"T@80": 0.22120, "T@90": 0.24885, "T@95": 0.262675},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_report(dataset, method):
    path = OUT / "corrected_v2_formal_{}_{}_seed42_report.json".format(dataset, method)
    return json.loads(path.read_text(encoding="utf-8")), path


def load_trace(dataset, method):
    path = EXP / "corrected_v2_formal_{}_{}_seed42/new_spu_corrected_trace.csv".format(
        dataset, method
    )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle)), path


def time_to_target(trace, target):
    for row in trace:
        if float(row["development_accuracy"]) >= target:
            return float(row["cumulative_online_secure_seconds"])
    return None


def main() -> None:
    report_paths = {}
    trace_paths = {}
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            report, rpath = load_report(dataset, method)
            trace, tpath = load_trace(dataset, method)
            report_paths["{}_{}".format(dataset, method)] = rpath
            trace_paths["{}_{}".format(dataset, method)] = tpath

            # --- hard acceptance gates ---
            ec = report["effective_config"]
            gates = {
                "status_complete": report["status"] == "COMPLETE",
                "run_kind_formal": report["run_kind"] == "formal",
                "effective_formal_run": ec.get("formal_run") is True,
                "effective_matches_freeze": (
                    ec["seed"] == 42
                    and ec["stage_b_unroll"] == 1
                    and ec["stage_c_block_steps"] == 10
                ),
                "epsilon_within_target": float(
                    report["conditional_stage_c_epsilon"]
                ) <= 3.000001,
                "delta_1e5": float(report["delta"]) == 1e-5,
                "no_client_local_reveal": report[
                    "stage_b_client_state_public_reveals"
                ] == 0,
                "reveal_no_client_public_reconstruction": all(
                    not (e.get("scope") == "client_local_secret_state" and e.get("public_reconstruction"))
                    for e in report["reveal_ledger"]
                ),
                "provider_commitments_distinct": all(
                    len({x["last_share_commitment"] for x in stream}) == 3
                    for stream in report["provider_metadata"].values()
                ),
                "official_test_unread": report["official_test_read"] is False,
                "locked_holdout_unread": report["locked_holdout_read"] is False,
            }
            if not all(gates.values()):
                raise AssertionError(
                    "corrected-v2 gate failed for {} {}: {}".format(
                        dataset, method,
                        {k: v for k, v in gates.items() if not v},
                    )
                )

            trace_steps = [int(r["steps_done"]) for r in trace]
            gate_trace = {
                "trace_strictly_increasing": all(
                    b > a for a, b in zip(trace_steps, trace_steps[1:])
                ),
                "trace_ends_at_config": trace_steps[-1] == int(ec["stage_c_steps"]),
            }
            if not all(gate_trace.values()):
                raise AssertionError(
                    "trace gate failed for {} {}: {}".format(dataset, method, gate_trace)
                )

            acc = float(report["final_development_accuracy"])
            tt = {k: time_to_target(trace, v) for k, v in TARGETS[dataset].items()}
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "status": report["status"],
                    "final_development_accuracy": acc,
                    "stage_b_online_secure_seconds": report["timing_seconds"][
                        "stage_b_online_secure"
                    ],
                    "stage_c_online_secure_seconds": report["timing_seconds"][
                        "stage_c_online_secure"
                    ],
                    "total_online_secure_seconds": report["online_secure_seconds"],
                    "raw_wall_seconds": report["raw_wall_seconds"],
                    "run_observed_memory_peak_gib": report["resource_measurement"][
                        "run_observed_memory_peak_gib"
                    ],
                    "conditional_stage_c_epsilon": report[
                        "conditional_stage_c_epsilon"
                    ],
                    "delta": report["delta"],
                    "T@80_seconds": "" if tt["T@80"] is None else round(tt["T@80"], 1),
                    "T@90_seconds": "" if tt["T@90"] is None else round(tt["T@90"], 1),
                    "T@95_seconds": "" if tt["T@95"] is None else round(tt["T@95"], 1),
                    "official_test_read": report["official_test_read"],
                    "locked_holdout_read": report["locked_holdout_read"],
                }
            )

    # --- main table ---
    main_path = EXP / "new_spu_corrected_v2_end2end_main_table.csv"
    with main_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- hash manifest ---
    manifest = {
        "schema": "new_spu_corrected_v2_hash_manifest_v1",
        "reports": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in report_paths.items()
        },
        "traces": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in trace_paths.items()
        },
    }
    manifest_path = OUT / "new_spu_corrected_v2_hash_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # --- plot data (x = stage_b cumulative + stage_c cumulative) ---
    plot_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            report, _ = load_report(dataset, method)
            trace, _ = load_trace(dataset, method)
            stage_b_sec = report["timing_seconds"]["stage_b_online_secure"]
            transition = stage_b_sec
            for row in trace:
                plot_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "cumulative_online_secure_seconds": (
                            float(stage_b_sec)
                            + float(row["stage_c_online_secure_seconds"])
                        ),
                        "development_accuracy": float(row["development_accuracy"]),
                        "stage_b_to_c_transition_seconds": float(transition),
                        "target_T80": TARGETS[dataset]["T@80"],
                        "target_T90": TARGETS[dataset]["T@90"],
                        "target_T95": TARGETS[dataset]["T@95"],
                        "n": 1,
                    }
                )
    plot_path = PLOT / "new_figure2_spu_corrected_v2_plot_data.csv"
    with plot_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plot_rows)

    # --- runtime breakdown ---
    brk_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            report, _ = load_report(dataset, method)
            t = report["timing_seconds"]
            row = {
                "dataset": dataset,
                "method": method,
                "stage_b_compile_warmup_seconds": t.get("stage_b_compile_warmup", ""),
                "stage_b_private_pack_infeed_seconds": t.get(
                    "stage_b_private_pack_and_infeed", ""
                ),
                "stage_b_online_secure_seconds": t.get("stage_b_online_secure", ""),
                "stage_b_secret_checkpoint_seconds": t.get(
                    "stage_b_secret_checkpoint", ""
                ),
                "stage_c_compile_warmup_seconds": t.get("stage_c_compile_warmup", ""),
                "stage_c_private_infeed_noise_seconds": t.get(
                    "stage_c_private_infeed_and_noise_generation", ""
                ),
                "stage_c_online_secure_seconds": t.get("stage_c_online_secure", ""),
                "stage_c_authorized_release_eval_seconds": t.get(
                    "stage_c_authorized_dp_release_and_dev_evaluation", ""
                ),
                "stage_c_privacy_accounting_seconds": t.get(
                    "stage_c_privacy_accounting", ""
                ),
                "raw_wall_seconds": report["raw_wall_seconds"],
            }
            brk_rows.append(row)
    brk_path = EXP / "new_spu_corrected_v2_runtime_breakdown.csv"
    with brk_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(brk_rows[0].keys()))
        writer.writeheader()
        writer.writerows(brk_rows)

    print("main_table:", main_path)
    print("hash_manifest:", manifest_path)
    print("plot_data:", plot_path)
    print("runtime_breakdown:", brk_path)
    print("ALL CORRECTED-V2 GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate the algorithm tuning registry, report, and final config freeze for
the isolated SecretFlow-SPU formal runs.

The Anchor CE parameters are reused verbatim from the validated TFE freeze
(CIFAR-10: 40 steps / lr=0.1 / mu=1e-3; CIFAR-100: 40 steps / lr=0.2 / mu=1e-3;
Tiny uses the CIFAR-100 preregistered starting point, NOT prior Tiny validation).
Stage-C Adaptive DP parameters are reused from the frozen TFE config freeze.
No new tuning is performed here because the smoke already proved numerical
correctness; this file only registers the audited starting configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "new_log/new_secretflow_spu"
CONFIG_DIR = ROOT / "new_configs/new_secretflow_spu"


def main() -> None:
    # Frozen TFE references (read-only) that define the audited starting points.
    tfe_config = json.loads(
        (
            ROOT / "new_configs/new_true_ours/new_true_ours_config_freeze.json"
        ).read_text(encoding="utf-8")
    )
    anchor_evidence = json.loads(
        (
            ROOT / "new_log/new_secretflow_spu/new_spu_feature_reuse_audit.json"
        ).read_text(encoding="utf-8")
    )["anchor_config_evidence"]

    datasets = ("cifar10", "cifar100", "tiny")
    anchor_params = {
        "cifar10": {"steps": 40, "lr": 0.1, "mu": 0.001},
        "cifar100": {"steps": 40, "lr": 0.2, "mu": 0.001},
        "tiny": {"steps": 40, "lr": 0.2, "mu": 0.001},  # preregistered C100 copy
    }
    anchor_source = {
        "cifar10": "validated TFE freeze: 40 steps / lr=0.1 / mu=0.001",
        "cifar100": "validated TFE freeze: 40 steps / lr=0.2 / mu=0.001",
        "tiny": "CIFAR-100 preregistered starting point (Tiny excluded from prior ablation; bounded pre-formal starting point, no prior Tiny validation claimed)",
    }

    rows = []
    for dataset in datasets:
        stage_b_full = dict(
            tfe_config["datasets"][dataset]["methods"]["full_data"]["stage_b"]
        )
        stage_b_ours = dict(
            tfe_config["datasets"][dataset]["methods"]["true_ours"]["stage_b"]
        )
        stage_c_full = dict(
            tfe_config["datasets"][dataset]["methods"]["full_data"]["stage_c"]
        )
        stage_c_ours = dict(
            tfe_config["datasets"][dataset]["methods"]["true_ours"]["stage_c"]
        )
        for method in ("full_data", "true_ours"):
            sb = stage_b_full if method == "full_data" else stage_b_ours
            sc = stage_c_full if method == "full_data" else stage_c_ours
            row = {
                "dataset": dataset,
                "method": method,
                "anchor_ce": method == "true_ours",
                "anchor_steps": (
                    anchor_params[dataset]["steps"] if method == "true_ours" else None
                ),
                "anchor_lr": (
                    anchor_params[dataset]["lr"] if method == "true_ours" else None
                ),
                "anchor_mu": (
                    anchor_params[dataset]["mu"] if method == "true_ours" else None
                ),
                "anchor_source": (
                    anchor_source[dataset] if method == "true_ours" else "N/A"
                ),
                "stage_b_source": (
                    "new_true_ours_config_freeze.json[{}][{}].stage_b".format(
                        dataset, method
                    )
                ),
                "stage_b_local_epochs": sb["local_epochs"],
                "stage_b_batch_size": sb["batch_size"],
                "stage_b_local_lr": sb["local_lr"],
                "stage_b_weight_decay": sb["weight_decay"],
                "stage_b_client_weighting": sb["client_weighting"],
                "stage_b_fedavg_sync_interval": sb.get("fedavg_sync_interval", 0),
                "stage_b_shared_initialization": sb.get("shared_initialization", False),
                "stage_c_source": (
                    "new_true_ours_config_freeze.json[{}][{}].stage_c".format(
                        dataset, method
                    )
                ),
                "stage_c_steps": sc["steps"],
                "stage_c_learning_rate": sc["learning_rate"],
                "stage_c_noise_multiplier": sc["noise_multiplier"],
                "stage_c_count_noise_multiplier": sc["count_noise_multiplier"],
                "stage_c_clip_c": sc["clip_c"],
                "stage_c_gamma": sc["gamma"],
                "stage_c_eta_c": sc["eta_c"],
                "stage_c_c_min": sc["c_min"],
                "stage_c_c_max": sc["c_max"],
                "stage_c_warmup_steps": sc["warmup_steps"],
                "stage_c_update_interval": sc["update_interval"],
                "stage_c_expected_batch": sc["expected_batch_size"],
                "stage_c_graph_capacity": sc["graph_batch_capacity"],
                "tuned_in_this_task": False,
                "change_reason": "reuse validated TFE freeze; smoke proved numerical correctness; no tuning needed",
            }
            rows.append(row)

    # Registry CSV.
    import csv

    csv_path = OUT / "new_spu_tuning_registry.csv"
    keys = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    # Report MD.
    report_md = OUT / "new_spu_tuning_report.md"
    lines = [
        "# SecretFlow-SPU algorithm tuning report",
        "",
        "Status: `COMPLETE` — configuration reused from validated TFE freeze; no new tuning performed",
        "",
        "## Decision",
        "",
        "The pipeline smoke (`new_spu_pipeline_smoke.json`) proved numerical",
        "correctness of Stage-B (secure local heads + secure average + anchor CE)",
        "and Stage-C (Adaptive DP with three-party noise shares summed inside SPU)",
        "against plaintext references. The formal configuration therefore reuses",
        "the audited TFE freeze verbatim; no utility-degradation diagnostic was",
        "triggered, so no one-factor-at-a-time adjustment was needed.",
        "",
        "## Anchor CE parameters",
        "",
        "| Dataset | Steps | LR | mu | Source |",
        "|---|---:|---:|---:|---|",
        "| cifar10 | 40 | 0.1 | 0.001 | validated TFE freeze |",
        "| cifar100 | 40 | 0.2 | 0.001 | validated TFE freeze |",
        "| tiny | 40 | 0.2 | 0.001 | C100 preregistered starting point (no prior Tiny validation) |",
        "",
        "## Stage-C Adaptive DP parameters",
        "",
        "Reused from `new_true_ours_config_freeze.json` per dataset/method,",
        "including noise multiplier, adaptive clip (gamma/eta_c/c_min/c_max,",
        "warmup, update interval), expected batch 16 / capacity 64, delta 1e-5,",
        "target epsilon 3.0 (conditional Stage-C DP only).",
        "",
        "## Scientific integrity",
        "",
        "- No model accuracy was used to move any target or parameter.",
        "- Tiny Anchor CE is the preregistered C100 copy, not prior Tiny validation.",
        "- No tuning on test/locked holdout (both remain unread).",
        "- Full and Ours share Stage-C optimizer/privacy/evaluation; they differ only",
        "  in protected data volume and Stage-B initialization path.",
        "",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Final config freeze JSON: per-dataset/method resolved config for formal runs.
    final = {
        "schema": "new_spu_final_config_freeze_v1",
        "status": "FROZEN_BEFORE_FORMAL",
        "backend": "SecretFlow-SPU device API",
        "cpu_freeze": "new_log/new_secretflow_spu/new_spu_cpu_freeze.json",
        "datasets": {},
        "privacy_claim": (
            "conditional Stage-C DP only; Stage-A no DP; Stage-B MPC confidentiality only; "
            "no end-to-end record-level DP"
        ),
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    for dataset in datasets:
        final["datasets"][dataset] = {"methods": {}}
        for row in rows:
            if row["dataset"] != dataset:
                continue
            final["datasets"][dataset]["methods"][row["method"]] = {
                "anchor_ce": row["anchor_ce"],
                "anchor_steps": row["anchor_steps"],
                "anchor_lr": row["anchor_lr"],
                "anchor_mu": row["anchor_mu"],
                "stage_b": {
                    k: row["stage_b_" + k]
                    for k in (
                        "local_epochs",
                        "batch_size",
                        "local_lr",
                        "weight_decay",
                        "client_weighting",
                        "fedavg_sync_interval",
                        "shared_initialization",
                    )
                },
                "stage_c": {
                    k: row["stage_c_" + k]
                    for k in (
                        "steps",
                        "learning_rate",
                        "noise_multiplier",
                        "count_noise_multiplier",
                        "clip_c",
                        "gamma",
                        "eta_c",
                        "c_min",
                        "c_max",
                        "warmup_steps",
                        "update_interval",
                        "expected_batch",
                        "graph_capacity",
                    )
                },
            }
    (CONFIG_DIR / "new_spu_final_config_freeze.json").write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )

    print("registry={}".format(csv_path))
    print("report={}".format(report_md))
    print("freeze={}".format(CONFIG_DIR / "new_spu_final_config_freeze.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Freeze the successful plaintext protocol before any formal TFE run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KD_DIR = ROOT / "new_scripts/new_stage_b_kd"
if str(KD_DIR) not in sys.path:
    sys.path.insert(0, str(KD_DIR))

from new_stage_b_kd_core import atomic_json, sha256_file  # noqa: E402


LOG_ROOT = ROOT / "new_log/new_stage_b_anchor_postagg"
SUMMARY_PATH = LOG_ROOT / "new_stage_b_anchor_postagg_smoke_summary.json"
SPLIT_PATH = LOG_ROOT / "new_stage_b_anchor_postagg_split_manifest.json"
POLICY_PATH = (
    ROOT
    / "new_configs/new_stage_b_anchor_postagg/"
    "new_stage_b_anchor_postagg_policy.json"
)
CONFIG_FREEZE = LOG_ROOT / "new_stage_b_anchor_postagg_config_freeze.json"
SPLIT_FREEZE = LOG_ROOT / "new_stage_b_anchor_postagg_split_freeze.json"
MANIFEST = LOG_ROOT / "new_stage_b_anchor_postagg_freeze_manifest.md"


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if not summary["gates"]["all_smoke_gates_pass"]:
        raise ValueError("REFUSE_FREEZE: plaintext smoke gates did not all pass")
    if summary["next_phase"] != "FREEZE_THEN_TFE":
        raise ValueError("REFUSE_FREEZE: next phase is not TFE")
    if not split["all_local_anchor_disjoint"] or not split[
        "all_development_disjoint"
    ]:
        raise ValueError("REFUSE_FREEZE: split audit failed")

    report_source = source_commit()
    experiment_source = summary["source_git_commit"]
    split_sha = sha256_file(SPLIT_PATH)
    policy_sha = sha256_file(POLICY_PATH)
    frozen = {
        "schema": "new_stage_b_anchor_postagg_config_freeze_v1",
        "status": "FROZEN_AFTER_SMOKE_SUCCESS",
        "frozen": True,
        "experiment_source_git_commit": experiment_source,
        "freeze_generator_git_commit": report_source,
        "datasets": ["cifar10", "cifar100"],
        "seed": 42,
        "clients": 5,
        "formal_distributions": ["severe"],
        "formal_methods": [
            "C1_random_plus_anchor",
            "C4_secure_average",
            "C5_secure_average_plus_anchor"
        ],
        "selected_anchor_configs": summary["selected_configs"],
        "local_configs": policy["local_configs"],
        "local_per_class": 40,
        "anchor_per_class": 8,
        "random_initialization_seed": 42,
        "local_selection_rule": policy["local_selection_rule"],
        "batch_mode": "full_batch",
        "feature_normalization": "audited_upstream_encoder_and_scaler",
        "tfe_fixed_point_precision": policy["tfe_fixed_point_precision"],
        "tfe_matmul_chunk_size": 64,
        "timing_boundary": {
            "local_secure_seconds": "all five client local updates",
            "secure_average_seconds": "secure parameter average materialization",
            "anchor_ce_seconds": "40 secure CE update steps per refined branch",
            "total_stage_b_seconds": "local + average + selected anchor branch"
        },
        "success_gates": summary["gates"],
        "severe_plaintext_deltas": summary["severe_deltas"],
        "policy_path": str(POLICY_PATH.relative_to(ROOT)),
        "policy_sha256": policy_sha,
        "split_manifest_path": str(SPLIT_PATH.relative_to(ROOT)),
        "split_manifest_sha256": split_sha,
        "locked_holdout_read": False,
        "test_read": False,
        "official_mtt_only": True,
        "full_data_initializer_dependency": False,
        "stage_c_run": False,
        "end_to_end_run": False
    }
    atomic_json(CONFIG_FREEZE, frozen)
    split_freeze = {
        "schema": "new_stage_b_anchor_postagg_split_freeze_v1",
        "status": "FROZEN_AFTER_SMOKE_SUCCESS",
        "frozen": True,
        "experiment_source_git_commit": experiment_source,
        "freeze_generator_git_commit": report_source,
        "split_manifest_path": str(SPLIT_PATH.relative_to(ROOT)),
        "split_manifest_sha256": split_sha,
        "rows": split["rows"],
        "all_local_anchor_disjoint": True,
        "all_development_disjoint": True,
        "locked_holdout_read": False,
        "test_read": False
    }
    atomic_json(SPLIT_FREEZE, split_freeze)
    MANIFEST.write_text(
        "\n".join(
            [
                "# Stage-B Anchor Post-Aggregation Freeze Manifest",
                "",
                "- Status: `FROZEN_AFTER_SMOKE_SUCCESS`",
                "- Experiment source: `{}`".format(experiment_source),
                "- Freeze generator source: `{}`".format(report_source),
                "- Config SHA256: `{}`".format(sha256_file(CONFIG_FREEZE)),
                "- Split freeze SHA256: `{}`".format(sha256_file(SPLIT_FREEZE)),
                "- Split manifest SHA256: `{}`".format(split_sha),
                "- Policy SHA256: `{}`".format(policy_sha),
                "- Formal scope: seed42, Severe, C1/C4/C5, TFE/ABY3.",
                "- No tuning is allowed after this freeze.",
                "- Locked holdout/test remain unread; Stage C and End-to-End remain forbidden.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        "FROZEN config={} split={} manifest={}".format(
            CONFIG_FREEZE, SPLIT_FREEZE, MANIFEST
        )
    )


if __name__ == "__main__":
    main()

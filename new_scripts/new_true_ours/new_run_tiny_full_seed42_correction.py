#!/usr/bin/env python3
"""Run only the clean Tiny Full-Data seed42 formal correction."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_scripts.new_true_ours.new_run_true_ours_formal_three_seed import (
    CONFIG_PATH,
    TFE_PYTHON,
    atomic_json,
    file_sha256,
    run_logged,
    validate_stage_b,
    validate_stage_c,
    write_timeline,
)


DATASET = "tiny"
METHOD = "full_data"
SEED = 42
NAMESPACE = "full_data_formal_corrected_v2"
POLICY_PATH = (
    ROOT
    / "new_configs/new_true_ours/new_true_ours_minimal_correction_policy.json"
)
STATUS_PATH = (
    ROOT / "new_log/new_true_ours/new_tiny_full_seed42_correction_status.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correction_paths():
    stage_b_output = (
        ROOT
        / "new_output/new_true_ours/stage_b"
        / NAMESPACE
        / DATASET
        / "seed42"
    )
    stage_c_output = (
        ROOT
        / "new_output/new_true_ours/stage_c"
        / NAMESPACE
        / DATASET
        / "seed42"
    )
    stage_b_checkpoint = (
        ROOT
        / "new_checkpoints/new_true_ours/stage_b"
        / NAMESPACE
        / DATASET
        / "seed42/new_stage_b"
    )
    stage_c_checkpoint = (
        ROOT
        / "new_checkpoints/new_true_ours/stage_c"
        / NAMESPACE
        / DATASET
        / "seed42/new_stage_c"
    )
    return {
        "lineage": "full_data_disjoint",
        "cache_mode": "memmap",
        "stage_b_cache": (
            ROOT
            / "new_cache/new_true_ours/full_data_stage_b/tiny/"
            "new_full_stage_b_tiny_seed42.npz"
        ),
        "stage_c_cache": (
            ROOT
            / "new_cache/new_true_ours/full_data_stage_c/tiny/"
            "new_full_stage_c_tiny_seed42.npz"
        ),
        "stage_b_output": stage_b_output,
        "stage_c_output": stage_c_output,
        "stage_b_checkpoint": stage_b_checkpoint,
        "stage_c_checkpoint": stage_c_checkpoint,
        "stage_b_report": (
            stage_b_output
            / "new_secure_stage_b_report_tiny_balanced_seed42.json"
        ),
        "initializer": (
            stage_b_output
            / "new_secure_stage_b_candidate_nokd_tiny_balanced_seed42.npz"
        ),
        "stage_c_report": (
            stage_c_output
            / "new_secure_stage_c_report_tiny_adaptive_dp_seed42.json"
        ),
        "stage_b_log": (
            ROOT
            / "new_log/new_true_ours/txt/formal_corrected_v2/"
            "new_formal_stage_b_full_data_tiny_seed42.log"
        ),
        "stage_c_log": (
            ROOT
            / "new_log/new_true_ours/txt/formal_corrected_v2/"
            "new_formal_stage_c_full_data_tiny_seed42.log"
        ),
        "runner_log": (
            ROOT
            / "new_log/new_true_ours/txt/formal_corrected_v2/"
            "new_tiny_full_seed42_correction_runner.log"
        ),
        "timeline": (
            ROOT
            / "new_log/new_true_ours/formal_timelines/tiny/"
            "full_data_corrected_v2/new_true_ours_timeline_seed42.json"
        ),
        "checkpoint_manifest": (
            ROOT
            / "new_log/new_true_ours/"
            "new_tiny_full_seed42_corrected_checkpoint_manifest.json"
        ),
    }


def preflight(config, policy, paths):
    authorized = policy["only_authorized_rerun"]
    if (
        authorized["dataset"] != DATASET
        or authorized["method"] != METHOD
        or int(authorized["seed"]) != SEED
        or authorized["namespace"] != NAMESPACE
    ):
        raise ValueError("policy does not authorize this exact correction")
    if policy.get("locked_holdout_read") is not False:
        raise ValueError("locked holdout policy is not false")
    if policy.get("official_test_read") is not False:
        raise ValueError("official test policy is not false")
    frozen = config["datasets"][DATASET]["methods"][METHOD]
    stage_b = frozen["stage_b"]
    stage_c = frozen["stage_c"]
    expected = {
        "stage_b_local_epochs": 40,
        "stage_b_batch_size": 16,
        "stage_b_local_lr": 0.1,
        "stage_c_steps": 180,
        "stage_c_learning_rate": 0.002,
        "stage_c_noise_multiplier": 0.5685304635176568,
        "stage_c_gamma": 0.5,
        "stage_c_eta_c": 0.1,
    }
    actual = {
        "stage_b_local_epochs": int(stage_b["local_epochs"]),
        "stage_b_batch_size": int(stage_b["batch_size"]),
        "stage_b_local_lr": float(stage_b["local_lr"]),
        "stage_c_steps": int(stage_c["steps"]),
        "stage_c_learning_rate": float(stage_c["learning_rate"]),
        "stage_c_noise_multiplier": float(stage_c["noise_multiplier"]),
        "stage_c_gamma": float(stage_c["gamma"]),
        "stage_c_eta_c": float(stage_c["eta_c"]),
    }
    if actual != expected:
        raise ValueError("frozen Tiny Full configuration mismatch: {}".format(actual))
    if frozen["initializer_lineage"] != "full_data_disjoint":
        raise ValueError("Tiny Full lineage is not full_data_disjoint")
    for key in ("stage_b_cache", "stage_c_cache"):
        if not paths[key].is_file():
            raise FileNotFoundError(str(paths[key]))
    split = config["datasets"][DATASET]["splits"]["42"]
    if file_sha256(split["split_npz"]) != split["split_npz_sha256"]:
        raise ValueError("frozen Tiny seed42 split SHA mismatch")
    for relative, expected_sha in config["exact_code_sha256"].items():
        if file_sha256(ROOT / relative) != expected_sha:
            raise ValueError("frozen code SHA mismatch: {}".format(relative))
    old_prefix = (
        ROOT
        / "new_checkpoints/new_true_ours/stage_b/full_data_formal/"
        "tiny/seed42/new_stage_b"
    )
    if paths["stage_b_checkpoint"] == old_prefix:
        raise ValueError("correction checkpoint aliases superseded checkpoint")
    return {"expected": expected, "actual": actual, "passed": True}


def stage_b_command(paths, frozen):
    config = frozen["stage_b"]
    command = [
        str(TFE_PYTHON),
        str(ROOT / "new_stage_b_secure_init.py"),
        "--feature_cache",
        str(paths["stage_b_cache"]),
        "--feature_cache_mode",
        paths["cache_mode"],
        "--dataset",
        DATASET,
        "--output_dir",
        str(paths["stage_b_output"]),
        "--checkpoint_prefix",
        str(paths["stage_b_checkpoint"]),
        "--seed",
        str(SEED),
        "--num_clients",
        str(config["num_clients"]),
        "--split",
        config["split"],
        "--batch_size",
        str(config["batch_size"]),
        "--local_epochs",
        str(config["local_epochs"]),
        "--local_lr",
        str(config["local_lr"]),
        "--weight_decay",
        str(config["weight_decay"]),
        "--client_weighting",
        config["client_weighting"],
        "--fedavg_sync_interval",
        str(config["fedavg_sync_interval"]),
        "--distill_epochs",
        "0",
        "--lambda_kd",
        "0",
        "--reference_supervised_weight",
        "0",
        "--precision_fractional",
        str(config["precision_fractional"]),
        "--matmul_chunk_size",
        str(config["matmul_chunk_size"]),
        "--checkpoint_interval",
        str(config["checkpoint_interval_epochs"]),
        "--formal_candidate_release",
        "--secure_average_only",
        "--development_checkpoint_evaluation",
        "--input_lineage",
        frozen["initializer_lineage"],
    ]
    if config["shared_initialization"]:
        command.append("--shared_initialization")
    return ["timeout", "--foreground", "-k", "60", "14400"] + command


def stage_c_command(paths, frozen):
    config = frozen["stage_c"]
    return [
        str(TFE_PYTHON),
        str(ROOT / "new_stage_c_secure_dpsgd.py"),
        "--feature_cache",
        str(paths["stage_c_cache"]),
        "--feature_cache_mode",
        paths["cache_mode"],
        "--initializer",
        str(paths["initializer"]),
        "--dataset",
        DATASET,
        "--mode",
        "adaptive_dp",
        "--output_dir",
        str(paths["stage_c_output"]),
        "--checkpoint_prefix",
        str(paths["stage_c_checkpoint"]),
        "--seed",
        str(SEED),
        "--batch_size",
        str(config["graph_batch_capacity"]),
        "--expected_batch_size",
        str(config["expected_batch_size"]),
        "--sampling_mode",
        "poisson",
        "--steps",
        str(config["steps"]),
        "--learning_rate",
        str(config["learning_rate"]),
        "--noise_multiplier",
        str(config["noise_multiplier"]),
        "--count_noise_multiplier",
        str(config["count_noise_multiplier"]),
        "--clip_c",
        str(config["clip_c"]),
        "--clipping_scope",
        config["clipping_scope"],
        "--gamma",
        str(config["gamma"]),
        "--eta_c",
        str(config["eta_c"]),
        "--c_min",
        str(config["c_min"]),
        "--c_max",
        str(config["c_max"]),
        "--warmup_steps",
        str(config["warmup_steps"]),
        "--update_interval",
        str(config["update_interval"]),
        "--delta",
        str(config["delta"]),
        "--precision_fractional",
        str(config["precision_fractional"]),
        "--matmul_chunk_size",
        str(config["matmul_chunk_size"]),
        "--checkpoint_interval",
        str(config["checkpoint_interval_steps"]),
        "--initializer_privacy_status",
        config["initializer_privacy_status"],
        "--input_lineage",
        frozen["initializer_lineage"],
        "--evaluation_split",
        "development_validation",
    ]


def checkpoint_manifest(paths):
    files = []
    for prefix in (paths["stage_b_checkpoint"], paths["stage_c_checkpoint"]):
        for path in sorted(prefix.parent.glob(prefix.name + "*")):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    payload = {
        "schema": "new_true_ours_local_checkpoint_manifest_v1",
        "git_policy": "local_only",
        "dataset": DATASET,
        "method": METHOD,
        "seed": SEED,
        "namespace": NAMESPACE,
        "files": files,
    }
    atomic_json(paths["checkpoint_manifest"], payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    paths = correction_paths()
    frozen = config["datasets"][DATASET]["methods"][METHOD]
    checks = preflight(config, policy, paths)
    status = {
        "schema": "new_tiny_full_seed42_correction_status_v1",
        "status": "PREFLIGHT_PASS",
        "dataset": DATASET,
        "method": METHOD,
        "seed": SEED,
        "namespace": NAMESPACE,
        "preflight": checks,
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    atomic_json(STATUS_PATH, status)
    print("CORRECTION_PREFLIGHT_PASS tiny full_data seed42")
    if args.preflight_only:
        return 0
    for key in ("stage_b_output", "stage_c_output"):
        paths[key].mkdir(parents=True, exist_ok=True)
    for key in ("stage_b_checkpoint", "stage_c_checkpoint"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    status["status"] = "STAGE_B_RUNNING"
    atomic_json(STATUS_PATH, status)
    if not paths["stage_b_report"].is_file():
        run_logged(
            stage_b_command(paths, frozen),
            paths["stage_b_log"],
            max_attempts=2,
        )
    stage_b, stage_b_checks = validate_stage_b(
        paths["stage_b_report"], DATASET, METHOD, SEED, frozen
    )
    if not paths["initializer"].is_file():
        raise FileNotFoundError(str(paths["initializer"]))
    status["status"] = "STAGE_B_COMPLETE"
    status["stage_b_checks"] = stage_b_checks
    status["stage_b_report"] = str(paths["stage_b_report"].relative_to(ROOT))
    status["stage_b_report_sha256"] = sha256_file(paths["stage_b_report"])
    atomic_json(STATUS_PATH, status)
    status["status"] = "STAGE_C_RUNNING"
    atomic_json(STATUS_PATH, status)
    if not paths["stage_c_report"].is_file():
        run_logged(stage_c_command(paths, frozen), paths["stage_c_log"])
    stage_c, stage_c_checks = validate_stage_c(
        paths["stage_c_report"], DATASET, METHOD, SEED, frozen
    )
    timeline = write_timeline(
        paths, DATASET, METHOD, SEED, stage_b, stage_c
    )
    manifest = checkpoint_manifest(paths)
    status.update(
        {
            "status": "COMPLETE",
            "stage_c_checks": stage_c_checks,
            "stage_c_report": str(paths["stage_c_report"].relative_to(ROOT)),
            "stage_c_report_sha256": sha256_file(paths["stage_c_report"]),
            "timeline": str(paths["timeline"].relative_to(ROOT)),
            "timeline_sha256": sha256_file(paths["timeline"]),
            "total_online_secure_seconds": timeline[
                "total_online_secure_seconds"
            ],
            "checkpoint_manifest": str(
                paths["checkpoint_manifest"].relative_to(ROOT)
            ),
            "checkpoint_file_count": len(manifest["files"]),
        }
    )
    atomic_json(STATUS_PATH, status)
    print("CORRECTION_COMPLETE {}".format(STATUS_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

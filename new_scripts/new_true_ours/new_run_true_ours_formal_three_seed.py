#!/usr/bin/env python3
"""Checkpointed formal Full/True Stage-B + Adaptive Stage-C queue."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_stage_b_secure_init import sha256_file


TFE_PYTHON = Path("/root/autodl-tmp/new_envs/tfe/bin/python")
MODERN_PYTHON = Path("/root/autodl-tmp/new_envs/distillation/bin/python")
CONFIG_PATH = (
    ROOT / "new_configs/new_true_ours/new_true_ours_config_freeze.json"
)
TARGET_PATH = (
    ROOT / "new_configs/new_true_ours/new_true_ours_target_freeze.json"
)
STATUS_PATH = (
    ROOT / "new_log/new_true_ours/new_true_ours_formal_queue_status.json"
)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command, log_path, timeout_seconds=None, max_attempts=1):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            time.sleep(30)  # cool-down between attempts
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\nRUN_START {} ATTEMPT {}/{}\nCOMMAND {}\n".format(
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    attempt, max_attempts,
                    " ".join(command),
                )
            )
            handle.flush()
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(ROOT),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as e:
                handle.write(f"\nTIMEOUT after {timeout_seconds}s\n")
                handle.flush()
                last_exc = e
                continue
        if completed.returncode != 0:
            raise RuntimeError(
                "command failed with exit {}: {}".format(
                    completed.returncode, log_path
                )
            )
        return  # success
    raise RuntimeError(
        f"command timed out after {max_attempts} attempts: {log_path}"
    )


def conditional_epsilon_at_step(stage_c_report, step):
    config = stage_c_report["configuration"]
    privacy = stage_c_report["privacy"]
    update_count = sum(
        1
        for index in range(int(step))
        if index >= int(config["warmup_steps"])
        and (index + 1) % int(config["update_interval"]) == 0
    )
    command = [
        str(MODERN_PYTHON),
        str(ROOT / "new_privacy_accountant.py"),
        "--steps",
        str(int(step)),
        "--sample_rate",
        str(float(privacy["sample_rate"])),
        "--noise_multiplier",
        str(float(privacy["gradient_noise_multiplier"])),
        "--delta",
        str(float(privacy["stage_c_delta"])),
        "--count_noise",
        str(float(privacy["clipping_count_noise_multiplier"])),
        "--count_steps",
        str(update_count),
    ]
    payload = json.loads(
        subprocess.check_output(
            command, cwd=str(ROOT), universal_newlines=True
        )
    )
    return float(payload["epsilon"])


def task_paths(dataset, method, seed):
    lineage = (
        "full_data_disjoint" if method == "full_data" else "official_mtt_only"
    )
    if method == "full_data":
        stage_b_cache = (
            ROOT
            / "new_cache/new_true_ours/full_data_stage_b"
            / dataset
            / "new_full_stage_b_{}_seed{}.npz".format(dataset, seed)
        )
        stage_c_cache = (
            ROOT
            / "new_cache/new_true_ours/full_data_stage_c"
            / dataset
            / "new_full_stage_c_{}_seed{}.npz".format(dataset, seed)
        )
        cache_mode = "memmap"
        stage_b_namespace = "full_data_formal"
    else:
        stage_b_cache = (
            ROOT
            / "new_cache/new_true_ours/official_mtt/stage_b"
            / dataset
            / "new_true_ours_stage_b_{}_seed{}.npz".format(dataset, seed)
        )
        stage_c_cache = (
            ROOT
            / "new_cache/new_true_ours/official_mtt/stage_c"
            / dataset
            / "new_true_ours_stage_c_{}_seed{}.npz".format(dataset, seed)
        )
        cache_mode = "npz"
        stage_b_namespace = "true_ours_formal"
    stage_b_output = (
        ROOT
        / "new_output/new_true_ours/stage_b"
        / stage_b_namespace
        / dataset
        / "seed{}".format(seed)
    )
    stage_b_checkpoint = (
        ROOT
        / "new_checkpoints/new_true_ours/stage_b"
        / stage_b_namespace
        / dataset
        / "seed{}".format(seed)
        / "new_stage_b"
    )
    stage_b_report = stage_b_output / (
        "new_secure_stage_b_report_{}_balanced_seed{}.json".format(
            dataset, seed
        )
    )
    initializer = stage_b_output / (
        "new_secure_stage_b_candidate_nokd_{}_balanced_seed{}.npz".format(
            dataset, seed
        )
    )
    stage_c_output = (
        ROOT
        / "new_output/new_true_ours/stage_c"
        / "{}_formal".format(method)
        / dataset
        / "seed{}".format(seed)
    )
    stage_c_checkpoint = (
        ROOT
        / "new_checkpoints/new_true_ours/stage_c"
        / "{}_formal".format(method)
        / dataset
        / "seed{}".format(seed)
        / "new_stage_c"
    )
    stage_c_report = stage_c_output / (
        "new_secure_stage_c_report_{}_adaptive_dp_seed{}.json".format(
            dataset, seed
        )
    )
    log_dir = ROOT / "new_log/new_true_ours/txt/formal"
    return {
        "lineage": lineage,
        "cache_mode": cache_mode,
        "stage_b_cache": stage_b_cache,
        "stage_c_cache": stage_c_cache,
        "stage_b_output": stage_b_output,
        "stage_b_checkpoint": stage_b_checkpoint,
        "stage_b_report": stage_b_report,
        "initializer": initializer,
        "stage_c_output": stage_c_output,
        "stage_c_checkpoint": stage_c_checkpoint,
        "stage_c_report": stage_c_report,
        "stage_b_log": log_dir
        / "new_formal_stage_b_{}_{}_seed{}.log".format(
            method, dataset, seed
        ),
        "stage_c_log": log_dir
        / "new_formal_stage_c_{}_{}_seed{}.log".format(
            method, dataset, seed
        ),
        "timeline": (
            ROOT
            / "new_log/new_true_ours/formal_timelines"
            / dataset
            / method
            / "new_true_ours_timeline_seed{}.json".format(seed)
        ),
    }


def validate_stage_b(report_path, dataset, method, seed, frozen):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    config = report["configuration"]
    security = report["security"]
    expected = frozen["stage_b"]
    checkpoint_rows = [
        row
        for row in report["history"]
        if row.get("development_validation_accuracy") is not None
    ]
    checks = {
        "complete_identity": (
            report.get("status") == "COMPLETE"
            and report.get("dataset") == dataset
            and int(report.get("seed")) == int(seed)
            and report.get("split") == "balanced"
        ),
        "lineage": (
            report["initializer_lineage"].get("validated") is True
            and report["initializer_lineage"].get("initializer_source")
            == frozen["initializer_lineage"]
            and bool(
                report["initializer_lineage"].get(
                    "full_data_initializer_dependency"
                )
            )
            == bool(frozen["full_data_initializer_dependency"])
        ),
        "secure_average_only": (
            config.get("secure_average_only") is True
            and security.get("backend") == "tfe_mpc"
            and security.get("protocol") == "ABY3"
            and security.get("silent_plaintext_fallback") is False
        ),
        "matched_stage_b_config": all(
            config.get(key) == value
            for key, value in expected.items()
            if key
            in (
                "local_epochs",
                "batch_size",
                "local_lr",
                "weight_decay",
                "client_weighting",
                "fedavg_sync_interval",
                "shared_initialization",
                "distill_epochs",
                "lambda_kd",
                "reference_supervised_weight",
                "precision_fractional",
                "matmul_chunk_size",
                "secure_average_only",
                "development_checkpoint_evaluation",
            )
        ),
        "checkpoint_development_curve": (
            len(checkpoint_rows)
            == int(expected["local_epochs"])
            // int(expected["checkpoint_interval_epochs"])
            and all(
                row.get("evaluation_split") == "development_validation"
                and row.get("locked_holdout_read") is False
                and row.get("official_test_read") is False
                and float(row.get("cumulative_online_secure_seconds")) > 0
                for row in checkpoint_rows
            )
        ),
        "timing": float(report.get("online_secure_runtime_seconds", 0)) > 0,
        "candidate": "NoKD" in report.get("formal_candidate_models", {}),
    }
    if not all(checks.values()):
        raise ValueError(
            "Stage-B validation failed {} {} seed{}: {}".format(
                method, dataset, seed, checks
            )
        )
    return report, checks


def validate_stage_c(report_path, dataset, method, seed, frozen):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    config = report["configuration"]
    privacy = report["privacy"]
    security = report["security"]
    evaluation = report["evaluation"]
    expected = frozen["stage_c"]
    checks = {
        "complete_identity": (
            report.get("status") == "COMPLETE"
            and report.get("dataset") == dataset
            and int(report.get("seed")) == int(seed)
            and report.get("mode") == "adaptive_dp"
        ),
        "lineage": (
            report["initializer_lineage"].get("validated") is True
            and config.get("input_lineage") == frozen["initializer_lineage"]
        ),
        "tfe_aby3_no_fallback": (
            security.get("backend") == "tfe_mpc"
            and security.get("protocol") == "ABY3"
            and security.get("silent_plaintext_fallback") is False
        ),
        "poisson_accountant": (
            privacy.get("sampling_mode") == "poisson"
            and privacy.get("sampling_rng") == "os_csprng"
            and int(privacy.get("expected_batch_size"))
            == int(expected["expected_batch_size"])
            and int(privacy.get("graph_batch_capacity"))
            == int(expected["graph_batch_capacity"])
            and float(
                privacy.get("stage_c_training_feature_record_epsilon")
            )
            <= 3.000001
        ),
        "conditional_claim": (
            privacy.get("initializer_privacy_status")
            == "data_dependent_non_dp"
            and privacy.get("released_model_record_dp_valid") is False
            and privacy.get("end_to_end_record_dp_claimed") is False
        ),
        "development_only": (
            evaluation.get("split") == "development_validation"
            and evaluation.get("locked_holdout_read") is False
            and evaluation.get("official_test_read") is False
        ),
        "matched_stage_c_config": all(
            abs(float(config[key]) - float(expected[key])) <= 1e-12
            for key in (
                "steps",
                "learning_rate",
                "clip_c",
                "gamma",
                "eta_c",
                "c_min",
                "c_max",
                "warmup_steps",
                "update_interval",
                "count_noise_multiplier",
                "noise_multiplier",
            )
        ),
        "history_and_timing": (
            len(report.get("history", [])) == int(expected["steps"])
            and float(report.get("online_secure_runtime_seconds", 0)) > 0
            and all(
                float(row.get("cumulative_stage_c_online_secure_seconds", 0))
                > 0
                for row in report["history"]
            )
        ),
        "finite_nonboundary_c": (
            float(expected["c_min"])
            < float(report.get("final_c"))
            < float(expected["c_max"])
        ),
    }
    if not all(checks.values()):
        raise ValueError(
            "Stage-C validation failed {} {} seed{}: {}".format(
                method, dataset, seed, checks
            )
        )
    return report, checks


def write_timeline(paths, dataset, method, seed, stage_b, stage_c):
    stage_b_points = [
        {
            "stage": "stage_b",
            "epoch_or_step": int(row["epoch"]),
            "cumulative_online_secure_seconds": float(
                row["cumulative_online_secure_seconds"]
            ),
            "development_validation_accuracy": float(
                row["development_validation_accuracy"]
            ),
            "epsilon_so_far": None,
            "epsilon_scope": "not_applicable_non_dp_stage_b",
            "processed_secure_examples": None,
        }
        for row in stage_b["history"]
        if row.get("development_validation_accuracy") is not None
    ]
    stage_b_total = float(stage_b["online_secure_runtime_seconds"])
    stage_c_points = [
        {
            "stage": "stage_c",
            "epoch_or_step": int(row["step"]),
            "cumulative_online_secure_seconds": (
                stage_b_total
                + float(row["cumulative_stage_c_online_secure_seconds"])
            ),
            "development_validation_accuracy": float(
                row["development_validation_accuracy"]
            ),
            "epsilon_so_far": conditional_epsilon_at_step(
                stage_c, int(row["step"])
            ),
            "epsilon_scope": "conditional_stage_c_mechanisms_only",
            "processed_secure_examples": (
                int(row["step"]) * 16
            ),
        }
        for row in stage_c["history"]
        if row.get("development_validation_accuracy") is not None
    ]
    payload = {
        "schema": "new_true_ours_formal_timeline_v1",
        "status": "COMPLETE_DEVELOPMENT_ONLY",
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "timing_boundary": (
            "Stage-B secure graph/session start through Stage-C final model"
        ),
        "stage_b_online_secure_seconds": stage_b_total,
        "stage_c_online_secure_seconds": float(
            stage_c["online_secure_runtime_seconds"]
        ),
        "total_online_secure_seconds": (
            stage_b_total
            + float(stage_c["online_secure_runtime_seconds"])
        ),
        "points": stage_b_points + stage_c_points,
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    atomic_json(paths["timeline"], payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    if (
        config.get("status") != "FROZEN_BEFORE_FORMAL"
        or target.get("status") != "FROZEN_BEFORE_FORMAL"
        or config.get("locked_holdout_read") is not False
        or target.get("locked_holdout_read") is not False
    ):
        raise ValueError("formal protocol/target is not frozen")
    for relative, expected_sha in config["exact_code_sha256"].items():
        if file_sha256(ROOT / relative) != expected_sha:
            raise ValueError("frozen code SHA mismatch: {}".format(relative))
    tasks = [
        (dataset, method, int(seed))
        for dataset in ("cifar10", "cifar100", "tiny")
        for method in ("true_ours", "full_data")
        for seed in config["seeds"]
    ]
    for dataset, method, seed in tasks:
        paths = task_paths(dataset, method, seed)
        for key in ("stage_b_cache", "stage_c_cache"):
            if not paths[key].is_file():
                raise FileNotFoundError(str(paths[key]))
        split = config["datasets"][dataset]["splits"][str(seed)]
        if file_sha256(split["split_npz"]) != split["split_npz_sha256"]:
            raise ValueError(
                "frozen split SHA mismatch: {} seed{}".format(dataset, seed)
            )
    if args.preflight_only:
        print(
            "FORMAL_PREFLIGHT_PASS tasks={} locked_holdout_read=false "
            "official_test_read=false".format(len(tasks))
        )
        return 0
    status = {
        "schema": "new_true_ours_formal_queue_status_v1",
        "status": "RUNNING",
        "total": len(tasks),
        "completed": 0,
        "rows": [],
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    for dataset, method, seed in tasks:
        frozen = config["datasets"][dataset]["methods"][method]
        paths = task_paths(dataset, method, seed)
        for key in ("stage_b_output", "stage_c_output"):
            paths[key].mkdir(parents=True, exist_ok=True)
        for key in ("stage_b_checkpoint", "stage_c_checkpoint"):
            paths[key].parent.mkdir(parents=True, exist_ok=True)
        stage_b_config = frozen["stage_b"]
        if not paths["stage_b_report"].is_file():
            command = [
                str(TFE_PYTHON),
                str(ROOT / "new_stage_b_secure_init.py"),
                "--feature_cache",
                str(paths["stage_b_cache"]),
                "--feature_cache_mode",
                paths["cache_mode"],
                "--dataset",
                dataset,
                "--output_dir",
                str(paths["stage_b_output"]),
                "--checkpoint_prefix",
                str(paths["stage_b_checkpoint"]),
                "--seed",
                str(seed),
                "--num_clients",
                str(stage_b_config["num_clients"]),
                "--split",
                stage_b_config["split"],
                "--batch_size",
                str(stage_b_config["batch_size"]),
                "--local_epochs",
                str(stage_b_config["local_epochs"]),
                "--local_lr",
                str(stage_b_config["local_lr"]),
                "--weight_decay",
                str(stage_b_config["weight_decay"]),
                "--client_weighting",
                stage_b_config["client_weighting"],
                "--fedavg_sync_interval",
                str(stage_b_config["fedavg_sync_interval"]),
                "--distill_epochs",
                "0",
                "--lambda_kd",
                "0",
                "--reference_supervised_weight",
                "0",
                "--precision_fractional",
                str(stage_b_config["precision_fractional"]),
                "--matmul_chunk_size",
                str(stage_b_config["matmul_chunk_size"]),
                "--checkpoint_interval",
                str(stage_b_config["checkpoint_interval_epochs"]),
                "--formal_candidate_release",
                "--secure_average_only",
                "--development_checkpoint_evaluation",
                "--input_lineage",
                frozen["initializer_lineage"],
            ]
            if stage_b_config["shared_initialization"]:
                command.append("--shared_initialization")
            # Full Data Stage B is heavy; wrap with timeout to survive TF 1.15 hangs
            if method == "full_data":
                # Use shell timeout (kills entire process group, handles TF threads)
                command = ["timeout", "--foreground", "-k", "60", "7200"] + command
            sb_attempts = 2 if method == "full_data" else 1
            run_logged(command, paths["stage_b_log"], max_attempts=sb_attempts)
        stage_b, stage_b_checks = validate_stage_b(
            paths["stage_b_report"], dataset, method, seed, frozen
        )
        if not paths["initializer"].is_file():
            raise FileNotFoundError(str(paths["initializer"]))
        stage_c_config = frozen["stage_c"]
        if not paths["stage_c_report"].is_file():
            command = [
                str(TFE_PYTHON),
                str(ROOT / "new_stage_c_secure_dpsgd.py"),
                "--feature_cache",
                str(paths["stage_c_cache"]),
                "--feature_cache_mode",
                paths["cache_mode"],
                "--initializer",
                str(paths["initializer"]),
                "--dataset",
                dataset,
                "--mode",
                "adaptive_dp",
                "--output_dir",
                str(paths["stage_c_output"]),
                "--checkpoint_prefix",
                str(paths["stage_c_checkpoint"]),
                "--seed",
                str(seed),
                "--batch_size",
                str(stage_c_config["graph_batch_capacity"]),
                "--expected_batch_size",
                str(stage_c_config["expected_batch_size"]),
                "--sampling_mode",
                "poisson",
                "--steps",
                str(stage_c_config["steps"]),
                "--learning_rate",
                str(stage_c_config["learning_rate"]),
                "--noise_multiplier",
                str(stage_c_config["noise_multiplier"]),
                "--count_noise_multiplier",
                str(stage_c_config["count_noise_multiplier"]),
                "--clip_c",
                str(stage_c_config["clip_c"]),
                "--clipping_scope",
                stage_c_config["clipping_scope"],
                "--gamma",
                str(stage_c_config["gamma"]),
                "--eta_c",
                str(stage_c_config["eta_c"]),
                "--c_min",
                str(stage_c_config["c_min"]),
                "--c_max",
                str(stage_c_config["c_max"]),
                "--warmup_steps",
                str(stage_c_config["warmup_steps"]),
                "--update_interval",
                str(stage_c_config["update_interval"]),
                "--delta",
                str(stage_c_config["delta"]),
                "--precision_fractional",
                str(stage_c_config["precision_fractional"]),
                "--matmul_chunk_size",
                str(stage_c_config["matmul_chunk_size"]),
                "--checkpoint_interval",
                str(stage_c_config["checkpoint_interval_steps"]),
                "--initializer_privacy_status",
                stage_c_config["initializer_privacy_status"],
                "--input_lineage",
                frozen["initializer_lineage"],
                "--evaluation_split",
                "development_validation",
            ]
            run_logged(command, paths["stage_c_log"])
        stage_c, stage_c_checks = validate_stage_c(
            paths["stage_c_report"], dataset, method, seed, frozen
        )
        timeline = write_timeline(
            paths, dataset, method, seed, stage_b, stage_c
        )
        row = {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "stage_b_report": str(paths["stage_b_report"]),
            "stage_b_report_sha256": sha256_file(
                str(paths["stage_b_report"])
            ),
            "stage_c_report": str(paths["stage_c_report"]),
            "stage_c_report_sha256": sha256_file(
                str(paths["stage_c_report"])
            ),
            "timeline": str(paths["timeline"]),
            "timeline_sha256": sha256_file(str(paths["timeline"])),
            "stage_b_checks": stage_b_checks,
            "stage_c_checks": stage_c_checks,
            "total_online_secure_seconds": timeline[
                "total_online_secure_seconds"
            ],
        }
        status["rows"] = [
            existing
            for existing in status["rows"]
            if not (
                existing["dataset"] == dataset
                and existing["method"] == method
                and int(existing["seed"]) == seed
            )
        ] + [row]
        status["completed"] = len(status["rows"])
        atomic_json(STATUS_PATH, status)
        print(
            "[FORMAL COMPLETE {}/{}] {} {} seed{}".format(
                status["completed"], status["total"], method, dataset, seed
            ),
            flush=True,
        )
    status["status"] = "COMPLETE"
    atomic_json(STATUS_PATH, status)
    print("status={}".format(STATUS_PATH), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run seed-42 MTT-only Stage-B finalists through TFE/ABY3 sequentially."""

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(
    os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parents[2])
)
LEGACY_PYTHON = Path(
    os.environ.get(
        "NEW_TFE_PYTHON",
        "/root/autodl-tmp/new_envs/tfe/bin/python",
    )
)
MODERN_PYTHON = Path(
    os.environ.get("NEW_MODERN_PYTHON", "/root/miniconda3/bin/python")
)


def is_complete(report_path):
    if not report_path.is_file():
        return False
    try:
        return json.loads(report_path.read_text(encoding="utf-8")).get(
            "status"
        ) == "COMPLETE"
    except (OSError, ValueError):
        return False


def run_one(dataset, seed, config):
    output_dir = (
        ROOT
        / "new_output/new_true_ours/stage_b/true_ours_finalist"
        / dataset
        / "seed{}".format(seed)
    )
    checkpoint = (
        ROOT
        / "new_checkpoints/new_true_ours/stage_b/true_ours_finalist"
        / dataset
        / "seed{}".format(seed)
        / "new_stage_b"
    )
    log_path = (
        ROOT
        / "new_log/new_true_ours/txt"
        / "new_stage_b_tfe_finalist_{}_seed{}.log".format(dataset, seed)
    )
    report = output_dir / (
        "new_secure_stage_b_report_{}_balanced_seed{}.json".format(
            dataset, seed
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not is_complete(report):
        cache = (
            ROOT
            / "new_cache/new_true_ours/official_mtt/stage_b"
            / dataset
            / "new_true_ours_stage_b_{}_seed{}.npz".format(dataset, seed)
        )
        command = [
            str(LEGACY_PYTHON),
            str(ROOT / "new_stage_b_secure_init.py"),
            "--feature_cache",
            str(cache),
            "--dataset",
            dataset,
            "--output_dir",
            str(output_dir),
            "--checkpoint_prefix",
            str(checkpoint),
            "--seed",
            str(seed),
            "--num_clients",
            "5",
            "--split",
            "balanced",
            "--batch_size",
            str(config["batch_size"]),
            "--local_epochs",
            str(config["local_epochs"]),
            "--local_lr",
            str(config["local_lr"]),
            "--weight_decay",
            str(config["weight_decay"]),
            "--client_weighting",
            str(config["client_weighting"]),
            "--fedavg_sync_interval",
            str(config.get("fedavg_sync_interval", 0)),
            "--distill_epochs",
            "0",
            "--lambda_kd",
            "0",
            "--reference_supervised_weight",
            "0",
            "--checkpoint_interval",
            "5",
            "--formal_candidate_release",
            "--input_lineage",
            "official_mtt_only",
        ]
        if config.get("shared_initialization", False):
            command.append("--shared_initialization")
        environment = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = environment.get(name, "32")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\nFINALIST_COMMAND_WITHOUT_SECRETS={}\n".format(
                    json.dumps(command)
                )
            )
            handle.flush()
            subprocess.check_call(
                command,
                cwd=str(ROOT),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
    candidate = output_dir / (
        "new_secure_stage_b_candidate_nokd_{}_balanced_seed{}.npz".format(
            dataset, seed
        )
    )
    evaluation = (
        ROOT
        / "new_log/new_true_ours"
        / "new_true_ours_stage_b_development_{}_seed{}.json".format(
            dataset, seed
        )
    )
    subprocess.check_call(
        [
            str(MODERN_PYTHON),
            str(
                ROOT
                / "new_scripts/new_true_ours/"
                "new_evaluate_stage_b_development.py"
            ),
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--candidate",
            str(candidate),
            "--output",
            str(evaluation),
        ],
        cwd=str(ROOT),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["cifar10", "cifar100", "tiny"],
        default=["cifar10", "cifar100", "tiny"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    payload = json.loads(
        (
            ROOT
            / "new_log/new_true_ours/new_true_ours_stage_b_fedavg_screen.json"
        ).read_text(encoding="utf-8")
    )
    configs = {
        (row["dataset"], int(row["seed"])): row["best_config"]
        for row in payload["results"]
    }
    rescue_path = (
        ROOT
        / "new_log/new_true_ours/new_true_ours_stage_b_rescue_screen.json"
    )
    if rescue_path.is_file():
        rescue = json.loads(rescue_path.read_text(encoding="utf-8"))
        for row in rescue.get("results", []):
            configs[(row["dataset"], int(row["seed"]))] = row[
                "best_config"
            ]
    for dataset in args.datasets:
        run_one(dataset, args.seed, configs[(dataset, args.seed)])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sequential resumable seed42 TFE validation: tiny -> C10 -> C100."""

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_PYTHON = Path("/root/autodl-tmp/new_envs/tfe/bin/python")
MODERN_PYTHON = Path("/root/miniconda3/bin/python")
SCRIPT_ROOT = ROOT / "new_scripts/new_stage_b_kd"
OUTPUT_ROOT = ROOT / "new_output/new_stage_b_kd"
CHECKPOINT_ROOT = ROOT / "new_checkpoints/new_stage_b_kd/tfe_validation"
LOG_ROOT = ROOT / "new_log/new_stage_b_kd/txt"


def complete(report_path):
    if not report_path.is_file():
        return False
    try:
        return json.loads(report_path.read_text(encoding="utf-8")).get(
            "status"
        ) == "COMPLETE"
    except (OSError, ValueError):
        return False


def run_one(dataset):
    input_path = (
        OUTPUT_ROOT
        / "tfe_inputs"
        / dataset
        / "new_stage_b_kd_tfe_input_{}_seed42.npz".format(dataset)
    )
    output_dir = OUTPUT_ROOT / "tfe_validation" / dataset / "seed42"
    report_path = output_dir / (
        "new_stage_b_kd_tfe_report_{}_seed42.json".format(dataset)
    )
    checkpoint = (
        CHECKPOINT_ROOT / dataset / "seed42" / "new_stage_b_kd"
    )
    log_path = LOG_ROOT / (
        "new_stage_b_kd_tfe_{}_seed42.log".format(dataset)
    )
    if not input_path.is_file():
        raise ValueError("missing frozen TFE input: {}".format(input_path))
    if complete(report_path):
        print("SKIP COMPLETE {}".format(dataset), flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(LEGACY_PYTHON),
        str(SCRIPT_ROOT / "new_stage_b_kd_tfe_worker.py"),
        "--input",
        str(input_path),
        "--output_dir",
        str(output_dir),
        "--checkpoint_prefix",
        str(checkpoint),
    ]
    environment = os.environ.copy()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\nTFE_COMMAND_WITHOUT_SECRETS={}\n".format(
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


def validate(required):
    subprocess.check_call(
        [
            str(MODERN_PYTHON),
            str(SCRIPT_ROOT / "new_validate_stage_b_kd_tfe.py"),
            "--require",
            required,
        ],
        cwd=str(ROOT),
    )


def main():
    run_one("tiny_synthetic")
    validate("tiny_synthetic")
    run_one("cifar10")
    validate("cifar10")
    run_one("cifar100")
    validate("all")
    print("COMPLETE seed42 TFE ladder tiny+C10+C100", flush=True)


if __name__ == "__main__":
    main()

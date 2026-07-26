#!/usr/bin/env python3
"""Create the complete serial TFE/ABY3 Stage-B formal task manifest."""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
FROZEN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_stage_b_frozen_config.json"
)
QUEUE = (
    ROOT
    / "new_configs/new_public_mtt_mainline"
    / "new_tfe_stage_b_formal.tsv"
)
PLAN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_stage_b_formal_plan.json"
)
TFE_PYTHON = os.environ.get(
    "NEW_TFE_PYTHON", str(ROOT.parent / "new_envs" / "tfe" / "bin" / "python")
)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resolve the complete formal secure Stage-B queue"
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Start the resolved queue in the persistent tmux TFE window",
    )
    return parser.parse_args()


def make_command(dataset, seed, split, cache, params, output_dir, checkpoint):
    values = {
        "python": TFE_PYTHON,
        "script": ROOT / "new_stage_b_secure_init.py",
        "cache": cache,
        "dataset": dataset,
        "output": output_dir,
        "checkpoint": checkpoint,
        "seed": seed,
        "split": split,
        "local_epochs": params["local_epochs"],
        "local_lr": params["local_lr"],
        "distill_epochs": params["distill_epochs"],
        "distill_lr": params["distill_lr"],
        "lambda_kd": params["lambda_kd"],
        "reference_supervised_weight": params[
            "reference_supervised_weight"
        ],
        "weight_temperature": params["weight_temperature"],
    }
    return (
        "{python} {script} --feature_cache {cache} --dataset {dataset} "
        "--output_dir {output} --checkpoint_prefix {checkpoint} "
        "--seed {seed} --num_clients 5 --split {split} --batch_size 256 "
        "--local_epochs {local_epochs} --local_lr {local_lr} "
        "--distill_epochs {distill_epochs} --distill_lr {distill_lr} "
        "--lambda_kd {lambda_kd} "
        "--reference_supervised_weight {reference_supervised_weight} "
        "--weight_temperature {weight_temperature} --init_scale 0.01 "
        "--precision_fractional 16 --matmul_chunk_size 64 "
        "--checkpoint_interval 1 --formal_candidate_release"
    ).format(**values)


def main():
    args = parse_args()
    frozen = load_json(FROZEN)
    if frozen.get("status") != "FROZEN":
        raise RuntimeError("Stage-B configuration has not passed its freeze gate")
    if not frozen.get("selection_only_tuning"):
        raise RuntimeError("Stage-B configuration was not selection-only tuned")
    if frozen.get("test_loaded_during_tuning") is not False:
        raise RuntimeError("Test leakage detected in Stage-B frozen config")

    rows = ["# task_id\tcheckpoint\tcommand"]
    tasks = []
    for dataset in ("cifar10", "cifar100", "tiny"):
        params = frozen["datasets"][dataset]["hyperparameters"]
        for seed in frozen["formal_seeds"]:
            # Frozen reports retain source-machine absolute paths as provenance.
            # Runtime queues must instead resolve the portable cache location.
            cache = (
                ROOT
                / "new_cache/new_public_mtt_mainline/new_stage_b_npz_round2"
                / dataset
                / "new_stage_b_features_{}_seed{}.npz".format(dataset, seed)
            )
            expected_sha = frozen["feature_caches"][dataset][str(seed)].get("sha256")
            if not cache.exists():
                raise FileNotFoundError(
                    "Portable Stage-B feature cache missing: {}".format(cache)
                )
            if expected_sha:
                import hashlib

                digest = hashlib.sha256()
                with cache.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_sha:
                    raise RuntimeError("Stage-B feature cache SHA256 mismatch: {}".format(cache))
            for split in frozen["splits"]:
                task_id = "public_mainline_stage_b_formal_{}_seed{}_{}".format(
                    dataset, seed, split
                )
                output_dir = (
                    ROOT
                    / "new_init_model/new_public_mtt_mainline/stage_b"
                    / dataset
                    / "seed{}".format(seed)
                    / split
                )
                checkpoint = (
                    ROOT
                    / "new_checkpoint/new_public_mtt_mainline/stage_b"
                    / dataset
                    / "seed{}".format(seed)
                    / split
                    / "new_stage_b"
                )
                report = output_dir / (
                    "new_secure_stage_b_report_{}_{}_seed{}.json".format(
                        dataset, split, seed
                    )
                )
                command = make_command(
                    dataset,
                    seed,
                    split,
                    cache,
                    params,
                    output_dir,
                    checkpoint,
                )
                rows.append(
                    "{}\t{}\t{}".format(task_id, checkpoint, command)
                )
                tasks.append(
                    {
                        "phase": "secure_stage_b_formal",
                        "dataset": dataset,
                        "seed": seed,
                        "split": split,
                        "methods_released_at_formal_boundary": [
                            "NoKD",
                            "KD-Avg",
                            "KD-ReferenceWeighted",
                        ],
                        "assignment": "TFE_CPU_core0_serial",
                        "expected_output": str(report),
                        "checkpoint": str(checkpoint),
                        "configuration": params,
                    }
                )

    report_output = (
        ROOT
        / "new_log/new_public_mtt_mainline/json"
        / "new_stage_b_initialization_report.json"
    )
    report_task = {
        "phase": "secure_stage_b_frozen_test_report",
        "assignment": "TFE_CPU_core0_serial_after_training",
        "expected_output": str(report_output),
        "test_role": "readonly_final_reporting_after_selection_freeze",
    }
    rows.append(
        (
            "public_mainline_stage_b_formal_report\t{output}\t"
            "{python} {script}"
        ).format(
            output=report_output,
            python=os.environ.get(
                "NEW_MODERN_PYTHON",
                str(ROOT.parent / "new_envs" / "distillation" / "bin" / "python"),
            ),
            script=ROOT / "new_report_stage_b_initialization.py",
        )
    )
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    payload = {
        "schema": "new_public_mtt_stage_b_formal_plan_v1",
        "status": "READY",
        "backend": "tfe_mpc",
        "protocol": "ABY3",
        "feature_dim": 2048,
        "cpu_threads": 1,
        "num_workers": 0,
        "gpu_assignment": "none",
        "serial": True,
        "checkpoint_each_epoch": True,
        "test_used_for_tuning": False,
        "formal_task_count": len(tasks),
        "post_training_report_task": report_task,
        "queue": str(QUEUE),
        "tasks": tasks,
    }
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("Formal Secure Stage-B resolved configuration")
    print("backend/protocol : tfe_mpc / ABY3")
    print("datasets         : cifar10, cifar100, tiny")
    print("seeds            : 42, 43, 44")
    print("splits           : iid, mild, severe")
    print("formal tasks     : {}".format(len(tasks)))
    print("report tasks     : 1 (runs only after all 27 formal tasks)")
    print("CPU/GPU          : core0, threads=1 / GPU disabled")
    print("queue            : {}".format(QUEUE))
    print("plan             : {}".format(PLAN))
    print("expected outputs :")
    for task in tasks:
        print(
            "  {dataset} seed={seed} split={split} -> {expected_output}".format(
                **task
            )
        )
    print("=" * 100)

    if args.activate:
        tmux = os.environ.get(
            "NEW_TMUX", "tmux"
        )
        session = "new_secure_feature_single_gpu_formal"
        command = (
            "cd {root} && bash new_scripts/new_run_task_queue.sh tfe {queue}"
        ).format(root=ROOT, queue=QUEUE)
        result = os.system(
            "{} send-keys -t {}:tfe '{}' Enter".format(
                tmux, session, command
            )
        )
        if result != 0:
            raise RuntimeError("Failed to activate tmux TFE queue")
        print("activated tmux session {}:tfe".format(session))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

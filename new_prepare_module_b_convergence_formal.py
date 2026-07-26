#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the formal Module B convergence queue after the Stage C gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from new_public_mtt_mainline_common import ROOT, load_config
from new_stage_c_privacy_calibration import calibrate_noise_multiplier


TFE_PYTHON = os.environ.get(
    "NEW_TFE_PYTHON", str(ROOT.parent / "new_envs" / "tfe" / "bin" / "python")
)
QUEUE = (
    ROOT
    / "new_configs/new_public_mtt_mainline"
    / "new_tfe_module_b_convergence_formal.tsv"
)
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
PLAN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_module_b_convergence_formal_plan.json"
)
VALIDATION = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_stage_c_validation_round3_summary.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_random_initializer(path, feature_dim, num_classes, seed, scale):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(int(seed) + 73001)
    weight = rng.normal(
        0.0, float(scale), size=(feature_dim, num_classes)
    ).astype(np.float32)
    bias = np.zeros((num_classes,), dtype=np.float32)
    temporary = Path(str(path) + ".new_tmp_{}".format(os.getpid()))
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            weight=weight,
            bias=bias,
            initializer_privacy_status=np.asarray(["data_independent"]),
            seed=np.asarray([seed], dtype=np.int64),
            scale=np.asarray([scale], dtype=np.float32),
        )
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    config = load_config()
    formal = config["module_b_convergence_formal"]
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if validation["status"] != "PASS" or validation["test_loaded"]:
        raise ValueError("Stage C validation gate is not a test-free PASS")
    if config["dp_mtt_status"] != "EXCLUDED_BY_USER":
        raise ValueError("DP-MTT must remain excluded")
    if formal["test_loaded"]:
        raise ValueError("formal convergence preparation cannot load test")

    rows = ["# task_id\tcheckpoint\tcommand"]
    tasks = []
    random_assets = []
    for dataset in config["datasets"]:
        dataset_cfg = formal["datasets"][dataset]
        steps = int(dataset_cfg["steps"])
        batch_size = int(dataset_cfg["batch_size"])
        num_samples = {"cifar10": 400, "cifar100": 4000, "tiny": 8000}[dataset]
        noise, achieved = calibrate_noise_multiplier(
            target_epsilon=formal["optimizer_target_epsilon"],
            steps=steps,
            sample_rate=float(batch_size) / num_samples,
            delta=formal["delta"],
        )
        for seed in formal["seeds"]:
            cache = (
                ROOT
                / "new_cache/new_public_mtt_mainline/new_stage_b_npz_round2"
                / dataset
                / "new_stage_b_features_{}_seed{}.npz".format(dataset, seed)
            )
            selected = (
                ROOT
                / "new_init_model/new_public_mtt_mainline/stage_b"
                / dataset
                / "seed{}".format(seed)
                / formal["selected_split"]
                / "new_secure_stage_b_selected_{}_{}_seed{}.npz".format(
                    dataset, formal["selected_split"], seed
                )
            )
            if not cache.is_file() or not selected.is_file():
                raise FileNotFoundError("missing migrated asset for {} {}".format(
                    dataset, seed
                ))
            with np.load(str(selected), allow_pickle=False) as model:
                feature_dim, num_classes = model["weight"].shape
            random_path = (
                ROOT
                / "new_init_model/new_public_mtt_mainline/module_b_random"
                / dataset
                / "seed{}".format(seed)
                / "new_random_initializer_{}_seed{}.npz".format(dataset, seed)
            )
            atomic_random_initializer(
                random_path,
                feature_dim,
                num_classes,
                seed,
                formal["random_init_scale"],
            )
            random_assets.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "path": str(random_path),
                    "sha256": sha256_file(random_path),
                    "initializer_privacy_status": "data_independent",
                }
            )
            method_specs = [
                ("random_initialization", random_path, "data_independent"),
                (
                    "secure_consensus_initialization",
                    selected,
                    "data_dependent_non_dp",
                ),
            ]
            for method, initializer, initializer_status in method_specs:
                task_id = "public_mainline_module_b_convergence_{}_{}_seed{}".format(
                    dataset, method, seed
                )
                output_dir = (
                    ROOT
                    / "new_output/new_public_mtt_mainline/module_b_convergence_formal"
                    / dataset
                    / method
                    / "seed{}".format(seed)
                )
                checkpoint = (
                    ROOT
                    / "new_checkpoint/new_public_mtt_mainline/module_b_convergence_formal"
                    / dataset
                    / method
                    / "seed{}/new_stage_c".format(seed)
                )
                command = (
                    "{python} {script} --feature_cache {cache} "
                    "--initializer {initializer} --dataset {dataset} "
                    "--mode fixed_dp --output_dir {output} "
                    "--checkpoint_prefix {checkpoint} --seed {seed} "
                    "--batch_size {batch} --steps {steps} "
                    "--learning_rate {lr} --noise_multiplier {noise} "
                    "--count_noise_multiplier 8.0 --clip_c 1.0 "
                    "--warmup_steps 5 --update_interval 5 "
                    "--checkpoint_interval {checkpoint_interval} "
                    "--initializer_privacy_status {initializer_status}"
                ).format(
                    python=TFE_PYTHON,
                    script=ROOT / "new_stage_c_secure_dpsgd.py",
                    cache=cache,
                    initializer=initializer,
                    dataset=dataset,
                    output=output_dir,
                    checkpoint=checkpoint,
                    seed=seed,
                    batch=batch_size,
                    steps=steps,
                    lr=formal["learning_rate"],
                    noise=noise,
                    checkpoint_interval=formal["checkpoint_interval"],
                    initializer_status=initializer_status,
                )
                rows.append("{}\t{}\t{}".format(task_id, checkpoint, command))
                tasks.append(
                    {
                        "task_id": task_id,
                        "phase": "module_b_convergence_formal",
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "steps": steps,
                        "batch_size": batch_size,
                        "processed_examples": steps * batch_size,
                        "mode": "fixed_dp",
                        "initializer": str(initializer),
                        "initializer_sha256": sha256_file(initializer),
                        "initializer_privacy_status": initializer_status,
                        "optimizer_target_epsilon": formal[
                            "optimizer_target_epsilon"
                        ],
                        "accounted_epsilon": achieved,
                        "gradient_noise_multiplier": noise,
                        "released_model_record_dp_valid": (
                            initializer_status == "data_independent"
                        ),
                        "test_loaded": False,
                        "resource": "tfe",
                        "expected_output_dir": str(output_dir),
                    }
                )
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(
        json.dumps(
            {
                "schema": "new_module_b_convergence_formal_plan_v1",
                "status": "READY",
                "formal_result": True,
                "test_loaded": False,
                "dp_mtt_status": "EXCLUDED_BY_USER",
                "validation_gate": str(VALIDATION),
                "validation_gate_status": "PASS",
                "privacy_interpretation": {
                    "random_initialization": "released_model_record_dp",
                    "secure_consensus_initialization": (
                        "conditional_gradient_mechanism_only_not_released_model_dp"
                    ),
                },
                "resource_policy": {
                    "resource": "tfe",
                    "cpu_affinity": "unrestricted",
                    "max_concurrent_tasks": 1,
                },
                "random_initializer_assets": random_assets,
                "queue": str(QUEUE),
                "tasks": tasks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.activate:
        ACTIVE.write_text("tfe\t{}\n".format(QUEUE), encoding="utf-8")
    print(
        "phase=module_b_convergence_formal tasks={} test_loaded=False "
        "dp_mtt_status=EXCLUDED_BY_USER".format(len(tasks))
    )
    print("queue={}".format(QUEUE))
    print("plan={}".format(PLAN))
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

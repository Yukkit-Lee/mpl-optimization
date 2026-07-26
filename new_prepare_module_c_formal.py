#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the record-DP-valid formal Module C privacy/utility grid."""

import argparse
import hashlib
import json
import os

from new_public_mtt_mainline_common import ROOT, load_config
from new_stage_c_privacy_calibration import adaptive_count_steps
from new_stage_c_privacy_calibration import calibrate_noise_multiplier


TFE_PYTHON = os.environ.get(
    "NEW_TFE_PYTHON", str(ROOT.parent / "new_envs" / "tfe" / "bin" / "python")
)
QUEUE = (
    ROOT / "new_configs/new_public_mtt_mainline/new_tfe_module_c_formal.tsv"
)
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
PLAN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_module_c_formal_plan.json"
)
MODULE_B_REPORT = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_module_b_convergence_formal_report.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epsilon_tag(value):
    return "eps{}".format(str(float(value)).replace(".", "p"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    config = load_config()
    formal = config["module_c_formal"]
    if config["dp_mtt_status"] != "EXCLUDED_BY_USER":
        raise ValueError("DP-MTT must remain excluded")
    if formal["initializer_privacy_status"] != "data_independent":
        raise ValueError("Module C released-model DP requires data-independent init")
    if formal["test_loaded"]:
        raise ValueError("Module C formal preparation cannot load test")
    module_b = json.loads(MODULE_B_REPORT.read_text(encoding="utf-8"))
    if (
        module_b["status"] != "COMPLETE"
        or module_b["runs_completed"] != module_b["runs_expected"]
        or module_b["test_loaded"]
    ):
        raise ValueError("Module B convergence gate is incomplete")

    rows = ["# task_id\tcheckpoint\tcommand"]
    tasks = []
    calibrations = {}
    for dataset in config["datasets"]:
        dataset_cfg = formal["datasets"][dataset]
        steps = int(dataset_cfg["steps"])
        batch_size = int(dataset_cfg["batch_size"])
        num_samples = {"cifar10": 400, "cifar100": 4000, "tiny": 8000}[dataset]
        sample_rate = float(batch_size) / num_samples
        for mode in formal["modes"]:
            count_steps = (
                adaptive_count_steps(
                    steps, formal["warmup_steps"], formal["update_interval"]
                )
                if mode == "adaptive_dp"
                else 0
            )
            for target_epsilon in formal["target_epsilons"]:
                noise, achieved = calibrate_noise_multiplier(
                    target_epsilon=target_epsilon,
                    steps=steps,
                    sample_rate=sample_rate,
                    delta=formal["delta"],
                    count_noise=(
                        formal["count_noise_multiplier"] if count_steps else None
                    ),
                    count_steps=count_steps,
                )
                key = "{}:{}:{}".format(dataset, mode, target_epsilon)
                calibrations[key] = {
                    "dataset": dataset,
                    "mode": mode,
                    "target_epsilon": target_epsilon,
                    "achieved_epsilon": achieved,
                    "gradient_noise_multiplier": noise,
                    "count_noise_multiplier": (
                        formal["count_noise_multiplier"] if count_steps else None
                    ),
                    "count_steps": count_steps,
                    "steps": steps,
                    "sample_rate": sample_rate,
                }
                for seed in formal["seeds"]:
                    cache = (
                        ROOT
                        / "new_cache/new_public_mtt_mainline/new_stage_b_npz_round2"
                        / dataset
                        / "new_stage_b_features_{}_seed{}.npz".format(
                            dataset, seed
                        )
                    )
                    initializer = (
                        ROOT
                        / "new_init_model/new_public_mtt_mainline/module_b_random"
                        / dataset
                        / "seed{}".format(seed)
                        / "new_random_initializer_{}_seed{}.npz".format(
                            dataset, seed
                        )
                    )
                    if not cache.is_file() or not initializer.is_file():
                        raise FileNotFoundError(
                            "missing formal asset for {} {}".format(dataset, seed)
                        )
                    tag = epsilon_tag(target_epsilon)
                    task_id = "public_mainline_module_c_{}_{}_{}_seed{}".format(
                        dataset, mode, tag, seed
                    )
                    output_dir = (
                        ROOT
                        / "new_output/new_public_mtt_mainline/module_c_formal"
                        / dataset
                        / mode
                        / tag
                        / "seed{}".format(seed)
                    )
                    checkpoint = (
                        ROOT
                        / "new_checkpoint/new_public_mtt_mainline/module_c_formal"
                        / dataset
                        / mode
                        / tag
                        / "seed{}/new_stage_c".format(seed)
                    )
                    command = (
                        "{python} {script} --feature_cache {cache} "
                        "--initializer {initializer} --dataset {dataset} "
                        "--mode {mode} --output_dir {output} "
                        "--checkpoint_prefix {checkpoint} --seed {seed} "
                        "--batch_size {batch} --steps {steps} "
                        "--learning_rate {lr} --noise_multiplier {noise} "
                        "--count_noise_multiplier {count_noise} "
                        "--clip_c {clip_c} --warmup_steps {warmup} "
                        "--update_interval {interval} "
                        "--checkpoint_interval {checkpoint_interval} "
                        "--initializer_privacy_status data_independent"
                    ).format(
                        python=TFE_PYTHON,
                        script=ROOT / "new_stage_c_secure_dpsgd.py",
                        cache=cache,
                        initializer=initializer,
                        dataset=dataset,
                        mode=mode,
                        output=output_dir,
                        checkpoint=checkpoint,
                        seed=seed,
                        batch=batch_size,
                        steps=steps,
                        lr=formal["learning_rate"],
                        noise=noise,
                        count_noise=formal["count_noise_multiplier"],
                        clip_c=formal["clip_c"],
                        warmup=formal["warmup_steps"],
                        interval=formal["update_interval"],
                        checkpoint_interval=formal["checkpoint_interval"],
                    )
                    rows.append("{}\t{}\t{}".format(task_id, checkpoint, command))
                    tasks.append(
                        {
                            "task_id": task_id,
                            "phase": "module_c_formal",
                            "dataset": dataset,
                            "mode": mode,
                            "target_epsilon": target_epsilon,
                            "achieved_epsilon": achieved,
                            "gradient_noise_multiplier": noise,
                            "count_steps": count_steps,
                            "seed": seed,
                            "steps": steps,
                            "batch_size": batch_size,
                            "initializer": str(initializer),
                            "initializer_sha256": sha256_file(initializer),
                            "initializer_privacy_status": "data_independent",
                            "released_model_record_dp_valid": True,
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
                "schema": "new_module_c_formal_plan_v1",
                "status": "READY",
                "formal_result": True,
                "test_loaded": False,
                "dp_mtt_status": "EXCLUDED_BY_USER",
                "module_b_gate": str(MODULE_B_REPORT),
                "module_b_gate_status": "COMPLETE",
                "privacy_scope": "stage_c_distilled_feature_records",
                "released_model_record_dp_valid": True,
                "initializer_privacy_status": "data_independent",
                "accountant": "Opacus RDPAccountant",
                "resource_policy": {
                    "resource": "tfe",
                    "cpu_affinity": "unrestricted",
                    "max_concurrent_tasks": 1,
                },
                "calibrations": calibrations,
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
        "phase=module_c_formal tasks={} calibrations={} test_loaded=False "
        "released_model_record_dp_valid=True".format(
            len(tasks), len(calibrations)
        )
    )
    print("queue={}".format(QUEUE))
    print("plan={}".format(PLAN))
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

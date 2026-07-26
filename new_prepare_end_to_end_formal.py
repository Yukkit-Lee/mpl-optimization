#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the two-curve formal End-to-End Stage C queue."""

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
    ROOT / "new_configs/new_public_mtt_mainline/new_tfe_end_to_end_formal.tsv"
)
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
PLAN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_end_to_end_formal_plan.json"
)
FEATURE_MANIFEST = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_end_to_end_feature_manifest.json"
)
MODULE_C_REPORT = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_module_c_formal_report.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    config = load_config()
    formal = config["end_to_end_formal"]
    features = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    module_c = json.loads(MODULE_C_REPORT.read_text(encoding="utf-8"))
    if (
        module_c["status"] != "COMPLETE"
        or module_c["runs_completed"] != module_c["runs_expected"]
        or module_c["test_loaded"]
    ):
        raise ValueError("Module C formal gate is incomplete")
    if features["status"] != "COMPLETE" or features["test_loaded"]:
        raise ValueError("End-to-End feature gate is incomplete")
    if config["dp_mtt_status"] != "EXCLUDED_BY_USER":
        raise ValueError("DP-MTT must remain excluded")
    by_key = {
        (row["dataset"], int(row["seed"])): row
        for row in features["rows"]
    }

    rows = ["# task_id\tcheckpoint\tcommand"]
    tasks = []
    calibrations = {}
    for dataset in config["datasets"]:
        dataset_cfg = formal["datasets"][dataset]
        steps = int(dataset_cfg["steps"])
        batch_size = int(dataset_cfg["batch_size"])
        num_samples = {"cifar10": 400, "cifar100": 4000, "tiny": 8000}[dataset]
        sample_rate = float(batch_size) / num_samples
        fixed_noise, fixed_epsilon = calibrate_noise_multiplier(
            target_epsilon=formal["target_epsilon"],
            steps=steps,
            sample_rate=sample_rate,
            delta=formal["delta"],
        )
        count_steps = adaptive_count_steps(
            steps, formal["warmup_steps"], formal["update_interval"]
        )
        adaptive_noise, adaptive_epsilon = calibrate_noise_multiplier(
            target_epsilon=formal["target_epsilon"],
            steps=steps,
            sample_rate=sample_rate,
            delta=formal["delta"],
            count_noise=formal["count_noise_multiplier"],
            count_steps=count_steps,
        )
        calibrations[dataset] = {
            "target_epsilon": formal["target_epsilon"],
            "baseline_fixed": {
                "gradient_noise_multiplier": fixed_noise,
                "achieved_epsilon": fixed_epsilon,
                "count_steps": 0,
            },
            "ours_adaptive": {
                "gradient_noise_multiplier": adaptive_noise,
                "achieved_epsilon": adaptive_epsilon,
                "count_noise_multiplier": formal["count_noise_multiplier"],
                "count_steps": count_steps,
            },
        }
        for seed in formal["seeds"]:
            asset = by_key[(dataset, seed)]
            baseline_initializer = (
                ROOT
                / "new_init_model/new_public_mtt_mainline/end_to_end_baseline"
                / dataset
                / "seed{}".format(seed)
                / "new_secure_stage_b_candidate_nokd_{}_iid_seed{}.npz".format(
                    dataset, seed
                )
            )
            ours_initializer = (
                ROOT
                / "new_init_model/new_public_mtt_mainline/stage_b"
                / dataset
                / "seed{}".format(seed)
                / "iid"
                / "new_secure_stage_b_selected_{}_iid_seed{}.npz".format(
                    dataset, seed
                )
            )
            if not baseline_initializer.is_file():
                raise FileNotFoundError(
                    "missing End-to-End baseline initializer {}".format(
                        baseline_initializer
                    )
                )
            if not ours_initializer.is_file():
                raise FileNotFoundError(
                    "missing frozen selected initializer {}".format(
                        ours_initializer
                    )
                )
            specs = [
                {
                    "method": "baseline_random_secure_average_fixed",
                    "feature_cache": asset["random_subset_cache"],
                    "feature_cache_sha256": asset["random_subset_sha256"],
                    "initializer": baseline_initializer,
                    "mode": "fixed_dp",
                    "noise": fixed_noise,
                    "achieved": fixed_epsilon,
                    "count_steps": 0,
                },
                {
                    "method": "ours_official_secure_consensus_adaptive",
                    "feature_cache": asset["official_mtt_cache"],
                    "feature_cache_sha256": asset["official_mtt_sha256"],
                    "initializer": ours_initializer,
                    "mode": "adaptive_dp",
                    "noise": adaptive_noise,
                    "achieved": adaptive_epsilon,
                    "count_steps": count_steps,
                },
            ]
            for spec in specs:
                task_id = "public_mainline_end_to_end_{}_{}_seed{}".format(
                    dataset, spec["method"], seed
                )
                output_dir = (
                    ROOT
                    / "new_output/new_public_mtt_mainline/end_to_end_formal"
                    / dataset
                    / spec["method"]
                    / "seed{}".format(seed)
                )
                checkpoint = (
                    ROOT
                    / "new_checkpoint/new_public_mtt_mainline/end_to_end_formal"
                    / dataset
                    / spec["method"]
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
                    "--initializer_privacy_status data_dependent_non_dp"
                ).format(
                    python=TFE_PYTHON,
                    script=ROOT / "new_stage_c_secure_dpsgd.py",
                    cache=spec["feature_cache"],
                    initializer=spec["initializer"],
                    dataset=dataset,
                    mode=spec["mode"],
                    output=output_dir,
                    checkpoint=checkpoint,
                    seed=seed,
                    batch=batch_size,
                    steps=steps,
                    lr=formal["learning_rate"],
                    noise=spec["noise"],
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
                        "phase": "end_to_end_formal",
                        "dataset": dataset,
                        "seed": seed,
                        "split": "iid",
                        "method": spec["method"],
                        "feature_cache": spec["feature_cache"],
                        "feature_cache_sha256": spec["feature_cache_sha256"],
                        "shared_reference": True,
                        "initializer": str(spec["initializer"]),
                        "initializer_sha256": sha256_file(spec["initializer"]),
                        "initializer_privacy_status": "data_dependent_non_dp",
                        "mode": spec["mode"],
                        "target_epsilon": formal["target_epsilon"],
                        "accounted_epsilon": spec["achieved"],
                        "count_steps": spec["count_steps"],
                        "released_model_record_dp_valid": False,
                        "epsilon_interpretation": (
                            "conditional_gradient_mechanism_only_not_released_model_dp"
                        ),
                        "steps": steps,
                        "batch_size": batch_size,
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
                "schema": "new_end_to_end_formal_plan_v1",
                "status": "READY",
                "formal_result": True,
                "test_loaded": False,
                "dp_mtt_status": "EXCLUDED_BY_USER",
                "curves": [
                    "Random Subset + NoKD Secure Average + Fixed",
                    "Official MTT + Frozen Secure Consensus Selected + Adaptive",
                ],
                "portable_artifact_boundary": (
                    "selected is not mislabeled as standalone KD-ReferenceWeighted"
                ),
                "shared_reference_protocol": True,
                "privacy_interpretation": (
                    "conditional_gradient_mechanism_only_not_released_model_dp"
                ),
                "released_model_record_dp_valid": False,
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
    for task in tasks:
        print(
            "phase={phase} dataset={dataset} seed={seed} split={split} "
            "method={method} privacy={epsilon_interpretation} resource={resource} "
            "expected={expected_output_dir}".format(**task)
        )
    print("queue={}".format(QUEUE))
    print("plan={}".format(PLAN))
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare the test-free, persistent Stage C validation queue."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from new_public_mtt_mainline_common import ROOT, load_config


TFE_PYTHON = os.environ.get(
    "NEW_TFE_PYTHON", str(ROOT.parent / "new_envs" / "tfe" / "bin" / "python")
)
QUEUE = (
    ROOT
    / "new_configs/new_public_mtt_mainline/new_tfe_stage_c_validation.tsv"
)
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
PLAN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_stage_c_validation_plan.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_cache(path):
    with np.load(str(path), allow_pickle=False) as cache:
        required = {
            "x_encoder_train",
            "y_encoder_train",
            "x_reference",
            "y_reference",
            "feature_dim",
            "test_loaded",
        }
        missing = sorted(required.difference(cache.files))
        if missing:
            raise ValueError("{} missing {}".format(path, missing))
        if bool(np.asarray(cache["test_loaded"]).reshape(-1)[0]):
            raise ValueError("{} records test_loaded=True".format(path))
        feature_dim = int(np.asarray(cache["feature_dim"]).reshape(-1)[0])
        if feature_dim != 2048:
            raise ValueError("{} feature_dim={}".format(path, feature_dim))
        return {
            "train_samples": int(cache["x_encoder_train"].shape[0]),
            "reference_samples": int(cache["x_reference"].shape[0]),
            "feature_dim": feature_dim,
            "test_loaded": False,
        }


def inspect_initializer(path, feature_dim):
    with np.load(str(path), allow_pickle=False) as model:
        if "weight" not in model.files or "bias" not in model.files:
            raise ValueError("{} is not a portable initializer".format(path))
        weight_shape = tuple(model["weight"].shape)
        bias_shape = tuple(model["bias"].shape)
    if weight_shape[0] != feature_dim or weight_shape[1] != bias_shape[0]:
        raise ValueError("{} has incompatible shapes".format(path))
    return {"weight_shape": weight_shape, "bias_shape": bias_shape}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activate",
        action="store_true",
        help="activate the validated TFE queue using the resource-aware pointer",
    )
    args = parser.parse_args()
    config = load_config()
    if config["dp_mtt_status"] != "EXCLUDED_BY_USER":
        raise ValueError("DP-MTT must remain excluded")
    pilot = config["module_c_validation"]
    if pilot["status"] != "PILOT_ONLY_NOT_FORMAL" or pilot["test_loaded"]:
        raise ValueError("invalid Stage C validation boundary")

    rows = ["# task_id\tcheckpoint\tcommand"]
    resolved = []
    for dataset in config["datasets"]:
        dataset_cfg = pilot["datasets"][dataset]
        steps = int(dataset_cfg["steps"])
        batch_size = int(dataset_cfg["batch_size"])
        cache = (
            ROOT
            / "new_cache/new_public_mtt_mainline/new_stage_b_npz_round2"
            / dataset
            / "new_stage_b_features_{}_seed42.npz".format(dataset)
        )
        initializer = (
            ROOT
            / "new_init_model/new_public_mtt_mainline/stage_b"
            / dataset
            / "seed42"
            / pilot["split"]
            / "new_secure_stage_b_selected_{}_{}_seed42.npz".format(
                dataset, pilot["split"]
            )
        )
        if not cache.is_file() or not initializer.is_file():
            raise FileNotFoundError("missing migrated Stage B asset for {}".format(dataset))
        cache_info = inspect_cache(cache)
        initializer_info = inspect_initializer(initializer, cache_info["feature_dim"])
        for mode in pilot["modes"]:
            task_id = "public_mainline_stage_c_validation_{}_{}_seed42".format(
                dataset, mode
            )
            output_dir = (
                ROOT
                / "new_output/new_public_mtt_mainline/stage_c_validation"
                / dataset
                / mode
                / "seed42"
            )
            checkpoint = (
                ROOT
                / "new_checkpoint/new_public_mtt_mainline/stage_c_validation"
                / dataset
                / mode
                / "seed42/new_stage_c"
            )
            command = (
                "{python} {script} --feature_cache {cache} "
                "--initializer {initializer} --dataset {dataset} --mode {mode} "
                "--output_dir {output_dir} --checkpoint_prefix {checkpoint} "
                "--seed 42 --batch_size {batch_size} --steps {steps} "
                "--learning_rate {learning_rate} "
                "--noise_multiplier {noise_multiplier} "
                "--count_noise_multiplier {count_noise_multiplier} "
                "--clip_c {clip_c} --warmup_steps {warmup_steps} "
                "--update_interval {update_interval} "
                "--checkpoint_interval {checkpoint_interval}"
            ).format(
                python=TFE_PYTHON,
                script=ROOT / "new_stage_c_secure_dpsgd.py",
                cache=cache,
                initializer=initializer,
                dataset=dataset,
                mode=mode,
                output_dir=output_dir,
                checkpoint=checkpoint,
                batch_size=batch_size,
                steps=steps,
                learning_rate=pilot["learning_rate"],
                noise_multiplier=pilot["noise_multiplier"],
                count_noise_multiplier=pilot["count_noise_multiplier"],
                clip_c=pilot["clip_c"],
                warmup_steps=pilot["warmup_steps"],
                update_interval=pilot["update_interval"],
                checkpoint_interval=pilot["checkpoint_interval"],
            )
            rows.append("{}\t{}\t{}".format(task_id, checkpoint, command))
            resolved.append(
                {
                    "task_id": task_id,
                    "phase": "module_c_validation",
                    "dataset": dataset,
                    "seed": 42,
                    "split": pilot["split"],
                    "method": mode,
                    "privacy_scope": "stage_c_distilled_feature_records",
                    "steps": steps,
                    "batch_size": batch_size,
                    "processed_examples": steps * batch_size,
                    "epochs_equivalent": steps * batch_size
                    / float(cache_info["train_samples"]),
                    "resource": "tfe",
                    "feature_cache": str(cache),
                    "feature_cache_sha256": sha256_file(cache),
                    "initializer": str(initializer),
                    "initializer_sha256": sha256_file(initializer),
                    "cache": cache_info,
                    "initializer_shapes": initializer_info,
                    "expected_output_dir": str(output_dir),
                    "checkpoint_prefix": str(checkpoint),
                }
            )

    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(
        json.dumps(
            {
                "schema": "new_public_mtt_stage_c_validation_plan_v1",
                "status": "READY_FOR_VALIDATION",
                "formal_result": False,
                "test_loaded": False,
                "dp_mtt_status": "EXCLUDED_BY_USER",
                "resource_policy": {
                    "resource": "tfe",
                    "cpu_thread_policy": "benchmark_max_throughput",
                    "cpu_affinity": "unrestricted",
                    "visible_gpu": None,
                },
                "queue": str(QUEUE),
                "tasks": resolved,
                "next_gate": (
                    "review reference-only convergence and freeze dataset-specific "
                    "formal budgets before formal Module C"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.activate:
        ACTIVE.write_text("tfe\t{}\n".format(QUEUE), encoding="utf-8")
    for task in resolved:
        print(
            "phase={phase} dataset={dataset} seed={seed} split={split} "
            "method={method} privacy={privacy_scope} resource={resource} "
            "steps={steps} expected={expected_output_dir}".format(**task)
        )
    print("queue={}".format(QUEUE))
    print("plan={}".format(PLAN))
    print("dp_mtt_status=EXCLUDED_BY_USER test_loaded=False")
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

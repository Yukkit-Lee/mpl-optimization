#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare small/medium/largest-feasible TFE system experiments."""

import argparse
import json
import os

from new_public_mtt_mainline_common import ROOT, load_config


MODERN_PYTHON = os.environ.get(
    "NEW_MODERN_PYTHON",
    str(ROOT.parent / "new_envs" / "distillation" / "bin" / "python"),
)
TFE_PYTHON = os.environ.get(
    "NEW_TFE_PYTHON", str(ROOT.parent / "new_envs" / "tfe" / "bin" / "python")
)
QUEUE = (
    ROOT / "new_configs/new_public_mtt_mainline/new_tfe_system_formal.tsv"
)
ACTIVE = ROOT / "new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
PLAN = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_tfe_system_formal_plan.json"
)
END_TO_END_REPORT = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_end_to_end_formal_report.json"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    config = load_config()
    formal = config["tfe_system_formal"]
    if args.activate:
        end_to_end = json.loads(END_TO_END_REPORT.read_text(encoding="utf-8"))
        if (
            end_to_end["status"] != "COMPLETE"
            or end_to_end["runs_completed"] != end_to_end["runs_expected"]
        ):
            raise ValueError("End-to-End formal gate is incomplete")
    if formal["test_loaded"]:
        raise ValueError("TFE system experiments cannot load test")
    if config["dp_mtt_status"] != "EXCLUDED_BY_USER":
        raise ValueError("DP-MTT must remain excluded")
    class_counts = {"cifar10": 10, "cifar100": 100, "tiny": 200}
    sample_counts = {"cifar10": 400, "cifar100": 4000, "tiny": 8000}
    rows = ["# task_id\tcheckpoint\tcommand"]
    tasks = []
    for dataset in config["datasets"]:
        seed = formal["seed"]
        feature_cache = (
            ROOT
            / "new_cache/new_public_mtt_mainline/new_stage_b_npz_round2"
            / dataset
            / "new_stage_b_features_{}_seed{}.npz".format(dataset, seed)
        )
        initializer = (
            ROOT
            / "new_init_model/new_public_mtt_mainline/module_b_random"
            / dataset
            / "seed{}".format(seed)
            / "new_random_initializer_{}_seed{}.npz".format(dataset, seed)
        )
        if not feature_cache.is_file() or not initializer.is_file():
            raise FileNotFoundError("missing TFE system asset for {}".format(dataset))
        for scale, scale_cfg in formal["scales"].items():
            batch_size = int(scale_cfg["batch_size"])
            iterations = int(formal["datasets"][dataset]["iterations"])
            task_id = "public_mainline_tfe_system_{}_{}".format(dataset, scale)
            output_dir = (
                ROOT
                / "new_output/new_public_mtt_mainline/tfe_system_formal"
                / dataset
                / scale
            )
            checkpoint = (
                ROOT
                / "new_checkpoint/new_public_mtt_mainline/tfe_system_formal"
                / dataset
                / scale
                / "new_stage_c"
            )
            inner_report = (
                output_dir
                / "new_secure_stage_c_report_{}_fixed_dp_seed{}.json".format(
                    dataset, seed
                )
            )
            system_report = (
                output_dir
                / "new_tfe_system_report_{}_{}.json".format(dataset, scale)
            )
            inner = (
                "{tfe_python} {stage_c} --feature_cache {cache} "
                "--initializer {initializer} --dataset {dataset} "
                "--mode fixed_dp --output_dir {output} "
                "--checkpoint_prefix {checkpoint} --seed {seed} "
                "--batch_size {batch} --steps {iterations} "
                "--learning_rate {lr} --noise_multiplier {noise} "
                "--count_noise_multiplier 8.0 --clip_c {clip_c} "
                "--warmup_steps 5 --update_interval 5 "
                "--checkpoint_interval {checkpoint_interval} "
                "--precision_fractional {precision} "
                "--matmul_chunk_size {chunk} "
                "--initializer_privacy_status data_independent"
            ).format(
                tfe_python=TFE_PYTHON,
                stage_c=ROOT / "new_stage_c_secure_dpsgd.py",
                cache=feature_cache,
                initializer=initializer,
                dataset=dataset,
                output=output_dir,
                checkpoint=checkpoint,
                seed=seed,
                batch=batch_size,
                iterations=iterations,
                lr=formal["learning_rate"],
                noise=formal["noise_multiplier"],
                clip_c=formal["clip_c"],
                checkpoint_interval=formal["checkpoint_interval"],
                precision=formal["precision_fractional"],
                chunk=formal["matmul_chunk_size"],
            )
            command = (
                "{python} {measure} --system_report {system_report} "
                "--inner_report {inner_report} --dataset {dataset} "
                "--scale {scale} --num_samples {samples} --feature_dim 2048 "
                "--num_classes {classes} --batch_size {batch} "
                "--iterations {iterations} --precision_fractional {precision} "
                "-- {inner}"
            ).format(
                python=MODERN_PYTHON,
                measure=ROOT / "new_measure_tfe_system.py",
                system_report=system_report,
                inner_report=inner_report,
                dataset=dataset,
                scale=scale,
                samples=sample_counts[dataset],
                classes=class_counts[dataset],
                batch=batch_size,
                iterations=iterations,
                precision=formal["precision_fractional"],
                inner=inner,
            )
            rows.append("{}\t{}\t{}".format(task_id, system_report, command))
            tasks.append(
                {
                    "task_id": task_id,
                    "phase": "tfe_system_formal",
                    "dataset": dataset,
                    "seed": seed,
                    "scale": scale,
                    "method": "Secure-Fixed-DP-SGD",
                    "privacy": "measured_inner_report",
                    "resource": "tfe",
                    "num_samples": sample_counts[dataset],
                    "feature_dim": 2048,
                    "num_classes": class_counts[dataset],
                    "batch_size": batch_size,
                    "iterations": iterations,
                    "model_parameters": (
                        2048 * class_counts[dataset] + class_counts[dataset]
                    ),
                    "fixed_point_precision_fractional": formal[
                        "precision_fractional"
                    ],
                    "communication_bytes": "N/A",
                    "communication_rounds": "N/A",
                    "test_loaded": False,
                    "expected_output": str(system_report),
                }
            )
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(
        json.dumps(
            {
                "schema": "new_tfe_system_formal_plan_v1",
                "status": "READY",
                "formal_result": True,
                "test_loaded": False,
                "dp_mtt_status": "EXCLUDED_BY_USER",
                "not_smoke": True,
                "backend": "tfe_mpc",
                "protocol": "ABY3",
                "communication_limitation": (
                    "single-machine TFE does not expose bytes or rounds; N/A"
                ),
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
            "phase={phase} dataset={dataset} seed={seed} scale={scale} "
            "method={method} privacy={privacy} resource={resource} "
            "expected={expected_output}".format(**task)
        )
    print("queue={}".format(QUEUE))
    print("plan={}".format(PLAN))
    print("activated={}".format(bool(args.activate)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

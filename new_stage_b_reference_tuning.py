#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference/selection-only hyperparameter tuning for secure Stage B."""

import argparse
import itertools
import json
from types import SimpleNamespace

import numpy as np

import new_stage_b_secure_init as secure_b
from new_public_mtt_mainline_common import ROOT, load_config
from new_secure_feature_models import configure_single_thread_runtime


JSON_ROOT = ROOT / "new_log/new_public_mtt_mainline/json/new_stage_b_tuning"


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new_tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def accuracy(x, y, weight, bias):
    predictions = np.argmax(np.matmul(x, weight) + bias, axis=1)
    return float(np.mean(predictions == y))


def entropy(weights):
    values = np.asarray(weights, dtype=np.float64)
    values = np.clip(values, 1e-12, 1.0)
    return float(-np.sum(values * np.log(values)))


def teacher_disagreement(x, weights, biases):
    logits = np.stack(
        [np.matmul(x, weight) + bias for weight, bias in zip(weights, biases)],
        axis=0,
    )
    return float(np.mean(np.var(logits, axis=0)))


def candidate_grid(dataset, config):
    base = config["module_b"]["epoch_budget"][dataset]
    grid = config["module_b"]["tuning_grid_by_dataset"][dataset]
    combinations = itertools.product(
        grid["local_lr"],
        grid["distill_lr"],
        grid["lambda_kd"],
        grid["weight_temperature"],
        grid["reference_supervised_weight"],
    )
    return [
        {
            "candidate_id": "new_b_{:02d}".format(index),
            "local_epochs": int(base["local_epochs"]),
            "distill_epochs": int(base["distill_epochs"]),
            "local_lr": local_lr,
            "distill_lr": distill_lr,
            "lambda_kd": lambda_kd,
            "weight_temperature": temperature,
            "reference_supervised_weight": reference_supervised_weight,
        }
        for index, (
            local_lr,
            distill_lr,
            lambda_kd,
            temperature,
            reference_supervised_weight,
        ) in enumerate(combinations, start=1)
    ]


def run_candidate(dataset, candidate, config, cache_tag):
    rows = []
    num_classes = {"cifar10": 10, "cifar100": 100, "tiny": 200}[dataset]
    for seed in config["tuning_seeds"]:
        cache_path = (
            ROOT
            / "new_cache/new_public_mtt_mainline"
            / cache_tag
            / dataset
            / "new_stage_b_features_{}_seed{}.npz".format(dataset, seed)
        )
        data = secure_b.load_feature_cache(str(cache_path))
        for split in config["module_b"]["splits"]:
            indices = secure_b.make_client_splits(
                data["y_encoder_train"],
                config["module_b"]["num_clients"],
                split,
                seed,
            )
            initial_w, initial_b = secure_b.initialize_heads(
                config["module_b"]["num_clients"],
                data["feature_dim"],
                num_classes,
                seed,
                config["module_b"]["init_scale"],
            )
            args = SimpleNamespace(
                num_classes=num_classes,
                local_epochs=candidate["local_epochs"],
                batch_size=config["module_b"]["batch_size"],
                seed=seed,
                local_lr=candidate["local_lr"],
                distill_epochs=candidate["distill_epochs"],
                lambda_kd=candidate["lambda_kd"],
                distill_lr=candidate["distill_lr"],
                weight_temperature=candidate["weight_temperature"],
                reference_supervised_weight=candidate[
                    "reference_supervised_weight"
                ],
            )
            result = secure_b.plaintext_reference(
                data["x_encoder_train"],
                data["y_encoder_train"],
                data["x_reference"],
                data["y_reference"],
                data["x_selection"],
                data["y_selection"],
                indices,
                initial_w,
                initial_b,
                args,
            )
            methods = {
                "NoKD": (result["nokd_avg_w"], result["nokd_avg_b"]),
                "KD-Avg": (result["avg_w"], result["avg_b"]),
                "KD-ReferenceWeighted": (result["rw_w"], result["rw_b"]),
            }
            accuracies = {
                method: accuracy(
                    data["x_selection"],
                    data["y_selection"],
                    weight,
                    bias,
                )
                for method, (weight, bias) in methods.items()
            }
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "split": split,
                    "selection_accuracy": accuracies,
                    "kd_avg_gain": accuracies["KD-Avg"] - accuracies["NoKD"],
                    "kd_rw_gain": (
                        accuracies["KD-ReferenceWeighted"] - accuracies["NoKD"]
                    ),
                    "teacher_disagreement": teacher_disagreement(
                        data["x_reference"],
                        result["nokd_weights"],
                        result["nokd_biases"],
                    ),
                    "reference_weight_entropy": entropy(
                        result["reference_weights"]
                    ),
                    "selected_name": result["selected_name"],
                    "test_loaded": False,
                }
            )
    return rows


def candidate_score(rows):
    severe = [row for row in rows if row["split"] == "severe"]
    all_rw = [
        row["selection_accuracy"]["KD-ReferenceWeighted"] for row in rows
    ]
    severe_gain = np.mean(
        [
            max(row["kd_avg_gain"], row["kd_rw_gain"])
            for row in severe
        ]
    )
    return float(np.mean(all_rw) + 0.25 * severe_gain)


def main():
    parser = argparse.ArgumentParser(
        description="Tune Stage-B KD using reference/selection only"
    )
    parser.add_argument(
        "--dataset", choices=("cifar10", "cifar100", "tiny"), required=True
    )
    parser.add_argument("--cache_tag", default="new_stage_b_npz_round2")
    parser.add_argument("--round_tag", default="round2")
    args = parser.parse_args()
    configure_single_thread_runtime(42)
    config = load_config()
    candidates = candidate_grid(args.dataset, config)
    results = []
    print("=" * 92, flush=True)
    print("Stage B reference-only tuning", flush=True)
    print(
        "dataset={} candidates={} seeds={} splits={} test_loaded=False".format(
            args.dataset,
            len(candidates),
            config["tuning_seeds"],
            config["module_b"]["splits"],
        ),
        flush=True,
    )
    print("cpu_threads=1 num_workers=0", flush=True)
    print("=" * 92, flush=True)
    log_points = set(
        np.linspace(1, len(candidates), min(10, len(candidates)))
        .round()
        .astype(np.int64)
        .tolist()
    )
    for index, candidate in enumerate(candidates, start=1):
        rows = run_candidate(args.dataset, candidate, config, args.cache_tag)
        score = candidate_score(rows)
        results.append(
            {"candidate": candidate, "selection_score": score, "rows": rows}
        )
        if index in log_points:
            print(
                "  candidate={}/{} id={} score={:.5f}".format(
                    index, len(candidates), candidate["candidate_id"], score
                ),
                flush=True,
            )
    winner = max(results, key=lambda row: row["selection_score"])
    payload = {
        "schema": "new_stage_b_reference_tuning_v1",
        "status": "COMPLETE",
        "dataset": args.dataset,
        "test_loaded": False,
        "selection_rule": (
            "mean KD-ReferenceWeighted selection accuracy plus 0.25 times "
            "mean best KD gain on severe split"
        ),
        "winner": winner,
        "candidates": results,
    }
    output = JSON_ROOT / "new_stage_b_tuning_{}_{}.json".format(
        args.round_tag, args.dataset
    )
    atomic_json(payload, output)
    print("winner={}".format(winner["candidate"]), flush=True)
    print("output={}".format(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

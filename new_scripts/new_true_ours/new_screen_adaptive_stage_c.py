#!/usr/bin/env python3
"""Development-only float diagnostic for aligned Adaptive Stage C.

Run with the distillation environment (Opacus available).  This screen uses
common random numbers for one-factor comparisons, recalibrates the composed
accountant for every step budget, and never reads locked holdout or test.
TFE/ABY3 finalists remain mandatory before protocol freeze.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_stage_b_secure_init import load_feature_cache, one_hot, sha256_file
from new_stage_c_privacy_calibration import calibrate_noise_multiplier
from new_stage_c_secure_dpsgd import adaptive_update_count


DATASETS = ("cifar10", "cifar100", "tiny")
STEP_GRIDS = {
    "cifar10": (10, 20, 40, 80),
    "cifar100": (20, 40, 80, 120),
    "tiny": (30, 60, 120, 180),
}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def poisson_ids(size, expected_batch, seed, step):
    rng = np.random.RandomState(int(seed) + int(step) * 104729)
    return np.flatnonzero(
        rng.random_sample(int(size))
        < float(expected_batch) / float(size)
    ).astype(np.int64)


def evaluate(weight, bias, x_dev, y_dev):
    prediction = np.argmax(np.matmul(x_dev, weight) + bias, axis=1)
    return float(np.mean(prediction == y_dev))


def calibrate(config, sample_count):
    count_steps = adaptive_update_count(
        config["steps"], config["warmup_steps"], config["update_interval"]
    )
    noise, epsilon = calibrate_noise_multiplier(
        target_epsilon=3.0,
        steps=config["steps"],
        sample_rate=float(config["batch_size"]) / float(sample_count),
        delta=1e-5,
        count_noise=config["count_noise_multiplier"],
        count_steps=count_steps,
    )
    return noise, epsilon, count_steps


def run_once(data, initializer, seed, config):
    x_train = data["x_encoder_train"]
    y_train = data["y_encoder_train"]
    x_dev = data["x_reference"]
    y_dev = data["y_reference"]
    weight = initializer["weight"].copy()
    bias = initializer["bias"].copy()
    noise_multiplier, epsilon, count_steps = calibrate(config, len(y_train))
    rng = np.random.RandomState(int(seed) + 7282026)
    current_c = float(config["initial_c"])
    history = []
    evaluation_steps = set(
        np.linspace(
            1, int(config["steps"]), min(10, int(config["steps"]))
        )
        .round()
        .astype(np.int64)
        .tolist()
    )
    started = time.time()
    for step in range(int(config["steps"])):
        ids = poisson_ids(
            len(y_train), config["batch_size"], seed, step
        )
        x_batch = x_train[ids]
        targets = one_hot(y_train[ids], bias.shape[0])
        error = np.matmul(x_batch, weight) + bias - targets
        norm_squared = np.sum(error * error, axis=1) * (
            np.sum(x_batch * x_batch, axis=1) + 1.0
        )
        factors = np.minimum(
            1.0, current_c / np.sqrt(norm_squared + 1e-5)
        )
        clipped = error * factors[:, None]
        grad_w = np.matmul(x_batch.T, clipped) / float(
            config["batch_size"]
        )
        grad_b = np.sum(clipped, axis=0) / float(config["batch_size"])
        noise_scale = (
            noise_multiplier * current_c / float(config["batch_size"])
        )
        grad_w += rng.normal(0.0, noise_scale, grad_w.shape)
        grad_b += rng.normal(0.0, noise_scale, grad_b.shape)
        weight -= float(config["learning_rate"]) * grad_w
        bias -= float(config["learning_rate"]) * grad_b
        adaptive = (
            step >= int(config["warmup_steps"])
            and (step + 1) % int(config["update_interval"]) == 0
        )
        noisy_fraction = None
        if adaptive:
            fraction_below = float(
                np.sum(norm_squared <= current_c**2)
                / float(config["batch_size"])
            )
            noisy_fraction = float(
                np.clip(
                    fraction_below
                    + rng.normal()
                    * config["count_noise_multiplier"]
                    / float(config["batch_size"]),
                    0.0,
                    1.0,
                )
            )
            current_c = float(
                np.clip(
                    current_c
                    * math.exp(
                        -config["eta_c"]
                        * (noisy_fraction - config["gamma"])
                    ),
                    config["c_min"],
                    config["c_max"],
                )
            )
        row = {
            "step": step + 1,
            "development_validation_accuracy": None,
            "clipping_bound": current_c,
            "dp_noised_fraction_below": noisy_fraction,
            "sampled_batch_size": int(len(ids)),
        }
        if step + 1 in evaluation_steps:
            row["development_validation_accuracy"] = evaluate(
                weight, bias, x_dev, y_dev
            )
        history.append(row)
    evaluated = [
        row["development_validation_accuracy"]
        for row in history
        if row["development_validation_accuracy"] is not None
    ]
    return {
        "development_validation_accuracy": evaluated[-1],
        "best_development_validation_accuracy": max(evaluated),
        "noise_multiplier": noise_multiplier,
        "accounted_epsilon": epsilon,
        "count_steps": count_steps,
        "final_c": current_c,
        "runtime_seconds": time.time() - started,
        "history": history,
    }


def load_inputs(dataset, seed):
    cache_path = (
        ROOT
        / "new_cache/new_true_ours/official_mtt/stage_c"
        / dataset
        / "new_true_ours_stage_c_{}_seed{}.npz".format(dataset, seed)
    )
    initializer_path = (
        ROOT
        / "new_output/new_true_ours/stage_b/true_ours_finalist"
        / dataset
        / "seed{}".format(seed)
        / "new_secure_stage_b_candidate_nokd_{}_balanced_seed{}.npz".format(
            dataset, seed
        )
    )
    data = load_feature_cache(str(cache_path))
    with np.load(str(initializer_path), allow_pickle=False) as model:
        if (
            str(model["initializer_source"].reshape(-1)[0])
            != "official_mtt_only"
            or int(model["full_data_initializer_dependency"].reshape(-1)[0])
            != 0
        ):
            raise ValueError("True Ours initializer lineage rejected")
        initializer = {
            "weight": np.asarray(model["weight"], dtype=np.float32),
            "bias": np.asarray(model["bias"], dtype=np.float32),
        }
    return cache_path, initializer_path, data, initializer


def screen_dataset(dataset, seed):
    cache_path, initializer_path, data, initializer = load_inputs(dataset, seed)
    current = {
        "steps": STEP_GRIDS[dataset][0],
        "batch_size": 16,
        "learning_rate": 0.002,
        "initial_c": 1.0,
        "gamma": 0.5,
        "eta_c": 0.05,
        "c_min": 0.05,
        "c_max": 4.0,
        "warmup_steps": 5,
        "update_interval": 5,
        "count_noise_multiplier": 8.0,
    }
    factors = [
        ("steps", STEP_GRIDS[dataset]),
        ("learning_rate", (0.001, 0.002, 0.005, 0.01)),
        ("initial_c", (0.5, 1.0, 2.0)),
        ("gamma", (0.1, 0.3, 0.5)),
        ("eta_c", (0.02, 0.05, 0.1)),
    ]
    rows = []
    for factor, values in factors:
        candidates = []
        for value in values:
            config = dict(current)
            config[factor] = value
            result = run_once(data, initializer, seed, config)
            row = {
                "dataset": dataset,
                "seed": int(seed),
                "factor": factor,
                "factor_value": value,
                "config": config,
                **result,
                "locked_holdout_read": False,
                "official_test_read": False,
            }
            rows.append(row)
            candidates.append(row)
            print(
                "{} {}={} dev={:.5f} eps={:.6f}".format(
                    dataset,
                    factor,
                    value,
                    row["development_validation_accuracy"],
                    row["accounted_epsilon"],
                ),
                flush=True,
            )
        chosen = max(
            candidates,
            key=lambda row: (
                row["development_validation_accuracy"],
                row["best_development_validation_accuracy"],
                -row["runtime_seconds"],
            ),
        )
        current = dict(chosen["config"])
    final = run_once(data, initializer, seed, current)
    return {
        "dataset": dataset,
        "seed": int(seed),
        "feature_cache": str(cache_path),
        "feature_cache_sha256": sha256_file(str(cache_path)),
        "initializer": str(initializer_path),
        "initializer_sha256": sha256_file(str(initializer_path)),
        "best_config": current,
        "best_result": final,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = [screen_dataset(dataset, args.seed) for dataset in args.datasets]
    payload = {
        "schema": "new_adaptive_stage_c_development_screen_v1",
        "status": "COMPLETE",
        "backend": "numpy_float_dp_diagnostic",
        "formal_backend_required": "TFE_ABY3",
        "target_epsilon": 3.0,
        "delta": 1e-5,
        "common_random_numbers": True,
        "sampling_mode": "poisson_development_emulation",
        "formal_sampling_rng_required": "os_csprng",
        "results": results,
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    output = (
        ROOT / "new_log/new_true_ours/new_adaptive_stage_c_development_screen.json"
    )
    atomic_json(output, payload)
    print("output={}".format(output))


if __name__ == "__main__":
    main()

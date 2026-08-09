#!/usr/bin/env python3
"""Bounded development screen for latest-initializer Fixed vs Adaptive DP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "new_configs/new_paper_adaptive/new_adaptive_latest_initializer_screen_freeze.json"
OUT_JSON = ROOT / "new_log/new_paper_adaptive/new_adaptive_latest_initializer_screen.json"
OUT_CSV = ROOT / "new_expTab/new_paper_adaptive/new_adaptive_latest_initializer_screen.csv"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_steps(steps, warmup, interval):
    return sum(1 for step in range(steps) if step >= warmup and (step + 1) % interval == 0)


def calibrate(target_epsilon, steps, rate, delta, count_noise=None, releases=0):
    from new_stage_c_privacy_calibration import calibrate_noise_multiplier

    return calibrate_noise_multiplier(
        target_epsilon=target_epsilon,
        steps=steps,
        sample_rate=rate,
        delta=delta,
        count_noise=count_noise,
        count_steps=releases,
    )


def evaluate(weight, bias, x_dev, y_dev):
    return float(np.mean(np.argmax(np.matmul(x_dev, weight) + bias, axis=1) == y_dev))


def poisson_ids(size, expected_batch, seed, step):
    rng = np.random.RandomState(int(seed) + (step + 1) * 104729)
    return np.flatnonzero(rng.random_sample(size) < float(expected_batch) / float(size))


def load_inputs(freeze):
    model_path = ROOT / freeze["initializer"]
    cache_path = ROOT / freeze["feature_cache"]
    with np.load(model_path, allow_pickle=False) as model:
        if str(model["initializer_source"].reshape(-1)[0]) != "official_mtt_only":
            raise ValueError("latest initializer lineage mismatch")
        if int(model["full_data_initializer_dependency"].reshape(-1)[0]) != 0:
            raise ValueError("latest initializer has full-data dependency")
        weight = np.asarray(model["c5_weight"], dtype=np.float32)
        bias = np.asarray(model["c5_bias"], dtype=np.float32)
    with np.load(cache_path, allow_pickle=False) as cache:
        if int(cache["test_loaded"].reshape(-1)[0]) != 0:
            raise ValueError("screen cache contains official test")
        data = {
            "x_train": np.asarray(cache["x_encoder_train"], dtype=np.float32),
            "y_train": np.asarray(cache["y_encoder_train"], dtype=np.int64),
            "x_dev": np.asarray(cache["x_reference"], dtype=np.float32),
            "y_dev": np.asarray(cache["y_reference"], dtype=np.int64),
        }
    if weight.shape != (data["x_train"].shape[1], int(bias.shape[0])):
        raise ValueError("initializer shape mismatch")
    return model_path, cache_path, weight, bias, data


def run_candidate(payload):
    config, weight0, bias0, data = payload
    started = time.time()
    weight = weight0.copy()
    bias = bias0.copy()
    adaptive = config["mode"] == "adaptive_dp"
    releases = count_steps(config["steps"], config["warmup_steps"], config["update_interval"]) if adaptive else 0
    noise, epsilon = calibrate(
        config["target_epsilon"],
        config["steps"],
        float(config["expected_batch_size"]) / len(data["y_train"]),
        config["delta"],
        config["count_noise_multiplier"] if adaptive else None,
        releases,
    )
    gradient_rng = np.random.RandomState(config["rng_seed"] + 7282026)
    count_rng = np.random.RandomState(config["rng_seed"] + 19088743)
    current_c = float(config["clip_c"])
    evaluation_steps = set(np.linspace(1, config["steps"], min(12, config["steps"])).round().astype(int))
    history = []
    for step in range(config["steps"]):
        ids = poisson_ids(len(data["y_train"]), config["expected_batch_size"], config["rng_seed"], step)
        if len(ids):
            x_batch = data["x_train"][ids]
            targets = np.eye(bias.shape[0], dtype=np.float32)[data["y_train"][ids]]
            error = np.matmul(x_batch, weight) + bias - targets
            norm_squared = np.sum(error * error, axis=1) * (np.sum(x_batch * x_batch, axis=1) + 1.0)
            factors = np.minimum(1.0, current_c / np.sqrt(norm_squared + 1e-5))
            clipped = error * factors[:, None]
            denom = float(config["expected_batch_size"])
            grad_w = np.matmul(x_batch.T, clipped) / denom
            grad_b = np.sum(clipped, axis=0) / denom
            scale = noise * current_c / denom
            grad_w += gradient_rng.normal(0.0, scale, grad_w.shape)
            grad_b += gradient_rng.normal(0.0, scale, grad_b.shape)
            weight -= config["learning_rate"] * grad_w
            bias -= config["learning_rate"] * grad_b
            fraction_below = float(np.mean(norm_squared <= current_c * current_c))
            clipped_fraction = 1.0 - fraction_below
            norm_mean = float(np.mean(np.sqrt(norm_squared)))
        else:
            fraction_below = 0.0
            clipped_fraction = 0.0
            norm_mean = 0.0
        update = adaptive and step >= config["warmup_steps"] and (step + 1) % config["update_interval"] == 0
        noisy_fraction = None
        if update:
            noisy_fraction = float(np.clip(fraction_below + count_rng.normal() * config["count_noise_multiplier"] / config["expected_batch_size"], 0.0, 1.0))
            current_c = float(np.clip(current_c * math.exp(-config["eta_c"] * (noisy_fraction - config["gamma"])), config["c_min"], config["c_max"]))
        accuracy = evaluate(weight, bias, data["x_dev"], data["y_dev"]) if step + 1 in evaluation_steps else None
        history.append({
            "step": step + 1,
            "development_validation_accuracy": accuracy,
            "clipping_bound": current_c,
            "fraction_clipped_nonprivate_diagnostic": clipped_fraction,
            "mean_gradient_norm_nonprivate_diagnostic": norm_mean,
            "dp_noised_fraction_below": noisy_fraction,
            "sampled_batch_size": int(len(ids)),
        })
    evaluated = [row["development_validation_accuracy"] for row in history if row["development_validation_accuracy"] is not None]
    return {
        "config": config,
        "noise_multiplier": noise,
        "accounted_epsilon": epsilon,
        "count_releases": releases,
        "final_development_validation_accuracy": evaluated[-1],
        "best_development_validation_accuracy": max(evaluated),
        "early_auc": float(np.mean(evaluated)),
        "final_c": current_c,
        "runtime_seconds": time.time() - started,
        "history": history,
    }


def run_parallel(configs, weight, bias, data, workers):
    payloads = [(config, weight, bias, data) for config in configs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run_candidate, payloads))


def choose(rows):
    return max(rows, key=lambda row: (row["final_development_validation_accuracy"], row["early_auc"], -row["runtime_seconds"]))


def flatten(stage, row):
    config = row["config"]
    return {
        "stage": stage,
        "mode": config["mode"],
        "steps": config["steps"],
        "learning_rate": config["learning_rate"],
        "clip_c": config["clip_c"],
        "gamma": config["gamma"],
        "eta_c": config["eta_c"],
        "final_accuracy": row["final_development_validation_accuracy"],
        "best_accuracy": row["best_development_validation_accuracy"],
        "early_auc": row["early_auc"],
        "final_c": row["final_c"],
        "noise_multiplier": row["noise_multiplier"],
        "accounted_epsilon": row["accounted_epsilon"],
        "count_releases": row["count_releases"],
        "runtime_seconds": row["runtime_seconds"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_SCREEN":
        raise ValueError("screen configuration is not frozen")
    model_path, cache_path, weight, bias, data = load_inputs(freeze)
    base = {
        "mode": "fixed_dp",
        "steps": 80,
        "learning_rate": 0.002,
        "clip_c": 1.35,
        "gamma": 0.5,
        "eta_c": 0.05,
        "warmup_steps": freeze["warmup_steps"],
        "update_interval": freeze["update_interval"],
        "count_noise_multiplier": freeze["count_noise_multiplier"],
        "c_min": freeze["c_min"],
        "c_max": freeze["c_max"],
        "target_epsilon": freeze["target_epsilon"],
        "delta": freeze["delta"],
        "expected_batch_size": freeze["expected_batch_size"],
        "rng_seed": freeze["screen_rng_seed"],
    }
    stage_rows = {}
    configs = []
    for steps in freeze["stages"]["budget_lr"]["steps"]:
        for lr in freeze["stages"]["budget_lr"]["learning_rate"]:
            config = dict(base, steps=int(steps), learning_rate=float(lr), clip_c=float(freeze["stages"]["budget_lr"]["fixed_c"]))
            configs.append(config)
    stage_rows["budget_lr"] = run_parallel(configs, weight, bias, data, args.workers)
    budget_best = choose(stage_rows["budget_lr"])

    configs = [dict(budget_best["config"], clip_c=float(value)) for value in freeze["stages"]["fixed_bound"]["clip_c"]]
    stage_rows["fixed_bound"] = run_parallel(configs, weight, bias, data, args.workers)
    fixed_best = choose(stage_rows["fixed_bound"])

    configs = []
    for initial_c in freeze["stages"]["adaptive"]["initial_c"]:
        for gamma in freeze["stages"]["adaptive"]["gamma"]:
            for eta_c in freeze["stages"]["adaptive"]["eta_c"]:
                configs.append(dict(fixed_best["config"], mode="adaptive_dp", clip_c=float(initial_c), gamma=float(gamma), eta_c=float(eta_c)))
    stage_rows["adaptive"] = run_parallel(configs, weight, bias, data, args.workers)
    adaptive_best = choose(stage_rows["adaptive"])

    initial_accuracy = evaluate(weight, bias, data["x_dev"], data["y_dev"])
    payload = {
        "schema": "new_adaptive_latest_initializer_screen_v1",
        "status": "COMPLETE",
        "freeze": str(FREEZE.relative_to(ROOT)),
        "freeze_sha256": sha256(FREEZE),
        "initializer": str(model_path.relative_to(ROOT)),
        "initializer_sha256": sha256(model_path),
        "feature_cache": str(cache_path.relative_to(ROOT)),
        "feature_cache_sha256": sha256(cache_path),
        "initial_development_validation_accuracy": initial_accuracy,
        "best_fixed": fixed_best,
        "best_adaptive": adaptive_best,
        "adaptive_minus_fixed_pp": 100.0 * (adaptive_best["final_development_validation_accuracy"] - fixed_best["final_development_validation_accuracy"]),
        "stages": stage_rows,
        "official_test_read": False,
        "locked_holdout_read": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flat = [flatten(stage, row) for stage, rows in stage_rows.items() for row in rows]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"initial_accuracy": initial_accuracy, "best_fixed": flatten("fixed_bound", fixed_best), "best_adaptive": flatten("adaptive", adaptive_best), "adaptive_minus_fixed_pp": payload["adaptive_minus_fixed_pp"]}, indent=2))


if __name__ == "__main__":
    main()

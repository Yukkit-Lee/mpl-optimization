#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：完整 ConvNet A+B+C 的统一正式论文实验。

本入口严格区分 formal 与 quick/smoke；所有候选选择和 Adaptive 调参仅使用
heldout_reference_set，test set 只在预设步数只读评估并报告最终结果。
"""

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch

from project_paths import PATHS
import official_distilled_convnet_pipeline as pipe


SEEDS = [42, 43, 44]
DATASETS = ["cifar10", "cifar100", "tiny"]
SPLITS = {
    "cifar10": {"IID": None, "Mild Non-IID": 5, "Severe Non-IID": 2},
    "cifar100": {"IID": None, "Mild Non-IID": 50, "Severe Non-IID": 10},
    "tiny": {"IID": None, "Mild Non-IID": 100, "Severe Non-IID": 20},
}
FORMAL = {
    "cifar10": {
        "local_steps": 300, "kd_steps": 100, "b_curve_steps": 800,
        "c_steps": 300, "batch_size": 64, "test_samples": 0,
    },
    "cifar100": {
        "local_steps": 300, "kd_steps": 100, "b_curve_steps": 800,
        "c_steps": 300, "batch_size": 64, "test_samples": 0,
    },
    "tiny": {
        "local_steps": 300, "kd_steps": 100, "b_curve_steps": 800,
        "c_steps": 300, "batch_size": 32, "test_samples": 0,
    },
}
COMMON = {
    "cpu_threads": 1,
    "num_clients": 5,
    "reference_per_class": 5,
    "local_lr": 0.01,
    "kd_lr": 0.005,
    "tau": 2.0,
    "lambda_kd": 1.0,
    "kd_local_supervised_weight": 0.1,
    "ref_weight_temperature": 0.08,
    "teacher_weight_temperature": 0.20,
    "curve_lr": 0.002,
    "curve_eval_points": 10,
    "stage_c_batch_size": 16,
    "stage_c_lr": 0.001,
    "noise_multiplier": 1.2,
    "delta": 1e-5,
    "adaptive_gamma_candidates": [0.1, 0.2, 0.3],
    "adaptive_eta_candidates": [0.03, 0.05],
    "adaptive_pilot_steps": 40,
}
TFE_FORMAL = {
    "cifar10": {"samples": 50, "steps": 10, "lr": 1e-4},
    "cifar100": {"samples": 100, "steps": 10, "lr": 5e-5},
    "tiny": {"samples": 200, "steps": 10, "lr": 1e-5},
}

JSON_DIR = PATHS.output_root / "log" / "json"
TXT_DIR = PATHS.output_root / "log" / "txt"
TABLE_DIR = PATHS.output_root / "expTab"
PAPER_DIR = PATHS.output_root / "expPaperTab"
FIG_DIR = PATHS.output_root / "expFig" / "formal_cross_stage"


class Tee:
    def __init__(self, path, mode="w"):
        self.console = sys.stdout
        self.handle = Path(path).open(mode, encoding="utf-8", buffering=1)

    def write(self, value):
        self.console.write(value)
        self.handle.write(value)

    def flush(self):
        self.console.flush()
        self.handle.flush()

    def close(self):
        self.handle.close()


def _history_auc(history, key):
    points = [(float(row["step"]), row.get(key)) for row in history if row.get(key) is not None]
    if not points:
        return float("nan")
    if len(points) == 1:
        return float(points[0][1])
    x = np.asarray([p[0] for p in points], dtype=np.float64)
    y = np.asarray([p[1] for p in points], dtype=np.float64)
    return float(np.trapz(y, x) / max(x[-1] - x[0], 1.0))


def _mean_std(values):
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))])
    if len(arr) == 0:
        return None, None
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _json_result(result):
    return {key: value for key, value in result.items() if key != "model"}


def _release_device_cache():
    """长实验方法间主动释放 Python/CUDA 缓存，降低原生运行时段错误风险。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_local_and_kd(dataset, bundle, seed, classes_per_client, cfg, device):
    started = time.time()
    splits = pipe.make_client_splits(
        bundle["y_train"], int(COMMON["num_clients"]), classes_per_client, seed
    )
    shared_initial_model = pipe.build_convnet(dataset, seed, device).cpu()
    local_models = []
    for client_id, ids in enumerate(splits):
        local_models.append(pipe.train_local_convnet(
            dataset, bundle["x_train"][ids], bundle["y_train"][ids],
            seed + client_id, int(cfg["local_steps"]), int(cfg["batch_size"]),
            float(COMMON["local_lr"]), device,
            initial_model=shared_initial_model,
        ))
    nokd_avg = pipe.model_from_state(
        dataset, pipe.aggregate_full_state(local_models), device
    ).cpu()
    teacher, teacher_class_weights = pipe.reference_guided_teacher_probabilities(
        local_models,
        bundle["x_ref"],
        bundle["y_ref"],
        float(COMMON["tau"]),
        float(COMMON["teacher_weight_temperature"]),
        device,
    )
    distilled = []
    for client_id, (model, ids) in enumerate(zip(local_models, splits)):
        distilled.append(pipe.distill_full_convnet(
            model, bundle["x_train"][ids], bundle["y_train"][ids],
            bundle["x_ref"], teacher, seed + 100 + client_id,
            int(cfg["kd_steps"]), int(cfg["batch_size"]), float(COMMON["kd_lr"]),
            float(COMMON["tau"]), float(COMMON["lambda_kd"]), device,
            local_supervised_weight=float(COMMON["kd_local_supervised_weight"]),
        ))
    kd = pipe.build_stage_b_candidates(
        dataset, distilled, bundle["x_ref"], bundle["y_ref"], device,
        float(COMMON["ref_weight_temperature"]),
    )
    methods = {
        "NoKD": nokd_avg,
        "KD-Avg": kd["candidates"]["KD-Avg"],
        "KD-ReferenceWeighted": kd["candidates"]["KD-ReferenceWeighted"],
        "KD-Best": kd["candidates"]["KD-Best"],
    }
    ref_scores = {
        name: pipe.evaluate_model(model, bundle["x_ref"], bundle["y_ref"], device)[1]
        for name, model in methods.items()
    }
    chosen_name = max(
        ("KD-Avg", "KD-ReferenceWeighted", "KD-Best"),
        key=lambda name: ref_scores[name],
    )
    return {
        "methods": methods,
        "ref_scores": ref_scores,
        "chosen_name": chosen_name,
        "chosen_model": deepcopy(methods[chosen_name]),
        "splits": [ids.tolist() for ids in splits],
        "prep_runtime_s": float(time.time() - started),
        "reference_weights": kd["reference_weights"],
        "teacher_class_weights": teacher_class_weights.tolist(),
    }


def _run_module_b_curves(dataset, split_name, seed, bundle, prepared, cfg, device):
    target_ref = min(0.999, max(prepared["ref_scores"].values()) + 0.02)
    eval_interval = max(1, int(cfg["b_curve_steps"]) // int(COMMON["curve_eval_points"]))
    rows, curves = [], {}
    for method, model in prepared["methods"].items():
        print("    [B curve] {} steps={}".format(method, cfg["b_curve_steps"]))
        result = pipe.train_stage_c_convnet(
            model=model, bundle=bundle, mode="plain",
            steps=int(cfg["b_curve_steps"]), batch_size=int(cfg["batch_size"]),
            lr=float(COMMON["curve_lr"]), clip_c=1.0, noise_multiplier=0.0,
            gamma=0.5, eta_c=0.01, seed=seed + 5000, device=device,
            eval_interval=eval_interval, target_reference_accuracy=target_ref,
            delta=float(COMMON["delta"]), noise_mechanism="gaussian",
        )
        curves[method] = result["history"]
        rows.append({
            "Section": "ModuleB",
            "Dataset": dataset,
            "Seed": seed,
            "Split": split_name,
            "SplitParam": SPLITS[dataset][split_name],
            "Method": method,
            "FinalAccuracy": result["final_test_accuracy"],
            "StepAtTargetAcc": result["step_at_target_reference_accuracy"],
            "TargetSource": "heldout_reference_set",
            "TargetAccuracy": target_ref,
            "EarlyAUC": _history_auc(result["history"], "test_accuracy_readonly"),
            "RuntimeS": result["time_s"] + prepared["prep_runtime_s"] / 4.0,
            "InitialReferenceAccuracy": prepared["ref_scores"][method],
        })
        result.pop("model", None)
        _release_device_cache()
    return rows, curves


def _reference_calibration(model, bundle, device):
    norms = np.asarray(pipe.reference_gradient_norms(model, bundle, device, 64))
    q10, median, q75, q90 = np.quantile(norms, [0.10, 0.50, 0.75, 0.90])
    return {
        "reference_norms": norms.tolist(),
        # Fixed 与 Adaptive 从同一个 reference q75 起点开始，避免初始条件不公平。
        "c_init": float(max(q75, 1e-3)),
        "c_min": float(max(q10 * 0.25, 1e-3)),
        "c_max": float(max(q90 * 1.10, q75 * 1.05)),
        "reference_median": float(median),
    }


def _tune_adaptive(dataset, seed, model, bundle, cfg, calibration, device):
    """仅看 heldout reference history；pilot 中 evaluate_test=False。"""
    best, trials = None, []
    for gamma in COMMON["adaptive_gamma_candidates"]:
        for eta in COMMON["adaptive_eta_candidates"]:
            result = pipe.train_stage_c_convnet(
                model=model, bundle=bundle, mode="adaptive_dp",
                steps=int(COMMON["adaptive_pilot_steps"]),
                batch_size=int(COMMON["stage_c_batch_size"]),
                lr=float(COMMON["stage_c_lr"]),
                clip_c=float(calibration["c_init"]),
                noise_multiplier=float(COMMON["noise_multiplier"]),
                gamma=float(gamma), eta_c=float(eta), seed=seed + 7100,
                device=device, eval_interval=5,
                target_reference_accuracy=None, delta=float(COMMON["delta"]),
                noise_mechanism="gaussian",
                adaptive_c_min=float(calibration["c_min"]),
                adaptive_c_max=float(calibration["c_max"]),
                evaluate_test=False,
            )
            ref_auc = _history_auc(result["history"], "reference_accuracy")
            final_ref = result["final_reference_accuracy"]
            relative_noise_scale = float(result["final_c"] / calibration["c_init"])
            # 只用 reference utility 与噪声绝对尺度选择；不读取 test accuracy。
            score = float(final_ref + 0.25 * ref_auc - 0.03 * relative_noise_scale)
            trial = {
                "gamma": gamma, "eta_c": eta, "score": score,
                "final_reference_accuracy": final_ref,
                "reference_auc": ref_auc, "final_c": result["final_c"],
                "relative_noise_scale": relative_noise_scale,
                "test_evaluation_enabled": result["test_evaluation_enabled"],
            }
            trials.append(trial)
            if best is None or score > best["score"]:
                best = trial
    print(
        "    [C tune/ref-only] gamma={} eta={} score={:.4f} C={:.3f}->{:.3f} bounds=({:.3f},{:.3f})".format(
            best["gamma"], best["eta_c"], best["score"], calibration["c_init"],
            best["final_c"], calibration["c_min"], calibration["c_max"],
        )
    )
    return best, trials


def _run_module_c(dataset, seed, model, bundle, cfg, device):
    calibration = _reference_calibration(model, bundle, device)
    tuned, trials = _tune_adaptive(dataset, seed, model, bundle, cfg, calibration, device)
    target_ref = pipe.build_reference_target(model, bundle, device)
    eval_interval = max(1, int(cfg["c_steps"]) // int(COMMON["curve_eval_points"]))
    results, rows = {}, []
    for mode in ("plain", "fixed_dp", "adaptive_dp"):
        print("    [C formal] {} steps={}".format(mode, cfg["c_steps"]))
        result = pipe.train_stage_c_convnet(
            model=model, bundle=bundle, mode=mode, steps=int(cfg["c_steps"]),
            batch_size=int(COMMON["stage_c_batch_size"]),
            lr=float(COMMON["stage_c_lr"]),
            clip_c=float(calibration["c_init"]),
            noise_multiplier=0.0 if mode == "plain" else float(COMMON["noise_multiplier"]),
            gamma=float(tuned["gamma"]), eta_c=float(tuned["eta_c"]),
            seed=seed + 8000, device=device, eval_interval=eval_interval,
            target_reference_accuracy=target_ref, delta=float(COMMON["delta"]),
            noise_mechanism="gaussian",
            adaptive_c_min=float(calibration["c_min"]),
            adaptive_c_max=float(calibration["c_max"]),
            evaluate_test=True,
        )
        results[mode] = result
        privacy = result["privacy"]
        rows.append({
            "Section": "ModuleC",
            "Dataset": dataset,
            "Seed": seed,
            "Method": mode,
            "FinalAccuracy": result["final_test_accuracy"],
            "StepAtTargetAcc": result["step_at_target_reference_accuracy"],
            "TargetSource": "heldout_reference_set",
            "EarlyAUC": _history_auc(result["history"], "test_accuracy_readonly"),
            "RuntimeS": result["time_s"],
            "StageCEpsilon": privacy["stage_c_epsilon"],
            "StageCDelta": privacy["stage_c_delta"],
            "NoiseMultiplier": privacy.get("noise_multiplier", 0.0),
            "SampleRate": privacy["sample_rate"],
            "Steps": privacy["steps"],
            "FinalC": result["final_c"],
            "MeanClippedFraction": float(np.mean([
                row["clipped_fraction"] for row in result["history"]
            ])),
            "AdaptiveGamma": tuned["gamma"] if mode == "adaptive_dp" else None,
            "AdaptiveEta": tuned["eta_c"] if mode == "adaptive_dp" else None,
        })
        result.pop("model", None)
        _release_device_cache()
    fixed_priv = results["fixed_dp"]["privacy"]
    adaptive_priv = results["adaptive_dp"]["privacy"]
    for key in ("noise_multiplier", "sample_rate", "steps", "stage_c_delta"):
        if fixed_priv[key] != adaptive_priv[key]:
            raise RuntimeError("Fixed/Adaptive privacy settings differ: {}".format(key))
    return rows, results, {"calibration": calibration, "chosen": tuned, "trials": trials}


def _random_subset_bundle(dataset, bundle, seed):
    _, _, _, _, _, _, train_ds, _, *_ = pipe._mtt_dataset_objects(dataset)
    excluded = {
        int(sample_id.split(":", 1)[1])
        for sample_id in bundle["reference_sample_ids"]
    }
    candidates = np.asarray(
        [idx for idx in range(len(train_ds)) if idx not in excluded], dtype=np.int64
    )
    rng = np.random.RandomState(seed + 9900)
    rng.shuffle(candidates)
    selected = candidates[:len(bundle["y_train"])]
    x_random, y_random = pipe._materialize(train_ds, selected)
    random_bundle = dict(bundle)
    random_bundle["x_train"] = x_random
    random_bundle["y_train"] = y_random
    random_bundle["train_sample_ids"] = [
        "real_train:{}".format(int(idx)) for idx in selected
    ]
    pipe.assert_split_boundaries(random_bundle)
    return random_bundle


def _end2end_row(dataset, seed, method, result, stage_a, stage_b, stage_c):
    privacy = result["privacy"]
    return {
        "Section": "EndToEnd",
        "Dataset": dataset,
        "Seed": seed,
        "Method": method,
        "StageA": bool(stage_a),
        "StageB": bool(stage_b),
        "StageC": bool(stage_c),
        "FinalAccuracy": result["final_test_accuracy"],
        "StepAtTargetAcc": result["step_at_target_reference_accuracy"],
        "EarlyAUC": _history_auc(result["history"], "test_accuracy_readonly"),
        "RuntimeS": result["time_s"],
        "PrivacyScope": privacy["privacy_scope"],
        "StageCEpsilon": privacy["stage_c_epsilon"],
    }


def _run_end2end(dataset, seed, chosen_model, bundle, cfg, c_results, device):
    target_ref = pipe.build_reference_target(chosen_model, bundle, device)
    eval_interval = max(1, int(cfg["c_steps"]) // int(COMMON["curve_eval_points"]))
    random_bundle = _random_subset_bundle(dataset, bundle, seed)
    random_model = pipe.build_convnet(dataset, seed + 10000, device).cpu()
    distilled_model = pipe.build_convnet(dataset, seed + 11000, device).cpu()
    random_result = pipe.train_stage_c_convnet(
        random_model, random_bundle, "plain", int(cfg["c_steps"]),
        int(cfg["batch_size"]), float(COMMON["curve_lr"]), 1.0, 0.0,
        0.5, 0.01, seed + 12000, device, eval_interval, target_ref,
        delta=float(COMMON["delta"]), evaluate_test=True,
    )
    distilled_result = pipe.train_stage_c_convnet(
        distilled_model, bundle, "plain", int(cfg["c_steps"]),
        int(cfg["batch_size"]), float(COMMON["curve_lr"]), 1.0, 0.0,
        0.5, 0.01, seed + 13000, device, eval_interval, target_ref,
        delta=float(COMMON["delta"]), evaluate_test=True,
    )
    results = {
        "Random subset baseline": random_result,
        "Distilled only": distilled_result,
        "Distilled + Stage B": c_results["plain"],
        "Distilled + Stage B + Fixed-DP": c_results["fixed_dp"],
        "Distilled + Stage B + Adaptive-DP": c_results["adaptive_dp"],
    }
    rows = [
        _end2end_row(
            dataset, seed, method, result,
            stage_a=method != "Random subset baseline",
            stage_b="Stage B" in method,
            stage_c=("Fixed-DP" in method or "Adaptive-DP" in method),
        )
        for method, result in results.items()
    ]
    for result in results.values():
        result.pop("model", None)
    _release_device_cache()
    return rows, results


def _aggregate_curves(seed_curves):
    grouped = defaultdict(list)
    for seed, curves in seed_curves.items():
        for method, history in curves.items():
            grouped[method].append(history)
    output = {}
    for method, histories in grouped.items():
        common_steps = sorted(set.intersection(*[
            {int(row["step"]) for row in history} for history in histories
        ]))
        output[method] = []
        for step in common_steps:
            values = [
                next(row["test_accuracy_readonly"] for row in history if int(row["step"]) == step)
                for history in histories
            ]
            mean, std = _mean_std(values)
            output[method].append({"step": step, "mean": mean, "std": std})
    return output


def _plot_module_b(dataset, curves_by_split, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "NoKD": "#8a8a8a", "KD-Avg": "#4c78a8",
        "KD-ReferenceWeighted": "#f58518", "KD-Best": "#54a24b",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180, sharey=True)
    for axis, split in zip(axes, ("IID", "Mild Non-IID", "Severe Non-IID")):
        aggregate = _aggregate_curves(curves_by_split[split])
        for method, points in aggregate.items():
            x = np.asarray([p["step"] for p in points])
            mean = np.asarray([p["mean"] for p in points])
            std = np.asarray([p["std"] for p in points])
            axis.plot(x, mean, label=method, color=colors[method], linewidth=2)
            axis.fill_between(x, mean - std, mean + std, color=colors[method], alpha=0.14)
        axis.set_title(split)
        axis.set_xlabel("Optimizer Steps")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test Accuracy")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def _plot_module_c(dataset, seed_results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"plain": "#777777", "fixed_dp": "#4c78a8", "adaptive_dp": "#e45756"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180)
    for mode in ("plain", "fixed_dp", "adaptive_dp"):
        histories = [results[mode]["history"] for results in seed_results.values()]
        steps = sorted(set.intersection(*[{row["step"] for row in h} for h in histories]))
        means, stds = [], []
        for step in steps:
            values = [
                next(row["test_accuracy_readonly"] for row in h if row["step"] == step)
                for h in histories
            ]
            mean, std = _mean_std(values); means.append(mean); stds.append(std)
        axes[0].plot(steps, means, label=mode, color=colors[mode], linewidth=2)
        axes[0].fill_between(
            steps, np.asarray(means) - np.asarray(stds),
            np.asarray(means) + np.asarray(stds), color=colors[mode], alpha=0.14,
        )
    adaptive_histories = [result["adaptive_dp"]["history"] for result in seed_results.values()]
    steps = sorted(set.intersection(*[{row["step"] for row in h} for h in adaptive_histories]))
    c_mean, c_std = [], []
    for step in steps:
        values = [next(row["clipping_bound_t"] for row in h if row["step"] == step) for h in adaptive_histories]
        mean, std = _mean_std(values); c_mean.append(mean); c_std.append(std)
    axes[1].plot(steps, c_mean, color=colors["adaptive_dp"], linewidth=2)
    axes[1].fill_between(steps, np.asarray(c_mean)-np.asarray(c_std), np.asarray(c_mean)+np.asarray(c_std), color=colors["adaptive_dp"], alpha=0.14)
    for mode in ("fixed_dp", "adaptive_dp"):
        histories = [result[mode]["history"] for result in seed_results.values()]
        means = []
        for step in steps:
            values = [next(row["clipped_fraction"] for row in h if row["step"] == step) for h in histories]
            means.append(float(np.mean(values)))
        axes[2].plot(steps, means, label=mode, color=colors[mode], linewidth=2)
    axes[0].set_ylabel("Test Accuracy"); axes[0].set_xlabel("Optimizer Steps")
    axes[1].set_ylabel("Clipping Bound $C_t$"); axes[1].set_xlabel("Optimizer Steps")
    axes[2].set_ylabel("Clipped Fraction"); axes[2].set_xlabel("Optimizer Steps")
    axes[0].legend(); axes[2].legend()
    for axis in axes: axis.grid(alpha=0.25)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path)); plt.close(fig)


def _plot_end2end(dataset, seed_results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#888888", "#72b7b2", "#4c78a8", "#f58518", "#e45756"]
    aggregate = _aggregate_curves({
        seed: {method: result["history"] for method, result in results.items()}
        for seed, results in seed_results.items()
    })
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    for color, (method, points) in zip(colors, aggregate.items()):
        x = np.asarray([p["step"] for p in points])
        mean = np.asarray([p["mean"] for p in points])
        std = np.asarray([p["std"] for p in points])
        ax.plot(x, mean, label=method, color=color, linewidth=2)
        ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.14)
    ax.set_xlabel("Optimizer Steps"); ax.set_ylabel("Test Accuracy")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path)); plt.close(fig)


def _summary(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for group, items in sorted(groups.items(), key=lambda item: str(item[0])):
        record = dict(zip(keys, group))
        for metric in metrics:
            mean, std = _mean_std([item.get(metric) for item in items])
            record[metric + "Mean"] = mean
            record[metric + "Std"] = std
        record["Seeds"] = len(items)
        output.append(record)
    return output


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*[set(row.keys()) for row in rows]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _run_tfe_validations(datasets):
    outputs = {}
    python = os.environ.get(
        "TFE_PYTHON", str(Path.home() / ".conda/envs/tfe/bin/python")
    )
    for dataset in datasets:
        cfg = TFE_FORMAL[dataset]
        print("[TFE formal] dataset={} samples={} steps={}".format(
            dataset, cfg["samples"], cfg["steps"]
        ))
        before = set(JSON_DIR.glob("stageC_tfe_secure_system_{}_*.json".format(dataset)))
        command = [
            python, str(PATHS.output_root / "exp_stageC_tfe_secure_system.py"),
            "--dataset", dataset, "--samples", str(cfg["samples"]),
            "--steps", str(cfg["steps"]), "--learning_rate", str(cfg["lr"]),
        ]
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
        })
        completed = subprocess.run(
            command, cwd=str(PATHS.output_root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        print(completed.stdout[-1200:])
        if completed.returncode != 0:
            outputs[dataset] = {"status": "FAILED", "returncode": completed.returncode, "tail": completed.stdout[-4000:]}
            continue
        after = set(JSON_DIR.glob("stageC_tfe_secure_system_{}_*.json".format(dataset)))
        created = sorted(after - before)
        path = created[-1] if created else sorted(after)[-1]
        outputs[dataset] = json.loads(path.read_text(encoding="utf-8"))
        outputs[dataset]["source_json"] = str(path)
    return outputs


def _run_isolated_plotting(json_path):
    """在未初始化 CUDA 的独立进程中绘图，隔离 Matplotlib 原生库段错误。"""
    command = [
        sys.executable,
        str(PATHS.output_root / "plot_formal_cross_stage.py"),
        "--json",
        str(json_path),
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
    })
    return subprocess.run(
        command,
        cwd=str(PATHS.output_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _print_resolved(datasets, run_id):
    b_curves = len(datasets) * len(SEEDS) * 3 * 4
    c_curves = len(datasets) * len(SEEDS) * 3
    e_curves = len(datasets) * len(SEEDS) * 5
    pilots = len(datasets) * len(SEEDS) * len(COMMON["adaptive_gamma_candidates"]) * len(COMMON["adaptive_eta_candidates"])
    tfe = len(datasets)
    print("=" * 88)
    print("FORMAL CROSS-STAGE EXPERIMENT")
    print("run_id               :", run_id)
    print("datasets             :", datasets)
    print("seeds                :", SEEDS)
    print("cpu_threads          : 1")
    print("visible_gpu          :", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    print("splits               :", {d: SPLITS[d] for d in datasets})
    print("formal_config        :", {d: FORMAL[d] for d in datasets})
    print("privacy              : noise={}, delta={}, batch={}".format(
        COMMON["noise_multiplier"], COMMON["delta"], COMMON["stage_c_batch_size"]
    ))
    print("estimated_runs       : {} (B={} C={} E2E={} pilots={} TFE={})".format(
        b_curves + c_curves + e_curves + pilots + tfe,
        b_curves, c_curves, e_curves, pilots, tfe,
    ))
    print("expected_json        :", JSON_DIR / "formalCrossStage_{}.json".format(run_id))
    print("expected_csv         :", TABLE_DIR / "formalCrossStage_{}.csv".format(run_id))
    print("expected_txt         :", TXT_DIR / "formalCrossStage_{}.log".format(run_id))
    print("expected_fig_dir     :", FIG_DIR)
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(description="Formal full-ConvNet cross-stage experiments")
    parser.add_argument(
        "--dataset", default="all",
        choices=["all", "cifar10", "cifar100", "tiny"],
    )
    parser.add_argument("--skip_tfe", action="store_true")
    parser.add_argument(
        "--resume_partial",
        default="",
        help="从指定 formalCrossStage_*.partial.json 继续，跳过已完成 seed",
    )
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    resume_payload = None
    if args.resume_partial:
        resume_path = Path(args.resume_partial).resolve()
        resume_payload = json.loads(resume_path.read_text(encoding="utf-8"))
        run_id = str(resume_payload["run_id"])
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    for directory in (JSON_DIR, TXT_DIR, TABLE_DIR, PAPER_DIR, FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    txt_path = TXT_DIR / "formalCrossStage_{}.log".format(run_id)
    tee = Tee(txt_path, mode="a" if resume_payload else "w"); sys.stdout = tee
    started = time.time()
    try:
        _print_resolved(datasets, run_id)
        pipe.set_safe_runtime(42, 1)
        device = pipe.select_device("auto")
        if resume_payload:
            all_rows = list(resume_payload.get("rows", []))
            module_b_details = dict(resume_payload.get("module_b", {}))
            module_c_details = dict(resume_payload.get("module_c", {}))
            end_details = dict(resume_payload.get("end_to_end", {}))
            print("[RESUME] partial={} last_completed={}".format(
                args.resume_partial, resume_payload.get("last_completed")
            ))
        else:
            all_rows, module_b_details, module_c_details, end_details = [], {}, {}, {}
        for dataset in datasets:
            cfg = FORMAL[dataset]
            print("\n" + "#" * 88)
            print("[DATASET START] {} {}".format(dataset, time.strftime("%F %T")))
            module_b_details.setdefault(dataset, {})
            module_c_details.setdefault(dataset, {})
            end_details.setdefault(dataset, {})
            b_curves_for_plot = {split: {} for split in SPLITS[dataset]}
            c_for_plot, e_for_plot = {}, {}
            for seed_text, seed_details in module_b_details[dataset].items():
                for split_name in SPLITS[dataset]:
                    if split_name in seed_details:
                        b_curves_for_plot[split_name][int(seed_text)] = seed_details[split_name]["curves"]
            for seed_text, seed_details in module_c_details[dataset].items():
                c_for_plot[int(seed_text)] = seed_details["results"]
            for seed_text, seed_details in end_details[dataset].items():
                e_for_plot[int(seed_text)] = seed_details
            for seed in SEEDS:
                seed_key = str(seed)
                complete = (
                    seed_key in module_b_details[dataset]
                    and all(split in module_b_details[dataset][seed_key] for split in SPLITS[dataset])
                    and seed_key in module_c_details[dataset]
                    and seed_key in end_details[dataset]
                )
                if complete:
                    print("[RESUME SKIP] dataset={} seed={} 已完成".format(dataset, seed))
                    continue
                print("\n[SEED] dataset={} seed={}".format(dataset, seed))
                bundle = pipe.load_image_bundle(
                    dataset, int(COMMON["reference_per_class"]),
                    int(cfg["test_samples"]), seed,
                )
                pipe.assert_split_boundaries(bundle)
                module_b_details[dataset][str(seed)] = {}
                severe_prepared = None
                for split_name, classes_per_client in SPLITS[dataset].items():
                    print("  [B split] {} classes_per_client={}".format(
                        split_name, classes_per_client
                    ))
                    prepared = _build_local_and_kd(
                        dataset, bundle, seed, classes_per_client, cfg, device
                    )
                    rows, curves = _run_module_b_curves(
                        dataset, split_name, seed, bundle, prepared, cfg, device
                    )
                    all_rows.extend(rows)
                    b_curves_for_plot[split_name][seed] = curves
                    module_b_details[dataset][str(seed)][split_name] = {
                        "ref_scores": prepared["ref_scores"],
                        "chosen_name": prepared["chosen_name"],
                        "reference_weights": prepared["reference_weights"],
                        "teacher_class_weights": prepared["teacher_class_weights"],
                        "prep_runtime_s": prepared["prep_runtime_s"],
                        "curves": curves,
                    }
                    if split_name == "Severe Non-IID":
                        severe_prepared = prepared
                    else:
                        del prepared
                        _release_device_cache()
                chosen = severe_prepared["chosen_model"]
                checkpoint_stage = {
                    "input_type": bundle["input_type"],
                    "model_name": bundle["model_name"],
                    "model_class": type(chosen).__name__,
                    "model_schema": pipe.state_schema(chosen),
                    "chosen_name": severe_prepared["chosen_name"],
                    "chosen_model": chosen,
                    "chosen_signature": pipe.state_signature(chosen),
                    "candidate_ref_scores": severe_prepared["ref_scores"],
                }
                checkpoint = pipe.save_stage_b_checkpoint(
                    dataset, checkpoint_stage, seed,
                    PATHS.output_root / "init_model/cross_stage_convnet/formal_{}_seed{}.pt".format(dataset, seed),
                )
                loaded, metadata = pipe.load_stage_b_checkpoint(checkpoint, device)
                if pipe.state_signature(loaded) != checkpoint_stage["chosen_signature"]:
                    raise RuntimeError("formal Stage B checkpoint signature mismatch")
                c_rows, c_results, tuning = _run_module_c(
                    dataset, seed, loaded, bundle, cfg, device
                )
                all_rows.extend(c_rows)
                module_c_details[dataset][str(seed)] = {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "tuning": tuning,
                    "results": {k: _json_result(v) for k, v in c_results.items()},
                }
                c_for_plot[seed] = c_results
                e_rows, e_results = _run_end2end(
                    dataset, seed, loaded, bundle, cfg, c_results, device
                )
                all_rows.extend(e_rows)
                end_details[dataset][str(seed)] = {
                    method: _json_result(result) for method, result in e_results.items()
                }
                e_for_plot[seed] = e_results
                partial_path = JSON_DIR / "formalCrossStage_{}.partial.json".format(run_id)
                partial_path.write_text(json.dumps({
                    "run_id": run_id,
                    "status": "FORMAL_IN_PROGRESS",
                    "last_completed": {"dataset": dataset, "seed": seed},
                    "rows": all_rows,
                    "module_b": module_b_details,
                    "module_c": module_c_details,
                    "end_to_end": end_details,
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                print("[SEED COMPLETE] dataset={} seed={} {}".format(
                    dataset, seed, time.strftime("%F %T")
                ))
                _release_device_cache()
            print("[DATASET TRAINING COMPLETE] {}；绘图将在独立 CPU 进程执行".format(dataset))
        tfe = {} if args.skip_tfe else _run_tfe_validations(datasets)
        b_rows = [row for row in all_rows if row["Section"] == "ModuleB"]
        c_rows = [row for row in all_rows if row["Section"] == "ModuleC"]
        e_rows = [row for row in all_rows if row["Section"] == "EndToEnd"]
        b_summary = _summary(
            b_rows, ["Dataset", "Split", "Method"],
            ["FinalAccuracy", "StepAtTargetAcc", "EarlyAUC", "RuntimeS"],
        )
        c_summary = _summary(
            c_rows, ["Dataset", "Method"],
            ["FinalAccuracy", "StepAtTargetAcc", "EarlyAUC", "RuntimeS", "StageCEpsilon", "FinalC", "MeanClippedFraction"],
        )
        e_summary = _summary(
            e_rows, ["Dataset", "Method"],
            ["FinalAccuracy", "StepAtTargetAcc", "EarlyAUC", "RuntimeS", "StageCEpsilon"],
        )
        payload = {
            "run_id": run_id,
            "status": "FORMAL_COMPLETE",
            "resolved_config": {
                "datasets": datasets, "seeds": SEEDS, "splits": SPLITS,
                "formal": FORMAL, "common": COMMON, "tfe": TFE_FORMAL,
            },
            "module_b": {"summary": b_summary, "details": module_b_details},
            "module_c": {"summary": c_summary, "details": module_c_details},
            "end_to_end": {"summary": e_summary, "details": end_details},
            "tfe_lightweight_secure_system_validation": tfe,
            "privacy_scope": "stage_c_only",
            "total_runtime_s": float(time.time() - started),
        }
        json_path = JSON_DIR / "formalCrossStage_{}.json".format(run_id)
        csv_path = TABLE_DIR / "formalCrossStage_{}.csv".format(run_id)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_csv(csv_path, all_rows)
        _write_csv(PAPER_DIR / "formalModuleB_{}.csv".format(run_id), b_summary)
        _write_csv(PAPER_DIR / "formalModuleC_{}.csv".format(run_id), c_summary)
        _write_csv(PAPER_DIR / "formalEnd2End_{}.csv".format(run_id), e_summary)
        plotted = _run_isolated_plotting(json_path)
        print("[ISOLATED PLOTTING] returncode={}".format(plotted.returncode))
        print(plotted.stdout[-4000:])
        if plotted.returncode != 0:
            raise RuntimeError("独立绘图失败，正式数值已保存: {}".format(json_path))
        print("\nFORMAL_COMPLETE")
        print("json:", json_path)
        print("csv :", csv_path)
        print("txt :", txt_path)
        print("runtime_s:", payload["total_runtime_s"])
    finally:
        sys.stdout = tee.console
        tee.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

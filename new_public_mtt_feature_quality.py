#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune and formally verify Public-MTT frozen 2048-D linear-probe quality."""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from new_public_mtt_mainline_common import ROOT, load_config, load_feature_cache
from new_secure_feature_models import atomic_torch_save


OUT = ROOT / "new_log/new_public_mtt_mainline"
TUNING_JSON = OUT / "json/new_feature_quality_tuning.json"
FORMAL_JSON = OUT / "json/new_feature_quality_formal.json"
CHECKPOINT_ROOT = (
    ROOT / "new_cache/new_public_mtt_mainline/new_feature_quality"
)


def atomic_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new_tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(model, x, y, device, batch_size=512) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(y), batch_size):
        xb = x[start : start + batch_size].to(device)
        pred = model(xb).argmax(dim=1).cpu()
        correct += int((pred == y[start : start + batch_size]).sum())
    return correct / max(len(y), 1)


def build_optimizer(model, candidate):
    if candidate["optimizer"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        momentum=0.9,
        weight_decay=float(candidate["weight_decay"]),
    )


def train_probe(
    dataset: str,
    seed: int,
    data: Dict,
    candidate: Dict,
    steps: int,
    batch_size: int,
    device: torch.device,
    checkpoint_path: Path,
    formal: bool,
) -> Dict:
    set_seed(seed)
    num_classes = {
        "cifar10": 10,
        "cifar100": 100,
        "tiny": 200,
    }[dataset]
    model = nn.Linear(2048, num_classes).to(device)
    optimizer = build_optimizer(model, candidate)
    criterion = nn.CrossEntropyLoss()
    if formal:
        x_train = torch.cat(
            [
                data["x_encoder_train"],
                data["x_reference"],
                data["x_selection"],
            ]
        )
        y_train = torch.cat(
            [
                data["y_encoder_train"],
                data["y_reference"],
                data["y_selection"],
            ]
        )
    else:
        x_train = data["x_encoder_train"]
        y_train = data["y_encoder_train"]
    start_step = 0
    history: List[Dict] = []
    generator = torch.Generator()
    generator.manual_seed(seed)
    if checkpoint_path.exists():
        state = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False
        )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_step = int(state["next_step"])
        history = list(state["history"])
        generator.set_state(state["batch_rng_state"].cpu())
    eval_steps = set(
        np.linspace(1, steps, min(20, steps), dtype=np.int64).tolist()
    )
    started = time.time()
    model.train()
    for step in range(start_step, steps):
        ids = torch.randint(
            0, len(y_train), (batch_size,), generator=generator
        )
        xb = x_train[ids].to(device)
        yb = y_train[ids].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        if step + 1 in eval_steps:
            row = {
                "step": step + 1,
                "train_batch_loss": float(loss.detach().cpu()),
            }
            if formal:
                row["selection_accuracy"] = accuracy(
                    model,
                    data["x_selection"],
                    data["y_selection"],
                    device,
                )
            else:
                row["reference_accuracy"] = accuracy(
                    model,
                    data["x_reference"],
                    data["y_reference"],
                    device,
                )
            history.append(row)
        if (step + 1) % 100 == 0 or step + 1 == steps:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "next_step": step + 1,
                    "history": history,
                    "batch_rng_state": generator.get_state(),
                },
                checkpoint_path,
            )
    result = {
        "dataset": dataset,
        "seed": seed,
        "candidate": candidate,
        "steps": steps,
        "batch_size": batch_size,
        "history": history,
        "runtime_seconds": time.time() - started,
        "encoder_sha256": data["encoder_sha256"],
        "scaler_sha256": data["scaler_sha256"],
        "test_loaded_during_tuning": False,
    }
    if formal:
        result["selection_accuracy"] = accuracy(
            model, data["x_selection"], data["y_selection"], device
        )
        result["test_accuracy"] = accuracy(
            model, data["x_test"], data["y_test"], device
        )
        result["test_usage"] = "single_final_readonly_after_config_freeze"
    else:
        result["reference_accuracy"] = accuracy(
            model, data["x_reference"], data["y_reference"], device
        )
    return result


def run_tuning(config, device) -> Dict:
    trials = []
    probe = config["linear_probe"]
    total = (
        len(config["datasets"])
        * len(config["tuning_seeds"])
        * len(probe["candidates"])
    )
    print("[FeatureQuality:Tuning] resolved_trials={}".format(total), flush=True)
    completed = 0
    for dataset in config["datasets"]:
        for seed in config["tuning_seeds"]:
            data = load_feature_cache(dataset, seed, allow_test=False)
            for candidate_id, candidate in enumerate(probe["candidates"]):
                checkpoint = (
                    CHECKPOINT_ROOT
                    / "new_tuning"
                    / dataset
                    / "new_seed{}_candidate{}.pt".format(seed, candidate_id)
                )
                result = train_probe(
                    dataset,
                    seed,
                    data,
                    candidate,
                    int(probe["tuning_steps"][dataset]),
                    int(probe["batch_size"][dataset]),
                    device,
                    checkpoint,
                    formal=False,
                )
                result["candidate_id"] = candidate_id
                trials.append(result)
                completed += 1
                print(
                    "  [{}/{}] {} seed={} candidate={} ref_acc={:.4f}".format(
                        completed,
                        total,
                        dataset,
                        seed,
                        candidate_id,
                        result["reference_accuracy"],
                    ),
                    flush=True,
                )
                atomic_json(
                    {
                        "schema": "new_feature_quality_tuning_v1",
                        "status": "RUNNING",
                        "test_loaded": False,
                        "trials": trials,
                    },
                    TUNING_JSON,
                )
    frozen = {}
    for dataset in config["datasets"]:
        rows = [row for row in trials if row["dataset"] == dataset]
        scores = {}
        for candidate_id in range(len(probe["candidates"])):
            values = [
                row["reference_accuracy"]
                for row in rows
                if row["candidate_id"] == candidate_id
            ]
            scores[candidate_id] = float(np.mean(values))
        best_id = max(scores, key=scores.get)
        frozen[dataset] = {
            "candidate_id": best_id,
            "candidate": probe["candidates"][best_id],
            "mean_reference_accuracy": scores[best_id],
            "selection_source": "reference_only_tuning_seeds_42_43",
        }
    payload = {
        "schema": "new_feature_quality_tuning_v1",
        "status": "COMPLETE",
        "test_loaded": False,
        "trials": trials,
        "frozen": frozen,
    }
    atomic_json(payload, TUNING_JSON)
    return payload


def run_formal(config, tuning, device) -> Dict:
    probe = config["linear_probe"]
    rows = []
    total = len(config["datasets"]) * len(config["formal_seeds"])
    print("[FeatureQuality:Formal] resolved_runs={}".format(total), flush=True)
    completed = 0
    for dataset in config["datasets"]:
        frozen = tuning["frozen"][dataset]
        for seed in config["formal_seeds"]:
            data = load_feature_cache(dataset, seed, allow_test=True)
            checkpoint = (
                CHECKPOINT_ROOT
                / "new_formal"
                / dataset
                / "new_seed{}.pt".format(seed)
            )
            result = train_probe(
                dataset,
                seed,
                data,
                frozen["candidate"],
                int(probe["formal_steps"][dataset]),
                int(probe["batch_size"][dataset]),
                device,
                checkpoint,
                formal=True,
            )
            rows.append(result)
            completed += 1
            print(
                "  [{}/{}] {} seed={} selection={:.4f} test={:.4f}".format(
                    completed,
                    total,
                    dataset,
                    seed,
                    result["selection_accuracy"],
                    result["test_accuracy"],
                ),
                flush=True,
            )
            atomic_json(
                {
                    "schema": "new_feature_quality_formal_v1",
                    "status": "RUNNING",
                    "frozen_config": tuning["frozen"],
                    "runs": rows,
                },
                FORMAL_JSON,
            )
    summary = {}
    passed = True
    for dataset in config["datasets"]:
        values = [
            row["test_accuracy"] for row in rows if row["dataset"] == dataset
        ]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        threshold = (
            float(config["quality_reference_means"][dataset])
            - float(config["quality_max_absolute_gap"])
        )
        gate = mean >= threshold
        passed = passed and gate
        summary[dataset] = {
            "mean_test_accuracy": mean,
            "sample_std": std,
            "required_accuracy": threshold,
            "quality_gate_pass": gate,
        }
    payload = {
        "schema": "new_feature_quality_formal_v1",
        "status": "QUALITY_GATE_PASS" if passed else "QUALITY_GATE_FAIL",
        "stage_a_track": "public_mtt_benchmark",
        "stage_a_epsilon": "N/A",
        "stage_a_dp_protected": False,
        "frozen_config": tuning["frozen"],
        "runs": rows,
        "summary": summary,
    }
    atomic_json(payload, FORMAL_JSON)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Public MTT frozen-feature linear quality gate"
    )
    parser.add_argument(
        "--phase", choices=("tune", "formal", "all"), default="all"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    config = load_config()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    print("=" * 88, flush=True)
    print("Public MTT mainline: frozen 2048-D feature quality gate", flush=True)
    print(
        "datasets={} tuning_seeds={} formal_seeds={}".format(
            config["datasets"],
            config["tuning_seeds"],
            config["formal_seeds"],
        ),
        flush=True,
    )
    print(
        "cpu_threads=1 num_workers=0 device={} test_tuning=False".format(
            device
        ),
        flush=True,
    )
    print("tuning_output={}".format(TUNING_JSON), flush=True)
    print("formal_output={}".format(FORMAL_JSON), flush=True)
    print("=" * 88, flush=True)
    tuning = None
    if args.phase in ("tune", "all"):
        tuning = run_tuning(config, device)
    else:
        tuning = json.loads(TUNING_JSON.read_text(encoding="utf-8"))
    if args.phase in ("formal", "all"):
        if tuning["status"] != "COMPLETE":
            raise RuntimeError("tuning configuration is not frozen")
        formal = run_formal(config, tuning, device)
        print(json.dumps(formal["summary"], ensure_ascii=False, indent=2))
        return 0 if formal["status"] == "QUALITY_GATE_PASS" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

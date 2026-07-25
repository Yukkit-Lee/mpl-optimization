#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one formal Module-A Full/Random/Public-MTT efficiency comparison."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from new_public_mtt_mainline_common import ROOT, load_config, load_feature_cache
from new_secure_feature_models import atomic_torch_save, configure_single_thread_runtime


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new_tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def auc(history):
    x = np.asarray([row["step"] for row in history], dtype=np.float64)
    y = np.asarray([row["test_accuracy"] for row in history], dtype=np.float64)
    if len(x) < 2:
        return float(y[-1])
    return float(np.trapz(y, x) / max(x[-1] - x[0], 1.0))


@torch.no_grad()
def evaluate(model, x, y, device):
    model.eval()
    correct = 0
    for start in range(0, len(y), 512):
        xb = x[start : start + 512].to(device)
        correct += int(
            (model(xb).argmax(1).cpu() == y[start : start + 512]).sum()
        )
    return correct / len(y)


def stratified_random_subset(labels, count, seed):
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    per_class = count // len(classes)
    remainder = count % len(classes)
    selected = []
    for position, class_id in enumerate(classes):
        ids = np.flatnonzero(labels == class_id)
        rng.shuffle(ids)
        take = per_class + (1 if position < remainder else 0)
        selected.extend(ids[:take].tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def source_batch(x, y, ids, device):
    if isinstance(x, np.ndarray):
        xb = torch.from_numpy(np.asarray(x[ids.numpy()]).copy())
        yb = torch.from_numpy(np.asarray(y[ids.numpy()]).copy()).long()
    else:
        xb = x[ids]
        yb = y[ids]
    return xb.to(device), yb.to(device)


def train_source(
    dataset,
    source,
    x_train,
    y_train,
    x_test,
    y_test,
    candidate,
    steps,
    batch_size,
    target,
    seed,
    device,
    checkpoint,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_classes = {"cifar10": 10, "cifar100": 100, "tiny": 200}[dataset]
    model = nn.Linear(2048, num_classes).to(device)
    if candidate["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(candidate["learning_rate"]),
            momentum=0.9,
            weight_decay=float(candidate["weight_decay"]),
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    history = []
    start_step = 0
    elapsed_offset = 0.0
    if checkpoint.exists():
        state = torch.load(
            str(checkpoint), map_location=device, weights_only=False
        )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        generator.set_state(state["batch_rng_state"].cpu())
        history = list(state["history"])
        start_step = int(state["next_step"])
        elapsed_offset = float(state.get("elapsed_seconds", 0.0))
    criterion = nn.CrossEntropyLoss()
    eval_steps = set(
        np.linspace(0, steps, 21, dtype=np.int64).tolist()
    )
    if start_step == 0 and not history:
        history.append(
            {
                "step": 0,
                "test_accuracy": evaluate(model, x_test, y_test, device),
                "elapsed_seconds": 0.0,
                "processed_examples": 0,
            }
        )
    started = time.time()
    for step in range(start_step, steps):
        ids = torch.randint(
            0, len(y_train), (batch_size,), generator=generator
        )
        xb, yb = source_batch(x_train, y_train, ids, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        if step + 1 in eval_steps:
            history.append(
                {
                    "step": step + 1,
                    "test_accuracy": evaluate(
                        model, x_test, y_test, device
                    ),
                    "elapsed_seconds": elapsed_offset + time.time() - started,
                    "processed_examples": (step + 1) * batch_size,
                }
            )
        if (step + 1) % 100 == 0 or step + 1 == steps:
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "batch_rng_state": generator.get_state(),
                    "history": history,
                    "next_step": step + 1,
                    "elapsed_seconds": elapsed_offset + time.time() - started,
                },
                checkpoint,
            )
    crossing = next(
        (row for row in history if row["test_accuracy"] >= target), None
    )
    return {
        "source": source,
        "num_train_samples": int(len(y_train)),
        "steps": steps,
        "batch_size": batch_size,
        "target_accuracy": target,
        "final_accuracy": history[-1]["test_accuracy"],
        "step_at_target": crossing["step"] if crossing else None,
        "time_at_target_seconds": (
            crossing["elapsed_seconds"] if crossing else None
        ),
        "processed_examples_at_target": (
            crossing["processed_examples"] if crossing else None
        ),
        "early_auc": auc(history),
        "runtime_seconds": elapsed_offset + time.time() - started,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Formal Module-A efficiency run")
    parser.add_argument(
        "--dataset", choices=("cifar10", "cifar100", "tiny"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    configure_single_thread_runtime(args.seed)
    config = load_config()
    device = torch.device(args.device)
    formal = json.loads(
        (
            ROOT
            / "new_log/new_public_mtt_mainline/json"
            / "new_feature_quality_formal.json"
        ).read_text(encoding="utf-8")
    )
    if formal["status"] != "QUALITY_GATE_PASS":
        raise RuntimeError("feature quality gate has not passed")
    candidate = formal["frozen_config"][args.dataset]["candidate"]
    official = load_feature_cache(args.dataset, args.seed, allow_test=True)
    full_dir = (
        ROOT
        / "new_cache/new_public_mtt_mainline/new_full_features"
        / args.dataset
        / "new_seed{}".format(args.seed)
    )
    manifest = json.loads(
        (full_dir / "new_full_feature_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    x_full = np.load(manifest["feature_path"], mmap_mode="r")
    y_full = np.load(manifest["label_path"], mmap_mode="r")
    official_count = len(official["y_encoder_train"])
    random_ids = stratified_random_subset(
        np.asarray(y_full), official_count, args.seed
    )
    sources = {
        "full_images": (x_full, y_full),
        "random_subset": (x_full[random_ids], y_full[random_ids]),
        "official_mtt": (
            official["x_encoder_train"],
            official["y_encoder_train"],
        ),
    }
    cfg = config["module_a_efficiency"]
    output_dir = (
        ROOT
        / "new_log/new_public_mtt_mainline/json/new_module_a_runs"
        / args.dataset
    )
    results = []
    print("=" * 88)
    print("Module A formal efficiency: Full vs Random vs Official MTT")
    print(
        "dataset={} seed={} steps={} target={} cpu_threads=1 num_workers=0".format(
            args.dataset,
            args.seed,
            cfg["steps"][args.dataset],
            cfg["target_accuracy"][args.dataset],
        )
    )
    print("=" * 88)
    for source, (x_train, y_train) in sources.items():
        checkpoint = (
            ROOT
            / "new_cache/new_public_mtt_mainline/new_module_a_checkpoints"
            / args.dataset
            / "new_seed{}_{}.pt".format(args.seed, source)
        )
        result = train_source(
            args.dataset,
            source,
            x_train,
            y_train,
            official["x_test"],
            official["y_test"],
            candidate,
            int(cfg["steps"][args.dataset]),
            int(cfg["batch_size"][args.dataset]),
            float(cfg["target_accuracy"][args.dataset]),
            args.seed,
            device,
            checkpoint,
        )
        result["compression_ratio"] = float(
            result["num_train_samples"] / len(y_full)
        )
        results.append(result)
        print(
            "  {} samples={} final={:.4f} step@target={}".format(
                source,
                result["num_train_samples"],
                result["final_accuracy"],
                result["step_at_target"],
            ),
            flush=True,
        )
    payload = {
        "schema": "new_module_a_efficiency_run_v1",
        "status": "COMPLETE",
        "dataset": args.dataset,
        "seed": args.seed,
        "stage_a_track": "public_mtt_benchmark",
        "stage_a_epsilon": "N/A",
        "stage_a_dp_protected": False,
        "encoder_sha256": official["encoder_sha256"],
        "scaler_sha256": official["scaler_sha256"],
        "optimizer": candidate,
        "test_usage": "fixed_interval_readonly_formal_reporting",
        "results": results,
    }
    output = output_dir / "new_module_a_efficiency_{}_seed{}.json".format(
        args.dataset, args.seed
    )
    atomic_json(payload, output)
    print("output={}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

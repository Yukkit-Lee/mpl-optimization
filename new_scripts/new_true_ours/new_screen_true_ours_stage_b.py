#!/usr/bin/env python3
"""One-factor plaintext screen for the MTT-only secure-average Stage B.

This is a development-validation diagnostic.  Finalists must be rerun through
TFE/ABY3 before they can become formal results.
"""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
LOG_ROOT = ROOT / "new_log/new_true_ours"
TABLE_ROOT = ROOT / "new_expTab/new_true_ours"
OUTPUT_ROOT = ROOT / "new_output/new_true_ours/stage_b_screen"
DATASETS = ("cifar10", "cifar100", "tiny")
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "tiny": 200}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def one_hot(labels, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    values = np.zeros((len(labels), num_classes), dtype=np.float32)
    values[np.arange(len(labels)), labels] = 1.0
    return values


def make_splits(labels, num_clients, seed):
    buckets = [[] for _ in range(num_clients)]
    for class_id in np.unique(labels):
        ids = np.flatnonzero(labels == class_id)
        rng = np.random.RandomState(seed + int(class_id) * 1009 + 728)
        for client_id, part in enumerate(np.array_split(rng.permutation(ids), num_clients)):
            buckets[client_id].extend(part.tolist())
    return [
        np.random.RandomState(seed + client_id * 7919).permutation(
            np.asarray(bucket, dtype=np.int64)
        )
        for client_id, bucket in enumerate(buckets)
    ]


def deterministic_batches(indices, batch_size, seed, epoch, client_id):
    rng = np.random.RandomState(seed + epoch * 1009 + client_id * 97)
    shuffled = rng.permutation(indices)
    count = int(np.ceil(float(len(shuffled)) / float(batch_size)))
    needed = count * batch_size
    if needed > len(shuffled):
        shuffled = np.concatenate(
            [shuffled, np.resize(shuffled, needed - len(shuffled))]
        )
    return [
        shuffled[offset : offset + batch_size]
        for offset in range(0, len(shuffled), batch_size)
    ]


def aggregation_weights(labels, client_indices, mode):
    if mode == "uniform":
        values = np.ones(len(client_indices), dtype=np.float64)
    elif mode == "sample_count":
        values = np.asarray([len(ids) for ids in client_indices], dtype=np.float64)
    elif mode == "class_balanced":
        global_counts = np.bincount(labels)
        values = []
        for ids in client_indices:
            values.append(float(np.sum(1.0 / global_counts[labels[ids]])))
        values = np.asarray(values, dtype=np.float64)
    else:
        raise ValueError(mode)
    return values / values.sum()


def accuracy(weight, bias, x_value, y_value):
    predictions = np.argmax(np.matmul(x_value, weight) + bias, axis=1)
    return float(np.mean(predictions == y_value)), predictions


def expected_calibration_error(weight, bias, x_value, y_value, bins=10):
    logits = np.matmul(x_value, weight) + bias
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_value
    result = 0.0
    for low in np.linspace(0.0, 0.9, bins):
        high = low + 0.1
        selected = (confidence > low) & (confidence <= high)
        if np.any(selected):
            result += float(np.mean(selected)) * abs(
                float(np.mean(confidence[selected]))
                - float(np.mean(correct[selected]))
            )
    return result


def train_once(x_train, y_train, x_dev, y_dev, dataset, seed, config):
    started = time.time()
    num_clients = int(config.get("num_clients", 5))
    num_classes = NUM_CLASSES[dataset]
    client_indices = make_splits(y_train, num_clients, seed)
    rng = np.random.RandomState(seed)
    weights = [
        rng.normal(0.0, 0.01, size=(x_train.shape[1], num_classes)).astype(np.float32)
        for _ in range(num_clients)
    ]
    biases = [np.zeros(num_classes, dtype=np.float32) for _ in range(num_clients)]
    if config["shared_initialization"]:
        weights = [weights[0].copy() for _ in weights]
        biases = [biases[0].copy() for _ in biases]
    y_hot = one_hot(y_train, num_classes)
    epoch_history = []
    for epoch in range(config["local_epochs"]):
        epoch_loss = 0.0
        epoch_examples = 0
        for client_id, ids in enumerate(client_indices):
            for batch in deterministic_batches(
                ids, config["batch_size"], seed, epoch, client_id
            ):
                logits = np.matmul(x_train[batch], weights[client_id]) + biases[client_id]
                error = logits - y_hot[batch]
                grad_w = np.matmul(x_train[batch].T, error) / float(len(batch))
                grad_w += config["weight_decay"] * weights[client_id]
                grad_b = np.mean(error, axis=0)
                weights[client_id] -= config["local_lr"] * grad_w
                biases[client_id] -= config["local_lr"] * grad_b
                epoch_loss += float(np.sum(error * error))
                epoch_examples += len(batch) * num_classes
        if (
            config["fedavg_sync_interval"] > 0
            and (epoch + 1) % config["fedavg_sync_interval"] == 0
        ):
            coefficients = aggregation_weights(
                y_train, client_indices, config["client_weighting"]
            )
            avg_weight = sum(
                coefficient * weight
                for coefficient, weight in zip(coefficients, weights)
            )
            avg_bias = sum(
                coefficient * bias
                for coefficient, bias in zip(coefficients, biases)
            )
            weights = [avg_weight.copy() for _ in weights]
            biases = [avg_bias.copy() for _ in biases]
        if epoch + 1 in (1, config["local_epochs"]):
            coefficients = aggregation_weights(
                y_train, client_indices, config["client_weighting"]
            )
            avg_w = sum(
                coefficient * weight
                for coefficient, weight in zip(coefficients, weights)
            )
            avg_b = sum(
                coefficient * bias
                for coefficient, bias in zip(coefficients, biases)
            )
            dev_acc, _ = accuracy(avg_w, avg_b, x_dev, y_dev)
            epoch_history.append(
                {
                    "epoch": epoch + 1,
                    "train_mse": epoch_loss / float(epoch_examples),
                    "development_validation_accuracy": dev_acc,
                }
            )

    coefficients = aggregation_weights(
        y_train, client_indices, config["client_weighting"]
    )
    avg_w = sum(
        coefficient * weight
        for coefficient, weight in zip(coefficients, weights)
    ).astype(np.float32)
    avg_b = sum(
        coefficient * bias
        for coefficient, bias in zip(coefficients, biases)
    ).astype(np.float32)
    dev_acc, predictions = accuracy(avg_w, avg_b, x_dev, y_dev)
    local_dev = [accuracy(w, b, x_dev, y_dev)[0] for w, b in zip(weights, biases)]
    distances = [
        float(np.sqrt(np.sum((weight - avg_w) ** 2) + np.sum((bias - avg_b) ** 2)))
        for weight, bias in zip(weights, biases)
    ]
    class_accuracy = {}
    for class_id in range(num_classes):
        selected = y_dev == class_id
        class_accuracy[str(class_id)] = float(
            np.mean(predictions[selected] == y_dev[selected])
        )
    return {
        "development_validation_accuracy": dev_acc,
        "local_development_validation_accuracy": local_dev,
        "inter_client_distance_to_average": distances,
        "class_accuracy": class_accuracy,
        "calibration_ece": expected_calibration_error(
            avg_w, avg_b, x_dev, y_dev
        ),
        "runtime_seconds": time.time() - started,
        "history": epoch_history,
        "weight": avg_w,
        "bias": avg_b,
        "client_counts": [int(len(ids)) for ids in client_indices],
    }


def load_assets(dataset, seed):
    cache_path = (
        ROOT / "new_cache/new_true_ours/official_mtt/stage_b" / dataset
        / "new_true_ours_stage_b_{}_seed{}.npz".format(dataset, seed)
    )
    split_path = (
        ROOT / "new_cache/new_true_ours/splits" / dataset
        / "new_true_ours_split_{}_seed{}.npz".format(dataset, seed)
    )
    full_dir = (
        ROOT / "new_cache/new_public_mtt_mainline/new_full_features"
        / dataset / "new_seed{}".format(seed)
    )
    with np.load(str(cache_path), allow_pickle=False) as cache:
        source = str(cache["initializer_source"].reshape(-1)[0])
        dependency = int(cache["full_data_initializer_dependency"].reshape(-1)[0])
        x_train = np.asarray(cache["x_encoder_train"], dtype=np.float32)
        y_train = np.asarray(cache["y_encoder_train"], dtype=np.int64)
    if source != "official_mtt_only" or dependency != 0:
        raise ValueError("True Ours asset lineage failed")
    with np.load(str(split_path), allow_pickle=False) as split:
        dev_ids = np.asarray(split["development_validation"], dtype=np.int64)
    x_full = np.load(
        str(full_dir / "new_full_train_features.npy"), mmap_mode="r"
    )
    y_full = np.load(
        str(full_dir / "new_full_train_labels.npy"), mmap_mode="r"
    )
    x_dev = np.asarray(x_full[dev_ids], dtype=np.float32)
    y_dev = np.asarray(y_full[dev_ids], dtype=np.int64)
    return cache_path, split_path, x_train, y_train, x_dev, y_dev


def screen_dataset(dataset, seed):
    cache_path, split_path, x_train, y_train, x_dev, y_dev = load_assets(
        dataset, seed
    )
    current = {
        "shared_initialization": True,
        "local_epochs": 20,
        "batch_size": 32,
        "local_lr": 0.1,
        "weight_decay": 0.0,
        "client_weighting": "uniform",
        "fedavg_sync_interval": 0,
    }
    factors = [
        ("shared_initialization", [False, True]),
        ("local_epochs", [5, 10, 20, 40]),
        ("batch_size", [8, 16, 32]),
        ("local_lr", [0.025, 0.05, 0.1, 0.2]),
        ("weight_decay", [0.0, 1e-5, 1e-4, 1e-3]),
        ("client_weighting", ["uniform", "sample_count", "class_balanced"]),
    ]
    rows = []
    run_id = 0
    for factor, values in factors:
        candidates = []
        for value in values:
            config = dict(current)
            config[factor] = value
            result = train_once(
                x_train, y_train, x_dev, y_dev, dataset, seed, config
            )
            row = {
                "run_id": run_id,
                "dataset": dataset,
                "seed": seed,
                "factor": factor,
                "factor_value": value,
                "config": config,
                "development_validation_accuracy": result[
                    "development_validation_accuracy"
                ],
                "calibration_ece": result["calibration_ece"],
                "mean_inter_client_distance": float(
                    np.mean(result["inter_client_distance_to_average"])
                ),
                "runtime_seconds": result["runtime_seconds"],
                "local_development_validation_accuracy": result[
                    "local_development_validation_accuracy"
                ],
                "class_accuracy": result["class_accuracy"],
                "history": result["history"],
                "client_counts": result["client_counts"],
                "locked_holdout_read": False,
                "official_test_read": False,
            }
            rows.append(row)
            candidates.append((row, result))
            print(
                "{} factor={} value={} dev={:.5f} time={:.2f}s".format(
                    dataset,
                    factor,
                    value,
                    row["development_validation_accuracy"],
                    row["runtime_seconds"],
                ),
                flush=True,
            )
            run_id += 1
        chosen_row, chosen_result = max(
            candidates,
            key=lambda item: (
                item[0]["development_validation_accuracy"],
                -item[0]["calibration_ece"],
                -item[0]["runtime_seconds"],
            ),
        )
        current = dict(chosen_row["config"])
        print(
            "{} KEEP {}={} dev={:.5f}".format(
                dataset,
                factor,
                current[factor],
                chosen_row["development_validation_accuracy"],
            ),
            flush=True,
        )

    final_result = train_once(
        x_train, y_train, x_dev, y_dev, dataset, seed, current
    )
    output_path = (
        OUTPUT_ROOT / dataset
        / "new_true_ours_stage_b_screen_best_{}_seed{}.npz".format(dataset, seed)
    )
    atomic_npz(
        output_path,
        weight=final_result["weight"],
        bias=final_result["bias"],
        initializer_source=np.asarray(["official_mtt_only"]),
        full_data_initializer_dependency=np.asarray([0], dtype=np.int64),
        formal_secure_result=np.asarray([0], dtype=np.int64),
        feature_cache_sha256=np.asarray([sha256_file(cache_path)]),
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "feature_cache": str(cache_path),
        "feature_cache_sha256": sha256_file(cache_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "best_config": current,
        "best_development_validation_accuracy": final_result[
            "development_validation_accuracy"
        ],
        "best_calibration_ece": final_result["calibration_ece"],
        "best_screen_initializer": str(output_path),
        "formal_secure_result": False,
        "locked_holdout_read": False,
        "official_test_read": False,
        "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    json_path = LOG_ROOT / "new_true_ours_stage_b_screen.json"
    existing_results = []
    if json_path.is_file():
        existing_payload = json.loads(json_path.read_text(encoding="utf-8"))
        existing_results = [
            result
            for result in existing_payload.get("results", [])
            if not (
                result["dataset"] in args.datasets
                and int(result["seed"]) == int(args.seed)
            )
        ]
    new_results = [
        screen_dataset(dataset, args.seed) for dataset in args.datasets
    ]
    results = existing_results + new_results
    payload = {
        "schema": "new_true_ours_stage_b_screen_v1",
        "status": "COMPLETE",
        "backend": "numpy_plaintext_diagnostic",
        "not_formal_secure_result": True,
        "one_factor_greedy_screen": True,
        "locked_holdout_read": False,
        "official_test_read": False,
        "results": results,
    }
    atomic_json(json_path, payload)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_ROOT / "new_true_ours_stage_b_screen.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "dataset",
                "seed",
                "factor",
                "factor_value",
                "development_validation_accuracy",
                "calibration_ece",
                "mean_inter_client_distance",
                "runtime_seconds",
            ]
        )
        for result in results:
            for row in result["runs"]:
                writer.writerow(
                    [
                        row["dataset"],
                        row["seed"],
                        row["factor"],
                        row["factor_value"],
                        "{:.8f}".format(row["development_validation_accuracy"]),
                        "{:.8f}".format(row["calibration_ece"]),
                        "{:.8f}".format(row["mean_inter_client_distance"]),
                        "{:.6f}".format(row["runtime_seconds"]),
                    ]
                )
    print("screen={} rows={} locked_holdout_read=False".format(
        csv_path, sum(len(result["runs"]) for result in results)
    ))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Development-only Phase-5 screen for regularized MTT consensus.

The screen rebuilds client heads from Official-MTT features.  It never loads a
Full-Data initializer and never reads locked holdout or official test.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from new_stage_b_secure_init import (
    aggregation_weights,
    initialize_heads,
    load_feature_cache,
    make_client_splits,
    mse_loss,
    one_hot,
    plaintext_reference,
)
from new_screen_true_ours_stage_b import (
    DATASETS,
    LOG_ROOT,
    ROOT,
    TABLE_ROOT,
    accuracy,
    atomic_json,
    load_assets,
)


def capped_simplex(weights, cap):
    """Project positive normalized weights onto a capped simplex."""
    values = np.asarray(weights, dtype=np.float64)
    result = np.zeros_like(values)
    remaining = np.ones(len(values), dtype=bool)
    mass = 1.0
    while np.any(remaining):
        proposal = values[remaining] / values[remaining].sum() * mass
        over = proposal > float(cap)
        if not np.any(over):
            result[remaining] = proposal
            break
        ids = np.flatnonzero(remaining)
        result[ids[over]] = float(cap)
        remaining[ids[over]] = False
        mass = 1.0 - result.sum()
    return result / result.sum()


def aggregate(weights, biases, coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    weight = np.sum(
        np.stack(weights) * coefficients[:, None, None], axis=0
    ).astype(np.float32)
    bias = np.sum(
        np.stack(biases) * coefficients[:, None], axis=0
    ).astype(np.float32)
    return weight, bias


def train_clients(dataset, seed, split_name, config):
    cache_path, _, x_train, y_train, x_dev, y_dev = load_assets(dataset, seed)
    data = load_feature_cache(str(cache_path))
    client_indices = make_client_splits(y_train, 5, split_name, seed)
    weights, biases = initialize_heads(
        5, x_train.shape[1], int(y_train.max()) + 1, seed, 0.01
    )
    if config.get("shared_initialization", False):
        weights = [weights[0].copy() for _ in weights]
        biases = [biases[0].copy() for _ in biases]
    coefficients = aggregation_weights(
        y_train, client_indices, config["client_weighting"]
    )
    args = SimpleNamespace(
        num_classes=int(y_train.max()) + 1,
        local_epochs=int(config["local_epochs"]),
        batch_size=int(config["batch_size"]),
        seed=int(seed),
        weight_decay=float(config["weight_decay"]),
        local_lr=float(config["local_lr"]),
        fedavg_sync_interval=int(config.get("fedavg_sync_interval", 0)),
        client_aggregation_weights=coefficients.tolist(),
        distill_epochs=0,
        lambda_kd=0.0,
        reference_supervised_weight=0.0,
        distill_lr=float(config["local_lr"]),
        weight_temperature=0.2,
    )
    result = plaintext_reference(
        x_train,
        y_train,
        data["x_reference"],
        data["y_reference"],
        data["x_selection"],
        data["y_selection"],
        client_indices,
        weights,
        biases,
        args,
    )
    return data, y_train, x_dev, y_dev, client_indices, result


def candidate_rows(dataset, seed, split_name, config):
    data, y_train, x_dev, y_dev, client_indices, result = train_clients(
        dataset, seed, split_name, config
    )
    weights = result["nokd_weights"]
    biases = result["nokd_biases"]
    avg_coeff = aggregation_weights(
        y_train, client_indices, config["client_weighting"]
    )
    class_coeff = aggregation_weights(y_train, client_indices, "class_balanced")
    losses = np.asarray(
        [
            mse_loss(
                np.matmul(data["x_reference"], weight) + bias,
                one_hot(data["y_reference"], bias.shape[0]),
            )
            for weight, bias in zip(weights, biases)
        ]
    )
    base_temperature = 0.2
    raw = np.exp(
        -(losses - losses.min()) / base_temperature
    )
    consensus_coeff = raw / raw.sum()
    specifications = [
        ("average", {}, avg_coeff),
        ("class_balanced", {}, class_coeff),
        ("original_consensus", {"temperature": 1.0}, consensus_coeff),
    ]
    for multiplier in (2.0, 4.0, 8.0):
        smooth = np.exp(
            -(losses - losses.min()) / (base_temperature * multiplier)
        )
        specifications.append(
            (
                "temperature_consensus",
                {"temperature": multiplier},
                smooth / smooth.sum(),
            )
        )
    for cap in (0.30, 0.40, 0.50):
        specifications.append(
            ("capped_consensus", {"cap": cap}, capped_simplex(consensus_coeff, cap))
        )
    for top_k in (2, 3):
        keep = np.argsort(consensus_coeff)[-top_k:]
        top = np.zeros_like(consensus_coeff)
        top[keep] = consensus_coeff[keep]
        specifications.append(("topk_consensus", {"top_k": top_k}, top / top.sum()))

    avg_w, avg_b = aggregate(weights, biases, avg_coeff)
    con_w, con_b = aggregate(weights, biases, consensus_coeff)
    for beta in (0.05, 0.10, 0.25, 0.50):
        specifications.append(
            (
                "shrinkage_consensus",
                {"beta": beta},
                None,
            )
        )

    rows = []
    for method, params, coefficients in specifications:
        if method == "shrinkage_consensus":
            beta = params["beta"]
            weight = (1.0 - beta) * avg_w + beta * con_w
            bias = (1.0 - beta) * avg_b + beta * con_b
        else:
            weight, bias = aggregate(weights, biases, coefficients)
        dev_accuracy, _ = accuracy(weight, bias, x_dev, y_dev)
        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "partition": split_name,
                "method": method,
                "parameters": params,
                "development_validation_accuracy": dev_accuracy,
                "reference_losses": losses.tolist(),
                "locked_holdout_read": False,
                "official_test_read": False,
                "full_data_initializer_dependency": 0,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rescue = json.loads(
        (LOG_ROOT / "new_true_ours_stage_b_rescue_screen.json").read_text()
    )
    configs = {
        (row["dataset"], int(row["seed"])): row["best_config"]
        for row in rescue["results"]
    }
    rows = []
    for dataset in args.datasets:
        for partition in ("balanced", "severe"):
            rows.extend(
                candidate_rows(
                    dataset,
                    args.seed,
                    partition,
                    configs[(dataset, args.seed)],
                )
            )
    decisions = []
    for dataset in args.datasets:
        dataset_rows = [
            row
            for row in rows
            if row["dataset"] == dataset and row["seed"] == int(args.seed)
        ]
        average = {
            row["partition"]: row["development_validation_accuracy"]
            for row in dataset_rows
            if row["method"] == "average"
        }
        candidates = []
        identities = sorted(
            {
                (
                    row["method"],
                    json.dumps(row["parameters"], sort_keys=True),
                )
                for row in dataset_rows
                if row["method"] != "average"
            }
        )
        for method, encoded_parameters in identities:
            parameters = json.loads(encoded_parameters)
            matched = {
                row["partition"]: row["development_validation_accuracy"]
                for row in dataset_rows
                if row["method"] == method
                and row["parameters"] == parameters
            }
            if set(matched) != {"balanced", "severe"}:
                raise ValueError(
                    "candidate partition pairing incomplete: {} {}".format(
                        method, parameters
                    )
                )
            iid_delta = matched["balanced"] - average["balanced"]
            severe_delta = matched["severe"] - average["severe"]
            candidates.append(
                {
                    "method": method,
                    "parameters": parameters,
                    "balanced_accuracy": matched["balanced"],
                    "severe_accuracy": matched["severe"],
                    "iid_delta_vs_average": iid_delta,
                    "severe_delta_vs_average": severe_delta,
                    "passes_seed42_gate": (
                        iid_delta >= -0.005 and severe_delta >= 0.01
                    ),
                }
            )
        passing = [
            candidate
            for candidate in candidates
            if candidate["passes_seed42_gate"]
        ]
        selected = (
            max(
                passing,
                key=lambda candidate: (
                    candidate["severe_delta_vs_average"],
                    candidate["iid_delta_vs_average"],
                ),
            )
            if passing
            else None
        )
        decisions.append(
            {
                "dataset": dataset,
                "seed": int(args.seed),
                "average_balanced_accuracy": average["balanced"],
                "average_severe_accuracy": average["severe"],
                "gate": {
                    "minimum_iid_delta": -0.005,
                    "minimum_severe_delta": 0.01,
                },
                "candidates": candidates,
                "selected_candidate": selected,
                "expand_to_three_seeds": selected is not None,
            }
        )
    payload = {
        "schema": "new_regularized_consensus_development_screen_v1",
        "status": "COMPLETE",
        "backend": "numpy_plaintext_diagnostic",
        "rows": rows,
        "decisions": decisions,
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    output = LOG_ROOT / "new_regularized_consensus_development_screen.json"
    atomic_json(output, payload)
    print("output={} rows={}".format(output, len(rows)))


if __name__ == "__main__":
    main()

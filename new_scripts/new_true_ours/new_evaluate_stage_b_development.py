#!/usr/bin/env python3
"""Evaluate a Stage-B candidate on development validation only."""

import argparse
import json
from pathlib import Path

import numpy as np

from new_screen_true_ours_stage_b import (
    DATASETS,
    LOG_ROOT,
    atomic_json,
    load_assets,
    sha256_file,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    candidate_path = Path(args.candidate).resolve()
    model = np.load(str(candidate_path), allow_pickle=False)
    if "initializer_source" not in model.files:
        raise ValueError("candidate lacks initializer_source lineage")
    initializer_source = str(model["initializer_source"].reshape(-1)[0])
    dependency = int(
        model["full_data_initializer_dependency"].reshape(-1)[0]
    )
    if initializer_source != "official_mtt_only" or dependency != 0:
        raise ValueError(
            "True Ours candidate lineage rejected: source={} dependency={}".format(
                initializer_source, dependency
            )
        )

    cache_path, split_path, _, _, x_dev, y_dev = load_assets(
        args.dataset, args.seed
    )
    weight = np.asarray(model["weight"], dtype=np.float32)
    bias = np.asarray(model["bias"], dtype=np.float32)
    prediction = np.argmax(np.matmul(x_dev, weight) + bias, axis=1)
    class_accuracy = {}
    for class_id in np.unique(y_dev):
        selected = y_dev == class_id
        class_accuracy[str(int(class_id))] = float(
            np.mean(prediction[selected] == y_dev[selected])
        )
    payload = {
        "schema": "new_true_ours_stage_b_development_evaluation_v1",
        "status": "COMPLETE",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "candidate": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "feature_cache": str(cache_path.resolve()),
        "split_manifest": str(split_path.resolve()),
        "initializer_source": initializer_source,
        "full_data_initializer_dependency": dependency,
        "development_validation_accuracy": float(
            np.mean(prediction == y_dev)
        ),
        "class_accuracy": class_accuracy,
        "development_examples": int(len(y_dev)),
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    output = (
        Path(args.output)
        if args.output
        else LOG_ROOT
        / "new_true_ours_stage_b_development_{}_seed{}.json".format(
            args.dataset, args.seed
        )
    )
    atomic_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

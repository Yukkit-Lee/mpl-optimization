#!/usr/bin/env python3
"""Recover the non-repeated Tiny rescue incumbent from its completed log rows."""

import json
import re

from new_screen_true_ours_stage_b import LOG_ROOT, atomic_json


def main():
    log_path = (
        LOG_ROOT / "txt/new_stage_b_rescue_tiny_seed42.log"
    )
    text = log_path.read_text(encoding="utf-8")
    matches = re.findall(
        r"tiny rescue factor=local_epochs value=(40|60|80) "
        r"dev=([0-9.]+) time=([0-9.]+)s",
        text,
    )
    if len(matches) != 3:
        raise RuntimeError(
            "expected three completed Tiny epoch rows, got {}".format(
                len(matches)
            )
        )
    rows = [
        {
            "local_epochs": int(epoch),
            "development_validation_accuracy": float(accuracy),
            "runtime_seconds": float(runtime),
        }
        for epoch, accuracy, runtime in matches
    ]
    best = max(
        rows,
        key=lambda row: row["development_validation_accuracy"],
    )
    output_path = LOG_ROOT / "new_true_ours_stage_b_rescue_screen.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    results = [
        row for row in payload["results"] if row["dataset"] != "tiny"
    ]
    results.append(
        {
            "dataset": "tiny",
            "seed": 42,
            "base_config": {
                "shared_initialization": False,
                "local_epochs": 40,
                "batch_size": 8,
                "local_lr": 0.2,
                "weight_decay": 1e-4,
                "client_weighting": "uniform",
                "fedavg_sync_interval": 1,
            },
            "best_config": {
                "shared_initialization": False,
                "local_epochs": best["local_epochs"],
                "batch_size": 8,
                "local_lr": 0.2,
                "weight_decay": 1e-4,
                "client_weighting": "uniform",
                "fedavg_sync_interval": 1,
            },
            "best_development_validation_accuracy": best[
                "development_validation_accuracy"
            ],
            "epoch_extension_rows": rows,
            "recovered_from_completed_log": str(log_path),
            "stopped_before_repeated_batch_candidate": True,
            "backend": "numpy_plaintext_diagnostic",
            "formal_secure_result": False,
            "locked_holdout_read": False,
            "official_test_read": False,
        }
    )
    payload["results"] = results
    payload["recovery_note"] = (
        "Tiny retained completed epoch-80 incumbent; redundant remaining "
        "one-factor rows were intentionally not rerun."
    )
    atomic_json(output_path, payload)
    print(
        "tiny_rescue_recovered dev={:.5f} epochs={}".format(
            best["development_validation_accuracy"],
            best["local_epochs"],
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the frozen three-RNG latest-initializer clipping comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_scripts.new_paper_adaptive.new_screen_latest_initializer_fixed_vs_adaptive import (
    load_inputs,
    run_candidate,
)


FREEZE = ROOT / "new_configs/new_paper_adaptive/new_adaptive_latest_initializer_formal_freeze.json"
SCREEN_FREEZE = ROOT / "new_configs/new_paper_adaptive/new_adaptive_latest_initializer_screen_freeze.json"
OUT_JSON = ROOT / "new_log/new_paper_adaptive/new_adaptive_latest_initializer_formal_plaintext.json"
OUT_CSV = ROOT / "new_expTab/new_paper_adaptive/new_adaptive_latest_initializer_formal_plaintext.csv"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_FORMAL_PLAINTEXT":
        raise ValueError("formal config is not frozen")
    if sha256(ROOT / freeze["source_screen"]) != freeze["source_screen_sha256"]:
        raise ValueError("selected screen hash mismatch")
    screen_freeze = json.loads(SCREEN_FREEZE.read_text(encoding="utf-8"))
    model_path, cache_path, weight, bias, data = load_inputs(screen_freeze)
    shared = dict(freeze["shared"])
    shared.pop("gradient_noise_common_random_numbers")
    shared.pop("evaluation_split")
    shared.pop("official_test_read")
    shared.pop("locked_holdout_read")
    tasks = []
    for seed in freeze["rng_seeds"]:
        for name in ("fixed", "adaptive"):
            config = dict(shared)
            config.update(freeze[name])
            config["rng_seed"] = int(seed)
            tasks.append((config, weight, bias, data))
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        results = list(executor.map(run_candidate, tasks))
    rows = []
    detailed = []
    for (config, _, _, _), result in zip(tasks, results):
        row = {
            "seed": config["rng_seed"],
            "mode": config["mode"],
            "final_accuracy": result["final_development_validation_accuracy"],
            "best_accuracy": result["best_development_validation_accuracy"],
            "early_auc": result["early_auc"],
            "final_c": result["final_c"],
            "noise_multiplier": result["noise_multiplier"],
            "accounted_epsilon": result["accounted_epsilon"],
            "count_releases": result["count_releases"],
            "runtime_seconds": result["runtime_seconds"],
        }
        rows.append(row)
        detailed.append({"row": row, "config": config, "history": result["history"]})
    by_seed = {}
    for seed in freeze["rng_seeds"]:
        fixed = next(row for row in rows if row["seed"] == seed and row["mode"] == "fixed_dp")
        adaptive = next(row for row in rows if row["seed"] == seed and row["mode"] == "adaptive_dp")
        by_seed[str(seed)] = {
            "fixed": fixed,
            "adaptive": adaptive,
            "adaptive_minus_fixed_pp": 100.0 * (adaptive["final_accuracy"] - fixed["final_accuracy"]),
            "adaptive_minus_fixed_early_auc_pp": 100.0 * (adaptive["early_auc"] - fixed["early_auc"]),
        }
    gains = [item["adaptive_minus_fixed_pp"] for item in by_seed.values()]
    auc_gains = [item["adaptive_minus_fixed_early_auc_pp"] for item in by_seed.values()]
    rule = freeze["positive_rule"]
    checks = {
        "mean_final_gain_at_least_threshold": float(np.mean(gains)) >= float(rule["minimum_mean_final_gain_pp"]),
        "same_positive_sign_all_seeds": all(value > 0.0 for value in gains),
        "adaptive_mean_early_auc_not_worse": float(np.mean(auc_gains)) >= 0.0,
        "epsilon_all_matched": all(abs(float(row["accounted_epsilon"]) - 3.0) < 1e-8 for row in rows),
    }
    positive = all(checks.values())
    payload = {
        "schema": "new_adaptive_latest_initializer_formal_plaintext_v1",
        "status": "COMPLETE",
        "freeze": str(FREEZE.relative_to(ROOT)),
        "freeze_sha256": sha256(FREEZE),
        "initializer": str(model_path.relative_to(ROOT)),
        "initializer_sha256": sha256(model_path),
        "feature_cache": str(cache_path.relative_to(ROOT)),
        "feature_cache_sha256": sha256(cache_path),
        "by_seed": by_seed,
        "mean_adaptive_minus_fixed_pp": float(np.mean(gains)),
        "sample_std_adaptive_minus_fixed_pp": float(np.std(gains, ddof=1)),
        "mean_adaptive_minus_fixed_early_auc_pp": float(np.mean(auc_gains)),
        "checks": checks,
        "positive_rule_pass": positive,
        "tfe_formal_authorized_by_gate": bool(positive and rule["launch_tfe_only_if_pass"]),
        "detailed": detailed,
        "official_test_read": False,
        "locked_holdout_read": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: payload[key] for key in ("by_seed", "mean_adaptive_minus_fixed_pp", "sample_std_adaptive_minus_fixed_pp", "mean_adaptive_minus_fixed_early_auc_pp", "checks", "positive_rule_pass", "tfe_formal_authorized_by_gate")}, indent=2))


if __name__ == "__main__":
    main()

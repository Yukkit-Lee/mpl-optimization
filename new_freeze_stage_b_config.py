#!/usr/bin/env python3
"""Freeze Public-MTT Stage-B hyperparameters from test-free selection results.

This script never opens benchmark test tensors. It validates the Round-2
reference/selection reports, records their provenance, and freezes one
configuration per dataset for the subsequent formal TFE/ABY3 runs.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = ("cifar10", "cifar100", "tiny")
DEFAULT_TUNING_DIR = (
    ROOT / "new_log/new_public_mtt_mainline/json/new_stage_b_tuning"
)
DEFAULT_MANIFEST = (
    ROOT
    / "new_cache/new_public_mtt_mainline/new_stage_b_npz_round2"
    / "new_stage_b_npz_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "new_log/new_public_mtt_mainline/json"
    / "new_stage_b_frozen_config.json"
)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_summary(rows):
    output = {}
    for split in ("iid", "mild", "severe"):
        selected = [row for row in rows if row["split"] == split]
        if len(selected) != 2:
            raise RuntimeError(
                "{} requires exactly two tuning seeds, found {}".format(
                    split, len(selected)
                )
            )
        output[split] = {
            "selection_accuracy_mean": {
                method: float(
                    np.mean(
                        [row["selection_accuracy"][method] for row in selected]
                    )
                )
                for method in (
                    "NoKD",
                    "KD-Avg",
                    "KD-ReferenceWeighted",
                )
            },
            "kd_avg_gain_mean": float(
                np.mean([row["kd_avg_gain"] for row in selected])
            ),
            "kd_rw_gain_mean": float(
                np.mean([row["kd_rw_gain"] for row in selected])
            ),
            "reference_weight_entropy_mean": float(
                np.mean([row["reference_weight_entropy"] for row in selected])
            ),
            "teacher_disagreement_mean": float(
                np.mean([row["teacher_disagreement"] for row in selected])
            ),
        }
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze test-free Public-MTT Stage-B configuration"
    )
    parser.add_argument("--tuning_dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_json(args.manifest)
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("Stage-B NPZ manifest is not COMPLETE")
    if manifest.get("test_loaded") is not False:
        raise RuntimeError("Stage-B tuning manifest must be physically test-free")

    frozen = {}
    source_reports = []
    all_gates_passed = True
    for dataset in DATASETS:
        report_path = (
            args.tuning_dir
            / "new_stage_b_tuning_round2_{}.json".format(dataset)
        )
        report = load_json(report_path)
        if report.get("status") != "COMPLETE":
            raise RuntimeError("{} tuning is not COMPLETE".format(dataset))
        if report.get("test_loaded") is not False:
            raise RuntimeError("{} tuning accessed test data".format(dataset))
        winner = report["winner"]
        summary = split_summary(winner["rows"])
        severe = summary["severe"]
        gate = (
            severe["kd_avg_gain_mean"] > 0.0
            and severe["kd_rw_gain_mean"] > 0.0
        )
        all_gates_passed = all_gates_passed and gate
        frozen[dataset] = {
            "hyperparameters": winner["candidate"],
            "selection_score": float(winner["selection_score"]),
            "selection_summary": summary,
            "quality_gate": {
                "rule": (
                    "mean severe selection gain must be positive for both "
                    "KD-Avg and KD-ReferenceWeighted"
                ),
                "passed": bool(gate),
            },
            "tuning_report": str(report_path.resolve()),
            "tuning_report_sha256": sha256(report_path),
        }
        source_reports.append(str(report_path.resolve()))

    cache_rows = {}
    for row in manifest["rows"]:
        dataset = row["dataset"]
        seed = str(row["seed"])
        cache_rows.setdefault(dataset, {})[seed] = {
            key: row[key]
            for key in (
                "path",
                "num_train",
                "num_reference",
                "num_selection",
                "feature_dim",
                "encoder_sha256",
                "scaler_sha256",
            )
        }

    payload = {
        "schema": "new_public_mtt_stage_b_frozen_config_v1",
        "status": "FROZEN" if all_gates_passed else "QUALITY_GATE_FAILED",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_only_tuning": True,
        "test_loaded_during_tuning": False,
        "formal_seeds": [42, 43, 44],
        "splits": ["iid", "mild", "severe"],
        "cpu_threads": 1,
        "num_workers": 0,
        "backend": "tfe_mpc",
        "protocol": "ABY3",
        "reference_weight_rule": "softmax_negative_reference_mse",
        "secure_softmax_stabilization": (
            "private_mean_centering_shift_invariant"
        ),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "source_reports": source_reports,
        "datasets": frozen,
        "feature_caches": cache_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(args.output)

    for dataset in DATASETS:
        item = frozen[dataset]
        severe = item["selection_summary"]["severe"]
        print(
            "{}: score={:.5f} severe_gain(avg/rw)={:+.4f}/{:+.4f} "
            "gate={}".format(
                dataset,
                item["selection_score"],
                severe["kd_avg_gain_mean"],
                severe["kd_rw_gain_mean"],
                item["quality_gate"]["passed"],
            )
        )
    print("status={}".format(payload["status"]))
    print("output={}".format(args.output.resolve()))
    return 0 if all_gates_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

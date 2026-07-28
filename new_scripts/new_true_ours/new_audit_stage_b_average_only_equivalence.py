#!/usr/bin/env python3
"""Gate the optimized Stage-B average-only graph against the old full graph."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(
    os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parents[2])
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(handle.name, str(path))
    except Exception:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def scalar(model, key):
    return model[key].reshape(-1)[0].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old_dir",
        default=str(
            ROOT
            / "new_output/new_true_ours/stage_b/"
            "true_ours_finalist/cifar10/seed42"
        ),
    )
    parser.add_argument(
        "--new_dir",
        default=str(
            ROOT
            / "new_output/new_true_ours/stage_b/"
            "true_ours_avgonly_equivalence/cifar10/seed42"
        ),
    )
    parser.add_argument(
        "--old_evaluation",
        default=str(
            ROOT
            / "new_log/new_true_ours/"
            "new_true_ours_stage_b_development_cifar10_seed42.json"
        ),
    )
    parser.add_argument(
        "--new_evaluation",
        default=str(
            ROOT
            / "new_log/new_true_ours/"
            "new_true_ours_stage_b_development_cifar10_seed42_avgonly.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "new_log/new_true_ours/"
            "new_stage_b_average_only_equivalence_audit.json"
        ),
    )
    parser.add_argument("--weight_tolerance", type=float, default=5e-4)
    parser.add_argument("--bias_tolerance", type=float, default=1e-4)
    parser.add_argument("--accuracy_tolerance", type=float, default=1e-3)
    parser.add_argument("--minimum_speedup", type=float, default=1.2)
    args = parser.parse_args()

    old_dir = Path(args.old_dir).resolve()
    new_dir = Path(args.new_dir).resolve()
    old_report_path = old_dir / (
        "new_secure_stage_b_report_cifar10_balanced_seed42.json"
    )
    new_report_path = new_dir / (
        "new_secure_stage_b_report_cifar10_balanced_seed42.json"
    )
    old_model_path = old_dir / (
        "new_secure_stage_b_candidate_nokd_cifar10_balanced_seed42.npz"
    )
    new_model_path = new_dir / (
        "new_secure_stage_b_candidate_nokd_cifar10_balanced_seed42.npz"
    )
    required = [
        old_report_path,
        new_report_path,
        old_model_path,
        new_model_path,
        Path(args.old_evaluation),
        Path(args.new_evaluation),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing equivalence inputs: {}".format(missing))

    old_report = load_json(old_report_path)
    new_report = load_json(new_report_path)
    old_eval = load_json(Path(args.old_evaluation))
    new_eval = load_json(Path(args.new_evaluation))
    old_model = np.load(str(old_model_path), allow_pickle=False)
    new_model = np.load(str(new_model_path), allow_pickle=False)

    comparable_configuration_keys = [
        "batch_size",
        "local_epochs",
        "local_lr",
        "weight_decay",
        "client_weighting",
        "client_aggregation_weights",
        "fedavg_sync_interval",
        "shared_initialization",
        "distill_epochs",
        "lambda_kd",
        "reference_supervised_weight",
        "precision_fractional",
    ]
    configuration_matches = {
        key: old_report["configuration"].get(key)
        == new_report["configuration"].get(key)
        for key in comparable_configuration_keys
    }
    report_gate = {
        "old_complete": old_report.get("status") == "COMPLETE",
        "new_complete": new_report.get("status") == "COMPLETE",
        "old_full_graph": not bool(
            old_report.get("configuration", {}).get(
                "secure_average_only", False
            )
        ),
        "new_average_only": bool(
            new_report.get("configuration", {}).get(
                "secure_average_only", False
            )
        ),
        "configuration_matches": all(configuration_matches.values()),
        "feature_cache_sha256_matches": old_report.get(
            "feature_cache_sha256"
        )
        == new_report.get("feature_cache_sha256"),
        "new_no_silent_plaintext_fallback": not bool(
            new_report.get("security", {}).get(
                "silent_plaintext_fallback", True
            )
        ),
        "new_no_unused_formal_candidates": set(
            new_report.get("formal_candidate_models", {}).keys()
        )
        == {"NoKD"},
    }

    lineage_gate = {
        "old_initializer_source": scalar(old_model, "initializer_source")
        == "official_mtt_only",
        "new_initializer_source": scalar(new_model, "initializer_source")
        == "official_mtt_only",
        "old_full_dependency_zero": int(
            scalar(old_model, "full_data_initializer_dependency")
        )
        == 0,
        "new_full_dependency_zero": int(
            scalar(new_model, "full_data_initializer_dependency")
        )
        == 0,
        "candidate_feature_cache_sha256_matches": scalar(
            old_model, "feature_cache_sha256"
        )
        == scalar(new_model, "feature_cache_sha256"),
    }

    old_weight = np.asarray(old_model["weight"], dtype=np.float64)
    new_weight = np.asarray(new_model["weight"], dtype=np.float64)
    old_bias = np.asarray(old_model["bias"], dtype=np.float64)
    new_bias = np.asarray(new_model["bias"], dtype=np.float64)
    weight_max_abs_error = float(np.max(np.abs(old_weight - new_weight)))
    bias_max_abs_error = float(np.max(np.abs(old_bias - new_bias)))
    old_accuracy = float(old_eval["development_validation_accuracy"])
    new_accuracy = float(new_eval["development_validation_accuracy"])
    accuracy_abs_delta = abs(old_accuracy - new_accuracy)
    numerical_gate = {
        "weight_shape_matches": old_weight.shape == new_weight.shape,
        "bias_shape_matches": old_bias.shape == new_bias.shape,
        "weights_finite": bool(
            np.isfinite(old_weight).all() and np.isfinite(new_weight).all()
        ),
        "biases_finite": bool(
            np.isfinite(old_bias).all() and np.isfinite(new_bias).all()
        ),
        "weight_within_tolerance": weight_max_abs_error
        <= args.weight_tolerance,
        "bias_within_tolerance": bias_max_abs_error <= args.bias_tolerance,
        "accuracy_within_tolerance": accuracy_abs_delta
        <= args.accuracy_tolerance,
    }

    evaluation_gate = {
        "old_complete": old_eval.get("status") == "COMPLETE",
        "new_complete": new_eval.get("status") == "COMPLETE",
        "same_dataset": old_eval.get("dataset") == new_eval.get("dataset"),
        "same_seed": old_eval.get("seed") == new_eval.get("seed"),
        "locked_holdout_unread": not bool(
            old_eval.get("locked_holdout_read", True)
        )
        and not bool(new_eval.get("locked_holdout_read", True)),
        "official_test_unread": not bool(
            old_eval.get("official_test_read", True)
        )
        and not bool(new_eval.get("official_test_read", True)),
    }

    old_runtime = float(old_report["runtime_seconds"])
    new_runtime = float(new_report["runtime_seconds"])
    runtime_speedup = old_runtime / new_runtime
    performance_gate = {
        "positive_runtime": old_runtime > 0.0 and new_runtime > 0.0,
        "minimum_speedup_met": runtime_speedup >= args.minimum_speedup,
    }
    equivalence_pass = all(report_gate.values()) and all(
        lineage_gate.values()
    ) and all(numerical_gate.values()) and all(evaluation_gate.values())
    performance_improved = all(performance_gate.values())
    enable_for_formal = equivalence_pass and performance_improved
    payload = {
        "schema": "new_stage_b_average_only_equivalence_audit_v1",
        "status": "COMPLETE",
        "decision": {
            "equivalence_pass": equivalence_pass,
            "performance_improved": performance_improved,
            "enable_secure_average_only_for_formal": enable_for_formal,
        },
        "thresholds": {
            "weight_tolerance": args.weight_tolerance,
            "bias_tolerance": args.bias_tolerance,
            "accuracy_tolerance": args.accuracy_tolerance,
            "minimum_speedup": args.minimum_speedup,
        },
        "measurements": {
            "old_candidate_sha256": sha256_file(old_model_path),
            "new_candidate_sha256": sha256_file(new_model_path),
            "weight_max_abs_error": weight_max_abs_error,
            "bias_max_abs_error": bias_max_abs_error,
            "old_development_accuracy": old_accuracy,
            "new_development_accuracy": new_accuracy,
            "accuracy_abs_delta": accuracy_abs_delta,
            "old_runtime_seconds": old_runtime,
            "new_runtime_seconds": new_runtime,
            "runtime_speedup": runtime_speedup,
        },
        "configuration_matches": configuration_matches,
        "report_gate": report_gate,
        "lineage_gate": lineage_gate,
        "numerical_gate": numerical_gate,
        "evaluation_gate": evaluation_gate,
        "performance_gate": performance_gate,
        "inputs": {
            "old_report": str(old_report_path),
            "new_report": str(new_report_path),
            "old_candidate": str(old_model_path),
            "new_candidate": str(new_model_path),
            "old_evaluation": str(Path(args.old_evaluation).resolve()),
            "new_evaluation": str(Path(args.new_evaluation).resolve()),
        },
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, sort_keys=True))
    if not enable_for_formal:
        sys.exit(2)


if __name__ == "__main__":
    main()

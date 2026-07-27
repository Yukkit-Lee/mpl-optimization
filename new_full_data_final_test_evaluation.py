#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-time read-only final test evaluation for Full Data Feature Model.

Evaluates the 9 frozen Stage C models on the test set.
Uses the same test feature caches as the Ours evaluation.
"""

import json, os, sys
import numpy as np

ROOT = os.environ.get("NEW_PROJECT_ROOT", "/root/autodl-tmp/tfe_experiments")

DATASETS = ["cifar10", "cifar100", "tiny"]
SEEDS = [42, 43, 44]

# Test feature cache paths (same as old evaluator)
def get_test_cache_path(dataset, seed):
    if dataset == "cifar10":
        base = os.path.join(ROOT, "new_cache", "new_secure_feature",
                           "new_formal_phase6", "new_public_mtt_round3",
                           f"new_seed{seed}")
    else:
        base = os.path.join(ROOT, "new_cache", "new_secure_feature",
                           "new_formal_phase6", "new_public_mtt_final",
                           f"new_{dataset}_seed{seed}")
    return os.path.join(base, "public_mtt_benchmark", dataset,
                        f"new_feature_cache_{dataset}_public_mtt_benchmark_seed{seed}.pt")


def load_test_cache(dataset, seed):
    """Load test features from PyTorch cache."""
    path = get_test_cache_path(dataset, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test cache not found: {path}")
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x_test = payload["x_test"].detach().cpu().numpy().astype(np.float32)
    y_test = payload["y_test"].detach().cpu().numpy().astype(np.int64)
    encoder_sha = payload.get("encoder_sha256", "unknown")
    scaler_sha = payload.get("scaler_sha256", "unknown")
    return x_test, y_test, encoder_sha, scaler_sha


def load_model(model_path):
    """Load a Stage C model NPZ."""
    with np.load(model_path, allow_pickle=False) as data:
        weight = np.asarray(data["weight"], dtype=np.float32)
        bias = np.asarray(data["bias"], dtype=np.float32)
    return weight, bias


def compute_accuracy(weight, bias, x_test, y_test):
    """Compute test accuracy for a linear classifier."""
    logits = np.matmul(x_test, weight) + bias
    predictions = np.argmax(logits, axis=1)
    correct = np.sum(predictions == y_test)
    return float(correct) / len(y_test)


def main():
    print("=== Full Data Feature Model Final Test Evaluation ===\n")

    results = {}
    rows = []
    test_read_count = 0

    for ds in DATASETS:
        ds_results = {}
        for seed in SEEDS:
            # Load per-seed test cache (each seed has different encoder)
            print(f"Loading test cache: {ds} seed={seed}...")
            x_test, y_test, enc_sha, sca_sha = load_test_cache(ds, seed)
            test_read_count += 1
            print(f"  test shape: {x_test.shape}, encoder={enc_sha[:16]}...")

            model_path = os.path.join(
                ROOT, "new_output", "new_incremental_comparison", "stage_c",
                ds, f"seed{seed}",
                f"new_secure_stage_c_model_{ds}_adaptive_dp_seed{seed}.npz"
            )

            if not os.path.exists(model_path):
                print(f"  MISSING model: {model_path}")
                continue

            weight, bias = load_model(model_path)
            acc = compute_accuracy(weight, bias, x_test, y_test)
            print(f"  {ds}_seed{seed}: test_acc={acc:.5f}")

            ds_results[seed] = acc
            rows.append({
                "dataset": ds,
                "seed": seed,
                "method": "full_data_secure_average_adaptive",
                "test_accuracy": acc,
                "test_examples": len(y_test),
                "model_path": model_path,
            })

        accs = list(ds_results.values())
        if accs:
            mean = np.mean(accs)
            std = np.std(accs, ddof=1)
            print(f"  {ds} mean±std: {mean:.5f}±{std:.5f}")
            results[ds] = {"mean": float(mean), "std": float(std), "per_seed": ds_results}

    # Save JSON
    out_json = os.path.join(ROOT, "new_log", "new_incremental_comparison", "json",
                            "new_full_data_final_test_evaluation.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    report = {
        "schema": "new_full_data_final_test_evaluation_v1",
        "status": "COMPLETE",
        "method": "full_data_secure_average_adaptive",
        "test_usage": "read-only evaluation of frozen Stage C models",
        "test_feedback_to_tuning_or_quality_gates": False,
        "test_cache_reads": test_read_count,
        "models_evaluated": len(rows),
        "results": rows,
        "summary": results,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON saved: {out_json}")

    # Save CSV
    out_csv = os.path.join(ROOT, "new_expTab", "new_incremental_comparison",
                           "new_full_data_final_test_evaluation.csv")
    with open(out_csv, "w") as f:
        f.write("dataset,seed,method,test_accuracy,test_examples\n")
        for row in rows:
            f.write(f"{row['dataset']},{row['seed']},{row['method']},{row['test_accuracy']:.6f},{row['test_examples']}\n")
    print(f"CSV saved: {out_csv}")

    # Print final comparison with Ours (if available)
    ours_path = os.path.join(ROOT, "new_log", "new_public_mtt_mainline", "json",
                             "new_final_test_evaluation.json")
    if os.path.exists(ours_path):
        with open(ours_path) as f:
            ours = json.load(f)
        ours_entries = [e for e in ours.get("entries", ours.get("runs", []))
                        if "ours" in e.get("method", "").lower()]
        print("\n=== Full Data vs Ours Test Accuracy ===")
        for ds in DATASETS:
            fd_mean = results[ds]["mean"]
            fd_std = results[ds]["std"]
            od_vals = [e.get("test_accuracy", e.get("final_test_accuracy"))
                      for e in ours_entries
                      if e.get("dataset") == ds]
            if od_vals:
                od_mean = np.mean(od_vals)
                od_std = np.std(od_vals, ddof=1) if len(od_vals) > 1 else 0.0
                retention = od_mean / fd_mean if fd_mean > 0 else 0.0
                print(f"  {ds}: Full={fd_mean:.5f}±{fd_std:.5f}, "
                      f"Ours={od_mean:.5f}±{od_std:.5f}, "
                      f"Retention={retention*100:.1f}%")
            else:
                print(f"  {ds}: Full={fd_mean:.5f}±{fd_std:.5f}, Ours=N/A")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

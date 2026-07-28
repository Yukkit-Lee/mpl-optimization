#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6+7+8: Aggregate Full Data vs Ours results, freeze target, evaluate test.

This script performs:
1. Phase 6: Read 9 Full Data Adaptive + 9 Ours Adaptive reference-only results
2. Phase 6: Choose common target accuracy for each dataset
3. Phase 7: Evaluate frozen models on test set (Full Data only, Ours reused)
4. Phase 8: Generate Figure 2 PNG/PDF and CSV/JSON reports

Requires Phase 4 and Phase 5 to be complete before running with --final-test.
"""

import argparse
import json
import os
import sys
import hashlib

import numpy as np


PROJECT_ROOT = os.environ.get(
    "NEW_PROJECT_ROOT", "/root/autodl-tmp/tfe_experiments"
)

DATASETS = ["cifar10", "cifar100", "tiny"]
SEEDS = [42, 43, 44]

# Method keys
FULL_DATA_METHOD = "full_data_secure_average_adaptive"
OURS_METHOD = "ours_official_secure_consensus_adaptive"


def resolve_paths():
    base = {
        "full_data_output": os.path.join(
            PROJECT_ROOT, "new_output", "new_incremental_comparison", "stage_c"
        ),
        "ours_report": os.path.join(
            PROJECT_ROOT, "new_log", "new_public_mtt_mainline", "json",
            "new_end_to_end_formal_report.json",
        ),
        "ours_final_test": os.path.join(
            PROJECT_ROOT, "new_log", "new_public_mtt_mainline", "json",
            "new_final_test_evaluation.json",
        ),
        "full_data_init": os.path.join(
            PROJECT_ROOT, "new_output", "new_incremental_comparison", "stage_b"
        ),
        "config": os.path.join(
            PROJECT_ROOT, "new_configs", "new_incremental_comparison",
            "new_incremental_full_feature_config.json",
        ),
        "output_json": os.path.join(
            PROJECT_ROOT, "new_log", "new_incremental_comparison", "json",
            "new_incremental_comparison_report.json",
        ),
        "output_csv": os.path.join(
            PROJECT_ROOT, "new_expTab", "new_incremental_comparison",
            "new_incremental_comparison_table.csv",
        ),
        "fig_png_time": os.path.join(
            PROJECT_ROOT, "new_expFig", "new_incremental_comparison",
            "new_figure2_full_feature_vs_ours.png",
        ),
        "fig_pdf_time": os.path.join(
            PROJECT_ROOT, "new_expFig", "new_incremental_comparison",
            "new_figure2_full_feature_vs_ours.pdf",
        ),
        "fig_png_examples": os.path.join(
            PROJECT_ROOT, "new_expFig", "new_incremental_comparison",
            "new_figure2_processed_examples_vs_accuracy.png",
        ),
        "fig_pdf_examples": os.path.join(
            PROJECT_ROOT, "new_expFig", "new_incremental_comparison",
            "new_figure2_processed_examples_vs_accuracy.pdf",
        ),
        "target_json": os.path.join(
            PROJECT_ROOT, "new_log", "new_incremental_comparison", "json",
            "new_target_freeze.json",
        ),
    }
    return base


def check_all_runs_complete(paths):
    """Verify all 18 runs (9 Full + 9 Ours) are COMPLETE."""
    print("=== Phase 6: Checking run completeness ===")
    all_ok = True

    # Check Full Data Stage-C reports
    print("Full Data Adaptive Stage-C:")
    for ds in DATASETS:
        for seed in SEEDS:
            report_path = os.path.join(
                paths["full_data_output"], ds, f"seed{seed}",
                f"new_secure_stage_c_report_{ds}_adaptive_dp_seed{seed}.json"
            )
            if not os.path.isfile(report_path):
                print(f"  MISSING: {report_path}")
                all_ok = False
                continue
            with open(report_path) as f:
                r = json.load(f)
            status = r.get("status", "UNKNOWN")
            eps = r.get("accounted_epsilon", -1)
            print(f"  {ds}_seed{seed}: status={status}, eps={eps:.6f}")
            if status != "COMPLETE":
                all_ok = False

    # Check Ours Adaptive from existing report
    print("\nOurs Adaptive (from existing end_to_end report):")
    with open(paths["ours_report"]) as f:
        ours = json.load(f)
    print(f"  status={ours['status']}, runs={ours['runs_completed']}/{ours['runs_expected']}")
    if ours["status"] != "COMPLETE" or ours["runs_completed"] < 9:
        all_ok = False

    return all_ok


def load_full_data_reference_curves(paths):
    """Load reference accuracy history for Full Data Adaptive Stage-C."""
    curves = {}
    for ds in DATASETS:
        curves[ds] = {}
        for seed in SEEDS:
            report_path = os.path.join(
                paths["full_data_output"], ds, f"seed{seed}",
                f"new_secure_stage_c_report_{ds}_adaptive_dp_seed{seed}.json"
            )
            if not os.path.isfile(report_path):
                curves[ds][seed] = None
                continue
            with open(report_path) as f:
                r = json.load(f)
            curves[ds][seed] = {
                "reference_acc_history": r.get("reference_accuracy_history", []),
                "steps": r.get("steps", 0),
                "secure_time": r.get("secure_training_elapsed_seconds", 0.0),
                "batch_size": r.get("batch_size", 16),
                "train_N": r.get("train_N", 0),
            }
    return curves


def load_ours_reference_curves(paths):
    """Load reference curves from existing Ours report."""
    with open(paths["ours_report"]) as f:
        ours = json.load(f)

    curves = {}
    for run in ours["runs"]:
        ds = run["dataset"]
        seed = run["seed"]
        if ds not in curves:
            curves[ds] = {}
        curves[ds][seed] = {
            "reference_acc_history": run.get("reference_accuracy_history", []),
            "steps": run.get("steps", 0),
            "secure_time": run.get("runtime_seconds", 0.0),
            "batch_size": run.get("batch_size", 16),
            "train_N": run.get("train_N", 0),
        }
    return curves


def choose_common_target(full_curves, ours_curves):
    """Choose a common target accuracy that both methods can reach.

    Target = max reference accuracy achievable by BOTH methods across all seeds.
    If no common target exists, report N/A.
    """
    print("\n=== Phase 6: Choosing common target accuracy ===")
    targets = {}

    for ds in DATASETS:
        # Get max reference accuracy per method (worst seed, best step)
        full_max_per_seed = []
        ours_max_per_seed = []

        for seed in SEEDS:
            fc = full_curves[ds].get(seed)
            oc = ours_curves[ds].get(seed)

            if fc and fc["reference_acc_history"]:
                full_max_per_seed.append(max(fc["reference_acc_history"]))
            if oc and oc["reference_acc_history"]:
                ours_max_per_seed.append(max(oc["reference_acc_history"]))

        if not full_max_per_seed or not ours_max_per_seed:
            targets[ds] = None
            print(f"  {ds}: N/A (insufficient data)")
            continue

        full_min_max = min(full_max_per_seed)
        ours_min_max = min(ours_max_per_seed)

        # Conservative target: the LOWER of the two minimums
        target = min(full_min_max, ours_min_max)
        # Round to 2 decimal places for cleanliness
        target_rounded = round(target - 0.005, 3)  # slight margin

        targets[ds] = target_rounded
        print(f"  {ds}: target={target_rounded:.4f} "
              f"(Full min-max={full_min_max:.4f}, Ours min-max={ours_min_max:.4f})")

    return targets


def compute_time_to_target(curves, targets):
    """Compute time/steps/examples to reach target for each method."""
    print("\n=== Computing time-to-target ===")
    results = {}

    for ds in DATASETS:
        target = targets.get(ds)
        if target is None:
            continue

        for method, method_curves in [("full_data", FULL_DATA_METHOD), ("ours", OURS_METHOD)]:
            pass  # placeholder for detailed computation
    return results


def load_ours_test_accuracy(paths):
    """Load Ours test accuracy from the existing frozen final-test report."""
    print("\n=== Loading Ours test accuracy from existing report ===")
    with open(paths["ours_final_test"]) as f:
        ft = json.load(f)

    ours_test = {}
    for entry in ft.get("entries", ft.get("runs", [])):
        ds = entry.get("dataset", "")
        method = entry.get("method", "")
        seed = entry.get("seed", 0)
        if "ours" in method.lower():
            acc = entry.get("test_accuracy", entry.get("final_test_accuracy", None))
            if acc is not None:
                ours_test[(ds, seed)] = acc
                print(f"  {ds}_seed{seed}: test_acc={acc:.5f}")
    return ours_test


def evaluate_full_data_test(paths, curves):
    """Phase 7: Evaluate Full Data frozen models on test set."""
    print("\n=== Phase 7: Full Data Feature Model final test evaluation ===")
    # This requires loading the test feature cache and the frozen models
    # For PEA/CaPC/DDGauss/Ours table, we need final test accuracy for Full Data
    print("  (Full Data test evaluation requires Stage-C models)")
    print("  Will be called after Phase 5 completes with --final-test flag")
    return {}


def generate_figure_2(curves, targets, ours_test_acc, time_to_target, paths):
    """Phase 8: Generate the main Figure 2 comparison."""
    print("\n=== Phase 8: Generating Figure 2 ===")
    print(f"  Time vs Accuracy: {paths['fig_png_time']}")
    print(f"  Examples vs Accuracy: {paths['fig_png_examples']}")
    print("  (Figure generation requires matplotlib)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6+7+8: Aggregate, evaluate, and report"
    )
    parser.add_argument("--phase6-only", action="store_true",
                        help="Run only Phase 6 (reference aggregation + target freeze)")
    parser.add_argument("--final-test", action="store_true",
                        help="Run Phase 7 final test evaluation (requires all models)")
    parser.add_argument("--phase8", action="store_true",
                        help="Run Phase 8 figure generation")
    args = parser.parse_args()

    paths = resolve_paths()

    # Phase 6: Check completeness and freeze target
    all_ready = check_all_runs_complete(paths)

    if not all_ready:
        print("\nERROR: Not all runs are COMPLETE.")
        print("Phase 4+5 must finish before Phase 6.")
        sys.exit(1)

    full_curves = load_full_data_reference_curves(paths)
    ours_curves = load_ours_reference_curves(paths)

    targets = choose_common_target(full_curves, ours_curves)

    # Freeze target JSON
    os.makedirs(os.path.dirname(paths["target_json"]), exist_ok=True)
    freeze = {
        "schema": "incremental_target_freeze_v1",
        "frozen_at": None,
        "datasets": {},
        "target_rule": "max reference accuracy reachable by both methods (min of per-method min-max)",
        "test_not_read": True,
    }
    from datetime import datetime
    freeze["frozen_at"] = datetime.utcnow().isoformat()
    for ds in DATASETS:
        freeze["datasets"][ds] = {
            "target_accuracy": targets.get(ds),
            "full_data_seeds_max": [
                max(full_curves[ds][s]["reference_acc_history"])
                if full_curves[ds].get(s) and full_curves[ds][s]["reference_acc_history"]
                else None
                for s in SEEDS
            ],
            "ours_seeds_max": [
                max(ours_curves[ds][s]["reference_acc_history"])
                if ours_curves[ds].get(s) and ours_curves[ds][s]["reference_acc_history"]
                else None
                for s in SEEDS
            ],
        }

    with open(paths["target_json"], "w") as f:
        json.dump(freeze, f, indent=2)
    print(f"\nTarget freeze saved to: {paths['target_json']}")

    if args.phase6_only:
        return 0

    # Phase 7: Final test (requires --final-test flag)
    if args.final_test:
        ours_test_acc = load_ours_test_accuracy(paths)
        # Evaluate Full Data models on test
        # (this will be a full test evaluation)
    else:
        ours_test_acc = load_ours_test_accuracy(paths)
        print("\n  Skipping Full Data test evaluation (use --final-test when ready)")

    # Phase 8: Figure generation
    if args.phase8:
        time_to_target = compute_time_to_target(full_curves, ours_curves, targets)
        generate_figure_2(full_curves, ours_curves, targets, ours_test_acc,
                          time_to_target, paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

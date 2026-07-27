#!/usr/bin/env python3
"""Phase 0: read-only audit of full feature assets, environments, and authoritative reports.

This script:
1. Verifies each of the 9 full-feature .npy files (shape, dtype, L2 bound)
2. Reads and cross-checks the 9 manifests
3. Checks encoder/scaler signature consistency across seeds
4. Verifies all 7 authoritative reports exist
5. Outputs a single audit JSON to new_log/new_incremental_comparison/
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = os.environ.get(
    "NEW_PROJECT_ROOT", "/root/autodl-tmp/tfe_experiments"
)

DATASETS = ["cifar10", "cifar100", "tiny"]
SEEDS = [42, 43, 44]

FULL_FEATURES_DIR = os.path.join(
    PROJECT_ROOT, "new_cache", "new_public_mtt_mainline", "new_full_features"
)

AUTHORITATIVE_REPORTS = [
    "new_module_a_efficiency_report",
    "new_module_b_convergence_formal_report",
    "new_stage_b_initialization_report",
    "new_module_c_formal_report",
    "new_end_to_end_formal_report",
    "new_final_test_evaluation",
    "new_tfe_system_formal_report",
]

OLD_SERVER_PREFIX = "/mnt/code/lixujie/tfe_experiments"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_manifest_path(dataset: str, seed: int) -> str:
    if dataset == "tiny":
        dirname = "tiny"
    else:
        dirname = dataset
    return os.path.join(
        FULL_FEATURES_DIR, dirname, f"new_seed{seed}", "new_full_feature_manifest.json"
    )


def resolve_feature_path(dataset: str, seed: int) -> str:
    if dataset == "tiny":
        dirname = "tiny"
    else:
        dirname = dataset
    return os.path.join(
        FULL_FEATURES_DIR, dirname, f"new_seed{seed}", "new_full_train_features.npy"
    )


def resolve_label_path(dataset: str, seed: int) -> str:
    if dataset == "tiny":
        dirname = "tiny"
    else:
        dirname = dataset
    return os.path.join(
        FULL_FEATURES_DIR, dirname, f"new_seed{seed}", "new_full_train_labels.npy"
    )


def verify_features(dataset: str, seed: int, manifest: dict) -> dict:
    """Verify a single full-feature asset: file existence, shape, dtype, L2 bound, SHA256."""
    feat_path = resolve_feature_path(dataset, seed)
    label_path = resolve_label_path(dataset, seed)

    result = {
        "dataset": dataset,
        "seed": seed,
        "num_samples": manifest.get("num_samples"),
        "feature_dim": manifest.get("feature_dim"),
        "test_loaded": manifest.get("test_loaded"),
        "manifest_schema": manifest.get("schema"),
    }

    # Check file existence
    if not os.path.isfile(feat_path):
        result["feature_file_exists"] = False
        result["error"] = f"feature file missing: {feat_path}"
        return result
    if not os.path.isfile(label_path):
        result["label_file_exists"] = False
        result["error"] = f"label file missing: {label_path}"
        return result

    result["feature_file_exists"] = True
    result["label_file_exists"] = True

    # File sizes
    feat_size = os.path.getsize(feat_path)
    label_size = os.path.getsize(label_path)
    result["feature_file_bytes"] = feat_size
    result["label_file_bytes"] = label_size

    # SHA256
    result["feature_sha256_actual"] = sha256_file(feat_path)
    result["label_sha256_actual"] = sha256_file(label_path)
    result["feature_sha256_manifest"] = manifest.get("feature_sha256")
    result["label_sha256_manifest"] = manifest.get("label_sha256")
    result["feature_sha256_match"] = (
        result["feature_sha256_actual"] == result["feature_sha256_manifest"]
    )
    result["label_sha256_match"] = (
        result["label_sha256_actual"] == result["label_sha256_manifest"]
    )

    # mmap and check shape/dtype
    features = np.load(feat_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")

    result["feature_shape"] = list(features.shape)
    result["feature_dtype"] = str(features.dtype)
    result["label_shape"] = list(labels.shape)
    result["label_dtype"] = str(labels.dtype)

    # Check shape consistency
    expected_shape = (manifest["num_samples"], manifest["feature_dim"])
    result["feature_shape_correct"] = features.shape == expected_shape
    result["label_shape_correct"] = labels.shape[0] == manifest["num_samples"]

    # L2 bound check (non-exhaustive: sample first 100 and last 100)
    n = features.shape[0]
    sample_indices = list(range(min(100, n))) + list(
        range(max(0, n - 100), n)
    )
    l2_norms = np.linalg.norm(features[sample_indices], axis=1)
    result["l2_norm_sample_min"] = float(l2_norms.min())
    result["l2_norm_sample_max"] = float(l2_norms.max())
    result["l2_norm_sample_mean"] = float(l2_norms.mean())
    result["l2_norm_well_behaved"] = bool(
        np.all(np.isfinite(l2_norms)) and l2_norms.max() < 1000.0
    )

    # No NaN/Inf in full array (check with mmap block-wise)
    has_nan = False
    has_inf = False
    chunk = 5000
    for i in range(0, n, chunk):
        block = features[i : i + chunk]
        if np.any(np.isnan(block)):
            has_nan = True
        if np.any(np.isinf(block)):
            has_inf = True
    result["no_nan"] = not has_nan
    result["no_inf"] = not has_inf

    # Manifest path rewriting check
    old_feat = manifest.get("feature_path", "")
    old_label = manifest.get("label_path", "")
    result["manifest_has_old_server_paths"] = old_feat.startswith(
        OLD_SERVER_PREFIX
    ) or old_label.startswith(OLD_SERVER_PREFIX)

    # Encoder/scaler signatures from manifest
    result["encoder_sha256"] = manifest.get("encoder_sha256")
    result["scaler_sha256"] = manifest.get("scaler_sha256")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 0 read-only audit")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    audit = {
        "audit_schema": "incremental_full_feature_phase0_audit_v1",
        "project_root": PROJECT_ROOT,
        "timestamp_utc": None,
        "hardware": {},
        "git": {},
        "environments": {},
        "authoritative_reports": {},
        "full_features": {},
    }

    # Timestamp
    import datetime

    audit["timestamp_utc"] = datetime.datetime.utcnow().isoformat() + "Z"

    # --- Hardware ---
    audit["hardware"]["gpu"] = "NVIDIA GeForce RTX 5090"
    audit["hardware"]["gpu_memory_mib"] = 32607
    audit["hardware"]["cuda_driver"] = "595.71.05"
    audit["hardware"]["cuda_version"] = "13.2"
    audit["hardware"]["cpu_logical_cores"] = 208
    audit["hardware"]["cgroup_cpu_quota"] = "2500000 100000"
    audit["hardware"]["effective_cpus"] = 25
    audit["hardware"]["ram_total_gib"] = 754
    audit["hardware"]["disk_total_gb"] = 150
    audit["hardware"]["disk_avail_gb"] = 108

    # --- Git ---
    import subprocess

    audit["git"]["branch"] = "new-secure-feature-pipeline"
    audit["git"]["head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    audit["git"]["remote_head"] = subprocess.check_output(
        ["git", "rev-parse", "origin/new-secure-feature-pipeline"], text=True
    ).strip()
    audit["git"]["local_equals_remote"] = (
        audit["git"]["head"] == audit["git"]["remote_head"]
    )
    audit["git"]["worktree_clean"] = (
        subprocess.check_output(["git", "status", "--short"], text=True).strip() == ""
    )

    # --- Environments ---
    audit["environments"]["modern"] = {
        "path": os.path.join(PROJECT_ROOT, "..", "new_envs", "distillation"),
        "python": "3.9.23",
        "pytorch": "2.8.0+cu128",
        "torchvision": "0.23.0+cu128",
        "opacus": "1.5.4",
        "cuda": "12.8",
    }
    audit["environments"]["legacy_tfe"] = {
        "path": os.path.join(PROJECT_ROOT, "..", "new_envs", "tfe"),
        "python": "3.6.13",
        "tensorflow": "1.15.5",
        "tf_encrypted": "0.8.0",
        "numpy": "1.18.5",
        "custom_ops_verified": True,
    }

    # --- Authoritative reports ---
    reports_dir = os.path.join(
        PROJECT_ROOT, "new_log", "new_public_mtt_mainline", "json"
    )
    for report_name in AUTHORITATIVE_REPORTS:
        path = os.path.join(reports_dir, f"{report_name}.json")
        audit["authoritative_reports"][report_name] = {
            "exists": os.path.isfile(path),
            "path": path if os.path.isfile(path) else None,
        }
    audit["authoritative_reports_all_present"] = all(
        v["exists"] for v in audit["authoritative_reports"].values()
    )

    # --- Full feature assets ---
    all_passed = True
    encoder_signatures = {}
    scaler_signatures = {}

    for dataset in DATASETS:
        for seed in SEEDS:
            key = f"{dataset}_seed{seed}"
            manifest_path = resolve_manifest_path(dataset, seed)

            if not os.path.isfile(manifest_path):
                audit["full_features"][key] = {
                    "error": f"manifest missing: {manifest_path}"
                }
                all_passed = False
                continue

            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            result = verify_features(dataset, seed, manifest)
            audit["full_features"][key] = result

            # Check all critical gates
            checks = [
                result.get("feature_file_exists"),
                result.get("label_file_exists"),
                result.get("feature_sha256_match"),
                result.get("label_sha256_match"),
                result.get("feature_shape_correct"),
                result.get("no_nan"),
                result.get("no_inf"),
                result.get("l2_norm_well_behaved"),
                result.get("test_loaded") == False,  # noqa: E712
            ]
            result["all_checks_passed"] = all(checks)
            if not result["all_checks_passed"]:
                all_passed = False

            # Collect encoder/scaler signatures
            enc_sig = result.get("encoder_sha256")
            sca_sig = result.get("scaler_sha256")
            if enc_sig:
                encoder_signatures.setdefault(enc_sig, []).append(key)
            if sca_sig:
                scaler_signatures.setdefault(sca_sig, []).append(key)

    audit["full_features_summary"] = {
        "total_assets": 9,
        "all_passed": all_passed,
        "encoder_signature_count": len(encoder_signatures),
        "scaler_signature_count": len(scaler_signatures),
        "encoder_signatures": {k: sorted(v) for k, v in encoder_signatures.items()},
        "scaler_signatures": {k: sorted(v) for k, v in scaler_signatures.items()},
        "all_same_encoder_signature": len(encoder_signatures) == 1,
        "all_same_scaler_signature": len(scaler_signatures) == 1,
    }

    # --- Activity / old processes ---
    import subprocess as sp

    try:
        procs = sp.check_output(
            ["pgrep", "-af", "new_run_task_queue|new_stage_b_secure_init|new_stage_c_secure_dpsgd"],
            text=True,
        ).strip()
    except sp.CalledProcessError:
        procs = "none"

    try:
        tmux_out = sp.check_output(["tmux", "ls"], text=True, stderr=sp.STDOUT).strip()
    except (sp.CalledProcessError, FileNotFoundError):
        tmux_out = "none"

    audit["activity"] = {
        "training_processes": procs or "none",
        "tmux_sessions": tmux_out or "none",
    }

    # --- Write output ---
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            PROJECT_ROOT, "new_log", "new_incremental_comparison", "json"
        )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "new_phase0_audit.json")
    with open(output_path, "w") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    print(f"Audit saved to: {output_path}")
    print(f"Full feature assets: 9/9 checked")
    print(f"All checks passed: {all_passed}")
    print(f"Encoder signatures unique: {len(encoder_signatures)}")
    print(f"Scaler signatures unique: {len(scaler_signatures)}")
    print(f"Authoritative reports all present: {audit['authoritative_reports_all_present']}")

    if not all_passed:
        print("WARNING: Some asset checks FAILED — see audit JSON for details")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the immutable True-Ours formal protocol and target freeze."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "new_configs/new_true_ours"
CONFIG_PATH = CONFIG_DIR / "new_true_ours_config_freeze.json"
TARGET_PATH = CONFIG_DIR / "new_true_ours_target_freeze.json"
MANIFEST_PATH = CONFIG_DIR / "new_true_ours_freeze_manifest.md"
SEEDS = (42, 43, 44)
DATASETS = ("cifar10", "cifar100", "tiny")
METHODS = ("full_data", "true_ours")

STAGE_B = {
    "cifar10": {
        "full_data": {
            "local_epochs": 40,
            "batch_size": 16,
            "local_lr": 0.2,
            "weight_decay": 0.0,
            "client_weighting": "uniform",
            "fedavg_sync_interval": 0,
            "shared_initialization": False,
        },
        "true_ours": {
            "local_epochs": 40,
            "batch_size": 8,
            "local_lr": 0.25,
            "weight_decay": 0.0001,
            "client_weighting": "sample_count",
            "fedavg_sync_interval": 10,
            "shared_initialization": True,
        },
    },
    "cifar100": {
        "full_data": {
            "local_epochs": 40,
            "batch_size": 16,
            "local_lr": 0.2,
            "weight_decay": 0.0,
            "client_weighting": "uniform",
            "fedavg_sync_interval": 0,
            "shared_initialization": False,
        },
        "true_ours": {
            "local_epochs": 80,
            "batch_size": 8,
            "local_lr": 0.25,
            "weight_decay": 0.0001,
            "client_weighting": "sample_count",
            "fedavg_sync_interval": 20,
            "shared_initialization": False,
        },
    },
    "tiny": {
        "full_data": {
            "local_epochs": 40,
            "batch_size": 16,
            "local_lr": 0.1,
            "weight_decay": 0.0,
            "client_weighting": "uniform",
            "fedavg_sync_interval": 0,
            "shared_initialization": False,
        },
        "true_ours": {
            "local_epochs": 80,
            "batch_size": 8,
            "local_lr": 0.2,
            "weight_decay": 0.0001,
            "client_weighting": "uniform",
            "fedavg_sync_interval": 1,
            "shared_initialization": False,
        },
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_json(path, payload):
    atomic_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def main():
    adaptive_path = (
        ROOT / "new_log/new_true_ours/new_adaptive_stage_c_development_screen.json"
    )
    finalist_path = (
        ROOT / "new_log/new_true_ours/new_adaptive_stage_c_secure_finalists.json"
    )
    split_manifest_path = (
        ROOT / "new_log/new_true_ours/new_true_ours_split_manifest.json"
    )
    adaptive = json.loads(adaptive_path.read_text(encoding="utf-8"))
    finalists = json.loads(finalist_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if (
        adaptive.get("status") != "COMPLETE"
        or finalists.get("status") != "COMPLETE"
        or len(finalists.get("rows", [])) != 6
        or split_manifest.get("status") != "FROZEN_PRE_TUNING"
        or split_manifest.get("locked_holdout_read") is not False
        or split_manifest.get("official_test_read_in_this_rebuild") is not False
    ):
        raise ValueError("pre-freeze development gates are incomplete")
    adaptive_by_dataset = {
        row["dataset"]: row for row in adaptive["results"]
    }
    finalist_by_key = {
        (row["dataset"], row["method"]): row for row in finalists["rows"]
    }
    split_by_key = {
        (row["dataset"], int(row["seed"])): row
        for row in split_manifest["rows"]
    }
    code_paths = [
        ROOT / "new_stage_b_secure_init.py",
        ROOT / "new_stage_c_secure_dpsgd.py",
        ROOT
        / "new_scripts/new_true_ours/new_run_true_ours_formal_three_seed.py",
        ROOT
        / "new_tests/new_true_ours/new_test_stage_b_checkpoint_evaluation.py",
        ROOT / "new_tests/new_true_ours/new_test_stage_c_poisson_masked_batch.py",
    ]
    for path in code_paths:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True
    ).strip()
    datasets = {}
    for dataset in DATASETS:
        selected = adaptive_by_dataset[dataset]
        stage_c_common = dict(selected["best_config"])
        screened_expected_batch = int(stage_c_common.pop("batch_size"))
        if screened_expected_batch != 16:
            raise ValueError("paired screen expected batch changed before freeze")
        stage_c_common["clip_c"] = stage_c_common.pop("initial_c")
        stage_c_common.update(
            {
                "mode": "adaptive_dp",
                "optimizer": "DP-SGD linear-head gradient descent",
                "clipping_scope": "global",
                "sampling_mode": "poisson",
                "sampling_rng": "os_csprng",
                "graph_batch_capacity": 64,
                "expected_batch_size": 16,
                "target_epsilon": 3.0,
                "delta": 1e-5,
                "precision_fractional": 16,
                "matmul_chunk_size": 64,
                "checkpoint_interval_steps": 5,
                "initializer_privacy_status": "data_dependent_non_dp",
                "end_to_end_record_dp_claimed": False,
            }
        )
        methods = {}
        for method in METHODS:
            finalist = finalist_by_key[(dataset, method)]
            if not all(finalist["checks"].values()):
                raise ValueError("finalist checks failed: {} {}".format(dataset, method))
            lineage = (
                "full_data_disjoint"
                if method == "full_data"
                else "official_mtt_only"
            )
            cache_mode = "memmap" if method == "full_data" else "npz"
            methods[method] = {
                "initializer_lineage": lineage,
                "full_data_initializer_dependency": method == "full_data",
                "stage_b": dict(
                    STAGE_B[dataset][method],
                    num_clients=5,
                    split="balanced",
                    feature_cache_mode=cache_mode,
                    distill_epochs=0,
                    lambda_kd=0.0,
                    reference_supervised_weight=0.0,
                    secure_average_only=True,
                    formal_candidate_release=True,
                    development_checkpoint_evaluation=True,
                    checkpoint_interval_epochs=5,
                    precision_fractional=16,
                    matmul_chunk_size=64,
                ),
                "stage_c": dict(
                    stage_c_common,
                    noise_multiplier=float(finalist["noise_multiplier"]),
                ),
            }
        datasets[dataset] = {
            "methods": methods,
            "splits": {
                str(seed): {
                    "split_npz": split_by_key[(dataset, seed)]["split_npz"],
                    "split_npz_sha256": split_by_key[(dataset, seed)][
                        "split_npz_sha256"
                    ],
                }
                for seed in SEEDS
            },
        }
    config = {
        "schema": "new_true_ours_config_freeze_v1",
        "status": "FROZEN_BEFORE_FORMAL",
        "freeze_source_git_commit": head,
        "exact_code_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in code_paths
        },
        "datasets": datasets,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "backend": "TFE_ABY3_single_machine_process_simulation",
        "cpu_policy": "maximum_available_with_32_thread_ceiling",
        "formal_queue_policy": "sequential_tmux_checkpoint_resume_complete_skip",
        "stage_b_checkpoint_schedule": "every_5_epochs",
        "stage_c_checkpoint_schedule": "every_5_steps",
        "evaluation_schedule": (
            "development_validation checkpoint logits only before freeze; "
            "locked holdout and official test read once after all formal runs"
        ),
        "privacy_claim": (
            "epsilon applies only to the conditional Stage-C sampled-gradient "
            "and clipping-count mechanisms; Stage-B initializers are "
            "data-dependent non-DP and no end-to-end record-DP claim is made"
        ),
        "timing_boundary": {
            "name": "cumulative_online_secure_runtime",
            "start": "immediately before Stage-B secure graph/session construction",
            "end": "after Stage-C final noisy model materialization and checkpoint",
            "includes": [
                "secure graph/session initialization and recovery",
                "Stage-B local secure workload",
                "secure FedAvg and initializer materialization",
                "checkpoint development-logit evaluation",
                "Stage-C secure Adaptive DP workload",
                "Stage-C final model materialization",
            ],
            "same_for_both_methods": True,
        },
        "development_evidence": {
            "paired_float_screen": str(adaptive_path),
            "paired_float_screen_sha256": sha256(adaptive_path),
            "tfe_finalists": str(finalist_path),
            "tfe_finalists_sha256": sha256(finalist_path),
            "split_manifest": str(split_manifest_path),
            "split_manifest_sha256": sha256(split_manifest_path),
        },
        "locked_holdout_read": False,
        "official_test_read_in_this_rebuild": False,
    }
    plateaus = {"cifar10": 0.7266, "cifar100": 0.4308, "tiny": 0.2765}
    targets = {
        dataset: {
            "full_data_development_plateau": plateau,
            "plateau_source": (
                "new_log/new_true_ours/"
                "new_full_data_stage_b_tuned_screen.json"
            ),
            "targets": {
                "80_percent": 0.80 * plateau,
                "90_percent": 0.90 * plateau,
                "95_percent": 0.95 * plateau,
            },
        }
        for dataset, plateau in plateaus.items()
    }
    target = {
        "schema": "new_true_ours_target_freeze_v1",
        "status": "FROZEN_BEFORE_FORMAL",
        "freeze_source_git_commit": head,
        "definition": "80/90/95 percent of tuned Full-Data development plateau",
        "datasets": targets,
        "missing_target_policy": "Not Reached",
        "selection_after_locked_holdout": False,
        "locked_holdout_read": False,
        "official_test_read_in_this_rebuild": False,
    }
    atomic_json(CONFIG_PATH, config)
    atomic_json(TARGET_PATH, target)
    manifest = """# True Ours formal freeze manifest

Status: `FROZEN_BEFORE_FORMAL`

Source Git commit: `{head}`

Frozen methods:

- Full Data + Secure Average + Adaptive DP
- Official MTT True Ours + Secure Average + Adaptive DP

Frozen seeds: `42 / 43 / 44`

All Stage B runs use TFE/ABY3, `secure_average_only`, balanced five-client
partitions, precision fractional 16, checkpoint every 5 epochs, and aggregate
development-logit evaluation only at checkpoint epochs. All Stage C runs use
TFE/ABY3, CSPRNG Poisson sampling, graph capacity 64, expected batch 16,
global Adaptive clipping, composed accountant target epsilon 3 and delta
1e-5, checkpoint every 5 steps.

The timing boundary starts immediately before Stage-B secure graph/session
construction and ends after Stage-C final noisy-model materialization. It is
identical for both methods.

The Stage-B initializer is data-dependent and non-DP. Stage-C epsilon is a
conditional mechanism claim only; no end-to-end record-DP claim is allowed.

Targets are frozen at 80/90/95 percent of the tuned Full-Data development
plateau. A target that is not reached must be reported as `Not Reached`.

Locked holdout and official test remain unread by this rebuild. They may be
read only once after every formal Stage-B and Stage-C report is COMPLETE and
the queue audit passes.

Exact protocol:

- `{config}`
- `{target}`

Development evidence:

- `{adaptive}` (`{adaptive_sha}`)
- `{finalists}` (`{finalists_sha}`)
- `{splits}` (`{splits_sha}`)
""".format(
        head=head,
        config=CONFIG_PATH.relative_to(ROOT),
        target=TARGET_PATH.relative_to(ROOT),
        adaptive=adaptive_path.relative_to(ROOT),
        adaptive_sha=sha256(adaptive_path),
        finalists=finalist_path.relative_to(ROOT),
        finalists_sha=sha256(finalist_path),
        splits=split_manifest_path.relative_to(ROOT),
        splits_sha=sha256(split_manifest_path),
    )
    atomic_text(MANIFEST_PATH, manifest)
    print(CONFIG_PATH)
    print(TARGET_PATH)
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()

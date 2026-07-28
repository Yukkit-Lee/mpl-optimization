#!/usr/bin/env python3
"""Run paired seed42 Adaptive Stage-C finalists with real TFE/ABY3.

The Full-Data development initializer is a metadata-complete wrapper around
the already tuned plaintext weights.  It is explicitly non-formal and exists
only to validate the secure Stage-C graph before protocol freeze.  Formal runs
must consume post-freeze TFE Stage-B initializers instead.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from new_stage_b_secure_init import sha256_file


DATASETS = ("cifar10", "cifar100", "tiny")
METHODS = ("full_data", "true_ours")
TFE_PYTHON = Path("/root/autodl-tmp/new_envs/tfe/bin/python")
SCREEN_PATH = (
    ROOT / "new_log/new_true_ours/new_adaptive_stage_c_development_screen.json"
)
AUDIT_PATH = (
    ROOT / "new_log/new_true_ours/new_adaptive_stage_c_secure_finalists.json"
)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    os.replace(str(temporary), str(path))


def full_development_initializer(dataset, seed):
    source = (
        ROOT
        / "new_output/new_true_ours/stage_b_screen/full_data_tuned"
        / dataset
        / "new_full_data_tuned_{}_seed{}.npz".format(dataset, seed)
    )
    stage_b_cache = (
        ROOT
        / "new_cache/new_true_ours/full_data_stage_b"
        / dataset
        / "new_full_stage_b_{}_seed{}.npz".format(dataset, seed)
    )
    output = (
        ROOT
        / "new_output/new_true_ours/stage_b/full_data_finalist"
        / dataset
        / "seed{}".format(seed)
        / "new_full_data_development_initializer_{}_seed{}.npz".format(
            dataset, seed
        )
    )
    with np.load(str(source), allow_pickle=False) as model:
        if (
            str(model["initializer_source"].reshape(-1)[0])
            != "full_data_disjoint"
            or int(model["full_data_initializer_dependency"].reshape(-1)[0])
            != 1
            or int(model["formal_secure_result"].reshape(-1)[0]) != 0
        ):
            raise ValueError("Full development initializer lineage rejected")
        atomic_npz(
            output,
            weight=np.asarray(model["weight"], dtype=np.float32),
            bias=np.asarray(model["bias"], dtype=np.float32),
            initializer_source=np.asarray(["full_data_disjoint"]),
            full_data_initializer_dependency=np.asarray([1], dtype=np.int64),
            feature_cache_sha256=np.asarray(
                [sha256_file(str(stage_b_cache))]
            ),
            source_initializer_path=np.asarray([str(source)]),
            source_initializer_sha256=np.asarray([sha256_file(str(source))]),
            formal_secure_result=np.asarray([0], dtype=np.int64),
            development_only=np.asarray([1], dtype=np.int64),
        )
    return output


def true_initializer(dataset, seed):
    return (
        ROOT
        / "new_output/new_true_ours/stage_b/true_ours_finalist"
        / dataset
        / "seed{}".format(seed)
        / "new_secure_stage_b_candidate_nokd_{}_balanced_seed{}.npz".format(
            dataset, seed
        )
    )


def paths(dataset, method, seed):
    if method == "full_data":
        cache = (
            ROOT
            / "new_cache/new_true_ours/full_data_stage_c"
            / dataset
            / "new_full_stage_c_{}_seed{}.npz".format(dataset, seed)
        )
        initializer = full_development_initializer(dataset, seed)
        cache_mode = "memmap"
        lineage = "full_data_disjoint"
    else:
        cache = (
            ROOT
            / "new_cache/new_true_ours/official_mtt/stage_c"
            / dataset
            / "new_true_ours_stage_c_{}_seed{}.npz".format(dataset, seed)
        )
        initializer = true_initializer(dataset, seed)
        cache_mode = "npz"
        lineage = "official_mtt_only"
    output = (
        ROOT
        / "new_output/new_true_ours/stage_c/adaptive_finalist"
        / method
        / dataset
        / "seed{}".format(seed)
    )
    checkpoint = (
        ROOT
        / "new_checkpoints/new_true_ours/stage_c/adaptive_finalist"
        / method
        / dataset
        / "seed{}".format(seed)
        / "new_stage_c"
    )
    log = (
        ROOT
        / "new_log/new_true_ours/txt"
        / "new_adaptive_stage_c_finalist_{}_{}_seed{}.log".format(
            method, dataset, seed
        )
    )
    report = output / (
        "new_secure_stage_c_report_{}_adaptive_dp_seed{}.json".format(
            dataset, seed
        )
    )
    return cache, initializer, cache_mode, lineage, output, checkpoint, log, report


def validate_report(report_path, dataset, method, seed, config, noise):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    expected_lineage = (
        "full_data_disjoint" if method == "full_data" else "official_mtt_only"
    )
    privacy = report["privacy"]
    security = report["security"]
    evaluation = report["evaluation"]
    actual = report["configuration"]
    checks = {
        "status_complete": report.get("status") == "COMPLETE",
        "dataset_seed_mode": (
            report.get("dataset") == dataset
            and int(report.get("seed")) == int(seed)
            and report.get("mode") == "adaptive_dp"
        ),
        "tfe_aby3_no_fallback": (
            security.get("backend") == "tfe_mpc"
            and security.get("protocol") == "ABY3"
            and security.get("silent_plaintext_fallback") is False
        ),
        "lineage": (
            actual.get("input_lineage") == expected_lineage
            and report["initializer_lineage"].get("validated") is True
        ),
        "poisson_csprng": (
            privacy.get("sampling_mode") == "poisson"
            and privacy.get("sampling_rng") == "os_csprng"
            and int(privacy.get("expected_batch_size")) == 16
            and int(privacy.get("graph_batch_capacity")) == 64
        ),
        "epsilon": float(
            privacy.get("stage_c_training_feature_record_epsilon")
        )
        <= 3.000001,
        "conditional_dp_claim_only": (
            privacy.get("initializer_privacy_status")
            == "data_dependent_non_dp"
            and privacy.get("released_model_record_dp_valid") is False
            and privacy.get("end_to_end_record_dp_claimed") is False
        ),
        "development_only_no_leakage": (
            evaluation.get("split") == "development_validation"
            and evaluation.get("locked_holdout_read") is False
            and evaluation.get("official_test_read") is False
        ),
        "matched_config": all(
            abs(float(actual[key]) - float(config[key])) <= 1e-12
            for key in (
                "steps",
                "learning_rate",
                "clip_c",
                "gamma",
                "eta_c",
                "c_min",
                "c_max",
                "warmup_steps",
                "update_interval",
                "count_noise_multiplier",
            )
        )
        and abs(float(actual["noise_multiplier"]) - float(noise)) <= 1e-12,
        "complete_history": (
            len(report.get("history", [])) == int(config["steps"])
        ),
        "finite_nonboundary_c": (
            np.isfinite(float(report.get("final_c")))
            and float(config["c_min"]) < float(report.get("final_c"))
            and float(report.get("final_c")) < float(config["c_max"])
        ),
    }
    if not all(checks.values()):
        raise ValueError(
            "Secure finalist validation failed for {} {}: {}".format(
                method, dataset, checks
            )
        )
    return report, checks


def main():
    if not TFE_PYTHON.is_file():
        raise FileNotFoundError(str(TFE_PYTHON))
    screen = json.loads(SCREEN_PATH.read_text(encoding="utf-8"))
    if (
        screen.get("status") != "COMPLETE"
        or screen.get("schema")
        != "new_adaptive_stage_c_development_screen_v2_paired"
        or screen.get("locked_holdout_read") is not False
        or screen.get("official_test_read") is not False
    ):
        raise ValueError("Paired Adaptive development screen is not valid")
    screen_by_dataset = {
        row["dataset"]: row for row in screen["results"]
    }
    rows = []
    seed = 42
    for dataset in DATASETS:
        selected = screen_by_dataset[dataset]
        config = dict(selected["best_config"])
        config["clip_c"] = config.pop("initial_c")
        for method in METHODS:
            noise = selected["best_result"]["methods"][method][
                "noise_multiplier"
            ]
            (
                cache,
                initializer,
                cache_mode,
                lineage,
                output,
                checkpoint,
                log,
                report_path,
            ) = paths(dataset, method, seed)
            if report_path.is_file():
                report, checks = validate_report(
                    report_path, dataset, method, seed, config, noise
                )
                print(
                    "[SKIP COMPLETE] {} {}".format(method, dataset),
                    flush=True,
                )
            else:
                output.mkdir(parents=True, exist_ok=True)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    str(TFE_PYTHON),
                    str(ROOT / "new_stage_c_secure_dpsgd.py"),
                    "--feature_cache",
                    str(cache),
                    "--feature_cache_mode",
                    cache_mode,
                    "--initializer",
                    str(initializer),
                    "--dataset",
                    dataset,
                    "--mode",
                    "adaptive_dp",
                    "--output_dir",
                    str(output),
                    "--checkpoint_prefix",
                    str(checkpoint),
                    "--seed",
                    str(seed),
                    "--batch_size",
                    "64",
                    "--expected_batch_size",
                    "16",
                    "--sampling_mode",
                    "poisson",
                    "--steps",
                    str(config["steps"]),
                    "--learning_rate",
                    str(config["learning_rate"]),
                    "--noise_multiplier",
                    str(noise),
                    "--count_noise_multiplier",
                    str(config["count_noise_multiplier"]),
                    "--clip_c",
                    str(config["clip_c"]),
                    "--clipping_scope",
                    "global",
                    "--gamma",
                    str(config["gamma"]),
                    "--eta_c",
                    str(config["eta_c"]),
                    "--c_min",
                    str(config["c_min"]),
                    "--c_max",
                    str(config["c_max"]),
                    "--warmup_steps",
                    str(config["warmup_steps"]),
                    "--update_interval",
                    str(config["update_interval"]),
                    "--precision_fractional",
                    "16",
                    "--matmul_chunk_size",
                    "64",
                    "--checkpoint_interval",
                    "5",
                    "--initializer_privacy_status",
                    "data_dependent_non_dp",
                    "--input_lineage",
                    lineage,
                    "--evaluation_split",
                    "development_validation",
                ]
                started = time.time()
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        "\nRUN_START {}\nCOMMAND {}\n".format(
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            " ".join(command),
                        )
                    )
                    handle.flush()
                    completed = subprocess.run(
                        command,
                        cwd=str(ROOT),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        "{} {} failed with exit {}".format(
                            method, dataset, completed.returncode
                        )
                    )
                report, checks = validate_report(
                    report_path, dataset, method, seed, config, noise
                )
                print(
                    "[COMPLETE] {} {} wall={:.2f}s".format(
                        method, dataset, time.time() - started
                    ),
                    flush=True,
                )
            history = report["history"]
            development = [
                row["development_validation_accuracy"]
                for row in history
                if row.get("development_validation_accuracy") is not None
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "report_path": str(report_path),
                    "report_sha256": sha256_file(str(report_path)),
                    "initializer_path": str(initializer),
                    "initializer_sha256": sha256_file(str(initializer)),
                    "development_initializer_is_formal_secure": (
                        method == "true_ours"
                    ),
                    "configuration": config,
                    "noise_multiplier": noise,
                    "accounted_epsilon": report["privacy"][
                        "stage_c_training_feature_record_epsilon"
                    ],
                    "final_c": report["final_c"],
                    "final_development_validation_accuracy": development[-1],
                    "runtime_seconds": report["runtime_seconds"],
                    "checks": checks,
                }
            )
    payload = {
        "schema": "new_adaptive_stage_c_secure_finalists_v1",
        "status": "COMPLETE",
        "backend": "TFE_ABY3",
        "seed": seed,
        "paired_screen_sha256": sha256_file(str(SCREEN_PATH)),
        "rows": rows,
        "full_data_stage_b_initializer_scope": (
            "development-only metadata wrapper around tuned plaintext weights; "
            "not a formal secure Stage-B result"
        ),
        "formal_requirement": (
            "post-freeze runs must consume real TFE Stage-B initializers for "
            "both methods"
        ),
        "locked_holdout_read": False,
        "official_test_read": False,
    }
    atomic_json(AUDIT_PATH, payload)
    print("audit={}".format(AUDIT_PATH), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

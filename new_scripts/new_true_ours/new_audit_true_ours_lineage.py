#!/usr/bin/env python3
"""Audit True Ours feature/initializer lineage before and after Stage B."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATASETS = ("cifar10", "cifar100", "tiny")
SEEDS = (42, 43, 44)
LOG_ROOT = ROOT / "new_log/new_true_ours"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def resolve_initializer(dataset, seed):
    """Resolve only this rebuild's isolated Stage-B output namespaces."""
    filename = (
        "new_secure_stage_b_candidate_nokd_{}_balanced_seed{}.npz".format(
            dataset, seed
        )
    )
    candidates = [
        ROOT
        / "new_output/new_true_ours/stage_b/formal"
        / dataset
        / "seed{}".format(seed)
        / filename,
        ROOT
        / "new_output/new_true_ours/stage_b/true_ours_finalist"
        / dataset
        / "seed{}".format(seed)
        / filename,
        ROOT
        / "new_output/new_true_ours/stage_b/true_ours"
        / dataset
        / "seed{}".format(seed)
        / filename,
    ]
    existing = [path for path in candidates if path.is_file()]
    return (
        existing[0] if existing else candidates[0],
        candidates,
        {str(path): sha256_file(path) for path in existing},
    )


def main():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    forbidden_fragment = os.path.join(
        "new_output", "new_incremental_comparison", "stage_b"
    )
    stage_b_code = (ROOT / "new_stage_b_secure_init.py").read_text(
        encoding="utf-8"
    )
    stage_c_code = (ROOT / "new_stage_c_secure_dpsgd.py").read_text(
        encoding="utf-8"
    )
    hard_guard_present = (
        "official_mtt_only" in stage_b_code
        and "full_data_initializer_dependency" in stage_b_code
        and "Forbidden legacy Full-Data initializer dependency" in stage_c_code
    )
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            cache = (
                ROOT / "new_cache/new_true_ours/official_mtt/stage_b" / dataset
                / "new_true_ours_stage_b_{}_seed{}.npz".format(dataset, seed)
            )
            (
                initializer,
                initializer_candidates,
                existing_initializer_hashes,
            ) = resolve_initializer(dataset, seed)
            with np.load(str(cache), allow_pickle=False) as payload:
                source = str(payload["initializer_source"].reshape(-1)[0])
                dependency = bool(
                    int(
                        payload[
                            "full_data_initializer_dependency"
                        ].reshape(-1)[0]
                    )
                )
                split_path = str(payload["split_manifest_path"].reshape(-1)[0])
                split_sha = str(
                    payload["split_manifest_sha256"].reshape(-1)[0]
                )
            cache_resolved = str(cache.resolve())
            cache_valid = (
                source == "official_mtt_only"
                and not dependency
                and forbidden_fragment not in cache_resolved
                and not cache.is_symlink()
                and Path(split_path).is_file()
                and sha256_file(split_path) == split_sha
            )
            initializer_exists = initializer.is_file()
            initializer_valid = False
            initializer_sha = None
            if initializer_exists:
                with np.load(str(initializer), allow_pickle=False) as payload:
                    init_source = str(
                        payload["initializer_source"].reshape(-1)[0]
                    )
                    init_dependency = bool(
                        int(
                            payload[
                                "full_data_initializer_dependency"
                            ].reshape(-1)[0]
                        )
                    )
                initializer_valid = (
                    init_source == "official_mtt_only"
                    and not init_dependency
                    and forbidden_fragment not in str(initializer.resolve())
                    and not initializer.is_symlink()
                )
                initializer_sha = sha256_file(initializer)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "stage_b_input": str(cache),
                    "stage_b_input_sha256": sha256_file(cache),
                    "initializer_path": str(initializer),
                    "initializer_candidates": [
                        str(path) for path in initializer_candidates
                    ],
                    "existing_initializer_hashes": existing_initializer_hashes,
                    "initializer_sha256": initializer_sha,
                    "initializer_exists": initializer_exists,
                    "initializer_source": source,
                    "full_data_dependency": dependency,
                    "cache_lineage_valid": cache_valid,
                    "initializer_lineage_valid": initializer_valid,
                    "valid_true_ours_pre_run": cache_valid and hard_guard_present,
                    "valid_true_ours": (
                        cache_valid and hard_guard_present and initializer_valid
                    ),
                }
            )
    payload = {
        "schema": "new_true_ours_lineage_audit_v1",
        "status": (
            "COMPLETE_POST_STAGE_B"
            if all(row["valid_true_ours"] for row in rows)
            else "PRE_STAGE_B_ASSET_GATE_PASS"
        ),
        "hard_lineage_guard_present": hard_guard_present,
        "forbidden_full_data_initializer_fragment": forbidden_fragment,
        "all_pre_run_assets_valid": all(
            row["valid_true_ours_pre_run"] for row in rows
        ),
        "all_initializers_valid": all(row["valid_true_ours"] for row in rows),
        "rows": rows,
    }
    json_path = LOG_ROOT / "new_true_ours_lineage_audit.json"
    atomic_json(json_path, payload)
    lines = [
        "# True Ours lineage audit",
        "",
        "- Status: {}".format(payload["status"]),
        "- Hard lineage guard present: {}".format(hard_guard_present),
        "- Full-Data initializer paths are forbidden for True Ours.",
        "",
        "| Dataset | Seed | Stage B input | Initializer exists | Full Data dependency | Pre-run valid | True Ours valid |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {seed} | `{stage_b_input}` | {initializer_exists} | "
            "{full_data_dependency} | {valid_true_ours_pre_run} | "
            "{valid_true_ours} |".format(**row)
        )
    (LOG_ROOT / "new_true_ours_lineage_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        "lineage_status={} rows={} all_pre_run_assets_valid={}".format(
            payload["status"], len(rows), payload["all_pre_run_assets_valid"]
        )
    )


if __name__ == "__main__":
    main()

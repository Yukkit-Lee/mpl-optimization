#!/usr/bin/env python3
"""Upgrade only Full-Data memmap metadata; preserve frozen split/MTT bytes."""

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from new_prepare_true_ours_assets import atomic_npz, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("cifar10", "cifar100", "tiny")
SEEDS = (42, 43, 44)
STAGES = ("stage_b", "stage_c")


def metadata_path(dataset, seed, stage):
    directory = "full_data_stage_b" if stage == "stage_b" else "full_data_stage_c"
    stem = "new_full_stage_b" if stage == "stage_b" else "new_full_stage_c"
    return (
        ROOT
        / "new_cache/new_true_ours"
        / directory
        / dataset
        / "{}_{}_seed{}.npz".format(stem, dataset, seed)
    )


def preserved_paths():
    paths = []
    for dataset in DATASETS:
        for seed in SEEDS:
            paths.append(
                ROOT
                / "new_cache/new_true_ours/splits"
                / dataset
                / "new_true_ours_split_{}_seed{}.npz".format(dataset, seed)
            )
            for stage in STAGES:
                paths.append(
                    ROOT
                    / "new_cache/new_true_ours/official_mtt"
                    / stage
                    / dataset
                    / "new_true_ours_{}_{}_seed{}.npz".format(
                        stage, dataset, seed
                    )
                )
    return paths


def main():
    frozen_before = {
        str(path): sha256_file(path) for path in preserved_paths()
    }
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            split_path = (
                ROOT
                / "new_cache/new_true_ours/splits"
                / dataset
                / "new_true_ours_split_{}_seed{}.npz".format(dataset, seed)
            )
            split_sha = sha256_file(split_path)
            for stage in STAGES:
                path = metadata_path(dataset, seed, stage)
                before_sha = sha256_file(path)
                with np.load(str(path), allow_pickle=False) as source:
                    arrays = {
                        name: np.asarray(source[name]) for name in source.files
                    }
                arrays.update(
                    {
                        "initializer_source": np.asarray(
                            ["full_data_disjoint"]
                        ),
                        "full_data_initializer_dependency": np.asarray(
                            [1], dtype=np.int64
                        ),
                        "split_manifest_path": np.asarray([str(split_path)]),
                        "split_manifest_sha256": np.asarray([split_sha]),
                    }
                )
                atomic_npz(path, **arrays)
                with np.load(str(path), allow_pickle=False) as upgraded:
                    assert (
                        str(upgraded["initializer_source"].reshape(-1)[0])
                        == "full_data_disjoint"
                    )
                    assert (
                        int(
                            upgraded[
                                "full_data_initializer_dependency"
                            ].reshape(-1)[0]
                        )
                        == 1
                    )
                    assert (
                        str(
                            upgraded["split_manifest_sha256"].reshape(-1)[0]
                        )
                        == split_sha
                    )
                    assert int(upgraded["test_loaded"].reshape(-1)[0]) == 0
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "stage": stage,
                        "metadata_path": str(path),
                        "before_sha256": before_sha,
                        "after_sha256": sha256_file(path),
                        "split_manifest_path": str(split_path),
                        "split_manifest_sha256": split_sha,
                        "test_loaded": False,
                    }
                )
    frozen_after = {
        str(path): sha256_file(path) for path in preserved_paths()
    }
    if frozen_after != frozen_before:
        changed = [
            path
            for path in frozen_before
            if frozen_before[path] != frozen_after.get(path)
        ]
        raise AssertionError(
            "frozen split/Official-MTT files changed: {}".format(changed)
        )
    payload = {
        "schema": "new_full_data_lineage_metadata_upgrade_v1",
        "status": "COMPLETE",
        "rows_expected": 18,
        "rows_completed": len(rows),
        "frozen_split_files_rewritten": False,
        "official_mtt_cache_files_rewritten": False,
        "preserved_files_verified": len(frozen_before),
        "preserved_files_sha256": frozen_after,
        "rows": rows,
    }
    if len(rows) != 18:
        raise AssertionError("expected 18 upgraded metadata files")
    output = (
        ROOT
        / "new_log/new_true_ours/new_full_data_lineage_metadata_upgrade.json"
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("status=COMPLETE rows=18 output={}".format(output))


if __name__ == "__main__":
    main()

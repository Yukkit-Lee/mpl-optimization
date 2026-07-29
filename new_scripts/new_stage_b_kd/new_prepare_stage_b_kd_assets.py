#!/usr/bin/env python3
"""Create audited Official-MTT local/anchor manifests for seed42 only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from new_stage_b_kd_core import (
    ANCHOR_SIZES,
    DATASETS,
    LOG_ROOT,
    SEED,
    atomic_json,
    load_dataset_assets,
    mtt_anchor_split,
    sha256_array,
    source_commit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=SEED, choices=[SEED])
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        assets = load_dataset_assets(dataset, args.seed)
        variants = []
        for anchor_size in ANCHOR_SIZES:
            local_ids, anchor_ids = mtt_anchor_split(
                assets.y_mtt, assets.num_classes, anchor_size, args.seed
            )
            local_histogram = np.bincount(
                assets.y_mtt[local_ids], minlength=assets.num_classes
            )
            anchor_histogram = np.bincount(
                assets.y_mtt[anchor_ids], minlength=assets.num_classes
            )
            variants.append(
                {
                    "anchor_samples_per_class": anchor_size,
                    "local_samples_per_class": 48 - anchor_size,
                    "local_count": int(len(local_ids)),
                    "anchor_count": int(len(anchor_ids)),
                    "local_ids": local_ids.astype(int).tolist(),
                    "anchor_ids": anchor_ids.astype(int).tolist(),
                    "local_ids_sha256": sha256_array(local_ids),
                    "anchor_ids_sha256": sha256_array(anchor_ids),
                    "intersection_size": int(
                        len(np.intersect1d(local_ids, anchor_ids))
                    ),
                    "union_count": int(
                        len(np.unique(np.concatenate([local_ids, anchor_ids])))
                    ),
                    "local_class_histogram": local_histogram.astype(int).tolist(),
                    "anchor_class_histogram": anchor_histogram.astype(int).tolist(),
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "seed": args.seed,
                "num_classes": assets.num_classes,
                "official_mtt_cache": str(assets.mtt_cache_path),
                "official_mtt_cache_sha256": assets.mtt_cache_sha256,
                "original_split_manifest": str(assets.original_split_path),
                "original_split_manifest_sha256": assets.original_split_sha256,
                "development_ids_sha256": assets.development_ids_sha256,
                "development_count": int(len(assets.y_dev)),
                "development_source_namespace": "original_train_disjoint_split",
                "mtt_source_namespace": "fixed_official_mtt_upstream_asset",
                "initializer_source": "official_mtt_only",
                "full_data_initializer_dependency": False,
                "variants": variants,
            }
        )

    payload = {
        "schema": "new_stage_b_kd_split_manifest_v1",
        "status": "COMPLETE",
        "source_commit": source_commit(),
        "seed_policy": [SEED],
        "preferred_split": {
            "local_samples_per_class": 40,
            "anchor_samples_per_class": 8,
        },
        "pairwise_local_anchor_disjoint": True,
        "development_not_loaded_into_training_or_anchor": True,
        "locked_holdout_read": False,
        "official_test_read": False,
        "rows": rows,
    }
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    output = LOG_ROOT / "new_stage_b_kd_split_manifest.json"
    atomic_json(output, payload)

    lines = [
        "# Stage-B MTT-anchor split audit",
        "",
        "- Status: COMPLETE",
        "- Current task seed: 42 only",
        "- Official-MTT local and anchor IDs are mutually exclusive and exhaustive.",
        "- Development validation comes from the frozen original-training split and is never loaded into local training or anchor KD.",
        "- Locked holdout and official test were not read.",
        "",
        "| Dataset | Anchor/class | Local | Anchor | Intersection | Union | Dev |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for variant in row["variants"]:
            lines.append(
                "| {dataset} | {anchor} | {local_count} | {anchor_count} | "
                "{intersection} | {union} | {dev} |".format(
                    dataset=row["dataset"],
                    anchor=variant["anchor_samples_per_class"],
                    local_count=variant["local_count"],
                    anchor_count=variant["anchor_count"],
                    intersection=variant["intersection_size"],
                    union=variant["union_count"],
                    dev=row["development_count"],
                )
            )
    audit = LOG_ROOT / "new_stage_b_kd_split_audit.md"
    audit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("manifest={} rows={} seed=42".format(output, len(rows)))
    print("audit={} locked_holdout_read=False official_test_read=False".format(audit))


if __name__ == "__main__":
    main()

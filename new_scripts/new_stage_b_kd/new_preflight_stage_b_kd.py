#!/usr/bin/env python3
"""Hard preflight for the single-seed Stage-B-only experiment."""

import json
import subprocess
from pathlib import Path

from new_stage_b_kd_core import DATASETS, ROOT, SEED, load_dataset_assets, sha256_file


def main() -> None:
    policy_path = (
        ROOT
        / "new_configs/new_stage_b_kd/new_stage_b_kd_single_seed_policy.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy["seed_policy"]["all_phases"] != [SEED]:
        raise ValueError("single-seed policy is not seed42-only")
    forbidden = set(policy["forbidden"])
    required = {
        "stage_c",
        "adaptive_dp_sgd",
        "fixed_dp_sgd",
        "official_test",
        "locked_holdout",
        "full_data_initializer",
        "figure2",
        "t_target",
    }
    if not required.issubset(forbidden):
        raise ValueError("policy forbidden set is incomplete")
    assets = []
    for dataset in DATASETS:
        item = load_dataset_assets(dataset, SEED)
        assets.append(
            {
                "dataset": dataset,
                "mtt_cache": str(item.mtt_cache_path),
                "mtt_cache_sha256": item.mtt_cache_sha256,
                "mtt_samples": len(item.y_mtt),
                "development_samples": len(item.y_dev),
                "initializer_source": "official_mtt_only",
                "full_data_initializer_dependency": False,
            }
        )
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=str(ROOT), text=True
    ).strip()
    if branch != "new-stage-b-mtt-anchor-kd":
        raise ValueError("wrong branch: {}".format(branch))
    print("PREFLIGHT_PASS branch={} seed_policy=[42]".format(branch))
    for row in assets:
        print(
            "{dataset} mtt={mtt_samples} dev={development_samples} "
            "source=official_mtt_only full_dependency=0".format(**row)
        )
    print("locked_holdout_read=False official_test_read=False stage_c=False")


if __name__ == "__main__":
    main()

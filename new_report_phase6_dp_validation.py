#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用途：报告 CIFAR-10 DP-MTT 首轮完整 validation 的一次冻结 test 结果。"""

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
BASE = (
    ROOT
    / "new_cache/new_secure_feature/new_tuning_phase6_dp_mtt_cifar10_round2"
    / "new_seed42/new_encoder_frozen/dp_mtt/cifar10"
)
METADATA = BASE / "new_shared_feature_metadata_cifar10_dp_mtt_seed42.json"
EXPERT_MANIFEST = (
    ROOT
    / "new_cache/new_secure_feature/new_tuning_phase6_dp_mtt_cifar10_round2"
    / "new_seed42/new_trajectories/cifar10"
    / "new_stage_a_dp_trajectory_manifest_cifar10.json"
)
OUTPUT = ROOT / "new_log/new_secure_feature/json/new_phase6_dp_validation_result.json"


def main() -> int:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    manifest = json.loads(EXPERT_MANIFEST.read_text(encoding="utf-8"))
    accuracy = float(metadata["final_test_accuracy"])
    public_mean = 0.6782666666666667
    gap = public_mean - accuracy
    status = "QUALITY_GATE_PASS" if gap <= 0.10 else "NOT_READY_FOR_PAPER"
    payload = {
        "schema": "new_phase6_dp_validation_result_v1",
        "status": status,
        "dataset": "cifar10",
        "seed": 42,
        "stage_a_epsilon": manifest["stage_a_epsilon"],
        "stage_a_delta": manifest["stage_a_delta"],
        "num_expert_trajectories": manifest["num_expert_trajectories"],
        "dp_mtt_test_accuracy": accuracy,
        "public_mtt_mean_accuracy": public_mean,
        "absolute_gap": gap,
        "maximum_allowed_gap": 0.10,
        "test_tuning": False,
        "next_step": (
            "freeze formal DP-MTT configuration"
            if status == "QUALITY_GATE_PASS"
            else "continue validation tuning of experts/trajectories/MTT"
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "QUALITY_GATE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

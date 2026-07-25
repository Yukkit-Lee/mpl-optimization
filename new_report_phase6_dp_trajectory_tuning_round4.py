#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report the test-free 5000-step refinement of the selected DP-MTT candidate."""

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
JSON_ROOT = ROOT / "new_log/new_secure_feature/json"
BASELINE = JSON_ROOT / "new_phase6_dp_round3_late10_public_reference.json"
REFINED = JSON_ROOT / "new_phase6_dp_round4_late10_5000_public_reference.json"
OUTPUT = JSON_ROOT / "new_phase6_dp_trajectory_tuning_round4_result.json"


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    refined = json.loads(REFINED.read_text(encoding="utf-8"))
    before = float(baseline["public_reference_accuracy"])
    after = float(refined["public_reference_accuracy"])
    gain = after - before
    public_mean = 0.6782666666666667
    gap = public_mean - after
    ready = gap <= 0.10
    result = {
        "schema": "new_phase6_dp_trajectory_tuning_round4_result_v1",
        "status": (
            "REFERENCE_GATE_READY"
            if ready
            else "TUNING_IMPROVED_NOT_READY"
            if gain >= 0.01
            else "TUNING_NO_MATERIAL_GAIN"
        ),
        "dataset": "cifar10",
        "seed": 42,
        "selection_metric": "external_public_mtt_reference_accuracy",
        "round3_accuracy_2500": before,
        "round4_accuracy_5000": after,
        "absolute_gain": gain,
        "public_mtt_formal_mean": public_mean,
        "remaining_gap": gap,
        "within_ten_percentage_points": ready,
        "test_loaded": False,
        "next_step": (
            "freeze configuration and release one readonly test"
            if ready
            else "tune DP expert utility using public reference only"
        ),
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

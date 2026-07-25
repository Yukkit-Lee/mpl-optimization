#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report test-free DP-MTT trajectory-window tuning round 3."""

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
JSON_ROOT = ROOT / "new_log/new_secure_feature/json"
OUTPUT = JSON_ROOT / "new_phase6_dp_trajectory_tuning_round3_result.json"
BASELINE = JSON_ROOT / "new_phase6_dp_round2_public_reference.json"
CANDIDATES = {
    name: JSON_ROOT
    / "new_phase6_dp_round3_{}_public_reference.json".format(name)
    for name in ("late3", "late10")
}


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_score = float(baseline["public_reference_accuracy"])
    rows = []
    for name, path in CANDIDATES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "candidate": name,
                "public_reference_accuracy": float(
                    payload["public_reference_accuracy"]
                ),
                "test_loaded": bool(payload["test_loaded"]),
                "result_path": str(path),
            }
        )
    rows.sort(key=lambda row: row["public_reference_accuracy"], reverse=True)
    best = rows[0]
    gain = float(best["public_reference_accuracy"] - baseline_score)
    status = "TUNING_IMPROVED" if gain >= 0.02 else "TUNING_NO_MATERIAL_GAIN"
    result = {
        "schema": "new_phase6_dp_trajectory_tuning_round3_result_v1",
        "status": status,
        "dataset": "cifar10",
        "seed": 42,
        "selection_metric": "external_public_mtt_reference_accuracy",
        "baseline_public_reference_accuracy": baseline_score,
        "candidates": rows,
        "selected_candidate": best["candidate"],
        "selected_public_reference_accuracy": best["public_reference_accuracy"],
        "absolute_reference_gain": gain,
        "material_gain_threshold": 0.02,
        "test_loaded": False,
        "next_step": (
            "continue selected candidate to 5000 iterations"
            if status == "TUNING_IMPROVED"
            else "tune DP expert utility before another test release"
        ),
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

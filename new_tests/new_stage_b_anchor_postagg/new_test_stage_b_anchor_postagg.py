#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "new_scripts/new_stage_b_anchor_postagg"))

from new_stage_b_anchor_postagg_core import (  # noqa: E402
    model_sha256,
    select_registered_local_client,
    select_viable_causal_group,
    step_efficiency,
    train_anchor_ce_trajectory,
)


def toy_problem():
    x = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
        dtype=np.float32,
    )
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    weight = np.zeros((2, 2), dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)
    return x, y, weight, bias


def test_zero_steps_preserves_initializer():
    x, y, weight, bias = toy_problem()
    final_weight, final_bias, timing = train_anchor_ce_trajectory(
        x, y, x, y, weight, bias, 0, 0.1, 0.0
    )
    assert np.array_equal(final_weight, weight)
    assert np.array_equal(final_bias, bias)
    assert len(timing["history"]) == 1
    assert timing["history"][0]["step"] == 0


def test_anchor_ce_improves_toy_accuracy():
    x, y, weight, bias = toy_problem()
    _, _, timing = train_anchor_ce_trajectory(
        x, y, x, y, weight, bias, 20, 0.5, 0.0
    )
    assert timing["final_metrics"]["accuracy"] == 1.0
    assert len(timing["history"]) == 21


def test_step_efficiency_uses_c5_target():
    history = [
        {"step": 0, "development_accuracy": 0.2, "elapsed_refinement_seconds": 0.0},
        {"step": 1, "development_accuracy": 0.5, "elapsed_refinement_seconds": 0.1},
        {"step": 2, "development_accuracy": 0.8, "elapsed_refinement_seconds": 0.2},
    ]
    values = step_efficiency(history, 0.8)
    assert values["steps_at_80"] == 2
    assert values["steps_at_90"] == 2
    assert values["steps_at_95"] == 2


def test_local_rule_is_development_independent_fixed_client():
    assert select_registered_local_client() == 0


def test_model_hash_changes_with_parameters():
    _, _, weight, bias = toy_problem()
    original = model_sha256(weight, bias)
    changed = weight.copy()
    changed[0, 0] = 1.0
    assert original != model_sha256(changed, bias)


def test_causal_selection_rejects_pathologically_low_c5_endpoint():
    def group(c1, c3, c5, steps):
        return {
            "random": {"accuracy": c1},
            "local": {"accuracy": c3},
            "average": {
                "accuracy": c5,
                "steps": steps,
                "mu": 0.0,
                "learning_rate": 0.1,
            },
        }

    low = group(0.10, 0.10, 0.20, 5)
    useful = group(0.55, 0.30, 0.58, 40)
    assert select_viable_causal_group([low, useful]) is useful

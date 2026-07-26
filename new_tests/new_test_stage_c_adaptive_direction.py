#!/usr/bin/env python3
"""Regression tests for the direction of the adaptive clipping update."""

from new_stage_c_secure_dpsgd import adaptive_clip_direction_reference


def test_many_examples_below_threshold_decreases_c():
    updated = adaptive_clip_direction_reference(
        current_c=1.0, fraction_below=0.9, gamma=0.5, eta_c=0.05
    )
    assert updated < 1.0


def test_few_examples_below_threshold_increases_c():
    updated = adaptive_clip_direction_reference(
        current_c=1.0, fraction_below=0.1, gamma=0.5, eta_c=0.05
    )
    assert updated > 1.0


def test_fraction_equal_to_target_keeps_c():
    updated = adaptive_clip_direction_reference(
        current_c=1.0, fraction_below=0.5, gamma=0.5, eta_c=0.05
    )
    assert abs(updated - 1.0) < 1e-12

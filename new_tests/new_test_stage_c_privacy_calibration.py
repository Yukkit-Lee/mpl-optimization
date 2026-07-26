#!/usr/bin/env python3
"""Tests for fixed/adaptive target-epsilon calibration."""

from new_stage_c_privacy_calibration import calibrate_noise_multiplier


def test_fixed_calibration_hits_target():
    _, achieved = calibrate_noise_multiplier(
        target_epsilon=3.0,
        steps=10,
        sample_rate=0.04,
        delta=1e-5,
    )
    assert abs(achieved - 3.0) <= 1e-4


def test_adaptive_calibration_composes_count_mechanism():
    fixed_noise, _ = calibrate_noise_multiplier(
        target_epsilon=1.0,
        steps=20,
        sample_rate=0.004,
        delta=1e-5,
    )
    adaptive_noise, achieved = calibrate_noise_multiplier(
        target_epsilon=1.0,
        steps=20,
        sample_rate=0.004,
        delta=1e-5,
        count_noise=8.0,
        count_steps=3,
    )
    assert adaptive_noise >= fixed_noise
    assert abs(achieved - 1.0) <= 1e-4

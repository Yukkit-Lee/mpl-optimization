#!/usr/bin/env python3
"""Regression tests for the direction of the adaptive clipping update."""

from new_stage_c_secure_dpsgd import adaptive_clip_direction_reference
from new_stage_c_secure_dpsgd import poisson_batch_indices
from new_stage_c_secure_dpsgd import released_model_dp_valid


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


def test_data_dependent_non_dp_initializer_blocks_release_claim():
    assert not released_model_dp_valid("data_dependent_non_dp")
    assert released_model_dp_valid("data_independent")
    assert released_model_dp_valid("disjoint_public")
    assert released_model_dp_valid("dp_sanitized")


def test_poisson_batch_sampler_uses_independent_threshold_draws():
    zeros = lambda size: b"\x00" * size
    maximum = lambda size: b"\xff" * size
    assert poisson_batch_indices(10, 2, zeros).tolist() == list(range(10))
    assert poisson_batch_indices(10, 2, maximum).tolist() == []
    assert poisson_batch_indices(10, 10, maximum).tolist() == list(range(10))

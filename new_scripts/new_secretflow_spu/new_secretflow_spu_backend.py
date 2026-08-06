#!/usr/bin/env python3
"""SecretFlow 1.14 / source-SFL compatibility adapter for SPU simulation."""

from __future__ import annotations

from typing import Any


def register_official_sfl_strategies() -> dict[str, Any]:
    """Register the official source-SFL scheduler strategies explicitly.

    SecretFlow 1.14 lazily imports ``sfl.distributed`` for non-core modes, but
    source-SFL commit c383e40 currently exposes the strategy classes without an
    import-time registration side effect.  Use SecretFlow's public plugin hook
    instead of patching either installed package.
    """
    import ray._private
    try:
        from ray._private import ray_option_utils  # type: ignore[attr-defined]
        ray_option_location = "ray._private"
    except ImportError:
        # SFL's pinned pyproject selects Ray 2.52, whose own modules import
        # this API from ray._common.  Supply the old import alias expected by
        # the pinned SFL source without modifying either installed package.
        from ray._common import ray_option_utils

        ray._private.ray_option_utils = ray_option_utils
        ray_option_location = "ray._common (compatibility alias for source-SFL)"

    import secretflow.distributed as sfd
    import secretflow.distributed.primitive as primitive
    from secretflow.distributed.const import DISTRIBUTION_MODE
    from sfl.distributed.op_strategy import (
        InterConnStrategy,
        RayProdStrategy,
        SimulationStrategy,
    )

    requested = {
        DISTRIBUTION_MODE.SIMULATION: SimulationStrategy,
        DISTRIBUTION_MODE.RAY_PRODUCTION: RayProdStrategy,
        DISTRIBUTION_MODE.INTERCONNECTION: InterConnStrategy,
    }
    existing = primitive._sf_op_context._strategies
    missing = {mode: strategy for mode, strategy in requested.items() if mode not in existing}
    if missing:
        sfd.add_distribution_modes(missing)
    registered = primitive._sf_op_context._strategies
    if registered.get(DISTRIBUTION_MODE.SIMULATION) is not SimulationStrategy:
        raise RuntimeError("official SFL SimulationStrategy registration failed")
    return {
        "adapter": "SecretFlow public add_distribution_modes plugin hook",
        "simulation_strategy": f"{SimulationStrategy.__module__}.{SimulationStrategy.__name__}",
        "ray_option_utils": ray_option_location,
        "registered_modes": sorted(mode.name for mode in registered),
        "plaintext_debug_mode": False,
    }

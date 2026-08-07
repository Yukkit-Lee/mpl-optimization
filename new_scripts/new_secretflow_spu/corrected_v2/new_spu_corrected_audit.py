#!/usr/bin/env python3
"""Accounting, timing and cgroup measurements for corrected-v2 runs."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ACCOUNTANT_PYTHON = Path(
    os.environ.get(
        "NEW_ACCOUNTANT_PYTHON",
        "/root/autodl-tmp/new_envs/distillation/bin/python",
    )
)


def conditional_stage_c_epsilon(
    config: dict[str, Any], steps: int, data_size: int
) -> float:
    """Run the same Opacus RDP accountant used by the audited TFE path."""
    update_interval = int(config["update_interval"])
    warmup = int(config["warmup_steps"])
    count_steps = sum(
        step >= warmup and (step + 1) % update_interval == 0
        for step in range(int(steps))
    )
    command = [
        str(ACCOUNTANT_PYTHON),
        str(ROOT / "new_privacy_accountant.py"),
        "--steps", str(int(steps)),
        "--sample_rate", str(float(config["expected_batch"]) / float(data_size)),
        "--noise_multiplier", str(float(config["noise_multiplier"])),
        "--delta", str(float(config.get("delta", 1e-5))),
        "--count_noise", str(float(config["count_noise_multiplier"])),
        "--count_steps", str(count_steps),
    ]
    result = subprocess.run(
        command,
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["epsilon"])


def cgroup_snapshot() -> dict[str, int]:
    root = Path("/sys/fs/cgroup")
    cpu = {}
    for line in (root / "cpu.stat").read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        cpu[key] = int(value)
    memory_current = int((root / "memory.current").read_text().strip())
    peak_path = root / "memory.peak"
    memory_peak = int(peak_path.read_text().strip()) if peak_path.exists() else memory_current
    return {
        "monotonic_ns": time.monotonic_ns(),
        "cpu_usage_usec": int(cpu["usage_usec"]),
        "memory_current_bytes": memory_current,
        "memory_peak_bytes": memory_peak,
    }


def cgroup_delta(start: dict[str, int], end: dict[str, int], allocated_cpus: int) -> dict[str, float]:
    wall_seconds = (end["monotonic_ns"] - start["monotonic_ns"]) / 1e9
    cpu_seconds = (end["cpu_usage_usec"] - start["cpu_usage_usec"]) / 1e6
    denominator = max(wall_seconds * int(allocated_cpus), 1e-12)
    return {
        "wall_seconds": wall_seconds,
        "cgroup_cpu_seconds": cpu_seconds,
        "cgroup_cpu_utilization_fraction_of_allocated_capacity": cpu_seconds / denominator,
        "memory_peak_gib": end["memory_peak_bytes"] / float(1 << 30),
        "measurement_scope": "whole_cgroup; run only without concurrent experiments",
    }


class PhaseTimer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def add(self, phase: str, elapsed: float) -> None:
        self.seconds[phase] = self.seconds.get(phase, 0.0) + float(elapsed)

    def measure(self, phase: str):
        return _TimerContext(self, phase)


class _TimerContext:
    def __init__(self, parent: PhaseTimer, phase: str) -> None:
        self.parent = parent
        self.phase = phase

    def __enter__(self):
        self.started = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.parent.add(self.phase, time.monotonic() - self.started)

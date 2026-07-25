#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用途：Stage C 专用的正式 RDP 隐私会计封装。"""

from typing import Dict, Optional

from opacus.accountants import RDPAccountant


PRIVACY_LIMITATION = (
    "The reported privacy budget covers Stage C only. Stages A and B are "
    "data-dependent and are not included in the current record-level DP "
    "accounting; therefore, the reported epsilon does not constitute "
    "end-to-end DP for the original local datasets."
)


class StageCAccountant:
    """Gaussian sampled mechanism 使用 Opacus RDP；其他机制不错误复用其 ε。"""

    def __init__(
        self,
        noise_mechanism: str,
        noise_multiplier: float,
        sample_rate: float,
        delta: float,
    ) -> None:
        self.noise_mechanism = str(noise_mechanism)
        self.noise_multiplier = float(noise_multiplier)
        self.sample_rate = float(sample_rate)
        self.delta = float(delta)
        self.steps = 0
        self._accountant = RDPAccountant() if self.noise_mechanism == "gaussian" else None

    def step(self) -> None:
        self.steps += 1
        if self._accountant is not None:
            if self.noise_multiplier <= 0:
                raise ValueError("Gaussian DP requires noise_multiplier > 0")
            self._accountant.step(
                noise_multiplier=self.noise_multiplier,
                sample_rate=self.sample_rate,
            )

    def epsilon(self) -> Optional[float]:
        if self._accountant is None:
            return None
        return float(self._accountant.get_epsilon(delta=self.delta))

    def metadata(self) -> Dict:
        epsilon = self.epsilon()
        return {
            "privacy_scope": "stage_c_only",
            "stage_c_epsilon": epsilon if epsilon is not None else "N/A",
            "stage_c_delta": self.delta,
            "accountant_type": "opacus_rdp" if self._accountant is not None else "N/A",
            "noise_mechanism": self.noise_mechanism,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "steps": self.steps,
            "limitation": PRIVACY_LIMITATION,
        }

#!/usr/bin/env python3
# 用途：使用 Opacus RDP accountant 计算新 Stage C 的隐私预算。

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from opacus.accountants import RDPAccountant


def compute_epsilon(
    steps,
    sample_rate,
    noise_multiplier,
    delta,
    count_noise=None,
    count_steps=None,
):
    accountant = RDPAccountant()
    for _ in range(int(steps)):
        accountant.step(
            noise_multiplier=float(noise_multiplier),
            sample_rate=float(sample_rate),
        )
    if count_noise is not None:
        if count_steps is None:
            count_steps = steps
        for _ in range(int(count_steps)):
            accountant.step(
                noise_multiplier=float(count_noise),
                sample_rate=float(sample_rate),
            )
    return float(accountant.get_epsilon(delta=float(delta)))


def main():
    parser = argparse.ArgumentParser(description="Opacus RDP 隐私会计")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--noise_multiplier", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--count_noise", type=float, default=None)
    parser.add_argument("--count_steps", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "epsilon": compute_epsilon(
                    args.steps,
                    args.sample_rate,
                    args.noise_multiplier,
                    args.delta,
                    args.count_noise,
                    args.count_steps,
                )
            }
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Calibrate Stage C gradient noise against the actual composed accountant."""

import argparse
import json

from new_privacy_accountant import compute_epsilon


def adaptive_count_steps(steps, warmup_steps, update_interval):
    return sum(
        1
        for step in range(int(steps))
        if step >= int(warmup_steps)
        and (step + 1) % int(update_interval) == 0
    )


def calibrate_noise_multiplier(
    target_epsilon,
    steps,
    sample_rate,
    delta,
    count_noise=None,
    count_steps=0,
    tolerance=1e-4,
):
    target_epsilon = float(target_epsilon)
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")

    def epsilon_at(noise):
        return compute_epsilon(
            steps=steps,
            sample_rate=sample_rate,
            noise_multiplier=noise,
            delta=delta,
            count_noise=count_noise if count_steps else None,
            count_steps=count_steps if count_steps else None,
        )

    lower = 0.05
    upper = 1.0
    while epsilon_at(upper) > target_epsilon and upper < 1024.0:
        upper *= 2.0
    epsilon_upper = epsilon_at(upper)
    if epsilon_upper > target_epsilon:
        raise ValueError(
            "target epsilon is infeasible with the configured count mechanism"
        )
    for _ in range(70):
        midpoint = (lower + upper) / 2.0
        if epsilon_at(midpoint) > target_epsilon:
            lower = midpoint
        else:
            upper = midpoint
    achieved = epsilon_at(upper)
    if abs(achieved - target_epsilon) > tolerance:
        raise ValueError(
            "calibration error {} exceeds tolerance {}".format(
                abs(achieved - target_epsilon), tolerance
            )
        )
    return float(upper), float(achieved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_epsilon", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--count_noise", type=float, default=None)
    parser.add_argument("--count_steps", type=int, default=0)
    args = parser.parse_args()
    noise, achieved = calibrate_noise_multiplier(
        target_epsilon=args.target_epsilon,
        steps=args.steps,
        sample_rate=args.sample_rate,
        delta=args.delta,
        count_noise=args.count_noise,
        count_steps=args.count_steps,
    )
    print(
        json.dumps(
            {
                "target_epsilon": args.target_epsilon,
                "achieved_epsilon": achieved,
                "gradient_noise_multiplier": noise,
                "steps": args.steps,
                "sample_rate": args.sample_rate,
                "delta": args.delta,
                "count_noise_multiplier": args.count_noise,
                "count_steps": args.count_steps,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

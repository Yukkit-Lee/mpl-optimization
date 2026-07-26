#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate a Stage C validation round without reading any test split."""

import argparse
import json
import re
from pathlib import Path

from new_public_mtt_mainline_common import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-tag", required=True)
    parser.add_argument("--max-reference-drop", type=float, default=0.02)
    args = parser.parse_args()
    if not re.match(r"^[a-z0-9_]+$", args.round_tag):
        raise ValueError("invalid round-tag")
    source_root = (
        ROOT / "new_output/new_public_mtt_mainline" / args.round_tag
    )
    rows = []
    for dataset in ("cifar10", "cifar100", "tiny"):
        for mode in ("fixed_dp", "adaptive_dp"):
            matches = list(
                (source_root / dataset / mode / "seed42").glob(
                    "new_secure_stage_c_report_*.json"
                )
            )
            if len(matches) != 1:
                raise FileNotFoundError(
                    "expected one report for {}/{}; got {}".format(
                        dataset, mode, len(matches)
                    )
                )
            report = json.loads(matches[0].read_text(encoding="utf-8"))
            if report["status"] != "COMPLETE":
                raise ValueError("{} is incomplete".format(matches[0]))
            history = [
                item for item in report["history"] if "reference_accuracy" in item
            ]
            if not history:
                raise ValueError("{} has no reference diagnostics".format(matches[0]))
            first_accuracy = float(history[0]["reference_accuracy"])
            final_accuracy = float(history[-1]["reference_accuracy"])
            drop = first_accuracy - final_accuracy
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "seed": int(report["seed"]),
                    "steps": int(report["privacy"]["steps"]),
                    "learning_rate": float(report["configuration"]["learning_rate"]),
                    "epsilon_c": float(
                        report["privacy"][
                            "stage_c_distilled_feature_record_epsilon"
                        ]
                    ),
                    "count_mechanism_steps": int(
                        report["privacy"]["clipping_count_mechanism_steps"]
                    ),
                    "first_logged_reference_accuracy": first_accuracy,
                    "final_reference_accuracy": final_accuracy,
                    "reference_drop": drop,
                    "stability_pass": drop <= args.max_reference_drop,
                    "final_c": float(report["final_c"]),
                    "runtime_seconds": float(report["runtime_seconds"]),
                    "backend": report["security"]["backend"],
                    "protocol": report["security"]["protocol"],
                    "silent_plaintext_fallback": report["security"][
                        "silent_plaintext_fallback"
                    ],
                    "test_loaded": False,
                    "source_report": str(matches[0]),
                }
            )
    passed = all(
        row["stability_pass"]
        and row["backend"] == "tfe_mpc"
        and row["protocol"] == "ABY3"
        and not row["silent_plaintext_fallback"]
        for row in rows
    )
    payload = {
        "schema": "new_public_mtt_stage_c_validation_summary_v1",
        "round_tag": args.round_tag,
        "status": "PASS" if passed else "NEEDS_TUNING",
        "formal_result": False,
        "test_loaded": False,
        "dp_mtt_status": "EXCLUDED_BY_USER",
        "gate": {
            "rule": "every Fixed/Adaptive run keeps reference drop <= threshold",
            "max_reference_drop": args.max_reference_drop,
            "passed": passed,
        },
        "rows": rows,
    }
    output = (
        ROOT
        / "new_log/new_public_mtt_mainline/json"
        / "new_{}_summary.json".format(args.round_tag)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("round_tag={} status={} test_loaded=False".format(
        args.round_tag, payload["status"]
    ))
    for row in rows:
        print(
            "{dataset} {mode} steps={steps} lr={learning_rate} "
            "first={first_logged_reference_accuracy:.4f} "
            "final={final_reference_accuracy:.4f} "
            "drop={reference_drop:.4f} epsilon_C={epsilon_c:.6f}".format(**row)
        )
    print("output={}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

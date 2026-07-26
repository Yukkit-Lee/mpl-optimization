#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate measured TFE system experiments."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from new_public_mtt_mainline_common import ROOT, load_config


SOURCE = ROOT / "new_output/new_public_mtt_mainline/tfe_system_formal"
JSON_OUTPUT = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_tfe_system_formal_report.json"
)
TXT_OUTPUT = (
    ROOT
    / "new_log/new_public_mtt_mainline/txt/new_tfe_system_formal_report.txt"
)
CSV_OUTPUT = (
    ROOT / "new_expTab/new_public_mtt_mainline/new_tfe_system_formal.csv"
)
FIGURE_ROOT = ROOT / "new_expFig/new_public_mtt_mainline"


def main():
    config = load_config()
    formal = config["tfe_system_formal"]
    rows = []
    scale_order = list(formal["scales"])
    for dataset in config["datasets"]:
        for scale in scale_order:
            path = (
                SOURCE
                / dataset
                / scale
                / "new_tfe_system_report_{}_{}.json".format(dataset, scale)
            )
            if not path.is_file():
                raise FileNotFoundError("missing {}".format(path))
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["status"] != "COMPLETE":
                raise ValueError("{} is incomplete".format(path))
            if int(report["iterations"]) <= 10:
                raise ValueError("{} is only a sanity run".format(path))
            if not report.get("revealed_values"):
                raise ValueError("{} omits reveal boundary".format(path))
            if report["communication_bytes"] != "N/A":
                raise ValueError("unexpected fabricated communication bytes")
            rows.append(
                {
                    "dataset": dataset,
                    "scale": scale,
                    "num_samples": int(report["num_samples"]),
                    "feature_dim": int(report["feature_dim"]),
                    "num_classes": int(report["num_classes"]),
                    "batch_size": int(report["batch_size"]),
                    "iterations": int(report["iterations"]),
                    "model_parameters": int(report["model_parameters"]),
                    "fixed_point_precision_fractional": int(
                        report["fixed_point_precision_fractional"]
                    ),
                    "secure_runtime_seconds": float(
                        report["secure_runtime_seconds"]
                    ),
                    "wall_seconds": float(report["wall_seconds"]),
                    "peak_rss_bytes": int(report["peak_rss_bytes"]),
                    "communication_bytes": "N/A",
                    "communication_rounds": "N/A",
                    "revealed_values": report["revealed_values"],
                    "released_model_record_dp_valid": report[
                        "released_model_record_dp_valid"
                    ],
                    "test_loaded": False,
                    "source_report": str(path),
                }
            )
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            key for key in rows[0] if key not in ("revealed_values", "source_report")
        ]
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            [{key: row[key] for key in fieldnames} for row in rows]
        )
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for dataset in config["datasets"]:
        selected = [row for row in rows if row["dataset"] == dataset]
        axes[0].plot(
            [row["batch_size"] for row in selected],
            [row["wall_seconds"] for row in selected],
            marker="o",
            label=dataset,
        )
        axes[1].plot(
            [row["batch_size"] for row in selected],
            [row["peak_rss_bytes"] / (1024.0 ** 3) for row in selected],
            marker="o",
            label=dataset,
        )
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Wall time (s)")
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Peak RSS (GiB)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    png = FIGURE_ROOT / "new_figureS_tfe_system_scaling.png"
    pdf = FIGURE_ROOT / "new_figureS_tfe_system_scaling.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    payload = {
        "schema": "new_tfe_system_formal_report_v1",
        "status": "COMPLETE",
        "formal_result": True,
        "runs_expected": 9,
        "runs_completed": len(rows),
        "test_loaded": False,
        "dp_mtt_status": "EXCLUDED_BY_USER",
        "backend": "tfe_mpc",
        "protocol": "ABY3",
        "rows": rows,
        "communication_limitation": (
            "single-machine TF-Encrypted 0.8.0 does not expose bytes or rounds"
        ),
        "artifacts": {
            "csv": str(CSV_OUTPUT),
            "figure_png": str(png),
            "figure_pdf": str(pdf),
        },
    }
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["TFE system formal report", "runs=9/9 test_loaded=False"]
    for row in rows:
        lines.append(
            "{dataset} {scale} batch={batch_size} iterations={iterations} "
            "wall={wall_seconds:.3f}s peak_rss={peak_rss_bytes}".format(**row)
        )
    TXT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TXT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("tfe_system status=COMPLETE runs=9")
    for row in rows:
        print(
            "{dataset} {scale} wall={wall_seconds:.3f}s peak_rss={peak_rss_bytes}".format(
                **row
            )
        )
    print("json={}".format(JSON_OUTPUT))
    print("figure={}".format(png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

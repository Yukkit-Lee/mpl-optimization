#!/usr/bin/env python3
"""Repair TFE system-table field alignment without rerunning 9 formal jobs."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "new_log/new_public_mtt_mainline/json/new_tfe_system_formal_report.json"
)
JSON_OUTPUT = (
    ROOT / "new_log/new_true_ours/new_tfe_system_formal_repaired.json"
)
CSV_OUTPUT = (
    ROOT / "new_expTab/new_true_ours/new_tfe_system_formal_repaired.csv"
)


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("status") != "COMPLETE":
        raise ValueError("authoritative TFE system report is incomplete")
    rows = []
    for row in source["rows"]:
        source_report = Path(row["source_report"])
        if not source_report.is_file():
            raise FileNotFoundError(source_report)
        raw = json.loads(source_report.read_text(encoding="utf-8"))
        if raw.get("status") != "COMPLETE":
            raise ValueError("{} is incomplete".format(source_report))
        runtime = float(row["secure_runtime_seconds"])
        peak_rss = int(row["peak_rss_bytes"])
        if runtime <= 0:
            raise AssertionError("runtime must be positive")
        if peak_rss <= 0:
            raise AssertionError("peak RSS must be positive")
        rows.append(
            {
                "dataset": row["dataset"],
                "scale": row["scale"],
                "batch_size": int(row["batch_size"]),
                "iterations": int(row["iterations"]),
                "feature_dim": int(row["feature_dim"]),
                "num_classes": int(row["num_classes"]),
                "fixed_point_precision_fractional": int(
                    row["fixed_point_precision_fractional"]
                ),
                "secure_runtime_seconds": runtime,
                "wall_seconds": float(row["wall_seconds"]),
                "peak_rss_bytes": peak_rss,
                "peak_rss_gib": peak_rss / float(1024**3),
                "communication_bytes": "N/A",
                "communication_rounds": "N/A",
                "source_report": str(source_report),
            }
        )
    if len(rows) != 9:
        raise AssertionError("source_rows must equal 9, got {}".format(len(rows)))
    payload = {
        "schema": "new_tfe_system_formal_repaired_v1",
        "status": "COMPLETE",
        "source_rows": len(rows),
        "backend": "tfe_mpc",
        "protocol": "ABY3",
        "test_loaded": False,
        "usage_scope": (
            "historical fixed-kernel system scaling supplement only; "
            "not the Adaptive main-figure timing source"
        ),
        "communication_limitation": source["communication_limitation"],
        "rows": rows,
    }
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key in rows[0] if key != "source_report"]
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            [{key: row[key] for key in fieldnames} for row in rows]
        )
    print(
        "status=COMPLETE source_rows={} json={} csv={}".format(
            len(rows), JSON_OUTPUT, CSV_OUTPUT
        )
    )


if __name__ == "__main__":
    main()

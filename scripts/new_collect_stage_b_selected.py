#!/usr/bin/env python3
"""Collect only final Stage-B selected initializers into a portable Git asset."""

import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
SOURCE = ROOT / "new_init_model/new_public_mtt_mainline/stage_b"
DESTINATION = ROOT / "portable_artifacts/stage_b_selected"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    records = []
    for report_path in sorted(SOURCE.glob("*/seed*/*/new_secure_stage_b_report_*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE":
            raise RuntimeError("Incomplete Stage-B report: {}".format(report_path))
        dataset = report["dataset"]
        seed = int(report["seed"])
        split = report["split"]
        source_path = report_path.parent / Path(report["selected_model_path"]).name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        actual_sha = sha256_file(source_path)
        if actual_sha != report["selected_model_sha256"]:
            raise RuntimeError("Selected initializer SHA mismatch: {}".format(source_path))
        selected_strategy = "unknown"
        for strategy, candidate in report.get("formal_candidate_models", {}).items():
            if candidate.get("sha256") == actual_sha:
                selected_strategy = strategy
                break
        relative = Path(dataset) / "seed{}".format(seed) / split / source_path.name
        destination = DESTINATION / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # The project mount may disallow chmod/utime, so preserve bytes rather
        # than host metadata. SHA256 below is the portability contract.
        shutil.copyfile(str(source_path), str(destination))
        records.append(
            {
                "dataset": dataset,
                "seed": seed,
                "split": split,
                "selected_strategy": selected_strategy,
                "relative_path": str(relative),
                "sha256": actual_sha,
                "size_bytes": destination.stat().st_size,
                "feature_dim": int(report["feature_dim"]),
                "num_classes": int(report["num_classes"]),
                "feature_cache_sha256": report["feature_cache_sha256"],
                "backend": report["security"]["backend"],
                "protocol": report["security"]["protocol"],
            }
        )
    if len(records) != 27:
        raise RuntimeError("Expected 27 selected initializers, found {}".format(len(records)))
    payload = {
        "schema": "new_portable_stage_b_selected_v1",
        "source_branch": "new-secure-feature-pipeline",
        "selected_initializer_count": len(records),
        "total_size_bytes": sum(item["size_bytes"] for item in records),
        "records": records,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Collected {} selected initializers ({} bytes)".format(
        len(records), payload["total_size_bytes"]
    ))


if __name__ == "__main__":
    main()

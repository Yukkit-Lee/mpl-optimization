#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用途：汇总 Stage A、full-ConvNet A+B+C 与 TFE 系统验证结果。"""

import csv
import json
import time
from pathlib import Path

from project_paths import PATHS


JSON_DIR = PATHS.output_root / "log" / "json"
TABLE_DIR = PATHS.output_root / "expTab"
DATASETS = ("cifar10", "cifar100", "tiny")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _formal_stage_a(dataset):
    candidates = []
    for path in JSON_DIR.glob("stageA_directImage_{}_*.json".format(dataset)):
        payload = _load(path)
        candidates.append((int(payload.get("train_steps_effective", 0)), path, payload))
    if not candidates:
        raise FileNotFoundError("missing Stage A result for {}".format(dataset))
    return max(candidates, key=lambda item: item[0])


def _latest(pattern):
    paths = sorted(JSON_DIR.glob(pattern))
    if not paths:
        return None, None
    return paths[-1], _load(paths[-1])


def main():
    run_id = time.strftime("%Y%m%d_%H%M%S")
    tfe_path, tfe_payload = _latest("stageC_tfe_secure_system_cifar10_*.json")
    datasets = {}
    rows = []
    for dataset in DATASETS:
        stage_a_steps, stage_a_path, stage_a = _formal_stage_a(dataset)
        end_path, end = _latest("crossStageConvNet_end2end_{}_*.json".format(dataset))
        if end is None:
            raise FileNotFoundError("missing cross-stage result for {}".format(dataset))
        datasets[dataset] = {
            "stage_a_source": str(stage_a_path),
            "stage_a": {
                "pipeline": stage_a["stage_a_pipeline"],
                "train_steps": stage_a_steps,
                "num_full_samples": stage_a["num_full_samples"],
                "num_distilled_samples": stage_a["num_distilled_samples"],
                "compression_ratio": stage_a["compression_ratio"],
                "results": stage_a["results"],
            },
            "algorithmic_main_source": str(end_path),
            "algorithmic_main": end,
            "secure_system_validation": (
                {"source": str(tfe_path), "result": tfe_payload}
                if dataset == "cifar10" else {
                    "status": "NOT_RUN_PER_DATASET",
                    "reason": "CIFAR-10 representative lightweight TFE validation is used.",
                }
            ),
        }
        for mode, result in end["stage_c"].items():
            privacy = result["privacy"]
            rows.append({
                "Dataset": dataset,
                "StageAInput": "official_mtt_distilled_images",
                "StageBChosen": end["stage_b"]["chosen_name"],
                "StageCMode": mode,
                "PrivacyScope": privacy["privacy_scope"],
                "StageCEpsilon": privacy["stage_c_epsilon"],
                "StageCDelta": privacy["stage_c_delta"],
                "Accountant": privacy["accountant_type"],
                "NoiseMechanism": privacy["noise_mechanism"],
                "FinalTestAcc": result["final_test_accuracy"],
                "FinalReferenceAcc": result["final_reference_accuracy"],
                "RuntimeS": result["time_s"],
                "ResultSource": str(end_path),
            })

    payload = {
        "run_id": run_id,
        "algorithmic_main_pipeline": (
            "Distilled Images -> PyTorch full ConvNet Stage B -> Chosen full "
            "ConvNet -> PyTorch full ConvNet Stage C"
        ),
        "secure_system_validation": (
            "Distilled Images -> Flatten -> TFE lightweight secret-shared training"
        ),
        "privacy_scope": "stage_c_only",
        "datasets": datasets,
    }
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    json_path = JSON_DIR / "crossStageConvNet_unified_{}.json".format(run_id)
    csv_path = TABLE_DIR / "crossStageConvNet_unified_{}.csv".format(run_id)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("json:", json_path)
    print("csv :", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

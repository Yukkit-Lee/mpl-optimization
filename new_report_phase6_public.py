#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：汇总 Phase 6 public MTT 三数据集正式结果并执行质量门检查。

表格中的 test accuracy 只来自冻结配置完成后的单次只读评估；曲线展示
validation/reference 训练诊断，不把 test 用于 checkpoint 或超参数选择。
"""

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parent))
TABLE_DIR = ROOT / "new_expTab/new_secure_feature"
FIG_DIR = ROOT / "new_expFig/new_secure_feature"
JSON_DIR = ROOT / "new_log/new_secure_feature/json"
STATUS_DIR = ROOT / "new_log/new_secure_feature/status"
OFFICIAL = {"cifar10": 0.6884, "cifar100": 0.4587, "tiny": 0.2558}
DISPLAY = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tiny": "Tiny ImageNet"}


def paths(dataset: str, seed: int):
    if dataset == "cifar10":
        base = (
            ROOT
            / "new_cache/new_secure_feature/new_formal_phase6/new_public_mtt_round3"
            / "new_seed{}".format(seed)
            / "public_mtt_benchmark"
            / dataset
        )
        status = STATUS_DIR / "new_formal_stageA_public_cifar10_round3_seed{}.json".format(seed)
    else:
        base = (
            ROOT
            / "new_cache/new_secure_feature/new_formal_phase6/new_public_mtt_final"
            / "new_{}_seed{}".format(dataset, seed)
            / "public_mtt_benchmark"
            / dataset
        )
        status = STATUS_DIR / "new_formal_phase6_public_{}_seed{}.json".format(dataset, seed)
    return {
        "base": base,
        "status": status,
        "metadata": base
        / "new_shared_feature_metadata_{}_public_mtt_benchmark_seed{}.json".format(
            dataset, seed
        ),
        "checkpoint": base
        / "new_shared_encoder_checkpoint_{}_public_mtt_benchmark_seed{}.pt".format(
            dataset, seed
        ),
    }


def require_complete(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise RuntimeError("正式任务未完成: {} status={}".format(path, payload.get("status")))


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    for directory in (TABLE_DIR, FIG_DIR, JSON_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = defaultdict(list)
    for dataset in ("cifar10", "cifar100", "tiny"):
        gate = max(0.95 * OFFICIAL[dataset], OFFICIAL[dataset] - 0.02)
        for seed in (42, 43, 44):
            item = paths(dataset, seed)
            require_complete(item["status"])
            metadata = json.loads(item["metadata"].read_text(encoding="utf-8"))
            checkpoint = torch.load(
                str(item["checkpoint"]), map_location="cpu", weights_only=False
            )
            accuracy = float(metadata["final_test_accuracy"])
            history = checkpoint["history"]
            curves[dataset].append(history)
            rows.append(
                {
                    "Dataset": DISPLAY[dataset],
                    "Seed": seed,
                    "Source": "Official MTT ipc=50",
                    "OfficialReferenceAccuracy": OFFICIAL[dataset],
                    "StrictQualityGate": gate,
                    "FinalTestAccuracy": accuracy,
                    "Passed": accuracy >= gate,
                    "Epochs": int(history[-1]["epoch"]),
                    "TestUsage": metadata["test_usage"],
                }
            )
    summary = []
    for dataset in ("cifar10", "cifar100", "tiny"):
        selected = [row for row in rows if row["Dataset"] == DISPLAY[dataset]]
        mean, std = mean_std([row["FinalTestAccuracy"] for row in selected])
        summary.append(
            {
                "Dataset": DISPLAY[dataset],
                "Seeds": 3,
                "MeanTestAccuracy": mean,
                "StdTestAccuracy": std,
                "StrictQualityGate": selected[0]["StrictQualityGate"],
                "AllSeedsPassed": all(row["Passed"] for row in selected),
            }
        )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=180, sharey=False)
    for axis, dataset in zip(axes, ("cifar10", "cifar100", "tiny")):
        for seed, history in zip((42, 43, 44), curves[dataset]):
            x = np.asarray([int(row["epoch"]) for row in history])
            y = np.asarray([float(row["reference_accuracy"]) for row in history])
            axis.plot(x, y, linewidth=1.6, label="seed {}".format(seed))
        axis.set_title(DISPLAY[dataset])
        axis.set_xlabel("Training Epoch")
        axis.set_ylabel("Reference Accuracy")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    png_path = FIG_DIR / "new_phase6_public_mtt_reference_curves.png"
    pdf_path = FIG_DIR / "new_phase6_public_mtt_reference_curves.pdf"
    figure.savefig(str(png_path), bbox_inches="tight")
    figure.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(figure)

    csv_path = TABLE_DIR / "new_phase6_public_mtt_results.csv"
    summary_path = TABLE_DIR / "new_phase6_public_mtt_summary.csv"
    json_path = JSON_DIR / "new_phase6_public_mtt_results.json"
    write_csv(csv_path, rows)
    write_csv(summary_path, summary)
    payload = {
        "schema": "new_phase6_public_mtt_results_v1",
        "status": (
            "QUALITY_GATE_PASS"
            if all(item["AllSeedsPassed"] for item in summary)
            else "NOT_READY_FOR_PAPER"
        ),
        "test_tuning": False,
        "rows": rows,
        "summary": summary,
        "outputs": {
            "csv": str(csv_path),
            "summary_csv": str(summary_path),
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Phase 6 public MTT formal summary")
    print("| Dataset | Mean ± Std | Gate | All seeds pass |")
    print("|---|---:|---:|:---:|")
    for item in summary:
        print(
            "| {Dataset} | {MeanTestAccuracy:.4f} ± {StdTestAccuracy:.4f} | "
            "{StrictQualityGate:.4f} | {AllSeedsPassed} |".format(**item)
        )
    print("status={}".format(payload["status"]))
    print("table={}".format(csv_path))
    print("figure={}".format(png_path))
    return 0 if payload["status"] == "QUALITY_GATE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

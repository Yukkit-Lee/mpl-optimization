#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：完整 ConvNet Stage B 初始化的 Figure 5 收敛实验。

候选池固定为 KD-Avg、KD-ReferenceWeighted、KD-Best；Random 仅作为收敛
对照，不参与 chosen selection。Test set 只按预设间隔只读评估。
"""

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

from project_paths import PATHS
import official_distilled_convnet_pipeline as pipe
import exp_module_b_kd_noniid_official as stage_b_entry

FIG_DIR = PATHS.output_root / "expFig" / "figure5_crossStageConvNet"
JSON_DIR = PATHS.output_root / "log" / "json"


def _plot(curves, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "Random Init": "#8a8a8a",
        "KD-Avg": "#4c78a8",
        "KD-ReferenceWeighted": "#f58518",
        "KD-Best": "#54a24b",
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=190)
    for name, result in curves.items():
        rows = result["history"]
        ax.plot(
            [row["step"] for row in rows],
            [row["test_accuracy_readonly"] for row in rows],
            label=name,
            color=colors[name],
            linewidth=2.0,
            marker="o",
        )
    ax.set_xlabel("Optimizer Steps")
    ax.set_ylabel("Test Accuracy")
    ax.grid(alpha=0.28)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Figure 5 full-ConvNet initialization convergence")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tiny"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    pipe.set_safe_runtime(args.seed, 1)
    device = pipe.select_device("auto")
    stage_b_out = stage_b_entry.run_stage_b_experiment(args.dataset, args.seed, args.quick)
    bundle = stage_b_out["bundle"]
    stage_b = stage_b_out["stage_b"]
    steps = 2 if args.quick else 200
    eval_interval = 1 if args.quick else 20
    _, chosen_ref_acc = pipe.evaluate_model(stage_b["chosen_model"], bundle["x_ref"], bundle["y_ref"], device)
    target_ref = min(0.999, chosen_ref_acc + 0.02)
    models = {"Random Init": pipe.build_convnet(args.dataset, args.seed + 900, device).cpu()}
    models.update({name: deepcopy(model) for name, model in stage_b["candidates"].items()})
    curves = {}
    for name, model in models.items():
        curves[name] = pipe.train_stage_c_convnet(
            model=model,
            bundle=bundle,
            mode="plain",
            steps=steps,
            batch_size=32,
            lr=0.002,
            clip_c=1.0,
            noise_multiplier=0.0,
            gamma=0.5,
            eta_c=0.1,
            seed=args.seed + 1000,
            device=device,
            eval_interval=eval_interval,
            target_reference_accuracy=target_ref,
        )
    run_id = time.strftime("%Y%m%d_%H%M%S")
    figure = FIG_DIR / "crossStageConvNet_figure5_{}_{}.png".format(args.dataset, run_id)
    _plot(curves, figure)
    payload = {
        "run_id": run_id,
        "dataset": args.dataset,
        "pipeline": "algorithmic_pytorch_convnet",
        "backend": "local_plaintext_pytorch",
        "input_type": "nchw_images",
        "feature_extraction_used": False,
        "candidate_pool": ["KD-Avg", "KD-ReferenceWeighted", "KD-Best"],
        "chosen_name": stage_b["chosen_name"],
        "heldout_reference_target": target_ref,
        "test_usage": "fixed_interval_readonly_reporting",
        "curves": {
            name: {key: value for key, value in result.items() if key != "model"}
            for name, result in curves.items()
        },
        "outputs": {"figure": str(figure), "stage_b_checkpoint": str(stage_b_out["checkpoint"])},
    }
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    json_path = JSON_DIR / "crossStageConvNet_figure5_{}_{}.json".format(args.dataset, run_id)
    pipe.write_json(json_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("figure:", figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

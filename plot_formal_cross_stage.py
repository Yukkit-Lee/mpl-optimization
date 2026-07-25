#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：仅依赖 JSON/NumPy/Matplotlib 绘制正式跨阶段实验曲线。

本脚本不导入 PyTorch、CUDA 或训练模块，避免长训练后的原生运行库冲突。
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLIT_NAMES = ("IID", "Mild Non-IID", "Severe Non-IID")


def aggregate_curves(seed_curves):
    grouped = defaultdict(list)
    for curves in seed_curves.values():
        for method, history in curves.items():
            grouped[method].append(history)
    output = {}
    for method, histories in grouped.items():
        steps = sorted(set.intersection(*[
            {int(row["step"]) for row in history} for history in histories
        ]))
        points = []
        for step in steps:
            values = [
                next(float(row["test_accuracy_readonly"]) for row in history
                     if int(row["step"]) == step)
                for history in histories
            ]
            points.append((step, float(np.mean(values)), float(np.std(values, ddof=1))))
        output[method] = points
    return output


def plot_module_b(details, path):
    colors = {"NoKD": "#8a8a8a", "KD-Avg": "#4c78a8",
              "KD-ReferenceWeighted": "#f58518", "KD-Best": "#54a24b"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180, sharey=True)
    for axis, split in zip(axes, SPLIT_NAMES):
        seed_curves = {int(seed): item[split]["curves"] for seed, item in details.items()}
        for method, points in aggregate_curves(seed_curves).items():
            x = np.asarray([p[0] for p in points]); mean = np.asarray([p[1] for p in points]); std = np.asarray([p[2] for p in points])
            axis.plot(x, mean, label=method, color=colors[method], linewidth=2)
            axis.fill_between(x, mean-std, mean+std, color=colors[method], alpha=0.14)
        axis.set_title(split); axis.set_xlabel("Optimizer Steps"); axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test Accuracy"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(str(path)); plt.close(fig)


def plot_module_c(details, path):
    colors = {"plain": "#777777", "fixed_dp": "#4c78a8", "adaptive_dp": "#e45756"}
    seed_results = {int(seed): item["results"] for seed, item in details.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180)
    for mode in ("plain", "fixed_dp", "adaptive_dp"):
        histories = [item[mode]["history"] for item in seed_results.values()]
        steps = sorted(set.intersection(*[{int(r["step"]) for r in h} for h in histories]))
        means=[]; stds=[]
        for step in steps:
            vals=[next(float(r["test_accuracy_readonly"]) for r in h if int(r["step"])==step) for h in histories]
            means.append(float(np.mean(vals))); stds.append(float(np.std(vals,ddof=1)))
        axes[0].plot(steps,means,label=mode,color=colors[mode],linewidth=2)
        axes[0].fill_between(steps,np.asarray(means)-np.asarray(stds),np.asarray(means)+np.asarray(stds),color=colors[mode],alpha=.14)
    adaptive=[item["adaptive_dp"]["history"] for item in seed_results.values()]
    steps=sorted(set.intersection(*[{int(r["step"]) for r in h} for h in adaptive]))
    cmean=[]; cstd=[]
    for step in steps:
        vals=[next(float(r["clipping_bound_t"]) for r in h if int(r["step"])==step) for h in adaptive]
        cmean.append(float(np.mean(vals))); cstd.append(float(np.std(vals,ddof=1)))
    axes[1].plot(steps,cmean,color=colors["adaptive_dp"],linewidth=2)
    axes[1].fill_between(steps,np.asarray(cmean)-np.asarray(cstd),np.asarray(cmean)+np.asarray(cstd),color=colors["adaptive_dp"],alpha=.14)
    for mode in ("fixed_dp","adaptive_dp"):
        histories=[item[mode]["history"] for item in seed_results.values()]; means=[]
        for step in steps:
            vals=[next(float(r["clipped_fraction"]) for r in h if int(r["step"])==step) for h in histories]
            means.append(float(np.mean(vals)))
        axes[2].plot(steps,means,label=mode,color=colors[mode],linewidth=2)
    axes[0].set_ylabel("Test Accuracy"); axes[0].set_xlabel("Optimizer Steps")
    axes[1].set_ylabel("Clipping Bound $C_t$"); axes[1].set_xlabel("Optimizer Steps")
    axes[2].set_ylabel("Clipped Fraction"); axes[2].set_xlabel("Optimizer Steps")
    axes[0].legend(); axes[2].legend()
    for axis in axes: axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(str(path)); plt.close(fig)


def plot_end2end(details, path):
    seed_curves={int(seed):{method:r["history"] for method,r in item.items()} for seed,item in details.items()}
    colors=["#888888","#72b7b2","#4c78a8","#f58518","#e45756"]
    fig,ax=plt.subplots(figsize=(8.8,5.2),dpi=180)
    for color,(method,points) in zip(colors,aggregate_curves(seed_curves).items()):
        x=np.asarray([p[0] for p in points]); mean=np.asarray([p[1] for p in points]); std=np.asarray([p[2] for p in points])
        ax.plot(x,mean,label=method,color=color,linewidth=2); ax.fill_between(x,mean-std,mean+std,color=color,alpha=.14)
    ax.set_xlabel("Optimizer Steps"); ax.set_ylabel("Test Accuracy"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(str(path)); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--json",required=True); args=parser.parse_args()
    source=Path(args.json).resolve(); payload=json.loads(source.read_text(encoding="utf-8"))
    out_dir=source.parents[2]/"expFig"/"formal_cross_stage"; out_dir.mkdir(parents=True,exist_ok=True)
    run_id=str(payload["run_id"]); outputs=[]
    for dataset in payload["resolved_config"]["datasets"]:
        b=out_dir/f"moduleB_figure5_{dataset}_{run_id}.png"; c=out_dir/f"moduleC_{dataset}_{run_id}.png"; e=out_dir/f"end2end_{dataset}_{run_id}.png"
        plot_module_b(payload["module_b"]["details"][dataset],b)
        plot_module_c(payload["module_c"]["details"][dataset],c)
        plot_end2end(payload["end_to_end"]["details"][dataset],e)
        outputs += [str(b),str(c),str(e)]
    print(json.dumps({"status":"PLOT_COMPLETE","outputs":outputs},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())

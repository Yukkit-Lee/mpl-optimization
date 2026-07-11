#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：模块 C 的 Clipping Bound Trajectory 多随机种子实验。

图名建议：Figure X. Clipping bound trajectory

实验目标：
展示 Adaptive-DP 中裁剪阈值 C_t 会随训练过程动态变化，
并通过多个 seed 报告 mean ± std，避免单次曲线被质疑偶然性。

曲线：
- MNIST Adaptive-DP
- CIFAR-10 Adaptive-DP
- IMDb Adaptive-DP

输出：
1) 图像：expFig/moduleC_adaptive_clipping/clipping_bound_trajectory_multiseed_{run_id}.png
2) CSV：expTab/module_c_clipping_trajectory_multiseed_{run_id}.csv
3) JSON：log/json/moduleC_clipping_trajectory_multiseed_{run_id}.json
4) TXT：log/txt/moduleC_clipping_trajectory_multiseed_{run_id}.log

注意：
- 只读取模块 B latest initialization，不重新运行模块 B。
- 只跑 Adaptive-DP，不跑 Plain/Fixed-DP。
- 线程限制写在文件开头，保护机器。
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_TXT_DIR = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"
EXPFIG_DIR = BASE_DIR / "expFig"
FIG_DIR = EXPFIG_DIR / "moduleC_adaptive_clipping"
EXPTAB_DIR = BASE_DIR / "expTab"

CFG = {
    "safe_mode": True,
    "cpu_threads": 1,
    "device": "cpu",
    "max_test_cap": 1000,
    "global_epochs": 300,
    "global_lr": 0.01,
    "batch_size": 64,
    "clipping_norm": 0.7,
    "noise_multiplier": 1.0,
    "delta": 1e-5,
    "target_epsilon": None,
    "gamma": 0.85,
    "eta_c": 0.08,
    "C_init": 0.7,
    "sigma_b_scale": 20.0,
    "quantile_beta": 0.35,
    "warmup_steps": 40,
    "clip_min": 0.05,
    "clip_max": 3.0,
    "noise_backend": "gaussian",
    "max_epoch_log_lines": 10,
    "seed_base": 42,
}

DATASET_NAME = {
    "mnist": "MNIST",
    "cifar": "CIFAR-10",
    "imdb": "IMDb",
}

DATASET_OVERRIDES = {
    "mnist": {"clipping_norm": 0.7, "C_init": 0.7, "clip_max": 3.0, "global_lr": 0.01},
    "cifar": {"clipping_norm": 0.35, "C_init": 0.7, "clip_max": 3.0, "global_lr": 0.01},
    "imdb": {"clipping_norm": 1.2, "C_init": 0.7, "clip_max": 3.0, "global_lr": 0.01},
}


def ensure_dirs() -> None:
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EXPTAB_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def tee_console_to_file(log_path: Path):
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    old_out, old_err = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as f:
        tee = _Tee(old_out, f)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


def set_safe_runtime(max_threads: int = 1) -> None:
    t = max(1, int(max_threads))
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["MKL_NUM_THREADS"] = str(t)
    os.environ["OPENBLAS_NUM_THREADS"] = str(t)
    os.environ["NUMEXPR_NUM_THREADS"] = str(t)
    os.environ["TFE_SAFE_RUNTIME_APPLIED"] = "1"
    try:
        torch.set_num_threads(t)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(t)
    except Exception:
        pass


def save_csv(summary_rows: List[Dict], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Epoch", "C_t_mean", "C_t_std", "TestAcc_mean", "TestAcc_std", "Seeds"])
        for r in summary_rows:
            w.writerow([
                r["dataset_name"],
                r["epoch"],
                "{:.6f}".format(r["c_mean"]),
                "{:.6f}".format(r["c_std"]),
                "{:.6f}".format(r["acc_mean"]),
                "{:.6f}".format(r["acc_std"]),
                r["num_seeds"],
            ])


def plot_trajectory(summary_by_dataset: Dict[str, List[Dict]], run_id: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "mnist": "#1f77b4",
        "cifar": "#ff7f0e",
        "imdb": "#2ca02c",
    }

    plt.figure(figsize=(9, 5.2))
    for ds in ["mnist", "cifar", "imdb"]:
        rows = summary_by_dataset.get(ds, [])
        if not rows:
            continue
        xs = np.asarray([r["epoch"] for r in rows], dtype=np.float64)
        mean = np.asarray([r["c_mean"] for r in rows], dtype=np.float64)
        std = np.asarray([r["c_std"] for r in rows], dtype=np.float64)
        color = colors.get(ds)
        label = f"{DATASET_NAME.get(ds, ds)} Adaptive-DP"
        plt.plot(xs, mean, linewidth=2.0, color=color, label=label)
        plt.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)

    plt.xlabel("Epoch")
    plt.ylabel("Clipping Bound $C_t$")
    plt.grid(alpha=0.28)
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / f"clipping_bound_trajectory_multiseed_{run_id}.png"
    plt.savefig(str(out), dpi=190)
    plt.close()
    return out


def run_dataset_seed(dataset: str, seed: int) -> Dict:
    import module_c_adaptive_bridge as bridge
    import module_c_adaptive_dpsgd as mc

    data = bridge.load_dataset_features(dataset, max_test_cap=int(CFG["max_test_cap"]))
    xtr = np.asarray(data["X_train_feat"], dtype=np.float32)
    xte = np.asarray(data["X_test_feat"], dtype=np.float32)
    ytr = np.asarray(data["y_train"]).reshape(-1)
    yte = np.asarray(data["y_test"]).reshape(-1)
    feat_dim = int(xtr.shape[1])
    num_classes = int(len(np.unique(ytr)))

    b_out = bridge._load_module_b_init_or_raise(dataset, feat_dim, num_classes)
    init_model = b_out["initial_global_model"]
    ds_cfg = DATASET_OVERRIDES.get(dataset, {})

    result = mc.run_adaptive_dpsgd(
        init_model=init_model,
        X_lg_feat=xtr,
        y_lg=ytr,
        X_test_feat=xte,
        y_test=yte,
        global_epochs=int(CFG["global_epochs"]),
        global_lr=float(ds_cfg.get("global_lr", CFG["global_lr"])),
        batch_size=int(CFG["batch_size"]),
        clipping_norm=float(ds_cfg.get("clipping_norm", CFG["clipping_norm"])),
        noise_multiplier=float(CFG["noise_multiplier"]),
        delta=float(CFG["delta"]),
        target_epsilon=CFG["target_epsilon"],
        mode="adaptive_dp",
        gamma=float(CFG["gamma"]),
        eta_c=float(CFG["eta_c"]),
        C_init=float(ds_cfg.get("C_init", CFG["C_init"])),
        sigma_b_scale=float(CFG["sigma_b_scale"]),
        quantile_beta=float(CFG["quantile_beta"]),
        warmup_steps=int(CFG["warmup_steps"]),
        clip_min=float(CFG["clip_min"]),
        clip_max=float(ds_cfg.get("clip_max", CFG["clip_max"])),
        noise_backend=str(CFG["noise_backend"]),
        seed=int(seed),
        device_hint=str(CFG["device"]),
        max_epoch_log_lines=int(CFG["max_epoch_log_lines"]),
        safe_mode=bool(CFG["safe_mode"]),
        cpu_threads=int(CFG["cpu_threads"]),
        verbose=False,
    )

    return {
        "dataset": dataset,
        "seed": int(seed),
        "source_feat": data.get("source_feat"),
        "history": result.get("history", []),
        "final_metrics": result.get("final_metrics", {}),
        "config": result.get("config", {}),
        "module_b_cache": b_out.get("_cache_meta", {}),
    }


def summarize(all_runs: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    out = {}
    for ds, runs in all_runs.items():
        if not runs:
            continue
        # 以最短 history 为准，避免不同 seed 提前停止时长度不一致。
        min_len = min(len(r["history"]) for r in runs)
        rows = []
        for i in range(min_len):
            epoch = int(runs[0]["history"][i]["epoch"])
            c_vals = np.asarray([float(r["history"][i]["clipping_bound_t"]) for r in runs], dtype=np.float64)
            acc_vals = np.asarray([float(r["history"][i]["test_acc"]) for r in runs], dtype=np.float64)
            rows.append({
                "dataset": ds,
                "dataset_name": DATASET_NAME.get(ds, ds),
                "epoch": epoch,
                "c_mean": float(np.mean(c_vals)),
                "c_std": float(np.std(c_vals)),
                "acc_mean": float(np.mean(acc_vals)),
                "acc_std": float(np.std(acc_vals)),
                "num_seeds": int(len(runs)),
            })
        out[ds] = rows
    return out


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="模块 C：Adaptive-DP C_t trajectory 多 seed 图")
    p.add_argument("--seeds", type=int, default=3, help="默认 3；如需更稳可设 5")
    p.add_argument("--epochs", type=int, default=int(CFG["global_epochs"]), help="默认 300")
    return p


def main() -> int:
    args = make_parser().parse_args()
    CFG["global_epochs"] = int(args.epochs)

    ensure_dirs()
    if bool(CFG["safe_mode"]):
        set_safe_runtime(int(CFG["cpu_threads"]))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    txt_path = LOG_TXT_DIR / f"moduleC_clipping_trajectory_multiseed_{run_id}.log"
    json_path = LOG_JSON_DIR / f"moduleC_clipping_trajectory_multiseed_{run_id}.json"
    csv_path = EXPTAB_DIR / f"module_c_clipping_trajectory_multiseed_{run_id}.csv"

    seeds = [int(CFG["seed_base"]) + i for i in range(int(args.seeds))]
    datasets = ["mnist", "cifar", "imdb"]
    all_runs: Dict[str, List[Dict]] = {ds: [] for ds in datasets}
    t0 = time.time()

    with tee_console_to_file(txt_path):
        print("=" * 96)
        print("Figure X. Clipping bound trajectory（Adaptive-DP，多 seed）")
        print("=" * 96)
        print("run_id      :", run_id)
        print("datasets    :", datasets)
        print("seeds       :", seeds)
        print("epochs      :", CFG["global_epochs"])
        print("cpu_threads :", CFG["cpu_threads"])
        print("noise_backend:", CFG["noise_backend"])

        for ds in datasets:
            print("\n" + "-" * 96)
            print(f"[Dataset] {DATASET_NAME.get(ds, ds)}")
            print("-" * 96)
            for seed in seeds:
                print(f"  seed={seed} 运行 Adaptive-DP ...")
                run = run_dataset_seed(ds, seed)
                all_runs[ds].append(run)
                fm = run.get("final_metrics", {})
                hist = run.get("history", [])
                c_final = hist[-1]["clipping_bound_t"] if hist else float("nan")
                print("    final_acc={:.4f}, final_C_t={:.6f}, eps={}".format(
                    float(fm.get("final_test_accuracy", 0.0)),
                    float(c_final),
                    fm.get("final_epsilon_estimate", "N/A"),
                ))

        summary_by_dataset = summarize(all_runs)
        flat_rows = []
        for ds in datasets:
            flat_rows.extend(summary_by_dataset.get(ds, []))
        save_csv(flat_rows, csv_path)
        fig_path = plot_trajectory(summary_by_dataset, run_id)

        elapsed = time.time() - t0
        payload = {
            "run_id": run_id,
            "config": CFG,
            "seeds": seeds,
            "datasets": datasets,
            "runs": all_runs,
            "summary": summary_by_dataset,
            "files": {
                "txt": str(txt_path),
                "json": str(json_path),
                "csv": str(csv_path),
                "figure": str(fig_path),
            },
            "elapsed_s": float(elapsed),
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print("\n输出文件:")
        print("  FIG :", fig_path)
        print("  CSV :", csv_path)
        print("  JSON:", json_path)
        print("  TXT :", txt_path)
        print("总耗时(s): {:.2f}".format(elapsed))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

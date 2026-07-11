#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：模块 C 的 Adaptive Clipping 消融实验（Table X + Figure X）。

表名建议：Table X. Effect of adaptive clipping in private optimization
图名建议：Figure X. Adaptive clipping trajectory

实验目标：
比较 fixed clipping 与 adaptive clipping 在同一模块 C 私有优化骨架下的效果，
并额外验证 Adaptive-DP + BinomialMPC 噪声后端。

对比方法：
1) Plain：不加 DP，作为无隐私约束参考上界
2) Fixed-DP：固定 clipping bound + Gaussian noise
3) Adaptive-DP：adaptive clipping + Gaussian noise
4) Adaptive-DP + BinomialMPC：adaptive clipping + BinomialMPC 噪声后端

指标：
- epsilon
- final test accuracy
- Epoch@TargetAcc
- final clipping bound C_T
- training time

注意：
- 本脚本读取模块 B 已保存的 latest initialization model，不会重新运行模块 B。
- 线程限制写在代码开头，防止 CPU 线程爆炸。
"""

import os
# 必须尽早限制底层线程池，避免 DPSGD per-sample gradient 阶段把 CPU 打满。
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
    "seed": 42,
    "device": "cpu",
    "max_test_cap": 1000,
    # 为了全数据集快速稳定跑通，默认 300；如论文最终版要更长，可改这里。
    "global_epochs": 300,
    "global_lr": 0.01,
    "batch_size": 64,
    "clipping_norm": 0.7,
    "noise_multiplier": 1.0,
    "delta": 1e-5,
    "target_epsilon": None,
    # Adaptive clipping 参数沿用你前面已验证较稳定的一组。
    "gamma": 0.85,
    "eta_c": 0.08,
    "C_init": 0.7,
    "sigma_b_scale": 20.0,
    "quantile_beta": 0.35,
    "warmup_steps": 40,
    "clip_min": 0.05,
    "clip_max": 3.0,
    "max_epoch_log_lines": 10,
    # Epoch@TargetAcc 的固定目标，统一比较收敛速度。
    "target_acc": {
        "mnist": 0.90,
        "cifar": 0.90,
        "imdb": 0.68,
    },
}

# 数据集级微调：只影响本消融脚本，不改模块 C 核心实现。
# 说明：当前模块 C 的 plain 路径仍会调用梯度裁剪；因此 plain 使用极大 C，近似无裁剪训练。
# CIFAR 使用更严格的 fixed clipping，便于观察 adaptive clipping 对“过裁剪”的缓解作用。
DATASET_OVERRIDES = {
    "mnist": {
        "plain_clipping_norm": 1e6,
        "fixed_clipping_norm": 0.7,
        "adaptive_C_init": 0.7,
        "adaptive_clip_max": 3.0,
        "global_lr": 0.01,
    },
    "cifar": {
        "plain_clipping_norm": 1e6,
        "fixed_clipping_norm": 0.35,
        "adaptive_C_init": 0.7,
        "adaptive_clip_max": 3.0,
        "global_lr": 0.01,
    },
    "imdb": {
        "plain_clipping_norm": 1e6,
        "fixed_clipping_norm": 1.2,
        "adaptive_C_init": 0.7,
        "adaptive_clip_max": 3.0,
        "global_lr": 0.01,
    },
}

DATASET_NAME = {
    "mnist": "MNIST",
    "cifar": "CIFAR-10",
    "imdb": "IMDb",
}

METHODS = [
    {
        "key": "plain",
        "name": "Plain",
        "mode": "plain",
        "noise_backend": "gaussian",
        "desc": "no DP",
    },
    {
        "key": "fixed_dp",
        "name": "Fixed-DP",
        "mode": "fixed_dp",
        "noise_backend": "gaussian",
        "desc": "fixed clipping + Gaussian",
    },
    {
        "key": "adaptive_dp",
        "name": "Adaptive-DP",
        "mode": "adaptive_dp",
        "noise_backend": "gaussian",
        "desc": "adaptive clipping + Gaussian",
    },
    {
        "key": "adaptive_dp_binomial_mpc",
        "name": "Adaptive-DP + BinomialMPC",
        "mode": "adaptive_dp",
        "noise_backend": "binomial_mpc",
        "desc": "adaptive clipping + BinomialMPC",
    },
]


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


def epoch_to_target(history: List[Dict], target_acc: float) -> int:
    """返回首次达到目标 test accuracy 的 epoch；没达到则返回最后 epoch + 1。"""
    if not history:
        return -1
    for r in history:
        if float(r.get("test_acc", 0.0)) >= float(target_acc):
            return int(r.get("epoch", 0))
    return int(history[-1].get("epoch", 0)) + 1


def fmt_float(v, digits: int = 4) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    if np.isnan(x):
        return "N/A"
    if np.isfinite(x):
        return f"{x:.{digits}f}"
    return "inf"


def method_params(dataset: str, method: Dict) -> Dict:
    """为不同方法选择裁剪阈值等参数，避免 Plain 被误裁剪。"""
    ds_cfg = DATASET_OVERRIDES.get(dataset, {})
    mode = str(method["mode"])
    if mode == "plain":
        clipping = float(ds_cfg.get("plain_clipping_norm", 1e6))
        c_init = float(CFG["C_init"])
        clip_max = float(ds_cfg.get("adaptive_clip_max", CFG["clip_max"]))
    elif mode == "fixed_dp":
        clipping = float(ds_cfg.get("fixed_clipping_norm", CFG["clipping_norm"]))
        c_init = float(CFG["C_init"])
        clip_max = float(ds_cfg.get("adaptive_clip_max", CFG["clip_max"]))
    else:
        clipping = float(ds_cfg.get("fixed_clipping_norm", CFG["clipping_norm"]))
        c_init = float(ds_cfg.get("adaptive_C_init", CFG["C_init"]))
        clip_max = float(ds_cfg.get("adaptive_clip_max", CFG["clip_max"]))
    return {
        "clipping_norm": clipping,
        "C_init": c_init,
        "clip_max": clip_max,
        "global_lr": float(ds_cfg.get("global_lr", CFG["global_lr"])),
    }


def save_csv(rows: List[Dict], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Method", "epsilon", "Final Acc", "Epoch@TargetAcc", "Final C_T", "Time(s)", "TargetAcc", "NoiseBackend"])
        for r in rows:
            w.writerow([
                r["dataset_name"],
                r["method"],
                fmt_float(r["epsilon"], 4),
                fmt_float(r["final_acc"], 4),
                r["epoch_target"],
                fmt_float(r["final_c_t"], 6),
                fmt_float(r["time_s"], 2),
                fmt_float(r["target_acc"], 4),
                r["noise_backend"],
            ])


def print_table(rows: List[Dict]) -> None:
    print("\n" + "=" * 118)
    print("Table X. Effect of adaptive clipping in private optimization")
    print("-" * 118)
    print("{:<10} {:<28} {:<10} {:<12} {:<14} {:<12} {:<10}".format(
        "Dataset", "Method", "ε", "Final Acc", "Epoch@Acc", "Final C_T", "Time(s)"
    ))
    print("-" * 118)
    for r in rows:
        print("{:<10} {:<28} {:<10} {:<12} {:<14} {:<12} {:<10}".format(
            r["dataset_name"],
            r["method"],
            fmt_float(r["epsilon"], 3),
            fmt_float(r["final_acc"], 4),
            str(r["epoch_target"]),
            fmt_float(r["final_c_t"], 4),
            fmt_float(r["time_s"], 2),
        ))
    print("=" * 118)


def save_trajectory_plot(dataset: str, histories: Dict[str, List[Dict]], run_id: str) -> Path:
    """绘制 C_t trajectory，只展示 Fixed/Adaptive DP 方法，不画 Plain。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "Fixed-DP": "#7f7f7f",
        "Adaptive-DP": "#1f77b4",
        "Adaptive-DP + BinomialMPC": "#ff7f0e",
    }
    plt.figure(figsize=(8.5, 5.0))
    for name, hist in histories.items():
        if name == "Plain" or not hist:
            continue
        xs = [int(r["epoch"]) for r in hist]
        ys = [float(r["clipping_bound_t"]) for r in hist]
        plt.plot(xs, ys, linewidth=2.0, label=name, color=colors.get(name, None))
    plt.xlabel("Training Epoch")
    plt.ylabel("Clipping Bound $C_t$")
    plt.grid(alpha=0.28)
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / f"adaptive_clipping_trajectory_{dataset}_{run_id}.png"
    plt.savefig(str(out), dpi=180)
    plt.close()
    return out


def save_accuracy_plot(dataset: str, histories: Dict[str, List[Dict]], run_id: str) -> Path:
    """补充绘制 test accuracy 收敛曲线，方便观察 adaptive clipping 的效果。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "Plain": {"color": "#2ca02c", "ls": "--"},
        "Fixed-DP": {"color": "#7f7f7f", "ls": "-"},
        "Adaptive-DP": {"color": "#1f77b4", "ls": "-"},
        "Adaptive-DP + BinomialMPC": {"color": "#ff7f0e", "ls": "-"},
    }
    plt.figure(figsize=(8.5, 5.0))
    for name, hist in histories.items():
        if not hist:
            continue
        xs = [int(r["epoch"]) for r in hist]
        ys = [float(r["test_acc"]) for r in hist]
        st = styles.get(name, {})
        plt.plot(xs, ys, linewidth=2.0, label=name, color=st.get("color"), linestyle=st.get("ls", "-"))
    plt.xlabel("Training Epoch")
    plt.ylabel("Test Accuracy")
    plt.grid(alpha=0.28)
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / f"adaptive_clipping_accuracy_{dataset}_{run_id}.png"
    plt.savefig(str(out), dpi=180)
    plt.close()
    return out


def run_one_dataset(dataset: str, run_id: str) -> Dict:
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

    print("\n" + "-" * 118)
    print(f"[Dataset] {DATASET_NAME.get(dataset, dataset)}")
    print("-" * 118)
    print("source_feat     :", data.get("source_feat"))
    print("X_train_feat    :", xtr.shape)
    print("X_test_feat     :", xte.shape)
    print("moduleB_init    :", b_out.get("_cache_meta", {}).get("weights_pt"))
    print("chosen_init     :", b_out.get("candidate", {}).get("chosen_init_type", "unknown"))

    rows = []
    histories = {}
    raw_results = {}
    target_acc = float(CFG["target_acc"].get(dataset, 0.7))

    for method in METHODS:
        print("\n  运行方法:", method["name"], "|", method["desc"])
        mp = method_params(dataset, method)
        print("  参数: clipping_norm={:.6g}, C_init={:.6g}, clip_max={:.6g}, lr={:.6g}".format(
            mp["clipping_norm"], mp["C_init"], mp["clip_max"], mp["global_lr"]
        ))
        t0 = time.time()
        result = mc.run_adaptive_dpsgd(
            init_model=init_model,
            X_lg_feat=xtr,
            y_lg=ytr,
            X_test_feat=xte,
            y_test=yte,
            global_epochs=int(CFG["global_epochs"]),
            global_lr=float(mp["global_lr"]),
            batch_size=int(CFG["batch_size"]),
            clipping_norm=float(mp["clipping_norm"]),
            noise_multiplier=float(CFG["noise_multiplier"]),
            delta=float(CFG["delta"]),
            target_epsilon=CFG["target_epsilon"],
            mode=str(method["mode"]),
            gamma=float(CFG["gamma"]),
            eta_c=float(CFG["eta_c"]),
            C_init=float(mp["C_init"]),
            sigma_b_scale=float(CFG["sigma_b_scale"]),
            quantile_beta=float(CFG["quantile_beta"]),
            warmup_steps=int(CFG["warmup_steps"]),
            clip_min=float(CFG["clip_min"]),
            clip_max=float(mp["clip_max"]),
            noise_backend=str(method["noise_backend"]),
            seed=int(CFG["seed"]),
            device_hint=str(CFG["device"]),
            max_epoch_log_lines=int(CFG["max_epoch_log_lines"]),
            safe_mode=bool(CFG["safe_mode"]),
            cpu_threads=int(CFG["cpu_threads"]),
            verbose=True,
        )
        elapsed = time.time() - t0
        hist = result.get("history", [])
        fm = result.get("final_metrics", {})
        histories[method["name"]] = hist
        raw_results[method["key"]] = {
            "final_metrics": fm,
            "dp_info": result.get("dp_info", {}),
            "config": result.get("config", {}),
            "history": hist,
        }
        final_c = float(hist[-1].get("clipping_bound_t", np.nan)) if hist else float("nan")
        if method["mode"] == "plain":
            final_c = float("nan")
        row = {
            "dataset": dataset,
            "dataset_name": DATASET_NAME.get(dataset, dataset),
            "method": method["name"],
            "mode": method["mode"],
            "noise_backend": method["noise_backend"],
            "epsilon": float(fm.get("final_epsilon_estimate", float("inf"))),
            "final_acc": float(fm.get("final_test_accuracy", 0.0)),
            "epoch_target": int(epoch_to_target(hist, target_acc)),
            "final_c_t": final_c,
            "time_s": float(fm.get("total_train_time_s", elapsed)),
            "target_acc": target_acc,
        }
        rows.append(row)
        print("  => final_acc={:.4f}, eps={}, epoch@target={}, C_T={:.6f}, time={:.2f}s".format(
            row["final_acc"], fmt_float(row["epsilon"], 4), row["epoch_target"], row["final_c_t"], row["time_s"]
        ))

    traj_fig = save_trajectory_plot(dataset, histories, run_id)
    acc_fig = save_accuracy_plot(dataset, histories, run_id)
    return {
        "dataset": dataset,
        "rows": rows,
        "histories": histories,
        "raw_results": raw_results,
        "figures": {
            "trajectory": str(traj_fig),
            "accuracy": str(acc_fig),
        },
    }


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="模块 C Adaptive Clipping Table X 消融实验")
    p.add_argument("--dataset", choices=["all", "mnist", "cifar", "imdb"], default="all", help="默认 all")
    p.add_argument("--epochs", type=int, default=int(CFG["global_epochs"]), help="覆盖 CFG['global_epochs']")
    return p


def main() -> int:
    args = make_parser().parse_args()
    CFG["global_epochs"] = int(args.epochs)

    ensure_dirs()
    if bool(CFG["safe_mode"]):
        set_safe_runtime(int(CFG["cpu_threads"]))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    tag = args.dataset
    txt_path = LOG_TXT_DIR / f"moduleC_adaptive_clipping_table_{tag}_{run_id}.log"
    json_path = LOG_JSON_DIR / f"moduleC_adaptive_clipping_table_{tag}_{run_id}.json"
    csv_path = EXPTAB_DIR / f"module_c_adaptive_clipping_table_{tag}_{run_id}.csv"
    datasets = ["mnist", "cifar", "imdb"] if args.dataset == "all" else [args.dataset]

    all_rows = []
    outputs = {}
    t0 = time.time()

    with tee_console_to_file(txt_path):
        print("=" * 118)
        print("Table X. Effect of adaptive clipping in private optimization")
        print("=" * 118)
        print("run_id          :", run_id)
        print("datasets        :", datasets)
        print("epochs/lr       : {}/{}".format(CFG["global_epochs"], CFG["global_lr"]))
        print("batch_size      :", CFG["batch_size"])
        print("noise_multiplier:", CFG["noise_multiplier"])
        print("fixed C         :", CFG["clipping_norm"])
        print("adaptive C_init :", CFG["C_init"])
        print("target_acc      :", CFG["target_acc"])
        print("cpu_threads     :", CFG["cpu_threads"])
        print("说明            : 读取模块 B latest init，不重跑模块 B")

        for ds in datasets:
            out = run_one_dataset(ds, run_id)
            outputs[ds] = out
            all_rows.extend(out["rows"])

        print_table(all_rows)
        save_csv(all_rows, csv_path)

        elapsed = time.time() - t0
        payload = {
            "run_id": run_id,
            "config": CFG,
            "datasets": datasets,
            "rows": all_rows,
            "outputs": outputs,
            "files": {
                "txt": str(txt_path),
                "json": str(json_path),
                "csv": str(csv_path),
            },
            "elapsed_s": float(elapsed),
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print("\n输出文件:")
        print("  CSV :", csv_path)
        print("  JSON:", json_path)
        print("  TXT :", txt_path)
        for ds, out in outputs.items():
            print(f"  FIG[{ds}] trajectory:", out["figures"]["trajectory"])
            print(f"  FIG[{ds}] accuracy   :", out["figures"]["accuracy"])
        print("总耗时(s): {:.2f}".format(elapsed))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

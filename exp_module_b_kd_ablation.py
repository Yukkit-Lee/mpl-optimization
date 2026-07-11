#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：模块 B 的 KD 消融实验（Table X 风格）。

对应论文实验：Effect of consensus-guided distillation under heterogeneous data。
本实验只评估“初始化模型”本身的 accuracy，不评估后续模块 C 私有训练后的 final model。

比较对象：
1) NoKD-Avg  ：不做 KL 蒸馏，直接平均 local models
2) NoKD-Best ：不做 KL 蒸馏，选择 reference set 上最优 local model
3) KD-Avg    ：teacher ensemble + KL 蒸馏后，再平均 distilled local models
4) KD-Best   ：teacher ensemble + KL 蒸馏后，再选择最优 distilled local model

输出：
1) CSV：expTab/module_b_kd_ablation_{dataset}_{run_id}.csv
2) JSON：log/json/moduleB_kd_ablation_{dataset}_{run_id}.json
3) TXT日志：log/txt/moduleB_kd_ablation_{dataset}_{run_id}.log
4) 对比图：expFig/table3_NonIID/module_b_kd_ablation_{dataset}_{run_id}.png

注意：
- 默认跑 mnist/cifar/imdb 三个数据集，且每个数据集都跑 IID + Non-IID。
- 默认关闭 ABY3 密态聚合，只看 KD 对初始化质量的贡献；如需密态聚合，可加 --secure_aggregation。
- 线程限制写在代码开头，防止 CPU 线程爆炸导致死机。
"""

import os
# 必须尽早限制底层 BLAS/OpenMP 线程，避免 PyTorch/NumPy 把 CPU 打满。
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
from typing import Dict, List, Tuple

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_TXT_DIR = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"
EXPFIG_DIR = BASE_DIR / "expFig"
FIG_DIR = EXPFIG_DIR / "table3_NonIID"
EXPTAB_DIR = BASE_DIR / "expTab"

# 实验主配置：以后优先在这里改，不要在命令行后面挂一长串参数。
CFG = {
    "safe_mode": True,
    "cpu_threads": 1,
    "max_test_cap": 1000,
    "num_repeats": 5,
    "seed_base": 42,
    "num_clients": 5,
    "classes_per_client": 0,       # 0=沿用模块 B 默认 class-skew
    "ref_size": 200,
    "local_epochs": 80,
    "local_lr": 1e-2,
    "tau": 2.0,
    "lambda_kd": 1.0,
    "distill_epochs": 30,
    "distill_lr": 5e-3,
    "device": "cpu",
    "max_epoch_log_lines": 10,
    # 这个实验验证 KD 贡献，默认明文聚合以减少资源消耗；需要 ABY3 时用 --secure_aggregation。
    "use_secure_aggregation": False,
    "secure_max_threads": 1,
    # local model 默认使用普通 SGD，和你当前模块 B 主线保持一致。
    "local_use_dpsgd": False,
    "local_dpsgd_batch_size": 64,
    "local_dpsgd_clipping_norm": 1.0,
    "local_dpsgd_noise_multiplier": 1.0,
}

DATASET_NAME = {
    "mnist": "MNIST",
    "cifar": "CIFAR-10",
    "imdb": "IMDb",
}


def ensure_dirs() -> None:
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EXPTAB_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def tee_console_to_file(log_path: Path):
    """控制台和 txt 日志同时输出。"""
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
    """统一限制 CPU 线程，防止实验脚本把服务器拖死。"""
    t = max(1, int(max_threads))
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["MKL_NUM_THREADS"] = str(t)
    os.environ["OPENBLAS_NUM_THREADS"] = str(t)
    os.environ["NUMEXPR_NUM_THREADS"] = str(t)
    try:
        torch.set_num_threads(t)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(t)
    except Exception:
        # 如果 PyTorch 已经启动并行线程池，再设置 interop 可能报错；这里静默跳过。
        pass


def sync_base_cfg(secure_aggregation: bool) -> None:
    """
    把本脚本配置同步到已有 Table3 单次运行函数。
    这样可以复用已有模块 B 训练/蒸馏/聚合逻辑，而不破坏主流程。
    """
    import exp_table3_init_compare as base

    base.CFG.update({
        "safe_mode": bool(CFG["safe_mode"]),
        "cpu_threads": int(CFG["cpu_threads"]),
        "max_test_cap": int(CFG["max_test_cap"]),
        "num_clients": int(CFG["num_clients"]),
        "classes_per_client": int(CFG["classes_per_client"]),
        "ref_size": int(CFG["ref_size"]),
        "local_epochs": int(CFG["local_epochs"]),
        "local_lr": float(CFG["local_lr"]),
        "tau": float(CFG["tau"]),
        "lambda_kd": float(CFG["lambda_kd"]),
        "distill_epochs": int(CFG["distill_epochs"]),
        "distill_lr": float(CFG["distill_lr"]),
        "device": str(CFG["device"]),
        "use_secure_aggregation": bool(secure_aggregation),
        "secure_max_threads": int(CFG["secure_max_threads"]),
        "max_epoch_log_lines": int(CFG["max_epoch_log_lines"]),
        "local_use_dpsgd": bool(CFG["local_use_dpsgd"]),
        "local_dpsgd_batch_size": int(CFG["local_dpsgd_batch_size"]),
        "local_dpsgd_clipping_norm": float(CFG["local_dpsgd_clipping_norm"]),
        "local_dpsgd_noise_multiplier": float(CFG["local_dpsgd_noise_multiplier"]),
    })
    return base


def summarize_records(records: List[Dict]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """按 dataset/split 聚合 5 次重复实验的 mean/std。"""
    summary = {}
    pairs = sorted(set((r["dataset"], r["distribution"]) for r in records))
    for dataset, split in pairs:
        no = [r for r in records if r["dataset"] == dataset and r["distribution"] == split and not r["use_kd"]]
        kd = [r for r in records if r["dataset"] == dataset and r["distribution"] == split and r["use_kd"]]
        def stat(rows, key):
            vals = np.asarray([x[key] for x in rows], dtype=np.float64)
            return float(np.mean(vals)), float(np.std(vals))
        no_avg_m, no_avg_s = stat(no, "avg_test_acc")
        no_best_m, no_best_s = stat(no, "best_test_acc")
        kd_avg_m, kd_avg_s = stat(kd, "avg_test_acc")
        kd_best_m, kd_best_s = stat(kd, "best_test_acc")
        summary[(dataset, split)] = {
            "nokd_avg_mean": no_avg_m,
            "nokd_avg_std": no_avg_s,
            "nokd_best_mean": no_best_m,
            "nokd_best_std": no_best_s,
            "kd_avg_mean": kd_avg_m,
            "kd_avg_std": kd_avg_s,
            "kd_best_mean": kd_best_m,
            "kd_best_std": kd_best_s,
            "n": int(len(no)),
        }
    return summary


def fmt_ms(mean: float, std: float) -> str:
    return "{:.4f} ± {:.4f}".format(float(mean), float(std))


def print_table(summary: Dict[Tuple[str, str], Dict[str, float]]) -> None:
    """打印用户指定格式的 Table X。"""
    print("\n" + "=" * 104)
    print("Table X. Effect of consensus-guided distillation under heterogeneous data")
    print("比较的是模块 B candidate initialization model 的 test accuracy，不是模块 C final model。")
    print("-" * 104)
    print("{:<10} {:<10} {:<18} {:<18} {:<18} {:<18}".format(
        "Dataset", "Split", "NoKD-Avg", "NoKD-Best", "KD-Avg", "KD-Best"
    ))
    print("-" * 104)
    for dataset in ["mnist", "cifar", "imdb"]:
        for split in ["iid", "noniid"]:
            key = (dataset, split)
            if key not in summary:
                continue
            s = summary[key]
            print("{:<10} {:<10} {:<18} {:<18} {:<18} {:<18}".format(
                DATASET_NAME.get(dataset, dataset),
                "IID" if split == "iid" else "Non-IID",
                fmt_ms(s["nokd_avg_mean"], s["nokd_avg_std"]),
                fmt_ms(s["nokd_best_mean"], s["nokd_best_std"]),
                fmt_ms(s["kd_avg_mean"], s["kd_avg_std"]),
                fmt_ms(s["kd_best_mean"], s["kd_best_std"]),
            ))
    print("=" * 104)


def save_csv(summary: Dict[Tuple[str, str], Dict[str, float]], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Split", "NoKD-Avg", "NoKD-Best", "KD-Avg", "KD-Best"])
        for dataset in ["mnist", "cifar", "imdb"]:
            for split in ["iid", "noniid"]:
                key = (dataset, split)
                if key not in summary:
                    continue
                s = summary[key]
                w.writerow([
                    DATASET_NAME.get(dataset, dataset),
                    "IID" if split == "iid" else "Non-IID",
                    fmt_ms(s["nokd_avg_mean"], s["nokd_avg_std"]),
                    fmt_ms(s["nokd_best_mean"], s["nokd_best_std"]),
                    fmt_ms(s["kd_avg_mean"], s["kd_avg_std"]),
                    fmt_ms(s["kd_best_mean"], s["kd_best_std"]),
                ])


def save_plot(summary: Dict[Tuple[str, str], Dict[str, float]], dataset: str, run_id: str) -> Path:
    """按数据集画 IID/Non-IID 下 KD 前后的 avg/best 对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = ["iid", "noniid"]
    labels = ["IID", "Non-IID"]
    metrics = [
        ("NoKD-Avg", "nokd_avg_mean", "nokd_avg_std"),
        ("NoKD-Best", "nokd_best_mean", "nokd_best_std"),
        ("KD-Avg", "kd_avg_mean", "kd_avg_std"),
        ("KD-Best", "kd_best_mean", "kd_best_std"),
    ]

    x = np.arange(len(labels), dtype=np.float32)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32) * width

    plt.figure(figsize=(9, 5))
    colors = ["#9aa0a6", "#6f7782", "#4c78a8", "#f58518"]
    for i, (name, mean_key, std_key) in enumerate(metrics):
        means, stds = [], []
        for split in splits:
            s = summary.get((dataset, split))
            means.append(float(s[mean_key]) if s else 0.0)
            stds.append(float(s[std_key]) if s else 0.0)
        plt.bar(x + offsets[i], means, width, yerr=stds, capsize=4, label=name, color=colors[i], alpha=0.90)

    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Initial global model test accuracy")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    out = FIG_DIR / f"module_b_kd_ablation_{dataset}_{run_id}.png"
    plt.savefig(str(out), dpi=180)
    plt.close()
    return out


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="模块 B KD 消融实验：NoKD vs KD 初始化模型精度对比")
    p.add_argument("--dataset", choices=["all", "mnist", "cifar", "imdb"], default="all", help="默认 all，跑三个数据集")
    p.add_argument("--secure_aggregation", action="store_true", help="启用 ABY3 密态聚合；默认关闭以节省资源")
    p.add_argument("--repeats", type=int, default=int(CFG["num_repeats"]), help="重复随机种子次数，默认 5")
    return p


def main() -> int:
    args = make_parser().parse_args()
    CFG["num_repeats"] = int(args.repeats)
    CFG["use_secure_aggregation"] = bool(args.secure_aggregation)

    ensure_dirs()
    if bool(CFG["safe_mode"]):
        set_safe_runtime(int(CFG["cpu_threads"]))

    base = sync_base_cfg(secure_aggregation=bool(args.secure_aggregation))
    import feature_kd_bridge as bridge

    run_id = time.strftime("%Y%m%d_%H%M%S")
    tag = args.dataset
    txt_path = LOG_TXT_DIR / f"moduleB_kd_ablation_{tag}_{run_id}.log"
    json_path = LOG_JSON_DIR / f"moduleB_kd_ablation_{tag}_{run_id}.json"
    csv_path = EXPTAB_DIR / f"module_b_kd_ablation_{tag}_{run_id}.csv"

    datasets = ["mnist", "cifar", "imdb"] if args.dataset == "all" else [args.dataset]
    splits = ["iid", "noniid"]
    records: List[Dict] = []
    fig_paths: List[str] = []

    t0 = time.time()
    with tee_console_to_file(txt_path):
        print("=" * 104)
        print("模块 B KD 消融实验：NoKD vs KD initialization accuracy")
        print("=" * 104)
        print("run_id              :", run_id)
        print("datasets            :", datasets)
        print("splits              :", ["IID", "Non-IID"])
        print("num_repeats         :", CFG["num_repeats"])
        print("cpu_threads         :", CFG["cpu_threads"])
        print("use_secure_agg      :", CFG["use_secure_aggregation"])
        print("local_epochs/lr     :", CFG["local_epochs"], CFG["local_lr"])
        print("distill_epochs/lr   :", CFG["distill_epochs"], CFG["distill_lr"])
        print("tau/lambda_kd       :", CFG["tau"], CFG["lambda_kd"])
        print("说明                : 只比较模块 B 初始化模型，不接入模块 C 后续训练")

        for ds in datasets:
            print("\n" + "-" * 104)
            print(f"[Dataset] {DATASET_NAME.get(ds, ds)}")
            print("-" * 104)
            data = bridge.load_dataset_features(ds, max_test_cap=int(CFG["max_test_cap"]))
            xtr, ytr = data["X_train_feat"], data["y_train"]
            xte, yte = data["X_test_feat"], data["y_test"]
            print("source_feat         :", data.get("source_feat"))
            print("X_train_feat        :", xtr.shape)
            print("X_test_feat         :", xte.shape)

            for split in splits:
                print(f"\n  [Split] {'IID' if split == 'iid' else 'Non-IID'}")
                for rep in range(int(CFG["num_repeats"])):
                    seed = int(CFG["seed_base"]) + rep
                    # 同一个 seed 下先跑 NoKD，再跑 KD；二者 local training/split 逻辑一致，便于公平比较。
                    for use_kd in [False, True]:
                        row = base.run_single(
                            dataset_name=ds,
                            distribution=split,
                            use_kd=bool(use_kd),
                            X_train_feat=xtr,
                            y_train=ytr,
                            X_test_feat=xte,
                            y_test=yte,
                            seed=seed,
                        )
                        records.append(row)
                        print(
                            "    seed={:<4d} {:<5} avg={:.4f} best={:.4f}".format(
                                seed,
                                "KD" if use_kd else "NoKD",
                                row["avg_test_acc"],
                                row["best_test_acc"],
                            )
                        )

        summary = summarize_records(records)
        print_table(summary)
        save_csv(summary, csv_path)
        for ds in datasets:
            fig_paths.append(str(save_plot(summary, ds, run_id)))

        elapsed = time.time() - t0
        payload = {
            "run_id": run_id,
            "config": CFG,
            "datasets": datasets,
            "records": records,
            "summary": {"|".join(k): v for k, v in summary.items()},
            "outputs": {
                "txt": str(txt_path),
                "json": str(json_path),
                "csv": str(csv_path),
                "figures": fig_paths,
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
        for p in fig_paths:
            print("  FIG :", p)
        print("总耗时(s): {:.2f}".format(elapsed))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

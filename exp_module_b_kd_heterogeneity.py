#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：模块 B 的异构程度敏感性实验（KD gain vs heterogeneity）。

实验目标：证明在数据越异构时，consensus-guided KD 对初始化模型越有价值。

比较设置：
1) IID
2) Mild Non-IID
3) Severe Non-IID

多分类数据集（MNIST/CIFAR-10）：
- IID：随机等分
- Mild Non-IID：classes_per_client = 5
- Severe Non-IID：classes_per_client = 2

二分类数据集（IMDb）：
- IID：随机等分
- Mild Non-IID：label-skew alpha = 0.5
- Severe Non-IID：label-skew alpha = 0.1

比较对象：
- NoKD-Avg / KD-Avg / Delta-Avg
- NoKD-Best / KD-Best / Delta-Best

注意：
本实验只评估模块 B 的 candidate initialization model accuracy，
不是模块 C 后续 private training 的最终精度。

输出：
1) CSV：expTab/module_b_kd_heterogeneity_{dataset}_{run_id}.csv
2) JSON：log/json/moduleB_kd_heterogeneity_{dataset}_{run_id}.json
3) TXT日志：log/txt/moduleB_kd_heterogeneity_{dataset}_{run_id}.log
4) 图：expFig/table3_NonIID/module_b_kd_heterogeneity_{dataset}_{run_id}.png
"""

import os
# 尽早限制线程，避免 NumPy/PyTorch 底层线程池把 CPU 打满。
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

CFG = {
    "safe_mode": True,
    "cpu_threads": 1,
    "max_test_cap": 1000,
    "num_repeats": 5,
    "seed_base": 42,
    "num_clients": 5,
    "ref_size": 200,
    "local_epochs": 80,
    "local_lr": 1e-2,
    "tau": 2.0,
    "lambda_kd": 1.0,
    "distill_epochs": 30,
    "distill_lr": 5e-3,
    "device": "cpu",
    "max_epoch_log_lines": 10,
    # 默认关闭 ABY3，聚焦 KD 对异构性的贡献；需要时可加 --secure_aggregation。
    "use_secure_aggregation": False,
    "secure_max_threads": 1,
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

# 异构级别配置：多分类使用每客户端可见类别数，二分类使用 label-skew alpha。
HETERO_LEVELS = [
    {"name": "IID", "key": "iid", "classes_per_client": None, "binary_alpha": None},
    {"name": "Mild Non-IID", "key": "mild", "classes_per_client": 5, "binary_alpha": 0.5},
    {"name": "Severe Non-IID", "key": "severe", "classes_per_client": 2, "binary_alpha": 0.1},
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
    try:
        torch.set_num_threads(t)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(t)
    except Exception:
        pass


def sync_table3_cfg(secure_aggregation: bool) -> object:
    """复用已有 Table3 工具函数，并同步安全配置。"""
    import exp_table3_init_compare as base
    base.CFG.update({
        "safe_mode": bool(CFG["safe_mode"]),
        "cpu_threads": int(CFG["cpu_threads"]),
        "max_test_cap": int(CFG["max_test_cap"]),
        "num_clients": int(CFG["num_clients"]),
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


def run_one_setting(
    dataset_name: str,
    level: Dict,
    use_kd: bool,
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    X_test_feat: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    base,
) -> Dict:
    """
    单次运行：给定异构级别 + 是否 KD，返回 avg/best 初始化模型测试精度。
    这里复用已有模块 B 训练/蒸馏逻辑，只改变 split 强度。
    """
    import feature_kd_init as kd

    kd.set_seed(seed)
    X_train_feat = np.asarray(X_train_feat, dtype=np.float32)
    X_test_feat = np.asarray(X_test_feat, dtype=np.float32)
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    feat_dim = int(X_train_feat.shape[1])
    num_classes = kd.infer_num_classes(y_train)
    device = torch.device(str(CFG["device"]))

    if level["key"] == "iid":
        clients = base.make_iid_splits(
            X_train_feat=X_train_feat,
            y_train=y_train,
            num_clients=int(CFG["num_clients"]),
            seed=seed,
        )
        split_param = "IID"
    else:
        if num_classes == 2:
            alpha = float(level["binary_alpha"])
            clients = base.make_binary_label_skew_splits(
                X_train_feat=X_train_feat,
                y_train=y_train,
                num_clients=int(CFG["num_clients"]),
                seed=seed,
                alpha=alpha,
            )
            split_param = f"alpha={alpha}"
        else:
            cpc = int(level["classes_per_client"])
            clients = kd.make_class_skew_splits(
                X_train_feat=X_train_feat,
                y_train=y_train,
                num_clients=int(CFG["num_clients"]),
                classes_per_client=cpc,
                seed=seed,
            )
            split_param = f"classes/client={cpc}"

    local_models = []
    for c in clients:
        out = kd.train_local_model(
            c["X"], c["y"],
            feat_dim=feat_dim,
            num_classes=num_classes,
            device=device,
            epochs=int(CFG["local_epochs"]),
            lr=float(CFG["local_lr"]),
            seed=seed + int(c["client_id"]),
            max_epoch_log_lines=int(CFG["max_epoch_log_lines"]),
            verbose=False,
            use_dpsgd=bool(CFG["local_use_dpsgd"]),
            dpsgd_batch_size=int(CFG["local_dpsgd_batch_size"]),
            dpsgd_clipping_norm=float(CFG["local_dpsgd_clipping_norm"]),
            dpsgd_noise_multiplier=float(CFG["local_dpsgd_noise_multiplier"]),
        )
        local_models.append(out["model"])

    X_ref, y_ref = kd.build_reference_set(clients, ref_size=int(CFG["ref_size"]), seed=seed)

    if use_kd:
        teacher_probs = kd.build_ensemble_teacher(
            local_models=local_models,
            X_ref=X_ref,
            num_classes=num_classes,
            device=device,
            tau=float(CFG["tau"]),
        )
        final_models = []
        for c, m in zip(clients, local_models):
            d_out = kd.distill_local_model(
                model=m,
                X_local=c["X"],
                y_local=c["y"],
                X_ref=X_ref,
                teacher_probs_ref=teacher_probs,
                num_classes=num_classes,
                device=device,
                tau=float(CFG["tau"]),
                lambda_kd=float(CFG["lambda_kd"]),
                distill_epochs=int(CFG["distill_epochs"]),
                lr=float(CFG["distill_lr"]),
                max_epoch_log_lines=int(CFG["max_epoch_log_lines"]),
                verbose=False,
            )
            final_models.append(d_out["model"])
    else:
        final_models = local_models

    best_model_plain, _, _ = kd.select_best_local_model(
        models=final_models,
        X_ref=X_ref,
        y_ref=y_ref,
        num_classes=num_classes,
        device=device,
    )

    if bool(CFG["use_secure_aggregation"]):
        import secure_aggregation as sa
        sa_result = sa.secure_aggregate_and_select(
            distill_models=final_models,
            best_local_model=best_model_plain,
            X_ref=X_ref,
            y_ref=y_ref,
            feat_dim=feat_dim,
            num_classes=num_classes,
            max_threads=int(CFG["secure_max_threads"]),
            verbose=False,
        )
        avg_model = sa_result["avg_model"].to(device)
        best_model = sa_result["best_model"].to(device)
    else:
        avg_model = kd.average_model_weights(final_models, feat_dim, num_classes, device)
        best_model = best_model_plain

    avg_acc = kd.evaluate_model(avg_model, X_test_feat, y_test, num_classes, device)
    best_acc = kd.evaluate_model(best_model, X_test_feat, y_test, num_classes, device)

    return {
        "dataset": dataset_name,
        "heterogeneity": level["key"],
        "heterogeneity_name": level["name"],
        "split_param": split_param,
        "use_kd": bool(use_kd),
        "seed": int(seed),
        "avg_test_acc": float(avg_acc),
        "best_test_acc": float(best_acc),
    }


def summarize(records: List[Dict]) -> Dict[Tuple[str, str], Dict[str, float]]:
    out = {}
    for dataset in sorted(set(r["dataset"] for r in records)):
        for level in [x["key"] for x in HETERO_LEVELS]:
            no = [r for r in records if r["dataset"] == dataset and r["heterogeneity"] == level and not r["use_kd"]]
            kd = [r for r in records if r["dataset"] == dataset and r["heterogeneity"] == level and r["use_kd"]]
            if not no or not kd:
                continue
            def stat(rows, key):
                vals = np.asarray([r[key] for r in rows], dtype=np.float64)
                return float(np.mean(vals)), float(np.std(vals))
            no_avg_m, no_avg_s = stat(no, "avg_test_acc")
            no_best_m, no_best_s = stat(no, "best_test_acc")
            kd_avg_m, kd_avg_s = stat(kd, "avg_test_acc")
            kd_best_m, kd_best_s = stat(kd, "best_test_acc")
            out[(dataset, level)] = {
                "level_name": no[0]["heterogeneity_name"],
                "split_param": no[0]["split_param"],
                "nokd_avg_mean": no_avg_m,
                "nokd_avg_std": no_avg_s,
                "nokd_best_mean": no_best_m,
                "nokd_best_std": no_best_s,
                "kd_avg_mean": kd_avg_m,
                "kd_avg_std": kd_avg_s,
                "kd_best_mean": kd_best_m,
                "kd_best_std": kd_best_s,
                "delta_avg": kd_avg_m - no_avg_m,
                "delta_best": kd_best_m - no_best_m,
                "n": int(len(no)),
            }
    return out


def fmt_ms(mean: float, std: float) -> str:
    return "{:.4f} ± {:.4f}".format(float(mean), float(std))


def print_table(summary: Dict[Tuple[str, str], Dict[str, float]]) -> None:
    print("\n" + "=" * 130)
    print("模块 B 异构程度敏感性实验：数据越异构，KD 是否越有价值")
    print("-" * 130)
    print("{:<10} {:<16} {:<18} {:<18} {:<10} {:<18} {:<18} {:<10}".format(
        "Dataset", "Heterogeneity", "NoKD-Avg", "KD-Avg", "ΔAvg", "NoKD-Best", "KD-Best", "ΔBest"
    ))
    print("-" * 130)
    for dataset in ["mnist", "cifar", "imdb"]:
        for level in [x["key"] for x in HETERO_LEVELS]:
            key = (dataset, level)
            if key not in summary:
                continue
            s = summary[key]
            print("{:<10} {:<16} {:<18} {:<18} {:+.4f}    {:<18} {:<18} {:+.4f}".format(
                DATASET_NAME.get(dataset, dataset),
                s["level_name"],
                fmt_ms(s["nokd_avg_mean"], s["nokd_avg_std"]),
                fmt_ms(s["kd_avg_mean"], s["kd_avg_std"]),
                s["delta_avg"],
                fmt_ms(s["nokd_best_mean"], s["nokd_best_std"]),
                fmt_ms(s["kd_best_mean"], s["kd_best_std"]),
                s["delta_best"],
            ))
    print("=" * 130)


def save_csv(summary: Dict[Tuple[str, str], Dict[str, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Dataset", "Heterogeneity", "SplitParam",
            "NoKD-Avg", "KD-Avg", "Delta-Avg",
            "NoKD-Best", "KD-Best", "Delta-Best",
        ])
        for dataset in ["mnist", "cifar", "imdb"]:
            for level in [x["key"] for x in HETERO_LEVELS]:
                key = (dataset, level)
                if key not in summary:
                    continue
                s = summary[key]
                w.writerow([
                    DATASET_NAME.get(dataset, dataset),
                    s["level_name"],
                    s["split_param"],
                    fmt_ms(s["nokd_avg_mean"], s["nokd_avg_std"]),
                    fmt_ms(s["kd_avg_mean"], s["kd_avg_std"]),
                    "{:+.4f}".format(s["delta_avg"]),
                    fmt_ms(s["nokd_best_mean"], s["nokd_best_std"]),
                    fmt_ms(s["kd_best_mean"], s["kd_best_std"]),
                    "{:+.4f}".format(s["delta_best"]),
                ])


def save_plot(summary: Dict[Tuple[str, str], Dict[str, float]], dataset: str, run_id: str) -> Path:
    """画 KD gain 随异构程度变化的图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [x["key"] for x in HETERO_LEVELS]
    labels = [summary[(dataset, lv)]["level_name"].replace(" Non-IID", "") for lv in levels if (dataset, lv) in summary]
    x = np.arange(len(labels), dtype=np.float32)
    delta_avg = [summary[(dataset, lv)]["delta_avg"] for lv in levels if (dataset, lv) in summary]
    delta_best = [summary[(dataset, lv)]["delta_best"] for lv in levels if (dataset, lv) in summary]
    kd_avg = [summary[(dataset, lv)]["kd_avg_mean"] for lv in levels if (dataset, lv) in summary]
    no_avg = [summary[(dataset, lv)]["nokd_avg_mean"] for lv in levels if (dataset, lv) in summary]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    axes[0].plot(x, no_avg, marker="o", linewidth=2, label="NoKD-Avg", color="#7f7f7f")
    axes[0].plot(x, kd_avg, marker="o", linewidth=2, label="KD-Avg", color="#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Initial test accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    width = 0.34
    axes[1].bar(x - width / 2, delta_avg, width, label="KD gain on Avg", color="#1f77b4", alpha=0.85)
    axes[1].bar(x + width / 2, delta_best, width, label="KD gain on Best", color="#ff7f0e", alpha=0.85)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Accuracy gain after KD")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    out = FIG_DIR / f"module_b_kd_heterogeneity_{dataset}_{run_id}.png"
    fig.savefig(str(out), dpi=180)
    plt.close(fig)
    return out


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="模块 B KD 异构程度敏感性实验")
    p.add_argument("--dataset", choices=["all", "mnist", "cifar", "imdb"], default="all", help="默认 all")
    p.add_argument("--repeats", type=int, default=int(CFG["num_repeats"]), help="重复次数，默认 5")
    p.add_argument("--secure_aggregation", action="store_true", help="启用 ABY3 密态聚合，默认关闭")
    return p


def main() -> int:
    args = make_parser().parse_args()
    CFG["num_repeats"] = int(args.repeats)
    CFG["use_secure_aggregation"] = bool(args.secure_aggregation)

    ensure_dirs()
    if bool(CFG["safe_mode"]):
        set_safe_runtime(int(CFG["cpu_threads"]))

    base = sync_table3_cfg(secure_aggregation=bool(args.secure_aggregation))
    import feature_kd_bridge as bridge

    run_id = time.strftime("%Y%m%d_%H%M%S")
    tag = args.dataset
    txt_path = LOG_TXT_DIR / f"moduleB_kd_heterogeneity_{tag}_{run_id}.log"
    json_path = LOG_JSON_DIR / f"moduleB_kd_heterogeneity_{tag}_{run_id}.json"
    csv_path = EXPTAB_DIR / f"module_b_kd_heterogeneity_{tag}_{run_id}.csv"

    datasets = ["mnist", "cifar", "imdb"] if args.dataset == "all" else [args.dataset]
    records: List[Dict] = []
    fig_paths: List[str] = []
    t0 = time.time()

    with tee_console_to_file(txt_path):
        print("=" * 130)
        print("模块 B KD 异构程度敏感性实验：IID / Mild Non-IID / Severe Non-IID")
        print("=" * 130)
        print("run_id              :", run_id)
        print("datasets            :", datasets)
        print("num_repeats         :", CFG["num_repeats"])
        print("cpu_threads         :", CFG["cpu_threads"])
        print("use_secure_agg      :", CFG["use_secure_aggregation"])
        print("multi-class levels  : IID, classes_per_client=5, classes_per_client=2")
        print("binary levels       : IID, alpha=0.5, alpha=0.1")
        print("说明                : 只评估模块 B 初始化模型，不接入模块 C")

        for ds in datasets:
            print("\n" + "-" * 130)
            print(f"[Dataset] {DATASET_NAME.get(ds, ds)}")
            print("-" * 130)
            data = bridge.load_dataset_features(ds, max_test_cap=int(CFG["max_test_cap"]))
            xtr, ytr = data["X_train_feat"], data["y_train"]
            xte, yte = data["X_test_feat"], data["y_test"]
            print("source_feat         :", data.get("source_feat"))
            print("X_train_feat        :", xtr.shape)
            print("X_test_feat         :", xte.shape)

            for level in HETERO_LEVELS:
                print(f"\n  [Heterogeneity] {level['name']}")
                for rep in range(int(CFG["num_repeats"])):
                    seed = int(CFG["seed_base"]) + rep
                    rows_for_seed = []
                    for use_kd in [False, True]:
                        row = run_one_setting(
                            dataset_name=ds,
                            level=level,
                            use_kd=use_kd,
                            X_train_feat=xtr,
                            y_train=ytr,
                            X_test_feat=xte,
                            y_test=yte,
                            seed=seed,
                            base=base,
                        )
                        records.append(row)
                        rows_for_seed.append(row)
                    no, yes = rows_for_seed[0], rows_for_seed[1]
                    print(
                        "    seed={:<4d} {:<18} NoKD-Avg={:.4f} KD-Avg={:.4f} Δ={:+.4f} | NoKD-Best={:.4f} KD-Best={:.4f} Δ={:+.4f}".format(
                            seed,
                            no["split_param"],
                            no["avg_test_acc"], yes["avg_test_acc"], yes["avg_test_acc"] - no["avg_test_acc"],
                            no["best_test_acc"], yes["best_test_acc"], yes["best_test_acc"] - no["best_test_acc"],
                        )
                    )

        summary = summarize(records)
        print_table(summary)
        save_csv(summary, csv_path)
        for ds in datasets:
            fig_paths.append(str(save_plot(summary, ds, run_id)))

        elapsed = time.time() - t0
        payload = {
            "run_id": run_id,
            "config": CFG,
            "heterogeneity_levels": HETERO_LEVELS,
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

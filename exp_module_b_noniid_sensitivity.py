#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：模块 B 的 Non-IID 强度敏感性实验（Figure X 风格）。

实验问题：
数据越异构，consensus-guided KD 初始化是否越有价值？

图名建议：Figure X. Sensitivity to non-IID severity

对比曲线：
1) NoKD-Avg   ：PEA-like direct averaging，不蒸馏直接平均 local models
2) NoKD-Best  ：PEA-like accuracy strategy，不蒸馏选择 reference set 上最优 local model
3) KD-Chosen  ：本文方法，KD 后在 avg/best 中选择 reference set 上更优者

纵轴：
1) Init Accuracy：初始化模型在 test set 上的 accuracy
2) Epoch@TargetAcc：从该初始化继续做同样的轻量全局训练，到达目标 test accuracy 所需 epoch

注意：
- 本实验仍然只验证模块 B 初始化质量，不接入模块 C 的 DP/MPC 后续训练。
- 后续 global fine-tuning 只是为了衡量初始化收敛速度，不作为最终方法精度汇报。
- 线程限制写在文件开头，避免 CPU 占用过高导致死机。
"""

import os
# 尽早限制线程，防止 OpenMP/MKL/BLAS 过度并行。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import copy
import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_TXT_DIR = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"
EXPFIG_DIR = BASE_DIR / "expFig"
FIG_DIR = EXPFIG_DIR / "moduleB_nonIID_sensitivity"
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
    # Epoch@TargetAcc 的轻量全局训练配置。
    "target_epochs": 80,
    "target_lr": 1e-2,
    "target_acc": {
        "mnist": 0.78,
        "cifar": 0.70,
        "imdb": 0.68,
    },
}

DATASET_NAME = {
    "mnist": "MNIST",
    "cifar": "CIFAR-10",
    "imdb": "IMDb",
}

# 多分类按 classes per client 表达异构强度；数值越小，异构越强。
MULTICLASS_LEVELS = [
    {"key": "cpc10", "label": "10", "classes_per_client": 10, "severity": 0},
    {"key": "cpc5", "label": "5", "classes_per_client": 5, "severity": 1},
    {"key": "cpc3", "label": "3", "classes_per_client": 3, "severity": 2},
    {"key": "cpc2", "label": "2", "classes_per_client": 2, "severity": 3},
]

# 二分类没有 classes_per_client=10 的意义，改用 Dirichlet alpha；alpha 越小，异构越强。
BINARY_LEVELS = [
    {"key": "alpha1.0", "label": "1.0", "alpha": 1.0, "severity": 0},
    {"key": "alpha0.5", "label": "0.5", "alpha": 0.5, "severity": 1},
    {"key": "alpha0.3", "label": "0.3", "alpha": 0.3, "severity": 2},
    {"key": "alpha0.1", "label": "0.1", "alpha": 0.1, "severity": 3},
]

METHODS = ["NoKD-Avg", "NoKD-Best", "KD-Chosen"]


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


def make_binary_dirichlet_splits(
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[Dict]:
    """二分类 Dirichlet label-skew 划分，alpha 越小越 non-IID。"""
    rng = np.random.RandomState(seed)
    X = np.asarray(X_train_feat, dtype=np.float32)
    y = np.asarray(y_train).reshape(-1)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("make_binary_dirichlet_splits 仅用于二分类。")

    client_indices = [[] for _ in range(num_clients)]
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        # 避免极端 alpha 下某个 client 完全拿不到样本，加入很小平滑。
        proportions = rng.dirichlet(np.ones(num_clients) * float(alpha))
        cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
        parts = np.split(idx, cuts)
        for cid, part in enumerate(parts):
            client_indices[cid].extend(part.tolist())

    # 兜底：空 client 从最大 client 挪少量样本。
    for cid in range(num_clients):
        if len(client_indices[cid]) == 0:
            donor = int(np.argmax([len(v) for v in client_indices]))
            take = max(1, len(client_indices[donor]) // 20)
            moved = client_indices[donor][:take]
            client_indices[donor] = client_indices[donor][take:]
            client_indices[cid].extend(moved)

    clients = []
    for cid in range(num_clients):
        idx = np.asarray(client_indices[cid], dtype=np.int64)
        rng.shuffle(idx)
        Xi, yi = X[idx], y[idx]
        labels, counts = np.unique(yi, return_counts=True)
        hist = {int(k): int(v) for k, v in zip(labels, counts)}
        clients.append({
            "client_id": int(cid),
            "X": Xi,
            "y": yi,
            "num_samples": int(len(yi)),
            "class_hist": hist,
            "visible_classes": sorted([int(v) for v in labels.tolist()]),
        })
    return clients


def build_splits(dataset: str, level: Dict, X_train: np.ndarray, y_train: np.ndarray, num_classes: int, seed: int) -> Tuple[List[Dict], str]:
    """按数据集类型构造不同强度的 non-IID 划分。"""
    import exp_table3_init_compare as table3
    import feature_kd_init as kd

    if num_classes == 2:
        alpha = float(level["alpha"])
        clients = make_binary_dirichlet_splits(X_train, y_train, int(CFG["num_clients"]), alpha=alpha, seed=seed)
        return clients, f"Dirichlet alpha={alpha}"

    cpc = int(level["classes_per_client"])
    if cpc >= num_classes:
        clients = table3.make_iid_splits(X_train, y_train, int(CFG["num_clients"]), seed=seed)
    else:
        clients = kd.make_class_skew_splits(
            X_train_feat=X_train,
            y_train=y_train,
            num_clients=int(CFG["num_clients"]),
            classes_per_client=cpc,
            seed=seed,
        )
    return clients, f"classes_per_client={cpc}"


def train_global_until_target(
    init_model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    device: torch.device,
    target_acc: float,
    max_epochs: int,
    lr: float,
) -> Tuple[int, float]:
    """从初始化模型出发做轻量全局训练，返回达到目标精度所需 epoch。"""
    import feature_kd_init as kd

    model = copy.deepcopy(init_model).to(device)
    init_acc = kd.evaluate_model(model, X_test, y_test, num_classes, device)
    if init_acc >= target_acc:
        return 0, float(init_acc)

    criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=float(lr))
    tx = kd.to_tensor_x(X_train, device)
    ty = kd.to_tensor_y(y_train, num_classes, device)

    best_acc = float(init_acc)
    for ep in range(1, int(max_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(tx)
        if num_classes == 2:
            loss = criterion(logits, ty)
        else:
            loss = criterion(logits, ty.view(-1))
        loss.backward()
        optimizer.step()

        acc = kd.evaluate_model(model, X_test, y_test, num_classes, device)
        best_acc = max(best_acc, float(acc))
        if acc >= target_acc:
            return int(ep), float(acc)

    # 未达到目标时，用 max_epochs + 1 表示“超预算仍未达到”。
    return int(max_epochs) + 1, float(best_acc)


def run_one_seed(dataset: str, level: Dict, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed: int) -> List[Dict]:
    """一个 seed 下完成 NoKD-Avg / NoKD-Best / KD-Chosen 三种初始化对比。"""
    import feature_kd_init as kd

    kd.set_seed(seed)
    device = torch.device(str(CFG["device"]))
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    feat_dim = int(X_train.shape[1])
    num_classes = kd.infer_num_classes(y_train)

    clients, split_param = build_splits(dataset, level, X_train, y_train, num_classes, seed)

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
        )
        local_models.append(out["model"])

    X_ref, y_ref = kd.build_reference_set(clients, ref_size=int(CFG["ref_size"]), seed=seed)

    nokd_avg = kd.average_model_weights(local_models, feat_dim, num_classes, device)
    nokd_best, _, _ = kd.select_best_local_model(local_models, X_ref, y_ref, num_classes, device)

    teacher_probs = kd.build_ensemble_teacher(local_models, X_ref, num_classes, device, tau=float(CFG["tau"]))
    distilled = []
    for c, model in zip(clients, local_models):
        d_out = kd.distill_local_model(
            model=model,
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
        distilled.append(d_out["model"])

    kd_avg = kd.average_model_weights(distilled, feat_dim, num_classes, device)
    kd_best, _, _ = kd.select_best_local_model(distilled, X_ref, y_ref, num_classes, device)
    candidate = kd.evaluate_candidate_models(kd_avg, kd_best, X_ref, y_ref, num_classes, device)
    kd_chosen = candidate["chosen_model"]

    target = float(CFG["target_acc"].get(dataset, 0.70))
    models = {
        "NoKD-Avg": nokd_avg,
        "NoKD-Best": nokd_best,
        "KD-Chosen": kd_chosen,
    }

    rows = []
    for method, model in models.items():
        init_acc = kd.evaluate_model(model, X_test, y_test, num_classes, device)
        epoch_target, reached_acc = train_global_until_target(
            init_model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=num_classes,
            device=device,
            target_acc=target,
            max_epochs=int(CFG["target_epochs"]),
            lr=float(CFG["target_lr"]),
        )
        rows.append({
            "dataset": dataset,
            "level_key": level["key"],
            "level_label": str(level["label"]),
            "severity": int(level["severity"]),
            "split_param": split_param,
            "method": method,
            "seed": int(seed),
            "target_acc": float(target),
            "init_acc": float(init_acc),
            "epoch_to_target": int(epoch_target),
            "reached_acc": float(reached_acc),
            "chosen_type": candidate.get("chosen_init_type", "unknown") if method == "KD-Chosen" else "",
        })
    return rows


def summarize(records: List[Dict]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    out = {}
    keys = sorted(set((r["dataset"], r["level_key"], r["method"]) for r in records))
    for key in keys:
        rs = [r for r in records if (r["dataset"], r["level_key"], r["method"]) == key]
        init_vals = np.asarray([r["init_acc"] for r in rs], dtype=np.float64)
        epoch_vals = np.asarray([r["epoch_to_target"] for r in rs], dtype=np.float64)
        out[key] = {
            "dataset": key[0],
            "level_key": key[1],
            "level_label": rs[0]["level_label"],
            "severity": int(rs[0]["severity"]),
            "split_param": rs[0]["split_param"],
            "method": key[2],
            "target_acc": float(rs[0]["target_acc"]),
            "init_acc_mean": float(np.mean(init_vals)),
            "init_acc_std": float(np.std(init_vals)),
            "epoch_target_mean": float(np.mean(epoch_vals)),
            "epoch_target_std": float(np.std(epoch_vals)),
            "n": int(len(rs)),
        }
    return out


def fmt_ms(mean: float, std: float) -> str:
    return "{:.4f} ± {:.4f}".format(float(mean), float(std))


def print_table(summary: Dict[Tuple[str, str, str], Dict[str, float]]) -> None:
    print("\n" + "=" * 132)
    print("Figure X 数据表：Sensitivity to non-IID severity")
    print("-" * 132)
    print("{:<10} {:<10} {:<12} {:<18} {:<18} {:<18}".format(
        "Dataset", "Severity", "Method", "InitAcc", "Epoch@Target", "SplitParam"
    ))
    print("-" * 132)
    for dataset in ["mnist", "cifar", "imdb"]:
        levels = BINARY_LEVELS if dataset == "imdb" else MULTICLASS_LEVELS
        for level in levels:
            for method in METHODS:
                key = (dataset, level["key"], method)
                if key not in summary:
                    continue
                s = summary[key]
                print("{:<10} {:<10} {:<12} {:<18} {:<18} {:<18}".format(
                    DATASET_NAME.get(dataset, dataset),
                    s["level_label"],
                    method,
                    fmt_ms(s["init_acc_mean"], s["init_acc_std"]),
                    fmt_ms(s["epoch_target_mean"], s["epoch_target_std"]),
                    s["split_param"],
                ))
    print("=" * 132)


def save_csv(summary: Dict[Tuple[str, str, str], Dict[str, float]], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Dataset", "Severity", "Method", "SplitParam", "TargetAcc",
            "InitAccuracy", "Epoch@TargetAcc",
        ])
        for dataset in ["mnist", "cifar", "imdb"]:
            levels = BINARY_LEVELS if dataset == "imdb" else MULTICLASS_LEVELS
            for level in levels:
                for method in METHODS:
                    key = (dataset, level["key"], method)
                    if key not in summary:
                        continue
                    s = summary[key]
                    w.writerow([
                        DATASET_NAME.get(dataset, dataset),
                        s["level_label"],
                        method,
                        s["split_param"],
                        "{:.4f}".format(s["target_acc"]),
                        fmt_ms(s["init_acc_mean"], s["init_acc_std"]),
                        fmt_ms(s["epoch_target_mean"], s["epoch_target_std"]),
                    ])


def save_figure(summary: Dict[Tuple[str, str, str], Dict[str, float]], dataset: str, run_id: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = BINARY_LEVELS if dataset == "imdb" else MULTICLASS_LEVELS
    xs = np.arange(len(levels), dtype=np.float32)
    labels = [str(x["label"]) for x in levels]
    xlabel = "Dirichlet alpha (smaller = more heterogeneous)" if dataset == "imdb" else "Classes per client (smaller = more heterogeneous)"

    colors = {
        "NoKD-Avg": "#7f7f7f",
        "NoKD-Best": "#b07aa1",
        "KD-Chosen": "#1f77b4",
    }
    markers = {
        "NoKD-Avg": "o",
        "NoKD-Best": "s",
        "KD-Chosen": "D",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for method in METHODS:
        means, stds = [], []
        et_means, et_stds = [], []
        for level in levels:
            s = summary[(dataset, level["key"], method)]
            means.append(s["init_acc_mean"])
            stds.append(s["init_acc_std"])
            et_means.append(s["epoch_target_mean"])
            et_stds.append(s["epoch_target_std"])
        axes[0].errorbar(xs, means, yerr=stds, marker=markers[method], linewidth=2, capsize=4, label=method, color=colors[method])
        axes[1].errorbar(xs, et_means, yerr=et_stds, marker=markers[method], linewidth=2, capsize=4, label=method, color=colors[method])

    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Init Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Epoch@TargetAcc (lower is better)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    out = FIG_DIR / f"module_b_noniid_sensitivity_{dataset}_{run_id}.png"
    fig.savefig(str(out), dpi=180)
    plt.close(fig)
    return out


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="模块 B Non-IID 强度敏感性实验")
    p.add_argument("--dataset", choices=["all", "mnist", "cifar", "imdb"], default="all", help="默认 all")
    p.add_argument("--repeats", type=int, default=int(CFG["num_repeats"]), help="重复次数，默认 5")
    return p


def main() -> int:
    args = make_parser().parse_args()
    CFG["num_repeats"] = int(args.repeats)

    ensure_dirs()
    if bool(CFG["safe_mode"]):
        set_safe_runtime(int(CFG["cpu_threads"]))

    import feature_kd_bridge as bridge

    run_id = time.strftime("%Y%m%d_%H%M%S")
    tag = args.dataset
    txt_path = LOG_TXT_DIR / f"moduleB_noniid_sensitivity_{tag}_{run_id}.log"
    json_path = LOG_JSON_DIR / f"moduleB_noniid_sensitivity_{tag}_{run_id}.json"
    csv_path = EXPTAB_DIR / f"module_b_noniid_sensitivity_{tag}_{run_id}.csv"

    datasets = ["mnist", "cifar", "imdb"] if args.dataset == "all" else [args.dataset]
    records: List[Dict] = []
    fig_paths: List[str] = []
    t0 = time.time()

    with tee_console_to_file(txt_path):
        print("=" * 120)
        print("Figure X. Sensitivity to non-IID severity（模块 B）")
        print("=" * 120)
        print("run_id          :", run_id)
        print("datasets        :", datasets)
        print("num_repeats     :", CFG["num_repeats"])
        print("cpu_threads     :", CFG["cpu_threads"])
        print("target_epochs   :", CFG["target_epochs"])
        print("target_acc      :", CFG["target_acc"])
        print("说明            : Init Accuracy + Epoch@TargetAcc；只验证模块 B 初始化质量")

        for ds in datasets:
            print("\n" + "-" * 120)
            print(f"[Dataset] {DATASET_NAME.get(ds, ds)}")
            print("-" * 120)
            data = bridge.load_dataset_features(ds, max_test_cap=int(CFG["max_test_cap"]))
            X_train, y_train = data["X_train_feat"], data["y_train"]
            X_test, y_test = data["X_test_feat"], data["y_test"]
            print("source_feat     :", data.get("source_feat"))
            print("X_train_feat    :", X_train.shape)
            print("X_test_feat     :", X_test.shape)

            levels = BINARY_LEVELS if ds == "imdb" else MULTICLASS_LEVELS
            for level in levels:
                print(f"\n  [Severity] {level['label']}")
                for rep in range(int(CFG["num_repeats"])):
                    seed = int(CFG["seed_base"]) + rep
                    rows = run_one_seed(ds, level, X_train, y_train, X_test, y_test, seed)
                    records.extend(rows)
                    msg = []
                    for r in rows:
                        msg.append("{}: init={:.4f}, ep@target={}".format(r["method"], r["init_acc"], r["epoch_to_target"]))
                    print("    seed={:<4d} {} | {}".format(seed, rows[0]["split_param"], " ; ".join(msg)))

        summary = summarize(records)
        print_table(summary)
        save_csv(summary, csv_path)
        for ds in datasets:
            fig_paths.append(str(save_figure(summary, ds, run_id)))

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

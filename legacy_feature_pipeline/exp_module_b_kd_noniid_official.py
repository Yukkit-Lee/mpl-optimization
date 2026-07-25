#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LEGACY_FEATURE_PIPELINE: 2048-D feature + linear-head implementation.
# Retained only for historical reproducibility; not part of the current main method.
"""
用途：模块 B 的 Non-IID KD 消融实验。

对应论文问题：
- 在新模块 A（official MTT distilled compact set）输出的紧凑训练数据上，
  KD alignment 是否能缓解 non-IID 多客户端 local knowledge conflict？

实验流程：
1) official MTT distilled data -> 训练同款 MTT ConvNet。
2) 抽取 classifier 前 2048 维特征。
3) 构造 IID / Mild / Moderate / Severe Non-IID 多客户端划分。
4) 对比 NoKD 与 KD 两个分支下的 Avg / ReferenceWeighted 初始化。
5) test set 只用于最终报告；reference/validation set 只用于 KD、加权与候选选择。

输出：
- TXT: log/txt/moduleB_kd_noniid_official_{dataset}_{run_id}.log
- JSON: log/json/moduleB_kd_noniid_official_{dataset}_{run_id}.json
- CSV: expTab/moduleB_kd_noniid_official_{dataset}_{run_id}.csv
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import copy
import csv
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from project_paths import PATHS

SOURCE_DIR = Path(__file__).resolve().parent
BASE_DIR = PATHS.output_root
LOG_TXT_DIR = BASE_DIR / "log" / "txt"
LOG_JSON_DIR = BASE_DIR / "log" / "json"
EXPTAB_DIR = BASE_DIR / "expTab"
EXPPAPER_TAB_DIR = BASE_DIR / "expPaperTab"
PROGRESS_PATH = BASE_DIR / "progressLog" / "project_progress.md"

# 复用当前 Figure 5 的 official distilled -> MTT ConvNet -> feature extraction 流程。
import exp_official_distilled_figure5_init as fig5

CFG = {
    "dataset": "cifar10",
    "seeds": [42, 43, 44],
    "cpu_threads": 1,
    "num_clients": 5,
    # 论文表使用较短 local pretraining，让 non-IID 下 local conflict 更明显，
    # 从而更清楚观察 KD alignment 对初始化质量的改善。
    "local_epochs": 45,
    "local_lr": 0.01,
    "kd_epochs": 20,
    "kd_lr": 0.005,
    "tau": 2.0,
    "lambda_kd": 3.0,
    "ref_weight_temperature": 0.08,
    "global_steps": 220,
    "global_lr": 0.02,
    "eval_interval": 20,
    "target_acc_ratio": 0.98,
    "max_test_samples": 3000,
    "selection_ref_per_class": 10,
    "selection_ref_max_samples": 1000,
    "feature_extractor_steps": None,
}

HETEROGENEITY = [
    ("IID", "classes_per_client=all", None),
    ("Mild", "classes_per_client=5", 5),
    ("Moderate", "classes_per_client=3", 3),
    ("Severe", "classes_per_client=2", 2),
]

DATASET_HETEROGENEITY = {
    # CIFAR-10 类别少，直接沿用 5/3/2 class-skew。
    "cifar10": HETEROGENEITY,
    # CIFAR-100 / Tiny 类别更多，如果仍用 5/3/2 会过强且三个 split 差异不明显。
    # 这里按类别总数设置 Mild/Moderate/Severe 的相对异构强度。
    "cifar100": [
        ("IID", "classes_per_client=all", None),
        ("Mild", "classes_per_client=50", 50),
        ("Moderate", "classes_per_client=25", 25),
        ("Severe", "classes_per_client=10", 10),
    ],
    "tiny": [
        ("IID", "classes_per_client=all", None),
        ("Mild", "classes_per_client=100", 100),
        ("Moderate", "classes_per_client=50", 50),
        ("Severe", "classes_per_client=20", 20),
    ],
}


def get_heterogeneity(dataset: str):
    return DATASET_HETEROGENEITY.get(dataset, HETEROGENEITY)


def ensure_dirs() -> None:
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    EXPTAB_DIR.mkdir(parents=True, exist_ok=True)
    EXPPAPER_TAB_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def tee_console_to_file(path: Path):
    class Tee:
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
    with path.open("w", encoding="utf-8") as f:
        tee = Tee(old_out, f)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


def set_safe_runtime(seed: int) -> None:
    t = int(CFG["cpu_threads"])
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["MKL_NUM_THREADS"] = str(t)
    os.environ["OPENBLAS_NUM_THREADS"] = str(t)
    os.environ["NUMEXPR_NUM_THREADS"] = str(t)
    try:
        torch.set_num_threads(t)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(t)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LinearHead(nn.Module):
    """特征级线性分类头。"""
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        return self.fc(x)


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(X, dtype=np.float32))
        yy = torch.from_numpy(np.asarray(y, dtype=np.int64).reshape(-1))
        pred = model(x).argmax(1)
        return float((pred == yy).float().mean().item())


def cross_entropy_loss(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(X, dtype=np.float32))
        yy = torch.from_numpy(np.asarray(y, dtype=np.int64).reshape(-1))
        return float(F.cross_entropy(model(x), yy).item())


def make_splits(X: np.ndarray, y: np.ndarray, num_clients: int, classes_per_client, seed: int) -> List[Dict]:
    """构造 IID 或 class-skew Non-IID 划分。"""
    rng = np.random.RandomState(seed)
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    clients = []
    if classes_per_client is None:
        idx = np.arange(len(y))
        rng.shuffle(idx)
        parts = np.array_split(idx, num_clients)
    else:
        class_to_idx = {c: np.where(y == c)[0].copy() for c in classes}
        for ids in class_to_idx.values():
            rng.shuffle(ids)
        parts = [[] for _ in range(num_clients)]
        class_order = classes.copy()
        rng.shuffle(class_order)
        # 每个类别分配给若干客户端；classes_per_client 越小，class-skew 越强。
        for ci, c in enumerate(class_order):
            owner_count = max(1, min(num_clients, int(np.ceil(num_clients * float(classes_per_client) / len(classes)))))
            owners = [(ci + j) % num_clients for j in range(owner_count)]
            chunks = np.array_split(class_to_idx[c], owner_count)
            for owner, chunk in zip(owners, chunks):
                parts[owner].extend(chunk.tolist())
        parts = [np.asarray(p, dtype=np.int64) for p in parts]
    for cid, ids in enumerate(parts):
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            ids = rng.choice(np.arange(len(y)), size=1, replace=False)
        hist = {int(c): int(np.sum(y[ids] == c)) for c in np.unique(y[ids])}
        clients.append({"client_id": cid, "idx": ids, "X": X[ids], "y": y[ids], "hist": hist})
    return clients


def train_supervised(X: np.ndarray, y: np.ndarray, feat_dim: int, num_classes: int, seed: int, epochs: int, lr: float) -> nn.Module:
    set_safe_runtime(seed)
    model = LinearHead(feat_dim, num_classes)
    opt = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=0.0)
    x = torch.from_numpy(np.asarray(X, dtype=np.float32))
    yy = torch.from_numpy(np.asarray(y, dtype=np.int64).reshape(-1))
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), yy)
        loss.backward()
        opt.step()
    return model


def teacher_probs(models: List[nn.Module], X_ref: np.ndarray, tau: float) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(X_ref, dtype=np.float32))
    probs = []
    with torch.no_grad():
        for m in models:
            m.eval()
            probs.append(F.softmax(m(x) / float(tau), dim=1))
    return torch.stack(probs, 0).mean(0)


def distill_models(models: List[nn.Module], X_ref: np.ndarray, y_ref: np.ndarray, tau: float, lambda_kd: float, epochs: int, lr: float) -> List[nn.Module]:
    """在 reference set 上做 KL 蒸馏，不使用 test set。"""
    x = torch.from_numpy(np.asarray(X_ref, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y_ref, dtype=np.int64).reshape(-1))
    teacher = teacher_probs(models, X_ref, tau).detach()
    out = []
    for src in models:
        m = copy.deepcopy(src).cpu()
        opt = torch.optim.SGD(m.parameters(), lr=float(lr), momentum=0.0)
        for _ in range(int(epochs)):
            opt.zero_grad()
            logits = m(x)
            sup = F.cross_entropy(logits, y)
            logp = F.log_softmax(logits / float(tau), dim=1)
            kd = F.kl_div(logp, teacher, reduction="batchmean") * (float(tau) ** 2)
            loss = sup + float(lambda_kd) * kd
            loss.backward()
            opt.step()
        out.append(m)
    return out


def average_models(models: List[nn.Module], feat_dim: int, num_classes: int) -> nn.Module:
    out = LinearHead(feat_dim, num_classes)
    state = {}
    keys = list(models[0].state_dict().keys())
    for k in keys:
        state[k] = torch.stack([m.state_dict()[k].detach().cpu().float() for m in models], 0).mean(0)
    out.load_state_dict(state)
    return out


def reference_weighted(models: List[nn.Module], X_ref: np.ndarray, y_ref: np.ndarray, feat_dim: int, num_classes: int, temp: float) -> Tuple[nn.Module, Dict]:
    accs, losses = [], []
    for m in models:
        accs.append(accuracy(m, X_ref, y_ref))
        losses.append(cross_entropy_loss(m, X_ref, y_ref))
    z = np.asarray(accs, dtype=np.float64) / max(float(temp), 1e-6)
    z = z - np.max(z)
    w = np.exp(z)
    w = w / np.sum(w)
    out = LinearHead(feat_dim, num_classes)
    state = {}
    for k in models[0].state_dict().keys():
        val = None
        for wi, m in zip(w, models):
            cur = m.state_dict()[k].detach().cpu().float() * float(wi)
            val = cur if val is None else val + cur
        state[k] = val
    out.load_state_dict(state)
    return out, {"ref_accs": accs, "ref_losses": losses, "weights": w.tolist()}


def train_global_probe(init_model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, target_acc: float) -> Dict:
    """轻量后续训练，用于 FinalAcc / Step@TargetAcc / EarlyAUC。"""
    model = copy.deepcopy(init_model).cpu()
    opt = torch.optim.SGD(model.parameters(), lr=float(CFG["global_lr"]), momentum=0.0)
    x = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y_train, dtype=np.int64).reshape(-1))
    hist = [{"step": 0, "test_acc": accuracy(model, X_test, y_test)}]
    log_steps = set(range(int(CFG["eval_interval"]), int(CFG["global_steps"]) + 1, int(CFG["eval_interval"])))
    log_steps.add(int(CFG["global_steps"]))
    for step in range(1, int(CFG["global_steps"]) + 1):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        if step in log_steps:
            hist.append({"step": int(step), "test_acc": accuracy(model, X_test, y_test)})
    final_acc = float(hist[-1]["test_acc"])
    step_target = None
    for r in hist:
        if float(r["test_acc"]) >= float(target_acc):
            step_target = int(r["step"])
            break
    xs = np.asarray([r["step"] for r in hist], dtype=np.float64)
    ys = np.asarray([r["test_acc"] for r in hist], dtype=np.float64)
    auc = float(np.trapz(ys, xs) / max(float(xs[-1] - xs[0]), 1.0))
    return {"history": hist, "final_acc": final_acc, "step_target": step_target, "early_auc": auc}


def prepare_features(dataset: str, seed: int) -> Dict:
    """每个 seed 只训练一次 MTT ConvNet，再复用特征跑所有划分。"""
    set_safe_runtime(seed)
    fig5.CFG["cpu_threads"] = CFG["cpu_threads"]
    fig5.CFG["seed"] = seed
    cfg_d = fig5.DATASET_CFG[dataset]
    x_dist, y_dist, image_files = fig5.load_official_distilled(dataset)
    x_test, y_test, full_train_n, full_test_n = fig5.load_test_tensors(dataset, int(CFG["max_test_samples"]))
    steps = int(CFG["feature_extractor_steps"] or cfg_d["global_steps"])
    X_train, X_test, extractor_acc, model, scaler = fig5.make_convnet_features(
        dataset, x_dist, y_dist, x_test, y_test, steps=steps, eval_interval=int(cfg_d["eval_interval"])
    )
    x_ref, y_ref = fig5.load_real_reference_tensors(
        dataset,
        int(CFG["selection_ref_per_class"]),
        int(CFG["selection_ref_max_samples"]),
        seed,
    )
    device = fig5.select_device()
    X_ref_raw = fig5.extract_convnet_features(model, x_ref, device)
    X_ref = scaler.transform(X_ref_raw).astype(np.float32)
    return {
        "X_train": X_train,
        "y_train": y_dist.numpy().astype(np.int64),
        "X_test": X_test,
        "y_test": y_test.numpy().astype(np.int64),
        "X_ref": X_ref,
        "y_ref": y_ref.numpy().astype(np.int64),
        "feat_dim": int(X_train.shape[1]),
        "num_classes": int(len(np.unique(y_dist.numpy()))),
        "extractor_acc": float(extractor_acc),
        "full_train_n": int(full_train_n),
        "full_test_n": int(full_test_n),
    }


def run_one_setting(data: Dict, hetero_name: str, split_param: str, classes_per_client, seed: int) -> List[Dict]:
    X, y = data["X_train"], data["y_train"]
    feat_dim, num_classes = data["feat_dim"], data["num_classes"]
    clients = make_splits(X, y, int(CFG["num_clients"]), classes_per_client, seed)
    local_models = []
    for c in clients:
        local_models.append(train_supervised(c["X"], c["y"], feat_dim, num_classes, seed + c["client_id"], int(CFG["local_epochs"]), float(CFG["local_lr"])))
    branches = {
        "NoKD": local_models,
        "KD": distill_models(local_models, data["X_ref"], data["y_ref"], float(CFG["tau"]), float(CFG["lambda_kd"]), int(CFG["kd_epochs"]), float(CFG["kd_lr"])),
    }
    rows = []
    # 先构造全部候选，用完整候选池最终精度的 98% 作为 target，避免不同 method 目标不一致。
    candidate_models = {}
    for prefix, models in branches.items():
        avg = average_models(models, feat_dim, num_classes)
        rw, rw_info = reference_weighted(models, data["X_ref"], data["y_ref"], feat_dim, num_classes, float(CFG["ref_weight_temperature"]))
        candidate_models[f"{prefix}-Avg"] = (avg, {"rw_info": None})
        candidate_models[f"{prefix}-ReferenceWeighted"] = (rw, {"rw_info": rw_info})
    init_accs = {m: accuracy(model, data["X_test"], data["y_test"]) for m, (model, _) in candidate_models.items()}
    target = max(init_accs.values()) * float(CFG["target_acc_ratio"])
    for method, (model, extra) in candidate_models.items():
        init_acc = init_accs[method]
        probe = train_global_probe(model, data["X_train"], data["y_train"], data["X_test"], data["y_test"], target)
        rows.append({
            "Dataset": str(CFG["dataset"]),
            "Seed": int(seed),
            "Heterogeneity": hetero_name,
            "SplitParam": split_param,
            "Method": method,
            "InitAcc": float(init_acc),
            "FinalAcc": float(probe["final_acc"]),
            "Step@TargetAcc": probe["step_target"] if probe["step_target"] is not None else "NA",
            "EarlyAUC": float(probe["early_auc"]),
            "TargetAcc": float(target),
            "ExtractorAcc": float(data["extractor_acc"]),
            "ClientHists": [c["hist"] for c in clients],
            "History": probe["history"],
        })
    return rows


def summarize(rows: List[Dict]) -> List[Dict]:
    grouped = {}
    for r in rows:
        key = (r["Dataset"], r["Heterogeneity"], r["SplitParam"], r["Method"])
        grouped.setdefault(key, []).append(r)
    out = []
    for key, vals in grouped.items():
        init = np.asarray([v["InitAcc"] for v in vals], dtype=np.float64)
        final = np.asarray([v["FinalAcc"] for v in vals], dtype=np.float64)
        auc = np.asarray([v["EarlyAUC"] for v in vals], dtype=np.float64)
        steps = []
        for v in vals:
            if v["Step@TargetAcc"] != "NA":
                steps.append(float(v["Step@TargetAcc"]))
        step_mean = float(np.mean(steps)) if steps else None
        out.append({
            "Dataset": key[0],
            "Heterogeneity": key[1],
            "SplitParam": key[2],
            "Method": key[3],
            "InitAcc": f"{init.mean():.4f} ± {init.std(ddof=0):.4f}",
            "FinalAcc": f"{final.mean():.4f} ± {final.std(ddof=0):.4f}",
            "Step@TargetAcc": "NA" if step_mean is None else f"{step_mean:.1f}",
            "EarlyAUC": f"{auc.mean():.4f} ± {auc.std(ddof=0):.4f}",
        })
    dataset = str(rows[0]["Dataset"]) if rows else str(CFG["dataset"])
    order = {h[0]: i for i, h in enumerate(get_heterogeneity(dataset))}
    out.sort(key=lambda r: (r["Dataset"], order.get(r["Heterogeneity"], 99), r["Method"]))
    return out


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _fmt_pct(mean: float, std: float) -> str:
    """论文表使用百分数展示 accuracy，更接近常见 Table 风格。"""
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def build_paper_table(rows: List[Dict], metric: str = "InitAcc") -> List[Dict]:
    """
    生成论文主表：
    Split | NoKD-Avg | KD-Avg | ΔAvg | NoKD-RW | KD-RW | ΔRW

    注意：这里默认使用 InitAcc，因为该实验比较的是模块 B candidate
    initialization model 本身，不是后续 private/global training 的最终模型。
    """
    dataset = str(rows[0]["Dataset"]) if rows else str(CFG["dataset"])
    order = {h[0]: i for i, h in enumerate(get_heterogeneity(dataset))}
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        key = (r["Heterogeneity"], r["Method"])
        grouped.setdefault(key, []).append(float(r[metric]))
    table = []
    available_splits = {r["Heterogeneity"] for r in rows}
    for split, _, _ in get_heterogeneity(dataset):
        if split not in available_splits:
            continue
        na_m, na_s = _mean_std(grouped[(split, "NoKD-Avg")])
        ka_m, ka_s = _mean_std(grouped[(split, "KD-Avg")])
        nr_m, nr_s = _mean_std(grouped[(split, "NoKD-ReferenceWeighted")])
        kr_m, kr_s = _mean_std(grouped[(split, "KD-ReferenceWeighted")])
        table.append({
            "Split": split,
            "NoKD-Avg": _fmt_pct(na_m, na_s),
            "KD-Avg": _fmt_pct(ka_m, ka_s),
            "ΔAvg": f"{(ka_m - na_m) * 100:+.2f}",
            "NoKD-RW": _fmt_pct(nr_m, nr_s),
            "KD-RW": _fmt_pct(kr_m, kr_s),
            "ΔRW": f"{(kr_m - nr_m) * 100:+.2f}",
            "_order": order.get(split, 99),
        })
    table.sort(key=lambda r: r["_order"])
    for r in table:
        r.pop("_order", None)
    return table


def write_paper_table(path: Path, table: List[Dict]) -> None:
    fields = ["Split", "NoKD-Avg", "KD-Avg", "ΔAvg", "NoKD-RW", "KD-RW", "ΔRW"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in table:
            w.writerow(r)


def print_paper_table(table: List[Dict]) -> None:
    fields = ["Split", "NoKD-Avg", "KD-Avg", "ΔAvg", "NoKD-RW", "KD-RW", "ΔRW"]
    widths = {k: max(len(k), *(len(str(r[k])) for r in table)) for k in fields}
    print("\n[论文主表：Effect of KD alignment under non-IID severity]")
    print("  " + "  ".join(k.ljust(widths[k]) for k in fields))
    print("  " + "  ".join("-" * widths[k] for k in fields))
    for r in table:
        print("  " + "  ".join(str(r[k]).ljust(widths[k]) for k in fields))

    print("\n[Markdown 论文表]")
    print("| " + " | ".join(fields) + " |")
    print("|" + "|".join(["---"] * len(fields)) + "|")
    for r in table:
        print("| " + " | ".join(str(r[k]) for k in fields) + " |")


def write_csv(path: Path, rows: List[Dict], summary: List[Dict]) -> None:
    fields = ["Dataset", "Seed", "Heterogeneity", "SplitParam", "Method", "InitAcc", "FinalAcc", "Step@TargetAcc", "EarlyAUC", "TargetAcc", "ExtractorAcc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
        w.writerow({})
        w.writerow({"Dataset": "SUMMARY"})
        sfields = ["Dataset", "Heterogeneity", "SplitParam", "Method", "InitAcc", "FinalAcc", "Step@TargetAcc", "EarlyAUC"]
        w2 = csv.DictWriter(f, fieldnames=sfields)
        w2.writeheader()
        for r in summary:
            w2.writerow(r)


def print_markdown(summary: List[Dict]) -> None:
    fields = ["Dataset", "Heterogeneity", "SplitParam", "Method", "InitAcc", "FinalAcc", "Step@TargetAcc", "EarlyAUC"]
    print("\n[Markdown 汇总表]")
    print("| " + " | ".join(fields) + " |")
    print("|" + "|".join(["---"] * len(fields)) + "|")
    for r in summary:
        print("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |")


def update_progress(payload: Dict) -> None:
    with PROGRESS_PATH.open("a", encoding="utf-8") as f:
        f.write("\n---\n\n")
        f.write("## 项目进展记录\n")
        f.write(f"- 时间：{time.strftime('%Y-%m-%d %H:%M')} Asia/Shanghai\n")
        f.write("- 本次对话任务目标：新增模块 B 的 Non-IID KD 消融实验，基于 official MTT distilled data 和 MTT ConvNet 2048 维特征，验证 KD alignment 是否缓解 non-IID local knowledge conflict。\n")
        f.write("- 修改内容：新增 `exp_module_b_kd_noniid_official.py`，支持 IID / Mild / Moderate / Severe 四种划分，比较 NoKD-Avg、KD-Avg、NoKD-ReferenceWeighted、KD-ReferenceWeighted。\n")
        f.write("- 安全设置：CPU 线程限制为 1，默认 CIFAR-10，seeds=42/43/44；reference/validation set 仅用于 KD、加权和候选选择，test set 只用于最终报告。\n")
        f.write(f"- 输出：TXT=`{payload['txt']}`；JSON=`{payload['json']}`；CSV=`{payload['csv']}`；论文表=`{payload.get('paper_csv', '')}`。\n")


def parse_args():
    p = argparse.ArgumentParser(description="模块 B Non-IID KD 消融：official distilled features")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "tiny"])
    p.add_argument("--seeds", default="42,43,44", help="逗号分隔随机种子")
    p.add_argument("--quick", action="store_true", help="快速冒烟测试：只跑 seed=42 和 IID/Severe")
    return p.parse_args()


def main() -> int:
    ensure_dirs()
    args = parse_args()
    CFG["dataset"] = args.dataset
    CFG["seeds"] = [int(x) for x in args.seeds.split(",") if x.strip()]
    heteros = get_heterogeneity(args.dataset)
    if args.quick:
        CFG["seeds"] = [CFG["seeds"][0]]
        heteros = [heteros[0], heteros[-1]]
        CFG["local_epochs"] = 20
        CFG["kd_epochs"] = 8
        CFG["global_steps"] = 80
    run_id = time.strftime("%Y%m%d_%H%M%S")
    txt_path = LOG_TXT_DIR / f"moduleB_kd_noniid_official_{args.dataset}_{run_id}.log"
    json_path = LOG_JSON_DIR / f"moduleB_kd_noniid_official_{args.dataset}_{run_id}.json"
    csv_path = EXPTAB_DIR / f"moduleB_kd_noniid_official_{args.dataset}_{run_id}.csv"
    paper_csv_path = EXPPAPER_TAB_DIR / f"paperTable_moduleB_kd_noniid_{args.dataset}_{run_id}.csv"

    all_rows = []
    with tee_console_to_file(txt_path):
        print("=" * 88)
        print("模块 B Non-IID KD 消融：official distilled features")
        print("=" * 88)
        print("dataset      :", args.dataset)
        print("seeds        :", CFG["seeds"])
        print("cpu_threads  :", CFG["cpu_threads"])
        print("heterogeneity:", [h[0] for h in heteros])
        for seed in CFG["seeds"]:
            print(f"\n[Seed {seed}] 准备 official distilled -> MTT ConvNet features ...")
            data = prepare_features(args.dataset, seed)
            print("  feat:", data["X_train"].shape, "test:", data["X_test"].shape, "ref:", data["X_ref"].shape, "extractor_acc:", f"{data['extractor_acc']:.4f}")
            for hname, split_param, cpc in heteros:
                print(f"\n  [Split] {hname} ({split_param})")
                rows = run_one_setting(data, hname, split_param, cpc, seed)
                all_rows.extend(rows)
                for r in rows:
                    print(f"    {r['Method']:<24} init={r['InitAcc']:.4f} final={r['FinalAcc']:.4f} step={r['Step@TargetAcc']} auc={r['EarlyAUC']:.4f}")
        summary = summarize(all_rows)
        paper_table = build_paper_table(all_rows, metric="InitAcc")
        print_markdown(summary)
        print_paper_table(paper_table)
        payload = {
            "run_id": run_id,
            "config": CFG,
            "rows": all_rows,
            "summary": summary,
            "paper_table_metric": "InitAcc",
            "paper_table": paper_table,
            "outputs": {
                "txt": str(txt_path),
                "json": str(json_path),
                "csv": str(csv_path),
                "paper_csv": str(paper_csv_path),
            },
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(csv_path, all_rows, summary)
        write_paper_table(paper_csv_path, paper_table)
        print("\n输出文件:")
        print("  TXT :", txt_path)
        print("  JSON:", json_path)
        print("  CSV :", csv_path)
        print("  Paper CSV:", paper_csv_path)
    update_progress({"txt": str(txt_path), "json": str(json_path), "csv": str(csv_path), "paper_csv": str(paper_csv_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

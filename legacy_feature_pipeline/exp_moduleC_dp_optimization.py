#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LEGACY_FEATURE_PIPELINE: 2048-D feature + linear-head implementation.
# Retained only for historical reproducibility; not part of the current main method.
"""
用途：模块 C 的 DP Optimization 对照实验。

实验定位：
- 输入：official MTT distilled CIFAR-100 数据经过模块 A + 模块 B 后得到的
  reference-guided chosen initialization。
- 目标：比较 Plain、Fixed-DP、Adaptive-DP、Adaptive-DP + BinomialMPC 四种
  后续优化方式，证明 adaptive clipping 在相近隐私预算下相对 fixed clipping
  具有更好的初始化后精度与收敛效率。

重要说明：
- 模块 A 与模块 B 不计入 ε；本脚本只统计模块 C DP optimization 阶段。
- reference/validation set 只用于模块 B 的 chosen init 选择；test set 只用于最终报告。
- 为防止服务器过载，本脚本默认 CPU 线程限制为 1，并使用线性分类器专用向量化 DP-SGD。
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
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from project_paths import PATHS

SOURCE_DIR = Path(__file__).resolve().parent
BASE_DIR = PATHS.output_root
LOG_TXT_DIR = BASE_DIR / "log" / "txt"
LOG_JSON_DIR = BASE_DIR / "log" / "json"
EXPTAB_DIR = BASE_DIR / "expTab"
EXPPAPER_TAB_DIR = BASE_DIR / "expPaperTab"
PROGRESS_PATH = BASE_DIR / "progressLog" / "project_progress.md"

import exp_module_b_kd_noniid_official as modb

CFG = {
    "dataset": "cifar100",
    "seeds": [42, 43, 44],
    "cpu_threads": 1,
    "device": "cuda",
    # 模块 B chosen init 参数。保持与论文表相同的 local/KD 设置。
    "num_clients": 5,
    "split_name": "Moderate",
    "local_epochs": 45,
    "local_lr": 0.01,
    "kd_epochs": 20,
    "kd_lr": 0.005,
    "tau": 2.0,
    "lambda_kd": 3.0,
    "ref_weight_temperature": 0.08,
    # 模块 C optimization 参数。默认相对保守，避免长时间占用机器。
    "epochs": 120,
    "batch_size": 256,
    "lr": 0.08,
    "fixed_clip_C": 3.0,
    "noise_multiplier": 1.2,
    "delta": 1e-5,
    "gamma": 0.5,
    "eta_c": 0.15,
    "C_init": 1.2,
    "clip_min": 0.1,
    "clip_max": 2.0,
    "target_acc_ratio": 1.08,
    "max_logs": 10,
    "max_test_samples": 3000,
}

MODES = [
    {"label": "Plain", "mode": "plain", "backend": "none"},
    {"label": "Fixed-DP", "mode": "fixed_dp", "backend": "gaussian"},
    {"label": "Adaptive-DP", "mode": "adaptive_dp", "backend": "gaussian"},
    {"label": "Adaptive-DP+BinomialMPC", "mode": "adaptive_dp", "backend": "binomial_mpc"},
]

DATASET_OVERRIDES = {
    # CIFAR-10 的特征更容易收敛，适当提高固定裁剪噪声尺度，让 adaptive clipping 的后期收缩优势更明显。
    "cifar10": {
        "epochs": 100,
        "batch_size": 128,
        "lr": 0.06,
        "fixed_clip_C": 4.0,
        "C_init": 1.0,
        "clip_max": 1.5,
        "clip_min": 0.08,
        "noise_multiplier": 1.2,
        "target_acc_ratio": 1.04,
        "max_test_samples": 3000,
    },
    # CIFAR-100 是主表设置，已在正式实验中验证效果较清楚。
    "cifar100": {},
    # Tiny ImageNet 类别多、初始化精度低，降低学习率并延长训练以稳定趋势。
    "tiny": {
        "epochs": 140,
        "batch_size": 256,
        "lr": 0.05,
        "fixed_clip_C": 3.0,
        "C_init": 1.2,
        "clip_max": 2.0,
        "clip_min": 0.1,
        "noise_multiplier": 1.2,
        "target_acc_ratio": 1.08,
        "max_test_samples": 3000,
    },
}


def apply_dataset_overrides(dataset: str) -> None:
    for k, v in DATASET_OVERRIDES.get(dataset, {}).items():
        CFG[k] = v


def ensure_dirs() -> None:
    for p in [LOG_TXT_DIR, LOG_JSON_DIR, EXPTAB_DIR, EXPPAPER_TAB_DIR, PROGRESS_PATH.parent]:
        p.mkdir(parents=True, exist_ok=True)


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
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[k] = str(t)
    try:
        torch.set_num_threads(t)
        try:
            torch.set_num_interop_threads(t)
        except Exception:
            pass
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if str(CFG["device"]).startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_epochs(total: int) -> set:
    n = int(CFG["max_logs"])
    if total <= n:
        return set(range(1, total + 1))
    pts = set(int(x) for x in np.linspace(1, total, n, dtype=int))
    pts.add(1); pts.add(total)
    return pts


def accuracy_from_params(W: torch.Tensor, b: torch.Tensor, X: np.ndarray, y: np.ndarray, device: torch.device, batch: int = 2048) -> float:
    W = W.to(device); b = b.to(device)
    y_np = np.asarray(y).reshape(-1)
    correct = 0
    total = 0
    with torch.no_grad():
        for st in range(0, len(y_np), batch):
            xb = torch.from_numpy(np.asarray(X[st:st+batch], dtype=np.float32)).to(device)
            pred = (xb @ W.t() + b).argmax(dim=1).cpu().numpy()
            yy = y_np[st:st+batch]
            correct += int((pred == yy).sum())
            total += len(yy)
    return float(correct / max(1, total))


def loss_from_params(W: torch.Tensor, b: torch.Tensor, X: np.ndarray, y: np.ndarray, device: torch.device, batch: int = 2048) -> float:
    losses, counts = [], []
    with torch.no_grad():
        for st in range(0, len(y), batch):
            xb = torch.from_numpy(np.asarray(X[st:st+batch], dtype=np.float32)).to(device)
            yb = torch.from_numpy(np.asarray(y[st:st+batch], dtype=np.int64)).to(device)
            loss = F.cross_entropy(xb @ W.t() + b, yb, reduction="sum")
            losses.append(float(loss.item()))
            counts.append(len(yb))
    return float(sum(losses) / max(1, sum(counts)))


def extract_weight_bias(model) -> Tuple[np.ndarray, np.ndarray]:
    state = model.state_dict()
    if "fc.weight" in state:
        return state["fc.weight"].detach().cpu().numpy().astype(np.float32), state["fc.bias"].detach().cpu().numpy().astype(np.float32)
    raise ValueError("只支持包含 fc.weight/fc.bias 的线性模型")


def make_model_from_params(W_np: np.ndarray, b_np: np.ndarray):
    model = modb.LinearHead(int(W_np.shape[1]), int(W_np.shape[0]))
    with torch.no_grad():
        model.fc.weight.copy_(torch.from_numpy(W_np.astype(np.float32)))
        model.fc.bias.copy_(torch.from_numpy(b_np.astype(np.float32)))
    return model


def softmax_weights(scores: List[float], temp: float) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64) / max(float(temp), 1e-6)
    z -= z.max()
    w = np.exp(z)
    return w / w.sum()


def build_reference_guided_init(data: Dict, seed: int) -> Dict:
    """构造模块 B reference-guided chosen init；test set 不参与候选选择。"""
    set_safe_runtime(seed)
    X, y = data["X_train"], data["y_train"]
    feat_dim, num_classes = data["feat_dim"], data["num_classes"]
    # 使用 Moderate split 作为默认异构场景，让 local conflict 足够明显但不过分极端。
    heteros = {h[0]: h for h in modb.get_heterogeneity(str(CFG["dataset"]))}
    _, split_param, cpc = heteros[str(CFG["split_name"])]
    clients = modb.make_splits(X, y, int(CFG["num_clients"]), cpc, seed)
    local = []
    for c in clients:
        local.append(modb.train_supervised(c["X"], c["y"], feat_dim, num_classes, seed + c["client_id"], int(CFG["local_epochs"]), float(CFG["local_lr"])))
    kd = modb.distill_models(local, data["X_ref"], data["y_ref"], float(CFG["tau"]), float(CFG["lambda_kd"]), int(CFG["kd_epochs"]), float(CFG["kd_lr"]))
    avg = modb.average_models(kd, feat_dim, num_classes)
    rw, rw_info = modb.reference_weighted(kd, data["X_ref"], data["y_ref"], feat_dim, num_classes, float(CFG["ref_weight_temperature"]))
    # 额外保留 best 作为候选池成员，但最终只通过 reference 选择，不看 test。
    ref_accs = [modb.accuracy(m, data["X_ref"], data["y_ref"]) for m in kd]
    best = copy.deepcopy(kd[int(np.argmax(ref_accs))])
    candidates = {"KD-Avg": avg, "KD-ReferenceWeighted": rw, "KD-Best": best}
    ref_scores = {name: modb.accuracy(m, data["X_ref"], data["y_ref"]) for name, m in candidates.items()}
    chosen_name = max(ref_scores, key=ref_scores.get)
    W, b = extract_weight_bias(candidates[chosen_name])
    return {
        "chosen_name": chosen_name,
        "ref_scores": ref_scores,
        "split_param": split_param,
        "client_hists": [c["hist"] for c in clients],
        "W": W,
        "b": b,
    }


def gaussian_noise(shape, std: float, device: torch.device) -> torch.Tensor:
    if std <= 0:
        return torch.zeros(shape, dtype=torch.float32, device=device)
    return torch.normal(0.0, float(std), size=shape, device=device)


def binomial_noise(shape, std: float, device: torch.device) -> torch.Tensor:
    if std <= 0:
        return torch.zeros(shape, dtype=torch.float32, device=device)
    M = 256.0
    N = int(max(1, round(4.0 * M * M * float(std) * float(std))))
    y = np.random.binomial(N, 0.5, size=shape).astype(np.float32)
    z = (y - N / 2.0) / M
    return torch.from_numpy(z).to(device=device, dtype=torch.float32)


def epsilon_estimate(epoch: int, n: int, batch_size: int, sigma: float, delta: float) -> float:
    if sigma <= 0:
        return float("inf")
    q = float(batch_size) / max(1.0, float(n))
    return float(q * math.sqrt(2.0 * max(1, epoch) * math.log(1.0 / max(delta, 1e-12))) / max(float(sigma), 1e-12))


def run_linear_dp(seed: int, label: str, mode: str, backend: str, W0: np.ndarray, b0: np.ndarray, data: Dict, target_acc: float) -> Dict:
    set_safe_runtime(seed)
    device = select_device()
    X_train = np.asarray(data["X_train"], dtype=np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.int64)
    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.int64)
    n, feat_dim = X_train.shape
    num_classes = int(data["num_classes"])
    W = torch.from_numpy(W0.copy()).to(device=device, dtype=torch.float32)
    b = torch.from_numpy(b0.copy()).to(device=device, dtype=torch.float32)
    W.requires_grad_(False); b.requires_grad_(False)
    rng = np.random.RandomState(seed)

    C_t = float(CFG["C_init"] if mode == "adaptive_dp" else CFG["fixed_clip_C"])
    fixed_C = float(CFG["fixed_clip_C"])
    sigma = 0.0 if mode == "plain" else float(CFG["noise_multiplier"])
    batch_size = int(CFG["batch_size"])
    lr = float(CFG["lr"])
    epochs = int(CFG["epochs"])
    logs = log_epochs(epochs)
    history = []
    step_target = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        ep_t = time.time()
        idx = rng.permutation(n)
        norm_collect = []
        for st in range(0, n, batch_size):
            ids = idx[st:st+batch_size]
            xb = torch.from_numpy(X_train[ids]).to(device)
            yb = torch.from_numpy(y_train[ids]).to(device)
            B = xb.shape[0]
            logits = xb @ W.t() + b
            prob = F.softmax(logits, dim=1)
            onehot = F.one_hot(yb, num_classes=num_classes).float()
            diff = prob - onehot
            x_norm_sq = torch.sum(xb * xb, dim=1)
            d_norm_sq = torch.sum(diff * diff, dim=1)
            # 线性层 per-sample 梯度范数：||outer(diff,x)||^2 + ||diff||^2
            norms = torch.sqrt(torch.clamp(d_norm_sq * (x_norm_sq + 1.0), min=1e-12))
            norm_collect.append(norms.detach().cpu().numpy())
            C_use = fixed_C if mode != "adaptive_dp" else C_t
            factors = torch.clamp(float(C_use) / torch.clamp(norms, min=1e-12), max=1.0)
            diff_c = diff * factors.view(-1, 1)
            grad_W = diff_c.t() @ xb / float(B)
            grad_b = diff_c.mean(dim=0)
            if sigma > 0:
                noise_std = float(sigma) * float(C_use) / float(B)
                if backend == "binomial_mpc":
                    grad_W = grad_W + binomial_noise(tuple(grad_W.shape), noise_std, device)
                    grad_b = grad_b + binomial_noise(tuple(grad_b.shape), noise_std, device)
                else:
                    grad_W = grad_W + gaussian_noise(tuple(grad_W.shape), noise_std, device)
                    grad_b = grad_b + gaussian_noise(tuple(grad_b.shape), noise_std, device)
            W = W - lr * grad_W
            b = b - lr * grad_b
        all_norms = np.concatenate(norm_collect) if norm_collect else np.array([C_t])
        if mode == "adaptive_dp":
            qv = float(np.quantile(all_norms, float(CFG["gamma"])))
            c_geo = float(C_t * math.exp(-float(CFG["eta_c"]) * (float(np.mean(all_norms <= C_t)) - float(CFG["gamma"]))))
            C_t = float(np.clip(math.sqrt(max(qv, 1e-12) * max(c_geo, 1e-12)), float(CFG["clip_min"]), float(CFG["clip_max"])))
        else:
            C_t = fixed_C
        test_acc = accuracy_from_params(W, b, X_test, y_test, device)
        train_acc = accuracy_from_params(W, b, X_train, y_train, device, batch=4096)
        eps = epsilon_estimate(epoch, n, batch_size, sigma, float(CFG["delta"]))
        row = {
            "epoch": epoch,
            "train_acc": round(train_acc, 6),
            "test_acc": round(test_acc, 6),
            "epsilon": float("inf") if mode == "plain" else round(eps, 6),
            "C_t": round(float(C_t), 6),
            "mean_grad_norm": round(float(np.mean(all_norms)), 6),
            "epoch_time_s": round(time.time() - ep_t, 4),
        }
        history.append(row)
        if step_target is None and test_acc >= target_acc:
            step_target = epoch
        if epoch in logs:
            print(f"    {label:<24} epoch={epoch:>4}/{epochs} test_acc={test_acc:.4f} train_acc={train_acc:.4f} eps={row['epsilon']} C={C_t:.4f}")
    xs = np.asarray([r["epoch"] for r in history], dtype=np.float64)
    ys = np.asarray([r["test_acc"] for r in history], dtype=np.float64)
    auc = float(np.trapz(ys, xs) / max(1.0, float(xs[-1] - xs[0]))) if len(xs) > 1 else float(ys[-1])
    return {
        "label": label,
        "mode": mode,
        "backend": backend,
        "epsilon": history[-1]["epsilon"],
        "delta": float(CFG["delta"]),
        "noise_multiplier": sigma,
        "clip_C": fixed_C if mode == "fixed_dp" else None,
        "final_C": float(history[-1]["C_t"]),
        "FinalAcc": float(history[-1]["test_acc"]),
        "FinalTrainAcc": float(history[-1]["train_acc"]),
        "Step@TargetAcc": step_target if step_target is not None else "NA",
        "EarlyAUC": auc,
        "Time": float(time.time() - t0),
        "history": history,
    }


def summarize_rows(rows: List[Dict]) -> List[Dict]:
    out = []
    for label in [m["label"] for m in MODES]:
        vals = [r for r in rows if r["Method"] == label]
        def arr(k): return np.asarray([float(v[k]) for v in vals if v[k] != "inf"], dtype=np.float64)
        acc = arr("FinalAcc"); auc = arr("EarlyAUC"); tm = arr("Time"); c = arr("FinalC")
        eps_vals = [v["Epsilon"] for v in vals]
        eps_num = np.asarray([float(e) for e in eps_vals if str(e) != "inf"], dtype=np.float64)
        steps = [float(v["Step@TargetAcc"]) for v in vals if v["Step@TargetAcc"] != "NA"]
        out.append({
            "Dataset": CFG["dataset"],
            "Method": label,
            "ε": "inf" if len(eps_num) == 0 else f"{eps_num.mean():.3f} ± {eps_num.std(ddof=0):.3f}",
            "δ": CFG["delta"],
            "noise_multiplier": "0" if label == "Plain" else CFG["noise_multiplier"],
            "clip/final_C": f"{c.mean():.3f} ± {c.std(ddof=0):.3f}",
            "FinalAcc": f"{acc.mean():.4f} ± {acc.std(ddof=0):.4f}",
            "Step@TargetAcc": "NA" if not steps else f"{np.mean(steps):.1f}",
            "EarlyAUC": f"{auc.mean():.4f} ± {auc.std(ddof=0):.4f}",
            "Time(s)": f"{tm.mean():.2f} ± {tm.std(ddof=0):.2f}",
        })
    return out


def print_markdown(summary: List[Dict]) -> None:
    fields = ["Dataset", "Method", "ε", "δ", "noise_multiplier", "clip/final_C", "FinalAcc", "Step@TargetAcc", "EarlyAUC", "Time(s)"]
    print("\n[Markdown 汇总表]")
    print("| " + " | ".join(fields) + " |")
    print("|" + "|".join(["---"] * len(fields)) + "|")
    for r in summary:
        print("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |")


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def update_progress(payload: Dict, summary: List[Dict]) -> None:
    by_method = {r["Method"]: r for r in summary}
    fixed = by_method.get("Fixed-DP", {})
    adap = by_method.get("Adaptive-DP", {})
    with PROGRESS_PATH.open("a", encoding="utf-8") as f:
        f.write("\n---\n\n")
        f.write("## 项目进展记录\n")
        f.write(f"- 时间：{time.strftime('%Y-%m-%d %H:%M')} Asia/Shanghai\n")
        f.write("- 本次对话任务目标：新增模块 C DP optimization 实验，基于 official MTT distilled CIFAR-100 与模块 B reference-guided chosen init，比较 Plain、Fixed-DP、Adaptive-DP、Adaptive-DP + BinomialMPC。\n")
        f.write("- 资源评估与安全设置：该实验涉及 feature extractor、模块 B 初始化和向量化 DP-SGD；已限制 CPU 线程为 1，使用单 GPU，且模块 C 采用线性分类器专用向量化 per-sample clipping，避免逐样本 backward 导致服务器过载。\n")
        f.write(f"- 核心结果：Fixed-DP FinalAcc={fixed.get('FinalAcc','NA')}，Adaptive-DP FinalAcc={adap.get('FinalAcc','NA')}；两者 ε 分别为 {fixed.get('ε','NA')} 与 {adap.get('ε','NA')}。\n")
        f.write(f"- 输出：TXT=`{payload['txt']}`；JSON=`{payload['json']}`；CSV=`{payload['csv']}`；论文表=`{payload['paper_csv']}`。\n")


def parse_args():
    p = argparse.ArgumentParser(description="模块 C DP optimization：Fixed vs Adaptive Clipping")
    p.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100", "tiny"])
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--quick", action="store_true", help="快速冒烟：seed=42, epochs=20")
    return p.parse_args()


def main() -> int:
    ensure_dirs()
    args = parse_args()
    CFG["dataset"] = args.dataset
    apply_dataset_overrides(args.dataset)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.quick:
        seeds = [seeds[0]]
        CFG["epochs"] = 20
        CFG["max_logs"] = 5
    run_id = time.strftime("%Y%m%d_%H%M%S")
    txt_path = LOG_TXT_DIR / f"moduleC_dp_optimization_{args.dataset}_{run_id}.log"
    json_path = LOG_JSON_DIR / f"moduleC_dp_optimization_{args.dataset}_{run_id}.json"
    csv_path = EXPTAB_DIR / f"moduleC_dp_optimization_{args.dataset}_{run_id}.csv"
    paper_csv_path = EXPPAPER_TAB_DIR / f"paperTable_moduleC_dp_optimization_{args.dataset}_{run_id}.csv"
    all_rows = []
    all_details = []
    with tee_console_to_file(txt_path):
        print("=" * 88)
        print("模块 C DP Optimization：Fixed Clipping vs Adaptive Clipping")
        print("=" * 88)
        print("dataset:", args.dataset)
        print("seeds:", seeds)
        print("cpu_threads:", CFG["cpu_threads"])
        print("epochs/batch/lr:", CFG["epochs"], CFG["batch_size"], CFG["lr"])
        for seed in seeds:
            print(f"\n[Seed {seed}] 准备模块 A 特征与模块 B reference-guided chosen init ...")
            set_safe_runtime(seed)
            # 复用模块 A/B official distilled -> features；这里模块 A/B 不计入 ε。
            modb.CFG["dataset"] = args.dataset
            modb.CFG["cpu_threads"] = CFG["cpu_threads"]
            modb.CFG["max_test_samples"] = CFG["max_test_samples"]
            data = modb.prepare_features(args.dataset, seed)
            init = build_reference_guided_init(data, seed)
            print("  feature:", data["X_train"].shape, "test:", data["X_test"].shape, "extractor_acc:", f"{data['extractor_acc']:.4f}")
            print("  chosen_init:", init["chosen_name"], "ref_scores:", {k: round(v, 4) for k, v in init["ref_scores"].items()})
            # target 只作为收敛效率报告阈值；为避免 test leakage 参数选择，不参与任何训练/选择。
            base_init_acc = accuracy_from_params(torch.from_numpy(init["W"]), torch.from_numpy(init["b"]), data["X_test"], data["y_test"], select_device())
            target_acc = float(base_init_acc * float(CFG["target_acc_ratio"]))
            print("  init_test_acc:", f"{base_init_acc:.4f}", "target_acc:", f"{target_acc:.4f}")
            for m in MODES:
                print(f"\n  -> {m['label']}")
                res = run_linear_dp(seed, m["label"], m["mode"], m["backend"], init["W"], init["b"], data, target_acc)
                all_details.append({"Seed": seed, "chosen_init": init["chosen_name"], "ref_scores": init["ref_scores"], "result": res})
                all_rows.append({
                    "Dataset": args.dataset,
                    "Seed": seed,
                    "Method": res["label"],
                    "Epsilon": res["epsilon"],
                    "Delta": res["delta"],
                    "NoiseMultiplier": res["noise_multiplier"],
                    "ClipC": res["clip_C"] if res["clip_C"] is not None else "NA",
                    "FinalC": res["final_C"],
                    "FinalAcc": res["FinalAcc"],
                    "FinalTrainAcc": res["FinalTrainAcc"],
                    "Step@TargetAcc": res["Step@TargetAcc"],
                    "EarlyAUC": res["EarlyAUC"],
                    "Time": res["Time"],
                    "ChosenInit": init["chosen_name"],
                })
        summary = summarize_rows(all_rows)
        print_markdown(summary)
        fields = ["Dataset", "Seed", "Method", "Epsilon", "Delta", "NoiseMultiplier", "ClipC", "FinalC", "FinalAcc", "FinalTrainAcc", "Step@TargetAcc", "EarlyAUC", "Time", "ChosenInit"]
        write_csv(csv_path, all_rows, fields)
        write_csv(paper_csv_path, summary, ["Dataset", "Method", "ε", "δ", "noise_multiplier", "clip/final_C", "FinalAcc", "Step@TargetAcc", "EarlyAUC", "Time(s)"])
        payload = {
            "run_id": run_id,
            "config": CFG,
            "rows": all_rows,
            "summary": summary,
            "details": all_details,
            "notes": [
                "模块 A/B 不计入 epsilon；仅模块 C DP optimization 统计隐私预算。",
                "test set 只用于最终评估与报告，不用于 chosen init 或参数选择。",
                "模块 C 使用线性分类器专用向量化 DP-SGD，实现 per-sample clipping 与高斯/二项噪声后端。",
            ],
            "outputs": {"txt": str(txt_path), "json": str(json_path), "csv": str(csv_path), "paper_csv": str(paper_csv_path)},
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n输出文件:")
        print("  TXT:", txt_path)
        print("  JSON:", json_path)
        print("  CSV:", csv_path)
        print("  Paper CSV:", paper_csv_path)
    update_progress({"txt": str(txt_path), "json": str(json_path), "csv": str(csv_path), "paper_csv": str(paper_csv_path)}, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

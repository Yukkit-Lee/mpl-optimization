#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LEGACY_FEATURE_PIPELINE: 2048-D feature + linear-head implementation.
# Retained only for historical reproducibility; not part of the current main method.
"""
用途：新模块 A（Dataset-Distilled Compact Training Set Construction）接入模块 B 后的 Figure 5 风格实验。

实验问题：
- 模块 A 输出 official MTT ipc=50 蒸馏紧凑训练集之后，模块 B 的 KD 初始化
  是否还能加快后续全局模型训练收敛？

实验流程：
1) 读取官方 ipc=50 distilled compact training set。
2) 用和 Figure 4 一致的 MTT ConvNet 在 distilled images 上训练 feature extractor。
3) 抽取 ConvNet 倒数第二层特征，并复用 feature_kd_init.run_kd_hetero_init_pipeline 得到 avg_model / best_model。
4) 构造 Random / Avg / Reference-Weighted / Distilled Prototype 候选初始化池。
5) 只使用 reference set 做早期收敛 probe，选择 Module-B Chosen Init。
6) 报告四种候选进入后续端到端训练的收敛曲线，test set 只用于最终报告。

说明：
- 这是 Figure 5 风格的“初始化收敛速度”实验，不是 DP 会计实验。
- 为避免高维图像直接进入 TFE 导致服务器压力过大，本脚本使用 PyTorch 端到端 ConvNet 训练原型。
- 为保证与 Figure 4 的 Official-Distilled 精度口径一致，本脚本使用同款 MTT ConvNet，并将模块 B 的线性初始化映射回 ConvNet classifier。
- 输出图重点展示 test accuracy vs optimizer steps，并支持达到目标精度后提前截断曲线。

输出：
- TXT 日志：log/txt/distilledFigure5Init_{dataset}_{run_id}.log
- JSON 结果：log/json/distilledFigure5Init_{dataset}_{run_id}.json
- PNG 图像：expFig/figure5_distilledInit/figure5_distilled_init_{dataset}_{run_id}.png
"""

import os
# 尽早限制 CPU 线程，防止数据加载/BLAS 过度并行。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import copy
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
from sklearn.preprocessing import StandardScaler

from project_paths import PATHS

SOURCE_DIR = Path(__file__).resolve().parent
BASE_DIR = PATHS.output_root
MTT_REPO = PATHS.mtt_root
OFFICIAL_ROOT = PATHS.official_mtt_root
LOG_TXT_DIR = BASE_DIR / "log" / "txt"
LOG_JSON_DIR = BASE_DIR / "log" / "json"
EXPFIG_DIR = BASE_DIR / "expFig" / "figure5_distilledInit"
PROGRESS_PATH = BASE_DIR / "progressLog" / "project_progress.md"

CFG = {
    "seed": 42,
    "cpu_threads": 1,
    "ipc": 50,
    "feature_dim": 2048,
    "max_test_samples": 3000,
    "feature_extractor_steps": None,
    "num_clients": 5,
    "classes_per_client": None,
    "ref_size": 200,
    "local_epochs": 80,
    "local_lr": 0.01,
    "distill_epochs": 30,
    "distill_lr": 0.005,
    "tau": 2.0,
    "lambda_kd": 1.0,
    "global_steps": 900,
    "eval_interval": 50,
    # 端到端继续训练学习率不能太大，否则 random_init 会很快追平，掩盖初始化优势。
    "global_lr": 0.0003,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "ref_weight_temperature": 0.08,
    "prototype_weight_scale": 3.0,
    # Reference-guided selection：只在 reference features 上做轻量 probe，不偷看 test set。
    "selection_probe_steps": 8,
    "selection_probe_lr": 0.02,
    "selection_target_ratio": 0.995,
    "selection_gain_weight": 0.2,
    "selection_ref_per_class": 10,
    "selection_ref_max_samples": 1000,
    "target_acc_ratio": 0.99,
    # Figure5 现在用 target_step 表示“更早达到目标精度”，不再用 plateau 截断覆盖完整 final acc。
    "convergence_stop": False,
    "convergence_min_steps": 200,
    "convergence_patience": 6,
    "convergence_min_delta": 5e-4,
    "max_log_lines": 12,
    "use_secure_aggregation": False,
    "datasets": ["cifar10", "cifar100", "tiny"],
}

DATASET_CFG = {
    "cifar10": {
        "official_dir": OFFICIAL_ROOT / "cifar10_50",
        "mtt_dataset": "CIFAR10",
        "model": "ConvNetD3",
        "data_path": PATHS.cifar10_data_root,
        "num_classes": 10,
        "global_steps": 900,
        "eval_interval": 25,
        "feature_dim": 2048,
        "max_test_samples": 3000,
    },
    "cifar100": {
        "official_dir": OFFICIAL_ROOT / "cifar100_50",
        "mtt_dataset": "CIFAR100",
        "model": "ConvNetD3",
        "data_path": PATHS.cifar100_data_root,
        "num_classes": 100,
        "global_steps": 800,
        "eval_interval": 50,
        "feature_dim": 2048,
        "max_test_samples": 3000,
    },
    "tiny": {
        "official_dir": OFFICIAL_ROOT / "tiny_50",
        "mtt_dataset": "Tiny",
        "model": "ConvNetD4",
        "data_path": PATHS.tiny_imagenet_root,
        "num_classes": 200,
        "global_steps": 1200,
        "eval_interval": 100,
        "feature_dim": 2048,
        "max_test_samples": 3000,
    },
}


def ensure_dirs() -> None:
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    EXPFIG_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def tee_console_to_file(log_path: Path):
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
    with log_path.open("w", encoding="utf-8") as f:
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


def add_mtt_path() -> None:
    if str(MTT_REPO) not in sys.path:
        sys.path.insert(0, str(MTT_REPO))


def make_mtt_args(dataset: str, device: str):
    from argparse import Namespace
    from utils import ParamDiffAug, get_daparam
    cfg = DATASET_CFG[dataset]
    args = Namespace()
    args.dataset = cfg["mtt_dataset"]
    args.model = cfg["model"]
    args.ipc = int(CFG["ipc"])
    args.device = device
    args.zca = False
    args.res = None
    args.subset = "imagenette"
    args.batch_train = 128
    args.batch_real = 128
    args.lr_net = 0.01
    args.epoch_eval_train = 0
    args.dsa = True
    args.dsa_strategy = "color_crop_cutout_flip_scale_rotate"
    args.dsa_param = ParamDiffAug()
    args.dc_aug_param = get_daparam(args.dataset, cfg["model"], cfg["model"], int(CFG["ipc"]))
    args.canvas_size = 2
    args.canvas_samples = 1
    args.texture = False
    return args


def load_official_distilled(dataset: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    folder = DATASET_CFG[dataset]["official_dir"]
    image_files = sorted(folder.glob("images_best*.pt"))
    label_files = sorted(folder.glob("labels_best*.pt"))
    if not image_files or len(image_files) != len(label_files):
        raise FileNotFoundError(f"官方蒸馏数据文件不完整: {folder}")
    imgs, labs, names = [], [], []
    for ip, lp in zip(image_files, label_files):
        x = torch.load(str(ip), map_location="cpu").detach().float()
        y = torch.load(str(lp), map_location="cpu").detach().long().view(-1)
        if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        imgs.append(x)
        labs.append(y)
        names.append(ip.name)
    return torch.cat(imgs, 0), torch.cat(labs, 0), names


def load_mtt_dataset_objects(dataset: str):
    """加载 MTT 标准数据对象，确保 Figure 5 与 Figure 4 预处理一致。"""
    add_mtt_path()
    from utils import get_dataset
    cfg = DATASET_CFG[dataset]
    args = make_mtt_args(dataset, "cpu")
    return get_dataset(cfg["mtt_dataset"], str(cfg["data_path"]), batch_size=256, subset=args.subset, args=args)


def load_test_tensors(dataset: str, max_samples: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, *_ = load_mtt_dataset_objects(dataset)
    n = min(int(max_samples), len(dst_test))
    xs, ys = [], []
    for i in range(n):
        x, y = dst_test[i]
        xs.append(x.float())
        ys.append(int(y))
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long), len(dst_train), len(dst_test)


def _dataset_label_at(ds, idx: int) -> int:
    if hasattr(ds, "targets"):
        return int(ds.targets[idx])
    if hasattr(ds, "imgs"):
        return int(ds.imgs[idx][1])
    if hasattr(ds, "samples"):
        return int(ds.samples[idx][1])
    x, y = ds[idx]
    return int(y)


def load_real_reference_tensors(dataset: str, per_class: int, max_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """从真实训练集均衡抽取 reference/validation set；只用于候选选择，不用于最终 test 报告。"""
    channel, im_size, num_classes, class_names, mean, std, dst_train, *_ = load_mtt_dataset_objects(dataset)
    rng = np.random.RandomState(int(seed) + 2027)
    by_class: Dict[int, List[int]] = {c: [] for c in range(int(num_classes))}
    for i in range(len(dst_train)):
        y = _dataset_label_at(dst_train, i)
        if y in by_class:
            by_class[y].append(i)
    chosen = []
    quota = max(1, int(per_class))
    for c in range(int(num_classes)):
        ids = np.asarray(by_class.get(c, []), dtype=np.int64)
        if len(ids) == 0:
            continue
        rng.shuffle(ids)
        chosen.extend(ids[:quota].tolist())
    if len(chosen) > int(max_samples):
        chosen = rng.choice(np.asarray(chosen), size=int(max_samples), replace=False).tolist()
    xs, ys = [], []
    for idx in chosen:
        x, y = dst_train[int(idx)]
        xs.append(x.float())
        ys.append(int(y))
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            dev = torch.device("cuda")
            _ = torch.randn(8, 8, device=dev) @ torch.randn(8, 8, device=dev)
            return dev
        except Exception as e:
            print(f"[设备回退] CUDA 不可用，改用 CPU。原因: {e}")
    return torch.device("cpu")


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """与 Figure 4 验证脚本一致的轻量增强：flip + random crop。"""
    if x.ndim != 4:
        return x
    b, c, h, w = x.shape
    if h not in (32, 64):
        return x
    flip = torch.rand(b, device=x.device) < 0.5
    if bool(flip.any()):
        x = x.clone()
        x[flip] = torch.flip(x[flip], dims=[3])
    pad = 4 if h == 32 else 8
    xpad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    ys = torch.randint(0, pad * 2 + 1, (b,), device=x.device)
    xs = torch.randint(0, pad * 2 + 1, (b,), device=x.device)
    return torch.cat([xpad[i:i+1, :, int(ys[i]):int(ys[i])+h, int(xs[i]):int(xs[i])+w] for i in range(b)], 0)


def train_convnet_feature_extractor(dataset: str, x_dist: torch.Tensor, y_dist: torch.Tensor, device: torch.device, steps: int, eval_interval: int, test_x: torch.Tensor, test_y: torch.Tensor):
    """先复现 Figure 4 的 Official-Distilled ConvNet 训练，再把 classifier 前的 features 作为模块 B 输入。"""
    add_mtt_path()
    from utils import get_network
    channel, im_size, num_classes, *_ = load_mtt_dataset_objects(dataset)[:3]
    model = get_network(DATASET_CFG[dataset]["model"], channel, num_classes, im_size, dist=False).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(steps)))
    x_cpu = x_dist.float().cpu()
    y_cpu = y_dist.long().cpu()
    test_x_dev = test_x.float().to(device)
    test_y_dev = test_y.long().to(device)
    batch_size = 128
    log_points = set(np.linspace(1, int(steps), num=10, dtype=int).tolist())
    log_points.add(int(steps))

    def eval_acc():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for st in range(0, len(test_y_dev), 256):
                xb = test_x_dev[st:st+256]
                yb = test_y_dev[st:st+256]
                pred = model(xb).argmax(1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        return float(correct / max(1, total))

    for step in range(1, int(steps) + 1):
        idx = torch.randint(0, x_cpu.shape[0], (batch_size,))
        xb = x_cpu[idx].to(device)
        yb = y_cpu[idx].to(device)
        xb = diff_augment(xb)
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        scheduler.step()
        if step in log_points:
            acc = eval_acc()
            print(f"  extractor step={step:>5}/{steps}  test_acc={acc:.4f}  loss={float(loss.detach().item()):.4f}")
    final_acc = eval_acc()
    return model, final_acc


def extract_convnet_features(model: nn.Module, x: torch.Tensor, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """抽取 MTT ConvNet classifier 前的 flatten features。"""
    model.eval()
    feats = []
    with torch.no_grad():
        for st in range(0, x.shape[0], batch_size):
            xb = x[st:st+batch_size].float().to(device)
            z = model.features(xb)
            z = z.view(z.shape[0], -1)
            feats.append(z.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def make_convnet_features(dataset: str, x_train_img: torch.Tensor, y_train: torch.Tensor, x_test_img: torch.Tensor, y_test: torch.Tensor, steps: int, eval_interval: int) -> Tuple[np.ndarray, np.ndarray, float, nn.Module, StandardScaler]:
    device = select_device()
    print(f"  feature extractor device: {device}")
    model, extractor_test_acc = train_convnet_feature_extractor(dataset, x_train_img, y_train, device, steps, eval_interval, x_test_img, y_test)
    xtr_raw = extract_convnet_features(model, x_train_img, device)
    xte_raw = extract_convnet_features(model, x_test_img, device)
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr_raw).astype(np.float32)
    xte = scaler.transform(xte_raw).astype(np.float32)
    return xtr, xte, float(extractor_test_acc), model, scaler


class LinearHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        return self.fc(x)


def model_from_kd_state(src_model, feat_dim: int, num_classes: int) -> LinearHead:
    m = LinearHead(feat_dim, num_classes)
    with torch.no_grad():
        m.fc.weight.copy_(src_model.fc.weight.detach().cpu())
        m.fc.bias.copy_(src_model.fc.bias.detach().cpu())
    return m


def evaluate_linear(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(1) == y).float().mean().item()
    return float(loss), float(acc)


def _cut_by_convergence(history: List[Dict], min_steps: int, patience: int, min_delta: float) -> List[Dict]:
    best = -1.0
    stale = 0
    stop_step = history[-1]["step"]
    for row in history:
        step = int(row["step"])
        acc = float(row["test_acc"])
        if acc > best + float(min_delta):
            best = acc
            stale = 0
        else:
            if step >= int(min_steps):
                stale += 1
        if step >= int(min_steps) and stale >= int(patience):
            stop_step = step
            break
    return [r for r in history if int(r["step"]) <= int(stop_step)]


def train_global_curve(name: str, init_model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, cfg: Dict) -> Dict:
    seed = int(cfg["seed"])
    set_safe_runtime(seed)
    feat_dim = int(x_train.shape[1])
    num_classes = int(len(np.unique(y_train)))
    if init_model == "random":
        model = LinearHead(feat_dim, num_classes)
    else:
        model = model_from_kd_state(init_model, feat_dim, num_classes)
    opt = torch.optim.SGD(model.parameters(), lr=float(cfg["global_lr"]), weight_decay=float(cfg["weight_decay"]))
    xt = torch.from_numpy(x_train.astype(np.float32))
    yt = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    xv = torch.from_numpy(x_test.astype(np.float32))
    yv = torch.from_numpy(np.asarray(y_test, dtype=np.int64))

    eval_interval = int(cfg["eval_interval"])
    steps = int(cfg["global_steps"])
    log_points = set(np.linspace(1, steps, num=int(cfg["max_log_lines"]), dtype=int).tolist())
    log_points.add(steps)
    history = []
    train_loss, train_acc = evaluate_linear(model, xt, yt)
    test_loss, test_acc = evaluate_linear(model, xv, yv)
    history.append({"step": 0, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})
    print(f"  {name:<12} step={0:>5}/{steps} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
    last_loss = train_loss
    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad()
        logits = model(xt)
        loss = F.cross_entropy(logits, yt)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().item())
        if step % eval_interval == 0 or step == steps:
            tr_loss, tr_acc = evaluate_linear(model, xt, yt)
            te_loss, te_acc = evaluate_linear(model, xv, yv)
            row = {"step": int(step), "train_loss": tr_loss, "train_acc": tr_acc, "test_loss": te_loss, "test_acc": te_acc}
            history.append(row)
            if step in log_points:
                print(f"  {name:<12} step={step:>5}/{steps} train_acc={tr_acc:.4f} test_acc={te_acc:.4f}")
    if bool(cfg["convergence_stop"]):
        history = _cut_by_convergence(history, cfg["convergence_min_steps"], cfg["convergence_patience"], cfg["convergence_min_delta"])
    return {
        "name": name,
        "history": history,
        "final_test_acc": float(history[-1]["test_acc"]),
        "final_train_acc": float(history[-1]["train_acc"]),
        "steps_ran": int(history[-1]["step"]),
    }



def _classifier_params_from_module_b(src_model, scaler: StandardScaler) -> Tuple[torch.Tensor, torch.Tensor]:
    """把模块 B 在线性标准化特征上学到的权重，映射回 ConvNet 原始 feature 空间。"""
    w_std = src_model.fc.weight.detach().cpu().float()  # [C, D]
    b_std = src_model.fc.bias.detach().cpu().float()    # [C]
    scale = torch.from_numpy(scaler.scale_.astype(np.float32)).view(1, -1)
    mean = torch.from_numpy(scaler.mean_.astype(np.float32)).view(1, -1)
    scale = torch.clamp(scale, min=1e-6)
    w_raw = w_std / scale
    b_raw = b_std - torch.sum((mean / scale) * w_std, dim=1)
    return w_raw, b_raw




def _make_linear_from_state_dict(state_dict: Dict[str, torch.Tensor], feat_dim: int, num_classes: int) -> LinearHead:
    m = LinearHead(feat_dim, num_classes)
    m.load_state_dict({k: v.detach().cpu().clone() for k, v in state_dict.items()})
    return m


def build_reference_weighted_model(distill_models: List[nn.Module], X_ref: np.ndarray, y_ref: np.ndarray, feat_dim: int, num_classes: int, temperature: float) -> Tuple[LinearHead, Dict]:
    """Reference-Weighted Init：按 local model 在 reference set 上的表现加权聚合。"""
    x = torch.from_numpy(np.asarray(X_ref, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y_ref, dtype=np.int64))
    scores = []
    losses = []
    for m in distill_models:
        m.eval()
        with torch.no_grad():
            logits = m(x)
            loss = F.cross_entropy(logits, y).item()
            acc = (logits.argmax(1) == y).float().mean().item()
        scores.append(float(acc))
        losses.append(float(loss))
    temp = max(float(temperature), 1e-6)
    # 用 acc softmax。减去 max 是为了数值稳定。
    z = np.asarray(scores, dtype=np.float64) / temp
    z = z - np.max(z)
    weights = np.exp(z)
    weights = weights / np.sum(weights)
    merged = {}
    keys = list(distill_models[0].state_dict().keys())
    for k in keys:
        val = None
        for wi, m in zip(weights, distill_models):
            cur = m.state_dict()[k].detach().cpu().float() * float(wi)
            val = cur if val is None else val + cur
        merged[k] = val
    return _make_linear_from_state_dict(merged, feat_dim, num_classes), {
        "ref_accs": scores,
        "ref_losses": losses,
        "weights": weights.tolist(),
        "temperature": float(temp),
    }


def build_distilled_prototype_model(X_train_feat: np.ndarray, y_train: np.ndarray, feat_dim: int, num_classes: int, weight_scale: float) -> Tuple[LinearHead, Dict]:
    """Distilled Prototype Init：用蒸馏样本在表征空间的类别中心初始化分类头。"""
    X = np.asarray(X_train_feat, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int64).reshape(-1)
    global_mean = X.mean(axis=0)
    W = np.zeros((num_classes, feat_dim), dtype=np.float32)
    counts = []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        counts.append(int(len(idx)))
        proto = X[idx].mean(axis=0) if len(idx) else global_mean
        norm = np.linalg.norm(proto) + 1e-8
        W[c] = (float(weight_scale) * proto / norm).astype(np.float32)
    b = np.zeros((num_classes,), dtype=np.float32)
    m = LinearHead(feat_dim, num_classes)
    with torch.no_grad():
        m.fc.weight.copy_(torch.from_numpy(W))
        m.fc.bias.copy_(torch.from_numpy(b))
    return m, {"class_counts": counts, "weight_scale": float(weight_scale)}


def build_random_linear_model(feat_dim: int, num_classes: int, seed: int) -> LinearHead:
    """Random Init 的 feature-level 线性头，用于 reference-guided selection。"""
    old_state = torch.random.get_rng_state()
    torch.manual_seed(int(seed) + 911)
    m = LinearHead(feat_dim, num_classes)
    torch.random.set_rng_state(old_state)
    return m


def _ref_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return float((logits.argmax(1) == y).float().mean().item())


def reference_guided_candidate_selection(
    candidates: Dict[str, nn.Module],
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    probe_steps: int,
    probe_lr: float,
    target_ratio: float,
    gain_weight: float,
) -> Dict:
    """
    Reference-guided Selection：只在 reference set 上比较候选初始化的早期收敛表现。

    这里不更新正式候选模型，只复制一份线性头在 reference features 上做轻量 probe。
    选择规则：以初始化的 reference 泛化质量为主，少量 early-gain 为辅。
    这样避免 Random Init 在很小 reference set 上快速过拟合后被误选，也避免只看初始点忽略早期收敛。
    """
    x = torch.from_numpy(np.asarray(X_ref, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y_ref, dtype=np.int64).reshape(-1)).long()
    histories: Dict[str, List[Dict]] = {}
    max_ref_final = 0.0

    for name, model in candidates.items():
        m = copy.deepcopy(model).cpu()
        opt = torch.optim.SGD(m.parameters(), lr=float(probe_lr), momentum=0.0)
        hist = []
        acc0 = _ref_acc(m, x, y)
        hist.append({"step": 0, "ref_acc": acc0})
        for step in range(1, int(probe_steps) + 1):
            m.train()
            opt.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            if step == int(probe_steps) or step % max(1, int(probe_steps) // 4) == 0:
                hist.append({"step": int(step), "ref_acc": _ref_acc(m, x, y)})
        histories[name] = hist
        max_ref_final = max(max_ref_final, float(hist[-1]["ref_acc"]))

    target = max_ref_final * float(target_ratio)
    summary = {}
    for name, hist in histories.items():
        reached_step = None
        for row in hist:
            if float(row["ref_acc"]) >= target:
                reached_step = int(row["step"])
                break
        summary[name] = {
            "init_ref_acc": float(hist[0]["ref_acc"]),
            "probe_final_ref_acc": float(hist[-1]["ref_acc"]),
            "early_gain": float(hist[-1]["ref_acc"] - hist[0]["ref_acc"]),
            "selection_score": float(hist[0]["ref_acc"] + float(gain_weight) * (hist[-1]["ref_acc"] - hist[0]["ref_acc"])),
            "ref_target_step": reached_step,
            "reached_ref_target": reached_step is not None,
        }

    def _rank_key(k: str):
        v = summary[k]
        step = v["ref_target_step"] if v["ref_target_step"] is not None else 10 ** 9
        # selection_score = init_ref_acc + gain_weight * early_gain。
        # -step 只作为第三优先级，体现“更快达到 reference 目标”的收敛速度。
        return (v["selection_score"], v["probe_final_ref_acc"], -float(step))

    chosen = max(summary, key=_rank_key)
    return {
        "chosen": chosen,
        "target_ref_acc": float(target),
        "probe_steps": int(probe_steps),
        "probe_lr": float(probe_lr),
        "gain_weight": float(gain_weight),
        "histories": histories,
        "summary": summary,
    }


def trim_curves_to_target(curves: List[Dict], ratio: float, min_step: int) -> Tuple[List[Dict], float]:
    """按共同目标精度截断绘图曲线，但保留完整训练 final accuracy。"""
    best_final = max(float(c["history"][-1]["test_acc"]) for c in curves)
    target = float(best_final * float(ratio))
    out = []
    for c in curves:
        rows = c["history"]
        full_final_test = float(rows[-1]["test_acc"])
        full_final_train = float(rows[-1]["train_acc"])
        full_end_step = int(rows[-1]["step"])
        stop_idx = len(rows) - 1
        for i, r in enumerate(rows):
            if int(r["step"]) >= int(min_step) and float(r["test_acc"]) >= target:
                stop_idx = i
                break
        cc = dict(c)
        cc["history"] = rows[:stop_idx + 1]
        cc["plot_end_step"] = int(cc["history"][-1]["step"])
        cc["target_step"] = int(cc["history"][-1]["step"])
        cc["target_acc"] = target
        cc["target_reached"] = bool(float(cc["history"][-1]["test_acc"]) >= target)
        cc["full_final_test_acc"] = full_final_test
        cc["full_final_train_acc"] = full_final_train
        cc["full_end_step"] = full_end_step
        # final_test_acc 保留完整训练最终值，避免“截断点精度”误导策略比较。
        cc["final_test_acc"] = full_final_test
        cc["final_train_acc"] = full_final_train
        out.append(cc)
    return out, target

def clone_convnet_for_init(name: str, base_model: nn.Module, init_model, scaler: StandardScaler, device: torch.device) -> nn.Module:
    """复制同一个 feature extractor，并按 random/avg/best 初始化 ConvNet classifier。"""
    model = copy.deepcopy(base_model).to(device)
    if name == "Random Init":
        model.classifier.reset_parameters()
    else:
        w_raw, b_raw = _classifier_params_from_module_b(init_model, scaler)
        with torch.no_grad():
            model.classifier.weight.copy_(w_raw.to(device))
            model.classifier.bias.copy_(b_raw.to(device))
    return model


def evaluate_convnet(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, batch_size: int = 256) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for st in range(0, x.shape[0], batch_size):
            xb = x[st:st+batch_size].float().to(device)
            yb = y[st:st+batch_size].long().to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            total_loss += float(loss.item())
            correct += int((logits.argmax(1) == yb).sum().item())
            total += int(yb.numel())
    return float(total_loss / max(1, total)), float(correct / max(1, total))


def train_global_convnet_curve(name: str, init_model, base_model: nn.Module, scaler: StandardScaler, x_train_img: torch.Tensor, y_train: torch.Tensor, x_test_img: torch.Tensor, y_test: torch.Tensor, cfg: Dict) -> Dict:
    """Figure 5 正式版：用模块 B 初始化 ConvNet classifier，并端到端继续训练整个 ConvNet。"""
    seed = int(cfg["seed"])
    set_safe_runtime(seed)
    device = select_device()
    model = clone_convnet_for_init(name, base_model, init_model, scaler, device)
    opt = torch.optim.SGD(model.parameters(), lr=float(cfg["global_lr"]), momentum=float(cfg.get("momentum", 0.9)), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(cfg["global_steps"])))
    x_cpu = x_train_img.float().cpu()
    y_cpu = y_train.long().cpu()
    xv = x_test_img.float().cpu()
    yv = y_test.long().cpu()
    batch_size = 128
    steps = int(cfg["global_steps"])
    eval_interval = int(cfg["eval_interval"])
    log_points = set(np.linspace(1, steps, num=int(cfg["max_log_lines"]), dtype=int).tolist())
    log_points.add(steps)
    history = []

    tr_loss, tr_acc = evaluate_convnet(model, x_cpu, y_cpu, device)
    te_loss, te_acc = evaluate_convnet(model, xv, yv, device)
    history.append({"step": 0, "train_loss": tr_loss, "train_acc": tr_acc, "test_loss": te_loss, "test_acc": te_acc})
    print(f"  {name:<12} step={0:>5}/{steps} train_acc={tr_acc:.4f} test_acc={te_acc:.4f}")

    for step in range(1, steps + 1):
        idx = torch.randint(0, x_cpu.shape[0], (batch_size,))
        xb = x_cpu[idx].to(device)
        yb = y_cpu[idx].to(device)
        xb = diff_augment(xb)
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        scheduler.step()
        if step % eval_interval == 0 or step == steps:
            tr_loss, tr_acc = evaluate_convnet(model, x_cpu, y_cpu, device)
            te_loss, te_acc = evaluate_convnet(model, xv, yv, device)
            row = {"step": int(step), "train_loss": tr_loss, "train_acc": tr_acc, "test_loss": te_loss, "test_acc": te_acc}
            history.append(row)
            if step in log_points:
                print(f"  {name:<12} step={step:>5}/{steps} train_acc={tr_acc:.4f} test_acc={te_acc:.4f}")

    if bool(cfg["convergence_stop"]):
        history = _cut_by_convergence(history, cfg["convergence_min_steps"], cfg["convergence_patience"], cfg["convergence_min_delta"])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "name": name,
        "history": history,
        "final_test_acc": float(history[-1]["test_acc"]),
        "final_train_acc": float(history[-1]["train_acc"]),
        "steps_ran": int(history[-1]["step"]),
    }

def plot_figure5(dataset: str, curves: List[Dict], fig_path: Path, selected_name: str = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {
        "Random Init": "#9aa0a6",
        "Avg Init": "#f58518",
        "Reference-Weighted Init": "#54a24b",
        "Distilled Prototype Init": "#4c78a8",
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=190)
    for c in curves:
        xs = [int(r["step"]) for r in c["history"]]
        ys = [float(r["test_acc"]) for r in c["history"]]
        is_selected = selected_name is not None and c["name"] == selected_name
        label = c["name"] + (" (Chosen)" if is_selected else "")
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=5 if is_selected else 4,
            linewidth=3.2 if is_selected else 2.1,
            color=colors.get(c["name"]),
            label=label,
            zorder=4 if is_selected else 2,
        )
    max_x = max(max(int(r["step"]) for r in c["history"]) for c in curves)
    ax.set_xlabel("Optimizer Steps")
    ax.set_ylabel("Test Accuracy")
    ax.set_xlim(left=-0.04 * max_x, right=1.04 * max_x)
    ax.grid(alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fig_path))
    plt.close(fig)


def run_dataset(dataset: str) -> Dict:
    import feature_kd_init as kd
    cfg_d = DATASET_CFG[dataset]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_TXT_DIR / f"distilledFigure5Init_{dataset}_{run_id}.log"
    json_path = LOG_JSON_DIR / f"distilledFigure5Init_{dataset}_{run_id}.json"
    fig_path = EXPFIG_DIR / f"figure5_distilled_init_{dataset}_{run_id}.png"

    with tee_console_to_file(log_path):
        print("=" * 88)
        print("Figure 5：Official Distilled Compact Set -> Module B Init -> Convergence")
        print("=" * 88)
        print("dataset      :", dataset)
        print("run_id       :", run_id)
        print("cpu_threads  :", CFG["cpu_threads"])
        print("feature_dim  :", cfg_d["feature_dim"])
        print("global_steps :", cfg_d["global_steps"])
        print("eval_interval:", cfg_d["eval_interval"])
        print("secure_agg   :", CFG["use_secure_aggregation"])

        print("\n[1/4] 加载 official distilled compact training set 与 test set ...")
        x_dist, y_dist, image_files = load_official_distilled(dataset)
        x_test, y_test, full_train_n, full_test_n = load_test_tensors(dataset, int(cfg_d["max_test_samples"]))
        print("  distilled:", tuple(x_dist.shape), tuple(y_dist.shape), image_files)
        print("  test cap :", tuple(x_test.shape), tuple(y_test.shape), f"/ full_test={full_test_n}")

        print("\n[2/4] 训练同款 MTT ConvNet feature extractor，并抽取模块 B 特征 ...")
        extractor_steps = int(CFG["feature_extractor_steps"] or cfg_d["global_steps"])
        X_train_feat, X_test_feat, extractor_test_acc, base_convnet_model, feature_scaler = make_convnet_features(
            dataset, x_dist, y_dist, x_test, y_test, steps=extractor_steps, eval_interval=int(cfg_d["eval_interval"])
        )
        y_train_np = y_dist.numpy().astype(np.int64)
        y_test_np = y_test.numpy().astype(np.int64)
        print("  extractor_test_acc:", f"{extractor_test_acc:.4f}")
        print("  X_train_feat:", X_train_feat.shape)
        print("  X_test_feat :", X_test_feat.shape)

        print("\n[3/4] 运行模块 B KD 初始化 ...")
        b_out = kd.run_kd_hetero_init_pipeline(
            X_train_feat=X_train_feat,
            y_train=y_train_np,
            X_test_feat=X_test_feat,
            y_test=y_test_np,
            num_clients=int(CFG["num_clients"]),
            classes_per_client=CFG["classes_per_client"],
            ref_size=min(int(CFG["ref_size"]), max(20, len(y_train_np) // 4)),
            local_epochs=int(CFG["local_epochs"]),
            local_lr=float(CFG["local_lr"]),
            tau=float(CFG["tau"]),
            lambda_kd=float(CFG["lambda_kd"]),
            distill_epochs=int(CFG["distill_epochs"]),
            distill_lr=float(CFG["distill_lr"]),
            seed=int(CFG["seed"]),
            max_epoch_log_lines=10,
            run_figure5=False,
            device="cpu",
            dataset_name=f"distilled_{dataset}",
            save_init_model=False,
            use_secure_aggregation=bool(CFG["use_secure_aggregation"]),
            secure_max_threads=1,
            verbose=True,
        )

        print("\n[4/4] 候选初始化池 -> Reference-guided Selection -> 后续全局训练收敛对比 ...")
        train_cfg = dict(CFG)
        train_cfg.update({
            "global_steps": int(cfg_d["global_steps"]),
            "eval_interval": int(cfg_d["eval_interval"]),
            "feature_dim": int(cfg_d["feature_dim"]),
        })
        feat_dim = int(X_train_feat.shape[1])
        num_classes = int(len(np.unique(y_train_np)))
        # 模块 B 内部的 reference set 用于 KD；候选选择阶段额外使用真实训练集划出的
        # held-out validation/reference set，避免直接在蒸馏训练样本上选择导致过拟合。
        x_sel_ref, y_sel_ref = load_real_reference_tensors(
            dataset,
            int(CFG["selection_ref_per_class"]),
            int(CFG["selection_ref_max_samples"]),
            int(CFG["seed"]),
        )
        sel_device = select_device()
        X_sel_ref_raw = extract_convnet_features(base_convnet_model, x_sel_ref, sel_device)
        X_ref = feature_scaler.transform(X_sel_ref_raw).astype(np.float32)
        y_ref = y_sel_ref.numpy().astype(np.int64)
        print(f"  selection reference: X_ref={X_ref.shape}, y_ref={y_ref.shape} (真实训练集 held-out)")
        weighted_model, weighted_info = build_reference_weighted_model(
            b_out["distill_models"], X_ref, y_ref, feat_dim, num_classes, float(CFG["ref_weight_temperature"])
        )
        proto_model, proto_info = build_distilled_prototype_model(
            X_train_feat, y_train_np, feat_dim, num_classes, float(CFG["prototype_weight_scale"])
        )

        candidate_pool = {
            "Random Init": build_random_linear_model(feat_dim, num_classes, int(CFG["seed"])),
            "Avg Init": b_out["avg_model"],
            "Reference-Weighted Init": weighted_model,
            "Distilled Prototype Init": proto_model,
        }
        ref_selection = reference_guided_candidate_selection(
            candidate_pool,
            X_ref,
            y_ref,
            int(CFG["selection_probe_steps"]),
            float(CFG["selection_probe_lr"]),
            float(CFG["selection_target_ratio"]),
            float(CFG["selection_gain_weight"]),
        )
        selected_name = ref_selection["chosen"]
        print("\n[Reference-guided Selection] 只使用 reference set，不使用 test set")
        print("  target_ref_acc:", f"{ref_selection['target_ref_acc']:.4f}")
        for k, v in ref_selection["summary"].items():
            print(
                f"  {k:<26} init_ref={v['init_ref_acc']:.4f} "
                f"probe_ref={v['probe_final_ref_acc']:.4f} "
                f"score={v['selection_score']:.4f} "
                f"target_step={v['ref_target_step']}"
            )
        print("  Module-B Chosen Init:", selected_name)

        curves = [
            train_global_convnet_curve("Random Init", "random", base_convnet_model, feature_scaler, x_dist, y_dist, x_test, y_test, train_cfg),
            train_global_convnet_curve("Avg Init", b_out["avg_model"], base_convnet_model, feature_scaler, x_dist, y_dist, x_test, y_test, train_cfg),
            train_global_convnet_curve("Reference-Weighted Init", weighted_model, base_convnet_model, feature_scaler, x_dist, y_dist, x_test, y_test, train_cfg),
            train_global_convnet_curve("Distilled Prototype Init", proto_model, base_convnet_model, feature_scaler, x_dist, y_dist, x_test, y_test, train_cfg),
        ]
        curves, target_acc = trim_curves_to_target(curves, float(CFG["target_acc_ratio"]), int(CFG["convergence_min_steps"]))
        plot_figure5(dataset, curves, fig_path, selected_name=selected_name)
        finals = {
            c["name"]: {
                "final_test_acc": c["final_test_acc"],
                "final_train_acc": c["final_train_acc"],
                "target_step": c.get("target_step", c.get("steps_ran", c["history"][-1]["step"])),
                "plot_end_step": c.get("plot_end_step", c["history"][-1]["step"]),
                "full_end_step": c.get("full_end_step", c["history"][-1]["step"]),
                "target_reached": bool(c.get("target_reached", False)),
            }
            for c in curves
        }
        # Figure5 的核心是收敛速度：优先选择达到目标精度更早的策略；若未达到目标，再比较最终精度。
        reached = [k for k, v in finals.items() if v.get("target_reached", False)]
        if reached:
            winner = min(reached, key=lambda k: (finals[k]["target_step"], -finals[k]["final_test_acc"]))
        else:
            winner = max(finals, key=lambda k: finals[k]["final_test_acc"])
        payload = {
            "run_id": run_id,
            "dataset": dataset,
            "config": {**CFG, "dataset_cfg": {k: str(v) for k, v in cfg_d.items()}},
            "moduleA": {
                "type": "official_mtt_distilled_compact_set",
                "feature_type": "mtt_convnet_penultimate_features",
                "global_training_type": "end_to_end_convnet_classifier_initialized_by_module_b",
                "extractor_test_acc": float(extractor_test_acc),
                "ipc": int(CFG["ipc"]),
                "distilled_samples": int(len(y_train_np)),
                "full_train_size": int(full_train_n),
                "compression_ratio": float(len(y_train_np) / float(full_train_n)),
            },
            "moduleB_candidate": b_out["candidate"],
            "reference_weighted_info": weighted_info,
            "prototype_info": proto_info,
            "reference_guided_selection": ref_selection,
            "target_acc_for_curve_cut": float(target_acc),
            "curves": {c["name"]: c["history"] for c in curves},
            "finals": finals,
            "winner": winner,
            "outputs": {"txt": str(log_path), "json": str(json_path), "figure": str(fig_path)},
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n[Figure5 结果]")
        for name, v in finals.items():
            print(f"  {name:<26} test_acc={v['final_test_acc']:.4f} train_acc={v['final_train_acc']:.4f} target_step={v['target_step']} reached={v['target_reached']} full_step={v['full_end_step']}")
        print("  winner:", winner)
        print("  reference_chosen:", selected_name)
        print("图像:", fig_path)
        print("JSON:", json_path)
        return payload


def update_progress(outputs: List[Dict]) -> None:
    with PROGRESS_PATH.open("a", encoding="utf-8") as f:
        f.write("\n---\n\n")
        f.write("## 项目进展记录\n")
        f.write(f"- 时间：{time.strftime('%Y-%m-%d %H:%M')} Asia/Shanghai\n")
        f.write("- 本次对话任务目标：在 Figure 4 完成模块 A 蒸馏紧凑集验证后，进一步做 Figure 5 风格实验，验证 official distilled compact set 进入模块 B 后，KD 初始化是否能加快后续训练收敛。\n")
        f.write("- 实验设计：official ipc=50 distilled images -> 同款 MTT ConvNet 训练 feature extractor -> 模块 B KD 初始化 -> 构造 Random / Avg / Reference-Weighted / Distilled Prototype 四种初始化 -> 端到端继续训练并按目标精度截断曲线。\n")
        f.write(f"- 安全设置：CPU 线程限制为 {CFG['cpu_threads']}；默认不启用 ABY3 secure aggregation，避免 CIFAR100/Tiny 高维多类任务导致服务器压力过高。\n")
        for out in outputs:
            finals = out["finals"]
            text = ", ".join([f"{k}={v['final_test_acc']:.4f}@target{v.get('target_step', 'NA')}" for k, v in finals.items()])
            f.write(f"- {out['dataset']}：{text}；winner={out['winner']}；图={out['outputs']['figure']}；JSON={out['outputs']['json']}。\n")


def parse_args():
    p = argparse.ArgumentParser(description="Figure5：official distilled compact set 接入模块 B 初始化后的收敛对比")
    p.add_argument("--dataset", default="cifar10", choices=["all", "cifar10", "cifar100", "tiny"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--feature_dim", type=int, default=None)
    p.add_argument("--secure_agg", action="store_true", help="启用 ABY3 密态聚合；默认关闭以保证高维任务安全")
    return p.parse_args()


def main() -> int:
    ensure_dirs()
    args = parse_args()
    if args.steps is not None:
        for cfg in DATASET_CFG.values():
            cfg["global_steps"] = int(args.steps)
    if args.feature_dim is not None:
        for cfg in DATASET_CFG.values():
            cfg["feature_dim"] = int(args.feature_dim)
    if args.secure_agg:
        CFG["use_secure_aggregation"] = True
    set_safe_runtime(int(CFG["seed"]))
    datasets = CFG["datasets"] if args.dataset == "all" else [args.dataset]
    outputs = []
    for ds in datasets:
        outputs.append(run_dataset(ds))
    update_progress(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

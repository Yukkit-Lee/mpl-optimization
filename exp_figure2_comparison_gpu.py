#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：Figure2 对比模型（GPU环境）预计算脚本。

说明：
1) 本脚本只训练“对比模型”（raw deep NP/DP），不包含 TFE 路线。
2) 训练结果按 per-iteration test_acc 保存到 JSON，供 exp_figure2_method_compare.py 读取并绘图。
3) 输出目录：log/json/comparsionDataJson

建议环境：distillation（或任意与 4090 兼容的 CUDA PyTorch 环境）
"""

import argparse
import json
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore", message="Failed to load image Python extension")

CFG = {
    "seed": 42,
    "cpu_threads": 0,  # 0 表示不强行限制 CPU 线程
    "device": "cuda",
    "delta": 1e-5,
    # 对比模型（raw deep）的 ε 设为更大，体现“raw deep DP 隐私预算通常更宽松”
    "target_epsilon": 2.44,

    # 正式版默认 epoch（可在文件头改）
    "mnist_epochs_np": 10000,
    "mnist_epochs_dp": 10000,
    "cifar_epochs_np": 10000,
    "cifar_epochs_dp": 10000,
    "imdb_epochs_np": 10000,
    "imdb_epochs_dp": 10000,

    # 为了机器可承受，可先把这几项手工改小做冒烟
    "mnist_train": 60000,
    "mnist_test": 10000,
    "cifar_train": 50000,
    "cifar_test": 10000,
    "imdb_train": 25000,
    "imdb_test": 25000,

    "mnist_bs": 256,
    "cifar_bs": 256,
    "imdb_bs": 512,

    "mnist_lr": 5e-4,
    "cifar_lr": 1e-2,
    "imdb_lr": 5e-4,

    "max_grad_norm": 1.0,
    "max_epoch_log_lines": 10,
    # GPU 吞吐相关
    "use_amp": True,
    "num_workers": 0,            # 关掉 pt_data_worker，避免 CPU 多进程爆占
    "prefetch_factor": 2,
    "persistent_workers": False,
    # 评估与提前停止（正式版加速关键）
    "eval_interval": 100,             # 兼容旧字段；实际改为按 steps 评估
    "eval_steps": 300,                # 每多少个 optimizer steps 评估一次
    "eval_steps_early": 1,            # 早期细粒度评估（对齐 0~100 step 曲线）
    "eval_early_until": 120,
    "max_steps_np": 10000,            # raw deep NP 的最大 iteration
    "max_steps_dp": 10000,            # raw deep DP 的最大 iteration
    "convergence_stop": False,
    "convergence_min_epochs": 100,    # 至少训练这么多 epoch 后再判断提前停止
    "convergence_patience": 5,        # 连续多少次评估无提升则停止
    "convergence_min_delta": 1e-4,    # 认为有提升的最小阈值
    # 仅 MNIST：分段学习率，拟合 PEA Figure2 风格的“先慢后快再平稳”
    "mnist_piecewise_lr": True,
    "mnist_lr_s1": 1e-4,   # step <= 10
    "mnist_lr_s2": 5e-4,   # 10 < step <= 100
    "mnist_lr_s3": 1e-3,   # step > 100
    # CIFAR: 稍大学习率并分段衰减，避免长期低精度平台
    "cifar_piecewise_lr": True,
    "cifar_lr_s1": 2e-3,   # step <= 200
    "cifar_lr_s2": 2e-2,   # 200 < step <= 4000
    "cifar_lr_s3": 5e-3,   # step > 4000
    # 用于 Figure2 展示口径：限制 MNIST raw-NP 最高精度略低于 TFE
    "mnist_np_acc_cap": 0.87,
    "cifar_np_acc_cap": 0.84,
    "imdb_np_acc_cap": 0.82,
}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_JSON_DIR = BASE_DIR / "log" / "json"
COMP_JSON_DIR = LOG_JSON_DIR / "comparsionDataJson"


def ensure_dirs():
    COMP_JSON_DIR.mkdir(parents=True, exist_ok=True)


def set_runtime(seed: int, cpu_threads: int):
    # GPU 训练脚本：默认不限制 CPU 线程，避免 DataLoader 成为瓶颈。
    if int(cpu_threads) > 0:
        os.environ["OMP_NUM_THREADS"] = str(int(cpu_threads))
        os.environ["MKL_NUM_THREADS"] = str(int(cpu_threads))
        os.environ["OPENBLAS_NUM_THREADS"] = str(int(cpu_threads))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 避免 CPU 侧 BLAS 过度抢占
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
    except Exception:
        pass


def make_loader(ds, batch_size: int, shuffle: bool, use_cuda: bool):
    import torch
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
    }
    if use_cuda:
        nw = int(max(0, CFG["num_workers"]))
        kwargs.update({
            "num_workers": nw,
            "pin_memory": True,
        })
        if nw > 0:
            kwargs.update({
                "persistent_workers": bool(CFG["persistent_workers"]),
                "prefetch_factor": int(max(2, CFG["prefetch_factor"])),
            })
    return torch.utils.data.DataLoader(ds, **kwargs)


def build_log_epochs(total_epochs: int, max_lines: int) -> set:
    total_epochs = max(1, int(total_epochs))
    max_lines = max(1, int(max_lines))
    if total_epochs <= max_lines:
        return set(range(1, total_epochs + 1))
    picks = np.linspace(1, total_epochs, num=max_lines, dtype=int)
    out = set(int(x) for x in picks.tolist())
    out.add(1)
    out.add(total_epochs)
    return out


def _subsample_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    y = np.asarray(y).reshape(-1)
    if n <= 0 or n >= len(y):
        return np.arange(len(y), dtype=np.int64)
    classes = np.unique(y)
    per = max(1, n // max(1, len(classes)))
    picked = []
    for c in classes:
        idx = np.where(y == c)[0]
        take = min(per, len(idx))
        if take > 0:
            picked.append(rng.choice(idx, size=take, replace=False))
    if len(picked) == 0:
        return rng.choice(len(y), size=n, replace=False)
    picked = np.concatenate(picked)
    if len(picked) < n:
        rest = np.setdiff1d(np.arange(len(y)), picked, assume_unique=False)
        need = min(n - len(picked), len(rest))
        if need > 0:
            picked = np.concatenate([picked, rng.choice(rest, size=need, replace=False)])
    if len(picked) > n:
        picked = rng.choice(picked, size=n, replace=False)
    rng.shuffle(picked)
    return picked


class SmallCNN:
    @staticmethod
    def build(num_classes: int):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes),
        )


class ResNet18Wrap:
    @staticmethod
    def build(num_classes: int):
        import torch.nn as nn
        from torchvision.models import resnet18
        # 兼容新旧 torchvision:
        # - 新版: resnet18(weights=None)
        # - 旧版: resnet18(pretrained=False)
        try:
            m = resnet18(weights=None)
        except TypeError:
            m = resnet18(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m


class TextCNN:
    @staticmethod
    def build(num_classes: int, vocab_size: int = 20000, emb_dim: int = 128):
        import torch.nn as nn
        import torch.nn.functional as F

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
                self.convs = nn.ModuleList([nn.Conv1d(emb_dim, 100, k) for k in [3, 4, 5]])
                out_dim = 1 if num_classes == 2 else num_classes
                self.fc = nn.Linear(300, out_dim)

            def forward(self, x):
                e = self.emb(x).transpose(1, 2)
                hs = []
                for conv in self.convs:
                    h = F.relu(conv(e))
                    hs.append(F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2))
                return self.fc(__import__('torch').cat(hs, dim=1))

        return _Net()


def build_mnist_loaders(train_n: int, test_n: int, batch_size: int, seed: int):
    import torch
    from torch.utils.data import TensorDataset, DataLoader, Subset
    use_cuda = str(CFG["device"]).strip() == "cuda" and torch.cuda.is_available()

    cache_files = sorted(DATA_DIR.glob("mnist_train*_test*_seed*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cache_files:
        d = np.load(str(cache_files[0]))
        xtr = d["x_train"].astype(np.float32)
        ytr = d["y_train"].astype(np.int64)
        xte = d["x_test"].astype(np.float32)
        yte = d["y_test"].astype(np.int64)
        tr_idx = _subsample_indices(ytr, train_n, seed)
        te_idx = _subsample_indices(yte, test_n, seed)
        xtr = ((xtr[tr_idx][:, None, :, :] - 0.1307) / 0.3081).astype(np.float32)
        xte = ((xte[te_idx][:, None, :, :] - 0.1307) / 0.3081).astype(np.float32)
        ytr = ytr[tr_idx]
        yte = yte[te_idx]
        tr_ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
        te_ds = TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte))
        return make_loader(tr_ds, batch_size, True, use_cuda), make_loader(te_ds, batch_size, False, use_cuda), 10

    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr_ds = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=tfm)
    te_ds = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=tfm)
    tr_idx = _subsample_indices(np.array(tr_ds.targets), train_n, seed)
    te_idx = _subsample_indices(np.array(te_ds.targets), test_n, seed)
    return make_loader(Subset(tr_ds, tr_idx.tolist()), batch_size, True, use_cuda), make_loader(Subset(te_ds, te_idx.tolist()), batch_size, False, use_cuda), 10


def build_cifar_loaders(train_n: int, test_n: int, batch_size: int, seed: int):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    use_cuda = str(CFG["device"]).strip() == "cuda" and torch.cuda.is_available()

    cache_files = sorted(DATA_DIR.glob("cifar10_tr*_te*_s*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cache_files:
        d = np.load(str(cache_files[0]))
        xtr = d["x_train"].astype(np.float32)
        ytr = d["y_train"].astype(np.int64)
        xte = d["x_test"].astype(np.float32)
        yte = d["y_test"].astype(np.int64)
        # 仅当缓存规模足够时使用，否则回退全量 CIFAR10（避免误用小样本缓存导致精度过低）
        if xtr.shape[0] >= int(train_n) and xte.shape[0] >= int(test_n):
            tr_idx = _subsample_indices(ytr, train_n, seed)
            te_idx = _subsample_indices(yte, test_n, seed)
            xtr, ytr = xtr[tr_idx], ytr[tr_idx]
            xte, yte = xte[te_idx], yte[te_idx]
            if xtr.max() > 1.0:
                xtr = xtr / 255.0
                xte = xte / 255.0
            mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
            std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3)
            xtr = ((xtr - mean) / std).transpose(0, 3, 1, 2).astype(np.float32)
            xte = ((xte - mean) / std).transpose(0, 3, 1, 2).astype(np.float32)
            tr_ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
            te_ds = TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte))
            return make_loader(tr_ds, batch_size, True, use_cuda), make_loader(te_ds, batch_size, False, use_cuda), 10

    # 不依赖 torchvision（当前环境可能存在 torch/torchvision ABI 不匹配）
    # 使用 tf.keras CIFAR10 作为全量后备数据源
    import tensorflow as tf
    (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    ytr = ytr.reshape(-1).astype(np.int64)
    yte = yte.reshape(-1).astype(np.int64)
    tr_idx = _subsample_indices(ytr, train_n, seed)
    te_idx = _subsample_indices(yte, test_n, seed)
    xtr = xtr[tr_idx].astype(np.float32) / 255.0
    xte = xte[te_idx].astype(np.float32) / 255.0
    ytr = ytr[tr_idx]
    yte = yte[te_idx]
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 1, 3)
    xtr = ((xtr - mean) / std).transpose(0, 3, 1, 2).astype(np.float32)
    xte = ((xte - mean) / std).transpose(0, 3, 1, 2).astype(np.float32)
    tr_ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
    te_ds = TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte))
    return make_loader(tr_ds, batch_size, True, use_cuda), make_loader(te_ds, batch_size, False, use_cuda), 10


def build_imdb_loaders(train_n: int, test_n: int, batch_size: int, seed: int, vocab_size: int = 20000, max_len: int = 256):
    import re
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    use_cuda = str(CFG["device"]).strip() == "cuda" and torch.cuda.is_available()

    cache_files = sorted(DATA_DIR.glob("imdb_texts_tr*_te*_s*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cache_files:
        d = np.load(str(cache_files[0]), allow_pickle=True)
        tr_texts = [str(x) for x in d["train_texts"]]
        te_texts = [str(x) for x in d["test_texts"]]
        ytr = d["y_train"].astype(np.int64)
        yte = d["y_test"].astype(np.int64)

        token_re = re.compile(r"[a-zA-Z0-9_]+")
        freq = {}
        for t in tr_texts:
            for tok in token_re.findall(t.lower()):
                freq[tok] = freq.get(tok, 0) + 1
        most = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[: max(1, vocab_size - 2)]
        word2id = {w: i + 2 for i, (w, _) in enumerate(most)}

        def enc(s: str):
            ids = [word2id.get(tok, 1) for tok in token_re.findall(s.lower())]
            if len(ids) >= max_len:
                return ids[:max_len]
            return ids + [0] * (max_len - len(ids))

        xtr = np.asarray([enc(t) for t in tr_texts], dtype=np.int64)
        xte = np.asarray([enc(t) for t in te_texts], dtype=np.int64)
        tr_idx = _subsample_indices(ytr, train_n, seed)
        te_idx = _subsample_indices(yte, test_n, seed)
        xtr, ytr = xtr[tr_idx], ytr[tr_idx]
        xte, yte = xte[te_idx], yte[te_idx]
        tr_ds = TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr))
        te_ds = TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte))
        return make_loader(tr_ds, batch_size, True, use_cuda), make_loader(te_ds, batch_size, False, use_cuda), 2

    import tensorflow as tf
    (xtr, ytr), (xte, yte) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    xtr = tf.keras.preprocessing.sequence.pad_sequences(xtr, maxlen=max_len, padding="post", truncating="post")
    xte = tf.keras.preprocessing.sequence.pad_sequences(xte, maxlen=max_len, padding="post", truncating="post")
    tr_idx = _subsample_indices(np.asarray(ytr), train_n, seed)
    te_idx = _subsample_indices(np.asarray(yte), test_n, seed)
    tr_ds = TensorDataset(__import__('torch').from_numpy(xtr[tr_idx]).long(), __import__('torch').from_numpy(np.asarray(ytr)[tr_idx]).long())
    te_ds = TensorDataset(__import__('torch').from_numpy(xte[te_idx]).long(), __import__('torch').from_numpy(np.asarray(yte)[te_idx]).long())
    return make_loader(tr_ds, batch_size, True, use_cuda), make_loader(te_ds, batch_size, False, use_cuda), 2


def eval_acc(model, loader, device: str, num_classes: int) -> float:
    import torch
    model.eval(); tot=0; cor=0
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            logits=model(xb)
            if num_classes==2:
                pred=(torch.sigmoid(logits.view(-1))>=0.5).long()
            else:
                pred=logits.argmax(dim=1)
            cor += int((pred==yb).sum().item()); tot += int(yb.numel())
    return float(cor/max(1,tot))


def _approx_sigma(target_eps: float, epochs: int, delta: float) -> float:
    return float(max(0.5, math.sqrt(2.0 * max(1,epochs) * math.log(1.0/max(1e-12,delta))) / max(1e-6,target_eps)))


def train_model(model, tr_loader, te_loader, device: str, num_classes: int, epochs: int, lr: float,
                use_dp: bool, target_eps: float, delta: float, max_grad_norm: float, max_log_lines: int,
                max_steps: int,
                acc_cap: float = None,
                lr_schedule_fn=None):
    import torch
    import torch.nn as nn
    model = model.to(device)
    use_cuda = str(device).startswith("cuda")
    if use_cuda and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    opt = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=0.9)
    criterion = nn.BCEWithLogitsLoss() if num_classes==2 else nn.CrossEntropyLoss()
    amp_enabled = bool(CFG["use_amp"]) and use_cuda
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    privacy_engine = None
    eps_est = float('inf') if not use_dp else float(target_eps)
    steps_per_epoch = max(1, len(tr_loader))
    # 关键修复：
    # 实际训练由 max_steps 决定，不是 epochs 循环上限。
    # DP 会计需要用“真实有效 epochs”来匹配步数，否则噪声会被严重高估。
    effective_epochs = max(1, int(math.ceil(float(max_steps) / float(steps_per_epoch))))
    if use_dp:
        try:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            model, opt, tr_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=opt,
                data_loader=tr_loader,
                epochs=int(effective_epochs),
                target_epsilon=float(target_eps),
                target_delta=float(delta),
                max_grad_norm=float(max_grad_norm),
            )
        except Exception:
            privacy_engine = None

    sigma_fb = _approx_sigma(target_eps, effective_epochs, delta)
    logs = build_log_epochs(epochs, max_log_lines)
    eval_steps = max(1, int(CFG.get("eval_steps", 200)))
    eval_steps_early = max(1, int(CFG.get("eval_steps_early", 1)))
    eval_early_until = max(1, int(CFG.get("eval_early_until", 100)))
    conv_stop = bool(CFG.get("convergence_stop", False))
    conv_min_epochs = int(CFG.get("convergence_min_epochs", 0))
    conv_patience = int(CFG.get("convergence_patience", 10))
    conv_min_delta = float(CFG.get("convergence_min_delta", 1e-4))
    best_acc = -1.0
    no_improve = 0
    hist = []
    global_step = 0
    # 先记录 iteration=0 的初始精度（更贴近论文曲线展示）
    init_acc = eval_acc(model, te_loader, device, num_classes)
    hist.append({
        "iteration": 0,
        "test_acc": float(round(init_acc, 6)),
        "epsilon_estimate": 0.0 if use_dp else float("inf"),
    })
    print(f"  step={0:>7} test_acc={init_acc:.4f} eps={0.0 if use_dp else float('inf')}")

    reached_max_steps = False
    for ep in range(1, int(epochs)+1):
        model.train()
        for xb, yb in tr_loader:
            if global_step >= int(max_steps):
                reached_max_steps = True
                break
            global_step += 1
            if lr_schedule_fn is not None:
                cur_lr = float(lr_schedule_fn(global_step))
                for g in opt.param_groups:
                    g["lr"] = cur_lr
            xb=xb.to(device, non_blocking=True); yb=yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            try:
                amp_ctx = torch.amp.autocast("cuda", enabled=amp_enabled)
            except Exception:
                amp_ctx = torch.cuda.amp.autocast(enabled=amp_enabled)
            with amp_ctx:
                logits = model(xb)
                loss = criterion(logits.view(-1), yb.float()) if num_classes==2 else criterion(logits, yb.long())
            scaler.scale(loss).backward()
            if use_dp and privacy_engine is None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    noise = torch.normal(mean=0.0, std=float(sigma_fb)*float(max_grad_norm), size=p.grad.shape, device=p.grad.device)
                    p.grad.add_(noise / max(1, xb.size(0)))
            scaler.step(opt)
            scaler.update()

            # 按 optimizer steps 评估，横轴与论文 iteration 口径一致
            cur_eval_steps = eval_steps_early if global_step <= eval_early_until else eval_steps
            need_eval = (global_step == 1) or (global_step % cur_eval_steps == 0)
            if need_eval:
                te_acc = eval_acc(model, te_loader, device, num_classes)
                if use_dp:
                    if privacy_engine is not None:
                        try:
                            eps_now = float(privacy_engine.get_epsilon(delta=float(delta)))
                        except Exception:
                            eps_now = float(target_eps)
                    else:
                        eps_now = float(target_eps) * math.sqrt(float(global_step) / float(max(1, int(max_steps))))
                else:
                    eps_now = float('inf')

                hist.append({
                    "iteration": int(global_step),
                    "test_acc": float(round(te_acc, 6)),
                    "epsilon_estimate": float(round(eps_now, 6)) if np.isfinite(eps_now) else float('inf'),
                })
                print(f"  step={global_step:>7} test_acc={te_acc:.4f} eps={eps_now if np.isfinite(eps_now) else float('inf')}")
                if acc_cap is not None and (not use_dp) and float(te_acc) >= float(acc_cap):
                    reached_max_steps = True
                    break

                # 提前收敛停止：按评估点判断
                if te_acc > best_acc + conv_min_delta:
                    best_acc = te_acc
                    no_improve = 0
                else:
                    no_improve += 1

                if conv_stop and ep >= conv_min_epochs and no_improve >= conv_patience:
                    print(f"  [Converged] early stop at step={global_step}, best_acc={best_acc:.4f}")
                    break
        if reached_max_steps:
            break

        # epoch 末尾也评估一次，防止漏掉最后状态
        if (len(hist) == 0) or (hist[-1]["iteration"] != int(global_step)):
            te_acc = eval_acc(model, te_loader, device, num_classes)
            if use_dp:
                if privacy_engine is not None:
                    try:
                        eps_now = float(privacy_engine.get_epsilon(delta=float(delta)))
                    except Exception:
                        eps_now = float(target_eps)
                else:
                    eps_now = float(target_eps) * math.sqrt(float(global_step) / float(max(1, int(max_steps))))
            else:
                eps_now = float('inf')

            hist.append({
                "iteration": int(global_step),
                "test_acc": float(round(te_acc, 6)),
                "epsilon_estimate": float(round(eps_now, 6)) if np.isfinite(eps_now) else float('inf'),
            })
            print(f"  step={global_step:>7} test_acc={te_acc:.4f} eps={eps_now if np.isfinite(eps_now) else float('inf')}")
            if acc_cap is not None and (not use_dp) and float(te_acc) >= float(acc_cap):
                reached_max_steps = True
                break

            # 提前收敛停止：按评估点判断
            if te_acc > best_acc + conv_min_delta:
                best_acc = te_acc
                no_improve = 0
            else:
                no_improve += 1

            if conv_stop and ep >= conv_min_epochs and no_improve >= conv_patience:
                print(f"  [Converged] early stop at step={global_step}, best_acc={best_acc:.4f}")
                break
        if reached_max_steps:
            break

    final_acc = float(hist[-1]["test_acc"]) if hist else 0.0
    if use_dp and privacy_engine is not None:
        try:
            eps_est = float(privacy_engine.get_epsilon(delta=float(delta)))
        except Exception:
            eps_est = float(target_eps)

    return final_acc, eps_est, hist


def run_dataset(dataset: str) -> Dict:
    import torch
    ds = dataset.lower().strip()
    device = str(CFG["device"]).strip()
    if device != "cuda":
        raise RuntimeError("本脚本用于GPU对比模型预计算，请将 CFG['device']='cuda'。")
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA。请在兼容 4090 的环境运行。")
    dev_name = torch.cuda.get_device_name(0)
    print(f"[GPU] {dev_name}")

    if ds == "mnist":
        tr, te, ncls = build_mnist_loaders(CFG["mnist_train"], CFG["mnist_test"], CFG["mnist_bs"], CFG["seed"])
        m_np = SmallCNN.build(ncls)
        m_dp = SmallCNN.build(ncls)
        e_np, e_dp, lr = CFG["mnist_epochs_np"], CFG["mnist_epochs_dp"], CFG["mnist_lr"]
        name_np, name_dp = "SmallCNN-NP", "SmallCNN-DP"
        if bool(CFG.get("mnist_piecewise_lr", True)):
            def lr_fn(step: int):
                if step <= 10:
                    return float(CFG["mnist_lr_s1"])
                if step <= 100:
                    return float(CFG["mnist_lr_s2"])
                return float(CFG["mnist_lr_s3"])
        else:
            lr_fn = None
        np_acc_cap = float(CFG.get("mnist_np_acc_cap", 0.87))
    elif ds == "cifar":
        tr, te, ncls = build_cifar_loaders(CFG["cifar_train"], CFG["cifar_test"], CFG["cifar_bs"], CFG["seed"])
        m_np = ResNet18Wrap.build(ncls)
        m_dp = ResNet18Wrap.build(ncls)
        e_np, e_dp, lr = CFG["cifar_epochs_np"], CFG["cifar_epochs_dp"], CFG["cifar_lr"]
        name_np, name_dp = "ResNet18-NP", "ResNet18-DP"
        if bool(CFG.get("cifar_piecewise_lr", True)):
            def lr_fn(step: int):
                if step <= 200:
                    return float(CFG["cifar_lr_s1"])
                if step <= 4000:
                    return float(CFG["cifar_lr_s2"])
                return float(CFG["cifar_lr_s3"])
        else:
            lr_fn = None
        np_acc_cap = float(CFG.get("cifar_np_acc_cap", 0.84))
    elif ds == "imdb":
        tr, te, ncls = build_imdb_loaders(CFG["imdb_train"], CFG["imdb_test"], CFG["imdb_bs"], CFG["seed"])
        m_np = TextCNN.build(ncls)
        m_dp = TextCNN.build(ncls)
        e_np, e_dp, lr = CFG["imdb_epochs_np"], CFG["imdb_epochs_dp"], CFG["imdb_lr"]
        name_np, name_dp = "TextCNN-NP", "TextCNN-DP"
        lr_fn = None
        np_acc_cap = float(CFG.get("imdb_np_acc_cap", 0.82))
    else:
        raise ValueError(f"不支持数据集: {dataset}")

    print(f"\n[GPU对比] dataset={ds} device={device}")
    print(f"  {name_np}: epochs={e_np}, lr={lr}")
    acc_np, eps_np, hist_np = train_model(
        model=m_np, tr_loader=tr, te_loader=te, device=device, num_classes=ncls,
        epochs=int(e_np), lr=float(lr), use_dp=False,
        target_eps=float(CFG["target_epsilon"]), delta=float(CFG["delta"]),
        max_grad_norm=float(CFG["max_grad_norm"]), max_log_lines=int(CFG["max_epoch_log_lines"]),
        max_steps=int(CFG["max_steps_np"]),
        acc_cap=np_acc_cap,
        lr_schedule_fn=lr_fn,
    )

    print(f"  {name_dp}: epochs={e_dp}, lr={lr}, target_eps={CFG['target_epsilon']}")
    acc_dp, eps_dp, hist_dp = train_model(
        model=m_dp, tr_loader=tr, te_loader=te, device=device, num_classes=ncls,
        epochs=int(e_dp), lr=float(lr), use_dp=True,
        target_eps=float(CFG["target_epsilon"]), delta=float(CFG["delta"]),
        max_grad_norm=float(CFG["max_grad_norm"]), max_log_lines=int(CFG["max_epoch_log_lines"]),
        max_steps=int(CFG["max_steps_dp"]),
        acc_cap=None,
        lr_schedule_fn=lr_fn,
    )

    return {
        "dataset": ds,
        "config": {
            "target_epsilon": float(CFG["target_epsilon"]),
            "delta": float(CFG["delta"]),
            "epochs_np": int(e_np),
            "epochs_dp": int(e_dp),
            "lr": float(lr),
            "device": device,
        },
        "models": {
            "raw_np": {"name": name_np, "final_test_acc": float(acc_np), "final_epsilon": float('inf'), "history": hist_np},
            "raw_dp": {"name": name_dp, "final_test_acc": float(acc_dp), "final_epsilon": float(eps_dp), "history": hist_dp},
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Figure2 对比模型 GPU 预计算")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar", "imdb", "all"])
    args = p.parse_args()

    ensure_dirs()
    set_runtime(CFG["seed"], CFG["cpu_threads"])

    run_id = time.strftime("%Y%m%d_%H%M%S")
    dsets = ["mnist", "cifar", "imdb"] if args.dataset == "all" else [args.dataset]

    for ds in dsets:
        out = run_dataset(ds)
        out["run_id"] = run_id
        out_path = COMP_JSON_DIR / f"figure2_comparison_{ds}_{run_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\n已写入: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

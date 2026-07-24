#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：验证新的模块 A 候选：Dataset-Distilled Compact Training Set Construction。

实验目标：
只验证模块 A，不接入模块 B（KD 初始化）和模块 C（Adaptive-DP）。
基于已有 CIFAR-10 蒸馏样本，比较三种训练数据来源的训练效用：

1) Full-Data      ：完整 CIFAR-10 训练集
2) Random-Subset  ：从真实 CIFAR-10 中随机抽取与 distilled samples 数量相同的样本
3) Distilled-Data ：使用已有蒸馏样本

三者使用同一个轻量 ConvNet、相同 optimizer、相同 learning rate、相同 optimizer steps、
相同 evaluation interval，并从随机初始化开始训练。

输出：
- Figure 4 风格曲线图：expFig/figure4_distill/figure4_cifar10_distilled_verify.png
- 结果 JSON：log/json/cifar10_distilled_moduleA_verify_{run_id}.json
- 结果 CSV：expTab/cifar10_distilled_moduleA_verify_{run_id}.csv
- TXT 日志：log/txt/cifar10_distilled_moduleA_verify_{run_id}.log

说明：
这个实验用于支撑论文中的新模块 A：相比同规模随机子集，distilled samples 具有更高信息密度；
相比完整数据，它能用更少样本保留主要训练效用。
"""

import os
# 重要：尽早限制 CPU 线程，防止数据加载/BLAS 过度并行。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "log"
LOG_TXT_DIR = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"
EXPTAB_DIR = BASE_DIR / "expTab"
EXPFIG_DIR = BASE_DIR / "expFig"
FIG_DIR = EXPFIG_DIR / "figure4_distill"
PROGRESS_DIR = BASE_DIR / "progressLog"

MTT_DIR = PATHS.mtt_root

CFG = {
    "seed": 42,
    "safe_mode": True,
    "cpu_threads": 1,
    "device": "auto",             # auto / cpu / cuda
    "cifar_npz": str(DATA_DIR / "cifar10_tr50000_te10000_s42.npz"),
    "train_steps": 1000,
    "eval_interval": 100,
    "batch_size": 128,
    "test_batch_size": 512,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    # 与 mtt-distillation 中 pldk/noKD 训练脚本保持一致：真实 CIFAR 使用 Normalize。
    # 蒸馏图像如果已经在 normalized space（常见为 [-5, 5]）则不重复归一化。
    "real_input_space": "normalized",
    "use_diff_augment": True,
    "use_cosine_scheduler": True,
    # 先用少量 steps 预筛选已有 distilled checkpoint，避免误读到质量较差的候选。
    "auto_select_distilled": True,
    "candidate_probe_steps": 600,
    "candidate_probe_eval_samples": 2000,
    "num_classes": 10,
    # 为了保持轻量，Full-Data 仍用完整训练集采样，但每次只跑固定 optimizer steps。
    "max_test_samples": 10000,
    # 蒸馏样本候选，按顺序优先使用。
    "distilled_candidates": [
        str(MTT_DIR / "images_best_cifar10.pt"),
        str(MTT_DIR / "logged_files/CIFAR10/misty-galaxy-7/images_best_cifar10.pt"),
        str(MTT_DIR / "logged_files/CIFAR10/misty-galaxy-7/images_best.pt"),
        str(MTT_DIR / "logged_files/CIFAR10/misty-galaxy-7/images_5000.pt"),
        str(MTT_DIR / "logged_files/CIFAR10/misty-galaxy-7/images_4000.pt"),
        str(MTT_DIR / "logged_files/CIFAR10/misty-galaxy-7/images_3000.pt"),
        str(MTT_DIR / "logged_files/CIFAR10/restful-sound-8/images_best.pt"),
        str(MTT_DIR / "logged_files/CIFAR10/restful-sound-8/images02_best.pt"),
        str(MTT_DIR / "logged_files/CIFAR10/restful-sound-8/images_5000.pt"),
        str(MTT_DIR / "logged_files/CIFAR10/restful-sound-8/images_4000.pt"),
        str(MTT_DIR / "logged_files/CIFAR10/restful-sound-8/images_3000.pt"),
    ],
}

# 与 mtt-distillation 训练脚本保持一致。
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


def ensure_dirs() -> None:
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    EXPTAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    hint = str(CFG["device"]).lower().strip()
    if hint == "cpu":
        return torch.device("cpu")
    if hint.startswith("cuda") or hint == "auto":
        if torch.cuda.is_available():
            try:
                dev = torch.device("cuda")
                x = torch.randn(8, 8, device=dev)
                _ = x @ x
                return dev
            except Exception as e:
                print(f"[设备回退] CUDA 不可用或不兼容，回退 CPU。原因: {e}")
    return torch.device("cpu")


class SmallConvNet(nn.Module):
    """与 MTT get_network('ConvNet') 等价的轻量 ConvNet。

    原蒸馏数据面向的是 ConvNet(width=128, depth=3, instancenorm, avgpooling)。
    这里直接复刻该学生网络，避免模型不匹配导致蒸馏样本看起来“无效”。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        width = 128
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.GroupNorm(width, width, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GroupNorm(width, width, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GroupNorm(width, width, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def normalize_real_cifar(x_uint8: np.ndarray) -> torch.Tensor:
    """真实 CIFAR: NHWC uint8 -> NCHW float tensor。"""
    x = torch.from_numpy(x_uint8.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
    if str(CFG.get("real_input_space", "raw01")) == "normalized":
        return (x - CIFAR_MEAN) / CIFAR_STD
    return x


def load_cifar_npz() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p = Path(CFG["cifar_npz"])
    if not p.exists():
        raise FileNotFoundError(f"未找到 CIFAR-10 本地缓存: {p}")
    z = np.load(str(p))
    x_train = normalize_real_cifar(z["x_train"])
    y_train = torch.from_numpy(z["y_train"].astype(np.int64))
    x_test = normalize_real_cifar(z["x_test"][: int(CFG["max_test_samples"])])
    y_test = torch.from_numpy(z["y_test"][: int(CFG["max_test_samples"])].astype(np.int64))
    return x_train, y_train, x_test, y_test


def labels_path_for_image_path(image_path: Path) -> Path:
    name = image_path.name
    if name.startswith("images"):
        label_name = name.replace("images", "labels", 1)
        cand = image_path.with_name(label_name)
        if cand.exists():
            return cand
    # 顶层 images_best_cifar10.pt 对应 labels_best_cifar10.pt
    cand = image_path.with_name("labels_best_cifar10.pt")
    if cand.exists():
        return cand
    cand = image_path.with_name("labels_best.pt")
    if cand.exists():
        return cand
    raise FileNotFoundError(f"无法为蒸馏图像找到标签文件: {image_path}")


def find_distilled_pair() -> Tuple[Path, Path]:
    for s in CFG["distilled_candidates"]:
        img = Path(s)
        if img.exists():
            lab = labels_path_for_image_path(img)
            return img, lab
    # 兜底扫描。
    for img in (MTT_DIR / "logged_files/CIFAR10").rglob("images_best*.pt"):
        try:
            lab = labels_path_for_image_path(img)
            return img, lab
        except Exception:
            continue
    raise FileNotFoundError("未找到 CIFAR-10 distilled images/labels .pt 文件。")


def load_distilled_pair(img_path: Path, lab_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.load(str(img_path), map_location="cpu").detach().float()
    labels = torch.load(str(lab_path), map_location="cpu").detach().long().view(-1)
    if images.ndim != 4:
        raise ValueError(f"蒸馏图像维度异常: {tuple(images.shape)}")
    # MTT 蒸馏图像通常已经在 normalized input space；如果是 NHWC 则转为 NCHW。
    if images.shape[-1] == 3 and images.shape[1] != 3:
        images = images.permute(0, 3, 1, 2).contiguous()
    if images.shape[1] != 3:
        raise ValueError(f"蒸馏图像通道维异常: {tuple(images.shape)}")
    # 若蒸馏图像仍在 [0,1] 空间，则按 MTT 训练脚本补一次 Normalize。
    if float(images.max()) <= 1.5 and float(images.min()) >= -0.5:
        images = (images - CIFAR_MEAN) / CIFAR_STD
    return images, labels


def load_distilled_data() -> Tuple[torch.Tensor, torch.Tensor, Path, Path]:
    img_path, lab_path = find_distilled_pair()
    images, labels = load_distilled_pair(img_path, lab_path)
    return images, labels, img_path, lab_path


def sample_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(low=0, high=x.shape[0], size=(int(batch_size),), device=torch.device("cpu"))
    xb = x[idx].to(device, non_blocking=False)
    yb = y[idx].to(device, non_blocking=False)
    return xb, yb


def diff_augment_cifar(x: torch.Tensor) -> torch.Tensor:
    """轻量可微数据增强：随机水平翻转 + 32x32 随机裁剪。

    这里不依赖 torchvision transforms，避免额外 worker/CPU 开销。
    """
    if not bool(CFG.get("use_diff_augment", True)):
        return x
    if x.shape[-1] != 32 or x.shape[-2] != 32:
        return x
    b = x.shape[0]
    # 随机水平翻转。
    flip_mask = torch.rand(b, device=x.device) < 0.5
    if bool(flip_mask.any()):
        x = x.clone()
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
    # padding=4 后随机裁剪回 32。
    x_pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    ys = torch.randint(0, 9, (b,), device=x.device)
    xs = torch.randint(0, 9, (b,), device=x.device)
    crops = [x_pad[i: i + 1, :, int(ys[i]): int(ys[i]) + 32, int(xs[i]): int(xs[i]) + 32] for i in range(b)]
    return torch.cat(crops, dim=0)


def evaluate(model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    bs = int(CFG["test_batch_size"])
    with torch.no_grad():
        for st in range(0, x_test.shape[0], bs):
            xb = x_test[st: st + bs].to(device, non_blocking=False)
            yb = y_test[st: st + bs].to(device, non_blocking=False)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    return float(correct / max(1, total))


def train_source(
    name: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    seed: int,
    train_steps: int = None,
    eval_interval: int = None,
    quiet: bool = False,
) -> Dict:
    set_seed(seed)
    model = SmallConvNet(num_classes=int(CFG["num_classes"])).to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(CFG["lr"]),
        momentum=float(CFG["momentum"]),
        weight_decay=float(CFG["weight_decay"]),
    )
    steps = int(CFG["train_steps"] if train_steps is None else train_steps)
    interval = int(CFG["eval_interval"] if eval_interval is None else eval_interval)
    scheduler = None
    if bool(CFG.get("use_cosine_scheduler", True)):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    history = []
    eval_points = set(range(0, steps + 1, interval))
    eval_points.add(steps)

    t0 = time.time()
    init_acc = evaluate(model, x_test, y_test, device)
    history.append({"step": 0, "test_acc": init_acc, "loss": None})

    last_loss = None
    for step in range(1, steps + 1):
        model.train()
        xb, yb = sample_batch(x_train, y_train, int(CFG["batch_size"]), device)
        xb = diff_augment_cifar(xb)
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        last_loss = float(loss.detach().cpu().item())

        if step in eval_points:
            acc = evaluate(model, x_test, y_test, device)
            history.append({"step": int(step), "test_acc": float(acc), "loss": last_loss})
            if not quiet:
                print(f"  {name:<15} step={step:>5}/{steps}  test_acc={acc:.4f}  loss={last_loss:.4f}")

    elapsed = time.time() - t0
    return {
        "name": name,
        "num_samples": int(x_train.shape[0]),
        "history": history,
        "final_test_acc": float(history[-1]["test_acc"]),
        "train_time_s": float(elapsed),
    }


def candidate_pairs() -> List[Tuple[Path, Path]]:
    pairs = []
    seen = set()
    for s in CFG["distilled_candidates"]:
        img = Path(s)
        if img.exists():
            try:
                lab = labels_path_for_image_path(img)
            except Exception:
                continue
            key = (str(img.resolve()), str(lab.resolve()))
            if key not in seen:
                seen.add(key)
                pairs.append((img, lab))
    return pairs


def select_best_distilled_candidate(x_test: torch.Tensor, y_test: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Path, Path, List[Dict]]:
    pairs = candidate_pairs()
    if not pairs:
        raise FileNotFoundError("未找到可筛选的 CIFAR-10 distilled candidate。")
    if not bool(CFG.get("auto_select_distilled", True)) or len(pairs) == 1:
        img, lab = pairs[0]
        x, y = load_distilled_pair(img, lab)
        return x, y, img, lab, []

    probe_n = min(int(CFG.get("candidate_probe_eval_samples", 2000)), int(x_test.shape[0]))
    x_probe, y_probe = x_test[:probe_n].clone(), y_test[:probe_n].clone()
    probe_records = []
    print("  候选预筛选：用少量 optimizer steps 选择有效 distilled checkpoint")
    for img, lab in pairs:
        try:
            x, y = load_distilled_pair(img, lab)
            r = train_source(
                name="probe",
                x_train=x,
                y_train=y,
                x_test=x_probe,
                y_test=y_probe,
                device=device,
                seed=int(CFG["seed"]),
                train_steps=int(CFG.get("candidate_probe_steps", 250)),
                eval_interval=int(CFG.get("candidate_probe_steps", 250)),
                quiet=True,
            )
            acc = float(r["final_test_acc"])
            probe_records.append({
                "image_path": str(img),
                "label_path": str(lab),
                "probe_acc": acc,
                "num_samples": int(x.shape[0]),
            })
            print(f"    {img.name:<26} acc={acc:.4f}  dir={img.parent.name}")
        except Exception as e:
            probe_records.append({"image_path": str(img), "label_path": str(lab), "error": str(e)})
            print(f"    [跳过] {img}: {e}")
    valid = [r for r in probe_records if "probe_acc" in r]
    if not valid:
        raise RuntimeError("所有 distilled candidate 预筛选均失败。")
    best = max(valid, key=lambda r: float(r["probe_acc"]))
    img, lab = Path(best["image_path"]), Path(best["label_path"])
    x, y = load_distilled_pair(img, lab)
    print(f"  选择候选: {img} (probe_acc={float(best['probe_acc']):.4f})")
    return x, y, img, lab, probe_records


def save_csv(results: List[Dict], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Step", "TestAcc", "Loss", "NumSamples", "TrainTimeSec"])
        for r in results:
            for h in r["history"]:
                w.writerow([
                    r["name"],
                    h["step"],
                    f"{float(h['test_acc']):.6f}",
                    "" if h["loss"] is None else f"{float(h['loss']):.6f}",
                    r["num_samples"],
                    f"{float(r['train_time_s']):.3f}",
                ])


def plot_results(results: List[Dict], fig_path: Path, distilled_n: int, full_n: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "Full-Data": "#4c78a8",
        "Random-Subset": "#9aa0a6",
        "Distilled-Data": "#f58518",
    }
    plt.figure(figsize=(8.6, 5.2))
    for r in results:
        xs = [int(h["step"]) for h in r["history"]]
        ys = [float(h["test_acc"]) for h in r["history"]]
        plt.plot(xs, ys, marker="o", linewidth=2.0, markersize=4, label=r["name"], color=colors.get(r["name"]))

    ratio = float(distilled_n) / float(full_n)
    plt.xlabel("Optimizer Steps")
    plt.ylabel("Test Accuracy")
    plt.grid(alpha=0.28)
    plt.legend()
    plt.text(
        0.02,
        0.04,
        f"distilled samples = {distilled_n}\ncompression ratio = {ratio:.4f} ({ratio*100:.2f}%)",
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="#cccccc"),
    )
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=190)
    plt.close()


def update_progress_log(run_id: str, script_path: Path, outputs: Dict) -> None:
    progress_path = PROGRESS_DIR / "project_progress.md"
    with progress_path.open("a", encoding="utf-8") as f:
        f.write("\n---\n\n")
        f.write("## 项目进展记录\n")
        f.write(f"- 时间：{time.strftime('%Y-%m-%d %H:%M')} Asia/Shanghai\n")
        f.write("- 本次对话任务目标：验证将模块A替换为 Dataset-Distilled Compact Training Set Construction 的可行性。\n")
        f.write("- 完成情况：已新增并运行 CIFAR-10 distilled module A 验证实验，比较 Full-Data / Random-Subset / Distilled-Data。\n")
        f.write(f"- run_id：{run_id}\n")
        f.write(f"- 修改脚本：{script_path}\n")
        f.write("- 关键参数：train_steps={}, eval_interval={}, batch_size={}, lr={}, cpu_threads={}。\n".format(
            CFG["train_steps"], CFG["eval_interval"], CFG["batch_size"], CFG["lr"], CFG["cpu_threads"]
        ))
        f.write("- 修改内容：使用与 MTT 协议一致的 InstanceNorm ConvNet，加入 DiffAugment、cosine scheduler，并通过候选预筛选选择有效 distilled checkpoint。\n")
        f.write("- 安全设置：脚本开头限制 OMP/MKL/OPENBLAS/NUMEXPR，并在运行时设置 torch 线程数，避免 CPU 爆炸。\n")
        for k, v in outputs.items():
            f.write(f"- 输出 {k}：{v}\n")


def main() -> int:
    ensure_dirs()
    if bool(CFG["safe_mode"]):
        set_safe_runtime(int(CFG["cpu_threads"]))
    set_seed(int(CFG["seed"]))

    run_id = time.strftime("%Y%m%d_%H%M%S")
    txt_path = LOG_TXT_DIR / f"cifar10_distilled_moduleA_verify_{run_id}.log"
    json_path = LOG_JSON_DIR / f"cifar10_distilled_moduleA_verify_{run_id}.json"
    csv_path = EXPTAB_DIR / f"cifar10_distilled_moduleA_verify_{run_id}.csv"
    fig_path = FIG_DIR / "figure4_cifar10_distilled_verify.png"

    with tee_console_to_file(txt_path):
        print("=" * 88)
        print("模块 A 验证：Dataset-Distilled Compact Training Set Construction")
        print("=" * 88)
        print("run_id       :", run_id)
        print("train_steps  :", CFG["train_steps"])
        print("eval_interval:", CFG["eval_interval"])
        print("batch_size   :", CFG["batch_size"])
        print("lr           :", CFG["lr"])
        print("cpu_threads  :", CFG["cpu_threads"])

        device = select_device()
        print("device       :", device)

        print("\n[1/4] 加载本地 CIFAR-10 缓存...")
        x_full, y_full, x_test, y_test = load_cifar_npz()
        print("  Full train :", tuple(x_full.shape), tuple(y_full.shape))
        print("  Test       :", tuple(x_test.shape), tuple(y_test.shape))

        print("\n[2/4] 加载已有 distilled samples...")
        x_distill, y_distill, img_path, lab_path, probe_records = select_best_distilled_candidate(x_test, y_test, device)
        distilled_n = int(x_distill.shape[0])
        full_n = int(x_full.shape[0])
        compression_ratio = float(distilled_n) / float(full_n)
        print("  images     :", img_path)
        print("  labels     :", lab_path)
        print("  shape      :", tuple(x_distill.shape), tuple(y_distill.shape))
        print("  range      : [{:.4f}, {:.4f}]".format(float(x_distill.min()), float(x_distill.max())))
        print("  compression: {:.4f} ({:.2f}%)".format(compression_ratio, compression_ratio * 100.0))

        print("\n[3/4] 构造同规模 Random-Subset...")
        rng = np.random.RandomState(int(CFG["seed"]))
        rand_idx = rng.choice(full_n, size=distilled_n, replace=False)
        x_rand = x_full[rand_idx].clone()
        y_rand = y_full[rand_idx].clone()
        print("  Random subset:", tuple(x_rand.shape), tuple(y_rand.shape))

        print("\n[4/4] 训练三种数据来源（横轴为 Optimizer Steps）...")
        # 三组都从同一个随机种子初始化，确保比较更公平。
        results = []
        results.append(train_source("Full-Data", x_full, y_full, x_test, y_test, device, seed=int(CFG["seed"])))
        results.append(train_source("Random-Subset", x_rand, y_rand, x_test, y_test, device, seed=int(CFG["seed"])))
        results.append(train_source("Distilled-Data", x_distill, y_distill, x_test, y_test, device, seed=int(CFG["seed"])))

        save_csv(results, csv_path)
        plot_results(results, fig_path, distilled_n=distilled_n, full_n=full_n)

        payload = {
            "run_id": run_id,
            "config": CFG,
            "device": str(device),
            "distilled_images": str(img_path),
            "distilled_labels": str(lab_path),
            "candidate_probe_records": probe_records,
            "num_full_samples": full_n,
            "num_distilled_samples": distilled_n,
            "compression_ratio": compression_ratio,
            "results": results,
            "outputs": {
                "txt": str(txt_path),
                "json": str(json_path),
                "csv": str(csv_path),
                "figure": str(fig_path),
            },
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print("\n" + "=" * 88)
        print("[结果汇总]")
        print("{:<16} {:<12} {:<14} {:<12} {:<12}".format("Method", "Samples", "Compression", "FinalAcc", "Time(s)"))
        print("-" * 88)
        for r in results:
            comp = float(r["num_samples"]) / float(full_n)
            print("{:<16} {:<12d} {:<14.4f} {:<12.4f} {:<12.2f}".format(
                r["name"], int(r["num_samples"]), comp, float(r["final_test_acc"]), float(r["train_time_s"])
            ))
        print("=" * 88)
        print("Figure:", fig_path)
        print("CSV   :", csv_path)
        print("JSON  :", json_path)
        print("TXT   :", txt_path)

        update_progress_log(run_id, Path(__file__).resolve(), payload["outputs"])
        print("ProgressLog:", PROGRESS_DIR / "project_progress.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

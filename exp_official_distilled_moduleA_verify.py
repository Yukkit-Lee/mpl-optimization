#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：使用官网发布的 MTT ipc=50 蒸馏数据验证新的模块 A：
Dataset-Distilled Compact Training Set Construction。

实验只验证模块 A，不接入模块 B/KD 初始化，也不接入模块 C/DP 私有优化。
对每个数据集比较三种训练数据来源：
1) Full-Data：完整真实训练集采样训练
2) Random-Subset：从真实训练集中随机抽取与官方 distilled samples 同规模的子集
3) Official-Distilled：官网 MTT ipc=50 蒸馏样本

三者使用同一个 MTT ConvNet、同一 optimizer/lr，从随机初始化开始训练。
横轴是 Optimizer Steps，纵轴是 Test Accuracy；不同数据集使用不同 iteration 终点以贴近 PEA Figure 4 风格。
主图不标 epsilon，专注证明 distilled samples 用极少数据量保留主要训练效用。

输出：
- TXT 日志：log/txt/officialDistillModuleA_{dataset}_{run_id}.log
- JSON 结果：log/json/officialDistillModuleA_{dataset}_{run_id}.json
- CSV 表格：expTab/officialDistillModuleA_{dataset}_{run_id}.csv
- 曲线图：expFig/figure4_distill_official/figure4_official_distill_{dataset}.png
"""

import os
# 尽早限制 CPU 线程，避免 torchvision / BLAS 过度并行。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
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
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
MTT_REPO = Path("/mnt/code/lixujie/mtt-distillation")
OFFICIAL_ROOT = Path("/home/lixujie/tmpWork/mtt_data/georgecazenavette")
LOG_TXT_DIR = BASE_DIR / "log" / "txt"
LOG_JSON_DIR = BASE_DIR / "log" / "json"
EXPTAB_DIR = BASE_DIR / "expTab"
EXPFIG_DIR = BASE_DIR / "expFig" / "figure4_distill_official"
PROGRESS_PATH = BASE_DIR / "progressLog" / "project_progress.md"

CFG = {
    "seed": 42,
    "cpu_threads": 1,
    "ipc": 50,
    # 默认值仅作为 CLI fallback；正式绘图按 DATASET_CFG 中每个数据集自己的 iteration 终点。
    "train_steps": 1000,
    "eval_interval": 100,
    # Figure 4 主图先不标 epsilon；隐私预算留给后续 DP 版本表格/附图。
    "plot_epsilon_annotations": False,
    "batch_size": 128,
    "test_batch_size": 256,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "use_diff_augment": True,
    "use_cosine_scheduler": True,
    "device": "auto",
    # Tiny 全量训练较重，仍然按 optimizer steps 采样，不完整扫 epoch。
    "datasets": ["cifar10", "cifar100", "tiny"],
}

DATASET_CFG = {
    "cifar10": {
        "official_dir": OFFICIAL_ROOT / "cifar10_50",
        "mtt_dataset": "CIFAR10",
        "model": "ConvNetD3",
        "data_path": Path("/home/lixujie/tmpWork/mtt_cifar10_work/data"),
        "num_classes": 10,
        "full_train_size": 50000,
        "fig_name": "figure4_official_distill_cifar10.png",
        "train_steps": 900,
        "eval_interval": 100,
    },
    "cifar100": {
        "official_dir": OFFICIAL_ROOT / "cifar100_50",
        "mtt_dataset": "CIFAR100",
        "model": "ConvNetD3",
        "data_path": Path("/home/lixujie/tmpWork/mtt_cifar100_work/data"),
        "num_classes": 100,
        "full_train_size": 50000,
        "fig_name": "figure4_official_distill_cifar100.png",
        "train_steps": 800,
        "eval_interval": 100,
    },
    "tiny": {
        "official_dir": OFFICIAL_ROOT / "tiny_50",
        "mtt_dataset": "Tiny",
        "model": "ConvNetD4",
        "data_path": Path("/home/lixujie/tmpWork/mtt_tiny_work/data/tiny-imagenet-200"),
        "num_classes": 200,
        "full_train_size": 100000,
        "fig_name": "figure4_official_distill_tiny.png",
        # Tiny ImageNet 任务更难，400 steps 还明显没到较好状态；这里按实际曲线延长训练。
        "train_steps": 1200,
        "eval_interval": 100,
    },
}


def ensure_dirs() -> None:
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    EXPTAB_DIR.mkdir(parents=True, exist_ok=True)
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
    torch.backends.cudnn.benchmark = True


def select_device() -> torch.device:
    hint = str(CFG["device"]).lower()
    if hint == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            dev = torch.device("cuda")
            _ = torch.randn(8, 8, device=dev) @ torch.randn(8, 8, device=dev)
            return dev
        except Exception as e:
            print(f"[设备回退] CUDA 不可用，改用 CPU。原因: {e}")
    return torch.device("cpu")


def add_mtt_path() -> None:
    if str(MTT_REPO) not in sys.path:
        sys.path.insert(0, str(MTT_REPO))


def make_mtt_args(dataset: str, device: torch.device):
    from argparse import Namespace
    from utils import ParamDiffAug, get_daparam
    cfg = DATASET_CFG[dataset]
    args = Namespace()
    args.dataset = cfg["mtt_dataset"]
    args.model = cfg["model"]
    args.ipc = int(CFG["ipc"])
    args.device = str(device)
    args.zca = False
    args.res = None
    args.subset = "imagenette"
    args.batch_train = int(CFG["batch_size"])
    args.batch_real = int(CFG["batch_size"])
    args.lr_net = float(CFG["lr"])
    args.epoch_eval_train = 0
    args.dsa = True
    args.dsa_strategy = "color_crop_cutout_flip_scale_rotate"
    args.dsa_param = ParamDiffAug()
    args.dc_aug_param = get_daparam(args.dataset, cfg["model"], cfg["model"], int(CFG["ipc"]))
    args.canvas_size = 2
    args.canvas_samples = 1
    args.texture = False
    return args


def load_official_distilled(dataset: str) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    folder = DATASET_CFG[dataset]["official_dir"]
    image_files = sorted(folder.glob("images_best*.pt"))
    label_files = sorted(folder.glob("labels_best*.pt"))
    if not image_files or len(image_files) != len(label_files):
        raise FileNotFoundError(f"官方蒸馏数据文件不完整: {folder}")
    imgs, labs = [], []
    for ip, lp in zip(image_files, label_files):
        x = torch.load(str(ip), map_location="cpu").detach().float()
        y = torch.load(str(lp), map_location="cpu").detach().long().view(-1)
        if x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        imgs.append(x)
        labs.append(y)
    return torch.cat(imgs, 0), torch.cat(labs, 0), [p.name for p in image_files], [p.name for p in label_files]


def dataset_batch(ds, indices: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i in indices:
        x, y = ds[int(i)]
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs, 0).to(device), torch.tensor(ys, dtype=torch.long, device=device)


def tensor_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, x.shape[0], (batch_size,), device=torch.device("cpu"))
    return x[idx].to(device), y[idx].to(device)


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    if not CFG["use_diff_augment"]:
        return x
    # 与 MTT 常用 color/crop/flip 等策略保持轻量近似：flip + random crop。
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


def evaluate(model, testloader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in testloader:
            xb = xb.float().to(device)
            yb = yb.long().to(device)
            pred = model(xb).argmax(1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
    return float(correct / max(1, total))


def train_source(name: str, source: Dict, dataset: str, testloader, channel, num_classes, im_size, device: torch.device, seed: int, steps: int, interval: int) -> Dict:
    from utils import get_network
    set_safe_runtime(seed)
    model = get_network(DATASET_CFG[dataset]["model"], channel, num_classes, im_size, dist=False).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=float(CFG["lr"]), momentum=float(CFG["momentum"]), weight_decay=float(CFG["weight_decay"]))
    steps = int(steps)
    interval = int(interval)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps)) if CFG["use_cosine_scheduler"] else None
    eval_points = set(range(0, steps + 1, interval))
    eval_points.add(steps)
    history = []
    t0 = time.time()
    init_acc = evaluate(model, testloader, device)
    history.append({"step": 0, "test_acc": init_acc, "loss": None})

    rng = np.random.RandomState(seed + 123)
    last_loss = None
    for step in range(1, steps + 1):
        model.train()
        if source["kind"] == "tensor":
            xb, yb = tensor_batch(source["x"], source["y"], int(CFG["batch_size"]), device)
        else:
            pool = source["indices"]
            chosen = rng.choice(pool, size=int(CFG["batch_size"]), replace=True)
            xb, yb = dataset_batch(source["dataset"], chosen, device)
        xb = diff_augment(xb.float())
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb.long())
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        last_loss = float(loss.detach().cpu().item())
        if step in eval_points:
            acc = evaluate(model, testloader, device)
            history.append({"step": int(step), "test_acc": acc, "loss": last_loss})
            print(f"  {name:<18} step={step:>5}/{steps}  test_acc={acc:.4f}  loss={last_loss:.4f}")
    return {
        "name": name,
        "num_samples": int(source["num_samples"]),
        "history": history,
        "final_test_acc": float(history[-1]["test_acc"]),
        "train_time_s": float(time.time() - t0),
    }


def plot_results(dataset: str, results: List[Dict], distilled_n: int, full_n: int, fig_path: Path, max_step: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Full-Data": "#4c78a8", "Random-Subset": "#9aa0a6", "Official-Distilled": "#f58518"}
    # 固定画布尺寸和 dpi，保证三个数据集输出图大小一致。
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=190)
    for r in results:
        xs = [h["step"] for h in r["history"] if int(h["step"]) <= max_step]
        ys = [h["test_acc"] for h in r["history"] if int(h["step"]) <= max_step]
        ax.plot(xs, ys, marker="o", linewidth=2.1, markersize=4, color=colors.get(r["name"]), label=r["name"])

    ax.set_xlabel("Optimizer Steps")
    ax.set_ylabel("Test Accuracy")
    ax.set_xlim(left=-0.04 * max_step, right=1.04 * max_step)
    ax.grid(alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fig_path))
    plt.close(fig)


def save_csv(results: List[Dict], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Step", "TestAcc", "Loss", "NumSamples", "TrainTimeSec"])
        for r in results:
            for h in r["history"]:
                w.writerow([r["name"], h["step"], f"{h['test_acc']:.6f}", "" if h["loss"] is None else f"{h['loss']:.6f}", r["num_samples"], f"{r['train_time_s']:.3f}"])


def run_dataset(dataset: str) -> Dict:
    add_mtt_path()
    from utils import get_dataset
    cfg = DATASET_CFG[dataset]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_TXT_DIR / f"officialDistillModuleA_{dataset}_{run_id}.log"
    json_path = LOG_JSON_DIR / f"officialDistillModuleA_{dataset}_{run_id}.json"
    csv_path = EXPTAB_DIR / f"officialDistillModuleA_{dataset}_{run_id}.csv"
    fig_path = EXPFIG_DIR / cfg["fig_name"]

    with tee_console_to_file(log_path):
        print("=" * 88)
        print("官方 MTT ipc=50 蒸馏数据：模块 A 验证实验")
        print("=" * 88)
        print("dataset      :", dataset)
        print("run_id       :", run_id)
        steps = int(cfg.get("train_steps", CFG["train_steps"]))
        interval = int(cfg.get("eval_interval", CFG["eval_interval"]))
        print("steps        :", steps)
        print("eval_interval:", interval)
        print("cpu_threads  :", CFG["cpu_threads"])
        device = select_device()
        print("device       :", device)

        args = make_mtt_args(dataset, device)
        print("\n[1/4] 加载 MTT 标准真实数据与测试集...")
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, *_ = get_dataset(cfg["mtt_dataset"], str(cfg["data_path"]), batch_size=int(CFG["batch_size"]), subset=args.subset, args=args)
        full_n = len(dst_train)
        print("  train_size:", full_n, " test_size:", len(dst_test), " im_size:", im_size, " classes:", num_classes)

        print("\n[2/4] 加载官方 ipc=50 蒸馏数据...")
        x_dist, y_dist, img_files, lab_files = load_official_distilled(dataset)
        distilled_n = int(x_dist.shape[0])
        print("  images:", tuple(x_dist.shape), img_files)
        print("  labels:", tuple(y_dist.shape), lab_files)
        print("  range : [{:.4f}, {:.4f}]".format(float(x_dist.min()), float(x_dist.max())))
        print("  compression ratio: {:.4f} ({:.2f}%)".format(distilled_n / full_n, distilled_n / full_n * 100.0))

        print("\n[3/4] 构造 Full / Random / Official-Distilled 三组 source...")
        rng = np.random.RandomState(int(CFG["seed"]))
        all_indices = np.arange(full_n)
        rand_indices = rng.choice(full_n, size=distilled_n, replace=False)
        sources = [
            ("Full-Data", {"kind": "dataset", "dataset": dst_train, "indices": all_indices, "num_samples": full_n}),
            ("Random-Subset", {"kind": "dataset", "dataset": dst_train, "indices": rand_indices, "num_samples": distilled_n}),
            ("Official-Distilled", {"kind": "tensor", "x": x_dist, "y": y_dist, "num_samples": distilled_n}),
        ]

        print("\n[4/4] 开始训练并记录曲线...")
        results = []
        for name, source in sources:
            results.append(train_source(name, source, dataset, testloader, channel, num_classes, im_size, device, seed=int(CFG["seed"]), steps=steps, interval=interval))

        save_csv(results, csv_path)
        plot_results(dataset, results, distilled_n, full_n, fig_path, max_step=steps)
        payload = {"run_id": run_id, "dataset": dataset, "config": CFG, "official_dir": str(cfg["official_dir"]), "image_files": img_files, "label_files": lab_files, "num_full_samples": full_n, "num_distilled_samples": distilled_n, "compression_ratio": distilled_n / full_n, "train_steps_effective": steps, "eval_interval_effective": interval, "figure_note": "Figure 4 main plot does not annotate epsilon; DP privacy budget should be reported in a separate DP table/appendix figure.", "results": results, "outputs": {"txt": str(log_path), "json": str(json_path), "csv": str(csv_path), "figure": str(fig_path)}}
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n[结果汇总]")
        for r in results:
            print(f"  {r['name']:<18} final_acc={r['final_test_acc']:.4f}  time={r['train_time_s']:.1f}s  samples={r['num_samples']}")
        print("图像:", fig_path)
        print("JSON:", json_path)
        print("CSV :", csv_path)
        return payload


def update_progress(datasets: List[str], outputs: List[Dict]) -> None:
    with PROGRESS_PATH.open("a", encoding="utf-8") as f:
        f.write("\n---\n\n")
        f.write("## 项目进展记录\n")
        f.write(f"- 时间：{time.strftime('%Y-%m-%d %H:%M')} Asia/Shanghai\n")
        f.write("- 本次对话任务目标：使用官网高精度 ipc=50 蒸馏数据重新做模块 A 验证实验，覆盖 CIFAR-10、CIFAR-100、Tiny ImageNet。\n")
        f.write("- 实验设计：每个数据集比较 Full-Data、Random-Subset、Official-Distilled 三种训练数据来源，横轴为 Optimizer Steps，纵轴为 Test Accuracy。\n")
        f.write(f"- 安全设置：CPU 线程限制为 {CFG['cpu_threads']}，使用 distillation 环境和单卡 GPU，避免 CPU 过载。\n")
        f.write(f"- 关键参数：ipc=50，不同数据集使用不同 iteration 终点；batch_size={CFG['batch_size']}, lr={CFG['lr']}。\n")
        f.write("- 绘图规范：Figure 4 主图不标 epsilon，不显示 dataset/distilled samples/ratio 说明框；三张图固定相同画布尺寸。\n")
        for out in outputs:
            acc_text = ", ".join([f"{r['name']}={r['final_test_acc']:.4f}" for r in out["results"]])
            f.write(f"- {out['dataset']}：train_steps={out['train_steps_effective']}，distilled_samples={out['num_distilled_samples']}，compression_ratio={out['compression_ratio']:.4f}；final_acc: {acc_text}；输出图={out['outputs']['figure']}；JSON={out['outputs']['json']}。\n")


def parse_args():
    p = argparse.ArgumentParser(description="官方 MTT ipc=50 蒸馏数据模块 A 验证")
    p.add_argument("--dataset", default="all", choices=["all", "cifar10", "cifar100", "tiny"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    return p.parse_args()


def main() -> int:
    ensure_dirs()
    args = parse_args()
    if args.steps is not None:
        CFG["train_steps"] = int(args.steps)
        for _cfg in DATASET_CFG.values():
            _cfg["train_steps"] = int(args.steps)
    if args.eval_interval is not None:
        CFG["eval_interval"] = int(args.eval_interval)
        for _cfg in DATASET_CFG.values():
            _cfg["eval_interval"] = int(args.eval_interval)
    set_safe_runtime(int(CFG["seed"]))
    datasets = CFG["datasets"] if args.dataset == "all" else [args.dataset]
    outputs = []
    for ds in datasets:
        outputs.append(run_dataset(ds))
    update_progress(datasets, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

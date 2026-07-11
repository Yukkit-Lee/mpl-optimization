# -*- coding: utf-8 -*-
"""
用途：验证官网发布的 MTT 蒸馏数据在标准测试集上的 test accuracy。

本脚本只做“官方蒸馏数据质量检查”，不改动现有模块 A/B/C。
数据来源默认：/home/lixujie/tmpWork/mtt_data/georgecazenavette
结果输出默认：/mnt/code/lixujie/tfe_experiments/official_data

评估口径：尽量复用 MTT 官方仓库中的 get_dataset / get_network / evaluate_synset。
注意：当前用户确认下载的官方数据全部为非 ZCA 版本，因此本脚本统一使用 standard/non-ZCA 预处理。
"""

import os
# 必须尽早限制 CPU 线程，避免 DataLoader / BLAS 把服务器打满。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import csv
import json
import random
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path("/mnt/code/lixujie/tfe_experiments")
MTT_REPO = Path("/mnt/code/lixujie/mtt-distillation")
OFFICIAL_ROOT = Path("/home/lixujie/tmpWork/mtt_data/georgecazenavette")
OUTPUT_DIR = PROJECT_DIR / "official_data"
TMP_DATA_ROOT = Path("/home/lixujie/tmpWork")

# 正式验证配置：如需更快冒烟，可临时改小 EVAL_EPOCHS。
EVAL_EPOCHS = 300
NUM_EVAL = 1
BATCH_TRAIN_DEFAULT = 256
SEED = 42

# 用户已确认官方数据全部为非 ZCA；保留该开关仅用于以后排查预处理问题。
AUTO_PREPROCESS_CIFAR = False

DATA_CONFIG = {
    "cifar10": {
        "mtt_dataset": "CIFAR10",
        "model": "ConvNetD3",
        "data_path": TMP_DATA_ROOT / "mtt_cifar10_work" / "data",
        "preprocess_candidates": [False],
        "batch_train": 256,
    },
    "cifar100": {
        "mtt_dataset": "CIFAR100",
        "model": "ConvNetD3",
        "data_path": TMP_DATA_ROOT / "mtt_cifar100_work" / "data",
        "preprocess_candidates": [False],
        "batch_train": 256,
    },
    "tiny": {
        "mtt_dataset": "Tiny",
        "model": "ConvNetD4",
        "data_path": TMP_DATA_ROOT / "mtt_tiny_work" / "data" / "tiny-imagenet-200",
        "preprocess_candidates": [False],
        "batch_train": 256,
    },
}


def set_safe_runtime(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True


def add_mtt_repo_to_path():
    if str(MTT_REPO) not in sys.path:
        sys.path.insert(0, str(MTT_REPO))


def parse_dataset_name(folder_name):
    # 例如 cifar10_50 / cifar100_10 / tiny_50
    parts = folder_name.lower().split("_")
    if len(parts) < 2:
        raise ValueError("目录名无法解析 dataset/ipc: %s" % folder_name)
    dataset_key = parts[0]
    ipc = int(parts[1])
    if dataset_key not in DATA_CONFIG:
        raise ValueError("未知官方蒸馏数据目录: %s" % folder_name)
    return dataset_key, ipc


def load_synthetic(folder):
    """读取官方 images_best/labels_best；Tiny ipc=50 这种分片目录会自动拼接。"""
    folder = Path(folder)
    image_files = sorted(folder.glob("images_best*.pt"))
    label_files = sorted(folder.glob("labels_best*.pt"))
    if not image_files or not label_files or len(image_files) != len(label_files):
        raise FileNotFoundError("缺少 images_best/labels_best 或数量不匹配: %s" % folder)

    images_list = []
    labels_list = []
    for img_path, lab_path in zip(image_files, label_files):
        images = torch.load(str(img_path), map_location="cpu")
        labels = torch.load(str(lab_path), map_location="cpu")
        images_list.append(images.detach().float().cpu())
        labels_list.append(labels.detach().long().cpu())

    images_all = torch.cat(images_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    return images_all, labels_all, [p.name for p in image_files], [p.name for p in label_files]


def make_args(dataset_name, model, zca, device, batch_train, ipc):
    # evaluate_synset 只会用到其中一部分字段；这里补齐常见字段，保证和官方脚本兼容。
    from utils import ParamDiffAug, get_daparam

    args = Namespace()
    args.dataset = dataset_name
    args.model = model
    args.ipc = ipc
    args.device = device
    args.zca = bool(zca)
    args.res = None
    args.subset = "imagenette"
    args.batch_train = int(batch_train)
    args.batch_real = int(batch_train)
    args.lr_net = 0.01
    args.epoch_eval_train = int(EVAL_EPOCHS)
    args.dsa = True
    args.dsa_strategy = "color_crop_cutout_flip_scale_rotate"
    args.dsa_param = ParamDiffAug()
    args.dc_aug_param = get_daparam(dataset_name, model, model, ipc)
    args.canvas_size = 2
    args.canvas_samples = 1
    args.texture = False
    return args


def get_dataset_bundle(cfg, args):
    from utils import get_dataset

    cfg["data_path"].mkdir(parents=True, exist_ok=True)
    return get_dataset(
        cfg["mtt_dataset"],
        str(cfg["data_path"]),
        batch_size=args.batch_real,
        subset=args.subset,
        args=args,
    )


def evaluate_one(folder, zca, run_id, device):
    add_mtt_repo_to_path()
    from utils import evaluate_synset, get_network

    dataset_key, ipc = parse_dataset_name(folder.name)
    cfg = DATA_CONFIG[dataset_key]
    images, labels, image_files, label_files = load_synthetic(folder)

    args = make_args(
        dataset_name=cfg["mtt_dataset"],
        model=cfg["model"],
        zca=zca,
        device=device,
        batch_train=cfg["batch_train"],
        ipc=ipc,
    )

    t0 = time.time()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, *_ = get_dataset_bundle(cfg, args)

    accs = []
    for eval_id in range(NUM_EVAL):
        set_safe_runtime(SEED + eval_id)
        net = get_network(cfg["model"], channel, num_classes, im_size).to(device)
        _, _, acc_test = evaluate_synset(
            eval_id,
            net,
            images.clone(),
            labels.clone(),
            testloader,
            args,
            return_loss=False,
            texture=False,
        )
        accs.append(float(acc_test))

    elapsed = time.time() - t0
    hist = torch.bincount(labels.cpu(), minlength=num_classes)
    nonzero = int((hist > 0).sum().item())
    return {
        "run_id": run_id,
        "folder": folder.name,
        "dataset_key": dataset_key,
        "mtt_dataset": cfg["mtt_dataset"],
        "model": cfg["model"],
        "ipc": ipc,
        "zca": bool(zca),
        "preprocess": "zca" if zca else "standard",
        "num_eval": NUM_EVAL,
        "eval_epochs": EVAL_EPOCHS,
        "num_synthetic": int(images.shape[0]),
        "num_classes_seen": nonzero,
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "label_min": int(labels.min().item()),
        "label_max": int(labels.max().item()),
        "image_files": image_files,
        "label_files": label_files,
        "test_acc_mean": float(np.mean(accs)),
        "test_acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "test_acc_runs": accs,
        "elapsed_s": float(elapsed),
        "device": device,
    }


def discover_folders(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError("官方蒸馏数据目录不存在: %s" % root)
    folders = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and list(p.glob("images_best*.pt")):
            folders.append(p)
    return folders


def save_outputs(run_id, selected_rows, all_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / ("official_mtt_eval_%s.csv" % run_id)
    json_path = OUTPUT_DIR / ("official_mtt_eval_%s.json" % run_id)
    txt_path = OUTPUT_DIR / ("official_mtt_eval_%s.log" % run_id)

    fieldnames = [
        "folder", "dataset_key", "ipc", "preprocess", "zca", "model",
        "num_synthetic", "num_classes_seen", "eval_epochs", "num_eval",
        "test_acc_mean", "test_acc_std", "elapsed_s", "device",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in selected_rows:
            w.writerow({k: row.get(k) for k in fieldnames})

    payload = {
        "run_id": run_id,
        "official_root": str(OFFICIAL_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "eval_epochs": EVAL_EPOCHS,
        "num_eval": NUM_EVAL,
        "selected_results": selected_rows,
        "all_candidate_results": all_rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("官方 MTT 蒸馏数据验证结果")
    lines.append("run_id: %s" % run_id)
    lines.append("root  : %s" % OFFICIAL_ROOT)
    lines.append("eval  : epochs=%d, num_eval=%d" % (EVAL_EPOCHS, NUM_EVAL))
    lines.append("")
    lines.append("%-14s %-8s %-10s %-12s %-10s %-10s %-10s" % (
        "Folder", "IPC", "Preproc", "Samples", "Classes", "Acc", "Time(s)"))
    lines.append("-" * 84)
    for r in selected_rows:
        lines.append("%-14s %-8d %-10s %-12d %-10d %.4f     %.1f" % (
            r["folder"], r["ipc"], r["preprocess"], r["num_synthetic"],
            r["num_classes_seen"], r["test_acc_mean"], r["elapsed_s"]
        ))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, txt_path


def main():
    set_safe_runtime(SEED)
    add_mtt_repo_to_path()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 88)
    print("官方 MTT 蒸馏数据 test accuracy 验证")
    print("run_id       : %s" % run_id)
    print("official_root: %s" % OFFICIAL_ROOT)
    print("output_dir   : %s" % OUTPUT_DIR)
    print("device       : %s" % device)
    print("eval_epochs  : %d" % EVAL_EPOCHS)
    print("num_eval     : %d" % NUM_EVAL)
    print("cpu_threads  : 1")
    print("=" * 88)

    folders = discover_folders(OFFICIAL_ROOT)
    if not folders:
        raise RuntimeError("没有找到任何 images_best*.pt 官方数据")

    selected_rows = []
    all_rows = []
    for folder in folders:
        dataset_key, ipc = parse_dataset_name(folder.name)
        cfg = DATA_CONFIG[dataset_key]
        candidates = cfg["preprocess_candidates"]
        print("\n[验证] %s  candidates=%s" % (folder.name, ["zca" if c else "standard" for c in candidates]))

        candidate_rows = []
        for zca in candidates:
            print("  -> preprocess=%s" % ("zca" if zca else "standard"))
            row = evaluate_one(folder, zca=zca, run_id=run_id, device=device)
            candidate_rows.append(row)
            all_rows.append(row)
            print("     acc=%.4f  samples=%d  time=%.1fs" % (
                row["test_acc_mean"], row["num_synthetic"], row["elapsed_s"]))

        best = max(candidate_rows, key=lambda r: r["test_acc_mean"])
        selected_rows.append(best)
        print("  ✓ selected=%s acc=%.4f" % (best["preprocess"], best["test_acc_mean"]))

    csv_path, json_path, txt_path = save_outputs(run_id, selected_rows, all_rows)

    print("\n" + "=" * 88)
    print("[汇总] 官方蒸馏数据验证结果")
    print("%-14s %-8s %-10s %-12s %-10s %-10s" % (
        "Folder", "IPC", "Preproc", "Samples", "Classes", "Acc"))
    print("-" * 76)
    for r in selected_rows:
        print("%-14s %-8d %-10s %-12d %-10d %.4f" % (
            r["folder"], r["ipc"], r["preprocess"], r["num_synthetic"],
            r["num_classes_seen"], r["test_acc_mean"]
        ))
    print("-" * 76)
    print("CSV : %s" % csv_path)
    print("JSON: %s" % json_path)
    print("LOG : %s" % txt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
用途说明：
本脚本用于在固定 MNIST 数据划分下做 HOG 参数消融实验，
比较不同 `pixels_per_cell` 设置对以下指标的影响：
1) 特征维度
2) 压缩比
3) 测试精度
4) 训练时间

默认对比：
- pixels_per_cell=(7,7)
- pixels_per_cell=(4,4)
"""

import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import mnist_hog as base


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
# ====== 训练日志打印控制（控制 epoch 输出条数的总开关）======
# 例子：
# - 1000 个 epoch + MAX_EPOCH_LOG_LINES=10  -> 大约每 100 轮打印一次
# - 1000 个 epoch + MAX_EPOCH_LOG_LINES=20  -> 大约每  50 轮打印一次
MAX_EPOCH_LOG_LINES = 10


def run_one_setting(x_train_img, x_test_img, y_train_int, y_test_int, pixels_per_cell, log):
    """在单个 HOG 设置下完整执行一次：提特征 + TFE 训练 + 评估。"""
    raw_dim = int(x_train_img.shape[1] * x_train_img.shape[2])

    t_hog = time.time()
    x_train_hog = base.extract_hog_features(
        x_train_img,
        orientations=base.HOG_ORIENTATIONS,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=base.HOG_CELLS_PER_BLOCK,
        block_norm=base.HOG_BLOCK_NORM,
        log_fn=log,
    )
    x_test_hog = base.extract_hog_features(
        x_test_img,
        orientations=base.HOG_ORIENTATIONS,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=base.HOG_CELLS_PER_BLOCK,
        block_norm=base.HOG_BLOCK_NORM,
        log_fn=log,
    )
    hog_time = time.time() - t_hog

    hog_dim = int(x_train_hog.shape[1])
    compression = float(hog_dim / raw_dim)

    # 明文侧标准化，与主实验保持一致
    scaler = StandardScaler()
    x_train_hog = scaler.fit_transform(x_train_hog).astype(np.float32)
    x_test_hog = scaler.transform(x_test_hog).astype(np.float32)

    sparse_log = make_sparse_epoch_logger(
        log, total_epochs=base.EPOCHS, max_lines=MAX_EPOCH_LOG_LINES
    )
    history, test_acc, checks, train_time = base.run_tfe_multiclass(
        x_train_feat=x_train_hog,
        y_train_int=y_train_int,
        x_test_feat=x_test_hog,
        y_test_int=y_test_int,
        n_classes=base.N_CLASSES,
        epochs=base.EPOCHS,
        lr=base.LR,
        log_fn=sparse_log,
    )
    train_acc = float(history[-1]["train_accuracy"]) if history else 0.0
    final_loss = float(history[-1]["loss"]) if history else None

    return {
        "hog_setting": {
            "orientations": base.HOG_ORIENTATIONS,
            "pixels_per_cell": list(pixels_per_cell),
            "cells_per_block": list(base.HOG_CELLS_PER_BLOCK),
            "block_norm": base.HOG_BLOCK_NORM,
        },
        "feature_dim": hog_dim,
        "compression_ratio": compression,
        "hog_extract_time_s": hog_time,
        "train_time_s": train_time,
        "final_train_acc": train_acc,
        "final_test_acc": float(test_acc),
        "final_train_loss": final_loss,
        "checks": checks,
        "history": history,
    }


def make_sparse_epoch_logger(log_fn, total_epochs, max_lines=MAX_EPOCH_LOG_LINES):
    """
    只打印最多 max_lines 条 epoch 行，减少长训练日志噪音。
    例如 total_epochs=1000 且 max_lines=10，则按 %100 打印。
    """
    # 自动计算打印间隔：
    # interval = ceil(总epoch / 最多打印条数)
    interval = max(1, int(math.ceil(float(total_epochs) / float(max_lines))))

    def wrapped(msg=""):
        text = str(msg)
        parts = text.strip().split()
        if parts and parts[0].isdigit():
            ep = int(parts[0])
            if ep % interval != 0 and ep != total_epochs:
                return
        log_fn(text)

    return wrapped


def log_aligned_ablation_table(log, rows):
    """打印对齐后的消融表，避免字段串行。"""
    headers = [
        ("HOG设置", 28),
        ("特征维度", 8),
        ("压缩比", 8),
        ("TestAcc", 8),
        ("训练时间(s)", 10),
    ]
    line = " | ".join([f"{name:<{width}}" for name, width in headers])
    sep = "-+-".join(["-" * width for _, width in headers])
    log(line)
    log(sep)
    for row in rows:
        log(
            f"{row[0]:<28} | {row[1]:>8} | {row[2]:>8} | {row[3]:>8} | {row[4]:>10}"
        )


def main():
    base.ensure_dirs()
    base.configure_logging()
    np.random.seed(base.RANDOM_SEED)
    base.tf.compat.v1.set_random_seed(base.RANDOM_SEED)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"exp3_step2_hog_ablation_{run_ts}.log"
    report_file = LOG_DIR / "exp3_step2_hog_ablation_report.json"
    log = base.make_logger(log_file)

    ablation_settings = [(7, 7), (4, 4)]

    log("=" * 90)
    log("步骤2：MNIST 上的 HOG 参数消融")
    log("=" * 90)
    log(f"run_id           : {run_ts}")
    log(f"Python           : {sys.executable}")
    log(f"TensorFlow       : {base.tf.__version__}")
    log(f"TF Encrypted     : {getattr(base.tfe, '__version__', 'unknown')}")
    log(f"Platform         : {platform.platform()}")
    log("")
    log("[实验配置]")
    log(f"  MAX_TRAIN_SAMPLES   : {base.MAX_TRAIN_SAMPLES}")
    log(f"  MAX_TEST_SAMPLES    : {base.MAX_TEST_SAMPLES}")
    log(f"  EPOCHS              : {base.EPOCHS}")
    log(f"  LR                  : {base.LR}")
    log(f"  消融参数集合         : {ablation_settings}")
    log("")

    log("[1/2] 加载固定 MNIST 子集（仅一次）...")
    x_train_img, x_test_img, y_train_int, y_test_int = base.load_mnist_subset(
        max_train=base.MAX_TRAIN_SAMPLES,
        max_test=base.MAX_TEST_SAMPLES,
        seed=base.RANDOM_SEED,
    )
    raw_dim = int(x_train_img.shape[1] * x_train_img.shape[2])
    log(f"  训练集样本数      : {x_train_img.shape[0]}")
    log(f"  测试集样本数      : {x_test_img.shape[0]}")
    log(f"  原始特征维度      : {raw_dim}")
    log("")

    log("[2/2] 逐个运行 HOG 消融设置...")
    results = []
    for ppc in ablation_settings:
        log("")
        log("-" * 90)
        log(f"当前设置: pixels_per_cell={ppc}")
        log("-" * 90)
        res = run_one_setting(
            x_train_img=x_train_img,
            x_test_img=x_test_img,
            y_train_int=y_train_int,
            y_test_int=y_test_int,
            pixels_per_cell=ppc,
            log=log,
        )
        results.append(res)

    log("")
    log("=" * 90)
    log("对齐消融表")
    log("=" * 90)
    rows = []
    for r in results:
        ppc = tuple(r["hog_setting"]["pixels_per_cell"])
        rows.append(
            (
                f"ppc={ppc}, ori={r['hog_setting']['orientations']}",
                str(r["feature_dim"]),
                f"{r['compression_ratio']:.4f}",
                f"{r['final_test_acc']:.4f}",
                f"{r['train_time_s']:.3f}",
            )
        )
    log_aligned_ablation_table(log, rows)
    log("=" * 90)

    report = {
        "meta": {
            "experiment": "mnist_hog_step2_ablation",
            "run_id": run_ts,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tensorflow": base.tf.__version__,
            "tf_encrypted": getattr(base.tfe, "__version__", "unknown"),
        },
        "config": {
            "dataset": "MNIST",
            "max_train_samples": base.MAX_TRAIN_SAMPLES,
            "max_test_samples": base.MAX_TEST_SAMPLES,
            "random_seed": base.RANDOM_SEED,
            "epochs": base.EPOCHS,
            "lr": base.LR,
            "ablation_pixels_per_cell": [list(x) for x in ablation_settings],
            "fixed_orientations": base.HOG_ORIENTATIONS,
            "fixed_cells_per_block": list(base.HOG_CELLS_PER_BLOCK),
            "fixed_block_norm": base.HOG_BLOCK_NORM,
            "raw_dim": raw_dim,
        },
        "results": results,
    }

    with report_file.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, sort_keys=False, ensure_ascii=False)
        fp.write("\n")

    log(f"报告文件已写入: {report_file}")
    log(f"日志文件已写入: {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

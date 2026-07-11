#!/usr/bin/env python3
"""
用途说明：
本脚本用于在与 `mnist_hog.py` 完全一致的数据划分与 HOG 配置下，
完成第一步对照实验：
1) 明文 HOG + 逻辑回归（对照 A）
2) TFE 浅层模型下的 Raw(784维) vs HOG(低维)（对照 B）

输出内容：
- 终端与日志文件中的对齐汇总表
- JSON 结果报告
"""

import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import mnist_hog as base


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
# ====== 训练日志打印控制（你要找的“控制 epoch 输出”的核心开关）======
# 例子：
# - 1000 个 epoch + MAX_EPOCH_LOG_LINES=10  -> 大约每 100 轮打印一次
# - 1000 个 epoch + MAX_EPOCH_LOG_LINES=20  -> 大约每  50 轮打印一次
MAX_EPOCH_LOG_LINES = 10


def epoch_to_target(history, target_acc):
    """返回首次达到目标训练精度的 epoch；若未达到则返回 None。"""
    for item in history:
        if float(item.get("train_accuracy", 0.0)) >= target_acc:
            return int(item["epoch"])
    return None


def run_plain_hog_logreg(x_train_hog, y_train_int, x_test_hog, y_test_int):
    """对照 A：明文 HOG + 逻辑回归。"""
    t0 = time.perf_counter()
    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        multi_class="auto",
        n_jobs=1,
        random_state=base.RANDOM_SEED,
    )
    model.fit(x_train_hog, y_train_int.reshape(-1))
    train_acc = float(model.score(x_train_hog, y_train_int.reshape(-1)))
    test_acc = float(model.score(x_test_hog, y_test_int.reshape(-1)))
    elapsed = time.perf_counter() - t0
    return train_acc, test_acc, elapsed


def make_sparse_epoch_logger(log_fn, total_epochs, max_lines=MAX_EPOCH_LOG_LINES):
    """
    构造一个“总条数受控”的 epoch 日志包装器。
    说明：
    - 目标是最多打印 max_lines 条 epoch 记录（另保留最后一轮）；
    - 例如 total_epochs=1000 且 max_lines=10，则按 %100 打印。
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


def log_aligned_summary(log, rows):
    """打印字段对齐的汇总表，避免串行错位。"""
    headers = [
        ("Setup", 24),
        ("FeatDim", 9),
        ("Compress", 10),
        ("TrainAcc", 10),
        ("TestAcc", 9),
        ("TrainTime(s)", 12),
        ("Epoch@0.80", 10),
    ]
    line = " | ".join([f"{name:<{width}}" for name, width in headers])
    sep = "-+-".join(["-" * width for _, width in headers])
    log(line)
    log(sep)
    for row in rows:
        log(
            f"{row[0]:<24} | {row[1]:>9} | {row[2]:>10} | "
            f"{row[3]:>10} | {row[4]:>9} | {row[5]:>12} | {row[6]:>10}"
        )


def main():
    base.ensure_dirs()
    base.configure_logging()
    np.random.seed(base.RANDOM_SEED)
    base.tf.compat.v1.set_random_seed(base.RANDOM_SEED)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"exp2_step1_controls_{run_ts}.log"
    report_file = LOG_DIR / "exp2_step1_controls_report.json"
    log = base.make_logger(log_file)

    log("=" * 90)
    log("步骤1对照实验：明文HOG与TFE(Raw/HOG)对比")
    log("=" * 90)
    log(f"run_id           : {run_ts}")
    log(f"Python           : {sys.executable}")
    log(f"TensorFlow       : {base.tf.__version__}")
    log(f"TF Encrypted     : {getattr(base.tfe, '__version__', 'unknown')}")
    log(f"Platform         : {platform.platform()}")
    log("")
    log("[实验配置]")
    log(f"  MAX_TRAIN_SAMPLES  : {base.MAX_TRAIN_SAMPLES}")
    log(f"  MAX_TEST_SAMPLES   : {base.MAX_TEST_SAMPLES}")
    log(f"  HOG_ORIENTATIONS   : {base.HOG_ORIENTATIONS}")
    log(f"  HOG_PIXELS_PER_CELL: {base.HOG_PIXELS_PER_CELL}")
    log(f"  HOG_CELLS_PER_BLOCK: {base.HOG_CELLS_PER_BLOCK}")
    log(f"  HOG_BLOCK_NORM     : {base.HOG_BLOCK_NORM}")
    log(f"  EPOCHS             : {base.EPOCHS}")
    log(f"  LR                 : {base.LR}")
    log("")

    log("[1/5] 加载与基线一致的 MNIST 子集...")
    t_load = time.time()
    x_train_img, x_test_img, y_train_int, y_test_int = base.load_mnist_subset(
        max_train=base.MAX_TRAIN_SAMPLES,
        max_test=base.MAX_TEST_SAMPLES,
        seed=base.RANDOM_SEED,
    )
    load_time = time.time() - t_load
    raw_dim = int(x_train_img.shape[1] * x_train_img.shape[2])
    log(f"  训练集样本数      : {x_train_img.shape[0]}")
    log(f"  测试集样本数      : {x_test_img.shape[0]}")
    log(f"  原始特征维度      : {raw_dim}")
    log(f"  数据加载耗时(s)   : {load_time:.3f}")
    log("")

    log("[2/5] 提取与基线一致的 HOG 特征...")
    t_hog = time.time()
    x_train_hog = base.extract_hog_features(
        x_train_img,
        orientations=base.HOG_ORIENTATIONS,
        pixels_per_cell=base.HOG_PIXELS_PER_CELL,
        cells_per_block=base.HOG_CELLS_PER_BLOCK,
        block_norm=base.HOG_BLOCK_NORM,
        log_fn=log,
    )
    x_test_hog = base.extract_hog_features(
        x_test_img,
        orientations=base.HOG_ORIENTATIONS,
        pixels_per_cell=base.HOG_PIXELS_PER_CELL,
        cells_per_block=base.HOG_CELLS_PER_BLOCK,
        block_norm=base.HOG_BLOCK_NORM,
        log_fn=log,
    )
    hog_time = time.time() - t_hog
    hog_dim = int(x_train_hog.shape[1])
    compression = float(hog_dim / raw_dim)

    # Keep exactly same preprocessing style as mnist_hog.py
    hog_scaler = StandardScaler()
    x_train_hog = hog_scaler.fit_transform(x_train_hog).astype(np.float32)
    x_test_hog = hog_scaler.transform(x_test_hog).astype(np.float32)
    log(f"  HOG 特征维度      : {hog_dim}")
    log(f"  压缩比例          : {compression:.4f}")
    log(f"  HOG 提取耗时(s)   : {hog_time:.3f}")
    log("")

    log("[3/5] 对照A：明文 HOG + 逻辑回归...")
    plain_train_acc, plain_test_acc, plain_train_time = run_plain_hog_logreg(
        x_train_hog, y_train_int, x_test_hog, y_test_int
    )
    log(f"  明文训练集精度    : {plain_train_acc:.4f}")
    log(f"  明文测试集精度    : {plain_test_acc:.4f}")
    log(f"  明文训练耗时(s)   : {plain_train_time:.3f}")
    log("")

    log("[4/5] 对照B-1：TFE 输入 HOG 特征...")
    # 这里把“稀疏日志函数”传给 TFE 训练函数，真正生效的位置在这行
    sparse_log = make_sparse_epoch_logger(
        log, total_epochs=base.EPOCHS, max_lines=MAX_EPOCH_LOG_LINES
    )
    tfe_hog_history, tfe_hog_test_acc, tfe_hog_checks, tfe_hog_time = (
        base.run_tfe_multiclass(
            x_train_feat=x_train_hog,
            y_train_int=y_train_int,
            x_test_feat=x_test_hog,
            y_test_int=y_test_int,
            n_classes=base.N_CLASSES,
            epochs=base.EPOCHS,
            lr=base.LR,
            log_fn=sparse_log,
        )
    )
    tfe_hog_train_acc = (
        float(tfe_hog_history[-1]["train_accuracy"]) if tfe_hog_history else 0.0
    )
    # Epoch@0.80：训练精度首次达到 0.80（80%）时所在的 epoch
    tfe_hog_epoch80 = epoch_to_target(tfe_hog_history, 0.80)
    log("")

    log("[5/5] 对照B-2：TFE 输入原始像素...")
    x_train_raw = x_train_img.reshape(x_train_img.shape[0], -1).astype(np.float32)
    x_test_raw = x_test_img.reshape(x_test_img.shape[0], -1).astype(np.float32)
    tfe_raw_history, tfe_raw_test_acc, tfe_raw_checks, tfe_raw_time = (
        base.run_tfe_multiclass(
            x_train_feat=x_train_raw,
            y_train_int=y_train_int,
            x_test_feat=x_test_raw,
            y_test_int=y_test_int,
            n_classes=base.N_CLASSES,
            epochs=base.EPOCHS,
            lr=base.LR,
            log_fn=sparse_log,
        )
    )
    tfe_raw_train_acc = (
        float(tfe_raw_history[-1]["train_accuracy"]) if tfe_raw_history else 0.0
    )
    # 同上，记录 raw 输入下首次达到 80% 训练精度的 epoch
    tfe_raw_epoch80 = epoch_to_target(tfe_raw_history, 0.80)
    log("")

    log("=" * 90)
    log("对齐汇总表")
    log("=" * 90)
    rows = [
        (
            "明文 HOG + LR",
            str(hog_dim),
            f"{compression:.4f}",
            f"{plain_train_acc:.4f}",
            f"{plain_test_acc:.4f}",
            f"{plain_train_time:.3f}",
            "-",
        ),
        (
            "TFE HOG 浅层",
            str(hog_dim),
            f"{compression:.4f}",
            f"{tfe_hog_train_acc:.4f}",
            f"{tfe_hog_test_acc:.4f}",
            f"{tfe_hog_time:.3f}",
            str(tfe_hog_epoch80 if tfe_hog_epoch80 is not None else "NA"),
        ),
        (
            "TFE Raw 浅层",
            str(raw_dim),
            f"{1.0:.4f}",
            f"{tfe_raw_train_acc:.4f}",
            f"{tfe_raw_test_acc:.4f}",
            f"{tfe_raw_time:.3f}",
            str(tfe_raw_epoch80 if tfe_raw_epoch80 is not None else "NA"),
        ),
    ]
    log_aligned_summary(log, rows)
    log("=" * 90)

    report = {
        "meta": {
            "experiment": "mnist_hog_step1_controls",
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
            "hog_orientations": base.HOG_ORIENTATIONS,
            "hog_pixels_per_cell": list(base.HOG_PIXELS_PER_CELL),
            "hog_cells_per_block": list(base.HOG_CELLS_PER_BLOCK),
            "hog_block_norm": base.HOG_BLOCK_NORM,
        },
        "feature_stats": {
            "raw_dim": raw_dim,
            "hog_dim": hog_dim,
            "hog_compression_ratio": round(compression, 6),
            "hog_extract_time_s": round(hog_time, 6),
            "load_time_s": round(load_time, 6),
        },
        "results": {
            "control_a_plain_hog_logreg": {
                "train_acc": plain_train_acc,
                "test_acc": plain_test_acc,
                "train_time_s": plain_train_time,
            },
            "control_b_tfe_hog": {
                "train_acc": tfe_hog_train_acc,
                "test_acc": tfe_hog_test_acc,
                "train_time_s": tfe_hog_time,
                "epoch_to_0p80_train_acc": tfe_hog_epoch80,
                "checks": tfe_hog_checks,
                "history": tfe_hog_history,
            },
            "control_b_tfe_raw": {
                "train_acc": tfe_raw_train_acc,
                "test_acc": tfe_raw_test_acc,
                "train_time_s": tfe_raw_time,
                "epoch_to_0p80_train_acc": tfe_raw_epoch80,
                "checks": tfe_raw_checks,
                "history": tfe_raw_history,
            },
        },
    }

    with report_file.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, sort_keys=False, ensure_ascii=False)
        fp.write("\n")

    log(f"报告文件已写入: {report_file}")
    log(f"日志文件已写入: {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

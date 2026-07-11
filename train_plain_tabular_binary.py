#!/usr/bin/env python3
"""
明文 Logistic Regression 基线脚本（与 TFE 任务使用同一份数据缓存）。

输出目标：
1) 训练过程日志（写入 log 目录）
2) 结果报告 JSON（写入 log 目录）
3) 关键指标：train acc / test acc / total time
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "log"


def ensure_dirs():
    """确保实验目录存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_or_build_dataset(test_size=0.2, seed=7):
    """读取缓存数据；若不存在则按与 TFE 相同流程构建并缓存。"""
    cache_file = DATA_DIR / "breast_cancer_split.npz"
    if cache_file.exists():
        cached = np.load(str(cache_file))
        return (
            cached["x_train"].astype(np.float32),
            cached["x_test"].astype(np.float32),
            cached["y_train"].astype(np.float32),
            cached["y_test"].astype(np.float32),
            cache_file,
        )

    data = load_breast_cancer()
    x = data.data.astype(np.float32)
    y = data.target.astype(np.float32).reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    np.savez(
        str(cache_file),
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
    return x_train, x_test, y_train, y_test, cache_file


def main():
    ensure_dirs()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / "train_plain_tabular_binary.log"
    report_file = LOG_DIR / "plain_tabular_training_report.json"

    def log_line(text):
        """同时输出到终端与日志文件。"""
        print(text)
        with log_file.open("a", encoding="utf-8") as fp:
            fp.write(text + "\n")

    x_train, x_test, y_train, y_test, cache_file = load_or_build_dataset()
    y_train_1d = y_train.reshape(-1)
    y_test_1d = y_test.reshape(-1)

    log_line("Python:   {}".format(sys.executable))
    log_line("日志文件: {}".format(log_file))
    log_line("")
    log_line("=" * 60)
    log_line("Plain - 真实表格二分类训练")
    log_line("=" * 60)
    log_line("run_id: {}".format(run_ts))
    log_line("")
    log_line("[1/3] 准备数据...")
    log_line("      数据缓存: {}".format(cache_file))
    log_line("      训练集: {} 样本, {} 特征".format(x_train.shape[0], x_train.shape[1]))
    log_line("      测试集: {} 样本".format(x_test.shape[0]))
    log_line("")
    log_line("[2/3] 训练明文 Logistic Regression...")

    start = time.perf_counter()
    model = LogisticRegression(
        max_iter=2000, solver="lbfgs", random_state=7, n_jobs=1
    )
    model.fit(x_train, y_train_1d)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    total_time = time.perf_counter() - start

    train_acc = float(np.mean(train_pred == y_train_1d))
    test_acc = float(np.mean(test_pred == y_test_1d))

    log_line("      train_accuracy: {:.6f}".format(train_acc))
    log_line("      test_accuracy:  {:.6f}".format(test_acc))
    log_line("      total_time(s):  {:.6f}".format(total_time))
    log_line("")
    log_line("[3/3] 训练完成")

    report = {
        "run_id": run_ts,
        "task": {
            "dataset": "breast_cancer_wisconsin",
            "model": "plain_logistic_regression",
            "train_samples": int(x_train.shape[0]),
            "test_samples": int(x_test.shape[0]),
            "features": int(x_train.shape[1]),
        },
        "summary": {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "total_time_sec": float(total_time),
        },
        "paths": {
            "data_cache_file": str(cache_file),
            "train_log_file": str(log_file),
            "report_file": str(report_file),
        },
    }

    with report_file.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, sort_keys=True, ensure_ascii=False)
        fp.write("\n")

    log_line("训练报告已写入: {}".format(report_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())

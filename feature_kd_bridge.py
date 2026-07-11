#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：桥接脚本（读取模块 a 的缓存特征，一键运行模块 b: KD 初始化）。

支持数据源（简化命名）：
1) mnist -> 读取 MNIST 缓存图像并在线提取 HOG 特征
2) imdb  -> 读取 IMDb + RoBERTa 缓存特征
3) cifar -> 读取 CIFAR-10 + DINO 缓存特征

稳定性设计（默认开启 safe mode）：
- 限制 CPU 线程，避免进程“爆核”卡死
- 支持对测试集做上限截断（例如 1000 -> 500）
- 支持 dry-run 只检查数据和配置，不执行训练
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "log"
LOG_TXT_DIR = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"
EXPFIG_DIR = BASE_DIR / "expFig"
FEATURE_DATA_DIR = BASE_DIR / "featureData"

# 统一实验配置：后续优先在这里改参数，避免命令行过长。
CFG = {
    "safe_mode": True,
    "cpu_threads": 4,
    "max_test_cap": 1000,
    "num_clients": 5,
    "classes_per_client": 0,  # 0=自动
    "ref_size": 200,
    "local_epochs": 80,
    "local_lr": 1e-2,
    "tau": 2.0,
    "lambda_kd": 1.0,
    "distill_epochs": 30,
    "distill_lr": 5e-3,
    "run_fig5": False,
    "global_epochs": 800,
    "global_lr": 1e-2,
    "seed": 42,
    "max_epoch_log_lines": 10,
    "device": "",  # 留空表示走模块 b 默认设备选择
    "use_secure_aggregation": True,  # True=ABY3密态聚合, False=明文聚合
    # 模块 b 本地训练：默认仍是 SGD；MNIST 先切到 DPSGD（见 DATASET_LOCAL_OVERRIDES）
    "local_use_dpsgd": False,
    "local_dpsgd_batch_size": 64,
    "local_dpsgd_clipping_norm": 1.0,
    "local_dpsgd_noise_multiplier": 1.0,
}

# 先从 MNIST 开始对齐“特征数据 -> 本地标准 DPSGD”。
DATASET_LOCAL_OVERRIDES = {
    "mnist": {
        "local_use_dpsgd": True,
        "local_dpsgd_batch_size": 64,
        "local_dpsgd_clipping_norm": 1.0,
        "local_dpsgd_noise_multiplier": 1.0,
    },
    "imdb": {
        # IMDb 特征维度高，batch 稍小一些更稳。
        "local_use_dpsgd": True,
        "local_dpsgd_batch_size": 32,
        "local_dpsgd_clipping_norm": 1.0,
        "local_dpsgd_noise_multiplier": 1.0,
    },
    "cifar": {
        # CIFAR(DINO) 先用与 MNIST 一致的正式版基线配置，便于横向对照。
        "local_use_dpsgd": True,
        "local_dpsgd_batch_size": 64,
        "local_dpsgd_clipping_norm": 1.0,
        "local_dpsgd_noise_multiplier": 1.0,
    },
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    EXPFIG_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def tee_console_to_file(log_path: Path):
    """将控制台输出同时写入文件（人类可读日志）。"""
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

        def close(self):
            # 兼容某些 logging handler 在退出时调用 close()
            self.flush()

    orig_out, orig_err = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as f:
        tee = _Tee(orig_out, f)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err


def set_safe_runtime(max_threads: int = 4) -> None:
    """安全运行模式：限制线程，降低 CPU 打满概率。"""
    max_threads = max(1, int(max_threads))
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)

    try:
        import torch
        torch.set_num_threads(max_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max_threads)
    except Exception:
        pass


def _find_latest(pattern: str) -> Path:
    files = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"未找到缓存文件: pattern={pattern}")
    return files[0]


def _limit_test_if_needed(
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_test: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    测试集截断（安全版）：
    - 不再直接取前 N 条，避免类别有序缓存导致评估偏差；
    - 使用分层随机抽样，尽量保持类别分布。
    """
    y_test = np.asarray(y_test).reshape(-1)
    if max_test <= 0 or len(y_test) <= max_test:
        return x_test, y_test

    rng = np.random.RandomState(seed)
    classes = np.unique(y_test)
    k = int(max_test)
    per_cls = max(1, k // max(1, len(classes)))

    picked = []
    for c in classes:
        idx = np.where(y_test == c)[0]
        if len(idx) == 0:
            continue
        take = min(per_cls, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        picked.append(chosen)

    if len(picked) == 0:
        # 兜底：全局随机
        idx = rng.choice(len(y_test), size=k, replace=False)
        return x_test[idx], y_test[idx]

    picked = np.concatenate(picked)
    # 若数量不足 k，则从剩余样本补齐
    if len(picked) < k:
        remain_pool = np.setdiff1d(np.arange(len(y_test)), picked, assume_unique=False)
        need = min(k - len(picked), len(remain_pool))
        if need > 0:
            extra = rng.choice(remain_pool, size=need, replace=False)
            picked = np.concatenate([picked, extra])

    # 若数量超过 k，随机裁剪到 k
    if len(picked) > k:
        picked = rng.choice(picked, size=k, replace=False)

    rng.shuffle(picked)
    return x_test[picked], y_test[picked]


def _quick_binary_linear_probe_acc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    steps: int = 120,
    lr: float = 0.1,
) -> float:
    """
    IMDb 特征质量快速探针（二分类）：
    用极简 logistic 训练做 sanity check，判断“特征-标签是否基本可对齐”。
    """
    xtr = np.asarray(x_train, dtype=np.float32)
    xte = np.asarray(x_test, dtype=np.float32)
    ytr = np.asarray(y_train).reshape(-1).astype(np.float32)
    yte = np.asarray(y_test).reshape(-1).astype(np.int32)

    if xtr.ndim != 2 or xte.ndim != 2 or len(np.unique(ytr)) < 2:
        return 0.0

    # 标准化，避免尺度影响收敛
    mu = xtr.mean(axis=0, keepdims=True)
    std = xtr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    xtr = (xtr - mu) / std
    xte = (xte - mu) / std

    xtr1 = np.concatenate([xtr, np.ones((xtr.shape[0], 1), dtype=np.float32)], axis=1)
    xte1 = np.concatenate([xte, np.ones((xte.shape[0], 1), dtype=np.float32)], axis=1)
    w = np.zeros((xtr1.shape[1],), dtype=np.float32)

    for _ in range(max(20, int(steps))):
        z = xtr1.dot(w)
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        g = xtr1.T.dot(p - ytr) / float(len(ytr))
        w -= float(lr) * g

    pte = 1.0 / (1.0 + np.exp(-np.clip(xte1.dot(w), -30.0, 30.0)))
    pred = (pte >= 0.5).astype(np.int32)
    return float(np.mean(pred == yte))


def _load_imdb_tfidf_fallback(max_test_cap: int = 0) -> Dict:
    """
    IMDb 回退特征：
    当 RoBERTa 缓存疑似错配（探针准确率接近 0.5）时，回退到 TF-IDF 缓存。
    """
    tfidf_file = _find_latest("imdb_tfidf_tr*_te*_f*_ng*.npz")
    m = re.search(r"_tr(\d+)_te(\d+)_", tfidf_file.name)
    if not m:
        raise RuntimeError(f"无法从 TF-IDF 文件名解析 tr/te: {tfidf_file.name}")
    tr_n = int(m.group(1))
    te_n = int(m.group(2))

    label_file = DATA_DIR / f"imdb_texts_tr{tr_n}_te{te_n}_s{int(CFG['seed'])}.npz"
    if not label_file.exists():
        candidates = sorted(DATA_DIR.glob(f"imdb_texts_tr{tr_n}_te{te_n}_s*.npz"))
        if not candidates:
            raise FileNotFoundError(f"未找到匹配 IMDb 标签缓存: tr={tr_n}, te={te_n}")
        label_file = candidates[0]

    d_feat = np.load(str(tfidf_file))
    d_lbl = np.load(str(label_file), allow_pickle=True)
    x_train = d_feat["x_train"].astype(np.float32)
    x_test = d_feat["x_test"].astype(np.float32)
    y_train = d_lbl["y_train"].astype(int)
    y_test = d_lbl["y_test"].astype(int)
    x_test, y_test = _limit_test_if_needed(x_test, y_test, max_test_cap, seed=int(CFG["seed"]))

    return {
        "dataset_name": "imdb_tfidf_fallback",
        "source_feat": str(tfidf_file),
        "source_label": str(label_file),
        "X_train_feat": x_train,
        "X_test_feat": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def load_imdb_roberta_cached(max_test_cap: int = 0) -> Dict:
    """读取 IMDb + RoBERTa 缓存特征。"""
    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    featdata_files = sorted(
        FEATURE_DATA_DIR.glob("imdb_RoBERTa_tr*_te*_seed*.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if featdata_files:
        feat_file = featdata_files[0]
        d = np.load(str(feat_file))
        x_train = d["x_train_feat"].astype(np.float32)
        x_test = d["x_test_feat"].astype(np.float32)
        y_train = d["y_train"].astype(int)
        y_test = d["y_test"].astype(int)
        x_test, y_test = _limit_test_if_needed(x_test, y_test, max_test_cap, seed=int(CFG["seed"]))
        probe_acc = _quick_binary_linear_probe_acc(x_train, y_train, x_test, y_test)
        if probe_acc < 0.56:
            print(
                f"[IMDb加载告警] RoBERTa缓存疑似错配（probe_acc={probe_acc:.4f}），"
                "已自动回退到 TF-IDF 缓存。"
            )
            return _load_imdb_tfidf_fallback(max_test_cap=max_test_cap)
        return {
            "dataset_name": "imdb_roberta",
            "source_feat": str(feat_file),
            "source_label": str(feat_file),
            "X_train_feat": x_train,
            "X_test_feat": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    feat_file = _find_latest("imdb_roberta_*_tr*_te*_ml*.npz")
    m = re.search(r"_tr(\d+)_te(\d+)_ml", feat_file.name)
    if not m:
        raise RuntimeError(f"无法从文件名解析 tr/te: {feat_file.name}")
    tr_n = int(m.group(1))
    te_n = int(m.group(2))

    label_file = DATA_DIR / f"imdb_texts_tr{tr_n}_te{te_n}_s42.npz"
    if not label_file.exists():
        candidates = sorted(DATA_DIR.glob(f"imdb_texts_tr{tr_n}_te{te_n}_s*.npz"))
        if not candidates:
            raise FileNotFoundError(f"未找到匹配标签缓存: tr={tr_n}, te={te_n}")
        label_file = candidates[0]

    d_feat = np.load(str(feat_file))
    d_lbl = np.load(str(label_file), allow_pickle=True)

    x_train = d_feat["x_train"].astype(np.float32)
    x_test = d_feat["x_test"].astype(np.float32)
    y_train = d_lbl["y_train"].astype(int)
    y_test = d_lbl["y_test"].astype(int)
    x_test, y_test = _limit_test_if_needed(x_test, y_test, max_test_cap, seed=int(CFG["seed"]))
    probe_acc = _quick_binary_linear_probe_acc(x_train, y_train, x_test, y_test)
    if probe_acc < 0.56:
        print(
            f"[IMDb加载告警] RoBERTa原始缓存疑似错配（probe_acc={probe_acc:.4f}），"
            "已自动回退到 TF-IDF 缓存。"
        )
        return _load_imdb_tfidf_fallback(max_test_cap=max_test_cap)

    # 回填 featureData 统一缓存
    feat_cache = FEATURE_DATA_DIR / f"imdb_RoBERTa_tr{len(y_train)}_te{len(y_test)}_seed{int(CFG['seed'])}.npz"
    np.savez(
        str(feat_cache),
        x_train_feat=x_train.astype(np.float32),
        x_test_feat=x_test.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        extractor="RoBERTa",
        dataset="imdb",
    )

    return {
        "dataset_name": "imdb_roberta",
        "source_feat": str(feat_cache),
        "source_label": str(label_file),
        "X_train_feat": x_train,
        "X_test_feat": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def load_cifar_dino_cached(max_test_cap: int = 0) -> Dict:
    """读取 CIFAR-10 + DINO 缓存特征。"""
    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    featdata_files = sorted(
        FEATURE_DATA_DIR.glob("cifar10_dino_tr*_te*_seed*.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if featdata_files:
        feat_file = featdata_files[0]
        d = np.load(str(feat_file))
        x_train = d["x_train_feat"].astype(np.float32)
        x_test = d["x_test_feat"].astype(np.float32)
        y_train = d["y_train"].astype(int)
        y_test = d["y_test"].astype(int)
        x_test, y_test = _limit_test_if_needed(x_test, y_test, max_test_cap, seed=int(CFG["seed"]))
        return {
            "dataset_name": "cifar_dino",
            "source_feat": str(feat_file),
            "source_label": str(feat_file),
            "X_train_feat": x_train,
            "X_test_feat": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    train_feat_file = _find_latest("dino_feat_*_train_*_*.npz")

    # 从 train 文件名解析 extractor、train 数量、input_size，再定位匹配 test 文件
    m = re.search(r"^dino_feat_(.+)_train_(\d+)_(\d+)\.npz$", train_feat_file.name)
    if not m:
        raise RuntimeError(f"无法解析 DINO 训练缓存名: {train_feat_file.name}")
    extractor_name = m.group(1)
    tr_n = int(m.group(2))
    input_size = int(m.group(3))

    test_candidates = sorted(DATA_DIR.glob(f"dino_feat_{extractor_name}_test_*_{input_size}.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not test_candidates:
        raise FileNotFoundError("未找到匹配的 DINO 测试特征缓存")
    test_feat_file = test_candidates[0]

    m2 = re.search(r"_test_(\d+)_", test_feat_file.name)
    te_n = int(m2.group(1)) if m2 else 1000

    label_file = DATA_DIR / f"cifar10_tr{tr_n}_te{te_n}_s42.npz"
    if not label_file.exists():
        candidates = sorted(DATA_DIR.glob(f"cifar10_tr{tr_n}_te{te_n}_s*.npz"))
        if not candidates:
            raise FileNotFoundError(f"未找到匹配 CIFAR 标签缓存: tr={tr_n}, te={te_n}")
        label_file = candidates[0]

    d_tr = np.load(str(train_feat_file))
    d_te = np.load(str(test_feat_file))
    d_lbl = np.load(str(label_file))

    x_train = d_tr["features"].astype(np.float32)
    x_test = d_te["features"].astype(np.float32)
    y_train = d_lbl["y_train"].astype(int)
    y_test = d_lbl["y_test"].astype(int)
    x_test, y_test = _limit_test_if_needed(x_test, y_test, max_test_cap, seed=int(CFG["seed"]))

    # 回填 featureData 统一缓存
    feat_cache = FEATURE_DATA_DIR / f"cifar10_dino_tr{len(y_train)}_te{len(y_test)}_seed{int(CFG['seed'])}.npz"
    np.savez(
        str(feat_cache),
        x_train_feat=x_train.astype(np.float32),
        x_test_feat=x_test.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        extractor="dino",
        extractor_name=extractor_name,
        input_size=int(input_size),
        dataset="cifar10",
    )

    return {
        "dataset_name": "cifar_dino",
        "source_feat": str(feat_cache),
        "source_label": str(label_file),
        "X_train_feat": x_train,
        "X_test_feat": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def load_mnist_hog_cached(max_test_cap: int = 0) -> Dict:
    """读取 MNIST HOG 特征缓存；若不存在再尝试在线提取并保存到 featureData。"""
    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 优先读取 featureData 统一缓存，避免在无 TensorFlow 环境中触发 mnist_hog 导入失败。
    def _mnist_cache_score(p: Path):
        m = re.search(r"mnist_hog_tr(\d+)_te(\d+)_seed(\d+)\.npz$", p.name)
        if not m:
            return (0, 0, p.stat().st_mtime)
        tr_n = int(m.group(1))
        te_n = int(m.group(2))
        return (tr_n, te_n, p.stat().st_mtime)

    featdata_files = sorted(
        FEATURE_DATA_DIR.glob("mnist_hog_tr*_te*_seed*.npz"),
        key=_mnist_cache_score,
        reverse=True,
    )
    if featdata_files:
        feat_file = featdata_files[0]
        dc = np.load(str(feat_file))
        x_train_hog = dc["x_train_feat"].astype(np.float32)
        x_test_hog = dc["x_test_feat"].astype(np.float32)
        y_train = dc["y_train"].astype(int)
        y_test = dc["y_test"].astype(int)
        x_test_hog, y_test = _limit_test_if_needed(
            x_test_hog, y_test, max_test_cap, seed=int(CFG["seed"])
        )
        return {
            "dataset_name": "mnist_hog",
            "source_feat": str(feat_file),
            "source_label": str(feat_file),
            "X_train_feat": x_train_hog.astype(np.float32),
            "X_test_feat": x_test_hog.astype(np.float32),
            "y_train": y_train,
            "y_test": y_test,
        }

    cache_file = _find_latest("mnist_train*_test*_seed*.npz")
    d = np.load(str(cache_file))

    x_train_img = d["x_train"].astype(np.float32)
    x_test_img = d["x_test"].astype(np.float32)
    y_train = d["y_train"].astype(int)
    y_test = d["y_test"].astype(int)

    x_test_img, y_test = _limit_test_if_needed(x_test_img, y_test, max_test_cap, seed=int(CFG["seed"]))

    # 从原图缓存名中解析 seed，落盘名按：mnist_hog_tr1000_te1000_seed42.npz
    m = re.search(r"mnist_train(\d+)_test(\d+)_seed(\d+)\.npz$", cache_file.name)
    parsed_seed = int(m.group(3)) if m else int(CFG["seed"])
    tr_n = int(len(y_train))
    te_n = int(len(y_test))
    feat_cache = FEATURE_DATA_DIR / f"mnist_hog_tr{tr_n}_te{te_n}_seed{parsed_seed}.npz"

    # 旧逻辑兼容：若目标命名缓存已存在，直接读取。
    if feat_cache.exists():
        dc = np.load(str(feat_cache))
        x_train_hog = dc["x_train_feat"].astype(np.float32)
        x_test_hog = dc["x_test_feat"].astype(np.float32)
        y_train = dc["y_train"].astype(int)
        y_test = dc["y_test"].astype(int)
    else:
        # 仅在确实需要在线提取时再导入 mnist_hog（该模块依赖 TensorFlow）。
        try:
            import mnist_hog as mh
        except Exception as e:
            raise RuntimeError(
                "未找到可用的 MNIST HOG 特征缓存，且当前环境无法导入 mnist_hog（通常因为缺少 TensorFlow）。\n"
                "请先在 tfe 环境运行一次 MNIST 模块 A 生成 featureData 缓存，"
                "或改用 tfe 环境执行本脚本。\n"
                f"底层错误: {type(e).__name__}: {e}"
            )

        # HOG 特征提取（明文侧）+ 标准化 + 缓存落盘
        x_train_hog = mh.extract_hog_features(
            x_train_img,
            orientations=mh.HOG_ORIENTATIONS,
            pixels_per_cell=mh.HOG_PIXELS_PER_CELL,
            cells_per_block=mh.HOG_CELLS_PER_BLOCK,
            block_norm=mh.HOG_BLOCK_NORM,
            log_fn=lambda *_: None,
        )
        x_test_hog = mh.extract_hog_features(
            x_test_img,
            orientations=mh.HOG_ORIENTATIONS,
            pixels_per_cell=mh.HOG_PIXELS_PER_CELL,
            cells_per_block=mh.HOG_CELLS_PER_BLOCK,
            block_norm=mh.HOG_BLOCK_NORM,
            log_fn=lambda *_: None,
        )

        # 与 mnist_hog.py 训练前处理一致：标准化
        scaler = mh.StandardScaler()
        x_train_hog = scaler.fit_transform(x_train_hog).astype(np.float32)
        x_test_hog = scaler.transform(x_test_hog).astype(np.float32)

        np.savez(
            str(feat_cache),
            x_train_feat=x_train_hog.astype(np.float32),
            x_test_feat=x_test_hog.astype(np.float32),
            y_train=y_train.astype(np.int32),
            y_test=y_test.astype(np.int32),
            source_image_cache=str(cache_file),
            extractor="hog",
            dataset="mnist",
        )

    return {
        "dataset_name": "mnist_hog",
        "source_feat": str(feat_cache),
        "source_label": str(cache_file),
        "X_train_feat": x_train_hog.astype(np.float32),
        "X_test_feat": x_test_hog.astype(np.float32),
        "y_train": y_train,
        "y_test": y_test,
    }


def load_dataset_features(dataset: str, max_test_cap: int = 0) -> Dict:
    if dataset == "mnist":
        return load_mnist_hog_cached(max_test_cap=max_test_cap)
    if dataset == "imdb":
        return load_imdb_roberta_cached(max_test_cap=max_test_cap)
    if dataset == "cifar":
        return load_cifar_dino_cached(max_test_cap=max_test_cap)
    raise ValueError(f"不支持的数据集标识: {dataset}")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="桥接脚本：读取缓存特征，一键运行模块 b")
    p.add_argument("--dataset", type=str, default="cifar", choices=["mnist", "imdb", "cifar"], help="数据集名称")
    p.add_argument("--dry_run", action="store_true", help="只检查数据和配置，不训练")
    return p


def save_figure5_outputs(fig5: Dict, dataset: str, run_id: str) -> Dict:
    """
    保存 Figure5 收敛曲线：
    1) 原始 json
    2) 扁平 csv（便于 Origin/Excel 作图）
    3) 若 matplotlib 可用则自动画 png
    """
    out = {}
    fig5_json = EXPFIG_DIR / f"figure5_curve_{dataset}_{run_id}.json"
    with fig5_json.open("w", encoding="utf-8") as f:
        json.dump(fig5, f, indent=2, ensure_ascii=False)
        f.write("\n")
    out["figure5_json"] = str(fig5_json)

    fig5_csv = EXPFIG_DIR / f"figure5_curve_{dataset}_{run_id}.csv"
    with fig5_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["init_type", "iteration", "train_loss", "test_loss", "train_acc", "test_acc"])
        for init_name, rows in fig5["curves"].items():
            for r in rows:
                w.writerow([
                    init_name, r["epoch"], r["train_loss"], r["test_loss"], r["train_acc"], r["test_acc"]
                ])
    out["figure5_csv"] = str(fig5_csv)

    # 尝试自动画图（环境缺少 matplotlib 时跳过，不报错）
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        fig_png = EXPFIG_DIR / f"figure5_curve_{dataset}_{run_id}.png"
        plt.figure(figsize=(8, 5))
        for init_name, rows in fig5["curves"].items():
            epochs = [int(r["epoch"]) for r in rows]
            test_acc = [r["test_acc"] for r in rows]
            plt.plot(epochs, test_acc, label=init_name)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Iteration")
        plt.ylabel("Test Accuracy")
        plt.title("Figure5-like Convergence Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(fig_png), dpi=160)
        plt.close()
        out["figure5_png"] = str(fig_png)
    except Exception:
        out["figure5_png"] = None

    return out


def main() -> int:
    ensure_dirs()
    args = make_parser().parse_args()

    use_safe_mode = bool(CFG["safe_mode"])
    if use_safe_mode:
        set_safe_runtime(CFG["cpu_threads"])

    # 延迟导入，确保先设置线程限制
    import feature_kd_init as kd

    t0 = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    txt_log_file = LOG_TXT_DIR / f"conInitModel_{args.dataset}_{run_id}.log"
    json_log_file = LOG_JSON_DIR / f"conInitModel_{args.dataset}_{run_id}.json"

    with tee_console_to_file(txt_log_file):
        print("=" * 72)
        print("桥接脚本：模块 a 缓存特征 -> 模块 b KD 初始化")
        print("=" * 72)
        print("run_id          :", run_id)
        print("dataset         :", args.dataset)
        print("safe_mode       :", use_safe_mode)
        print("cpu_threads     :", CFG["cpu_threads"] if use_safe_mode else "(未限制)")

        data = load_dataset_features(args.dataset, max_test_cap=CFG["max_test_cap"])
        xtr = data["X_train_feat"]
        xte = data["X_test_feat"]
        ytr = data["y_train"]
        yte = data["y_test"]

        print("source_feat     :", data["source_feat"])
        print("source_label    :", data["source_label"])
        print("X_train_feat    :", xtr.shape)
        print("X_test_feat     :", xte.shape)
        print("y_train         :", ytr.shape)
        print("y_test          :", yte.shape)

        local_cfg = {
            "local_use_dpsgd": bool(CFG.get("local_use_dpsgd", False)),
            "local_dpsgd_batch_size": int(CFG.get("local_dpsgd_batch_size", 64)),
            "local_dpsgd_clipping_norm": float(CFG.get("local_dpsgd_clipping_norm", 1.0)),
            "local_dpsgd_noise_multiplier": float(CFG.get("local_dpsgd_noise_multiplier", 1.0)),
        }
        local_cfg.update(DATASET_LOCAL_OVERRIDES.get(args.dataset, {}))
        print("local_train_mode:", "DPSGD" if local_cfg["local_use_dpsgd"] else "SGD")
        if local_cfg["local_use_dpsgd"]:
            print("dpsgd(batch/C/sigma): {}/{}/{}".format(
                local_cfg["local_dpsgd_batch_size"],
                local_cfg["local_dpsgd_clipping_norm"],
                local_cfg["local_dpsgd_noise_multiplier"],
            ))

        if args.dry_run:
            dry_summary = {
                "run_id": run_id,
                "dataset": args.dataset,
                "safe_mode": use_safe_mode,
                "cpu_threads": int(CFG["cpu_threads"]) if use_safe_mode else None,
                "dry_run": True,
                "data": {
                    "source_feat": data["source_feat"],
                    "source_label": data["source_label"],
                    "x_train_shape": tuple(xtr.shape),
                    "x_test_shape": tuple(xte.shape),
                    "y_train_shape": tuple(ytr.shape),
                    "y_test_shape": tuple(yte.shape),
                },
                "bridge_cfg": CFG,
                "txt_log_file": str(txt_log_file),
            }
            with json_log_file.open("w", encoding="utf-8") as f:
                json.dump(dry_summary, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print("dry_run=True，数据检查完成，未执行训练。")
            print("TXT日志已写入:", txt_log_file)
            print("JSON日志已写入:", json_log_file)
            return 0

        cpc = None if int(CFG["classes_per_client"]) <= 0 else int(CFG["classes_per_client"])
        out = kd.run_kd_hetero_init_pipeline(
            X_train_feat=xtr,
            y_train=ytr,
            X_test_feat=xte,
            y_test=yte,
            num_clients=int(CFG["num_clients"]),
            classes_per_client=cpc,
            ref_size=int(CFG["ref_size"]),
            local_epochs=int(CFG["local_epochs"]),
            local_lr=float(CFG["local_lr"]),
            tau=float(CFG["tau"]),
            lambda_kd=float(CFG["lambda_kd"]),
            distill_epochs=int(CFG["distill_epochs"]),
            distill_lr=float(CFG["distill_lr"]),
            run_figure5=bool(CFG["run_fig5"]),
            global_epochs=int(CFG["global_epochs"]),
            global_lr=float(CFG["global_lr"]),
            seed=int(CFG["seed"]),
            max_epoch_log_lines=int(CFG["max_epoch_log_lines"]),
            device=(str(CFG["device"]).strip() if str(CFG["device"]).strip() else None),
            dataset_name=str(args.dataset),
            save_init_model=True,
            use_secure_aggregation=bool(CFG.get("use_secure_aggregation", True)),
            secure_max_threads=int(CFG.get("cpu_threads", 2)),
            local_use_dpsgd=bool(local_cfg["local_use_dpsgd"]),
            local_dpsgd_batch_size=int(local_cfg["local_dpsgd_batch_size"]),
            local_dpsgd_clipping_norm=float(local_cfg["local_dpsgd_clipping_norm"]),
            local_dpsgd_noise_multiplier=float(local_cfg["local_dpsgd_noise_multiplier"]),
            verbose=True,
        )

        fig5_files = None
        if CFG["run_fig5"] and out.get("figure5") is not None:
            fig5_files = save_figure5_outputs(out["figure5"], args.dataset, run_id)

        summary = {
            "run_id": run_id,
            "dataset": args.dataset,
            "safe_mode": use_safe_mode,
            "cpu_threads": int(CFG["cpu_threads"]) if use_safe_mode else None,
            "data": {
                "source_feat": data["source_feat"],
                "source_label": data["source_label"],
                "x_train_shape": tuple(xtr.shape),
                "x_test_shape": tuple(xte.shape),
                "y_train_shape": tuple(ytr.shape),
                "y_test_shape": tuple(yte.shape),
            },
            "candidate": out["candidate"],
            "figure5": out.get("figure5"),
            "figure5_files": fig5_files,
            "config": out["config"],
            "bridge_cfg": CFG,
            "elapsed_s": round(time.time() - t0, 3),
            "txt_log_file": str(txt_log_file),
        }

        with json_log_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print("\nTXT日志已写入:", txt_log_file)
        print("JSON日志已写入:", json_log_file)
        print("总耗时(s):", summary["elapsed_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())


# [test set check] 是“初始化时刻”的精度检查，[Figure5] 是“后续全局训练结束后”的最终精度比较，
# 所以两者数值不一样完全正常。前者说明 avg 是更好的起点，后者说明 avg 也是更好的训练起点并最终达到最高 test accuracy。

# [Step7/8] candidate comparison on reference set随机抽取的训练集200个样本
# reference set: X_ref=(200, 384), y_ref=(200,)

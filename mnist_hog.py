#!/usr/bin/env python3
"""
实验 1：MNIST + HOG 特征提取 + TF-Encrypted 密态多分类训练
==============================================================

复现并扩展 PEA 论文（IEEE S&P 2023）Section V-A 思路：
  "data-independent feature extraction + 密态浅层训练"

Pipeline：
  MNIST(784-dim)
    → [本地明文] HOG 特征提取 → HOG-feat(N_HOG_FEAT-dim)
    → [TFE 密态] define_private_variable → 加密分片
    → [TFE 密态] 单层 Dense(10) + MSE → 多分类
    → [明文监控] reveal() 计算 loss/acc → 日志 + JSON 报告

关键实验意义（论文用）：
  1. 验证 HOG 作为"数据无关"特征提取器在 MPC 场景下的可行性
  2. 量化特征维度压缩比例与训练效率的关系
  3. 与 baseline（原始像素入 MPC）对比，证明 Feature Extraction 的必要性

环境依赖：
  tensorflow==1.15.5
  tf-encrypted==0.8.0
  scikit-image          (HOG 提取)
  scikit-learn          (StandardScaler / metrics)
  numpy, tqdm

运行：
  python exp1_mnist_hog.py
"""

import json
import logging
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ============================================================================
# 在任何 TF 导入前设置日志等级
# ============================================================================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_encrypted").setLevel(logging.ERROR)

import tf_encrypted as tfe
from skimage.feature import hog
from skimage.transform import resize as sk_resize
from sklearn.preprocessing import StandardScaler
from tf_encrypted.keras import Sequential
from tf_encrypted.keras import backend as KE
from tf_encrypted.keras.layers import Dense
from tqdm import tqdm


# ============================================================================
# ★ 实验配置（所有超参数集中在此，方便论文对比实验调整）
# ============================================================================

# --- 数据配置 ---
MAX_TRAIN_SAMPLES = 1000   # TFE 单 batch 全量训练，太大会 OOM；论文消融实验用 500/1000/2000
MAX_TEST_SAMPLES  = 10000  # 测试集样本数
RANDOM_SEED       = 42

# --- HOG 特征提取参数（对应论文 Section: Feature Extractor Design）---
HOG_IMAGE_SIZE    = (28, 28)   # MNIST 原始尺寸，无需 resize
HOG_ORIENTATIONS  = 9          # 梯度方向数，PEA 论文推荐 9
HOG_PIXELS_PER_CELL = (7, 7)   # 每个 cell 的像素数；越大特征越少
HOG_CELLS_PER_BLOCK = (1, 1)   # 每个 block 的 cell 数
HOG_BLOCK_NORM    = "L2-Hys"   # block 归一化方式

# --- TFE 训练配置 ---
N_CLASSES  = 10       # MNIST 10 分类
EPOCHS     = 500      # 30         （400-500收敛）
LR         = 0.01     # 多分类比二分类学习率要小一些
MAX_EPOCH_LOG_LINES = 10  # 训练日志最多打印多少个 epoch 采样点

# --- Figure4（DP预算-精度）配置 ---
RUN_FIGURE4_DP            = True
FIG4_EPSILON_LIST         = [1.0, 2.0, 4.0, 8.0, 16.0]
FIG4_DELTA                = 1e-5
FIG4_DP_EPOCHS            = 120
FIG4_DP_LR                = 0.01
FIG4_MAX_EPOCH_LOG_LINES  = 10

# --- 路径配置 ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR  = BASE_DIR / "log"
FEATURE_DATA_DIR = BASE_DIR / "featureData"
EXPFIG_DIR = BASE_DIR / "expFig"
FIG4_DIR = EXPFIG_DIR / "figure4"


# ============================================================================
# 公共工具函数（三个实验脚本共享同一写法风格）
# ============================================================================

def ensure_dirs():
    """确保实验目录存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPFIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG4_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging():
    """压低运行期 TF / TFE 日志噪音。"""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("tf_encrypted").setLevel(logging.ERROR)


def make_logger(log_file: Path):
    """
    返回一个同时写终端和日志文件的 log 函数。
    风格与 baseline 脚本保持一致。
    """
    def log(text: str = ""):
        print(text)
        with log_file.open("a", encoding="utf-8") as fp:
            fp.write(text + "\n")
    return log


def compute_hog_dim(image_size, orientations, pixels_per_cell, cells_per_block):
    """
    预先计算 HOG 输出维度，用于日志和模型构建。

    公式：
      n_cells_x = image_w / pixels_per_cell_x
      n_cells_y = image_h / pixels_per_cell_y
      n_blocks_x = n_cells_x - cells_per_block_x + 1
      n_blocks_y = n_cells_y - cells_per_block_y + 1
      hog_dim = n_blocks_x * n_blocks_y * cells_per_block_x * cells_per_block_y * orientations
    """
    h, w = image_size
    pcx, pcy = pixels_per_cell
    cbx, cby = cells_per_block
    n_cells_x = w // pcx
    n_cells_y = h // pcy
    n_blocks_x = n_cells_x - cbx + 1
    n_blocks_y = n_cells_y - cby + 1
    return int(n_blocks_x * n_blocks_y * cbx * cby * orientations)


def to_onehot(y, n_classes):
    """将整数标签转为 one-hot 矩阵 (float32)。"""
    y = y.flatten().astype(int)
    oh = np.zeros((len(y), n_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def accuracy_from_logits(logits_np, y_int):
    """
    从 logit 矩阵（shape: [N, C]）和整数标签计算准确率。
    argmax 找最大输出作为预测类别。
    """
    pred = np.argmax(logits_np, axis=1)
    return float(np.mean(pred == y_int.flatten()))


def as_scalar_loss(loss_value):
    """
    将 TFE reveal() 返回的 loss 统一转成 Python float。
    某些图里返回标量，某些返回向量/矩阵（逐样本 loss），这里取均值。
    """
    arr = np.asarray(loss_value, dtype=np.float64)
    return float(arr.mean())


# ============================================================================
# ★ 数据加载
# ============================================================================

def load_mnist_subset(
    max_train=MAX_TRAIN_SAMPLES,
    max_test=MAX_TEST_SAMPLES,
    seed=RANDOM_SEED,
):
    """
    加载 MNIST 并截取子集。

    使用 tf.keras.datasets（不依赖额外网络，复用已下载缓存）。
    返回归一化到 [0,1] 的 float32 图像（shape: [N, 28, 28]）和整数标签。
    """
    cache_file = DATA_DIR / f"mnist_train{max_train}_test{max_test}_seed{seed}.npz"
    if cache_file.exists():
        d = np.load(str(cache_file))
        return (
            d["x_train"], d["x_test"],
            d["y_train"].astype(int), d["y_test"].astype(int),
        )

    (x_tr_full, y_tr_full), (x_te_full, y_te_full) = (
        tf.keras.datasets.mnist.load_data()
    )

    rng = np.random.RandomState(seed)

    # 从训练集随机选 max_train 个样本（保持各类均衡）
    train_idx = _balanced_sample(y_tr_full, max_train, rng)
    test_idx  = _balanced_sample(y_te_full, max_test,  rng)

    x_train = (x_tr_full[train_idx] / 255.0).astype(np.float32)
    x_test  = (x_te_full[test_idx]  / 255.0).astype(np.float32)
    y_train = y_tr_full[train_idx].astype(np.int32)
    y_test  = y_te_full[test_idx].astype(np.int32)

    np.savez(
        str(cache_file),
        x_train=x_train, x_test=x_test,
        y_train=y_train, y_test=y_test,
    )
    return x_train, x_test, y_train, y_test


def _balanced_sample(labels, n, rng):
    """各类均衡抽样，返回索引数组。"""
    classes = np.unique(labels)
    n_per_class = max(1, n // len(classes))
    idxs = []
    for c in classes:
        cidx = np.where(labels == c)[0]
        chosen = rng.choice(cidx, size=min(n_per_class, len(cidx)), replace=False)
        idxs.append(chosen)
    idxs = np.concatenate(idxs)
    # 若总数不足 n，补充随机样本
    if len(idxs) < n:
        extra = rng.choice(len(labels), size=n - len(idxs), replace=False)
        idxs = np.concatenate([idxs, extra])
    return idxs[:n]


# ============================================================================
# ★ Feature Extraction 模块：HOG
# ============================================================================

def extract_hog_features(
    images,
    orientations=HOG_ORIENTATIONS,
    pixels_per_cell=HOG_PIXELS_PER_CELL,
    cells_per_block=HOG_CELLS_PER_BLOCK,
    block_norm=HOG_BLOCK_NORM,
    log_fn=print,
):
    """
    对一批 28×28 灰度图像做 HOG 特征提取。

    对应 PEA 论文 Section V-A:
      "Classical feature extraction rules also have proven their
       effectiveness on various tasks."
    HOG 是真正的数据无关提取器：提取规则不依赖任何私有训练数据，
    所有参与方可以公开约定同一套参数，无需任何通信。

    参数
    ----
    images : np.ndarray, shape [N, H, W], float32, 归一化到 [0,1]

    返回
    ----
    features : np.ndarray, shape [N, hog_dim], float32
    """
    n_samples = len(images)
    # 用第一张图测试输出维度
    sample_feat = hog(
        images[0],
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        feature_vector=True,
    )
    hog_dim = len(sample_feat)
    log_fn(f"  HOG 配置: orientations={orientations}, "
           f"pixels_per_cell={pixels_per_cell}, "
           f"cells_per_block={cells_per_block}, "
           f"block_norm={block_norm}")
    log_fn(f"  HOG 输出维度: {hog_dim}")

    features = np.zeros((n_samples, hog_dim), dtype=np.float32)
    features[0] = sample_feat

    for i in tqdm(range(1, n_samples), desc="  HOG提取", ncols=80, leave=True):
        features[i] = hog(
            images[i],
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            feature_vector=True,
        )

    return features


# ============================================================================
# ★ TFE 密态多分类训练
# ============================================================================

def run_tfe_multiclass(
    x_train_feat: np.ndarray,
    y_train_int:  np.ndarray,
    x_test_feat:  np.ndarray,
    y_test_int:   np.ndarray,
    n_classes:    int,
    epochs:       int,
    lr:           float,
    log_fn,
):
    """
    在 TF-Encrypted ABY3 协议下训练单层 Dense 多分类器。

    多分类处理方式：
      - 标签转为 one-hot (float32)
      - 输出层 Dense(n_classes)，无激活函数
      - 损失函数 MSE（TFE 0.8.0 不支持 softmax/cross-entropy，与 baseline 保持一致）
      - 准确率：明文侧 argmax(reveal()) 计算

    返回
    ----
    history : list of dict，每个 epoch 的 loss / train_acc
    test_acc : float
    checks   : dict
    """
    n_train, d_feat = x_train_feat.shape
    y_train_oh = to_onehot(y_train_int, n_classes)   # [N, 10]

    history = []
    checks  = {}

    tf.compat.v1.reset_default_graph()

    with tfe.protocol.ABY3():

        # ----------------------------------------------------------------
        # 构建密态模型
        # Dense(10) 对应 10 类，无激活函数（MSE 损失不需要 softmax）
        # ----------------------------------------------------------------
        model = Sequential()
        model.add(Dense(n_classes, batch_input_shape=(n_train, d_feat)))

        # ----------------------------------------------------------------
        # 将提取后的低维特征加密分片
        # ★ 进入 MPC 的是 HOG 特征，不是原始像素
        # ----------------------------------------------------------------
        x_priv = tfe.define_private_variable(x_train_feat)
        y_priv = tfe.define_private_variable(y_train_oh)

        loss_fn   = tfe.keras.losses.MeanSquaredError()
        optimizer = tfe.keras.optimizers.SGD(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=loss_fn)
        model(x_priv)   # build graph

        fit_op, loss_tensor = model.fit_batch(x_priv, y_priv)
        pred_tensor         = model.predict(x_priv)

        sess = KE.get_session()
        sess.run(tf.compat.v1.global_variables_initializer())

        # 验证密态数据流
        probe     = x_priv.reveal()
        revealed  = sess.run(probe)
        checks["data_flow_ok"] = bool(revealed.shape == x_train_feat.shape)

        # 初始状态
        init_loss      = as_scalar_loss(sess.run(loss_tensor.reveal()))
        init_pred_np   = sess.run(pred_tensor.reveal())
        init_train_acc = accuracy_from_logits(init_pred_np, y_train_int)
        checks["loss_computable"] = bool(np.isfinite(init_loss))

        log_fn(f"  初始 Loss     : {init_loss:.6f}")
        log_fn(f"  初始 Train Acc: {init_train_acc:.4f}")
        log_fn("")
        log_fn("=" * 90)
        log_fn("{:<8} {:<14} {:<16} {:<14}".format(
            "Epoch", "Loss", "Train_Acc", "Time(s)"))
        log_fn("=" * 90)

        optimizer_ok = True
        epoch_start  = time.time()

        # 为避免大 epoch 时日志过长，只采样打印固定数量的 epoch 点
        log_epochs = set(np.linspace(1, epochs, num=min(MAX_EPOCH_LOG_LINES, epochs), dtype=int).tolist())
        log_epochs.add(1)
        log_epochs.add(epochs)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            bar = tqdm(
                total=1,
                desc=f"  Epoch {epoch:>2}/{epochs}",
                ncols=80,
                leave=False,
            )
            try:
                sess.run(fit_op)
            except Exception as exc:
                optimizer_ok = False
                log_fn(f"  Epoch {epoch}: 训练中断 → {exc}")
                bar.close()
                break

            loss_val  = as_scalar_loss(sess.run(loss_tensor.reveal()))
            pred_np   = sess.run(pred_tensor.reveal())
            train_acc = accuracy_from_logits(pred_np, y_train_int)
            w_ep, b_ep = sess.run([
                model.layers[0].weights[0].reveal(),
                model.layers[0].weights[1].reveal(),
            ])
            test_acc_ep = accuracy_from_logits(x_test_feat.dot(w_ep) + b_ep, y_test_int)
            elapsed   = time.time() - t0

            history.append({
                "epoch":          epoch,
                "loss":           round(loss_val, 6),
                "train_accuracy": round(train_acc, 6),
                "test_accuracy":  round(test_acc_ep, 6),
                "epoch_time_s":   round(elapsed, 3),
            })

            bar.set_postfix(
                loss=f"{loss_val:.4f}",
                train_acc=f"{train_acc:.4f}",
            )
            bar.update(1)
            bar.close()

            if epoch in log_epochs:
                log_fn("{:<8} {:<14.6f} {:<16.6f} {:<14.3f}".format(
                    epoch, loss_val, train_acc, elapsed))

        total_train_time = time.time() - epoch_start

        # ----------------------------------------------------------------
        # 测试集评估（明文侧）
        # 解密权重后在测试集 HOG 特征上做矩阵乘法
        # ----------------------------------------------------------------
        w_enc = model.layers[0].weights[0].reveal()
        b_enc = model.layers[0].weights[1].reveal()
        w_np, b_np = sess.run([w_enc, b_enc])   # shape: [d_feat, n_classes], [n_classes]

        test_logits = x_test_feat.dot(w_np) + b_np
        test_acc    = accuracy_from_logits(test_logits, y_test_int)
        if history and "test_accuracy" in history[-1]:
            test_acc = float(history[-1]["test_accuracy"])

        final_loss     = history[-1]["loss"]     if history else init_loss
        final_train_acc = history[-1]["train_accuracy"] if history else init_train_acc

        checks["backward_optimizer_ok"]   = bool(optimizer_ok)
        checks["loss_changed"]            = bool(abs(final_loss - init_loss) > 1e-6)
        checks["train_accuracy_changed"]  = bool(
            abs(final_train_acc - init_train_acc) > 1e-4
        )
        checks["loss_decreased"]          = bool(final_loss < init_loss)

        KE.clear_session()

    return history, test_acc, checks, total_train_time


def run_figure4_dp_budget_sweep(
    x_train_feat: np.ndarray,
    y_train_int: np.ndarray,
    x_test_feat: np.ndarray,
    y_test_int: np.ndarray,
    run_id: str,
    log_fn,
):
    """
    Figure4 风格：DP 预算 ε 与精度关系曲线。
    说明：复用模块 C 的 DPSGD 实现做 shallow training，保证 ε 会计口径一致。
    """
    import module_c_trust_dpsgd as mc

    rows = []
    log_fn("")
    log_fn("[Figure4] 模块A-DP shallow training（epsilon sweep）")
    log_fn(f"  eps_list      : {FIG4_EPSILON_LIST}")
    log_fn(f"  delta         : {FIG4_DELTA}")
    log_fn(f"  dp_epochs/lr  : {FIG4_DP_EPOCHS}/{FIG4_DP_LR}")

    for eps in FIG4_EPSILON_LIST:
        eps = float(eps)
        sigma = math.sqrt(2 * FIG4_DP_EPOCHS * math.log(1.0 / FIG4_DELTA)) / max(eps, 1e-12)
        log_fn(f"  -> epsilon={eps:.4f}, sigma={sigma:.6f}")

        out = mc.run_trust_dpsgd(
            init_model="random",
            feat_dim=int(x_train_feat.shape[1]),
            num_classes_hint=N_CLASSES,
            X_lg_feat=x_train_feat,
            y_lg=y_train_int,
            X_test_feat=x_test_feat,
            y_test=y_test_int,
            global_epochs=FIG4_DP_EPOCHS,
            global_lr=FIG4_DP_LR,
            batch_size=64,
            clipping_norm=1.0,
            noise_multiplier=float(sigma),
            delta=FIG4_DELTA,
            target_epsilon=eps,
            mode="fixed_dp",
            trust_t=0.0,
            sigma_min=float(sigma),
            sigma_max=float(sigma),
            seed=RANDOM_SEED,
            device_hint="cpu",
            max_epoch_log_lines=FIG4_MAX_EPOCH_LOG_LINES,
            safe_mode=True,
            cpu_threads=4,
            verbose=False,
        )

        fm = out["final_metrics"]
        row = {
            "epsilon_target": eps,
            "delta": FIG4_DELTA,
            "sigma": float(round(sigma, 6)),
            "test_acc": float(fm["final_test_accuracy"]),
            "train_acc": float(fm["final_train_accuracy"]),
            "epochs_ran": int(fm.get("epochs_ran", FIG4_DP_EPOCHS)),
            "epsilon_final": float(fm.get("final_epsilon_estimate", float("inf"))),
        }
        rows.append(row)
        log_fn("     test_acc={:.4f} train_acc={:.4f} epochs_ran={} epsilon_final={:.4f}".format(
            row["test_acc"], row["train_acc"], row["epochs_ran"], row["epsilon_final"]
        ))

    fig_json = FIG4_DIR / f"figure4_dp_mnist_{run_id}.json"
    with fig_json.open("w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "rows": rows}, f, indent=2, ensure_ascii=False)
        f.write("\n")

    fig_csv = FIG4_DIR / f"figure4_dp_mnist_{run_id}.csv"
    with fig_csv.open("w", encoding="utf-8") as f:
        f.write("epsilon_target,delta,sigma,test_acc,train_acc,epochs_ran,epsilon_final\n")
        for r in rows:
            f.write("{epsilon_target},{delta},{sigma},{test_acc},{train_acc},{epochs_ran},{epsilon_final}\n".format(**r))

    fig_png = None
    try:
        import matplotlib.pyplot as plt
        xs = [r["epsilon_target"] for r in rows]
        ys = [r["test_acc"] for r in rows]
        plt.figure(figsize=(7, 4.5))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Privacy Budget ε")
        plt.ylabel("Test Accuracy")
        plt.title("Figure4-style DP Curve (MNIST+HOG)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fig_png = FIG4_DIR / f"figure4_dp_mnist_{run_id}.png"
        plt.savefig(str(fig_png), dpi=160)
        plt.close()
    except Exception:
        fig_png = None

    return {
        "rows": rows,
        "json": str(fig_json),
        "csv": str(fig_csv),
        "png": str(fig_png) if fig_png is not None else None,
    }


# ============================================================================
# 主流程
# ============================================================================

def main():
    ensure_dirs()
    configure_logging()
    np.random.seed(RANDOM_SEED)
    tf.compat.v1.set_random_seed(RANDOM_SEED)

    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file    = LOG_DIR / f"exp1_mnist_hog_{run_ts}.log"
    report_file = LOG_DIR / "exp1_mnist_hog_report.json"
    log = make_logger(log_file)

    # ----------------------------------------------------------------
    # 实验头信息
    # ----------------------------------------------------------------
    log("=" * 70)
    log("实验 1：MNIST + HOG 特征提取 + TF-Encrypted 密态多分类")
    log("=" * 70)
    log(f"run_id           : {run_ts}")
    log(f"Python           : {sys.executable}")
    log(f"TensorFlow       : {tf.__version__}")
    log(f"TF Encrypted     : {getattr(tfe, '__version__', 'unknown')}")
    log(f"Platform         : {platform.platform()}")
    log("")
    log("[配置]")
    log(f"  MAX_TRAIN_SAMPLES  : {MAX_TRAIN_SAMPLES}")
    log(f"  MAX_TEST_SAMPLES   : {MAX_TEST_SAMPLES}")
    log(f"  HOG_ORIENTATIONS   : {HOG_ORIENTATIONS}")
    log(f"  HOG_PIXELS_PER_CELL: {HOG_PIXELS_PER_CELL}")
    log(f"  HOG_CELLS_PER_BLOCK: {HOG_CELLS_PER_BLOCK}")
    log(f"  HOG_BLOCK_NORM     : {HOG_BLOCK_NORM}")
    log(f"  EPOCHS             : {EPOCHS}")
    log(f"  LR                 : {LR}")
    log("")

    # 预先计算 HOG 理论维度（用于报告）
    hog_dim_theory = compute_hog_dim(
        HOG_IMAGE_SIZE,
        HOG_ORIENTATIONS,
        HOG_PIXELS_PER_CELL,
        HOG_CELLS_PER_BLOCK,
    )

    # ----------------------------------------------------------------
    # Step 1：加载数据
    # ----------------------------------------------------------------
    log("[1/4] 加载 MNIST 子集...")
    t_load = time.time()
    x_train_img, x_test_img, y_train_int, y_test_int = load_mnist_subset(
        max_train=MAX_TRAIN_SAMPLES,
        max_test=MAX_TEST_SAMPLES,
    )
    log(f"  训练集: {x_train_img.shape[0]} 样本, 图像尺寸: {x_train_img.shape[1:]}")
    log(f"  测试集: {x_test_img.shape[0]} 样本")
    log(f"  原始输入维度 (展平): {x_train_img.shape[1] * x_train_img.shape[2]}")
    log(f"  数据加载耗时: {time.time()-t_load:.2f}s")
    log("")

    raw_dim = x_train_img.shape[1] * x_train_img.shape[2]   # 784

    # ----------------------------------------------------------------
    # Step 2：本地明文 HOG 特征提取
    # ----------------------------------------------------------------
    log("[2/4] 本地明文侧 HOG 特征提取（不进入 MPC）...")
    t_hog = time.time()
    x_train_hog = extract_hog_features(
        x_train_img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        log_fn=log,
    )
    x_test_hog = extract_hog_features(
        x_test_img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        log_fn=lambda _: None,   # 测试集提取不重复打印配置
    )
    hog_feat_time = time.time() - t_hog

    actual_hog_dim = x_train_hog.shape[1]
    compression_ratio = actual_hog_dim / raw_dim

    # HOG 特征标准化（在明文侧，零均值单位方差）
    scaler = StandardScaler()
    x_train_hog = scaler.fit_transform(x_train_hog).astype(np.float32)
    x_test_hog  = scaler.transform(x_test_hog).astype(np.float32)

    # 将模块 a 特征缓存到 featureData，供模块 b/c 复用，避免重复提取。
    feat_cache_file = FEATURE_DATA_DIR / (
        f"mnist_hog_tr{len(y_train_int)}_te{len(y_test_int)}_seed{RANDOM_SEED}.npz"
    )
    np.savez(
        str(feat_cache_file),
        x_train_feat=x_train_hog.astype(np.float32),
        x_test_feat=x_test_hog.astype(np.float32),
        y_train=y_train_int.astype(np.int32),
        y_test=y_test_int.astype(np.int32),
        extractor="hog",
        dataset="mnist",
    )

    log(f"  ✓ 特征提取完成:")
    log(f"    原始维度      : {raw_dim}")
    log(f"    HOG 特征维度  : {actual_hog_dim}  (理论值: {hog_dim_theory})")
    log(f"    压缩比例      : {compression_ratio:.4f}  "
        f"({actual_hog_dim}/{raw_dim} = {compression_ratio*100:.1f}%)")
    log(f"    HOG提取耗时   : {hog_feat_time:.2f}s")
    log(f"    特征缓存路径   : {feat_cache_file}")
    log("")

    # ----------------------------------------------------------------
    # Step 3：TFE 密态多分类训练
    # ----------------------------------------------------------------
    log("[3/4] TFE 密态训练（输入为 HOG 特征）...")
    log(f"  进入 MPC 的特征维度: {actual_hog_dim}  "
        f"（原始 {raw_dim} 维压缩至 {compression_ratio*100:.1f}%）")
    log("")

    history, test_acc, checks, train_time = run_tfe_multiclass(
        x_train_feat=x_train_hog,
        y_train_int=y_train_int,
        x_test_feat=x_test_hog,
        y_test_int=y_test_int,
        n_classes=N_CLASSES,
        epochs=EPOCHS,
        lr=LR,
        log_fn=log,
    )

    figure4_dp = None
    if RUN_FIGURE4_DP:
        try:
            figure4_dp = run_figure4_dp_budget_sweep(
                x_train_feat=x_train_hog,
                y_train_int=y_train_int,
                x_test_feat=x_test_hog,
                y_test_int=y_test_int,
                run_id=run_ts,
                log_fn=log,
            )
            log("")
            log("[Figure4] 文件输出:")
            log(f"  json : {figure4_dp['json']}")
            log(f"  csv  : {figure4_dp['csv']}")
            log(f"  png  : {figure4_dp['png']}")
            log("")
        except Exception as exc:
            log("")
            log(f"[Figure4] 运行失败（不中断主实验）: {exc}")
            log("")

    # ----------------------------------------------------------------
    # Step 4：汇总报告
    # ----------------------------------------------------------------
    log("")
    log("[4/4] 实验结果汇总")
    log("=" * 70)

    final_train_acc = history[-1]["train_accuracy"] if history else 0.0
    final_loss      = history[-1]["loss"]            if history else 0.0

    log(f"  原始输入维度         : {raw_dim}")
    log(f"  HOG 特征维度         : {actual_hog_dim}")
    log(f"  特征压缩比           : {compression_ratio*100:.1f}%")
    log(f"  HOG 提取耗时         : {hog_feat_time:.2f}s")
    log(f"  总密态训练耗时       : {train_time:.2f}s")
    log(f"  平均每 epoch 耗时    : {train_time/EPOCHS:.2f}s")
    log(f"  最终 Train Loss      : {final_loss:.6f}")
    log(f"  最终 Train Accuracy  : {final_train_acc:.4f}")
    log(f"  最终 Test  Accuracy  : {test_acc:.4f}")
    log("")

    must_have = [
        "data_flow_ok",
        "loss_computable",
        "backward_optimizer_ok",
        "loss_changed",
    ]
    exp_ok = all(checks.get(k, False) for k in must_have)
    log(f"  实验验证状态: {'✓ PASS' if exp_ok else '✗ FAIL'}")
    for k, v in checks.items():
        log(f"    {k:<35}: {v}")

    # ----------------------------------------------------------------
    # 写入 JSON 报告
    # ----------------------------------------------------------------
    report = {
        "meta": {
            "experiment":    "exp1_mnist_hog",
            "run_id":        run_ts,
            "python":        sys.version.split()[0],
            "platform":      platform.platform(),
            "tensorflow":    tf.__version__,
            "tf_encrypted":  getattr(tfe, "__version__", "unknown"),
        },
        "config": {
            "dataset":            "MNIST",
            "max_train_samples":  MAX_TRAIN_SAMPLES,
            "max_test_samples":   MAX_TEST_SAMPLES,
            "random_seed":        RANDOM_SEED,
            "extractor":          "HOG",
            "hog_orientations":   HOG_ORIENTATIONS,
            "hog_pixels_per_cell":list(HOG_PIXELS_PER_CELL),
            "hog_cells_per_block":list(HOG_CELLS_PER_BLOCK),
            "hog_block_norm":     HOG_BLOCK_NORM,
            "n_classes":          N_CLASSES,
            "epochs":             EPOCHS,
            "lr":                 LR,
            "tfe_protocol":       "ABY3",
            "model":              "single_dense_no_activation",
            "loss":               "MSE",
        },
        "feature_extraction": {
            "raw_dim":            raw_dim,
            "hog_feat_dim":       actual_hog_dim,
            "compression_ratio":  round(compression_ratio, 4),
            "hog_extract_time_s": round(hog_feat_time, 3),
        },
        "training": {
            "total_train_time_s":    round(train_time, 3),
            "avg_epoch_time_s":      round(train_time / EPOCHS, 3),
            "final_train_loss":      final_loss,
            "final_train_accuracy":  final_train_acc,
            "final_test_accuracy":   test_acc,
        },
        "figure4_dp": figure4_dp,
        "checks":  checks,
        "exp_pass": exp_ok,
        "history": history,
    }

    with report_file.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, sort_keys=False, ensure_ascii=False)
        fp.write("\n")

    log(f"\n  报告已写入: {report_file}")
    log(f"  日志已写入: {log_file}")
    log("=" * 70)

    return 0 if exp_ok else 1


if __name__ == "__main__":
    sys.exit(main())

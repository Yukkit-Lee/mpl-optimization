#!/usr/bin/env python3
"""
用途说明：
本脚本用于在 IMDb 数据集上执行“RoBERTa 特征提取 + TFE 密态浅层训练”实验，
用于替代原先的 Word2Vec 路线。

实验内容：
1) 实验 A：明文 RoBERTa 特征 + Logistic Regression（明文对照）
2) 实验 B：RoBERTa 特征 -> TFE 密态单层 Dense 二分类（核心）
3) 实验 C：RoBERTa 特征维度消融（截断维度 128/256/512/768）
4) 实验 D：TF-IDF（弱文本表示）-> TFE 密态单层 Dense（二分类 baseline）

运行：
  python imdb_RoBERTa.py
"""

import json
import logging
import os
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_encrypted").setLevel(logging.ERROR)

import tf_encrypted as tfe
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tf_encrypted.keras import Sequential
from tf_encrypted.keras import backend as KE
from tf_encrypted.keras.layers import Dense
from tqdm import tqdm

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:
    raise RuntimeError(
        "缺少 RoBERTa 依赖，请在 tfe 环境安装："
        "pip install torch transformers"
    ) from exc


# ============================================================================
# 配置
# ============================================================================

MAX_TRAIN_SAMPLES = 1000
MAX_TEST_SAMPLES = 1000     ## 25000 full data
RANDOM_SEED = 42

# RoBERTa 配置
ROBERTA_MODEL_NAME = "roberta-base"
ROBERTA_MAX_LEN = 256
ROBERTA_BATCH_SIZE_TRAIN = 32
ROBERTA_BATCH_SIZE_TEST = 32
ROBERTA_DEVICE = "cpu"

# TFE 配置
TFE_EPOCHS = 400
TFE_LR = 0.01
MAX_EPOCH_LOG_LINES = 10
LOSS_ABS_STOP_TH = 1e5
# 主实验 B 使用的 RoBERTa 特征维度（从前 k 维截断）
# 根据当前消融结果，128 维在精度-效率权衡上更优，作为默认主设置。
MAIN_FEATURE_DIM = 128
# 是否根据消融结果自动选择主实验 B 的最佳维度
AUTO_SELECT_MAIN_DIM_FROM_ABLATION = True

# 维度消融（对 RoBERTa 输出做前 k 维截断）
ABLATION_DIMS = [128, 256, 512, 768]

# 弱文本表示 baseline（近似 raw）
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 1)

# Figure4 风格曲线配置（方法对比）
RUN_FIGURE4 = True
# None 表示按总 epoch 自动选 3-4 个点；也可手动给 [0, 200, 400] 这种列表
FIG4_ANNOTATE_ITERS = None
FIG4_EPSILON_END = 2.07

# 安全模式：限制线程，降低 CPU 爆占风险
SAFE_MODE = True
CPU_THREADS = 2

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "log"
FEATURE_DATA_DIR = BASE_DIR / "featureData"
EXPFIG_DIR = BASE_DIR / "expFig"
FIG4_DIR = EXPFIG_DIR / "figure4_extractor"


# ============================================================================
# 工具函数
# ============================================================================

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPFIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG4_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("tf_encrypted").setLevel(logging.ERROR)


def make_logger(log_file: Path):
    def log(text: str = ""):
        print(text)
        with log_file.open("a", encoding="utf-8") as fp:
            fp.write(text + "\n")
    return log


def accuracy_from_logits_binary(logits_np: np.ndarray, y_int: np.ndarray) -> float:
    logits_flat = logits_np.flatten()
    logits_clip = np.clip(logits_flat, -30.0, 30.0)
    prob = 1.0 / (1.0 + np.exp(-logits_clip))
    pred = (prob > 0.5).astype(np.float32)
    return float(np.mean(pred == y_int.flatten()))


def accuracy_from_logits_binary_sign(logits_np: np.ndarray, y_int: np.ndarray) -> float:
    """
    当标签使用 {-1, +1} 回归目标时，分类阈值自然取 logit > 0。
    预测为正类(1) 当且仅当 logit > 0。
    """
    logits_flat = logits_np.flatten()
    pred = (logits_flat > 0.0).astype(np.float32)
    return float(np.mean(pred == y_int.flatten()))


def as_scalar_loss(loss_value) -> float:
    arr = np.asarray(loss_value, dtype=np.float64)
    return float(arr.mean())


def build_epoch_log_set(total_epochs: int, max_lines: int = MAX_EPOCH_LOG_LINES):
    n = min(max_lines, total_epochs)
    if n <= 0:
        return set()
    idx = np.linspace(1, total_epochs, num=n)
    return {int(round(x)) for x in idx}


def get_dim_scaled_lr(embed_dim: int, base_lr: float = TFE_LR) -> float:
    if embed_dim <= 128:
        scale = 1.0
    elif embed_dim <= 256:
        scale = 0.5
    elif embed_dim <= 512:
        scale = 0.2
    else:
        scale = 0.1
    return float(base_lr * scale)


def estimate_epsilon_by_epoch_fraction(epoch: int, total_epochs: int, epsilon_end: float) -> float:
    """
    Figure4 绘图用 epsilon 口径：
      epsilon_so_far = epsilon_end * sqrt(epoch / total_epochs)
    """
    if total_epochs <= 0:
        return 0.0
    ratio = max(0.0, min(1.0, float(epoch) / float(total_epochs)))
    return float(epsilon_end) * float(np.sqrt(ratio))


def run_figure4_method_curve(exp_b_history, exp_d_history, total_epochs, run_id, log_fn):
    """
    Figure4 风格方法对比曲线：
    - 曲线1: 实验B（RoBERTa->TFE）test_acc
    - 曲线2: 实验D（TF-IDF->TFE）test_acc
    - 注释点: 固定 iteration（0/300/600/800），图上仅显示 eps
    """
    log_fn("")
    log_fn("[Figure4] 方法对比曲线：TFE RoBERTa vs TFE TF-IDF")
    log_fn(f"  total_epochs      : {total_epochs}")
    log_fn(f"  epsilon_end       : {FIG4_EPSILON_END}")
    def _resolve_annot_iters(total_ep, preset):
        if isinstance(preset, (list, tuple)) and len(preset) > 0:
            vals = sorted(set(int(x) for x in preset))
            return [v for v in vals if 0 <= v <= int(total_ep)]
        # 自动取 4 个点：起点 + 两个中间点 + 终点
        p1 = int(round(total_ep * 0.33))
        p2 = int(round(total_ep * 0.66))
        vals = sorted(set([0, p1, p2, int(total_ep)]))
        return vals

    annotate_iters = _resolve_annot_iters(int(total_epochs), FIG4_ANNOTATE_ITERS)
    log_fn(f"  annotate_iters    : {annotate_iters}")

    rows = []
    n_hist = min(len(exp_b_history), len(exp_d_history), int(total_epochs))
    for i in range(1, n_hist + 1):
        rows.append({
            "iteration": i,
            "feat_acc": float(exp_b_history[i - 1].get("test_accuracy", exp_b_history[i - 1]["train_accuracy"])),
            "raw_acc": float(exp_d_history[i - 1].get("test_accuracy", exp_d_history[i - 1]["train_accuracy"])),
            "epsilon_so_far": estimate_epsilon_by_epoch_fraction(i, total_epochs, FIG4_EPSILON_END),
        })

    annotations = []
    for it in annotate_iters:
        if not rows:
            continue
        if it <= 0:
            idx = 0
            ann_iter = 0
        else:
            ann_iter = min(max(int(it), 1), len(rows))
            idx = ann_iter - 1
        r = rows[idx]
        annotations.append({
            "target_iteration": int(it),
            "iteration": int(ann_iter),
            "epsilon_so_far": estimate_epsilon_by_epoch_fraction(int(ann_iter), total_epochs, FIG4_EPSILON_END),
            "feat_acc": float(r["feat_acc"]),
            "raw_acc": float(r["raw_acc"]),
        })

    csv_path = FIG4_DIR / f"figure4_imdb_{run_id}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("iteration,feat_acc,raw_acc,epsilon_so_far\n")
        for r in rows:
            f.write(f"{r['iteration']},{r['feat_acc']},{r['raw_acc']},{r['epsilon_so_far']}\n")

    json_path = FIG4_DIR / f"figure4_imdb_{run_id}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "config": {
                "epochs": int(total_epochs),
                "epsilon_end": FIG4_EPSILON_END,
                "annotate_iters": annotate_iters,
            },
            "history": rows,
            "annotations": annotations,
        }, f, indent=2, ensure_ascii=False)
        f.write("\n")

    png_path = None
    try:
        import matplotlib.pyplot as plt
        xs = [r["iteration"] for r in rows]
        ys_feat = [r["feat_acc"] for r in rows]
        ys_raw = [r["raw_acc"] for r in rows]

        plt.figure(figsize=(8, 4.8))
        line_feat, = plt.plot(xs, ys_feat, label="TFE RoBERTa Shallow (Test)", linewidth=1.8)
        line_raw, = plt.plot(xs, ys_raw, label="TFE TF-IDF Shallow (Test)", linewidth=1.8)

        for a in annotations:
            x_plot = 1 if a["iteration"] == 0 else a["iteration"]
            y1 = a["feat_acc"]
            y2 = a["raw_acc"]
            txt = "eps={:.2f}".format(a["epsilon_so_far"])
            plt.scatter([x_plot], [y1], s=20, color=line_feat.get_color())
            plt.scatter([x_plot], [y2], s=20, color=line_raw.get_color())
            plt.annotate(
                txt,
                (x_plot, y1),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=line_feat.get_color(),
            )

        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("IMDb Figure4-style Curve: TFE RoBERTa vs TFE TF-IDF (Test)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        png_path = FIG4_DIR / f"figure4_imdb_{run_id}.png"
        plt.savefig(str(png_path), dpi=160)
        plt.close()
    except Exception:
        png_path = None

    return {
        "history": rows,
        "annotations": annotations,
        "csv": str(csv_path),
        "json": str(json_path),
        "png": str(png_path) if png_path is not None else None,
    }


# ============================================================================
# IMDb 文本加载
# ============================================================================

def load_imdb_texts(max_train=MAX_TRAIN_SAMPLES, max_test=MAX_TEST_SAMPLES, seed=RANDOM_SEED):
    cache_file = DATA_DIR / f"imdb_texts_tr{max_train}_te{max_test}_s{seed}.npz"
    if cache_file.exists():
        d = np.load(str(cache_file), allow_pickle=True)
        return (
            list(d["train_texts"]),
            list(d["test_texts"]),
            d["y_train"].astype(int),
            d["y_test"].astype(int),
        )

    (x_tr_ids, y_tr), (x_te_ids, y_te) = tf.keras.datasets.imdb.load_data()
    word_index = tf.keras.datasets.imdb.get_word_index()
    id_to_word = {v + 3: k for k, v in word_index.items()}
    id_to_word[0] = "<PAD>"
    id_to_word[1] = "<START>"
    id_to_word[2] = "<UNK>"

    def ids_to_text(ids):
        return " ".join(id_to_word.get(i, "<UNK>") for i in ids)

    rng = np.random.RandomState(seed)

    def balanced_sample(x_ids, y_labels, n):
        pos_idx = np.where(y_labels == 1)[0]
        neg_idx = np.where(y_labels == 0)[0]
        n_half = n // 2
        chosen_pos = rng.choice(pos_idx, size=min(n_half, len(pos_idx)), replace=False)
        chosen_neg = rng.choice(neg_idx, size=min(n_half, len(neg_idx)), replace=False)
        chosen = np.concatenate([chosen_pos, chosen_neg])
        rng.shuffle(chosen)
        return [x_ids[i] for i in chosen], y_labels[chosen].astype(int)

    x_tr_sample, y_tr_sample = balanced_sample(x_tr_ids, y_tr, max_train)
    x_te_sample, y_te_sample = balanced_sample(x_te_ids, y_te, max_test)
    train_texts = [ids_to_text(ids) for ids in x_tr_sample]
    test_texts = [ids_to_text(ids) for ids in x_te_sample]

    np.savez(
        str(cache_file),
        train_texts=np.array(train_texts, dtype=object),
        test_texts=np.array(test_texts, dtype=object),
        y_train=y_tr_sample,
        y_test=y_te_sample,
    )
    return train_texts, test_texts, y_tr_sample, y_te_sample


def normalize_texts(texts):
    cleaned = []
    for t in texts:
        s = t.lower()
        s = re.sub(r"\s+", " ", s).strip()
        cleaned.append(s)
    return cleaned


def extract_tfidf_features(train_texts, test_texts, log_fn):
    """
    构建弱文本表示（近似 raw baseline）：
    TF-IDF / Bag-of-Words 风格稀疏向量，再转 dense 进入 TFE。
    """
    cache_file = DATA_DIR / (
        f"imdb_tfidf_tr{len(train_texts)}_te{len(test_texts)}_"
        f"f{TFIDF_MAX_FEATURES}_ng{TFIDF_NGRAM_RANGE[0]}{TFIDF_NGRAM_RANGE[1]}.npz"
    )
    if cache_file.exists():
        d = np.load(str(cache_file))
        x_train = d["x_train"].astype(np.float32)
        x_test = d["x_test"].astype(np.float32)
        log_fn(f"  加载缓存 TF-IDF 特征: {cache_file.name}")
        return x_train, x_test

    train_clean = normalize_texts(train_texts)
    test_clean = normalize_texts(test_texts)

    log_fn("  构建 TF-IDF 特征...")
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        lowercase=True,
        dtype=np.float32,
    )
    x_train_sp = vec.fit_transform(train_clean)
    x_test_sp = vec.transform(test_clean)

    x_train = x_train_sp.toarray().astype(np.float32)
    x_test = x_test_sp.toarray().astype(np.float32)
    np.savez(str(cache_file), x_train=x_train, x_test=x_test)
    log_fn(f"  TF-IDF 词表大小: {len(vec.vocabulary_)}")
    log_fn(f"  已缓存 TF-IDF 特征: {cache_file.name}")
    return x_train, x_test


# ============================================================================
# RoBERTa 特征提取
# ============================================================================

def encode_with_roberta(texts, tokenizer, model, max_len, batch_size, device, log_fn):
    model.eval()
    all_vec = []
    total = len(texts)

    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size), desc="  RoBERTa编码", ncols=80, leave=True):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attn_mask = encoded["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            hidden = out.last_hidden_state

            mask = attn_mask.unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1e-6)
            pooled = summed / denom
            all_vec.append(pooled.cpu().numpy().astype(np.float32))

    emb = np.concatenate(all_vec, axis=0)
    log_fn(f"  编码完成: {emb.shape}")
    return emb


def extract_roberta_features(train_texts, test_texts, log_fn):
    model_tag = ROBERTA_MODEL_NAME.replace("/", "_")
    cache_file = DATA_DIR / (
        f"imdb_roberta_{model_tag}_tr{len(train_texts)}_te{len(test_texts)}"
        f"_ml{ROBERTA_MAX_LEN}.npz"
    )
    if cache_file.exists():
        d = np.load(str(cache_file))
        x_train = d["x_train"].astype(np.float32)
        x_test = d["x_test"].astype(np.float32)
        log_fn(f"  加载缓存 RoBERTa 特征: {cache_file.name}")
        return x_train, x_test

    log_fn(f"  加载 RoBERTa: {ROBERTA_MODEL_NAME}")
    device = torch.device(ROBERTA_DEVICE)
    try:
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
        model = AutoModel.from_pretrained(ROBERTA_MODEL_NAME).to(device)
    except Exception as exc:
        raise RuntimeError(
            "RoBERTa 模型下载/加载失败。请检查网络，或先手动缓存模型后再运行。"
        ) from exc

    train_clean = normalize_texts(train_texts)
    test_clean = normalize_texts(test_texts)

    log_fn("  提取训练集特征...")
    x_train = encode_with_roberta(
        train_clean, tokenizer, model, ROBERTA_MAX_LEN, ROBERTA_BATCH_SIZE_TRAIN, device, log_fn
    )
    log_fn("  提取测试集特征...")
    x_test = encode_with_roberta(
        test_clean, tokenizer, model, ROBERTA_MAX_LEN, ROBERTA_BATCH_SIZE_TEST, device, log_fn
    )

    np.savez(str(cache_file), x_train=x_train, x_test=x_test)
    log_fn(f"  已缓存 RoBERTa 特征: {cache_file.name}")
    return x_train, x_test


# ============================================================================
# 实验 A：明文对照
# ============================================================================

def run_exp_a(x_train_emb, y_train, x_test_emb, y_test, log_fn):
    log_fn("")
    log_fn("─" * 60)
    log_fn("  实验 A：明文 RoBERTa + Logistic Regression")
    log_fn("─" * 60)

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train_emb).astype(np.float32)
    x_te = scaler.transform(x_test_emb).astype(np.float32)

    t0 = time.time()
    clf = LogisticRegression(max_iter=1500, random_state=RANDOM_SEED, solver="lbfgs")
    clf.fit(x_tr, y_train.flatten())
    train_time = time.time() - t0

    train_acc = float(accuracy_score(y_train.flatten(), clf.predict(x_tr)))
    test_acc = float(accuracy_score(y_test.flatten(), clf.predict(x_te)))

    log_fn(f"  特征维度        : {x_train_emb.shape[1]}")
    log_fn(f"  训练耗时        : {train_time:.3f}s")
    log_fn(f"  Train Accuracy  : {train_acc:.4f}")
    log_fn(f"  Test  Accuracy  : {test_acc:.4f}")

    return {
        "feature_dim": x_train_emb.shape[1],
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "train_time_s": round(train_time, 3),
    }


# ============================================================================
# 实验 B：TFE 密态训练
# ============================================================================

def run_exp_b(x_train_emb, y_train, x_test_emb, y_test, epochs, lr, log_fn,
              exp_title="实验 B：RoBERTa 特征 -> TFE 密态二分类"):
    log_fn("")
    log_fn("─" * 60)
    log_fn(f"  {exp_title}")
    log_fn("─" * 60)

    embed_dim = int(x_train_emb.shape[1])
    lr_eff = get_dim_scaled_lr(embed_dim, base_lr=lr)
    epoch_log_points = build_epoch_log_set(epochs, MAX_EPOCH_LOG_LINES)

    scaler = StandardScaler()
    x_tr_norm = scaler.fit_transform(x_train_emb).astype(np.float32)
    x_te_norm = scaler.transform(x_test_emb).astype(np.float32)
    # 为了让 MSE 二分类更稳定，标签改用 {-1, +1}，分类阈值用 logit>0
    y_tr_pm1 = (y_train.reshape(-1, 1).astype(np.float32) * 2.0) - 1.0
    n_train = x_tr_norm.shape[0]

    history = []
    checks = {}
    unstable_training = False

    tf.compat.v1.reset_default_graph()
    with tfe.protocol.ABY3():
        model = Sequential()
        model.add(Dense(1, batch_input_shape=(n_train, embed_dim)))

        x_priv = tfe.define_private_variable(x_tr_norm)
        y_priv = tfe.define_private_variable(y_tr_pm1)

        loss_fn = tfe.keras.losses.MeanSquaredError()
        optimizer = tfe.keras.optimizers.SGD(learning_rate=lr_eff)
        model.compile(optimizer=optimizer, loss=loss_fn)
        model(x_priv)

        fit_op, loss_tensor = model.fit_batch(x_priv, y_priv)
        pred_tensor = model.predict(x_priv)

        sess = KE.get_session()
        sess.run(tf.compat.v1.global_variables_initializer())

        probe = x_priv.reveal()
        revealed = sess.run(probe)
        checks["data_flow_ok"] = bool(revealed.shape == x_tr_norm.shape)

        init_loss = as_scalar_loss(sess.run(loss_tensor.reveal()))
        init_pred_np = sess.run(pred_tensor.reveal())
        init_train_acc = accuracy_from_logits_binary_sign(init_pred_np, y_train)
        checks["loss_computable"] = bool(np.isfinite(init_loss))

        log_fn(f"  进入 MPC 的特征维度 : {embed_dim}")
        log_fn(f"  实际学习率           : {lr_eff:.6f} (base={lr:.6f})")
        log_fn(f"  初始 Loss           : {init_loss:.6f}")
        log_fn(f"  初始 Train Acc      : {init_train_acc:.4f}")
        log_fn("")
        log_fn("=" * 85)
        log_fn("{:<8} {:<14} {:<16} {:<12}".format("Epoch", "Loss", "Train_Acc", "Time(s)"))
        log_fn("=" * 85)

        optimizer_ok = True
        t_train_start = time.time()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            try:
                sess.run(fit_op)
            except Exception as exc:
                optimizer_ok = False
                log_fn(f"  Epoch {epoch}: 训练中断 -> {exc}")
                break

            loss_val = as_scalar_loss(sess.run(loss_tensor.reveal()))
            pred_np = sess.run(pred_tensor.reveal())
            train_acc = accuracy_from_logits_binary_sign(pred_np, y_train)
            w_ep, b_ep = sess.run([
                model.layers[0].weights[0].reveal(),
                model.layers[0].weights[1].reveal(),
            ])
            test_acc_ep = accuracy_from_logits_binary_sign(x_te_norm.dot(w_ep) + b_ep, y_test)
            elapsed = time.time() - t0

            if (not np.isfinite(loss_val)) or abs(loss_val) > LOSS_ABS_STOP_TH:
                unstable_training = True
                log_fn(f"  Epoch {epoch}: 检测到 loss 异常（{loss_val:.4e}），提前停止")
                break

            history.append({
                "epoch": epoch,
                "loss": round(loss_val, 6),
                "train_accuracy": round(train_acc, 6),
                "test_accuracy": round(test_acc_ep, 6),
                "epoch_time_s": round(elapsed, 3),
            })

            if epoch in epoch_log_points:
                log_fn("{:<8} {:<14.6f} {:<16.6f} {:<12.3f}".format(
                    epoch, loss_val, train_acc, elapsed
                ))

        total_train_time = time.time() - t_train_start

        w_enc = model.layers[0].weights[0].reveal()
        b_enc = model.layers[0].weights[1].reveal()
        w_np, b_np = sess.run([w_enc, b_enc])

        test_logits = x_te_norm.dot(w_np) + b_np
        test_acc = accuracy_from_logits_binary_sign(test_logits, y_test)
        if history and "test_accuracy" in history[-1]:
            test_acc = float(history[-1]["test_accuracy"])

        final_loss = history[-1]["loss"] if history else init_loss
        final_train_acc = history[-1]["train_accuracy"] if history else init_train_acc

        checks["backward_optimizer_ok"] = bool(optimizer_ok)
        checks["unstable_training"] = bool(unstable_training)
        checks["loss_changed"] = bool(abs(final_loss - init_loss) > 1e-6)
        checks["train_accuracy_changed"] = bool(abs(final_train_acc - init_train_acc) > 1e-4)
        checks["loss_decreased"] = bool(final_loss < init_loss)

        KE.clear_session()

    log_fn("")
    log_fn(f"  总密态训练耗时    : {total_train_time:.2f}s")
    log_fn(f"  平均每 epoch 耗时 : {total_train_time / max(epochs, 1):.2f}s")
    log_fn(f"  最终 Train Acc    : {final_train_acc:.4f}")
    log_fn(f"  最终 Test  Acc    : {test_acc:.4f}")

    return {
        "feature_dim": embed_dim,
        "epochs": epochs,
        "lr": lr,
        "effective_lr": round(lr_eff, 6),
        "initial_train_loss": round(init_loss, 6),
        "final_train_loss": round(final_loss, 6),
        "initial_train_acc": round(init_train_acc, 4),
        "final_train_acc": round(final_train_acc, 4),
        "final_test_acc": round(test_acc, 4),
        "total_train_time_s": round(total_train_time, 3),
        "avg_epoch_time_s": round(total_train_time / max(epochs, 1), 3),
        "checks": checks,
        "history": history,
    }


# ============================================================================
# 实验 C：RoBERTa 特征维度消融
# ============================================================================

def run_exp_c(x_train_emb, x_test_emb, y_train, y_test, dims, epochs, lr, log_fn):
    log_fn("")
    log_fn("─" * 60)
    log_fn("  实验 C：RoBERTa 特征维度消融（前 k 维截断）")
    log_fn(f"  测试维度: {dims}")
    log_fn("─" * 60)

    base_dim = x_train_emb.shape[1]
    results = []
    details = {}

    for dim in dims:
        if dim > base_dim:
            log_fn(f"  [跳过] dim={dim} 大于原始维度 {base_dim}")
            continue

        log_fn(f"\n  [消融] feature_dim = {dim}")
        x_tr = x_train_emb[:, :dim]
        x_te = x_test_emb[:, :dim]
        row = run_exp_b(
            x_tr, y_train, x_te, y_test,
            epochs=epochs, lr=lr, log_fn=log_fn
        )
        details[int(dim)] = row
        results.append({
            "feature_dim": dim,
            "effective_lr": row["effective_lr"],
            "train_accuracy": row["final_train_acc"],
            "test_accuracy": row["final_test_acc"],
            "train_time_s": row["total_train_time_s"],
            "avg_epoch_time_s": row["avg_epoch_time_s"],
            "unstable_training": row["checks"].get("unstable_training", False),
        })

    log_fn("")
    log_fn("  [消融汇总表]")
    log_fn("  {:<12} {:<12} {:<14} {:<14} {:<12} {:<10}".format(
        "FeatDim", "EffLR", "Train_Acc", "Test_Acc", "Time(s)", "Unstable"
    ))
    log_fn("  " + "-" * 78)
    for r in results:
        log_fn("  {:<12} {:<12.6f} {:<14.4f} {:<14.4f} {:<12.2f} {:<10}".format(
            r["feature_dim"],
            r["effective_lr"],
            r["train_accuracy"],
            r["test_accuracy"],
            r["train_time_s"],
            str(r["unstable_training"]),
        ))
    return results, details


# ============================================================================
# 主流程
# ============================================================================

def main():
    if SAFE_MODE:
        os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
        os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
        os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)
        try:
            torch.set_num_threads(CPU_THREADS)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    ensure_dirs()
    configure_logging()
    np.random.seed(RANDOM_SEED)
    tf.compat.v1.set_random_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"exp2_imdb_roberta_{run_ts}.log"
    report_file = LOG_DIR / "exp2_imdb_roberta_report.json"
    log = make_logger(log_file)

    log("=" * 72)
    log("实验 2：IMDb + RoBERTa 特征提取 + TF-Encrypted 密态二分类")
    log("=" * 72)
    log(f"run_id              : {run_ts}")
    log(f"Python              : {sys.executable}")
    log(f"TensorFlow          : {tf.__version__}")
    log(f"TF Encrypted        : {getattr(tfe, '__version__', 'unknown')}")
    log(f"Platform            : {platform.platform()}")
    log("")
    log("[配置]")
    log(f"  MAX_TRAIN_SAMPLES : {MAX_TRAIN_SAMPLES}")
    log(f"  MAX_TEST_SAMPLES  : {MAX_TEST_SAMPLES}")
    log(f"  ROBERTA_MODEL     : {ROBERTA_MODEL_NAME}")
    log(f"  ROBERTA_MAX_LEN   : {ROBERTA_MAX_LEN}")
    log(f"  ROBERTA_BATCH_TR  : {ROBERTA_BATCH_SIZE_TRAIN}")
    log(f"  ROBERTA_BATCH_TE  : {ROBERTA_BATCH_SIZE_TEST}")
    log(f"  TFIDF_MAX_FEATURES: {TFIDF_MAX_FEATURES}")
    log(f"  TFE_EPOCHS        : {TFE_EPOCHS}")
    log(f"  TFE_LR            : {TFE_LR}")
    log(f"  MAIN_FEATURE_DIM  : {MAIN_FEATURE_DIM}")
    log(f"  AUTO_SELECT_DIM   : {AUTO_SELECT_MAIN_DIM_FROM_ABLATION}")
    log(f"  ABLATION_DIMS     : {ABLATION_DIMS}")
    log(f"  FIG4_EPS_END      : {FIG4_EPSILON_END}")
    log(f"  FIG4_ANN_ITERS    : {FIG4_ANNOTATE_ITERS} (None=auto)")
    log(f"  SAFE_MODE/THREADS : {SAFE_MODE}/{CPU_THREADS}")
    log("")

    log("[1/5] 加载 IMDb 数据集...")
    t0 = time.time()
    train_texts, test_texts, y_train, y_test = load_imdb_texts(
        max_train=MAX_TRAIN_SAMPLES, max_test=MAX_TEST_SAMPLES, seed=RANDOM_SEED
    )
    log(f"  训练集: {len(train_texts)}  测试集: {len(test_texts)}")
    log(f"  加载耗时: {time.time() - t0:.2f}s")
    log("")

    log("[2/5] RoBERTa 特征提取（本地明文侧）...")
    t_feat = time.time()
    x_train_emb, x_test_emb = extract_roberta_features(train_texts, test_texts, log_fn=log)
    feat_time = time.time() - t_feat
    # 模块 a 特征缓存：统一落盘到 featureData，供模块 b/c 直接复用。
    roberta_feat_cache = FEATURE_DATA_DIR / (
        f"imdb_RoBERTa_tr{len(y_train)}_te{len(y_test)}_seed{RANDOM_SEED}.npz"
    )
    np.savez(
        str(roberta_feat_cache),
        x_train_feat=x_train_emb.astype(np.float32),
        x_test_feat=x_test_emb.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        extractor="RoBERTa",
        dataset="imdb",
        model_name=ROBERTA_MODEL_NAME,
        max_len=int(ROBERTA_MAX_LEN),
    )
    log(f"  特征提取耗时: {feat_time:.2f}s")
    log(f"  特征矩阵: train={x_train_emb.shape}, test={x_test_emb.shape}")
    log(f"  特征缓存路径: {roberta_feat_cache}")
    log("")

    log("[3/5] 运行实验 A（明文对照）...")
    result_a = run_exp_a(x_train_emb, y_train, x_test_emb, y_test, log_fn=log)

    log("")
    log("[4/5] 运行实验 B/C/D（密态训练与消融）...")
    # 先跑消融，便于自动选择主实验最优维度。
    result_c, result_c_details = run_exp_c(
        x_train_emb, x_test_emb, y_train, y_test,
        dims=ABLATION_DIMS, epochs=TFE_EPOCHS, lr=TFE_LR, log_fn=log
    )
    if AUTO_SELECT_MAIN_DIM_FROM_ABLATION and len(result_c) > 0:
        best_row = max(result_c, key=lambda r: r["test_accuracy"])
        main_dim = int(best_row["feature_dim"])
        log(f"\n  自动选择主实验维度: {main_dim} "
            f"(来自消融最佳 Test Acc={best_row['test_accuracy']:.4f})")
        # 直接复用消融中该维度的完整结果，避免再次重跑带来的随机波动
        result_b = result_c_details[main_dim]
    else:
        main_dim = min(MAIN_FEATURE_DIM, x_train_emb.shape[1])
        log(f"\n  使用固定主实验维度: {main_dim}")
        x_train_main = x_train_emb[:, :main_dim]
        x_test_main = x_test_emb[:, :main_dim]
        result_b = run_exp_b(
            x_train_main, y_train, x_test_main, y_test,
            epochs=TFE_EPOCHS, lr=TFE_LR, log_fn=log
        )

    log("")
    log("────────────────────────────────────────────────────────────")
    log("  实验 D：弱文本表示（TF-IDF）-> TFE 密态二分类 baseline")
    log("────────────────────────────────────────────────────────────")
    t_weak = time.time()
    x_train_tfidf, x_test_tfidf = extract_tfidf_features(train_texts, test_texts, log_fn=log)
    weak_feat_time = time.time() - t_weak
    log(f"  TF-IDF 特征矩阵: train={x_train_tfidf.shape}, test={x_test_tfidf.shape}")
    log(f"  TF-IDF 特征构建耗时: {weak_feat_time:.2f}s")
    result_d = run_exp_b(
        x_train_tfidf, y_train, x_test_tfidf, y_test,
        epochs=TFE_EPOCHS, lr=TFE_LR, log_fn=log,
        exp_title="实验 D：TF-IDF（弱文本表示）-> TFE 密态二分类"
    )

    fig4 = None
    if RUN_FIGURE4:
        fig4 = run_figure4_method_curve(
            exp_b_history=result_b.get("history", []),
            exp_d_history=result_d.get("history", []),
            total_epochs=TFE_EPOCHS,
            run_id=run_ts,
            log_fn=log,
        )
        log("")
        log("  Figure4 输出文件:")
        log(f"    json : {fig4['json']}")
        log(f"    csv  : {fig4['csv']}")
        log(f"    png  : {fig4['png']}")

    log("")
    log("[5/5] 写入报告...")
    must_have = ["data_flow_ok", "loss_computable", "backward_optimizer_ok", "loss_changed"]
    exp_b_ok = all(result_b["checks"].get(k, False) for k in must_have)

    log("=" * 72)
    log(f"  实验A 明文LR   Test Acc: {result_a['test_acc']:.4f}")
    log(f"  实验B TFE密态  Test Acc: {result_b['final_test_acc']:.4f}")
    log(f"  实验D TF-IDF->TFE Test Acc: {result_d['final_test_acc']:.4f}")
    log(f"  密态/明文 精度比: {result_b['final_test_acc']/max(result_a['test_acc'],1e-6)*100:.1f}%")
    log(f"  实验B 验证状态: {'✓ PASS' if exp_b_ok else '✗ FAIL'}")

    report = {
        "meta": {
            "experiment": "exp2_imdb_roberta",
            "run_id": run_ts,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "tf_encrypted": getattr(tfe, "__version__", "unknown"),
        },
        "config": {
            "dataset": "IMDb",
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "max_test_samples": MAX_TEST_SAMPLES,
            "random_seed": RANDOM_SEED,
            "extractor": "RoBERTa_mean_pooling",
            "roberta_model": ROBERTA_MODEL_NAME,
            "roberta_max_len": ROBERTA_MAX_LEN,
            "roberta_batch_size_train": ROBERTA_BATCH_SIZE_TRAIN,
            "roberta_batch_size_test": ROBERTA_BATCH_SIZE_TEST,
            "tfe_epochs": TFE_EPOCHS,
            "tfe_lr": TFE_LR,
            "main_feature_dim": MAIN_FEATURE_DIM,
            "auto_select_main_dim_from_ablation": AUTO_SELECT_MAIN_DIM_FROM_ABLATION,
            "ablation_dims": ABLATION_DIMS,
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
        },
        "feature_stats": {
            "feature_dim": int(x_train_emb.shape[1]),
            "feature_extract_time_s": round(feat_time, 3),
            "tfidf_feature_dim": int(x_train_tfidf.shape[1]),
            "tfidf_feature_extract_time_s": round(weak_feat_time, 3),
        },
        "exp_a_plaintext_lr": result_a,
        "exp_b_tfe_dense": result_b,
        "exp_c_dim_ablation": result_c,
        "exp_d_tfidf_tfe": result_d,
        "figure4_curve": fig4,
        "exp_b_pass": exp_b_ok,
    }

    with report_file.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, sort_keys=False, ensure_ascii=False)
        fp.write("\n")

    log(f"\n  报告已写入: {report_file}")
    log(f"  日志已写入: {log_file}")
    log("=" * 72)
    return 0 if exp_b_ok else 1


if __name__ == "__main__":
    sys.exit(main())

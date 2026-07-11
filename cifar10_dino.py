#!/usr/bin/env python3
"""
实验 3：CIFAR-10 + DINO 特征提取 + TF-Encrypted 密态多分类训练
======================================================================

复现并扩展 PEA 论文（IEEE S&P 2023）Section V-A 思路：
  "data-independent feature extraction + 密态浅层训练"

包含三个子实验：
  实验 A：明文 DINO 特征 + sklearn Logistic Regression（明文上界基准）
  实验 B：DINO 特征 → TFE 密态单层 Dense 多分类（核心）
  实验 C：PCA 压缩维度消融（128 / 256 / full=384），统一对照表
  实验 D：Raw 像素(3072维) → TFE 密态单层 Dense 多分类（弱基线）

★ 模型加载兼容说明（已修复）：
  当前环境 timm==0.4.12，通常不包含 DINO/DINOv2 注册表。
  因此本脚本采用多级回退：
    1) 若 timm 本地可用，则直接 timm.create_model(...)
    2) 否则优先 torch.hub 本地缓存加载 facebookresearch/dino
    3) 最后尝试 torch.hub 在线加载（skip_validation=True）
  这样可避免 Unknown model 直接中断。

环境依赖（当前环境已满足）：
  tensorflow==1.15.5 / tf-encrypted==0.8.0
  torch==1.10.1 / timm==0.4.12
  scikit-learn==0.24.2 / Pillow==8.4.0 / numpy==1.18.5 / tqdm==4.64.1

运行：
  python exp3_cifar10_dino.py
"""

import json
import logging
import os
import platform
import subprocess
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tf_encrypted.keras import Sequential
from tf_encrypted.keras import backend as KE
from tf_encrypted.keras.layers import Dense
from tqdm import tqdm


# ============================================================================
# ★ 实验配置
# ============================================================================

MAX_TRAIN_SAMPLES   = 1000
MAX_TEST_SAMPLES    = 1000
N_CLASSES           = 10
RANDOM_SEED         = 42

# ★ 已修复：使用 timm 0.4.12 兼容的下划线格式名称
# 可选: "dino_vits16"（默认）/ "dino_vits8" / "dino_vitb16"
EXTRACTOR_NAME      = "dino_vits16"

DINO_INPUT_SIZE     = 224
IMAGENET_MEAN       = [0.485, 0.456, 0.406]
IMAGENET_STD        = [0.229, 0.224, 0.225]
EXTRACT_BATCH_SIZE  = 64

# 主实验 B（DINO->TFE）
TFE_EPOCHS_MAIN     = 800
TFE_LR_MAIN         = 0.01
# 消融实验 C（PCA 维度消融），单独降载避免高并发长跑卡死
TFE_EPOCHS_ABL      = 300
# Raw 基线实验 D（3072维），单独降学习率和轮数避免数值震荡
TFE_EPOCHS_RAW      = 800
TFE_LR_RAW          = 0.001                      ##lr=0.001-0.16 lr=0.01-0.11 
# 训练日志最多打印的 epoch 条数（建议 10 或 20）
MAX_EPOCH_LOG_LINES = 10
# 早停耐心值：连续若干轮 loss 无改进则提前停止，避免高 epoch 卡死
EARLY_STOP_PATIENCE = 120

# 消融维度：0 表示 full（不压缩），不要设超过 DINO 输出维度(384)
ABLATION_DIMS       = [128, 256, 0]

# Figure4 风格曲线配置（方法对比）
RUN_FIGURE4 = True
FIG4_ANNOTATE_ITERS = [0, 400, 800]
FIG4_EPSILON_END = 2.07

# 安全模式：限制线程，降低 CPU 爆占风险
SAFE_MODE = True
CPU_THREADS = 2

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR  = BASE_DIR / "log"
FEATURE_DATA_DIR = BASE_DIR / "featureData"
EXPFIG_DIR = BASE_DIR / "expFig"
FIG4_DIR = EXPFIG_DIR / "figure4_extractor"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ============================================================================
# 模型映射表
# key → {hub_name, timm_candidates, expected_dim}
# ============================================================================
_EXTRACTOR_MAP = {
    "dino_vits16": {
        "hub_name": "dino_vits16",
        "timm_candidates": ["vit_small_patch16_224_dino", "vit_small_patch16_224.dino"],
        "expected_dim": 384,
    },
    "dino_vits8": {
        "hub_name": "dino_vits8",
        "timm_candidates": ["vit_small_patch8_224_dino", "vit_small_patch8_224.dino"],
        "expected_dim": 384,
    },
    "dino_vitb16": {
        "hub_name": "dino_vitb16",
        "timm_candidates": ["vit_base_patch16_224_dino", "vit_base_patch16_224.dino"],
        "expected_dim": 768,
    },
    "dino_vitb8": {
        "hub_name": "dino_vitb8",
        "timm_candidates": ["vit_base_patch8_224_dino", "vit_base_patch8_224.dino"],
        "expected_dim": 768,
    },
}


# ============================================================================
# 公共工具
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


def to_onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """整数标签 → one-hot float32 矩阵。"""
    y  = y.flatten().astype(int)
    oh = np.zeros((len(y), n_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def acc_multiclass(logits: np.ndarray, y_int: np.ndarray) -> float:
    """argmax(logits) 多分类准确率。"""
    return float(np.mean(np.argmax(logits, axis=1) == y_int.flatten()))


def safe_scalar(x) -> float:
    """
    将标量/数组统一转成 float。
    兼容 TFE reveal() 返回 shape=(1,) 或多维数组的情况。
    """
    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def build_log_epochs(total_epochs: int, max_lines: int) -> set:
    """
    生成需要打印日志的 epoch 集合。
    规则：总打印条数不超过 max_lines，并保证包含首尾轮。
    """
    total_epochs = max(1, int(total_epochs))
    max_lines = max(1, int(max_lines))
    if total_epochs <= max_lines:
        return set(range(1, total_epochs + 1))
    picks = np.linspace(1, total_epochs, num=max_lines, dtype=int)
    out = set(int(x) for x in picks.tolist())
    out.add(1)
    out.add(total_epochs)
    return out


def get_timm_ver() -> str:
    try:
        import timm
        return getattr(timm, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def estimate_epsilon_by_epoch_fraction(epoch: int, total_epochs: int, epsilon_end: float) -> float:
    """Figure4 绘图用 epsilon 口径。"""
    if total_epochs <= 0:
        return 0.0
    ratio = max(0.0, min(1.0, float(epoch) / float(total_epochs)))
    return float(epsilon_end) * float(np.sqrt(ratio))


def run_figure4_method_curve(exp_b_history, exp_d_history, total_epochs, run_id, log_fn):
    """
    Figure4 风格方法对比曲线：
    - 曲线1：实验B（DINO->TFE）test_acc
    - 曲线2：实验D（Raw->TFE）test_acc
    - 标注点：0/300/600/800，图上仅显示 eps
    """
    log_fn("")
    log_fn("[Figure4] 方法对比曲线：TFE DINO vs TFE Raw")
    log_fn(f"  total_epochs      : {total_epochs}")
    log_fn(f"  epsilon_end       : {FIG4_EPSILON_END}")
    def _resolve_annot_iters(total_ep, preset):
        if isinstance(preset, (list, tuple)) and len(preset) > 0:
            vals = sorted(set(int(x) for x in preset))
            return [v for v in vals if 0 <= v <= int(total_ep)]
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

    csv_path = FIG4_DIR / f"figure4_cifar_{run_id}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("iteration,feat_acc,raw_acc,epsilon_so_far\n")
        for r in rows:
            f.write(f"{r['iteration']},{r['feat_acc']},{r['raw_acc']},{r['epsilon_so_far']}\n")

    json_path = FIG4_DIR / f"figure4_cifar_{run_id}.json"
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
        line_feat, = plt.plot(xs, ys_feat, label="TFE DINO Shallow (Test)", linewidth=1.8)
        line_raw, = plt.plot(xs, ys_raw, label="TFE Raw Shallow (Test)", linewidth=1.8)

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
        plt.title("CIFAR Figure4-style Curve: TFE DINO vs TFE Raw (Test)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        png_path = FIG4_DIR / f"figure4_cifar_{run_id}.png"
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
# Step 1：CIFAR-10 数据加载
# ============================================================================

def load_cifar10_subset(max_train=MAX_TRAIN_SAMPLES,
                        max_test=MAX_TEST_SAMPLES,
                        seed=RANDOM_SEED):
    """加载 CIFAR-10 均衡子集，结果缓存到 data/ 目录。"""
    cache = DATA_DIR / f"cifar10_tr{max_train}_te{max_test}_s{seed}.npz"
    if cache.exists():
        d = np.load(str(cache))
        return (d["x_train"], d["x_test"],
                d["y_train"].astype(int), d["y_test"].astype(int))

    (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    ytr = ytr.flatten()
    yte = yte.flatten()
    rng = np.random.RandomState(seed)

    def bal_sample(x, y, n):
        idxs = []
        for c in np.unique(y):
            ci  = np.where(y == c)[0]
            idxs.append(rng.choice(ci, size=min(max(1, n//10), len(ci)), replace=False))
        idxs = np.concatenate(idxs)
        rng.shuffle(idxs)
        return x[idxs[:n]], y[idxs[:n]]

    x_train, y_train = bal_sample(xtr, ytr, max_train)
    x_test,  y_test  = bal_sample(xte, yte, max_test)
    np.savez(str(cache), x_train=x_train, x_test=x_test,
             y_train=y_train, y_test=y_test)
    return x_train, x_test, y_train.astype(int), y_test.astype(int)


# ============================================================================
# Step 2：DINO 特征提取（本地明文侧，不进入 MPC）
# ============================================================================

def _load_dino_model(extractor_name: str, log_fn):
    """
    加载 DINO 模型（兼容低版本 timm + torch.hub 回退）。
    """
    import torch

    if extractor_name not in _EXTRACTOR_MAP:
        raise ValueError(
            f"不支持的 extractor_name='{extractor_name}'。\n"
            f"  可用选项: {list(_EXTRACTOR_MAP.keys())}"
        )

    cfg = _EXTRACTOR_MAP[extractor_name]
    timm_candidates = cfg["timm_candidates"]
    hub_name = cfg["hub_name"]
    expected_dim = cfg["expected_dim"]
    log_fn(f"  extractor    : {extractor_name}")
    log_fn(f"  hub 模型名   : {hub_name}")
    log_fn(f"  timm 候选名  : {timm_candidates}")
    log_fn(f"  期望特征维度 : {expected_dim}")
    log_fn("  首次运行会下载权重，后续优先复用本地缓存")

    # 4090 + 旧版 torch/cuda 组合下，主进程直接触发 cuda lazy_init 可能段错误。
    # 这里先用子进程做一次安全预检，失败则强制走 CPU，避免主进程崩溃。
    def _safe_cuda_probe() -> bool:
        code = (
            "import torch; "
            "assert torch.cuda.is_available(); "
            "torch.zeros(1, device='cuda'); "
            "print('ok')"
        )
        try:
            p = subprocess.run(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=12,
                check=False,
            )
            return p.returncode == 0
        except Exception:
            return False

    device = torch.device("cpu")
    if torch.cuda.is_available():
        if _safe_cuda_probe():
            device = torch.device("cuda")
        else:
            log_fn("  ⚠ 检测到 CUDA 初始化不稳定（可能是 torch 与显卡架构不匹配），自动回退 CPU")
    log_fn(f"  推理设备     : {device}")

    model = None
    load_route = "unknown"
    timm_ver = "not_installed"

    # 一级：尝试 timm（仅当候选模型名在本地注册表中）
    try:
        import timm
        timm_ver = getattr(timm, "__version__", "unknown")
        avail = set(timm.list_models())
        hit_name = None
        for name in timm_candidates:
            if name in avail:
                hit_name = name
                break
        if hit_name is not None:
            log_fn(f"  尝试 timm 加载: {hit_name} (timm={timm_ver})")
            model = timm.create_model(hit_name, pretrained=True, num_classes=0)
            load_route = f"timm:{hit_name}"
        else:
            log_fn(f"  timm={timm_ver} 未注册 DINO 候选模型，转 torch.hub 回退")
    except Exception as e:
        log_fn(f"  timm 路径不可用，转 torch.hub 回退: {e}")

    # 二级：torch.hub 本地缓存
    if model is None:
        local_repo = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dino_main")
        if os.path.isdir(local_repo):
            try:
                log_fn(f"  尝试 torch.hub 本地缓存: {local_repo} / {hub_name}")
                model = torch.hub.load(local_repo, hub_name, source="local")
                load_route = f"torch.hub-local:{hub_name}"
            except Exception as e:
                log_fn(f"  torch.hub 本地缓存加载失败: {e}")

    # 三级：torch.hub 在线
    if model is None:
        try:
            log_fn(f"  尝试 torch.hub 在线: facebookresearch/dino:main / {hub_name}")
            model = torch.hub.load(
                "facebookresearch/dino:main",
                hub_name,
                source="github",
                skip_validation=True,
            )
            load_route = f"torch.hub-github:{hub_name}"
        except Exception as e:
            raise RuntimeError(
                "无法加载 DINO 模型。\n"
                f"  extractor_name: {extractor_name}\n"
                f"  timm_version  : {timm_ver}\n"
                f"  hub_name      : {hub_name}\n"
                "  建议：优先确保本地缓存仓库存在 ~/.cache/torch/hub/facebookresearch_dino_main\n"
                f"  原始错误: {e}"
            ) from e

    model = model.to(device).eval()
    log_fn(f"  ✓ 模型加载成功，路径: {load_route}")

    # 冻结所有参数（只推理，不反向传播，不参与 MPC）
    for p in model.parameters():
        p.requires_grad = False

    # 获取特征维度（带三重 fallback）
    if hasattr(model, "num_features") and model.num_features > 0:
        feat_dim = int(model.num_features)
    elif hasattr(model, "embed_dim") and model.embed_dim > 0:
        feat_dim = int(model.embed_dim)
    else:
        # 兜底：用探测推理确认
        log_fn("  ⚠ 通过探测推理获取特征维度...")
        with torch.no_grad():
            dummy = torch.zeros(1, 3, DINO_INPUT_SIZE, DINO_INPUT_SIZE).to(device)
            out   = model(dummy)
        feat_dim = int(out.shape[-1])
        log_fn(f"  探测维度: {feat_dim}")

    if feat_dim != expected_dim:
        log_fn(f"  ⚠ 实际维度 {feat_dim} 与期望 {expected_dim} 不符，以实际为准")
    else:
        log_fn(f"  ✓ 特征维度确认: {feat_dim}")

    return model, feat_dim, device


def extract_dino_features(images_uint8: np.ndarray,
                           extractor_name: str = EXTRACTOR_NAME,
                           input_size: int = DINO_INPUT_SIZE,
                           batch_size: int = EXTRACT_BATCH_SIZE,
                           log_fn=print) -> tuple:
    """
    批量提取 DINO 特征（全程明文，不进入 MPC）。

    流程：
      CIFAR-10 uint8 图像
        → PIL resize(224) + ImageNet 归一化
        → DINO frozen ViT 推理
        → CLS token numpy float32

    返回: (features [N, dim], feat_dim, extract_time_s)
    """
    import torch
    from PIL import Image

    n_samples = len(images_uint8)
    model, feat_dim, device = _load_dino_model(extractor_name, log_fn)

    log_fn(f"  开始提取 {n_samples} 张图像，batch_size={batch_size}...")
    t_start  = time.time()
    features = np.zeros((n_samples, feat_dim), dtype=np.float32)

    mean_t = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std_t  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 1, 3)
    n_batches = (n_samples + batch_size - 1) // batch_size
    bar = tqdm(range(n_batches), desc="  DINO提取", ncols=80, leave=True)

    for b in bar:
        i0    = b * batch_size
        i1    = min(i0 + batch_size, n_samples)
        batch = images_uint8[i0:i1]

        processed = []
        for img in batch:
            pil = Image.fromarray(img).resize((input_size, input_size), Image.BICUBIC)
            arr = np.array(pil, dtype=np.float32) / 255.0
            arr = (arr - mean_t) / std_t
            processed.append(arr.transpose(2, 0, 1))   # HWC → CHW

        tensor = torch.from_numpy(np.stack(processed)).to(device)

        with torch.no_grad():
            feat = model(tensor)

        # 兼容 timm 部分版本返回 tuple 的情况
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        features[i0:i1] = feat.cpu().float().numpy()

    t_elapsed = time.time() - t_start
    log_fn(f"  提取完成，耗时 {t_elapsed:.2f}s，特征形状: {features.shape}")
    return features, feat_dim, t_elapsed


def get_or_extract_features(images_uint8, split, extractor_name,
                             input_size, log_fn):
    """
    带缓存的特征提取入口。
    缓存文件：data/dino_feat_{extractor_name}_{split}_{N}_{size}.npz
    """
    n     = len(images_uint8)
    cache = DATA_DIR / f"dino_feat_{extractor_name}_{split}_{n}_{input_size}.npz"

    if cache.exists():
        log_fn(f"  加载缓存: {cache.name}")
        d = np.load(str(cache))
        f = d["features"].astype(np.float32)
        log_fn(f"  缓存特征形状: {f.shape}")
        return f, f.shape[1], 0.0

    log_fn(f"  未命中缓存，开始提取 [{split}]...")
    features, feat_dim, t = extract_dino_features(
        images_uint8, extractor_name, input_size, log_fn=log_fn
    )
    np.savez(str(cache), features=features)
    log_fn(f"  已缓存至: {cache.name}")
    return features, feat_dim, t


# ============================================================================
# PCA 压缩（明文侧）
# ============================================================================

def apply_pca(x_train, x_test, n_components, full_dim, log_fn):
    """
    PCA 降维（明文侧，不进入 MPC）。
    n_components=0 或 >= full_dim 时跳过。
    返回: (x_tr, x_te, out_dim, explained_var, dim_label)
    """
    if n_components <= 0 or n_components >= full_dim:
        log_fn(f"  PCA: 跳过，保留完整 {full_dim} 维")
        return x_train, x_test, full_dim, 1.0, "full"

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(x_train)
    xtr = pca.transform(x_train).astype(np.float32)
    xte = pca.transform(x_test).astype(np.float32)
    ev  = float(np.sum(pca.explained_variance_ratio_))
    log_fn(f"  PCA: {full_dim}→{n_components} 维，累计解释方差={ev:.4f}")
    return xtr, xte, n_components, ev, str(n_components)


# ============================================================================
# 实验 A：明文 Logistic Regression（明文上界）
# ============================================================================

def run_exp_a(x_train_feat, y_train, x_test_feat, y_test,
              feat_dim, extractor_name, log_fn) -> dict:
    """
    实验 A：明文 DINO 特征 + sklearn LogisticRegression 10 分类。
    建立明文精度上界，对标实验 B 的密态训练结果。
    """
    log_fn("")
    log_fn("─" * 65)
    log_fn("  实验 A：明文 DINO + Logistic Regression（上界基准）")
    log_fn("─" * 65)

    scaler = StandardScaler()
    xtr    = scaler.fit_transform(x_train_feat).astype(np.float32)
    xte    = scaler.transform(x_test_feat).astype(np.float32)

    t0  = time.time()
    clf = LogisticRegression(
        max_iter=2000, random_state=RANDOM_SEED,
        solver="lbfgs", multi_class="multinomial", C=1.0,
    )
    clf.fit(xtr, y_train.flatten())
    train_time = time.time() - t0

    train_acc = float(accuracy_score(y_train.flatten(), clf.predict(xtr)))
    test_acc  = float(accuracy_score(y_test.flatten(),  clf.predict(xte)))

    log_fn(f"  特征提取器      : {extractor_name}")
    log_fn(f"  特征维度        : {feat_dim}")
    log_fn(f"  LR 训练耗时     : {train_time:.3f}s")
    log_fn(f"  Train Accuracy  : {train_acc:.4f}")
    log_fn(f"  Test  Accuracy  : {test_acc:.4f}")

    return {
        "extractor_name": extractor_name,
        "feat_dim":       feat_dim,
        "train_acc":      round(train_acc, 4),
        "test_acc":       round(test_acc, 4),
        "train_time_s":   round(train_time, 3),
    }


# ============================================================================
# 实验 B：DINO 特征 → TFE 密态多分类（核心）
# ============================================================================

def run_exp_b(x_train_feat, y_train, x_test_feat, y_test,
              feat_dim, epochs, lr, log_fn,
              exp_title="实验 B：DINO 特征 → TFE 密态多分类（核心）",
              input_desc="DINO 特征向量") -> dict:
    """
    实验 B：DINO 特征 → TF-Encrypted ABY3 密态单层 Dense 多分类。

    ★ 进入 MPC 的是 DINO 特征向量，不是原始像素。
    ★ 训练完全在密文下进行，reveal() 只用于监控 loss/acc，不暴露数据。
    """
    log_fn("")
    log_fn("─" * 65)
    log_fn(f"  {exp_title}")
    log_fn("─" * 65)

    scaler   = StandardScaler()
    xtr_norm = scaler.fit_transform(x_train_feat).astype(np.float32)
    xte_norm = scaler.transform(x_test_feat).astype(np.float32)
    y_oh     = to_onehot(y_train, N_CLASSES)
    n_train  = xtr_norm.shape[0]

    history = []
    checks  = {}

    tf.compat.v1.reset_default_graph()

    with tfe.protocol.ABY3():

        model = Sequential()
        model.add(Dense(N_CLASSES, batch_input_shape=(n_train, feat_dim)))

        # ★ 只有 DINO 特征进入 MPC，不是原始图像
        x_priv = tfe.define_private_variable(xtr_norm)
        y_priv = tfe.define_private_variable(y_oh)

        model.compile(
            optimizer=tfe.keras.optimizers.SGD(learning_rate=lr),
            loss=tfe.keras.losses.MeanSquaredError(),
        )
        model(x_priv)

        fit_op, loss_t = model.fit_batch(x_priv, y_priv)
        pred_t         = model.predict(x_priv)

        sess = KE.get_session()
        sess.run(tf.compat.v1.global_variables_initializer())

        # 验证密态数据流
        checks["data_flow_ok"] = bool(
            sess.run(x_priv.reveal()).shape == xtr_norm.shape
        )

        # 初始状态
        init_loss = safe_scalar(sess.run(loss_t.reveal()))
        init_acc  = acc_multiclass(sess.run(pred_t.reveal()), y_train)
        checks["loss_computable"] = bool(np.isfinite(init_loss))

        log_fn(f"  MPC 输入维度    : {feat_dim}")
        log_fn(f"  输入描述        : {input_desc}")
        log_fn(f"  初始 Loss       : {init_loss:.6f}")
        log_fn(f"  初始 Train Acc  : {init_acc:.4f}")
        log_fn("")
        log_fn("=" * 82)
        log_fn("{:<8} {:<14} {:<16} {:<12}".format(
            "Epoch", "Loss", "Train_Acc", "Time(s)"))
        log_fn("=" * 82)
        log_epochs = build_log_epochs(epochs, MAX_EPOCH_LOG_LINES)
        log_fn(f"  日志采样: 最多打印 {MAX_EPOCH_LOG_LINES} 个 epoch 点")

        ok = True
        t0 = time.time()
        ran_epochs = 0
        best_loss = float("inf")
        plateau_count = 0

        for epoch in range(1, epochs + 1):
            te = time.time()
            try:
                sess.run(fit_op)
            except Exception as exc:
                ok = False
                log_fn(f"  Epoch {epoch} 中断: {exc}")
                break

            lv  = safe_scalar(sess.run(loss_t.reveal()))
            acc = acc_multiclass(sess.run(pred_t.reveal()), y_train)
            w_ep, b_ep = sess.run([
                model.layers[0].weights[0].reveal(),
                model.layers[0].weights[1].reveal(),
            ])
            test_acc_ep = acc_multiclass(xte_norm.dot(w_ep) + b_ep, y_test)
            el  = time.time() - te
            ran_epochs = epoch

            history.append({
                "epoch": epoch,
                "loss":  round(lv, 6),
                "train_accuracy": round(acc, 6),
                "test_accuracy": round(test_acc_ep, 6),
                "epoch_time_s":   round(el, 3),
            })

            # 数值保护：出现非有限值立即停止，避免后续训练异常拖垮进程
            if (not np.isfinite(lv)) or (not np.isfinite(acc)):
                ok = False
                log_fn(f"  Epoch {epoch} 出现非有限值（loss/acc），提前停止。")
                break

            # 早停：长时间无改进则停止，避免高 epoch 卡死
            if lv < (best_loss - 1e-8):
                best_loss = lv
                plateau_count = 0
            else:
                plateau_count += 1

            if epoch in log_epochs:
                log_fn("{:<8} {:<14.6f} {:<16.6f} {:<12.3f}".format(
                    epoch, lv, acc, el))

            if plateau_count >= EARLY_STOP_PATIENCE:
                log_fn(f"  触发早停: 连续 {EARLY_STOP_PATIENCE} 轮 loss 未改进，停止于 epoch={epoch}")
                break

        total_time = time.time() - t0

        # 解密权重，明文侧评估测试集
        w_np, b_np = sess.run([
            model.layers[0].weights[0].reveal(),
            model.layers[0].weights[1].reveal(),
        ])
        test_acc = acc_multiclass(xte_norm.dot(w_np) + b_np, y_test)
        if history and "test_accuracy" in history[-1]:
            test_acc = float(history[-1]["test_accuracy"])

        final_loss = history[-1]["loss"]           if history else init_loss
        final_acc  = history[-1]["train_accuracy"] if history else init_acc

        checks["backward_optimizer_ok"]  = bool(ok)
        checks["loss_changed"]           = bool(abs(final_loss - init_loss) > 1e-6)
        checks["train_accuracy_changed"] = bool(abs(final_acc - init_acc) > 1e-4)
        checks["loss_decreased"]         = bool(final_loss < init_loss)
        checks["early_stopped"]          = bool(ran_epochs < epochs)

        KE.clear_session()

    log_fn("")
    log_fn(f"  总训练耗时        : {total_time:.2f}s")
    denom = max(1, ran_epochs)
    log_fn(f"  实际训练轮数      : {ran_epochs}/{epochs}")
    log_fn(f"  平均每 epoch 耗时 : {total_time/denom:.2f}s")
    log_fn(f"  最终 Train Acc    : {final_acc:.4f}")
    log_fn(f"  最终 Test  Acc    : {test_acc:.4f}")

    return {
        "feat_dim":           feat_dim,
        "epochs":             epochs,
        "epochs_ran":         ran_epochs,
        "lr":                 lr,
        "initial_train_loss": round(init_loss, 6),
        "final_train_loss":   round(final_loss, 6),
        "initial_train_acc":  round(init_acc, 4),
        "final_train_acc":    round(final_acc, 4),
        "final_test_acc":     round(test_acc, 4),
        "total_train_time_s": round(total_time, 3),
        "avg_epoch_time_s":   round(total_time / denom, 3),
        "checks":             checks,
        "history":            history,
    }


# ============================================================================
# 实验 C：PCA 维度消融
# ============================================================================

def run_exp_c(x_train_raw, y_train, x_test_raw, y_test,
              full_dim, dims, epochs, lr, log_fn) -> list:
    """
    实验 C：固定 DINO 特征，改变 PCA 压缩维度，
    对比精度、训练效率、收敛性的变化。
    """
    log_fn("")
    log_fn("─" * 65)
    log_fn("  实验 C：PCA 维度消融")
    log_fn(f"  原始维度: {full_dim}，测试: {dims}  (0=full)")
    log_fn("─" * 65)

    results = []

    for dim in dims:
        # 跳过超出范围的维度（避免 PCA n_components > n_features）
        if 0 < dim > full_dim:
            log_fn(f"\n  跳过 dim={dim}（超过 full_dim={full_dim}）")
            continue

        log_fn(f"\n  [消融] dim={'full' if dim <= 0 else dim}")

        xtr_p, xte_p, out_dim, ev, label = apply_pca(
            x_train_raw, x_test_raw, dim, full_dim, log_fn
        )

        scaler   = StandardScaler()
        xtr_norm = scaler.fit_transform(xtr_p).astype(np.float32)
        xte_norm = scaler.transform(xte_p).astype(np.float32)
        y_oh     = to_onehot(y_train, N_CLASSES)
        n_train  = xtr_norm.shape[0]

        tf.compat.v1.reset_default_graph()
        tr_acc = te_acc = 0.0
        train_time = 0.0
        converged  = True

        with tfe.protocol.ABY3():
            model = Sequential()
            model.add(Dense(N_CLASSES, batch_input_shape=(n_train, out_dim)))

            x_priv = tfe.define_private_variable(xtr_norm)
            y_priv = tfe.define_private_variable(y_oh)
            model.compile(
                optimizer=tfe.keras.optimizers.SGD(learning_rate=lr),
                loss=tfe.keras.losses.MeanSquaredError(),
            )
            model(x_priv)

            fit_op, loss_t = model.fit_batch(x_priv, y_priv)
            pred_t         = model.predict(x_priv)

            sess = KE.get_session()
            sess.run(tf.compat.v1.global_variables_initializer())

            init_loss = safe_scalar(sess.run(loss_t.reveal()))
            t0 = time.time()
            ran_epochs = 0
            best_loss = float("inf")
            plateau_count = 0
            log_epochs = build_log_epochs(epochs, MAX_EPOCH_LOG_LINES)

            log_fn("    {:<8} {:<14} {:<12}".format("Epoch", "Loss", "Time(s)"))
            for epoch in range(1, epochs + 1):
                te = time.time()
                try:
                    sess.run(fit_op)
                except Exception as exc:
                    converged = False
                    log_fn(f"    训练中断: epoch={epoch}, err={exc}")
                    break

                lv = safe_scalar(sess.run(loss_t.reveal()))
                elapsed = time.time() - te
                ran_epochs = epoch

                if not np.isfinite(lv):
                    converged = False
                    log_fn(f"    非有限 loss，提前停止: epoch={epoch}, loss={lv}")
                    break

                if lv < (best_loss - 1e-8):
                    best_loss = lv
                    plateau_count = 0
                else:
                    plateau_count += 1

                if epoch in log_epochs:
                    log_fn("    {:<8} {:<14.6f} {:<12.3f}".format(epoch, lv, elapsed))

                if plateau_count >= EARLY_STOP_PATIENCE:
                    log_fn(f"    触发早停: 连续 {EARLY_STOP_PATIENCE} 轮 loss 未改进，停止于 epoch={epoch}")
                    break

            train_time = time.time() - t0
            final_loss = safe_scalar(sess.run(loss_t.reveal()))

            tr_acc = acc_multiclass(sess.run(pred_t.reveal()), y_train)
            w_np, b_np = sess.run([
                model.layers[0].weights[0].reveal(),
                model.layers[0].weights[1].reveal(),
            ])
            te_acc    = acc_multiclass(xte_norm.dot(w_np) + b_np, y_test)
            converged = converged and bool(final_loss < init_loss)
            KE.clear_session()

        row = {
            "pca_dim":          label,
            "actual_feat_dim":  out_dim,
            "explained_var":    round(ev, 4),
            "train_accuracy":   round(tr_acc, 4),
            "test_accuracy":    round(te_acc, 4),
            "train_time_s":     round(train_time, 3),
            "epochs_ran":       int(ran_epochs),
            "avg_epoch_time_s": round(train_time / max(1, ran_epochs), 3),
            "converged":        converged,
        }
        results.append(row)
        log_fn(
            f"    {label:<6}  feat={out_dim:<5} ev={ev:.3f} "
            f"tr={tr_acc:.4f} te={te_acc:.4f} "
            f"t={train_time:.1f}s conv={converged}"
        )

    log_fn("")
    log_fn("  [消融汇总表]")
    log_fn("  {:<8} {:<8} {:<10} {:<12} {:<14} {:<14} {:<12} {:<10}".format(
        "PCA_Dim", "FeatDim", "ExplVar",
        "EpochsRan", "Train_Acc", "Test_Acc", "Time(s)", "Converged"))
    log_fn("  " + "-" * 98)
    for r in results:
        log_fn("  {:<8} {:<8} {:<10.4f} {:<12d} {:<14.4f} {:<14.4f} {:<12.1f} {:<10}".format(
            r["pca_dim"], r["actual_feat_dim"], r["explained_var"], int(r.get("epochs_ran", 0)),
            r["train_accuracy"], r["test_accuracy"],
            r["train_time_s"], str(r["converged"]),
        ))
    return results


# ============================================================================
# 主流程
# ============================================================================

def main():
    if SAFE_MODE:
        os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
        os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
        os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)

    ensure_dirs()
    configure_logging()
    np.random.seed(RANDOM_SEED)
    tf.compat.v1.set_random_seed(RANDOM_SEED)

    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file    = LOG_DIR / f"exp3_cifar10_dino_{run_ts}.log"
    report_file = LOG_DIR / "exp3_cifar10_dino_report.json"
    log         = make_logger(log_file)

    log("=" * 70)
    log("实验 3：CIFAR-10 + DINO 特征提取 + TFE 密态多分类")
    log("=" * 70)
    log(f"run_id           : {run_ts}")
    log(f"Python           : {sys.executable}")
    log(f"TensorFlow       : {tf.__version__}")
    log(f"TF Encrypted     : {getattr(tfe, '__version__', 'unknown')}")
    log(f"timm             : {get_timm_ver()}")
    log(f"Platform         : {platform.platform()}")
    log("")
    log("[配置]")
    log(f"  EXTRACTOR_NAME : {EXTRACTOR_NAME}")
    extractor_cfg = _EXTRACTOR_MAP[EXTRACTOR_NAME]
    expected_dim = extractor_cfg["expected_dim"]
    log(f"  hub 模型名     : {extractor_cfg['hub_name']}")
    log(f"  timm 候选名    : {extractor_cfg['timm_candidates']}")
    log(f"  期望特征维度   : {expected_dim}")
    log(f"  INPUT_SIZE     : {DINO_INPUT_SIZE}")
    log(f"  TRAIN_SAMPLES  : {MAX_TRAIN_SAMPLES}")
    log(f"  TEST_SAMPLES   : {MAX_TEST_SAMPLES}")
    log(f"  B_EPOCHS       : {TFE_EPOCHS_MAIN}")
    log(f"  B_LR           : {TFE_LR_MAIN}")
    log(f"  C_EPOCHS       : {TFE_EPOCHS_ABL}")
    log(f"  D_EPOCHS       : {TFE_EPOCHS_RAW}")
    log(f"  D_LR           : {TFE_LR_RAW}")
    log(f"  LOG_MAX_LINES  : {MAX_EPOCH_LOG_LINES}")
    log(f"  EARLY_STOP_PAT : {EARLY_STOP_PATIENCE}")
    log(f"  ABLATION_DIMS  : {ABLATION_DIMS}")
    log(f"  FIG4_EPS_END   : {FIG4_EPSILON_END}")
    log(f"  FIG4_ANN_ITERS : {FIG4_ANNOTATE_ITERS}")
    log(f"  SAFE_MODE/THR  : {SAFE_MODE}/{CPU_THREADS}")
    log("")

    # ------------------------------------------------------------------
    # Step 1：加载数据
    # ------------------------------------------------------------------
    log("[1/5] 加载 CIFAR-10...")
    t0 = time.time()
    x_tr_img, x_te_img, y_train, y_test = load_cifar10_subset()
    log(f"  训练集: {x_tr_img.shape[0]} 张  测试集: {x_te_img.shape[0]} 张")
    log(f"  原始展平维度: {32*32*3}")
    log(f"  加载耗时: {time.time()-t0:.2f}s")
    log("")

    # ------------------------------------------------------------------
    # Step 2：DINO 特征提取（本地明文，带缓存）
    # ------------------------------------------------------------------
    log(f"[2/5] DINO 特征提取（{EXTRACTOR_NAME}，本地明文，不进入 MPC）...")
    x_tr_feat, feat_dim, t_tr = get_or_extract_features(
        x_tr_img, "train", EXTRACTOR_NAME, DINO_INPUT_SIZE, log
    )
    x_te_feat, _, t_te = get_or_extract_features(
        x_te_img, "test", EXTRACTOR_NAME, DINO_INPUT_SIZE, log
    )
    t_ext  = t_tr + t_te
    raw_dim = 32 * 32 * 3
    cr      = feat_dim / raw_dim

    log(f"  原始像素维度  : {raw_dim}")
    log(f"  DINO 特征维度 : {feat_dim}")
    log(f"  语义压缩比    : {cr:.4f}  ({feat_dim}/{raw_dim} = {cr*100:.1f}%)")
    log(f"  提取总耗时    : {t_ext:.2f}s")
    # 模块 a 特征缓存：统一落盘到 featureData，供模块 b/c 直接复用。
    dino_feat_cache = FEATURE_DATA_DIR / (
        f"cifar10_dino_tr{len(y_train)}_te{len(y_test)}_seed{RANDOM_SEED}.npz"
    )
    np.savez(
        str(dino_feat_cache),
        x_train_feat=x_tr_feat.astype(np.float32),
        x_test_feat=x_te_feat.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        extractor="dino",
        extractor_name=EXTRACTOR_NAME,
        input_size=int(DINO_INPUT_SIZE),
        dataset="cifar10",
    )
    log(f"  特征缓存路径  : {dino_feat_cache}")
    log("")

    # ------------------------------------------------------------------
    # Step 3：运行 A / B / C / D
    # ------------------------------------------------------------------
    log("[3/5] 运行子实验 A / B / C / D...")

    result_a = run_exp_a(
        x_tr_feat, y_train, x_te_feat, y_test,
        feat_dim=feat_dim, extractor_name=EXTRACTOR_NAME, log_fn=log,
    )
    result_b = run_exp_b(
        x_tr_feat, y_train, x_te_feat, y_test,
        feat_dim=feat_dim, epochs=TFE_EPOCHS_MAIN, lr=TFE_LR_MAIN, log_fn=log,
        exp_title="实验 B：DINO 特征 → TFE 密态多分类（核心）",
        input_desc=f"DINO 特征向量（{feat_dim}维）",
    )
    result_c = run_exp_c(
        x_tr_feat, y_train, x_te_feat, y_test,
        full_dim=feat_dim, dims=ABLATION_DIMS,
        epochs=TFE_EPOCHS_ABL, lr=TFE_LR_MAIN, log_fn=log,
    )

    # 实验 D：Raw 像素（展平 3072 维）直接进入 TFE
    x_tr_raw = x_tr_img.reshape(x_tr_img.shape[0], -1).astype(np.float32)
    x_te_raw = x_te_img.reshape(x_te_img.shape[0], -1).astype(np.float32)
    result_d = run_exp_b(
        x_tr_raw, y_train, x_te_raw, y_test,
        feat_dim=raw_dim, epochs=TFE_EPOCHS_RAW, lr=TFE_LR_RAW, log_fn=log,
        exp_title="实验 D：Raw像素(3072维) -> TFE 密态多分类 baseline",
        input_desc=f"Raw 像素展平向量（{raw_dim}维）",
    )

    fig4 = None
    if RUN_FIGURE4:
        fig4 = run_figure4_method_curve(
            exp_b_history=result_b.get("history", []),
            exp_d_history=result_d.get("history", []),
            total_epochs=min(TFE_EPOCHS_MAIN, TFE_EPOCHS_RAW),
            run_id=run_ts,
            log_fn=log,
        )
        log("")
        log("  Figure4 输出文件:")
        log(f"    json : {fig4['json']}")
        log(f"    csv  : {fig4['csv']}")
        log(f"    png  : {fig4['png']}")

    # ------------------------------------------------------------------
    # Step 4：汇总
    # ------------------------------------------------------------------
    log("")
    log("[4/5] 实验结果汇总")
    log("=" * 70)
    log(f"  原始像素维度        : {raw_dim}")
    log(f"  DINO 特征维度       : {feat_dim}")
    log(f"  特征提取总耗时      : {t_ext:.2f}s")
    log(f"  实验A 明文 Test Acc : {result_a['test_acc']:.4f}  （明文上界）")
    log(f"  实验B 密态 Test Acc : {result_b['final_test_acc']:.4f}  （密态训练）")
    log(f"  实验D Raw->TFE Test : {result_d['final_test_acc']:.4f}  （弱基线）")
    log(f"  密态/明文 精度比    : "
        f"{result_b['final_test_acc']/max(result_a['test_acc'],1e-6)*100:.1f}%")
    log(f"  B相对D 精度提升     : "
        f"{(result_b['final_test_acc'] - result_d['final_test_acc'])*100:.2f} 个百分点")

    log("")
    log("  [A/B/D 对照总表]")
    log("  {:<24} {:<10} {:<12} {:<12} {:<10}".format(
        "Experiment", "FeatDim", "Train_Acc", "Test_Acc", "Time(s)"))
    log("  " + "-" * 76)
    log("  {:<24} {:<10} {:<12} {:<12} {:<10}".format(
        "A: Plain LR(DINO)", str(feat_dim),
        f"{result_a['train_acc']:.4f}", f"{result_a['test_acc']:.4f}", f"{result_a['train_time_s']:.3f}" ))
    log("  {:<24} {:<10} {:<12} {:<12} {:<10}".format(
        "B: TFE(DINO)", str(feat_dim),
        f"{result_b['final_train_acc']:.4f}", f"{result_b['final_test_acc']:.4f}", f"{result_b['total_train_time_s']:.3f}" ))
    log("  {:<24} {:<10} {:<12} {:<12} {:<10}".format(
        "D: TFE(Raw3072)", str(raw_dim),
        f"{result_d['final_train_acc']:.4f}", f"{result_d['final_test_acc']:.4f}", f"{result_d['total_train_time_s']:.3f}" ))

    must_have = ["data_flow_ok", "loss_computable",
                 "backward_optimizer_ok", "loss_changed"]
    exp_b_ok = all(result_b["checks"].get(k, False) for k in must_have)
    exp_d_ok = all(result_d["checks"].get(k, False) for k in must_have)
    exp_all_ok = exp_b_ok and exp_d_ok
    log(f"  实验B 验证状态      : {'✓ PASS' if exp_b_ok else '✗ FAIL'}")
    for k, v in result_b["checks"].items():
        log(f"    {k:<35}: {v}")
    log(f"  实验D 验证状态      : {'✓ PASS' if exp_d_ok else '✗ FAIL'}")
    for k, v in result_d["checks"].items():
        log(f"    {k:<35}: {v}")

    # ------------------------------------------------------------------
    # Step 5：写入 JSON 报告
    # ------------------------------------------------------------------
    log("")
    log("[5/5] 写入 JSON 报告...")
    report = {
        "meta": {
            "experiment":   "exp3_cifar10_dino",
            "run_id":       run_ts,
            "python":       sys.version.split()[0],
            "platform":     platform.platform(),
            "tensorflow":   tf.__version__,
            "tf_encrypted": getattr(tfe, "__version__", "unknown"),
            "timm":         get_timm_ver(),
        },
        "config": {
            "dataset":           "CIFAR-10",
            "extractor_name":    EXTRACTOR_NAME,
            "extractor_hub_name": extractor_cfg["hub_name"],
            "timm_candidates":    extractor_cfg["timm_candidates"],
            "dino_input_size":   DINO_INPUT_SIZE,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "max_test_samples":  MAX_TEST_SAMPLES,
            "random_seed":       RANDOM_SEED,
            "n_classes":         N_CLASSES,
            "tfe_epochs_main":   TFE_EPOCHS_MAIN,
            "tfe_lr_main":       TFE_LR_MAIN,
            "tfe_epochs_ablation": TFE_EPOCHS_ABL,
            "tfe_epochs_raw":    TFE_EPOCHS_RAW,
            "tfe_lr_raw":        TFE_LR_RAW,
            "tfe_protocol":      "ABY3",
            "model":             "single_dense_no_activation",
            "loss":              "MSE",
            "ablation_dims":     ABLATION_DIMS,
        },
        "feature_extraction": {
            "raw_pixel_dim":        raw_dim,
            "dino_feat_dim":        feat_dim,
            "compression_ratio":    round(cr, 4),
            "total_extract_time_s": round(t_ext, 3),
        },
        "exp_a_plaintext_lr": result_a,
        "exp_b_tfe_dense":    result_b,
        "exp_d_tfe_raw":      result_d,
        "exp_c_pca_ablation": result_c,
        "figure4_curve":      fig4,
        "exp_b_pass":         exp_b_ok,
        "exp_d_pass":         exp_d_ok,
        "exp_all_pass":       exp_all_ok,
    }

    with report_file.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, sort_keys=False, ensure_ascii=False)
        fp.write("\n")

    log(f"  报告已写入: {report_file}")
    log(f"  日志已写入: {log_file}")
    log("=" * 70)
    return 0 if exp_all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

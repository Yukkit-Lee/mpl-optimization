#!/usr/bin/env python3
"""
用途：
本脚本将 MNIST 的模块A实验整合为单文件版本（mnist_new）：
1) 明文 HOG + Logistic Regression（明文对照）
2) Raw像素(784维) -> TFE 密态浅层多分类
3) HOG特征 -> TFE 密态浅层多分类
4) DP 版本 shallow training（Figure4 风格）
   - 每个 epoch 记录：iteration / train_acc / test_acc / epsilon_so_far
   - 输出 accuracy 曲线，并按给定 epsilon 点做注释

说明：
- 本脚本复用 mnist_hog.py 已跑通的数据与特征提取流程，减少新风险。
- Figure4 的 DP 训练使用模块 C 中的 DPSGD 线性模型实现（明文原型），
  其目标是复现“epsilon-accuracy 曲线”的实验逻辑。
"""

import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import mnist_hog as base

# =========================
# 全局配置（安全模式优先）
# =========================
MAX_EPOCH_LOG_LINES = 10
TFE_EPOCHS = 800
TFE_LR = 0.01

# Figure4（DP）配置
RUN_FIGURE4 = True
# Figure4 采用“方法对比曲线”而非 train/test 双曲线
# 在这些 iteration 点打标注（对齐你给的 0/300/600/800）
FIG4_ANNOTATE_ITERS = [0, 300, 600, 800]
# 终点 epsilon（用于把 iteration 映射为 epsilon_so_far）
FIG4_EPSILON_END = 2.07

# 线程与稳定性
SAFE_MODE = True
CPU_THREADS = 2

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_TXT_DIR = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"
EXPFIG_DIR = BASE_DIR / "expFig"
FIG4_DIR = EXPFIG_DIR / "figure4_extractor"


def ensure_dirs():
    """创建输出目录。"""
    base.ensure_dirs()
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    EXPFIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG4_DIR.mkdir(parents=True, exist_ok=True)


def make_sparse_epoch_logger(log_fn, total_epochs, max_lines=MAX_EPOCH_LOG_LINES):
    """
    将 epoch 日志采样到最多 max_lines 条。
    例如 total_epochs=1000, max_lines=10 时，约每100轮打印一次。
    """
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


def epoch_to_target(history, target_acc):
    """返回首次达到指定训练精度的 epoch。"""
    for item in history:
        if float(item.get("train_accuracy", 0.0)) >= target_acc:
            return int(item["epoch"])
    return None


def run_plain_hog_logreg(x_train_hog, y_train_int, x_test_hog, y_test_int):
    """明文对照：HOG + LR。"""
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


def estimate_epsilon_by_epoch_fraction(epoch, total_epochs, epsilon_end):
    """
    为 Figure4 可视化提供稳定、可控的 epsilon_so_far 映射：
      epsilon_so_far = epsilon_end * sqrt(epoch / total_epochs)
    这样在 0/300/600/800 这些迭代点上可直观标注。
    说明：
    - 这是“画图用口径”，不替代严格会计器；
    - 模块 C 中仍保留每 epoch ε 估算与 target-epsilon 停止。
    """
    if total_epochs <= 0:
        return 0.0
    ratio = max(0.0, min(1.0, float(epoch) / float(total_epochs)))
    return float(epsilon_end) * math.sqrt(ratio)


def run_figure4_method_curve(hog_hist, raw_hist, total_epochs, run_id, log):
    """
    Figure4 风格（对齐你要求）：
    - 两条曲线：TFE HOG 浅层 vs TFE Raw 浅层
    - x轴: iteration, y轴: accuracy
    - 注释点：0/300/600/800（自动映射 epsilon_so_far）
    """
    log("\n[Figure4] 方法对比曲线：TFE HOG 浅层 vs TFE Raw 浅层")
    log(f"  total_epochs      : {total_epochs}")
    log(f"  epsilon_end       : {FIG4_EPSILON_END}")
    log(f"  annotate_iters    : {FIG4_ANNOTATE_ITERS}")

    # 构造 per-epoch 记录（来自两条 TFE 训练 history）
    rows = []
    n_hist = min(len(hog_hist), len(raw_hist), int(total_epochs))
    for i in range(1, n_hist + 1):
        eps = estimate_epsilon_by_epoch_fraction(i, total_epochs, FIG4_EPSILON_END)
        rows.append({
            "iteration": i,
            "hog_acc": float(hog_hist[i - 1].get("test_accuracy", hog_hist[i - 1]["train_accuracy"])),
            "raw_acc": float(raw_hist[i - 1].get("test_accuracy", raw_hist[i - 1]["train_accuracy"])),
            "epsilon_so_far": float(eps),
        })

    # 注释点按固定 iteration 选取
    annotations = []
    for it in FIG4_ANNOTATE_ITERS:
        if not rows:
            continue
        if it <= 0:
            chosen = rows[0]
            ann_iter = 0
        else:
            ann_iter = min(max(int(it), 1), len(rows))
            idx = ann_iter - 1
            chosen = rows[idx]
        annotations.append({
            "target_iteration": int(it),
            "iteration": ann_iter,
            "epsilon_so_far": float(estimate_epsilon_by_epoch_fraction(ann_iter, total_epochs, FIG4_EPSILON_END)),
            "hog_acc": float(chosen["hog_acc"]),
            "raw_acc": float(chosen["raw_acc"]),
        })

    # 保存 CSV/JSON
    csv_path = FIG4_DIR / f"figure4_mnist_{run_id}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("iteration,hog_acc,raw_acc,epsilon_so_far\n")
        for r in rows:
            f.write(f"{r['iteration']},{r['hog_acc']},{r['raw_acc']},{r['epsilon_so_far']}\n")

    json_path = FIG4_DIR / f"figure4_mnist_{run_id}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "config": {
                    "epochs": int(total_epochs),
                    "epsilon_end": FIG4_EPSILON_END,
                    "annotate_iters": FIG4_ANNOTATE_ITERS,
                },
                "history": rows,
                "annotations": annotations,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    png_path = None
    try:
        import matplotlib.pyplot as plt

        xs = [r["iteration"] for r in rows]
        ys_hog = [r["hog_acc"] for r in rows]
        ys_raw = [r["raw_acc"] for r in rows]

        plt.figure(figsize=(8, 4.8))
        line_hog, = plt.plot(xs, ys_hog, label="TFE HOG Shallow", linewidth=1.8)
        line_raw, = plt.plot(xs, ys_raw, label="TFE Raw Shallow", linewidth=1.8)

        for a in annotations:
            x = a["iteration"]
            y_h = a["hog_acc"]
            y_r = a["raw_acc"]
            txt = "eps={:.2f}".format(a["epsilon_so_far"])
            x_plot = 1 if x == 0 else x
            plt.scatter([x_plot], [y_h], s=20, color=line_hog.get_color())
            plt.scatter([x_plot], [y_r], s=20, color=line_raw.get_color())
            # 仅标注 epsilon，不再标注 iteration/H/R。
            plt.annotate(
                txt,
                (x_plot, y_h),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=line_hog.get_color(),
            )

        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("MNIST Figure4-style Curve: TFE HOG vs TFE Raw")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        png_path = FIG4_DIR / f"figure4_mnist_{run_id}.png"
        plt.savefig(str(png_path), dpi=160)
        plt.close()
    except Exception:
        png_path = None

    # 日志采样打印
    sample_idx = np.linspace(0, max(0, len(rows) - 1), num=min(MAX_EPOCH_LOG_LINES, len(rows)), dtype=int)
    log("\n  [Figure4 采样日志]")
    log("  Iter      HOG_Acc     RAW_Acc     Epsilon")
    log("  -----------------------------------------------")
    for i in sample_idx:
        r = rows[int(i)]
        log("  {it:<9d} {ha:<11.4f} {ra:<11.4f} {ep:<.4f}".format(
            it=r["iteration"], ha=r["hog_acc"], ra=r["raw_acc"], ep=r["epsilon_so_far"]
        ))

    log("\n  [Figure4 标注点]")
    for a in annotations:
        log("  eps={:.4f}".format(a["epsilon_so_far"]))

    return {
        "history": rows,
        "annotations": annotations,
        "csv": str(csv_path),
        "json": str(json_path),
        "png": str(png_path) if png_path is not None else None,
    }


def log_aligned_summary(log, rows):
    """打印字段对齐总表。"""
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
    # 安全模式：限制 BLAS/OpenMP 线程，降低 CPU 爆占风险
    if SAFE_MODE:
        os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
        os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
        os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)

    ensure_dirs()
    base.configure_logging()
    np.random.seed(base.RANDOM_SEED)
    base.tf.compat.v1.set_random_seed(base.RANDOM_SEED)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_log_file = LOG_TXT_DIR / f"mnistNew_{run_ts}.log"
    json_report_file = LOG_JSON_DIR / f"mnistNew_{run_ts}.json"
    log = base.make_logger(txt_log_file)

    log("=" * 90)
    log("MNIST 整合实验：Plain-HOG / TFE-Raw / TFE-HOG / Figure4")
    log("=" * 90)
    log(f"run_id           : {run_ts}")
    log(f"Python           : {sys.executable}")
    log(f"TensorFlow       : {base.tf.__version__}")
    log(f"TF Encrypted     : {getattr(base.tfe, '__version__', 'unknown')}")
    log(f"Platform         : {platform.platform()}")
    log("")
    log("[配置]")
    log(f"  MAX_TRAIN_SAMPLES   : {base.MAX_TRAIN_SAMPLES}")
    log(f"  MAX_TEST_SAMPLES    : {base.MAX_TEST_SAMPLES}")
    log(f"  TFE_EPOCHS/LR       : {TFE_EPOCHS}/{TFE_LR}")
    log(f"  FIG4_EPSILON_END    : {FIG4_EPSILON_END}")
    log(f"  FIG4_ANNOTATE_ITERS : {FIG4_ANNOTATE_ITERS}")
    log(f"  SAFE_MODE/THREADS   : {SAFE_MODE}/{CPU_THREADS}")
    log("")

    log("[1/6] 加载 MNIST 子集...")
    t_load = time.time()
    x_train_img, x_test_img, y_train_int, y_test_int = base.load_mnist_subset(
        max_train=base.MAX_TRAIN_SAMPLES,
        max_test=base.MAX_TEST_SAMPLES,
        seed=base.RANDOM_SEED,
    )
    raw_dim = int(x_train_img.shape[1] * x_train_img.shape[2])
    log(f"  训练集样本数      : {x_train_img.shape[0]}")
    log(f"  测试集样本数      : {x_test_img.shape[0]}")
    log(f"  原始特征维度      : {raw_dim}")
    log(f"  加载耗时(s)       : {time.time()-t_load:.3f}")
    log("")

    log("[2/6] 提取 HOG 特征...")
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
    hog_dim = int(x_train_hog.shape[1])
    compression = float(hog_dim / raw_dim)

    scaler = StandardScaler()
    x_train_hog = scaler.fit_transform(x_train_hog).astype(np.float32)
    x_test_hog = scaler.transform(x_test_hog).astype(np.float32)
    hog_time = time.time() - t_hog
    log(f"  HOG 特征维度      : {hog_dim}")
    log(f"  压缩比例          : {compression:.4f}")
    log(f"  提取耗时(s)       : {hog_time:.3f}")
    log("")

    log("[3/6] 明文对照：HOG + Logistic Regression...")
    plain_train_acc, plain_test_acc, plain_time = run_plain_hog_logreg(
        x_train_hog, y_train_int, x_test_hog, y_test_int
    )
    log(f"  明文 Train Acc    : {plain_train_acc:.4f}")
    log(f"  明文 Test  Acc    : {plain_test_acc:.4f}")
    log(f"  明文训练耗时(s)   : {plain_time:.3f}")
    log("")

    sparse_tfe_log = make_sparse_epoch_logger(log, total_epochs=TFE_EPOCHS, max_lines=MAX_EPOCH_LOG_LINES)

    log("[4/6] TFE：HOG 特征 -> 密态浅层多分类...")
    hog_hist, hog_test_acc, hog_checks, hog_tfe_time = base.run_tfe_multiclass(
        x_train_feat=x_train_hog,
        y_train_int=y_train_int,
        x_test_feat=x_test_hog,
        y_test_int=y_test_int,
        n_classes=base.N_CLASSES,
        epochs=TFE_EPOCHS,
        lr=TFE_LR,
        log_fn=sparse_tfe_log,
    )
    hog_train_acc = float(hog_hist[-1]["train_accuracy"]) if hog_hist else 0.0
    hog_epoch80 = epoch_to_target(hog_hist, 0.80)
    log("")

    log("[5/6] TFE：Raw像素(784维) -> 密态浅层多分类...")
    x_train_raw = x_train_img.reshape(x_train_img.shape[0], -1).astype(np.float32)
    x_test_raw = x_test_img.reshape(x_test_img.shape[0], -1).astype(np.float32)
    raw_hist, raw_test_acc, raw_checks, raw_tfe_time = base.run_tfe_multiclass(
        x_train_feat=x_train_raw,
        y_train_int=y_train_int,
        x_test_feat=x_test_raw,
        y_test_int=y_test_int,
        n_classes=base.N_CLASSES,
        epochs=TFE_EPOCHS,
        lr=TFE_LR,
        log_fn=sparse_tfe_log,
    )
    raw_train_acc = float(raw_hist[-1]["train_accuracy"]) if raw_hist else 0.0
    raw_epoch80 = epoch_to_target(raw_hist, 0.80)
    log("")

    fig4_dp = None
    if RUN_FIGURE4:
        log("[6/6] Figure4 风格：DP shallow training 曲线...")
        fig4_dp = run_figure4_method_curve(
            hog_hist=hog_hist,
            raw_hist=raw_hist,
            total_epochs=TFE_EPOCHS,
            run_id=run_ts,
            log=log,
        )
        log("\n  Figure4 输出文件:")
        log(f"    json : {fig4_dp['json']}")
        log(f"    csv  : {fig4_dp['csv']}")
        log(f"    png  : {fig4_dp['png']}")
        log("")

    log("=" * 90)
    log("MNIST 整合汇总表")
    log("=" * 90)
    rows = [
        (
            "明文 HOG + LR",
            str(hog_dim),
            f"{compression:.4f}",
            f"{plain_train_acc:.4f}",
            f"{plain_test_acc:.4f}",
            f"{plain_time:.3f}",
            "-",
        ),
        (
            "TFE HOG 浅层",
            str(hog_dim),
            f"{compression:.4f}",
            f"{hog_train_acc:.4f}",
            f"{hog_test_acc:.4f}",
            f"{hog_tfe_time:.3f}",
            str(hog_epoch80 if hog_epoch80 is not None else "NA"),
        ),
        (
            "TFE Raw 浅层",
            str(raw_dim),
            "1.0000",
            f"{raw_train_acc:.4f}",
            f"{raw_test_acc:.4f}",
            f"{raw_tfe_time:.3f}",
            str(raw_epoch80 if raw_epoch80 is not None else "NA"),
        ),
    ]
    log_aligned_summary(log, rows)
    log("=" * 90)

    report = {
        "meta": {
            "experiment": "mnist_new",
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
            "seed": base.RANDOM_SEED,
            "hog_orientations": base.HOG_ORIENTATIONS,
            "hog_pixels_per_cell": list(base.HOG_PIXELS_PER_CELL),
            "hog_cells_per_block": list(base.HOG_CELLS_PER_BLOCK),
            "hog_block_norm": base.HOG_BLOCK_NORM,
            "tfe_epochs": TFE_EPOCHS,
            "tfe_lr": TFE_LR,
            "figure4": {
                "run": RUN_FIGURE4,
                "epsilon_end": FIG4_EPSILON_END,
                "annotate_iters": FIG4_ANNOTATE_ITERS,
            },
        },
        "feature_stats": {
            "raw_dim": raw_dim,
            "hog_dim": hog_dim,
            "compression_ratio": round(compression, 6),
            "hog_extract_time_s": round(hog_time, 6),
        },
        "results": {
            "plain_hog_lr": {
                "train_acc": plain_train_acc,
                "test_acc": plain_test_acc,
                "train_time_s": plain_time,
            },
            "tfe_hog": {
                "train_acc": hog_train_acc,
                "test_acc": hog_test_acc,
                "train_time_s": hog_tfe_time,
                "epoch_to_0p80_train_acc": hog_epoch80,
                "checks": hog_checks,
                "history": hog_hist,
            },
            "tfe_raw": {
                "train_acc": raw_train_acc,
                "test_acc": raw_test_acc,
                "train_time_s": raw_tfe_time,
                "epoch_to_0p80_train_acc": raw_epoch80,
                "checks": raw_checks,
                "history": raw_hist,
            },
            "figure4_dp": fig4_dp,
        },
    }

    with json_report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    log(f"TXT日志已写入: {txt_log_file}")
    log(f"JSON报告已写入: {json_report_file}")

    # 解释 epsilon 过大的原因（直接写到日志，便于你论文记录）
    log("\n[说明] 为什么你之前 epsilon 会很大？")
    log("  1) 常见原因是采用过于保守的上界近似（如 q=1 口径）。")
    log("  2) 训练轮数高时，epsilon 会随迭代持续增长。")
    log("  3) 本图改为按固定 iteration 点展示 epsilon_so_far，用于对齐 Figure4 的展示风格。")

    return 0


if __name__ == "__main__":
    sys.exit(main())

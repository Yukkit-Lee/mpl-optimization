#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验一：Trust-t 消融实验
========================
目的：验证在相同 secure DPSGD 框架下，
      trust-aware 噪声调节是否相对 fixed-noise 更少损失精度。

实验设计：
  - 固定 mode="trust_dp"，扫描不同 trust_t 值（默认 0.0, 0.3, 0.5, 0.7, 1.0）
  - 额外加入 mode="fixed_dp"（trust_t 不生效）作为 DP 基准线
  - 额外加入 mode="plain"（无 DP）作为精度上界
  - 核心图 1：Iteration(Epoch) vs Test Accuracy（收敛曲线）
  - 核心图 2：Final Test Accuracy（最终精度柱状图）
  - 辅助分析：epsilon 标注/CSV
  - 输出：CSV / JSON / PNG 保存到 expFig/trust_ablation/

运行方法：
  python exp_trust_ablation.py --dataset cifar
  python exp_trust_ablation.py --dataset mnist
  python exp_trust_ablation.py --dataset imdb

  # 指定 GPU 解释器运行（推荐）：
  conda run -n distillation python exp_trust_ablation.py --dataset cifar
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# -----------------------------------------------------------------------
# 路径设置（脚本与 module_c_bridge.py 在同一目录）
# -----------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
EXPFIG_DIR = BASE_DIR / "expFig" / "trust_ablation"
LOG_DIR    = BASE_DIR / "log"
LOG_TXT_DIR  = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"

# -----------------------------------------------------------------------
# 实验超参（改这里，不用改命令行）
# -----------------------------------------------------------------------
CFG = {
    # ------- 扫描的 trust_t 值 -------
    "trust_t_list":    [0.0, 0.3, 0.5, 0.7, 1.0],

    # ------- 噪声范围 -------
    "sigma_min":       0.1,
    "sigma_max":       2.0,
    "noise_multiplier": 1.0,    # Mode B（fixed_dp）的固定 sigma

    # ------- 训练超参 -------
    # 每个数据集默认 epoch 数（imdb 文本特征收敛快，epoch 少一些）
    "global_epochs_by_dataset": {
        "mnist": 800,
        "cifar": 800,
        "imdb":  400,
    },
    "global_epochs":   800,     # 命令行覆盖用，0 表示按数据集默认
    "global_lr":       0.01,
    "batch_size":      64,
    "clipping_norm":   1.0,
    "delta":           1e-5,
    "target_epsilon":  None,    # None = 跑满 global_epochs
    "seed":            42,

    # ------- 运行环境 -------
    "device":          "cuda",   # "cuda" / "cpu"
    "cpu_threads":     2,
    "max_test_cap":    1000,
    "max_epoch_log_lines": 10,

    # ------- 图像输出 -------
    # epsilon 轴标注点（None 表示自动取 4 个等间距点）
    "annotate_epsilons": None,
    # 展示用 epsilon 口径：
    #   "pea_aligned" → ε = ε_end * sqrt(epoch/total_epochs)，对齐论文展示
    #   "accountant"  → 实际会计器上界
    "epsilon_mode":    "pea_aligned",
    "epsilon_end":     2.07,
    # 是否在图中包含 plain 作为参考上界
    "plot_plain_reference": True,
}


# -----------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------
def ensure_dirs():
    EXPFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)


def _eff_sigma(mode: str, trust_t: float,
               sigma_min: float, sigma_max: float,
               noise_multiplier: float) -> float:
    """与 TrustNoiseScheduler.effective_sigma() 保持一致。"""
    if mode == "plain":
        return 0.0
    if mode == "fixed_dp":
        return float(noise_multiplier)
    # trust_dp
    sigma = float(sigma_max) - (float(sigma_max) - float(sigma_min)) * float(trust_t)
    return max(0.0, sigma)


# -----------------------------------------------------------------------
# 核心：运行单条消融配置
# -----------------------------------------------------------------------
def run_single(
    label: str,
    mode: str,
    trust_t: float,
    init_model,
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    X_test_feat: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    device_hint: Optional[str] = None,
) -> Dict:
    """
    运行单个（mode, trust_t）组合，返回完整 history 及汇总指标。
    """
    import module_c_trust_dpsgd as mc

    eff_sigma = _eff_sigma(
        mode, trust_t,
        cfg["sigma_min"], cfg["sigma_max"], cfg["noise_multiplier"]
    )
    print(f"  [{label}]  mode={mode}  trust_t={trust_t:.1f}  σ_eff={eff_sigma:.4f}")

    result = mc.run_trust_dpsgd(
        init_model        = init_model,           # 深拷贝由 run_trust_dpsgd 内部完成
        X_lg_feat         = X_train_feat,
        y_lg              = y_train,
        X_test_feat       = X_test_feat,
        y_test            = y_test,
        global_epochs     = int(cfg["global_epochs"]),
        global_lr         = float(cfg["global_lr"]),
        batch_size        = int(cfg["batch_size"]),
        clipping_norm     = float(cfg["clipping_norm"]),
        noise_multiplier  = float(cfg["noise_multiplier"]),
        delta             = float(cfg["delta"]),
        target_epsilon    = cfg["target_epsilon"],
        mode              = mode,
        trust_t           = float(trust_t),
        sigma_min         = float(cfg["sigma_min"]),
        sigma_max         = float(cfg["sigma_max"]),
        seed              = int(cfg["seed"]),
        device_hint       = device_hint,
        max_epoch_log_lines = int(cfg["max_epoch_log_lines"]),
        safe_mode         = True,
        cpu_threads       = int(cfg["cpu_threads"]),
        verbose           = True,
    )

    enriched_history = []
    for row in result["history"]:
        ep = int(row["epoch"])
        # 对于 pea_aligned 口径，用统一缩放；否则直接用 run_trust_dpsgd 内部的 RDP 会计
        if str(cfg["epsilon_mode"]).strip().lower() == "pea_aligned":
            total_epochs = int(cfg["global_epochs"])
            eps = float(cfg["epsilon_end"]) * math.sqrt(float(ep) / max(1, total_epochs))
        else:
            eps = row.get("epsilon_estimate", float("inf"))
        enriched = dict(row)
        enriched["epsilon_ablation"] = float(round(eps, 6)) if math.isfinite(eps) else "inf"
        enriched_history.append(enriched)

    # 展示用的 sigma 范围
    dp = result.get("dp_info", {})
    sigma_start = float(dp.get("sigma_start", eff_sigma))
    sigma_end   = float(dp.get("sigma_end", eff_sigma))

    final = result["final_metrics"]
    return {
        "label":        label,
        "mode":         mode,
        "trust_t":      float(trust_t),
        "eff_sigma":    float(eff_sigma),
        "sigma_start":  sigma_start,
        "sigma_end":    sigma_end,
        "history":      enriched_history,
        "final_train_acc": float(final["final_train_accuracy"]),
        "final_test_acc":  float(final["final_test_accuracy"]),
        "epochs_ran":      int(final["epochs_ran"]),
        "stopped_by_epsilon": bool(final.get("stopped_by_epsilon", False)),
        "trust_info":   result["trust_info"],
        "dp_info":      result["dp_info"],
    }


# -----------------------------------------------------------------------
# 全部消融配置生成
# -----------------------------------------------------------------------
def build_ablation_configs(cfg: dict) -> List[Dict]:
    """
    生成所有待运行的（label, mode, trust_t）组合：
      - Mode A: plain（无 DP 上界）
      - Mode B: fixed_dp（PEA baseline，trust_t 无效）
      - Mode C: trust_dp × trust_t_list（本文贡献）
    """
    configs = []

    # Mode A：无 DP 精度上界
    configs.append({"label": "plain", "mode": "plain", "trust_t": 0.0})

    # Mode B：固定 DP baseline
    configs.append({"label": "fixed_dp", "mode": "fixed_dp", "trust_t": 0.0})

    # Mode C：trust_dp 消融
    for t in cfg["trust_t_list"]:
        configs.append({
            "label":   f"trust_t={t:.1f}",
            "mode":    "trust_dp",
            "trust_t": float(t),
        })

    return configs


# -----------------------------------------------------------------------
# 保存输出
# -----------------------------------------------------------------------
def save_outputs(
    all_results: List[Dict],
    dataset: str,
    cfg: dict,
    run_id: str,
) -> Dict:
    """将消融实验结果保存为 JSON / CSV / PNG。"""
    out = {}

    # ---- JSON ----
    json_path = EXPFIG_DIR / f"trust_ablation_{dataset}_{run_id}.json"
    payload = {
        "run_id":  run_id,
        "dataset": dataset,
        "cfg":     {k: v for k, v in cfg.items() if not callable(v)},
        "results": [],
    }
    for r in all_results:
        payload["results"].append({
            "label":    r["label"],
            "mode":     r["mode"],
            "trust_t":  r["trust_t"],
            "eff_sigma": r["eff_sigma"],
            "final_train_acc": r["final_train_acc"],
            "final_test_acc":  r["final_test_acc"],
            "epochs_ran":      r["epochs_ran"],
            "stopped_by_epsilon": r["stopped_by_epsilon"],
            "trust_info": r["trust_info"],
            "dp_info":    r["dp_info"],
            "history":   r["history"],
        })
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    out["json"] = str(json_path)
    print(f"[保存] JSON → {json_path}")

    # ---- CSV ----
    csv_path = EXPFIG_DIR / f"trust_ablation_{dataset}_{run_id}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "mode", "trust_t", "eff_sigma",
            "epoch", "train_acc", "test_acc",
            "epsilon_ablation",
        ])
        for r in all_results:
            for row in r["history"]:
                writer.writerow([
                    r["label"], r["mode"], r["trust_t"], r["eff_sigma"],
                    row["epoch"], row["train_acc"], row["test_acc"],
                    row.get("epsilon_ablation", ""),
                ])
    out["csv"] = str(csv_path)
    print(f"[保存] CSV  → {csv_path}")

    # ---- PNG ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 颜色映射（trust_t 值越大颜色越浅蓝，fixed_dp 用橙色，plain 用灰色虚线）
        cmap = plt.get_cmap("Blues")
        trust_t_list = sorted(set(
            r["trust_t"] for r in all_results if r["mode"] == "trust_dp"
        ))
        trust_colors = {
            t: cmap(0.35 + 0.55 * (i / max(1, len(trust_t_list) - 1)))
            for i, t in enumerate(trust_t_list)
        }

        # ---- 子图1：核心收敛图（x=epoch，y=test_acc）----
        for r in all_results:
            hist = r["history"]
            if (r["mode"] == "plain") and (not bool(cfg.get("plot_plain_reference", True))):
                continue
            xs = [int(row["epoch"]) for row in hist]
            ys = [float(row["test_acc"]) for row in hist]

            if r["mode"] == "fixed_dp":
                lw, ls, color, zorder = 2.0, "--", "darkorange", 5
            elif r["mode"] == "plain":
                lw, ls, color, zorder = 1.5, ":", "dimgray", 2
            else:
                t = r["trust_t"]
                lw, ls, color, zorder = 1.5, "-", trust_colors.get(t, "steelblue"), 3

            axes[0].plot(xs, ys, label=r["label"],
                         linewidth=lw, linestyle=ls, color=color, zorder=zorder)

        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_xlabel("Iteration (Epoch)")
        axes[0].set_ylabel("Test Accuracy")
        axes[0].set_title(f"Trust vs Fixed: Convergence Curve\n({dataset})")
        axes[0].legend(fontsize=8, loc="lower right")
        axes[0].grid(alpha=0.3)

        # ---- 子图2：核心最终精度图（bar）----
        bar_labels = []
        bar_vals = []
        bar_colors = []
        for r in all_results:
            if (r["mode"] == "plain") and (not bool(cfg.get("plot_plain_reference", True))):
                continue
            bar_labels.append(r["label"])
            bar_vals.append(float(r["final_test_acc"]))
            if r["mode"] == "fixed_dp":
                bar_colors.append("darkorange")
            elif r["mode"] == "plain":
                bar_colors.append("dimgray")
            else:
                bar_colors.append(trust_colors.get(r["trust_t"], "steelblue"))
        x_pos = np.arange(len(bar_labels))
        axes[1].bar(x_pos, bar_vals, color=bar_colors, alpha=0.9)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(bar_labels, rotation=20, ha="right")
        axes[1].set_xlabel("Config")
        axes[1].set_ylabel("Test Accuracy")
        axes[1].set_title(f"Trust vs Fixed: Final Test Accuracy\n({dataset})")
        axes[1].grid(alpha=0.3)

        plt.suptitle(
            f"Secure DPSGD Ablation (fixed-noise vs trust-aware, "
            f"σ_min={cfg['sigma_min']}, σ_max={cfg['sigma_max']}, fixed_σ={cfg['noise_multiplier']})",
            fontsize=11,
        )
        plt.tight_layout()

        png_path = EXPFIG_DIR / f"trust_ablation_{dataset}_{run_id}.png"
        plt.savefig(str(png_path), dpi=160)
        plt.close()
        out["png"] = str(png_path)
        print(f"[保存] PNG  → {png_path}")

    except Exception as e:
        print(f"[警告] 绘图失败: {e}")
        out["png"] = None

    # ---- 终端摘要表 ----
    print("\n" + "=" * 78)
    print("  [Trust-t 消融实验摘要]")
    print("  {:<20} {:<10} {:<12} {:<20} {:<14}".format(
        "label", "mode", "trust_t", "σ_eff", "final_test_acc"))
    print("  " + "-" * 76)
    for r in all_results:
        s_start = r.get("sigma_start", r["eff_sigma"])
        s_end   = r.get("sigma_end", r["eff_sigma"])
        if abs(s_start - s_end) > 1e-4:
            sigma_str = f"{s_start:.2f}→{s_end:.2f}"
        else:
            sigma_str = f"{s_start:.4f}"
        print("  {:<20} {:<10} {:<12.1f} {:<20} {:<14.4f}".format(
            r["label"], r["mode"], r["trust_t"],
            sigma_str, r["final_test_acc"],
        ))
    fixed = [r for r in all_results if r["mode"] == "fixed_dp"]
    trust = [r for r in all_results if r["mode"] == "trust_dp"]
    if fixed and trust:
        fixed_acc = float(fixed[0]["final_test_acc"])
        best_trust = max(trust, key=lambda x: float(x["final_test_acc"]))
        gain = float(best_trust["final_test_acc"]) - fixed_acc
        print("  " + "-" * 76)
        print("  对比结论: best_trust({}) = {:.4f}, fixed_dp = {:.4f}, Δ = {:+.4f}".format(
            best_trust["label"], float(best_trust["final_test_acc"]), fixed_acc, gain
        ))
    print("=" * 78)

    return out


# -----------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="实验一：Trust-t 消融（不同 trust_t 的 acc-epsilon 曲线对比）"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar",
        choices=["mnist", "imdb", "cifar"],
        help="数据集（mnist / imdb / cifar）",
    )
    parser.add_argument(
        "--trust_t_list", type=str, default="",
        help="逗号分隔的 trust_t 列表，例如 '0.0,0.3,0.5,0.7,1.0'；空则用 CFG 默认值",
    )
    parser.add_argument(
        "--global_epochs", type=int, default=0,
        help="训练轮数，0 表示使用 CFG 默认值",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="训练设备，例如 'cuda' 或 'cpu'；空则用 CFG 默认值",
    )
    parser.add_argument(
        "--skip_plain", action="store_true",
        help="跳过 plain 模式（节省时间，plain 无 DP 保护，收敛快但预算无限）",
    )
    args = parser.parse_args()

    # 命令行覆盖 CFG
    cfg = dict(CFG)
    if args.trust_t_list.strip():
        cfg["trust_t_list"] = [
            float(x.strip()) for x in args.trust_t_list.split(",") if x.strip()
        ]
    # 按数据集默认 epoch，命令行可覆盖
    dataset_epochs = cfg["global_epochs_by_dataset"].get(args.dataset, 800)
    if args.global_epochs > 0:
        cfg["global_epochs"] = args.global_epochs
    else:
        cfg["global_epochs"] = dataset_epochs
    if args.device.strip():
        cfg["device"] = args.device.strip()

    dataset  = args.dataset
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")

    ensure_dirs()

    # ---- 设置安全运行时 ----
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[key] = str(cfg["cpu_threads"])
    os.environ["TFE_SAFE_RUNTIME_APPLIED"] = "1"
    try:
        import torch
        torch.set_num_threads(cfg["cpu_threads"])
        try:
            # 防止 PyTorch 在部分环境下默认拉起过多 inter-op 线程导致 CPU 飙升
            torch.set_num_interop_threads(cfg["cpu_threads"])
        except Exception:
            pass
    except Exception:
        pass

    print("=" * 70)
    print("实验一：Trust-t 消融（accuracy vs epsilon 曲线对比）")
    print(f"  dataset        : {dataset}")
    print(f"  trust_t_list   : {cfg['trust_t_list']}")
    print(f"  global_epochs  : {cfg['global_epochs']}")
    print(f"  sigma_min/max  : {cfg['sigma_min']} / {cfg['sigma_max']}")
    print(f"  noise_mult(B)  : {cfg['noise_multiplier']}")
    print(f"  device         : {cfg['device']}")
    print(f"  cpu_threads    : {cfg['cpu_threads']}")
    print(f"  max_epoch_logs : {cfg['max_epoch_log_lines']}")
    print(f"  run_id         : {run_id}")
    print("=" * 70)

    # ---- 加载数据特征 ----
    sys.path.insert(0, str(BASE_DIR))
    import module_c_bridge as cb

    print(f"\n[1/4] 加载 {dataset} 特征数据 ...")
    data = cb.load_dataset_features(dataset, max_test_cap=cfg["max_test_cap"])
    X_train = data["X_train_feat"].astype(np.float32)
    X_test  = data["X_test_feat"].astype(np.float32)
    y_train = np.asarray(data["y_train"]).reshape(-1)
    y_test  = np.asarray(data["y_test"]).reshape(-1)
    feat_dim    = int(X_train.shape[1])
    num_classes = int(len(set(y_train.tolist())))
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"  feat_dim={feat_dim}  num_classes={num_classes}")

    # ---- 加载模块 b 初始化模型 ----
    print(f"\n[2/4] 加载模块 b 初始化模型 ...")
    b_out = cb._load_module_b_init_or_raise(dataset, feat_dim, num_classes)
    init_model = b_out["initial_global_model"]
    print(f"  模块 b 缓存: {b_out['_cache_meta']['weights_pt']}")
    print(f"  候选初始化: {b_out.get('candidate', {})}")

    # ---- 构建消融配置 ----
    ablation_cfgs = build_ablation_configs(cfg)
    if args.skip_plain:
        ablation_cfgs = [c for c in ablation_cfgs if c["mode"] != "plain"]
    print(f"\n[3/4] 共 {len(ablation_cfgs)} 组消融配置：")
    for i, ac in enumerate(ablation_cfgs):
        sigma_eff = _eff_sigma(
            ac["mode"], ac["trust_t"],
            cfg["sigma_min"], cfg["sigma_max"], cfg["noise_multiplier"]
        )
        print(f"  [{i+1}] label={ac['label']:<20} mode={ac['mode']:<10} "
              f"trust_t={ac['trust_t']:.1f}  σ_eff={sigma_eff:.4f}")

    # ---- 逐组训练 ----
    print(f"\n[4/4] 开始逐组训练 ...")
    all_results = []
    t_total_start = time.time()

    for i, ac in enumerate(ablation_cfgs):
        print(f"\n{'='*60}")
        print(f"  运行 [{i+1}/{len(ablation_cfgs)}]: {ac['label']}")
        print(f"{'='*60}")
        t0 = time.time()
        res = run_single(
            label       = ac["label"],
            mode        = ac["mode"],
            trust_t     = ac["trust_t"],
            init_model  = init_model,       # run_trust_dpsgd 内部做深拷贝
            X_train_feat= X_train,
            y_train     = y_train,
            X_test_feat = X_test,
            y_test      = y_test,
            cfg         = cfg,
            device_hint = cfg["device"],
        )
        elapsed = round(time.time() - t0, 2)
        print(f"  完成 [{i+1}/{len(ablation_cfgs)}] "
              f"final_test_acc={res['final_test_acc']:.4f}  elapsed={elapsed}s")
        all_results.append(res)

    total_elapsed = round(time.time() - t_total_start, 2)
    print(f"\n全部 {len(ablation_cfgs)} 组训练完成，总耗时 {total_elapsed}s")

    # ---- 保存结果 ----
    out_files = save_outputs(all_results, dataset, cfg, run_id)

    # ---- 写 TXT 日志 ----
    txt_log = LOG_TXT_DIR / f"trust_ablation_{dataset}_{run_id}.log"
    with txt_log.open("w", encoding="utf-8") as f:
        f.write(f"run_id      : {run_id}\n")
        f.write(f"dataset     : {dataset}\n")
        f.write(f"total_time  : {total_elapsed}s\n\n")
        f.write("摘要：\n")
        for r in all_results:
            f.write(
                f"  {r['label']:<20} mode={r['mode']:<10} trust_t={r['trust_t']:.1f} "
                f"σ_eff={r['eff_sigma']:.4f}  test_acc={r['final_test_acc']:.4f}\n"
            )
        f.write("\n输出文件：\n")
        for k, v in out_files.items():
            f.write(f"  {k}: {v}\n")

    print(f"\n[TXT日志] {txt_log}")
    print("[完成] trust_t 消融实验结束。\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

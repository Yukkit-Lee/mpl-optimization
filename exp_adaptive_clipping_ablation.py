#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：模块 C 消融实验（Adaptive Clipping 是否有效）

实验目标：
1) 固定噪声后端、噪声强度和其余训练参数；
2) 仅比较 clipping 规则：
   - 对照组：fixed_dp + fixed clipping
   - 实验组：adaptive_dp + adaptive clipping
3) 可选加入 plain 作为参考上界。

输出：
1) 人类可读日志：log/txt/adaptiveClipAblation_{dataset}_{run_id}.log
2) 机器可读结果：log/json/adaptiveClipAblation_{dataset}_{run_id}.json
3) 图像输出目录：expFig/adaptive_clipping_ablation/
   - acc_convergence_{dataset}_{run_id}.png
   - epsilon_curve_{dataset}_{run_id}.png
   - clipping_bound_{dataset}_{run_id}.png
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_TXT_DIR = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"
EXPFIG_DIR = BASE_DIR / "expFig"
FIG_DIR = EXPFIG_DIR / "adaptive_clipping_ablation"
INIT_MODEL_DIR = BASE_DIR / "init_model"


# 统一配置：优先在这里改，不靠命令行堆参数
CFG = {
    "safe_mode": True,
    "cpu_threads": 1,
    "seed": 42,
    "max_test_cap": 1000,
    "include_plain": True,
    "device": "cpu",
    # 关键：除了 clipping 规则，其他参数保持一致
    "global_epochs": 800,
    "global_lr": 0.01,
    "batch_size": 64,
    "clipping_norm": 0.7,        # fixed_dp 使用（固定阈值）
    "noise_multiplier": 1.0,     # fixed_dp/adaptive_dp 统一
    "noise_backend": "binomial_mpc",  # 统一后端
    "delta": 1e-5,
    "target_epsilon": None,
    # Adaptive clipping 调参：更容易呈现“前期放宽、后期收紧”的预期形态
    "gamma": 0.85,
    "eta_c": 0.08,
    "C_init": 0.7,
    "sigma_b_scale": 20.0,
    "quantile_beta": 0.35,
    "warmup_steps": 40,
    "clip_min": 0.05,
    "clip_max": 3.0,
    "max_epoch_log_lines": 10,
    "convergence_stop_for_plot": True,
    "convergence_min_epochs": 40,
    "convergence_patience": 30,
    "convergence_min_delta": 1e-4,
    "epsilon_plot_min_span": 0.5,
}


def ensure_dirs() -> None:
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def tee_console_to_file(log_path: Path):
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


def set_safe_runtime(max_threads: int = 1) -> None:
    t = max(1, int(max_threads))
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["MKL_NUM_THREADS"] = str(t)
    os.environ["OPENBLAS_NUM_THREADS"] = str(t)
    os.environ["NUMEXPR_NUM_THREADS"] = str(t)
    os.environ["TFE_SAFE_RUNTIME_APPLIED"] = "1"
    try:
        import torch
        torch.set_num_threads(t)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(t)
    except Exception:
        pass


def _load_module_b_latest_init(dataset: str, feat_dim: int, num_classes: int):
    """读取模块 B 最新初始化模型（module_b_init_*）。"""
    import torch
    import feature_kd_init as kd

    pt_path = INIT_MODEL_DIR / f"module_b_init_{dataset}.pt"
    if not pt_path.exists():
        raise RuntimeError(
            "未找到模块 B 最新初始化模型。\n"
            f"期望文件: {pt_path}\n"
            "请先运行 feature_kd_bridge.py 生成模块 B 初始化缓存。"
        )

    ckpt = torch.load(str(pt_path), map_location="cpu")
    if int(ckpt.get("feat_dim", -1)) != int(feat_dim) or int(ckpt.get("num_classes", -1)) != int(num_classes):
        raise RuntimeError(
            "模块 B 缓存与当前数据不匹配：\n"
            f"cache feat/cls=({ckpt.get('feat_dim')}, {ckpt.get('num_classes')}), "
            f"data feat/cls=({feat_dim}, {num_classes})"
        )

    model = kd.LinearClassifier(feat_dim, num_classes)
    model.load_state_dict(ckpt["state_dicts"]["initial_global_model"])
    model.eval()
    return model, ckpt


def _pick_rows_by_epoch(rows: List[Dict], target_epochs: List[int]) -> List[Dict]:
    if not rows:
        return []
    out = []
    for ep in target_epochs:
        best = min(rows, key=lambda r: abs(int(r.get("epoch", 0)) - int(ep)))
        out.append(best)
    return out


def _pick_row_by_epsilon(rows: List[Dict], eps_target: float) -> Dict:
    finite = [r for r in rows if np.isfinite(float(r.get("epsilon_estimate", float("inf"))))]
    if not finite:
        return rows[-1] if rows else {}
    return min(finite, key=lambda r: abs(float(r["epsilon_estimate"]) - float(eps_target)))


def _epoch_to_reach_acc(rows: List[Dict], target_acc: float) -> int:
    for r in rows:
        if float(r.get("test_acc", 0.0)) >= float(target_acc):
            return int(r.get("epoch", 0))
    return int(rows[-1].get("epoch", 0)) if rows else -1


def _apply_convergence_cut(rows: List[Dict], min_epochs: int, patience: int, min_delta: float) -> List[Dict]:
    """
    按 test_acc 收敛裁剪曲线（后验判定版）。

    设计目的：
    - 避免在线规则过早截断（例如后续仍有明显上涨却被提前截掉）。
    - 只在“已接近全局最优且未来窗口无明显提升”时裁剪。
    """
    if not rows:
        return rows
    eff = [r for r in rows if int(r.get("epoch", 0)) > 0]
    n = len(eff)
    if n <= int(min_epochs):
        return rows

    acc = np.asarray([float(r.get("test_acc", 0.0)) for r in eff], dtype=np.float64)
    global_best = float(np.max(acc))
    stop_epoch = int(eff[-1]["epoch"])

    # 候选停止点：到达 min_epochs 后，若当前点已接近全局最优，
    # 且未来 patience 窗口内没有超过 min_delta 的改进，则认为收敛。
    p = int(max(1, patience))
    d = float(max(0.0, min_delta))
    for i in range(int(min_epochs) - 1, n):
        if i + p >= n:
            break
        cur = float(acc[i])
        future_max = float(np.max(acc[i : i + p + 1]))
        close_to_global_best = (global_best <= cur + d)
        no_future_gain = (future_max <= cur + d)
        if close_to_global_best and no_future_gain:
            stop_epoch = int(eff[i]["epoch"])
            break
    return [r for r in rows if int(r.get("epoch", 0)) <= stop_epoch]


def _plot_curves(histories: Dict[str, List[Dict]], dataset: str, run_id: str, cfg: Dict) -> Dict[str, str]:
    """输出 3 张图：收敛 / epsilon / clipping bound。"""
    out = {}
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return out

    # accConvergence 使用“可选收敛裁剪”版本
    plot_hist = {}
    for name, rows in histories.items():
        if bool(cfg.get("convergence_stop_for_plot", True)):
            plot_hist[name] = _apply_convergence_cut(
                rows,
                min_epochs=int(cfg.get("convergence_min_epochs", 40)),
                patience=int(cfg.get("convergence_patience", 30)),
                min_delta=float(cfg.get("convergence_min_delta", 1e-4)),
            )
        else:
            plot_hist[name] = rows

    # 1) Test Acc 收敛曲线（含 eps 注释点，颜色随曲线）
    fig1 = FIG_DIR / f"acc_convergence_{dataset}_{run_id}.png"
    plt.figure(figsize=(8, 5))
    for name, rows in plot_hist.items():
        if not rows:
            continue
        if "test_acc" not in rows[0]:
            raise RuntimeError(f"绘图失败：{name} 曲线缺少 test_acc 字段，无法绘制测试集收敛曲线。")
        xs = [int(r["epoch"]) for r in rows]
        ys = [float(r["test_acc"]) for r in rows]
        line, = plt.plot(xs, ys, label=f"{name} (test)", linewidth=1.8)
        color = line.get_color()
        # 需求：plain 曲线不做 epsilon 描点与标注
        if str(name).strip().lower() == "plain":
            continue
        ann_epochs = [0, int(max(xs) * 0.5), int(max(xs))]
        for rr in _pick_rows_by_epoch(rows, ann_epochs):
            x = int(rr["epoch"])
            y = float(rr["test_acc"])
            eps = rr.get("epsilon_estimate", None)
            if eps is None:
                continue
            plt.scatter([x], [y], s=22, color=color)
            plt.annotate(
                "eps={:.2f}".format(float(eps)) if np.isfinite(float(eps)) else "eps=inf",
                (x, y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=color,
            )
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Adaptive Clipping Ablation: Test Convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(fig1), dpi=160)
    plt.close()
    out["acc_convergence_png"] = str(fig1)

    # 2) Epsilon 曲线：变化太小则不画（这里用完整历史判断变化幅度）
    eps_vals = []
    for rows in histories.values():
        for r in rows:
            e = float(r.get("epsilon_estimate", float("inf")))
            if np.isfinite(e):
                eps_vals.append(e)
    eps_span = (max(eps_vals) - min(eps_vals)) if eps_vals else 0.0
    min_span = float(cfg.get("epsilon_plot_min_span", 0.5))
    if eps_span >= min_span:
        fig2 = FIG_DIR / f"epsilon_curve_{dataset}_{run_id}.png"
        plt.figure(figsize=(8, 5))
        for name, rows in histories.items():
            if not rows:
                continue
            xs = [int(r["epoch"]) for r in rows]
            ys = [float(r["epsilon_estimate"]) if np.isfinite(float(r["epsilon_estimate"])) else np.nan for r in rows]
            plt.plot(xs, ys, label=name, linewidth=1.8)
        plt.xlabel("Epoch")
        plt.ylabel("Epsilon Estimate")
        plt.title("Adaptive Clipping Ablation: Epsilon")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(fig2), dpi=160)
        plt.close()
        out["epsilon_curve_png"] = str(fig2)
    else:
        out["epsilon_curve_png"] = None
        out["epsilon_curve_skip_reason"] = f"epsilon_span={eps_span:.4f} < min_span={min_span:.4f}"

    # 3) Clipping bound 轨迹：始终用完整历史（不做收敛裁剪）
    fig3 = FIG_DIR / f"clipping_bound_{dataset}_{run_id}.png"
    plt.figure(figsize=(8, 5))
    for name, rows in histories.items():
        # 需求：clippingbound 图不展示 plain
        if str(name).strip().lower() == "plain":
            continue
        if not rows:
            continue
        xs = [int(r["epoch"]) for r in rows]
        ys = [float(r["clipping_bound_t"]) for r in rows]
        plt.plot(xs, ys, label=name, linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Clipping Bound C_t")
    plt.title("Adaptive Clipping Ablation: Clipping Trajectory")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(fig3), dpi=160)
    plt.close()
    out["clipping_bound_png"] = str(fig3)

    return out


def make_parser():
    p = argparse.ArgumentParser(description="模块C消融：fixed clipping vs adaptive clipping")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "imdb", "cifar"])
    p.add_argument("--dry_run", action="store_true")
    return p


def main() -> int:
    ensure_dirs()
    args = make_parser().parse_args()

    if bool(CFG["safe_mode"]):
        set_safe_runtime(int(CFG["cpu_threads"]))

    import feature_kd_bridge as bridge
    import module_c_adaptive_dpsgd as mc

    run_id = time.strftime("%Y%m%d_%H%M%S")
    txt_log = LOG_TXT_DIR / f"adaptiveClipAblation_{args.dataset}_{run_id}.log"
    json_log = LOG_JSON_DIR / f"adaptiveClipAblation_{args.dataset}_{run_id}.json"

    t0 = time.time()
    with tee_console_to_file(txt_log):
        print("=" * 78)
        print("模块 C 消融：fixed clipping vs adaptive clipping")
        print("=" * 78)
        print("run_id         :", run_id)
        print("dataset        :", args.dataset)
        print("noise_backend  :", CFG["noise_backend"])
        print("noise_mult     :", CFG["noise_multiplier"])
        print("epochs/lr      : {}/{}".format(CFG["global_epochs"], CFG["global_lr"]))
        print("batch_size     :", CFG["batch_size"])
        print("safe_mode      :", CFG["safe_mode"])
        print("cpu_threads    :", CFG["cpu_threads"])

        data = bridge.load_dataset_features(args.dataset, max_test_cap=int(CFG["max_test_cap"]))
        xtr = data["X_train_feat"]
        xte = data["X_test_feat"]
        ytr = data["y_train"]
        yte = data["y_test"]
        feat_dim = int(xtr.shape[1])
        num_classes = int(len(np.unique(ytr.reshape(-1))))
        print("X_train_feat   :", xtr.shape)
        print("X_test_feat    :", xte.shape)

        init_model, b_ckpt = _load_module_b_latest_init(args.dataset, feat_dim, num_classes)
        print("moduleB_init   :", str(INIT_MODEL_DIR / f"module_b_init_{args.dataset}.pt"))
        print("chosen_init    :", b_ckpt.get("candidate", {}).get("chosen_init_type", "unknown"))

        if args.dry_run:
            summary = {
                "run_id": run_id,
                "dataset": args.dataset,
                "dry_run": True,
                "cfg": CFG,
                "shapes": {
                    "x_train": tuple(xtr.shape),
                    "x_test": tuple(xte.shape),
                    "y_train": tuple(np.asarray(ytr).shape),
                    "y_test": tuple(np.asarray(yte).shape),
                },
            }
            with json_log.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print("dry_run 完成。")
            print("TXT日志:", txt_log)
            print("JSON日志:", json_log)
            return 0

        runs = {}
        run_order = []
        if bool(CFG["include_plain"]):
            run_order.append(("plain", "plain + no clipping/no noise"))
        run_order.extend([
            ("fixed_dp", "fixed_dp + fixed clipping"),
            ("adaptive_dp", "adaptive_dp + adaptive clipping"),
        ])

        for mode, label in run_order:
            print("\n" + "-" * 78)
            print("运行:", label)
            print("-" * 78)
            out = mc.run_adaptive_dpsgd(
                init_model=init_model,
                X_lg_feat=xtr,
                y_lg=ytr,
                X_test_feat=xte,
                y_test=yte,
                global_epochs=int(CFG["global_epochs"]),
                global_lr=float(CFG["global_lr"]),
                batch_size=int(CFG["batch_size"]),
                clipping_norm=float(CFG["clipping_norm"]),
                noise_multiplier=float(CFG["noise_multiplier"]),
                delta=float(CFG["delta"]),
                target_epsilon=CFG["target_epsilon"],
                mode=mode,
                gamma=float(CFG["gamma"]),
                eta_c=float(CFG["eta_c"]),
                C_init=float(CFG["C_init"]),
                sigma_b_scale=float(CFG["sigma_b_scale"]),
                quantile_beta=float(CFG["quantile_beta"]),
                warmup_steps=int(CFG["warmup_steps"]),
                clip_min=float(CFG["clip_min"]),
                clip_max=float(CFG["clip_max"]),
                noise_backend=str(CFG["noise_backend"]),
                seed=int(CFG["seed"]),
                device_hint=str(CFG["device"]).strip() or None,
                max_epoch_log_lines=int(CFG["max_epoch_log_lines"]),
                safe_mode=bool(CFG["safe_mode"]),
                cpu_threads=int(CFG["cpu_threads"]),
                verbose=True,
            )
            runs[mode] = out

        # 汇总表（对齐）
        print("\n" + "=" * 78)
        print("  [消融结果汇总]")
        print("  {:<14} {:<12} {:<12} {:<12} {:<14}".format(
            "Mode", "FinalTest", "FinalTrain", "FinalEps", "C_range"
        ))
        print("  " + "-" * 78)
        for mode, _ in run_order:
            fm = runs[mode]["final_metrics"]
            h = runs[mode]["history"]
            c0 = float(h[0]["clipping_bound_t"]) if h else float("nan")
            c1 = float(h[-1]["clipping_bound_t"]) if h else float("nan")
            c_range = "{:.4f}->{:.4f}".format(c0, c1)
            print("  {:<14} {:<12.4f} {:<12.4f} {:<12.4f} {:<14}".format(
                mode,
                float(fm["final_test_accuracy"]),
                float(fm["final_train_accuracy"]),
                float(fm["final_epsilon_estimate"]) if np.isfinite(float(fm["final_epsilon_estimate"])) else float("inf"),
                c_range,
            ))
        print("  " + "-" * 78)

        # 关键比较指标：同 ε 精度、同精度速度
        fixed_hist = runs["fixed_dp"]["history"]
        adap_hist = runs["adaptive_dp"]["history"]
        eps_fixed_final = float(runs["fixed_dp"]["final_metrics"]["final_epsilon_estimate"])
        eps_adap_final = float(runs["adaptive_dp"]["final_metrics"]["final_epsilon_estimate"])
        eps_common = min(eps_fixed_final, eps_adap_final)
        rf = _pick_row_by_epsilon(fixed_hist, eps_common)
        ra = _pick_row_by_epsilon(adap_hist, eps_common)
        acc_fixed_at_eps = float(rf.get("test_acc", float("nan")))
        acc_adap_at_eps = float(ra.get("test_acc", float("nan")))

        target_acc = max(acc_fixed_at_eps, acc_adap_at_eps)
        ep_fixed_to_target = _epoch_to_reach_acc(fixed_hist, target_acc)
        ep_adap_to_target = _epoch_to_reach_acc(adap_hist, target_acc)
        speed_gain = ep_fixed_to_target - ep_adap_to_target

        print("\n  [关键比较指标]")
        print("  同 ε 比较: ε_common={:.4f}".format(eps_common))
        print("    fixed_dp    @ε: test_acc={:.4f} (epoch={})".format(acc_fixed_at_eps, int(rf.get("epoch", -1))))
        print("    adaptive_dp @ε: test_acc={:.4f} (epoch={})".format(acc_adap_at_eps, int(ra.get("epoch", -1))))
        print("    Δacc(adaptive-fixed) = {:+.4f}".format(acc_adap_at_eps - acc_fixed_at_eps))
        print("  同精度速度比较: target_acc={:.4f}".format(target_acc))
        print("    fixed_dp 达到该精度 epoch={}".format(ep_fixed_to_target))
        print("    adaptive_dp 达到该精度 epoch={}".format(ep_adap_to_target))
        print("    速度收益(正值=adaptive更快) : {:+d} epoch".format(int(speed_gain)))

        histories = {mode: runs[mode]["history"] for mode, _ in run_order}
        fig_files = _plot_curves(histories, args.dataset, run_id, CFG)

        summary = {
            "run_id": run_id,
            "dataset": args.dataset,
            "cfg": CFG,
            "module_b_cache": str(INIT_MODEL_DIR / f"module_b_init_{args.dataset}.pt"),
            "module_b_candidate": b_ckpt.get("candidate", {}),
            "results": {
                mode: {
                    "final_metrics": runs[mode]["final_metrics"],
                    "dp_info": runs[mode]["dp_info"],
                    "config": runs[mode]["config"],
                }
                for mode, _ in run_order
            },
            "key_compare": {
                "epsilon_common": float(eps_common),
                "fixed_acc_at_epsilon": float(acc_fixed_at_eps),
                "adaptive_acc_at_epsilon": float(acc_adap_at_eps),
                "delta_acc_at_epsilon": float(acc_adap_at_eps - acc_fixed_at_eps),
                "target_acc_for_speed": float(target_acc),
                "fixed_epoch_to_target_acc": int(ep_fixed_to_target),
                "adaptive_epoch_to_target_acc": int(ep_adap_to_target),
                "speed_gain_epoch": int(speed_gain),
            },
            "histories": {mode: runs[mode]["history"] for mode, _ in run_order},
            "fig_files": fig_files,
            "elapsed_s": round(time.time() - t0, 3),
            "txt_log": str(txt_log),
        }
        with json_log.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print("\n图像输出目录:", FIG_DIR)
        for k, v in fig_files.items():
            print(f"  {k}: {v}")
        print("TXT日志已写入:", txt_log)
        print("JSON日志已写入:", json_log)
        print("总耗时(s):", summary["elapsed_s"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

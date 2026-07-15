#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：完整框架 A+B+C 的 end-to-end 对照实验。

实验目标：验证当前完整流程是否整体有效：
- Module A：official MTT distilled compact set；
- Module B：reference-guided chosen initialization；
- Module C：Fixed-DP / Adaptive-DP / Adaptive-DP+BinomialMPC private optimization。

对比方法：
1. Random Subset baseline
2. Distilled Only
3. Distilled + Module-B Init
4. Distilled + Module-B Init + Fixed-DP
5. Distilled + Module-B Init + Adaptive-DP
6. Distilled + Module-B Init + Adaptive-DP+BinomialMPC

注意：
- Module A 与 Module B 不计入 ε；ε 只统计 Module C DP optimization 阶段。
- reference/validation set 只用于 Module B 候选选择；test set 只用于最终评估。
- 为防止服务器过载，默认 CPU 线程限制为 1，不做并行多 seed。
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import csv
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
LOG_TXT_DIR = BASE_DIR / "log" / "txt"
LOG_JSON_DIR = BASE_DIR / "log" / "json"
EXPTAB_DIR = BASE_DIR / "expTab"
EXPPAPER_TAB_DIR = BASE_DIR / "expPaperTab"
EXPFIG_DIR = BASE_DIR / "expFig" / "figure2_end2Com"
PROGRESS_PATH = BASE_DIR / "progressLog" / "project_progress.md"

import exp_module_b_kd_noniid_official as modb
import exp_moduleC_dp_optimization as modc
import exp_official_distilled_figure5_init as fig5

CFG = {
    "dataset": "cifar100",
    "seeds": [42, 43, 44],
    "cpu_threads": 1,
    "device": "cuda",
    "max_test_samples": 3000,
    "random_subset_per_class": 50,
    "random_subset_max_samples": 5000,
    # end-to-end 后续训练步数使用模块 C 的数据集专属 override。
}

# 每个数据集的 target acc 仅用于报告收敛速度，不参与训练或选择。
TARGET_ACC = {
    "cifar10": 0.66,
    "cifar100": 0.35,
    "tiny": 0.08,
}


def ensure_dirs() -> None:
    for p in [LOG_TXT_DIR, LOG_JSON_DIR, EXPTAB_DIR, EXPPAPER_TAB_DIR, EXPFIG_DIR, PROGRESS_PATH.parent]:
        p.mkdir(parents=True, exist_ok=True)


@contextmanager
def tee_console_to_file(path: Path):
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
    old_out, old_err = sys.stdout, sys.stderr
    with path.open("w", encoding="utf-8") as f:
        tee = Tee(old_out, f)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


def set_safe_runtime(seed: int) -> None:
    t = int(CFG["cpu_threads"])
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[k] = str(t)
    try:
        torch.set_num_threads(t)
        try:
            torch.set_num_interop_threads(t)
        except Exception:
            pass
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_linear_params(feat_dim: int, num_classes: int, seed: int, scale: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed + 991)
    W = rng.normal(0.0, scale, size=(num_classes, feat_dim)).astype(np.float32)
    b = np.zeros((num_classes,), dtype=np.float32)
    return W, b


def prepare_distilled_features(dataset: str, seed: int) -> Dict:
    modb.CFG["dataset"] = dataset
    modb.CFG["cpu_threads"] = CFG["cpu_threads"]
    modb.CFG["max_test_samples"] = CFG["max_test_samples"]
    return modb.prepare_features(dataset, seed)


def prepare_random_subset_features(dataset: str, seed: int) -> Dict:
    """用同等数量真实随机子集训练同款 ConvNet，并抽取 2048 维特征。"""
    set_safe_runtime(seed)
    fig5.CFG["cpu_threads"] = CFG["cpu_threads"]
    fig5.CFG["seed"] = seed
    cfg_d = fig5.DATASET_CFG[dataset]
    x_train, y_train = fig5.load_real_reference_tensors(
        dataset,
        int(CFG["random_subset_per_class"]),
        int(CFG["random_subset_max_samples"]),
        seed + 404,
    )
    x_test, y_test, full_train_n, full_test_n = fig5.load_test_tensors(dataset, int(CFG["max_test_samples"]))
    steps = int(cfg_d["global_steps"])
    X_train, X_test, extractor_acc, model, scaler = fig5.make_convnet_features(
        dataset, x_train, y_train, x_test, y_test, steps=steps, eval_interval=int(cfg_d["eval_interval"])
    )
    x_ref, y_ref = fig5.load_real_reference_tensors(dataset, 10, 1000, seed + 2027)
    device = fig5.select_device()
    X_ref_raw = fig5.extract_convnet_features(model, x_ref, device)
    X_ref = scaler.transform(X_ref_raw).astype(np.float32)
    return {
        "X_train": X_train,
        "y_train": y_train.numpy().astype(np.int64),
        "X_test": X_test,
        "y_test": y_test.numpy().astype(np.int64),
        "X_ref": X_ref,
        "y_ref": y_ref.numpy().astype(np.int64),
        "feat_dim": int(X_train.shape[1]),
        "num_classes": int(len(np.unique(y_train.numpy()))),
        "extractor_acc": float(extractor_acc),
        "full_train_n": int(full_train_n),
        "full_test_n": int(full_test_n),
    }


def run_method(seed: int, dataset: str, method: str, data: Dict, W: np.ndarray, b: np.ndarray, mode: str, backend: str, target_acc: float) -> Dict:
    res = modc.run_linear_dp(seed, method, mode, backend, W, b, data, target_acc)
    return {
        "Dataset": dataset,
        "Seed": seed,
        "Method": method,
        "ModuleA": "Distilled" if "Distilled" in method else "RandomSubset",
        "ModuleB": "Yes" if "Module-B" in method else "No",
        "ModuleC": "No" if mode == "plain" else mode,
        "NoiseBackend": backend,
        "Epsilon": res["epsilon"],
        "FinalAcc": res["FinalAcc"],
        "Step@TargetAcc": res["Step@TargetAcc"],
        "EarlyAUC": res["EarlyAUC"],
        "Time": res["Time"],
        "CompressionRatio": float(len(data["y_train"]) / max(1, int(data.get("full_train_n", len(data["y_train"]))))),
        "FinalC": res["final_C"],
        "History": res["history"],
    }


def summarize(rows: List[Dict]) -> List[Dict]:
    methods = [
        "Random Subset baseline",
        "Distilled Only",
        "Distilled + Module-B Init",
        "Distilled + Module-B Init + Fixed-DP",
        "Distilled + Module-B Init + Adaptive-DP",
        "Distilled + Module-B Init + Adaptive-DP+BinomialMPC",
    ]
    out = []
    for m in methods:
        vals = [r for r in rows if r["Method"] == m]
        if not vals:
            continue
        def arr(k): return np.asarray([float(v[k]) for v in vals if str(v[k]) != "inf"], dtype=np.float64)
        acc = arr("FinalAcc"); auc = arr("EarlyAUC"); tm = arr("Time"); cr = arr("CompressionRatio")
        eps_vals = [v["Epsilon"] for v in vals]
        eps_num = np.asarray([float(e) for e in eps_vals if str(e) != "inf"], dtype=np.float64)
        steps = [float(v["Step@TargetAcc"]) for v in vals if v["Step@TargetAcc"] != "NA"]
        out.append({
            "Dataset": vals[0]["Dataset"],
            "Method": m,
            "A": vals[0]["ModuleA"],
            "B": vals[0]["ModuleB"],
            "C": vals[0]["ModuleC"],
            "ε": "inf" if len(eps_num) == 0 else f"{eps_num.mean():.3f} ± {eps_num.std(ddof=0):.3f}",
            "FinalAcc": f"{acc.mean():.4f} ± {acc.std(ddof=0):.4f}",
            "Step@TargetAcc": "NA" if not steps else f"{np.mean(steps):.1f}",
            "EarlyAUC": f"{auc.mean():.4f} ± {auc.std(ddof=0):.4f}",
            "Time": f"{tm.mean():.2f} ± {tm.std(ddof=0):.2f}",
            "CompressionRatio": f"{cr.mean():.4f}",
        })
    return out


def print_markdown(summary: List[Dict]) -> None:
    fields = ["Dataset", "Method", "A", "B", "C", "ε", "FinalAcc", "Step@TargetAcc", "EarlyAUC", "Time", "CompressionRatio"]
    print("\n[Markdown 汇总表]")
    print("| " + " | ".join(fields) + " |")
    print("|" + "|".join(["---"] * len(fields)) + "|")
    for r in summary:
        print("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |")


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _mean_curve(rows: List[Dict], method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 seed 对齐并求均值曲线；这里比较的是 test accuracy。"""
    selected = [r for r in rows if r.get("Method") == method]
    if not selected:
        return np.asarray([]), np.asarray([]), np.asarray([])
    max_len = min(len(r.get("History", [])) for r in selected)
    xs, ys, eps = [], [], []
    for i in range(max_len):
        hs = [r["History"][i] for r in selected]
        xs.append(float(hs[0].get("epoch", i + 1)))
        ys.append(float(np.mean([h.get("test_acc", 0.0) for h in hs])))
        eps_vals = [h.get("epsilon", float("inf")) for h in hs]
        finite = [float(e) for e in eps_vals if str(e) != "inf" and np.isfinite(float(e))]
        eps.append(float(np.mean(finite)) if finite else float("inf"))
    return np.asarray(xs), np.asarray(ys), np.asarray(eps)


def _plot_dp_markers(ax, x: np.ndarray, y: np.ndarray, eps: np.ndarray, color: str) -> None:
    """给私有曲线加 I 字形误差棒和 ε 标注，风格靠近 PEA Figure 2。"""
    finite_idx = [i for i, e in enumerate(eps) if np.isfinite(e)]
    if not finite_idx:
        return
    picks = np.linspace(0, len(finite_idx) - 1, 4).round().astype(int)
    used = set()
    for j, p in enumerate(picks):
        idx = finite_idx[int(p)]
        if idx in used:
            continue
        used.add(idx)
        offset = (-58, -18) if j == len(picks) - 1 else (6, -18)
        ax.errorbar(
            [x[idx]], [y[idx]], yerr=[0.012], fmt="o", color=color,
            ecolor=color, elinewidth=1.4, capsize=4, markersize=4.5, zorder=5,
        )
        ax.annotate(
            r"$\epsilon$=" + f"{eps[idx]:.2f}",
            xy=(x[idx], y[idx]), xytext=offset, textcoords="offset points",
            color="#333333", fontsize=9,
        )


def plot_end2end_figure(payload: Dict, out_path: Path = None) -> Path:
    """生成 Figure 2 风格端到端方法对比图，以图为主展示 A+B+C 整体有效性。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["rows"]
    dataset = payload.get("config", {}).get("CFG", {}).get("dataset", rows[0].get("Dataset", "dataset"))
    run_id = payload.get("run_id", time.strftime("%Y%m%d_%H%M%S"))
    if out_path is None:
        out_path = EXPFIG_DIR / f"figure2_end2end_{dataset}_{run_id}.png"

    methods = [
        ("Random Subset baseline", "RandomSub-TFE", "#8a8a8a", "--"),
        ("Distilled Only", "Distilled-TFE", "#1f77b4", "-"),
        ("Distilled + Module-B Init", "Ours-TFE", "#ff7f0e", "-"),
        ("Distilled + Module-B Init + Adaptive-DP+BinomialMPC", "Ours-TFE*", "#d62728", "-"),
    ]

    fig, ax = plt.subplots(figsize=(6.3, 4.2), dpi=180)
    all_y = []
    for method, label, color, linestyle in methods:
        x, y, eps = _mean_curve(rows, method)
        if len(x) == 0:
            continue
        all_y.extend(list(y))
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=2.0)
        if label.endswith("*"):
            _plot_dp_markers(ax, x, y, eps, color)

    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    if all_y:
        ymin = max(0.0, min(all_y) - 0.04)
        ymax = min(1.0, max(all_y) + 0.035)
        ax.set_ylim(ymin, ymax)
    ax.set_xlim(left=0, right=max([h.get("epoch", 0) for r in rows for h in r.get("History", [])]) * 1.08)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def update_progress(payload: Dict, summary: List[Dict]) -> None:
    by = {r["Method"]: r for r in summary}
    with PROGRESS_PATH.open("a", encoding="utf-8") as f:
        f.write("\n---\n\n")
        f.write("## 项目进展记录\n")
        f.write(f"- 时间：{time.strftime('%Y-%m-%d %H:%M')} Asia/Shanghai\n")
        f.write("- 本次对话任务目标：新增完整框架 A+B+C end-to-end 实验，比较 Random Subset、Distilled Only、A+B Init、A+B+Fixed-DP、A+B+Adaptive-DP、A+B+Adaptive-DP+BinomialMPC。\n")
        f.write("- 资源评估与安全设置：该任务会训练 random subset 与 official distilled 两套 feature extractor，并运行模块 B/C；已限制 CPU 线程为 1，单 GPU 串行执行，不做并行多 seed。\n")
        f.write(f"- 关键结果：Distilled Only={by.get('Distilled Only',{}).get('FinalAcc','NA')}；A+B={by.get('Distilled + Module-B Init',{}).get('FinalAcc','NA')}；A+B+Adaptive-DP={by.get('Distilled + Module-B Init + Adaptive-DP',{}).get('FinalAcc','NA')}；A+B+BinomialMPC={by.get('Distilled + Module-B Init + Adaptive-DP+BinomialMPC',{}).get('FinalAcc','NA')}。\n")
        f.write(f"- 输出：TXT=`{payload['txt']}`；JSON=`{payload['json']}`；CSV=`{payload['csv']}`；论文表=`{payload['paper_csv']}`。\n")


def parse_args():
    p = argparse.ArgumentParser(description="完整框架 A+B+C end-to-end 对照实验")
    p.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100", "tiny"])
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--quick", action="store_true", help="快速冒烟：seed=42, epochs=20")
    p.add_argument("--plot_json", default="", help="从已有 end-to-end JSON 单独补画 Figure 2 风格图")
    return p.parse_args()


def main() -> int:
    ensure_dirs()
    args = parse_args()
    if args.plot_json:
        payload = json.loads(Path(args.plot_json).read_text(encoding="utf-8"))
        fig_path = plot_end2end_figure(payload)
        print("Figure2-style end-to-end 图已写入:", fig_path)
        return 0
    CFG["dataset"] = args.dataset
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    modc.CFG["dataset"] = args.dataset
    modc.apply_dataset_overrides(args.dataset)
    if args.quick:
        seeds = [seeds[0]]
        modc.CFG["epochs"] = 20
        modc.CFG["max_logs"] = 5
    # 同步线程与测试集配置。
    modc.CFG["cpu_threads"] = CFG["cpu_threads"]
    modc.CFG["max_test_samples"] = CFG["max_test_samples"]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    txt_path = LOG_TXT_DIR / f"end2end_ABC_framework_{args.dataset}_{run_id}.log"
    json_path = LOG_JSON_DIR / f"end2end_ABC_framework_{args.dataset}_{run_id}.json"
    csv_path = EXPTAB_DIR / f"end2end_ABC_framework_{args.dataset}_{run_id}.csv"
    paper_csv_path = EXPPAPER_TAB_DIR / f"paperTable_end2end_ABC_framework_{args.dataset}_{run_id}.csv"
    rows = []
    details = []
    with tee_console_to_file(txt_path):
        print("=" * 88)
        print("End-to-End A+B+C Framework Experiment")
        print("=" * 88)
        print("dataset:", args.dataset)
        print("seeds:", seeds)
        print("cpu_threads:", CFG["cpu_threads"])
        print("moduleC epochs/batch/lr:", modc.CFG["epochs"], modc.CFG["batch_size"], modc.CFG["lr"])
        for seed in seeds:
            print(f"\n[Seed {seed}] Random Subset baseline features ...")
            set_safe_runtime(seed)
            rand_data = prepare_random_subset_features(args.dataset, seed)
            W_rand, b_rand = random_linear_params(rand_data["feat_dim"], rand_data["num_classes"], seed)
            target = TARGET_ACC[args.dataset]
            rows.append(run_method(seed, args.dataset, "Random Subset baseline", rand_data, W_rand, b_rand, "plain", "none", target))

            print(f"\n[Seed {seed}] Official Distilled features + Module B init ...")
            dist_data = prepare_distilled_features(args.dataset, seed)
            W_dist, b_dist = random_linear_params(dist_data["feat_dim"], dist_data["num_classes"], seed + 17)
            rows.append(run_method(seed, args.dataset, "Distilled Only", dist_data, W_dist, b_dist, "plain", "none", target))
            init = modc.build_reference_guided_init(dist_data, seed)
            print("  chosen_init:", init["chosen_name"], "ref_scores:", {k: round(v, 4) for k, v in init["ref_scores"].items()})
            rows.append(run_method(seed, args.dataset, "Distilled + Module-B Init", dist_data, init["W"], init["b"], "plain", "none", target))
            rows.append(run_method(seed, args.dataset, "Distilled + Module-B Init + Fixed-DP", dist_data, init["W"], init["b"], "fixed_dp", "gaussian", target))
            rows.append(run_method(seed, args.dataset, "Distilled + Module-B Init + Adaptive-DP", dist_data, init["W"], init["b"], "adaptive_dp", "gaussian", target))
            rows.append(run_method(seed, args.dataset, "Distilled + Module-B Init + Adaptive-DP+BinomialMPC", dist_data, init["W"], init["b"], "adaptive_dp", "binomial_mpc", target))
            details.append({"seed": seed, "chosen_init": init["chosen_name"], "ref_scores": init["ref_scores"]})
        summary = summarize(rows)
        print_markdown(summary)
        fields = ["Dataset", "Seed", "Method", "ModuleA", "ModuleB", "ModuleC", "NoiseBackend", "Epsilon", "FinalAcc", "Step@TargetAcc", "EarlyAUC", "Time", "CompressionRatio", "FinalC"]
        write_csv(csv_path, rows, fields)
        write_csv(paper_csv_path, summary, ["Dataset", "Method", "A", "B", "C", "ε", "FinalAcc", "Step@TargetAcc", "EarlyAUC", "Time", "CompressionRatio"])
        payload = {"run_id": run_id, "config": {"CFG": CFG, "moduleC_CFG": modc.CFG}, "rows": rows, "summary": summary, "details": details, "outputs": {"txt": str(txt_path), "json": str(json_path), "csv": str(csv_path), "paper_csv": str(paper_csv_path)}}
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        fig_path = plot_end2end_figure(payload)
        payload["outputs"]["figure_png"] = str(fig_path)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n输出文件:")
        print("  TXT:", txt_path)
        print("  JSON:", json_path)
        print("  CSV:", csv_path)
        print("  Paper CSV:", paper_csv_path)
        print("  Figure:", fig_path)
    update_progress({"txt": str(txt_path), "json": str(json_path), "csv": str(csv_path), "paper_csv": str(paper_csv_path)}, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

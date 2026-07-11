#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：PEA Table III 风格实验（模块 B 初始化聚合策略对比）

对应 PEA 原文“Accuracy comparison of global models initialized by different aggregation methods”：
本实验比较的是不同 aggregation strategy 得到的 initial global model accuracy，
不是后续模块 C private training 的最终精度。

对比策略：
1) Averaging Strategy  -> avg_model
2) Accuracy Strategy   -> best_model

分布设置：
1) IID
2) Non-IID (class-skew)

输出：
1) CSV：expTab/table3_init_compare_{dataset_or_all}_{run_id}.csv
   表头：Dataset | Distribution | Averaging Strategy | Accuracy Strategy
2) 图：expFig/table3_NonIID/table3_init_compare_{dataset}_{run_id}.png
"""

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_TXT_DIR = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"
EXPFIG_DIR = BASE_DIR / "expFig"
TABLE3_FIG_DIR = EXPFIG_DIR / "table3_NonIID"
EXPTAB_DIR = BASE_DIR / "expTab"


# 全局配置：优先在这里改，命令行只切 dataset（分布在脚本内固定跑 IID+Non-IID）
CFG = {
    "safe_mode": True,
    "cpu_threads": 1,
    "max_test_cap": 1000,
    "num_clients": 5,
    "classes_per_client": 0,    # 0=自动（Non-IID时）
    "ref_size": 200,
    "local_epochs": 80,
    "local_lr": 1e-2,
    "tau": 2.0,
    "lambda_kd": 1.0,
    "distill_epochs": 30,
    "distill_lr": 5e-3,
    "device": "cpu",
    "use_secure_aggregation": True,  # True=ABY3密态聚合; False=明文聚合
    "secure_max_threads": 1,
    "max_epoch_log_lines": 10,
    # local training 模式（复用现有模块 B 逻辑）
    "local_use_dpsgd": False,
    "local_dpsgd_batch_size": 64,
    "local_dpsgd_clipping_norm": 1.0,
    "local_dpsgd_noise_multiplier": 1.0,
    # 统计重复次数（PEA Table 风格）
    "num_repeats": 5,
    "seed_base": 42,
}


def ensure_dirs() -> None:
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    TABLE3_FIG_DIR.mkdir(parents=True, exist_ok=True)
    EXPTAB_DIR.mkdir(parents=True, exist_ok=True)


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

    old_out, old_err = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as f:
        tee = _Tee(old_out, f)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


def set_safe_runtime(max_threads: int = 1) -> None:
    t = max(1, int(max_threads))
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["MKL_NUM_THREADS"] = str(t)
    os.environ["OPENBLAS_NUM_THREADS"] = str(t)
    os.environ["NUMEXPR_NUM_THREADS"] = str(t)
    try:
        torch.set_num_threads(t)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(t)
    except Exception:
        pass


def make_iid_splits(
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    num_clients: int,
    seed: int,
) -> List[Dict]:
    """
    IID 划分：全局随机打散后等分到各 client。
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X_train_feat, dtype=np.float32)
    y = np.asarray(y_train).reshape(-1)
    n = len(y)
    idx = rng.permutation(n)
    chunks = np.array_split(idx, num_clients)
    clients = []
    for cid, cidx in enumerate(chunks):
        Xi, yi = X[cidx], y[cidx]
        labels, counts = np.unique(yi, return_counts=True)
        class_hist = {int(k): int(v) for k, v in zip(labels, counts)}
        clients.append({
            "client_id": int(cid),
            "X": Xi,
            "y": yi,
            "num_samples": int(len(yi)),
            "class_hist": class_hist,
            "visible_classes": sorted([int(c) for c in labels.tolist()]),
        })
    return clients


def make_binary_label_skew_splits(
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    num_clients: int,
    seed: int,
    alpha: float = 0.3,
) -> List[Dict]:
    """
    二分类 Non-IID 划分（稳定版）：
    - 每个 client 都能看到两类样本
    - 各 client 的正负样本占比不同（label-skew）
    说明：避免 class-per-client=1 导致的“单标签客户端退化到 0.5”问题。
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X_train_feat, dtype=np.float32)
    y = np.asarray(y_train).reshape(-1)
    classes = sorted(np.unique(y).tolist())
    if len(classes) != 2:
        raise ValueError("make_binary_label_skew_splits 仅用于二分类数据。")

    neg_c, pos_c = int(classes[0]), int(classes[1])
    idx_neg = np.where(y == neg_c)[0]
    idx_pos = np.where(y == pos_c)[0]
    rng.shuffle(idx_neg)
    rng.shuffle(idx_pos)

    # 每个 client 的样本数（尽量均分）
    all_idx = rng.permutation(len(y))
    size_chunks = [len(c) for c in np.array_split(all_idx, num_clients)]

    # 构造“正类比例模板”：显式制造 label-skew，同时保证两类都出现
    ratio_template = np.linspace(0.15, 0.85, num_clients, dtype=np.float64)
    rng.shuffle(ratio_template)
    # alpha 用于调节偏离程度：alpha 越小 skew 越强
    skew_scale = max(0.2, min(1.0, float(alpha)))
    ratio_template = 0.5 + (ratio_template - 0.5) * (1.0 / skew_scale)
    ratio_template = np.clip(ratio_template, 0.10, 0.90)

    pos_counts = []
    for n_i, r_i in zip(size_chunks, ratio_template):
        p_i = int(round(float(n_i) * float(r_i)))
        p_i = max(1, min(n_i - 1, p_i))
        pos_counts.append(p_i)

    # 调整总数使其与可用正类样本一致
    target_pos = int(len(idx_pos))
    cur_pos = int(sum(pos_counts))
    while cur_pos != target_pos:
        if cur_pos > target_pos:
            cands = [i for i in range(num_clients) if pos_counts[i] > 1]
            if not cands:
                break
            i = int(rng.choice(cands))
            pos_counts[i] -= 1
            cur_pos -= 1
        else:
            cands = [i for i in range(num_clients) if pos_counts[i] < size_chunks[i] - 1]
            if not cands:
                break
            i = int(rng.choice(cands))
            pos_counts[i] += 1
            cur_pos += 1

    neg_counts = [int(n_i - p_i) for n_i, p_i in zip(size_chunks, pos_counts)]

    client_indices = [[] for _ in range(num_clients)]
    p_cur = 0
    n_cur = 0
    for cid in range(num_clients):
        p_take = int(pos_counts[cid])
        n_take = int(neg_counts[cid])
        p_part = idx_pos[p_cur:p_cur + p_take]
        n_part = idx_neg[n_cur:n_cur + n_take]
        p_cur += p_take
        n_cur += n_take
        mix = np.concatenate([p_part, n_part]).astype(np.int64)
        rng.shuffle(mix)
        client_indices[cid] = mix.tolist()

    clients = []
    for cid in range(num_clients):
        cidx = np.array(client_indices[cid], dtype=np.int64)
        rng.shuffle(cidx)
        Xi, yi = X[cidx], y[cidx]
        labels, counts = np.unique(yi, return_counts=True)
        class_hist = {int(k): int(v) for k, v in zip(labels, counts)}
        clients.append({
            "client_id": int(cid),
            "X": Xi,
            "y": yi,
            "num_samples": int(len(yi)),
            "class_hist": class_hist,
            "visible_classes": sorted([int(c) for c in labels.tolist()]),
        })
    return clients


def run_single(
    dataset_name: str,
    distribution: str,   # iid / noniid
    use_kd: bool,
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    X_test_feat: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict:
    """
    单次运行（一个 seed）：
      1) 划分
      2) local training
      3) （可选）teacher + distill
      4) 构造 avg_model / best_model
      5) 直接评估这两个初始化模型 test accuracy
    """
    import feature_kd_init as kd

    kd.set_seed(seed)

    X_train_feat = np.asarray(X_train_feat, dtype=np.float32)
    X_test_feat = np.asarray(X_test_feat, dtype=np.float32)
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    feat_dim = int(X_train_feat.shape[1])
    num_classes = kd.infer_num_classes(y_train)
    device = torch.device(str(CFG["device"]).strip() or "cpu")

    # Step2: IID / Non-IID 划分
    if distribution == "iid":
        clients = make_iid_splits(
            X_train_feat=X_train_feat,
            y_train=y_train,
            num_clients=int(CFG["num_clients"]),
            seed=seed,
        )
    else:
        if num_classes == 2:
            clients = make_binary_label_skew_splits(
                X_train_feat=X_train_feat,
                y_train=y_train,
                num_clients=int(CFG["num_clients"]),
                seed=seed,
                alpha=0.3,
            )
        else:
            cpc = None if int(CFG["classes_per_client"]) <= 0 else int(CFG["classes_per_client"])
            clients = kd.make_class_skew_splits(
                X_train_feat=X_train_feat,
                y_train=y_train,
                num_clients=int(CFG["num_clients"]),
                classes_per_client=cpc,
                seed=seed,
            )

    # Step3: local pretraining
    local_models = []
    for c in clients:
        out = kd.train_local_model(
            c["X"], c["y"],
            feat_dim=feat_dim,
            num_classes=num_classes,
            device=device,
            epochs=int(CFG["local_epochs"]),
            lr=float(CFG["local_lr"]),
            seed=seed + int(c["client_id"]),
            max_epoch_log_lines=int(CFG["max_epoch_log_lines"]),
            verbose=False,
            use_dpsgd=bool(CFG["local_use_dpsgd"]),
            dpsgd_batch_size=int(CFG["local_dpsgd_batch_size"]),
            dpsgd_clipping_norm=float(CFG["local_dpsgd_clipping_norm"]),
            dpsgd_noise_multiplier=float(CFG["local_dpsgd_noise_multiplier"]),
        )
        local_models.append(out["model"])

    X_ref, y_ref = kd.build_reference_set(clients, ref_size=int(CFG["ref_size"]), seed=seed)
    if use_kd:
        # Step4: teacher ensemble + KD
        teacher_probs = kd.build_ensemble_teacher(
            local_models=local_models,
            X_ref=X_ref,
            num_classes=num_classes,
            device=device,
            tau=float(CFG["tau"]),
        )
        final_models = []
        for c, base_model in zip(clients, local_models):
            d_out = kd.distill_local_model(
                model=base_model,
                X_local=c["X"],
                y_local=c["y"],
                X_ref=X_ref,
                teacher_probs_ref=teacher_probs,
                num_classes=num_classes,
                device=device,
                tau=float(CFG["tau"]),
                lambda_kd=float(CFG["lambda_kd"]),
                distill_epochs=int(CFG["distill_epochs"]),
                lr=float(CFG["distill_lr"]),
                max_epoch_log_lines=int(CFG["max_epoch_log_lines"]),
                verbose=False,
            )
            final_models.append(d_out["model"])
    else:
        # without KD：直接使用 local pretraining 模型参与聚合
        final_models = local_models

    # Step5: avg / best 初始化模型
    best_model_plain, _, _ = kd.select_best_local_model(
        models=final_models,
        X_ref=X_ref,
        y_ref=y_ref,
        num_classes=num_classes,
        device=device,
    )

    if bool(CFG["use_secure_aggregation"]):
        import secure_aggregation as sa
        sa_result = sa.secure_aggregate_and_select(
            distill_models=final_models,
            best_local_model=best_model_plain,
            X_ref=X_ref,
            y_ref=y_ref,
            feat_dim=feat_dim,
            num_classes=num_classes,
            max_threads=int(CFG["secure_max_threads"]),
            verbose=False,
        )
        avg_model = sa_result["avg_model"].to(device)
        best_model = sa_result["best_model"].to(device)
    else:
        avg_model = kd.average_model_weights(
            models=final_models,
            feat_dim=feat_dim,
            num_classes=num_classes,
            device=device,
        )
        best_model = best_model_plain

    # Step6: 只评估初始化模型 accuracy（不是模块 C 最终模型）
    avg_test_acc = kd.evaluate_model(avg_model, X_test_feat, y_test, num_classes, device)
    best_test_acc = kd.evaluate_model(best_model, X_test_feat, y_test, num_classes, device)

    return {
        "dataset": dataset_name,
        "distribution": distribution,
        "use_kd": bool(use_kd),
        "seed": int(seed),
        "avg_test_acc": float(avg_test_acc),
        "best_test_acc": float(best_test_acc),
    }


def summarize(records: List[Dict]) -> Dict[Tuple[str, str, bool], Dict[str, float]]:
    out = {}
    keys = sorted(set((r["dataset"], r["distribution"], bool(r["use_kd"])) for r in records))
    for k in keys:
        rs = [r for r in records if (r["dataset"], r["distribution"], bool(r["use_kd"])) == k]
        avg_vals = np.asarray([r["avg_test_acc"] for r in rs], dtype=np.float64)
        best_vals = np.asarray([r["best_test_acc"] for r in rs], dtype=np.float64)
        out[k] = {
            "avg_mean": float(np.mean(avg_vals)),
            "avg_std": float(np.std(avg_vals)),
            "best_mean": float(np.mean(best_vals)),
            "best_std": float(np.std(best_vals)),
            "n": int(len(rs)),
        }
    return out


def _fmt_mean_std(mean: float, std: float) -> str:
    return "{:.4f} ± {:.4f}".format(float(mean), float(std))


def print_table(summary: Dict[Tuple[str, str, bool], Dict[str, float]]) -> None:
    print("\n" + "=" * 86)
    print("Table III 扩展：with KD vs without KD 的初始化模型 accuracy")
    print("Dataset | Dist | Avg(w/o KD) | Avg(w/ KD) | Best(w/o KD) | Best(w/ KD)")
    print("-" * 86)
    keys = sorted(set((k[0], k[1]) for k in summary.keys()))
    for dataset, dist in keys:
        s_no = summary.get((dataset, dist, False))
        s_yes = summary.get((dataset, dist, True))
        avg_no = _fmt_mean_std(s_no["avg_mean"], s_no["avg_std"]) if s_no else "N/A"
        avg_yes = _fmt_mean_std(s_yes["avg_mean"], s_yes["avg_std"]) if s_yes else "N/A"
        best_no = _fmt_mean_std(s_no["best_mean"], s_no["best_std"]) if s_no else "N/A"
        best_yes = _fmt_mean_std(s_yes["best_mean"], s_yes["best_std"]) if s_yes else "N/A"
        print(
            "{:<7} | {:<6} | {:<12} | {:<12} | {:<13} | {:<12}".format(
                dataset, dist.upper(), avg_no, avg_yes, best_no, best_yes
            )
        )
    print("=" * 86)


def save_csv(summary: Dict[Tuple[str, str, bool], Dict[str, float]], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Dataset",
            "Distribution",
            "Averaging Strategy (w/o KD)",
            "Averaging Strategy (w/ KD)",
            "Accuracy Strategy (w/o KD)",
            "Accuracy Strategy (w/ KD)",
        ])
        keys = sorted(set((k[0], k[1]) for k in summary.keys()))
        for dataset, dist in keys:
            s_no = summary.get((dataset, dist, False))
            s_yes = summary.get((dataset, dist, True))
            avg_no = _fmt_mean_std(s_no["avg_mean"], s_no["avg_std"]) if s_no else "N/A"
            avg_yes = _fmt_mean_std(s_yes["avg_mean"], s_yes["avg_std"]) if s_yes else "N/A"
            best_no = _fmt_mean_std(s_no["best_mean"], s_no["best_std"]) if s_no else "N/A"
            best_yes = _fmt_mean_std(s_yes["best_mean"], s_yes["best_std"]) if s_yes else "N/A"
            w.writerow([dataset, dist.upper(), avg_no, avg_yes, best_no, best_yes])


def save_plot(summary: Dict[Tuple[str, str, bool], Dict[str, float]], dataset: str, run_id: str) -> Path:
    """
    输出 Table III 扩展图（同数据集下 IID / Non-IID，with KD vs without KD）。
    """
    import matplotlib.pyplot as plt

    labels = ["IID", "Non-IID"]
    avg_no_means, avg_no_stds = [], []
    avg_yes_means, avg_yes_stds = [], []
    best_no_means, best_no_stds = [], []
    best_yes_means, best_yes_stds = [], []
    for dist in ["iid", "noniid"]:
        s_no = summary.get((dataset, dist, False))
        s_yes = summary.get((dataset, dist, True))
        avg_no_means.append(s_no["avg_mean"] if s_no else 0.0)
        avg_no_stds.append(s_no["avg_std"] if s_no else 0.0)
        best_no_means.append(s_no["best_mean"] if s_no else 0.0)
        best_no_stds.append(s_no["best_std"] if s_no else 0.0)
        avg_yes_means.append(s_yes["avg_mean"] if s_yes else 0.0)
        avg_yes_stds.append(s_yes["avg_std"] if s_yes else 0.0)
        best_yes_means.append(s_yes["best_mean"] if s_yes else 0.0)
        best_yes_stds.append(s_yes["best_std"] if s_yes else 0.0)

    x = np.arange(len(labels), dtype=np.float32)
    width = 0.2

    fig_path = TABLE3_FIG_DIR / f"table3_init_compare_{dataset}_{run_id}.png"
    plt.figure(figsize=(9, 5))
    plt.bar(
        x - 1.5 * width,
        avg_no_means,
        width,
        yerr=avg_no_stds,
        capsize=4,
        label="Averaging (w/o KD)",
    )
    plt.bar(
        x - 0.5 * width,
        avg_yes_means,
        width,
        yerr=avg_yes_stds,
        capsize=4,
        label="Averaging (w/ KD)",
    )
    plt.bar(
        x + 0.5 * width,
        best_no_means,
        width,
        yerr=best_no_stds,
        capsize=4,
        label="Accuracy (w/o KD)",
    )
    plt.bar(
        x + 1.5 * width,
        best_yes_means,
        width,
        yerr=best_yes_stds,
        capsize=4,
        label="Accuracy (w/ KD)",
    )
    plt.xticks(x, labels)
    plt.ylabel("Initial Global Model Accuracy")
    plt.title(f"Table III-style Init Comparison with/without KD ({dataset})")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(fig_path), dpi=160)
    plt.close()
    return fig_path


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PEA Table III 风格初始化聚合对比实验")
    # 入口规则：
    # 1) 不传 --dataset：默认跑全部数据集（mnist/cifar/imdb）
    # 2) 传 --dataset：只跑该单个数据集
    p.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "mnist", "cifar", "imdb"],
        help="默认 all；可指定单个数据集",
    )
    return p


def main() -> int:
    ensure_dirs()
    args = make_parser().parse_args()
    if bool(CFG["safe_mode"]):
        set_safe_runtime(int(CFG["cpu_threads"]))

    import feature_kd_bridge as bridge

    run_id = time.strftime("%Y%m%d_%H%M%S")
    txt_log = LOG_TXT_DIR / f"table3InitCompare_{args.dataset}_{run_id}.log"
    json_log = LOG_JSON_DIR / f"table3InitCompare_{args.dataset}_{run_id}.json"
    csv_path = EXPTAB_DIR / f"table3_init_compare_{args.dataset}_{run_id}.csv"

    datasets = ["mnist", "cifar", "imdb"] if args.dataset == "all" else [args.dataset]
    dists = ["iid", "noniid"]

    t0 = time.time()
    with tee_console_to_file(txt_log):
        print("=" * 86)
        print("PEA Table III 风格实验：Initialization Aggregation Strategy Comparison")
        print("=" * 86)
        print("run_id             :", run_id)
        print("datasets           :", datasets)
        print("distribution       :", dists)
        print("num_repeats        :", CFG["num_repeats"])
        print("use_secure_agg     :", CFG["use_secure_aggregation"])
        print("kd_compare         : [without_kd, with_kd]")
        print("cpu_threads        :", CFG["cpu_threads"])

        records = []
        for ds in datasets:
            print("\n" + "-" * 86)
            print(f"[Dataset] {ds}")
            print("-" * 86)
            data = bridge.load_dataset_features(ds, max_test_cap=int(CFG["max_test_cap"]))
            xtr = data["X_train_feat"]
            xte = data["X_test_feat"]
            ytr = data["y_train"]
            yte = data["y_test"]
            print("X_train_feat       :", xtr.shape)
            print("X_test_feat        :", xte.shape)
            print("source_feat        :", data.get("source_feat"))

            for dist in dists:
                print(f"\n  [Distribution] {dist.upper()}")
                for i in range(int(CFG["num_repeats"])):
                    seed = int(CFG["seed_base"]) + i
                    for use_kd in [False, True]:
                        r = run_single(
                            dataset_name=ds,
                            distribution=dist,
                            use_kd=use_kd,
                            X_train_feat=xtr,
                            y_train=ytr,
                            X_test_feat=xte,
                            y_test=yte,
                            seed=seed,
                        )
                        records.append(r)
                        print(
                            "    seed={:<4d} {:<10} avg_test_acc={:.4f} best_test_acc={:.4f}".format(
                                seed,
                                "with_kd" if use_kd else "without_kd",
                                r["avg_test_acc"],
                                r["best_test_acc"],
                            )
                        )

        summary = summarize(records)
        print_table(summary)
        save_csv(summary, csv_path)
        fig_paths = []
        for ds in datasets:
            fig_paths.append(str(save_plot(summary, ds, run_id)))

        payload = {
            "run_id": run_id,
            "datasets": datasets,
            "distribution": dists,
            "cfg": CFG,
            "records": records,
            "summary": {
                f"{k[0]}::{k[1]}::{'with_kd' if k[2] else 'without_kd'}": v
                for k, v in summary.items()
            },
            "csv_path": str(csv_path),
            "fig_paths": fig_paths,
            "txt_log": str(txt_log),
            "elapsed_s": round(time.time() - t0, 3),
        }
        with json_log.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print("CSV已写入 :", csv_path)
        print("图已写入  :")
        for p in fig_paths:
            print("  -", p)
        print("TXT日志   :", txt_log)
        print("JSON日志  :", json_log)
        print("总耗时(s) :", payload["elapsed_s"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

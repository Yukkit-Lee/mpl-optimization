#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：实验对比模块 b 中「本地 SGD」与「本地 DPSGD」训练出的 local model，
再经过同样 KD 蒸馏后，在精度上的差异。

说明：
1) 输入直接使用模块 a 的特征缓存（默认先做 MNIST）。
2) 两组实验只改变 local 训练方式（SGD vs DPSGD），其余参数保持一致。
3) 输出两份日志：
   - 人类可读：log/txt/localModelDpCompare_{dataset}_{run_id}.log
   - 机器可读：log/json/localModelDpCompare_{dataset}_{run_id}.json
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_TXT_DIR = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"


# 实验配置（如需调参，优先改这里）
CFG = {
    "safe_mode": True,
    "cpu_threads": 1,
    "seed": 42,
    "max_test_cap": 1000,
    "max_epoch_log_lines": 10,
    "device": "",  # 留空=按 feature_kd_init 默认设备
    # 模块 b 共同参数
    "num_clients": 5,
    "classes_per_client": 0,  # 0=自动
    "ref_size": 200,
    "local_epochs": 80,
    "local_lr": 1e-2,
    "tau": 2.0,
    "lambda_kd": 1.0,
    "distill_epochs": 30,
    "distill_lr": 5e-3,
    "use_secure_aggregation": True,
    # DPSGD 参数（只用于 dpsgd 组）
    "dpsgd_batch_size": 64,
    "dpsgd_clipping_norm": 1.0,
    "dpsgd_noise_multiplier": 1.0,
}


def ensure_dirs() -> None:
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)


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


def set_safe_runtime(max_threads: int) -> None:
    t = max(1, int(max_threads))
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["MKL_NUM_THREADS"] = str(t)
    os.environ["OPENBLAS_NUM_THREADS"] = str(t)
    os.environ["NUMEXPR_NUM_THREADS"] = str(t)
    try:
        import torch
        torch.set_num_threads(t)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(t)
    except Exception:
        pass


def _mean_after_acc(distill_logs):
    vals = [float(x.get("after_acc", 0.0)) for x in distill_logs]
    return float(np.mean(vals)) if vals else 0.0


def _std_after_acc(distill_logs):
    vals = [float(x.get("after_acc", 0.0)) for x in distill_logs]
    return float(np.std(vals)) if vals else 0.0


def run_one(
    dataset: str,
    local_use_dpsgd: bool,
    data: Dict,
):
    import feature_kd_init as kd

    cpc = None if int(CFG["classes_per_client"]) <= 0 else int(CFG["classes_per_client"])
    out = kd.run_kd_hetero_init_pipeline(
        X_train_feat=data["X_train_feat"],
        y_train=data["y_train"],
        X_test_feat=data["X_test_feat"],
        y_test=data["y_test"],
        num_clients=int(CFG["num_clients"]),
        classes_per_client=cpc,
        ref_size=int(CFG["ref_size"]),
        local_epochs=int(CFG["local_epochs"]),
        local_lr=float(CFG["local_lr"]),
        tau=float(CFG["tau"]),
        lambda_kd=float(CFG["lambda_kd"]),
        distill_epochs=int(CFG["distill_epochs"]),
        distill_lr=float(CFG["distill_lr"]),
        seed=int(CFG["seed"]),
        max_epoch_log_lines=int(CFG["max_epoch_log_lines"]),
        run_figure5=False,
        global_epochs=80,
        global_lr=1e-2,
        device=(str(CFG["device"]).strip() if str(CFG["device"]).strip() else None),
        dataset_name=str(dataset),
        # 对比实验只看结果，不写入 init_model，避免覆盖主流程缓存。
        save_init_model=False,
        use_secure_aggregation=bool(CFG["use_secure_aggregation"]),
        secure_max_threads=int(CFG["cpu_threads"]),
        local_use_dpsgd=bool(local_use_dpsgd),
        local_dpsgd_batch_size=int(CFG["dpsgd_batch_size"]),
        local_dpsgd_clipping_norm=float(CFG["dpsgd_clipping_norm"]),
        local_dpsgd_noise_multiplier=float(CFG["dpsgd_noise_multiplier"]),
        verbose=True,
    )
    return out


def make_parser():
    p = argparse.ArgumentParser(description="本地 SGD vs DPSGD（蒸馏后精度）对比实验")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "imdb", "cifar"])
    return p


def main() -> int:
    ensure_dirs()
    args = make_parser().parse_args()

    if bool(CFG["safe_mode"]):
        set_safe_runtime(int(CFG["cpu_threads"]))

    import feature_kd_bridge as bridge

    run_id = time.strftime("%Y%m%d_%H%M%S")
    txt_log = LOG_TXT_DIR / f"localModelDpCompare_{args.dataset}_{run_id}.log"
    json_log = LOG_JSON_DIR / f"localModelDpCompare_{args.dataset}_{run_id}.json"

    t0 = time.time()
    with tee_console_to_file(txt_log):
        print("=" * 72)
        print("实验：本地 SGD vs 本地 DPSGD（蒸馏后 local model 精度对比）")
        print("=" * 72)
        print("run_id       :", run_id)
        print("dataset      :", args.dataset)
        print("safe_mode    :", CFG["safe_mode"])
        print("cpu_threads  :", CFG["cpu_threads"])

        data = bridge.load_dataset_features(args.dataset, max_test_cap=int(CFG["max_test_cap"]))
        print("X_train_feat :", data["X_train_feat"].shape)
        print("X_test_feat  :", data["X_test_feat"].shape)

        print("\n[1/2] 运行基线：local SGD -> KD")
        out_sgd = run_one(args.dataset, local_use_dpsgd=False, data=data)

        print("\n[2/2] 运行对照：local DPSGD -> KD")
        out_dpsgd = run_one(args.dataset, local_use_dpsgd=True, data=data)

        sgd_mean = _mean_after_acc(out_sgd["distill_logs"])
        dpsgd_mean = _mean_after_acc(out_dpsgd["distill_logs"])
        sgd_std = _std_after_acc(out_sgd["distill_logs"])
        dpsgd_std = _std_after_acc(out_dpsgd["distill_logs"])

        sgd_chosen = float(out_sgd["candidate"]["chosen_test_acc"])
        dpsgd_chosen = float(out_dpsgd["candidate"]["chosen_test_acc"])

        print("\n" + "=" * 72)
        print("  [结果汇总] 蒸馏后精度对比")
        print("  mode         mean_after_acc   std_after_acc    chosen_test_acc")
        print("  ---------------------------------------------------------------")
        print("  SGD          {:<16.4f} {:<15.4f} {:.4f}".format(sgd_mean, sgd_std, sgd_chosen))
        print("  DPSGD        {:<16.4f} {:<15.4f} {:.4f}".format(dpsgd_mean, dpsgd_std, dpsgd_chosen))
        print("  ---------------------------------------------------------------")
        print("  Δmean_after_acc  : {:+.4f} (DPSGD - SGD)".format(dpsgd_mean - sgd_mean))
        print("  Δchosen_test_acc : {:+.4f} (DPSGD - SGD)".format(dpsgd_chosen - sgd_chosen))
        print("=" * 72)

        summary = {
            "run_id": run_id,
            "dataset": args.dataset,
            "safe_mode": bool(CFG["safe_mode"]),
            "cpu_threads": int(CFG["cpu_threads"]),
            "cfg": CFG,
            "data_shapes": {
                "X_train_feat": tuple(data["X_train_feat"].shape),
                "X_test_feat": tuple(data["X_test_feat"].shape),
                "y_train": tuple(np.asarray(data["y_train"]).shape),
                "y_test": tuple(np.asarray(data["y_test"]).shape),
            },
            "sgd": {
                "mean_after_acc": sgd_mean,
                "std_after_acc": sgd_std,
                "chosen_test_acc": sgd_chosen,
                "candidate": out_sgd["candidate"],
            },
            "dpsgd": {
                "mean_after_acc": dpsgd_mean,
                "std_after_acc": dpsgd_std,
                "chosen_test_acc": dpsgd_chosen,
                "candidate": out_dpsgd["candidate"],
            },
            "delta": {
                "mean_after_acc": dpsgd_mean - sgd_mean,
                "chosen_test_acc": dpsgd_chosen - sgd_chosen,
            },
            "elapsed_s": round(time.time() - t0, 3),
            "txt_log": str(txt_log),
        }

        with json_log.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print("TXT日志已写入:", txt_log)
        print("JSON日志已写入:", json_log)
        print("总耗时(s):", summary["elapsed_s"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

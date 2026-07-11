#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：
独立测试“模块C中可迁移到GPU的流程”是否可运行，不修改现有实验代码。

设计目标：
1) 自动探测当前机器哪些 conda 环境支持 4090 的 CUDA 计算；
2) 选一个可用环境（优先非 tfe），在该环境执行模块C最小训练验证；
3) 输出清晰的 txt/json 报告，告诉你“能否迁移到GPU”。

说明：
- 该脚本只测试 module_c_trust_dpsgd（PyTorch段），不改 module_c_bridge.py。
- TFE/TF1.15 相关部分不在本脚本内迁移。
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_TXT_DIR = LOG_DIR / "txt"
LOG_JSON_DIR = LOG_DIR / "json"
FEATURE_DIR = BASE_DIR / "featureData"

# ===== 配置（尽量少参数，避免复杂命令行）=====
DATASET = "cifar"          # 可选: mnist / imdb / cifar
TEST_EPOCHS = 20            # 最小可运行验证轮数
CPU_THREADS = 1             # 安全模式：线程数限制
SEED = 42

# 候选环境优先级（可按需调整）
ENV_PRIORITY = ["distillation", "lixujieEnv", "nssmpc", "privacy_meter", "tfe"]


def ensure_dirs():
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, timeout=120):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr


def find_python_of_env(env_name: str):
    p = Path(f"/home/lixujie/.conda/envs/{env_name}/bin/python")
    return str(p) if p.exists() else None


def probe_env_cuda(py_exec: str):
    """探测某个解释器是否真的能在 GPU 上完成 matmul。"""
    code = r'''
import json
out = {"ok": False}
try:
    import torch
    out["torch_version"] = torch.__version__
    out["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        out["device_name"] = torch.cuda.get_device_name(0)
        x = torch.randn(64, 64, device="cuda")
        y = torch.randn(64, 64, device="cuda")
        z = x @ y
        out["ok"] = True
        out["sample"] = float(z[0,0].item())
except Exception as e:
    out["error"] = repr(e)
print(json.dumps(out, ensure_ascii=False))
'''
    rc, out, err = run_cmd([py_exec, "-c", code], timeout=60)
    if rc != 0:
        return {"ok": False, "error": f"rc={rc}", "stdout": out[-500:], "stderr": err[-500:]}
    try:
        obj = json.loads(out.strip().splitlines()[-1])
    except Exception:
        obj = {"ok": False, "error": "json_parse_fail", "stdout": out[-500:], "stderr": err[-500:]}
    return obj


def find_feature_file(dataset: str):
    pats = {
        "mnist": "mnist_hog_tr*_te*_seed*.npz",
        "imdb": "imdb_RoBERTa_tr*_te*_seed*.npz",
        "cifar": "cifar10_dino_tr*_te*_seed*.npz",
    }
    pt = pats.get(dataset, pats["cifar"])
    files = sorted(FEATURE_DIR.glob(pt))
    return str(files[-1]) if files else None


def run_worker(py_exec: str, dataset: str, epochs: int, cpu_threads: int, seed: int):
    """在目标环境执行模块C最小GPU训练。"""
    feature_file = find_feature_file(dataset)
    code = r'''
import json
import numpy as np
import torch
import importlib.util
from pathlib import Path

dataset = __DATASET__
epochs = __EPOCHS__
cpu_threads = __CPU_THREADS__
seed = __SEED__
feature_file = __FEATURE_FILE__

# 导入本仓库的 module_c_trust_dpsgd.py（不依赖 site-packages 安装）
mod_path = Path(__MODULE_PATH__)
spec = importlib.util.spec_from_file_location("module_c_trust_dpsgd", str(mod_path))
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

# 安全模式（低线程）
mc.set_safe_runtime(cpu_threads)

device = mc.select_device("cuda")

if feature_file is not None:
    d = np.load(feature_file)
    xtr = d["x_train_feat"].astype(np.float32)
    xte = d["x_test_feat"].astype(np.float32)
    ytr = d["y_train"].astype(int)
    yte = d["y_test"].astype(int)
else:
    # 无缓存就造最小数据
    rng = np.random.RandomState(seed)
    xtr = rng.randn(1000, 128).astype(np.float32)
    ytr = rng.randint(0, 10, size=(1000,), dtype=np.int32)
    xte = rng.randn(200, 128).astype(np.float32)
    yte = rng.randint(0, 10, size=(200,), dtype=np.int32)

num_classes = int(len(np.unique(ytr)))
feat_dim = int(xtr.shape[1])
init_model = mc.make_fresh_model(feat_dim, num_classes, device)

out = mc.run_trust_dpsgd(
    init_model=init_model,
    X_lg_feat=xtr,
    y_lg=ytr,
    X_test_feat=xte,
    y_test=yte,
    global_epochs=int(epochs),
    global_lr=0.01,
    batch_size=64,
    clipping_norm=1.0,
    mode="plain",
    delta=1e-5,
    target_epsilon=None,
    safe_mode=True,
    cpu_threads=int(cpu_threads),
    seed=int(seed),
    device_hint="cuda",
    max_epoch_log_lines=5,
    verbose=False,
)

fm = out["final_metrics"]
res = {
    "device": str(out["config"]["device"]),
    "feat_dim": feat_dim,
    "num_classes": num_classes,
    "epochs_ran": int(fm.get("epochs_ran", 0)),
    "final_train_acc": float(fm.get("final_train_accuracy", 0.0)),
    "final_test_acc": float(fm.get("final_test_accuracy", 0.0)),
}
print(json.dumps(res, ensure_ascii=False))
'''
    code = (
        code.replace("__DATASET__", repr(dataset))
            .replace("__EPOCHS__", repr(int(epochs)))
            .replace("__CPU_THREADS__", repr(int(cpu_threads)))
            .replace("__SEED__", repr(int(seed)))
            .replace("__FEATURE_FILE__", repr(feature_file))
            .replace("__MODULE_PATH__", repr(str((BASE_DIR / "module_c_trust_dpsgd.py").resolve())))
    )

    rc, out, err = run_cmd([py_exec, "-c", code], timeout=1800)
    return rc, out, err


def main():
    ensure_dirs()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = LOG_TXT_DIR / f"gpuMigrationTest_{DATASET}_{run_id}.log"
    json_path = LOG_JSON_DIR / f"gpuMigrationTest_{DATASET}_{run_id}.json"

    logs = []
    def log(s=""):
        print(s)
        logs.append(s)

    log("=" * 72)
    log("GPU迁移测试：模块C（PyTorch段）")
    log("=" * 72)
    log(f"run_id      : {run_id}")
    log(f"dataset     : {DATASET}")
    log(f"test_epochs : {TEST_EPOCHS}")
    log(f"cpu_threads : {CPU_THREADS}")

    probes = []
    for env in ENV_PRIORITY:
        py = find_python_of_env(env)
        if not py:
            continue
        info = probe_env_cuda(py)
        info.update({"env": env, "python": py})
        probes.append(info)
        log(f"  probe {env:<12} -> ok={info.get('ok')} torch={info.get('torch_version', 'NA')} device={info.get('device_name', 'NA')}")

    ok_envs = [p for p in probes if p.get("ok")]
    chosen = ok_envs[0] if ok_envs else None

    worker = None
    if chosen:
        log("")
        log(f"选择环境: {chosen['env']} ({chosen['python']})")
        rc, out, err = run_worker(chosen["python"], DATASET, TEST_EPOCHS, CPU_THREADS, SEED)
        worker = {"rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}
        if rc == 0:
            try:
                line = out.strip().splitlines()[-1]
                worker["result"] = json.loads(line)
                log("GPU最小训练验证: SUCCESS")
                log(f"  device         : {worker['result'].get('device')}")
                log(f"  epochs_ran     : {worker['result'].get('epochs_ran')}")
                log(f"  final_test_acc : {worker['result'].get('final_test_acc'):.4f}")
            except Exception:
                log("GPU最小训练验证: FAIL（输出解析失败）")
        else:
            log("GPU最小训练验证: FAIL（子进程返回非0）")
    else:
        log("\n未找到可用GPU环境（所有探测失败）。")

    with txt_path.open("w", encoding="utf-8") as f:
        for x in logs:
            f.write(x + "\n")

    report = {
        "run_id": run_id,
        "dataset": DATASET,
        "test_epochs": TEST_EPOCHS,
        "cpu_threads": CPU_THREADS,
        "probes": probes,
        "chosen_env": chosen,
        "worker": worker,
        "txt_log_file": str(txt_path),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\nTXT日志:", txt_path)
    print("JSON报告:", json_path)


if __name__ == "__main__":
    main()

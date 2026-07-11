#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用途：
模块 C 正式 GPU 运行器（不改现有代码），实现“一条命令跑 GPU 主训练”，
并可选自动切换到 tfe 环境执行 Figure5-TFE。

核心能力：
1) 自动切换到 GPU 兼容环境（默认 distillation）；
2) 在 GPU 上执行模块 C 主训练（PyTorch 路径）；
3) 可选在 tfe 环境执行 Figure5-TFE（tf-encrypted 路径）；
4) 输出 txt/json 日志到 log/txt 和 log/json。

说明：
- 模块 C 主训练：run_trust_dpsgd（PyTorch，当前原型实现，支持 GPU）
- Figure5-TFE：run_tfe_figure5_init_comparison（TF-Encrypted，通常走 tfe 环境/CPU）
"""

import argparse
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
DATA_DIR = BASE_DIR / "data"

# 默认解释器路径（可通过命令行覆盖）
DEFAULT_GPU_PY = "/home/lixujie/.conda/envs/distillation/bin/python"
DEFAULT_TFE_PY = "/home/lixujie/.conda/envs/tfe/bin/python"


def ensure_dirs():
    LOG_TXT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_JSON_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _run_cmd(cmd, timeout=None):
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr


def _tee_write(fp, text):
    print(text)
    fp.write(text + "\n")
    fp.flush()


def _probe_cuda(py_exec):
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
        out["sample"] = float(z[0, 0].item())
except Exception as e:
    out["error"] = repr(e)
print(json.dumps(out, ensure_ascii=False))
'''
    rc, out, err = _run_cmd([py_exec, "-c", code], timeout=60)
    if rc != 0:
        return {"ok": False, "error": f"rc={rc}", "stderr": err[-500:]}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "error": "json_parse_fail", "stdout": out[-500:], "stderr": err[-500:]}


def _reexec_if_needed(args):
    """
    若当前解释器不是 GPU 解释器，则自动重启到 GPU 解释器。
    这样用户在任意环境里一条命令即可。
    """
    current = os.path.realpath(sys.executable)
    target = os.path.realpath(args.gpu_python)
    if current == target:
        return None

    if not Path(target).exists():
        raise RuntimeError(f"GPU解释器不存在: {target}")

    new_argv = [target, str(Path(__file__).resolve())] + sys.argv[1:] + ["--_reexeced", "1"]
    p = subprocess.run(new_argv)
    return int(p.returncode)


def _run_figure5_tfe_subprocess(
    tfe_python,
    dataset,
    run_id,
    npz_payload,
    figure5_epochs,
    figure5_lr,
    max_log_lines,
    figure5_delta,
    figure5_target_epsilon,
    figure5_annotate_iters,
    figure5_epsilon_mode,
    figure5_epsilon_end,
):
    """在 tfe 环境中仅执行 Figure5-TFE，不重复跑模块 C 主训练。"""
    inline = r'''
import json
import numpy as np
import importlib.util
from pathlib import Path

payload = Path(__PAYLOAD__)
d = np.load(str(payload))
xtr = d["x_train_feat"].astype(np.float32)
xte = d["x_test_feat"].astype(np.float32)
ytr = d["y_train"].astype(int)
yte = d["y_test"].astype(int)

mod_path = Path(__BRIDGE_PATH__)
spec = importlib.util.spec_from_file_location("module_c_bridge", str(mod_path))
cb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cb)

feat_dim = int(xtr.shape[1])
num_classes = int(len(np.unique(ytr.reshape(-1))))
b_out = cb._load_module_b_init_or_raise(__DATASET__, feat_dim, num_classes)

fig5 = cb.run_tfe_figure5_init_comparison(
    x_train=xtr,
    y_train=ytr,
    x_test=xte,
    y_test=yte,
    b_out=b_out,
    epochs=int(__F5_EPOCHS__),
    lr=float(__F5_LR__),
    max_epoch_log_lines=int(__F5_LOG_LINES__),
    delta=float(__F5_DELTA__),
    target_epsilon=__F5_TARGET_EPS__,
    annotate_iters=__F5_ANN_ITERS__,
    epsilon_mode=__F5_EPS_MODE__,
    epsilon_end=float(__F5_EPS_END__),
    log_fn=print,
)
files = cb.save_tfe_figure5_outputs(fig5, __DATASET__, __RUN_ID__)
print("__FIG5_FILES__=" + json.dumps(files, ensure_ascii=False))
'''
    inline = (
        inline.replace("__PAYLOAD__", repr(str(npz_payload)))
              .replace("__BRIDGE_PATH__", repr(str((BASE_DIR / "module_c_bridge.py").resolve())))
              .replace("__DATASET__", repr(dataset))
              .replace("__RUN_ID__", repr(run_id))
              .replace("__F5_EPOCHS__", repr(int(figure5_epochs)))
              .replace("__F5_LR__", repr(float(figure5_lr)))
              .replace("__F5_LOG_LINES__", repr(int(max_log_lines)))
              .replace("__F5_DELTA__", repr(float(figure5_delta)))
              .replace("__F5_TARGET_EPS__", repr(figure5_target_epsilon))
              .replace("__F5_ANN_ITERS__", repr(figure5_annotate_iters))
              .replace("__F5_EPS_MODE__", repr(str(figure5_epsilon_mode)))
              .replace("__F5_EPS_END__", repr(float(figure5_epsilon_end)))
    )

    rc, out, err = _run_cmd([tfe_python, "-c", inline], timeout=7200)
    files = None
    for line in out.splitlines():
        if line.startswith("__FIG5_FILES__="):
            try:
                files = json.loads(line.split("=", 1)[1])
            except Exception:
                files = None
    return {
        "rc": rc,
        "stdout_tail": out[-3000:],
        "stderr_tail": err[-3000:],
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description="模块C GPU正式运行器")
    parser.add_argument("--dataset", type=str, default="cifar", choices=["mnist", "imdb", "cifar"])
    parser.add_argument("--gpu_python", type=str, default=DEFAULT_GPU_PY)
    parser.add_argument("--tfe_python", type=str, default=DEFAULT_TFE_PY)

    parser.add_argument("--mode", type=str, default="trust_dp", choices=["plain", "fixed_dp", "trust_dp"])
    parser.add_argument("--trust_t", type=float, default=0.7)
    parser.add_argument("--global_epochs", type=int, default=800)
    parser.add_argument("--global_lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sigma_min", type=float, default=0.1)
    parser.add_argument("--sigma_max", type=float, default=2.0)
    parser.add_argument("--noise_multiplier", type=float, default=1.0)
    parser.add_argument("--clipping_norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--target_epsilon", type=float, default=None)

    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--max_epoch_log_lines", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_test_cap", type=int, default=1000)

    parser.add_argument("--run_figure5_tfe", action="store_true")
    parser.add_argument("--figure5_epochs", type=int, default=800)
    parser.add_argument("--figure5_lr", type=float, default=0.01)
    parser.add_argument("--figure5_delta", type=float, default=1e-5)
    parser.add_argument("--figure5_target_epsilon", type=float, default=None)
    parser.add_argument("--figure5_annotate_iters", type=str, default="")  # 例如: 0,400,800
    parser.add_argument("--figure5_epsilon_mode", type=str, default="pea_aligned", choices=["pea_aligned", "accountant"])
    parser.add_argument("--figure5_epsilon_end", type=float, default=2.07)

    parser.add_argument("--_reexeced", type=str, default="0")
    args = parser.parse_args()

    ensure_dirs()

    if args._reexeced != "1":
        code = _reexec_if_needed(args)
        if code is not None:
            return code

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_log = LOG_TXT_DIR / f"trustDpsgdGpu_{args.dataset}_{run_id}.log"
    json_log = LOG_JSON_DIR / f"trustDpsgdGpu_{args.dataset}_{run_id}.json"

    with txt_log.open("w", encoding="utf-8") as fp:
        _tee_write(fp, "=" * 72)
        _tee_write(fp, "模块C GPU正式运行器")
        _tee_write(fp, "=" * 72)
        _tee_write(fp, f"run_id        : {run_id}")
        _tee_write(fp, f"python        : {sys.executable}")
        _tee_write(fp, f"dataset       : {args.dataset}")
        _tee_write(fp, f"mode          : {args.mode}")
        _tee_write(fp, f"global_epochs : {args.global_epochs}")
        _tee_write(fp, f"cpu_threads   : {args.cpu_threads}")

        probe = _probe_cuda(sys.executable)
        _tee_write(fp, f"cuda_probe    : {probe}")
        if not probe.get("ok", False):
            _tee_write(fp, "当前解释器无法稳定执行GPU算子，请更换 --gpu_python。")
            report = {
                "run_id": run_id,
                "status": "failed",
                "reason": "cuda_probe_fail",
                "probe": probe,
                "txt_log": str(txt_log),
            }
            with json_log.open("w", encoding="utf-8") as jf:
                json.dump(report, jf, indent=2, ensure_ascii=False)
                jf.write("\n")
            return 2

        import module_c_bridge as cb
        import module_c_trust_dpsgd as mc

        cb.set_safe_runtime(args.cpu_threads)
        _tee_write(fp, "安全模式线程限制已应用。")

        data = cb.load_dataset_features(args.dataset, max_test_cap=args.max_test_cap)
        xtr = data["X_train_feat"]
        xte = data["X_test_feat"]
        ytr = data["y_train"]
        yte = data["y_test"]
        feat_dim = int(xtr.shape[1])
        num_classes = int(len(set(ytr.reshape(-1).tolist())))
        _tee_write(fp, f"X_train_feat   : {xtr.shape}")
        _tee_write(fp, f"X_test_feat    : {xte.shape}")

        b_out = cb._load_module_b_init_or_raise(args.dataset, feat_dim, num_classes)
        _tee_write(fp, f"module_b_cache : {b_out.get('_cache_meta', {})}")

        t0 = time.time()
        c_results = cb.run_b_to_c(
            module_b_output=b_out,
            X_train_feat=xtr,
            y_train=ytr,
            X_test_feat=xte,
            y_test=yte,
            init_type="chosen",
            mode=args.mode,
            trust_t=args.trust_t,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            noise_multiplier=args.noise_multiplier,
            global_epochs=args.global_epochs,
            global_lr=args.global_lr,
            batch_size=args.batch_size,
            clipping_norm=args.clipping_norm,
            delta=args.delta,
            target_epsilon=args.target_epsilon,
            seed=args.seed,
            device_hint="cuda",
            run_comparison=False,
            max_epoch_log_lines=args.max_epoch_log_lines,
            verbose=True,
        )
        elapsed = round(time.time() - t0, 3)

        main_json_path = mc.save_report(
            c_results["module_c_main"],
            dataset_name=args.dataset,
            mode=args.mode,
            json_path=json_log,
        )
        _tee_write(fp, f"GPU主训练完成，main_json: {main_json_path}")

        fig5_info = None
        if args.run_figure5_tfe:
            _tee_write(fp, "开始在 tfe 环境执行 Figure5-TFE ...")
            payload = DATA_DIR / f"gpu_runner_payload_{args.dataset}_{run_id}.npz"
            np = __import__("numpy")
            np.savez(
                str(payload),
                x_train_feat=xtr.astype(np.float32),
                x_test_feat=xte.astype(np.float32),
                y_train=ytr.astype(np.int32),
                y_test=yte.astype(np.int32),
            )

            ann = None
            s = str(args.figure5_annotate_iters).strip()
            if s:
                ann = [int(x.strip()) for x in s.split(",") if x.strip()]

            fig5_info = _run_figure5_tfe_subprocess(
                tfe_python=args.tfe_python,
                dataset=args.dataset,
                run_id=run_id,
                npz_payload=payload,
                figure5_epochs=args.figure5_epochs,
                figure5_lr=args.figure5_lr,
                max_log_lines=args.max_epoch_log_lines,
                figure5_delta=args.figure5_delta,
                figure5_target_epsilon=args.figure5_target_epsilon,
                figure5_annotate_iters=ann,
                figure5_epsilon_mode=args.figure5_epsilon_mode,
                figure5_epsilon_end=args.figure5_epsilon_end,
            )
            _tee_write(fp, f"Figure5-TFE 子进程结果: rc={fig5_info.get('rc')} files={fig5_info.get('files')}")

        summary = {
            "run_id": run_id,
            "status": "ok",
            "python": sys.executable,
            "dataset": args.dataset,
            "mode": args.mode,
            "gpu_probe": probe,
            "elapsed_s": elapsed,
            "main_json": str(main_json_path),
            "txt_log": str(txt_log),
            "module_b_cache": b_out.get("_cache_meta", {}),
            "module_b_candidate": b_out.get("candidate", {}),
            "figure5_tfe": fig5_info,
        }
        with json_log.open("w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2, ensure_ascii=False)
            jf.write("\n")

        _tee_write(fp, f"TXT日志: {txt_log}")
        _tee_write(fp, f"JSON报告: {json_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

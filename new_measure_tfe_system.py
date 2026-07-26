#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one TFE command while measuring process-tree peak RSS."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def child_pids(pid):
    path = Path("/proc/{}/task/{}/children".format(pid, pid))
    try:
        return [int(value) for value in path.read_text().split()]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []


def process_tree(pid):
    seen = set()
    pending = [int(pid)]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(child_pids(current))
    return seen


def rss_kib(pid):
    try:
        for line in Path("/proc/{}/status".format(pid)).read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new_tmp_{}".format(os.getpid()))
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_report", required=True)
    parser.add_argument("--inner_report", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--precision_fractional", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("measured command is required after --")
    started = time.time()
    process = subprocess.Popen(command)
    peak_rss_kib = 0
    samples = 0
    while process.poll() is None:
        peak_rss_kib = max(
            peak_rss_kib,
            sum(rss_kib(pid) for pid in process_tree(process.pid)),
        )
        samples += 1
        time.sleep(0.1)
    return_code = process.wait()
    peak_rss_kib = max(
        peak_rss_kib,
        sum(rss_kib(pid) for pid in process_tree(process.pid)),
    )
    wall_seconds = time.time() - started
    payload = {
        "schema": "new_tfe_system_run_v1",
        "status": "COMPLETE" if return_code == 0 else "FAILED",
        "dataset": args.dataset,
        "scale": args.scale,
        "num_samples": args.num_samples,
        "feature_dim": args.feature_dim,
        "num_classes": args.num_classes,
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "model_parameters": (
            args.feature_dim * args.num_classes + args.num_classes
        ),
        "fixed_point_precision_fractional": args.precision_fractional,
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": int(peak_rss_kib * 1024),
        "memory_samples": samples,
        "return_code": return_code,
        "backend": "tfe_mpc",
        "protocol": "ABY3",
        "communication_bytes": "N/A",
        "communication_rounds": "N/A",
        "communication_limitation": (
            "single-machine TF-Encrypted 0.8.0 does not expose byte/round counters"
        ),
        "test_loaded": False,
        "dp_mtt_status": "EXCLUDED_BY_USER",
        "inner_report": args.inner_report,
        "command": command,
    }
    inner_path = Path(args.inner_report)
    if return_code == 0:
        if not inner_path.is_file():
            raise FileNotFoundError("missing inner report {}".format(inner_path))
        inner = json.loads(inner_path.read_text(encoding="utf-8"))
        payload.update(
            {
                "secure_runtime_seconds": inner["runtime_seconds"],
                "revealed_values": inner["security"]["revealed"],
                "privacy": inner["privacy"],
                "released_model_record_dp_valid": inner["privacy"][
                    "released_model_record_dp_valid"
                ],
            }
        )
    atomic_json(payload, args.system_report)
    print(
        "tfe_system dataset={} scale={} status={} wall={:.3f}s peak_rss={}".format(
            args.dataset,
            args.scale,
            payload["status"],
            wall_seconds,
            payload["peak_rss_bytes"],
        )
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
小实验台总控脚本：顺序运行 Plain logistic 与 TFE logistic，并输出对照表。
"""

import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"

PLAIN_SCRIPT = BASE_DIR / "train_plain_tabular_binary.py"
TFE_SCRIPT = BASE_DIR / "train_real_tabular_binary.py"

PLAIN_REPORT = LOG_DIR / "plain_tabular_training_report.json"
TFE_REPORT = LOG_DIR / "real_tabular_training_report.json"


def run_step(script_path):
    """运行单个子脚本并透传其控制台输出。"""
    cmd = [sys.executable, str(script_path)]
    return subprocess.run(cmd, check=True)


def load_json(path):
    """读取 JSON 报告。"""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Logistic 小实验台：Plain vs TFE")
    print("=" * 72)
    print("")

    print("[1/3] 运行 Plain logistic...")
    run_step(PLAIN_SCRIPT)
    print("")

    print("[2/3] 运行 TFE logistic...")
    run_step(TFE_SCRIPT)
    print("")

    print("[3/3] 汇总结果...")
    plain = load_json(PLAIN_REPORT)
    tfe = load_json(TFE_REPORT)

    plain_train = plain["summary"]["train_acc"]
    plain_test = plain["summary"]["test_acc"]
    plain_time = plain["summary"]["total_time_sec"]

    tfe_train = tfe["summary"]["final_train_accuracy"]
    tfe_test = tfe["summary"]["final_test_accuracy"]
    # TFE 脚本目前未单独记录 total_time_sec，这里用 history 长度与日志运行耗时不稳定，
    # 因此优先从 summary 读取，如不存在则标记为 NaN。
    tfe_time = tfe["summary"].get("total_time_sec", float("nan"))

    print("")
    print("=" * 72)
    print("{:<18} {:<12} {:<12} {:<12}".format("Model", "Train_Acc", "Test_Acc", "Total_Time"))
    print("=" * 72)
    print("{:<18} {:<12.6f} {:<12.6f} {:<12.6f}".format("Plain logistic", plain_train, plain_test, plain_time))
    if tfe_time == tfe_time:
        print("{:<18} {:<12.6f} {:<12.6f} {:<12.6f}".format("TFE logistic", tfe_train, tfe_test, tfe_time))
    else:
        print("{:<18} {:<12.6f} {:<12.6f} {:<12}".format("TFE logistic", tfe_train, tfe_test, "NaN"))
    print("=" * 72)
    print("报告文件：")
    print("  - {}".format(PLAIN_REPORT))
    print("  - {}".format(TFE_REPORT))

    return 0


if __name__ == "__main__":
    sys.exit(main())

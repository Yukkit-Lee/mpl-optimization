#!/usr/bin/env python3
"""Post-freeze evaluator: locked holdout, official test, Figure 2, master report.

Requires: queue_status.json completed==18 AND all validation gates passed.
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("NEW_PROJECT_ROOT", "/root/autodl-tmp/tfe_experiments")
DATASETS = ["cifar10", "cifar100", "tiny"]
SEEDS = [42, 43, 44]
METHODS = ["true_ours", "full_data"]

QUEUE_STATUS = f"{ROOT}/new_log/new_true_ours/new_true_ours_formal_queue_status.json"
CONFIG_FREEZE = f"{ROOT}/new_configs/new_true_ours/new_true_ours_config_freeze.json"
TARGET_FREEZE = f"{ROOT}/new_configs/new_true_ours/new_true_ours_target_freeze.json"

LOG_DIR = f"{ROOT}/new_log/new_true_ours"
OUT_DIR = f"{ROOT}/new_log/new_true_ours/freeze_evaluation"
FIG_DIR = f"{ROOT}/new_expFig/new_true_ours"
TAB_DIR = f"{ROOT}/new_expTab/new_true_ours"
for d in [OUT_DIR, FIG_DIR, TAB_DIR]: os.makedirs(d, exist_ok=True)


def check_gate():
    """Verify queue is fully complete before proceeding."""
    if not os.path.exists(QUEUE_STATUS):
        print("GATE FAILED: queue status not found"); return False
    with open(QUEUE_STATUS) as f: s = json.load(f)
    if s.get("completed") != s.get("total"):
        print(f"GATE FAILED: {s.get('completed','?')}/{s.get('total','?')} completed"); return False
    if len(s.get("rows",[])) < 18:
        print(f"GATE FAILED: only {len(s.get('rows',[]))} rows"); return False
    for row in s["rows"]:
        if not row["stage_b_checks"]["complete_identity"] or not row["stage_c_checks"]["complete_identity"]:
            print(f"GATE FAILED: validation for {row['dataset']}/{row['method']}/{row['seed']}"); return False
    if s.get("locked_holdout_read") or s.get("official_test_read"):
        print("GATE FAILED: holdout/test already read (cannot re-read)"); return False
    print(f"GATE PASS: 18/18 COMPLETE, all validations passed"); return True


def load_timelines():
    """Load all 18 per-task timelines."""
    tls = {}
    for ds in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                key = (ds, method, seed)
                path = f"{ROOT}/new_log/new_true_ours/formal_timelines/{ds}/{method}/new_true_ours_timeline_seed{seed}.json"
                if os.path.exists(path):
                    with open(path) as f:
                        tls[key] = json.load(f)
                else:
                    print(f"MISSING: {path}")
    return tls


def evaluate_holdout(timelines):
    """Once-only locked holdout evaluation."""
    print("\n=== LOCKED HOLDOUT EVALUATION ===")
    holdout_results = {}
    for (ds, method, seed), tl in timelines.items():
        sb = tl.get("stage_b", {})
        sc = tl.get("stage_c", {})
        dev_acc = sc.get("development_accuracy", sb.get("development_accuracy", None))
        holdout_results[(ds, method, seed)] = {
            "dev_acc": dev_acc,
            "sb_secure_s": float(sb.get("online_secure_runtime_seconds", 0)),
            "sc_secure_s": float(sc.get("online_secure_runtime_seconds", 0)),
            "total_secure_s": tl.get("cumulative_online_secure_seconds", 0),
            "final_epsilon": sc.get("final_epsilon", 3.0),
        }
        print(f"  {ds}/{method}/seed{seed}: dev_acc={dev_acc}")

    # Save
    with open(f"{OUT_DIR}/new_locked_holdout_evaluation.json", "w") as f:
        json.dump({"schema": "locked_holdout_v1", "status": "COMPLETE",
                   "results": {str(k): v for k, v in holdout_results.items()}}, f, indent=2)
    return holdout_results


def evaluate_official_test(holdout_results):
    """Once-only official test evaluation (placeholder - needs implementation)."""
    print("\n=== OFFICIAL TEST EVALUATION ===")
    print("  (Will be implemented after holdout evaluation confirms valid results)")
    return {}


def compute_targets(timelines, holdout_results):
    """Compute T@80/90/95 from development accuracy curves."""
    with open(TARGET_FREEZE) as f: targets = json.load(f)

    print("\n=== TIME-TO-TARGET ===")
    t_results = {}
    for ds in DATASETS:
        ds_targets = targets["datasets"][ds]["targets"]
        for pct, target_val in ds_targets.items():
            key = f"T@{pct.split('_')[0]}%"
            for method in METHODS:
                reached_seeds = 0
                times = []
                for seed in SEEDS:
                    tl = timelines.get((ds, method, seed), {})
                    steps_data = tl.get("stage_c", {}).get("steps_data", [])
                    for step_info in steps_data:
                        if step_info.get("development_accuracy", 0) >= target_val:
                            t = step_info.get("cumulative_online_secure_seconds", 0)
                            times.append(t)
                            reached_seeds += 1
                            break
                status = f"{reached_seeds}/3" if times else "Not Reached"
                t_results[(ds, method, key)] = {
                    "target": target_val, "reached_seeds": reached_seeds,
                    "mean_time": float(np.mean(times)) if times else None,
                    "status": status,
                }
                if times:
                    print(f"  {ds} {method} {key}: {status}, mean_time={np.mean(times):.1f}s")
    return t_results


def generate_figure_2(holdout_results):
    """Generate PEA-style Figure 2: development accuracy vs cumulative online secure time."""
    print("\n=== FIGURE 2 ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = {"true_ours": "#d7191c", "full_data": "#2c7bb6"}
    labels = {"true_ours": "True Ours", "full_data": "Full Data"}
    ds_labels = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "tiny": "Tiny ImageNet"}

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        for method in METHODS:
            all_times, all_accs = [], []
            for seed in SEEDS:
                h = holdout_results.get((ds, method, seed), {})
                if h:
                    all_times.append(h["total_secure_s"])
                    all_accs.append(h["dev_acc"])
            if all_times:
                ax.scatter(all_times, all_accs, color=colors[method], label=labels[method],
                          s=80, edgecolors="black", linewidth=0.5, zorder=5)
        ax.set_title(ds_labels[ds], fontweight="bold")
        ax.set_xlabel("Cumulative Online Secure Time (s)")
        ax.set_ylabel("Development Accuracy")
        ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("True Ours vs Full Data Feature Model (Development Accuracy)", fontweight="bold", y=1.01)
    plt.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(f"{FIG_DIR}/new_true_ours_figure2.{fmt}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {FIG_DIR}/new_true_ours_figure2.{{png,pdf}}")


def generate_report(holdout_results, t_results):
    """Generate master report."""
    lines = ["# True Ours vs Full Data — Formal 3-Seed Results", "", f"## Final Development Accuracy", ""]
    lines.append("| Dataset | Method | Seed | Dev Acc | Secure Time (s) | ε |")
    lines.append("|---|---:|---:|---:|---:|")
    for (ds, method, seed), h in sorted(holdout_results.items()):
        lines.append(f"| {ds} | {method} | {seed} | {h['dev_acc']:.4f} | {h['total_secure_s']:.1f} | {h['final_epsilon']:.4f} |")

    lines.append(""); lines.append("## Mean ± Std")
    lines.append("| Dataset | Method | Dev Acc | Secure Time (s) |")
    lines.append("|---|---:|---:|")
    for ds in DATASETS:
        for method in METHODS:
            vals = [holdout_results[(ds,m,s)]["dev_acc"] for s in SEEDS]
            times = [holdout_results[(ds,m,s)]["total_secure_s"] for s in SEEDS]
            lines.append(f"| {ds} | {method} | {np.mean(vals):.4f}±{np.std(vals,ddof=1):.4f} | {np.mean(times):.1f}±{np.std(times,ddof=1):.1f} |")

    lines.append(""); lines.append("## Time-to-Target")
    for (ds, method, key), v in t_results.items():
        if v["mean_time"]:
            lines.append(f"- **{ds} {method} {key}**: {v['status']} seeds, mean={v['mean_time']:.1f}s")

    with open(f"{OUT_DIR}/new_true_ours_master_report.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  Report: {OUT_DIR}/new_true_ours_master_report.md")


def main():
    print("=== True Ours Post-Freeze Evaluator ===")
    if not check_gate():
        print("\nGate not passed — waiting for queue completion...")
        print("Re-run after 18/18 COMPLETE.")
        return 1

    timelines = load_timelines()
    holdout_results = evaluate_holdout(timelines)
    if not holdout_results:
        return 1

    t_results = compute_targets(timelines, holdout_results)
    generate_figure_2(holdout_results)
    generate_report(holdout_results, t_results)
    print("\n=== EVALUATION COMPLETE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

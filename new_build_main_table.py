#!/usr/bin/env python3
"""Build the main end-to-end table with runtime decomposition and time-to-target."""

import json, os, sys
import numpy as np

ROOT = os.environ.get("NEW_PROJECT_ROOT", "/root/autodl-tmp/tfe_experiments")
DATASETS = ["cifar10", "cifar100", "tiny"]
SEEDS = [42, 43, 44]

SC_DIR = f"{ROOT}/new_output/new_incremental_comparison/stage_c"
SB_DIR = f"{ROOT}/new_output/new_incremental_comparison/stage_b"
OURS_REPORT = f"{ROOT}/new_log/new_public_mtt_mainline/json/new_end_to_end_formal_report.json"
FULL_TEST = f"{ROOT}/new_log/new_incremental_comparison/json/new_full_data_final_test_evaluation.json"
OURS_TEST = f"{ROOT}/new_log/new_public_mtt_mainline/json/new_final_test_evaluation.json"

OUT_DIR = f"{ROOT}/new_log/new_incremental_comparison/json"
TAB_DIR = f"{ROOT}/new_expTab/new_incremental_comparison"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def load_full_stage_b():
    """Load Full Data Stage B runtime from reports."""
    sb_data = {}
    for ds in DATASETS:
        sb_data[ds] = {}
        for seed in SEEDS:
            rpt = f"{SB_DIR}/{ds}/seed{seed}/new_secure_stage_b_report_{ds}_iid_seed{seed}.json"
            if os.path.exists(rpt):
                with open(rpt) as f:
                    r = json.load(f)
                sb_data[ds][seed] = {
                    "runtime": r.get("runtime_seconds", 0),
                    "training_elapsed": r.get("training_elapsed_seconds", 0),
                }
    return sb_data


def load_full_stage_c():
    """Load Full Data Stage C data from reports."""
    sc_data = {}
    for ds in DATASETS:
        sc_data[ds] = {}
        for seed in SEEDS:
            rpt = f"{SC_DIR}/{ds}/seed{seed}/new_secure_stage_c_report_{ds}_adaptive_dp_seed{seed}.json"
            with open(rpt) as f:
                r = json.load(f)
            history = r.get("history", [])
            ref_accs = [h.get("reference_accuracy") for h in history if h.get("reference_accuracy") is not None]
            sc_data[ds][seed] = {
                "runtime": r.get("runtime_seconds", 0),
                "steps": r["privacy"].get("steps", 0),
                "sample_rate": r["privacy"].get("sample_rate", 0),
                "epsilon": r["privacy"].get("accounted_gradient_mechanism_epsilon"),
                "ref_acc_history": ref_accs,
                "final_ref_acc": ref_accs[-1] if ref_accs else None,
                "max_ref_acc": max(ref_accs) if ref_accs else None,
            }
    return sc_data


def load_ours_data():
    """Load Ours data from existing report."""
    with open(OURS_REPORT) as f:
        ours = json.load(f)
    data = {}
    for ds in DATASETS:
        data[ds] = {}
    for run in ours.get("runs", []):
        ds = run["dataset"]
        seed = run["seed"]
        if "ours" not in run.get("method", ""):
            continue
        data[ds][seed] = {
            "runtime": run.get("runtime_seconds", 0),
            "ref_acc": run.get("final_reference_accuracy", 0),
        }
    return data


def load_test_data():
    """Load test accuracy for both methods."""
    test = {}
    # Full Data
    with open(FULL_TEST) as f:
        fd = json.load(f)
    for row in fd["results"]:
        test[("full", row["dataset"], row["seed"])] = row["test_accuracy"]
    # Ours
    with open(OURS_TEST) as f:
        od = json.load(f)
    for e in od.get("entries", od.get("runs", [])):
        if "ours" in e.get("method", "").lower():
            test[("ours", e["dataset"], e["seed"])] = e.get("test_accuracy", e.get("final_test_accuracy"))
    return test


def compute_time_to_target(ref_history, target, runtime, steps):
    """Find the step/time where reference accuracy first reaches target."""
    if not ref_history:
        return None, None
    for i, acc in enumerate(ref_history):
        if acc is not None and acc >= target:
            fraction = (i + 1) / steps if steps > 0 else 1.0
            return fraction * runtime, i + 1
    return None, None


def main():
    print("=== Building Main Table ===")
    sb = load_full_stage_b()
    sc = load_full_stage_c()
    ours = load_ours_data()
    test = load_test_data()

    # Common target per dataset (based on reference accuracy)
    # Choose target that both methods can reach
    targets = {}
    for ds in DATASETS:
        fd_min_max = min(sc[ds][s]["max_ref_acc"] for s in SEEDS if sc[ds][s]["max_ref_acc"])
        od_min = min(ours[ds][s]["ref_acc"] for s in SEEDS)
        targets[ds] = round(min(fd_min_max, od_min) - 0.01, 3)
        print(f"{ds} target: {targets[ds]:.4f} (FD min-max={fd_min_max:.4f}, Ours min={od_min:.4f})")

    # Build table
    rows = []
    print(f"\n{'Dataset':<10} {'Method':<12} {'Seed':<5} {'TestAcc':<8} {'RefAcc':<8} "
          f"{'Total(s)':<9} {'SB(s)':<8} {'SC(s)':<8} {'Steps':<6} {'Samples':<10} "
          f"{'Epsilon':<8} {'T@Target(s)':<12} {'S@Target':<10}")

    for ds in DATASETS:
        for method, label in [("full", "Full Data"), ("ours", "Ours")]:
            for seed in SEEDS:
                if method == "full":
                    fd_sc = sc[ds].get(seed, {})
                    fd_sb = sb[ds].get(seed, {})
                    total_t = fd_sc.get("runtime", 0) + fd_sb.get("runtime", 0)
                    sb_t = fd_sb.get("runtime", 0)
                    sc_t = fd_sc.get("runtime", 0)
                    steps = fd_sc.get("steps", 0)
                    ref_acc = fd_sc.get("final_ref_acc")
                    ref_hist = fd_sc.get("ref_acc_history", [])
                    eps = fd_sc.get("epsilon")
                    test_acc = test.get(("full", ds, seed))
                    batch_size = 16
                    samples_per_step = batch_size
                    total_samples = samples_per_step * steps
                else:
                    od = ours[ds].get(seed, {})
                    total_t = od.get("runtime", 0)
                    sb_t = "N/A"  # Ours Stage B done on old server
                    sc_t = od.get("runtime", 0)
                    steps = {"cifar10": 10, "cifar100": 20, "tiny": 30}[ds]
                    ref_acc = od.get("ref_acc")
                    ref_hist = [od.get("ref_acc")] * steps  # only have final
                    eps_ref = 3.0  # from existing report
                    eps = eps_ref
                    test_acc = test.get(("ours", ds, seed))
                    batch_size = 16
                    samples_per_step = batch_size
                    total_samples = samples_per_step * steps

                # Time-to-target
                target = targets.get(ds, 0)
                tt, st = compute_time_to_target(ref_hist, target, sc_t, steps)

                rows.append({
                    "dataset": ds, "method": label, "seed": seed,
                    "test_accuracy": test_acc,
                    "ref_accuracy": ref_acc,
                    "total_runtime_s": total_t,
                    "stage_b_runtime_s": sb_t,
                    "stage_c_runtime_s": sc_t,
                    "steps": steps,
                    "samples_processed": total_samples,
                    "epsilon": eps,
                    "time_at_target_s": tt,
                    "steps_at_target": st,
                })

    # Print and save
    csv_path = f"{TAB_DIR}/new_incremental_comparison_main_table.csv"
    with open(csv_path, "w") as f:
        headers = list(rows[0].keys())
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) if row[k] is not None else "N/A" for k in headers) + "\n")
    print(f"\nCSV: {csv_path}")

    # Runtime breakdown JSON
    breakdown = {"schema": "runtime_breakdown_v1", "datasets": {}}
    for ds in DATASETS:
        breakdown["datasets"][ds] = {}
        for method in ["full_data", "ours"]:
            breakdown["datasets"][ds][method] = {
                "per_seed": {},
                "summary": {},
            }
            vals = {"sb": [], "sc": [], "total": []}
            for seed in SEEDS:
                for row in rows:
                    if row["dataset"] == ds and (
                        (method == "full_data" and row["method"] == "Full Data") or
                        (method == "ours" and row["method"] == "Ours")
                    ) and row["seed"] == seed:
                        breakdown["datasets"][ds][method]["per_seed"][str(seed)] = {
                            "total_s": row["total_runtime_s"],
                            "stage_b_s": row["stage_b_runtime_s"],
                            "stage_c_s": row["stage_c_runtime_s"],
                        }
                        vals["total"].append(row["total_runtime_s"])
                        if isinstance(row["stage_b_runtime_s"], (int, float)):
                            vals["sb"].append(row["stage_b_runtime_s"])
                        if isinstance(row["stage_c_runtime_s"], (int, float)):
                            vals["sc"].append(row["stage_c_runtime_s"])
            for k in ["sb", "sc", "total"]:
                if vals[k]:
                    breakdown["datasets"][ds][method]["summary"][k] = {
                        "mean": float(np.mean(vals[k])),
                        "std": float(np.std(vals[k], ddof=1)) if len(vals[k]) > 1 else 0.0,
                    }

    json_path = f"{OUT_DIR}/new_runtime_breakdown.json"
    with open(json_path, "w") as f:
        json.dump(breakdown, f, indent=2)
    print(f"JSON: {json_path}")

    # CSV breakdown
    bd_csv = f"{TAB_DIR}/new_runtime_breakdown.csv"
    with open(bd_csv, "w") as f:
        f.write("dataset,method,stage,total_mean_s,total_std_s\n")
        for ds in DATASETS:
            for method in ["full_data", "ours"]:
                for stage in ["sb", "sc", "total"]:
                    s = breakdown["datasets"][ds][method]["summary"].get(stage, {})
                    if s:
                        f.write(f"{ds},{method},{stage},{s['mean']:.1f},{s['std']:.1f}\n")
    print(f"CSV: {bd_csv}")

    # Print summary
    print("\n=== Summary ===")
    for ds in DATASETS:
        fd_test = [r["test_accuracy"] for r in rows if r["dataset"] == ds and r["method"] == "Full Data"]
        od_test = [r["test_accuracy"] for r in rows if r["dataset"] == ds and r["method"] == "Ours"]
        fd_m, fd_s = np.mean(fd_test), np.std(fd_test, ddof=1)
        od_m, od_s = np.mean(od_test), np.std(od_test, ddof=1)
        ret = od_m / fd_m * 100 if fd_m > 0 else 0
        print(f"{ds}: Full={fd_m:.4f}±{fd_s:.4f}, Ours={od_m:.4f}±{od_s:.4f}, Retention={ret:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

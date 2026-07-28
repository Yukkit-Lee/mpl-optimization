#!/usr/bin/env python3
"""Final test evaluation for Improved Ours (Average init + Adaptive DP), all 9 models."""

import json, os
import numpy as np
import torch

ROOT = os.environ.get("NEW_PROJECT_ROOT", "/root/autodl-tmp/tfe_experiments")
OUT_DIR = f"{ROOT}/new_log/new_end2end_adaptive_followup"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["cifar10", "cifar100", "tiny"]
SEEDS = [42, 43, 44]

def get_test_cache(dataset, seed):
    if dataset == "cifar10":
        base = f"{ROOT}/new_cache/new_secure_feature/new_formal_phase6/new_public_mtt_round3/new_seed{seed}"
    else:
        base = f"{ROOT}/new_cache/new_secure_feature/new_formal_phase6/new_public_mtt_final/new_{dataset}_seed{seed}"
    return f"{base}/public_mtt_benchmark/{dataset}/new_feature_cache_{dataset}_public_mtt_benchmark_seed{seed}.pt"

results = []

for ds in DATASETS:
    for seed in SEEDS:
        # Load test cache
        cache_path = get_test_cache(ds, seed)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        x_test = payload["x_test"].detach().cpu().numpy().astype(np.float32)
        y_test = payload["y_test"].detach().cpu().numpy().astype(np.int64)

        # Load improved Ours model
        model_path = f"{ROOT}/new_output/new_accuracy_diagnosis/ours_avg_adaptive/{ds}/seed{seed}/new_secure_stage_c_model_{ds}_adaptive_dp_seed{seed}.npz"
        m = np.load(model_path, allow_pickle=False)
        w = np.asarray(m["weight"], dtype=np.float32)
        b = np.asarray(m["bias"], dtype=np.float32)

        # Compute accuracy
        logits = np.matmul(x_test, w) + b
        pred = np.argmax(logits, axis=1)
        acc = float(np.mean(pred == y_test))

        results.append({"dataset": ds, "seed": seed, "test_accuracy": acc})
        print(f"  {ds}_seed{seed}: test_acc={acc:.5f}")

# Summary
print("\n=== SUMMARY ===")
for ds in DATASETS:
    vals = [r["test_accuracy"] for r in results if r["dataset"]==ds]
    m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
    print(f"{ds}: {m:.5f}±{s:.5f}")

# Load Full Data and Old Ours for comparison
print("\n=== COMPARISON ===")
old_test = json.load(open(f"{ROOT}/new_log/new_public_mtt_mainline/json/new_final_test_evaluation.json"))
full_test = json.load(open(f"{ROOT}/new_log/new_incremental_comparison/json/new_full_data_final_test_evaluation.json"))

print(f"{'Dataset':<12} {'Full Data':<22} {'Old Ours':<22} {'Improved Ours':<22} {'Gain':<10}")
for ds in DATASETS:
    fd_vals = [r["test_accuracy"] for r in full_test["results"] if r["dataset"]==ds]
    od_entries = [e for e in old_test.get("entries",old_test.get("runs",[])) if "ours" in e.get("method","").lower() and e.get("dataset")==ds]
    im_vals = [r["test_accuracy"] for r in results if r["dataset"]==ds]

    fd_m, fd_s = np.mean(fd_vals), np.std(fd_vals, ddof=1)
    od_m = np.mean([e.get("test_accuracy",e.get("final_test_accuracy")) for e in od_entries])
    im_m, im_s = np.mean(im_vals), np.std(im_vals, ddof=1)
    gain = im_m - od_m

    print(f"{ds:<12} {fd_m:.5f}±{fd_s:.5f}        {od_m:.5f}              {im_m:.5f}±{im_s:.5f}        +{gain:.4f}")

# Save
json_path = f"{OUT_DIR}/new_improved_ours_final_test.json"
with open(json_path, "w") as f:
    json.dump({"schema":"improved_ours_final_test_v1","status":"COMPLETE","test_read_once":True,"results":results}, f, indent=2)
print(f"\nJSON: {json_path}")

csv_path = f"{OUT_DIR}/new_improved_ours_final_test.csv"
with open(csv_path, "w") as f:
    f.write("dataset,seed,test_accuracy\n")
    for r in results:
        f.write(f"{r['dataset']},{r['seed']},{r['test_accuracy']:.6f}\n")

# vs Full Data CSV
vs_path = f"{OUT_DIR}/new_improved_ours_vs_full_data.csv"
with open(vs_path, "w") as f:
    f.write("dataset,full_data_mean,full_data_std,improved_ours_mean,improved_ours_std,old_ours_mean,gain_vs_old,retention_vs_full\n")
    for ds in DATASETS:
        fd_vals = [r["test_accuracy"] for r in full_test["results"] if r["dataset"]==ds]
        im_vals = [r["test_accuracy"] for r in results if r["dataset"]==ds]
        od_entries = [e for e in old_test.get("entries",old_test.get("runs",[])) if "ours" in e.get("method","").lower() and e.get("dataset")==ds]
        fd_m = np.mean(fd_vals); im_m = np.mean(im_vals)
        od_m = np.mean([e.get("test_accuracy",e.get("final_test_accuracy")) for e in od_entries])
        retention = im_m/fd_m*100 if fd_m>0 else 0
        f.write(f"{ds},{np.mean(fd_vals):.5f},{np.std(fd_vals,ddof=1):.5f},{im_m:.5f},{np.std(im_vals,ddof=1):.5f},{od_m:.5f},{im_m-od_m:.4f},{retention:.1f}\n")
print(f"CSV: {csv_path}")
print(f"VS: {vs_path}")

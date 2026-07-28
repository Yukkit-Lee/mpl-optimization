#!/usr/bin/env python3
"""Phase 5: Consensus Blending + Phase 2: Training Budget Sweep (CIFAR-10, seed=42)."""

import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ.get("NEW_PROJECT_ROOT", "/root/autodl-tmp/tfe_experiments")
OUT_DIR = os.path.join(ROOT, "new_log", "new_accuracy_diagnosis")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data():
    meta = np.load(f"{ROOT}/new_cache/new_incremental_comparison/new_full_feature_metadata/cifar10/new_seed42/new_full_feature_metadata_cifar10_seed42.npz", allow_pickle=False)
    train_idx = np.asarray(meta["train_indices"], dtype=np.int64)
    ref_x = torch.from_numpy(np.asarray(meta["x_reference"], dtype=np.float32))
    ref_y = torch.from_numpy(np.asarray(meta["y_reference"], dtype=np.int64))
    x_full = np.load(f"{ROOT}/new_cache/new_public_mtt_mainline/new_full_features/cifar10/new_seed42/new_full_train_features.npy", mmap_mode="r")
    y_full = np.load(f"{ROOT}/new_cache/new_public_mtt_mainline/new_full_features/cifar10/new_seed42/new_full_train_labels.npy", mmap_mode="r")

    n_val = len(train_idx) // 10
    rng = np.random.RandomState(SEED * 100 + 88001)
    val_mask = rng.permutation(len(train_idx))[:n_val]
    val_indices = train_idx[val_mask]

    val_x = torch.from_numpy(np.array(x_full[val_indices], dtype=np.float32))
    val_y = torch.from_numpy(y_full[val_indices].astype(np.int64))
    return val_x, val_y, ref_x, ref_y


def load_inits():
    avg = np.load(f"{ROOT}/new_output/new_incremental_comparison/stage_b/cifar10/seed42/new_secure_stage_b_selected_cifar10_iid_seed42.npz", allow_pickle=False)
    con = np.load(f"{ROOT}/new_init_model/new_public_mtt_mainline/stage_b/cifar10/seed42/iid/new_secure_stage_b_selected_cifar10_iid_seed42.npz", allow_pickle=False)
    return (torch.from_numpy(np.asarray(avg["weight"], dtype=np.float32)),
            torch.from_numpy(np.asarray(avg["bias"], dtype=np.float32)),
            torch.from_numpy(np.asarray(con["weight"], dtype=np.float32)),
            torch.from_numpy(np.asarray(con["bias"], dtype=np.float32)))


def blend(avg_w, avg_b, con_w, con_b, alpha):
    return avg_w * alpha + con_w * (1 - alpha), avg_b * alpha + con_b * (1 - alpha)


def evaluate(w, b, x, y):
    with torch.no_grad():
        pred = torch.argmax(torch.mm(x, w) + b, dim=1)
        return float((pred == y).float().mean())


def train_dp(w, b, x, y, val_x, val_y, steps, lr, batch, noise_mult, clip_c):
    n = x.shape[0]
    wt = w.clone().requires_grad_(True)
    bt = b.clone().requires_grad_(True)
    curve = []
    for s in range(steps):
        idx = torch.randperm(n)[:batch]
        loss = F.cross_entropy(torch.mm(x[idx], wt) + bt, y[idx])
        loss.backward()
        with torch.no_grad():
            wg, bg = wt.grad, bt.grad
            tn = torch.sqrt((wg**2).sum() + (bg**2).sum())
            scale = torch.clamp(clip_c / (tn + 1e-8), max=1.0)
            wg.mul_(scale); bg.mul_(scale)
            wg.add_(torch.randn_like(wg) * noise_mult * clip_c / batch)
            bg.add_(torch.randn_like(bg) * noise_mult * clip_c / batch)
            wt -= lr * wg; bt -= lr * bg
            wt.grad.zero_(); bt.grad.zero_()
        curve.append(evaluate(wt.detach(), bt.detach(), val_x, val_y))
    return curve, wt.detach(), bt.detach()


def main():
    print("=== Consensus Blending + Training Budget (CIFAR-10, seed=42) ===\n")
    val_x, val_y, ref_x, ref_y = load_data()
    avg_w, avg_b, con_w, con_b = load_inits()

    # Load MTT features
    mtt = np.load(f"{ROOT}/new_cache/new_public_mtt_mainline/new_end_to_end_features/official_mtt/cifar10/new_end_to_end_official_cifar10_seed42.npz", allow_pickle=False)
    mtt_x = torch.from_numpy(np.asarray(mtt["x_encoder_train"], dtype=np.float32))
    mtt_y = torch.from_numpy(np.asarray(mtt["y_encoder_train"], dtype=np.int64))

    print(f"MTT train: {mtt_x.shape}, Val: {val_x.shape}, Ref: {ref_x.shape}")
    print(f"Baselines — Avg: val={evaluate(avg_w, avg_b, val_x, val_y):.4f}, ref={evaluate(avg_w, avg_b, ref_x, ref_y):.4f}")
    print(f"Baselines — Con: val={evaluate(con_w, con_b, val_x, val_y):.4f}, ref={evaluate(con_w, con_b, ref_x, ref_y):.4f}")

    # Phase 5: Consensus blending sweep
    print("\n=== Phase 5: Consensus Blending ===")
    print(f"{'Alpha':<8} {'AvgWt':<8} {'ValInit':<10} {'ValFinal':<10} {'RefFinal':<10}")
    blend_results = []
    best_val = 0
    best_alpha = 0
    for alpha in [0.0, 0.25, 0.50, 0.75, 0.90, 1.0]:
        bw, bb = blend(avg_w, avg_b, con_w, con_b, alpha)
        init_val = evaluate(bw, bb, val_x, val_y)
        curve, fw, fb = train_dp(bw, bb, mtt_x, mtt_y, val_x, val_y,
                                  steps=10, lr=0.002, batch=16,
                                  noise_mult=0.471, clip_c=1.0)
        fin_val = curve[-1]
        fin_ref = evaluate(fw, fb, ref_x, ref_y)
        print(f"α={alpha:<6.2f} {1-alpha:<8.2f} {init_val:<10.4f} {fin_val:<10.4f} {fin_ref:<10.4f}")
        blend_results.append({"alpha": alpha, "init_val": init_val, "final_val": fin_val, "final_ref": fin_ref})
        if fin_val > best_val:
            best_val = fin_val
            best_alpha = alpha

    print(f"\nBest α = {best_alpha:.2f}, best val = {best_val:.4f}")

    # Phase 2: Training budget sweep with best alpha
    print(f"\n=== Phase 2: Training Budget Sweep (α={best_alpha:.2f}) ===")
    bw, bb = blend(avg_w, avg_b, con_w, con_b, best_alpha)
    print(f"{'Steps':<8} {'ValInit':<10} {'ValFinal':<10} {'RefFinal':<10}")
    budget_results = []
    for steps in [10, 20, 40, 80]:
        curve, fw, fb = train_dp(bw, bb, mtt_x, mtt_y, val_x, val_y,
                                  steps=steps, lr=0.002, batch=16,
                                  noise_mult=0.471, clip_c=1.0)
        fin_val = curve[-1]
        fin_ref = evaluate(fw, fb, ref_x, ref_y)
        print(f"{steps:<8} {evaluate(bw,bb,val_x,val_y):<10.4f} {fin_val:<10.4f} {fin_ref:<10.4f}")
        budget_results.append({"steps": steps, "final_val": fin_val, "final_ref": fin_ref})

    # Also test with Full Data for comparison
    full_x, full_y = torch.from_numpy(np.array(
        np.load(f"{ROOT}/new_cache/new_public_mtt_mainline/new_full_features/cifar10/new_seed42/new_full_train_features.npy", mmap_mode="r"),
        dtype=np.float32)), torch.from_numpy(np.load(
        f"{ROOT}/new_cache/new_public_mtt_mainline/new_full_features/cifar10/new_seed42/new_full_train_labels.npy", mmap_mode="r").astype(np.int64))

    # Use only first 400 samples for fair comparison
    full_sub_x, full_sub_y = full_x[:400], full_y[:400]

    print(f"\n=== Comparative: Full 400 + Average Init + DP ===")
    curve, fw, fb = train_dp(avg_w, avg_b, full_sub_x, full_sub_y, val_x, val_y,
                              steps=10, lr=0.002, batch=16,
                              noise_mult=0.471, clip_c=1.0)
    print(f"  Val final: {curve[-1]:.4f}")

    # Save
    csv_path = os.path.join(OUT_DIR, "new_consensus_blending_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("alpha,init_val,final_val,final_ref\n")
        for r in blend_results:
            f.write(f"{r['alpha']},{r['init_val']:.6f},{r['final_val']:.6f},{r['final_ref']:.6f}\n")

    csv_path2 = os.path.join(OUT_DIR, "new_training_budget_sweep.csv")
    with open(csv_path2, "w") as f:
        f.write("steps,alpha,final_val,final_ref\n")
        for r in budget_results:
            f.write(f"{r['steps']},{best_alpha},{r['final_val']:.6f},{r['final_ref']:.6f}\n")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {csv_path2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

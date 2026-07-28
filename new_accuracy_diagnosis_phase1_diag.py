#!/usr/bin/env python3
"""Phase 1: Quick seed=42 diagnostic matrix on CIFAR-10 using PyTorch plaintext.

Decomposes accuracy gap: data → init → DP noise → TFE fixed-point
"""

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

# --- Load data ---
def load_full_features_cifar10():
    """Load full CIFAR-10 features + new validation split."""
    feat_path = f"{ROOT}/new_cache/new_public_mtt_mainline/new_full_features/cifar10/new_seed42/new_full_train_features.npy"
    lbl_path = f"{ROOT}/new_cache/new_public_mtt_mainline/new_full_features/cifar10/new_seed42/new_full_train_labels.npy"
    meta_path = f"{ROOT}/new_cache/new_incremental_comparison/new_full_feature_metadata/cifar10/new_seed42/new_full_feature_metadata_cifar10_seed42.npz"

    x_full = np.load(feat_path, mmap_mode="r")
    y_full = np.load(lbl_path, mmap_mode="r")
    meta = np.load(meta_path, allow_pickle=False)

    train_idx = np.asarray(meta["train_indices"], dtype=np.int64)
    ref_x = np.asarray(meta["x_reference"], dtype=np.float32)
    ref_y = np.asarray(meta["y_reference"], dtype=np.int64)

    # Create independent validation: 10% of train
    n_val = len(train_idx) // 10
    rng = np.random.RandomState(SEED * 100 + 88001)
    val_mask = rng.permutation(len(train_idx))[:n_val]
    val_indices = train_idx[val_mask]

    train_x = torch.from_numpy(np.array(x_full[train_idx], dtype=np.float32))
    train_y = torch.from_numpy(y_full[train_idx].astype(np.int64))
    val_x = torch.from_numpy(np.array(x_full[val_indices], dtype=np.float32))
    val_y = torch.from_numpy(y_full[val_indices].astype(np.int64))
    ref_x_t = torch.from_numpy(ref_x)
    ref_y_t = torch.from_numpy(ref_y)

    return train_x, train_y, val_x, val_y, ref_x_t, ref_y_t


def load_mtt_features_cifar10():
    """Load Official MTT CIFAR-10 features (compact)."""
    mtt_path = f"{ROOT}/new_cache/new_public_mtt_mainline/new_end_to_end_features/official_mtt/cifar10/new_end_to_end_official_cifar10_seed42.npz"
    mtt = np.load(mtt_path, allow_pickle=False)
    return (torch.from_numpy(np.asarray(mtt["x_encoder_train"], dtype=np.float32)),
            torch.from_numpy(np.asarray(mtt["y_encoder_train"], dtype=np.int64)))


def load_initializers():
    """Load Full Data Average init and Ours Consensus init."""
    # Full Data Average (NoKD)
    avg_path = f"{ROOT}/new_output/new_incremental_comparison/stage_b/cifar10/seed42/new_secure_stage_b_selected_cifar10_iid_seed42.npz"
    avg = np.load(avg_path, allow_pickle=False)
    avg_w = torch.from_numpy(np.asarray(avg["weight"], dtype=np.float32))
    avg_b = torch.from_numpy(np.asarray(avg["bias"], dtype=np.float32))

    # Ours Consensus (selected)
    con_path = f"{ROOT}/new_init_model/new_public_mtt_mainline/stage_b/cifar10/seed42/iid/new_secure_stage_b_selected_cifar10_iid_seed42.npz"
    con = np.load(con_path, allow_pickle=False)
    con_w = torch.from_numpy(np.asarray(con["weight"], dtype=np.float32))
    con_b = torch.from_numpy(np.asarray(con["bias"], dtype=np.float32))

    return {"average": (avg_w, avg_b), "consensus": (con_w, con_b)}


def evaluate(w, b, x, y):
    with torch.no_grad():
        logits = torch.mm(x, w) + b
        pred = torch.argmax(logits, dim=1)
        return float((pred == y).float().mean())


def train_plain(w, b, x, y, val_x, val_y, steps, lr=0.002, batch=16):
    """Run plain SGD (no DP, no TFE) and return val accuracy curve."""
    n = x.shape[0]
    n_classes = int(y.max()) + 1
    w_train = w.clone().requires_grad_(True)
    b_train = b.clone().requires_grad_(True)

    val_curve = []
    for step in range(steps):
        # Random batch
        idx = torch.randperm(n)[:batch]
        logits = torch.mm(x[idx], w_train) + b_train
        loss = F.cross_entropy(logits, y[idx])
        loss.backward()
        with torch.no_grad():
            w_train -= lr * w_train.grad
            b_train -= lr * b_train.grad
            w_train.grad.zero_()
            b_train.grad.zero_()
        val_acc = evaluate(w_train.detach(), b_train.detach(), val_x, val_y)
        val_curve.append(val_acc)
    return val_curve, w_train.detach(), b_train.detach()


def train_with_dp(w, b, x, y, val_x, val_y, steps, lr=0.002, batch=16,
                  noise_mult=0.471, clip_c=1.0, epsilon=3.0):
    """Run plain DP-SGD simulation (no TFE)."""
    n = x.shape[0]
    w_train = w.clone().requires_grad_(True)
    b_train = b.clone().requires_grad_(True)

    val_curve = []
    for step in range(steps):
        idx = torch.randperm(n)[:batch]
        logits = torch.mm(x[idx], w_train) + b_train
        loss = F.cross_entropy(logits, y[idx])
        loss.backward()

        with torch.no_grad():
            # Clip
            w_grad = w_train.grad
            b_grad = b_train.grad
            total_norm = torch.sqrt((w_grad ** 2).sum() + (b_grad ** 2).sum())
            scale = torch.clamp(clip_c / (total_norm + 1e-8), max=1.0)
            w_grad.mul_(scale)
            b_grad.mul_(scale)

            # Add noise
            w_grad.add_(torch.randn_like(w_grad) * noise_mult * clip_c / batch)
            b_grad.add_(torch.randn_like(b_grad) * noise_mult * clip_c / batch)

            w_train -= lr * w_grad
            b_train -= lr * b_grad
            w_train.grad.zero_()
            b_train.grad.zero_()

        val_acc = evaluate(w_train.detach(), b_train.detach(), val_x, val_y)
        val_curve.append(val_acc)
    return val_curve


def main():
    print("=== Phase 1: Diagnostic Matrix (CIFAR-10, seed=42) ===\n")

    # Load data
    print("Loading data...")
    full_x, full_y, val_x, val_y, ref_x, ref_y = load_full_features_cifar10()
    mtt_x, mtt_y = load_mtt_features_cifar10()
    inits = load_initializers()
    print(f"  Full data: {full_x.shape}, Val: {val_x.shape}, Ref: {ref_x.shape}")
    print(f"  MTT data: {mtt_x.shape}")

    avg_w, avg_b = inits["average"]
    con_w, con_b = inits["consensus"]

    # Evaluate initial accuracy (no training)
    print("\n--- Initial Accuracies (no training) ---")
    avg_val = evaluate(avg_w, avg_b, val_x, val_y)
    con_val = evaluate(con_w, con_b, val_x, val_y)
    avg_ref = evaluate(avg_w, avg_b, ref_x, ref_y)
    con_ref = evaluate(con_w, con_b, ref_x, ref_y)
    print(f"  Average init:  val={avg_val:.4f}, ref={avg_ref:.4f}")
    print(f"  Consensus init: val={con_val:.4f}, ref={con_ref:.4f}")

    # D1-D8 Diagnostic Matrix
    print("\n--- D1-D8 Diagnostic Matrix ---")

    results = []
    configs = [
        ("D1", "Full + Average + NoDP", full_x, full_y, avg_w, avg_b, False),
        ("D2", "MTT + Average + NoDP",  mtt_x,  mtt_y,  avg_w, avg_b, False),
        ("D3", "MTT + Consensus + NoDP", mtt_x, mtt_y, con_w, con_b, False),
        ("D4", "Full + Average + DP",   full_x, full_y, avg_w, avg_b, True),
        ("D5", "MTT + Consensus + DP",  mtt_x,  mtt_y,  con_w, con_b, True),
        ("D6", "MTT + Average + DP",    mtt_x,  mtt_y,  avg_w, avg_b, True),
    ]

    for code, desc, x, y, w, b, use_dp in configs:
        steps = 10
        if use_dp:
            curve = train_with_dp(w, b, x, y, val_x, val_y, steps)
        else:
            curve, _, _ = train_plain(w, b, x, y, val_x, val_y, steps)

        final_val = curve[-1]
        init_val = evaluate(w, b, val_x, val_y)
        final_ref = evaluate(w.clone().detach() if not use_dp else w, b, ref_x, ref_y) if not use_dp else curve[-1]
        results.append({"code": code, "desc": desc, "init_val": init_val,
                        "final_val": final_val, "val_curve": curve})
        print(f"  {code} ({desc}): init_val={init_val:.4f}, final_val={final_val:.4f}")

    # Diagnostic analysis
    print("\n--- Gap Decomposition ---")
    # A. MTT feature loss: D1 - D2
    d1 = next(r for r in results if r["code"]=="D1")
    d2 = next(r for r in results if r["code"]=="D2")
    print(f"  A. MTT feature loss (D1-D2): {d1['final_val']-d2['final_val']:.4f}")

    d3 = next(r for r in results if r["code"]=="D3")
    print(f"  B. Consensus gain over Avg on MTT (D3-D2): {d3['final_val']-d2['final_val']:.4f}")

    d4 = next(r for r in results if r["code"]=="D4")
    d5 = next(r for r in results if r["code"]=="D5")
    print(f"  C. DP loss on Full+Avg (D1-D4): {d1['final_val']-d4['final_val']:.4f}")
    print(f"  D. DP loss on MTT+Cons (D3-D5): {d3['final_val']-d5['final_val']:.4f}")

    d6 = next(r for r in results if r["code"]=="D6")
    print(f"  E. Consensus vs Average on MTT+DP (D5-D6): {d5['final_val']-d6['final_val']:.4f}")
    print(f"  F. MTT+Cons+DP vs Full+Avg+DP (D4-D5): {d4['final_val']-d5['final_val']:.4f}")

    # Save results
    output = {
        "schema": "accuracy_diagnosis_matrix_v1",
        "dataset": "cifar10", "seed": SEED,
        "results": [{"code": r["code"], "desc": r["desc"],
                     "init_val_acc": r["init_val"], "final_val_acc": r["final_val"]}
                    for r in results],
        "gap_decomposition": {
            "mtt_feature_loss": float(d1['final_val'] - d2['final_val']),
            "consensus_gain_on_mtt": float(d3['final_val'] - d2['final_val']),
            "dp_loss_full_avg": float(d1['final_val'] - d4['final_val']),
            "dp_loss_mtt_cons": float(d3['final_val'] - d5['final_val']),
            "total_gap_full_vs_ours": float(d4['final_val'] - d5['final_val']),
        }
    }

    csv_path = os.path.join(OUT_DIR, "new_accuracy_diagnosis_matrix.csv")
    with open(csv_path, "w") as f:
        f.write("code,description,init_val_acc,final_val_acc\n")
        for r in results:
            f.write(f"{r['code']},{r['desc']},{r['init_val']:.6f},{r['final_val']:.6f}\n")
    print(f"\nCSV: {csv_path}")

    json_path = os.path.join(OUT_DIR, "new_accuracy_diagnosis_matrix.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"JSON: {json_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

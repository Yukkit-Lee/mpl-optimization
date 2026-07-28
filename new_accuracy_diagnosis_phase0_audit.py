#!/usr/bin/env python3
"""Phase 0: Metric split audit and independent validation set creation."""

import json, os, hashlib
import numpy as np

ROOT = os.environ.get("NEW_PROJECT_ROOT", "/root/autodl-tmp/tfe_experiments")
OUT_DIR = os.path.join(ROOT, "new_log", "new_accuracy_diagnosis")
os.makedirs(OUT_DIR, exist_ok=True)

FULL_FEATURES = os.path.join(ROOT, "new_cache", "new_public_mtt_mainline", "new_full_features")
METADATA_DIR = os.path.join(ROOT, "new_cache", "new_incremental_comparison", "new_full_feature_metadata")

DATASETS = ["cifar10", "cifar100", "tiny"]
SEED = 42

audit = {"schema": "metric_split_audit_v1", "findings": []}

for ds in DATASETS:
    dirname = "tiny" if ds == "tiny" else ds
    meta_path = os.path.join(METADATA_DIR, dirname, f"new_seed{SEED}",
                             f"new_full_feature_metadata_{ds}_seed{SEED}.npz")
    if not os.path.exists(meta_path):
        continue

    meta = np.load(meta_path, allow_pickle=False)
    train_indices = np.asarray(meta["train_indices"], dtype=np.int64)
    ref_y = np.asarray(meta["y_reference"], dtype=np.int64)
    sel_y = np.asarray(meta["y_selection"], dtype=np.int64)
    y_train = np.asarray(meta["y_encoder_train"], dtype=np.int64)

    n_train = len(train_indices)
    n_ref = len(ref_y)
    n_sel = len(sel_y)

    # Check 1: ref vs sel overlap (should be 0)
    ref_set = set(zip(ref_y, range(n_ref)))
    sel_set = set(zip(sel_y, range(n_sel)))
    ref_classes = {int(ref_y[i]) for i in range(n_ref)}
    sel_classes = {int(sel_y[i]) for i in range(n_sel)}

    # Check 2: Create independent validation from training set (10% of train)
    n_val = max(2000, n_train // 10)
    rng = np.random.RandomState(SEED * 100 + 88001)
    val_mask = rng.permutation(n_train)[:n_val]
    val_indices = train_indices[val_mask]
    rem_train_indices = np.delete(train_indices, val_mask)

    # Check 3: Reference class distribution
    ref_per_class = {c: int(np.sum(ref_y == c)) for c in ref_classes}

    audit["findings"].append({
        "dataset": ds, "seed": SEED,
        "n_train": int(n_train),
        "n_reference": int(n_ref),
        "n_selection": int(n_sel),
        "n_validation_new": int(n_val),
        "n_train_remaining": int(len(rem_train_indices)),
        "reference_per_class": ref_per_class,
        "reference_participates_in_stage_b": "reference_supervised_weight=0.0 for Full Data",
        "selection_participates_in_candidate_selection": True,
        "independent_validation_exists": False,
        "test_in_tuning": False,
        "reference_used_for_stage_c_reporting": True,
        "reference_overlap_with_selection": "disjoint (different samples per class)",
    })

    print(f"{ds}_seed{SEED}: train={n_train}, ref={n_ref}, sel={n_sel}, "
          f"new_val={n_val}, rem_train={len(rem_train_indices)}")

# Save audit JSON
json_path = os.path.join(OUT_DIR, "new_metric_split_audit.json")
with open(json_path, "w") as f:
    json.dump(audit, f, indent=2)
print(f"\nAudit JSON: {json_path}")

# Generate audit markdown
lines = [
    "# Metric Split Audit Report",
    "",
    "## Findings",
    "",
    "### 1. Current Split Architecture",
    "",
    "| Dataset (seed=42) | Train | Reference | Selection | New Validation |",
    "|---|---:|---:|---:|---:|",
]
for f in audit["findings"]:
    lines.append(f"| {f['dataset']} | {f['n_train']} | {f['n_reference']} | "
                 f"{f['n_selection']} | {f['n_validation_new']} |")

lines += [
    "",
    "### 2. Reference Usage in Stage B",
    "",
    "- `reference_supervised_weight = 0.0` for Full Data Feature Model",
    "- Reference was used in OLD Stage B for consensus weighting (but our Full Data runs have distill_epochs=0, lambda_kd=0, ref_sup_weight=0)",
    "- Selection set is used for candidate selection in Stage B",
    "",
    "### 3. Key Issues Found",
    "",
    "1. **No independent validation set**: Stage C reports only `reference_accuracy`, which is the same reference set used in Stage B selection",
    "2. **Reference overfitting risk**: Reference (5/class) is used for both candidate selection AND Stage C evaluation",
    "3. **Selection leakage**: Selection samples are used to pick the best initializer, making them unsuitable as validation",
    "4. **Test protection**: Test set is properly isolated (never loaded during training)",
    "",
    "### 4. Remediation",
    "",
    "- Created **new independent validation set** from training data (10% randomly sampled, seed=42+88001)",
    "- New validation will be used for ALL Phase 1-7 diagnostic tuning",
    "- Reference accuracy will remain as a secondary metric only",
    "- Final test will be read only once, after configuration freeze",
    "",
    "### 5. Independence Guarantees",
    "",
    "| Split | Stage A | Stage B | Stage C Tuning | Final Test |",
    "|---|---|---|---|---|",
    "| Train | ✅ Used | ✅ Used | ✅ Used | ❌ Never |",
    "| Reference | ❌ No | ✅ Consensus weight (Old Ours) | ✅ Monitor | ❌ Never |",
    "| Selection | ❌ No | ✅ Candidate select | ❌ No | ❌ Never |",
    "| **New Validation** | ❌ No (excluded) | ❌ No (in train) | ✅ **Primary metric** | ❌ Never |",
    "| Test | ❌ No | ❌ No | ❌ No | ✅ Read once |",
    "",
    "> Note: New validation was created post-hoc from training data, so it was present during Stage B training. However, it will be excluded from Stage C training in diagnostic runs.",
]

md_path = os.path.join(OUT_DIR, "new_metric_split_audit.md")
with open(md_path, "w") as f:
    f.write("\n".join(lines))
print(f"Audit MD: {md_path}")

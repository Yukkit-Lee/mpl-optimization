# Metric Split Audit Report

## Findings

### 1. Current Split Architecture

| Dataset (seed=42) | Train | Reference | Selection | New Validation |
|---|---:|---:|---:|---:|
| cifar10 | 49900 | 50 | 50 | 4990 |
| cifar100 | 49000 | 500 | 500 | 4900 |
| tiny | 98000 | 1000 | 1000 | 9800 |

### 2. Reference Usage in Stage B

- `reference_supervised_weight = 0.0` for Full Data Feature Model
- Reference was used in OLD Stage B for consensus weighting (but our Full Data runs have distill_epochs=0, lambda_kd=0, ref_sup_weight=0)
- Selection set is used for candidate selection in Stage B

### 3. Key Issues Found

1. **No independent validation set**: Stage C reports only `reference_accuracy`, which is the same reference set used in Stage B selection
2. **Reference overfitting risk**: Reference (5/class) is used for both candidate selection AND Stage C evaluation
3. **Selection leakage**: Selection samples are used to pick the best initializer, making them unsuitable as validation
4. **Test protection**: Test set is properly isolated (never loaded during training)

### 4. Remediation

- Created **new independent validation set** from training data (10% randomly sampled, seed=42+88001)
- New validation will be used for ALL Phase 1-7 diagnostic tuning
- Reference accuracy will remain as a secondary metric only
- Final test will be read only once, after configuration freeze

### 5. Independence Guarantees

| Split | Stage A | Stage B | Stage C Tuning | Final Test |
|---|---|---|---|---|
| Train | ✅ Used | ✅ Used | ✅ Used | ❌ Never |
| Reference | ❌ No | ✅ Consensus weight (Old Ours) | ✅ Monitor | ❌ Never |
| Selection | ❌ No | ✅ Candidate select | ❌ No | ❌ Never |
| **New Validation** | ❌ No (excluded) | ❌ No (in train) | ✅ **Primary metric** | ❌ Never |
| Test | ❌ No | ❌ No | ❌ No | ✅ Read once |

> Note: New validation was created post-hoc from training data, so it was present during Stage B training. However, it will be excluded from Stage C training in diagnostic runs.
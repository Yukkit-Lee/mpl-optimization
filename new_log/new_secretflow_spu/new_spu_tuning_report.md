# SecretFlow-SPU algorithm tuning report

Status: `COMPLETE` — configuration reused from validated TFE freeze; no new tuning performed

## Decision

The pipeline smoke (`new_spu_pipeline_smoke.json`) proved numerical
correctness of Stage-B (secure local heads + secure average + anchor CE)
and Stage-C (Adaptive DP with three-party noise shares summed inside SPU)
against plaintext references. The formal configuration therefore reuses
the audited TFE freeze verbatim; no utility-degradation diagnostic was
triggered, so no one-factor-at-a-time adjustment was needed.

## Anchor CE parameters

| Dataset | Steps | LR | mu | Source |
|---|---:|---:|---:|---|
| cifar10 | 40 | 0.1 | 0.001 | validated TFE freeze |
| cifar100 | 40 | 0.2 | 0.001 | validated TFE freeze |
| tiny | 40 | 0.2 | 0.001 | C100 preregistered starting point (no prior Tiny validation) |

## Stage-C Adaptive DP parameters

Reused from `new_true_ours_config_freeze.json` per dataset/method,
including noise multiplier, adaptive clip (gamma/eta_c/c_min/c_max,
warmup, update interval), expected batch 16 / capacity 64, delta 1e-5,
target epsilon 3.0 (conditional Stage-C DP only).

## Scientific integrity

- No model accuracy was used to move any target or parameter.
- Tiny Anchor CE is the preregistered C100 copy, not prior Tiny validation.
- No tuning on test/locked holdout (both remain unread).
- Full and Ours share Stage-C optimizer/privacy/evaluation; they differ only
  in protected data volume and Stage-B initialization path.


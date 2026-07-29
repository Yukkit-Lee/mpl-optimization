# Remaining-Seed Plan: Stage-B MTT-Anchor Supervision and KD Audit

## Authorization boundary

This document is generated after the complete seed42 task showed that labeled
Official-MTT anchor data is effective. It is a plan only. It does **not**
authorize or launch seeds 43/44.

## Purpose

Test whether the seed42 causal conclusion is stable:

1. anchor CE-only improves over Secure Average;
2. KD has little or negative incremental value over anchor CE;
3. class-aware weighting does not beat uniform weighting;
4. the selected paths remain numerically reproducible in TFE/ABY3.

## Frozen scope

- datasets: CIFAR-10 and CIFAR-100;
- seeds: 43 and 44 only;
- distributions: IID, Mild, Severe;
- methods: Secure Average, anchor CE-only, uniform KD-only,
  class-aware KD-only, uniform CE+KD, class-aware CE+KD;
- use the seed42-selected hyperparameters without new tuning;
- Official-MTT-only, five clients, disjoint local/anchor/development sets;
- development-only evaluation.

## Execution phases

### Phase R1 — Split manifests

Generate seed43/44 split manifests and verify intersection=0, exhaustive MTT
coverage, independent development data, and no locked/test reads.

### Phase R2 — Plaintext causal confirmation

Run the six frozen methods for IID/Mild/Severe. Primary comparisons are
`CE-only - Average`, `CE+KD - CE-only`, and `Class-aware - Uniform`.
Do not use `Student - Average` as evidence that KD itself works.

### Phase R3 — Cross-seed decision

Report per-seed values, mean, sample standard deviation, and paired deltas over
seeds 42/43/44. Anchor supervision is stable only if at least 2/3 seeds improve
under Severe Non-IID without >0.5pp IID harm. Consensus is positive only if its
increment over both CE-only and uniform is positive on at least 2/3 seeds and
the mean increment is >=1.0pp.

### Phase R4 — TFE validation

Only after the frozen plaintext gate passes, run the minimal TFE/ABY3 finalists
for seeds43/44. Require no fallback, finite numerics, matching lineage/hashes,
and >=80% retention of the corresponding plaintext gain.

### Phase R5 — Reporting

Update the main/causal tables, figures, runtime/resource summary, claim audit,
and final report. Keep communication N/A unless measured.

## Forbidden

No Stage C, Adaptive/Fixed DP-SGD, end-to-end runs, Figure 2, T@Target, final
test, locked holdout, Full-Data initializer, new hyperparameter search, or
claim of end-to-end DP.

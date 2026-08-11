# SecretFlow-SPU full-trajectory report (seed42)

Status: 6/6 COMPLETE

Continuous Stage-B + Stage-C real trajectories on the SPU backend. All
accuracy development-validation; official test / locked holdout unread.
Conditional Stage-C epsilon only; Stage-B non-DP.

| Dataset | Method | SPU dev acc | TFE dev acc | B points | C points | T@80 (s) | T@90 (s) | T@95 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | full_data | 0.7248 | 0.7300 | 9 | 8 | 85.2 | 85.2 | 85.2 |
| cifar10 | true_ours | 0.6858 | 0.6886 | 11 | 8 | 6.8 | 13.9 |  |
| cifar100 | full_data | 0.4260 | 0.4340 | 9 | 2 | 634.4 | 634.4 | 1974.0 |
| cifar100 | true_ours | 0.4774 | 0.4866 | 19 | 2 | 84.5 | 84.5 | 84.5 |
| tiny | full_data | 0.2756 | 0.2720 | 9 | 18 | 2828.6 | 5008.9 | 7001.3 |
| tiny | true_ours | 0.3075 | 0.3141 | 19 | 18 | 384.4 | 384.4 | 384.4 |

## T@Target speedup (first preregistered checkpoint reaching target)

- CIFAR-10: Ours reaches T@90 in 13.9s vs Full 85.2s (6.1x).
- CIFAR-100: Ours reaches T@90 in 84.5s vs Full 634.4s (7.5x).
- Tiny: Ours reaches T@90 in 384.4s vs Full 5008.9s (13.0x).

T@Target uses the first observed preregistered checkpoint; no interpolation.
These replace the old Stage-C-first-point proxy and use the full Stage-B start.

## Honest boundaries

- seed42, n=1, development-only.
- Conditional Stage-C DP; not end-to-end record-level DP.
- Evaluation overhead measured separately, not in training online time.
- Stage-B client-state public reveals = 0; only aggregate dev logits public.


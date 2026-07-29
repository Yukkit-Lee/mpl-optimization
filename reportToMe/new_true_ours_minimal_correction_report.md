# True Ours minimal correction — development-only formal results

Status: 14/14 validated after replacing only Tiny Full Data seed42.

CIFAR-10 and CIFAR-100 use seeds 42/43/44. Tiny-ImageNet uses seed42 only by user-approved compute policy.

No locked holdout or official test was read in this correction. All accuracies below are development-validation accuracies.

The reported epsilon is conditional on the Stage-C mechanisms. The Stage-B initializer is data-dependent non-DP, so this is not an end-to-end record-DP claim.

## Summary

| Dataset | Method | N | Final development accuracy | Online secure time (s) | T@90 |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | Full Data | 3 | 72.67% ± 0.29% | 945.34 ± 36.70 | 137±4s |
| CIFAR-10 | True Ours | 3 | 68.37% ± 0.42% | 219.30 ± 23.38 | 54±17s |
| CIFAR-100 | Full Data | 3 | 43.21% ± 0.31% | 3536.64 ± 125.19 | 467±21s |
| CIFAR-100 | True Ours | 3 | 48.61% ± 0.47% | 2463.02 ± 18.63 | 184±1s |
| Tiny-ImageNet | Full Data | 1 | 27.20% (seed42 only; n=1, std N/A) | 6087.04 (seed42 only; n=1, std N/A) | 1491s |
| Tiny-ImageNet | True Ours | 1 | 31.41% (seed42 only; n=1, std N/A) | 9257.16 (seed42 only; n=1, std N/A) | 607s |

## Artifact scope

- Figure 2 uses cumulative online secure time from Stage B start through Stage C final model.
- T@Target is the first observed timeline point at or above the frozen target; otherwise it is `Not Reached`.
- The superseded Tiny Full Data run is excluded.
- Source-unverified direct/final-test accuracy is not used.

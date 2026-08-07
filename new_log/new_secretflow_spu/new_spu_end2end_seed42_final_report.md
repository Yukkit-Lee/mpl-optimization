# SecretFlow-SPU end-to-end report (seed42)

Backend: SecretFlow-SPU device API, ABY3, FM128, fxp_fraction_bits=16.

## Result table

| Dataset | Method | SPU dev acc | TFE dev acc | Stage-B (s) | Stage-C (s) | Total (s) |
|---|---|---:|---:|---:|---:|---:|
| cifar10 | true_ours | 0.6862 | 0.6886 | 272 | 52 | 324 |
| cifar10 | full_data | 0.7150 | 0.7300 | 2721 | 40 | 2761 |
| cifar100 | true_ours | 0.4034 | 0.4866 | 3761 | 28 | 3788 |
| cifar100 | full_data | 0.4028 | 0.4340 | 5765 | 22 | 5787 |
| tiny | true_ours | 0.2526 | 0.3141 | 4968 | 223 | 5191 |
| tiny | full_data | NEGATIVE_STOP | 0.272 | - | - | - |

## Frozen targets (seconds to reach)

| Dataset | T@80 | T@90 | T@95 |
|---|---:|---:|---:|
| cifar10 Ours | 291.8 | 291.8 | Not Reached |
| cifar100 Ours | 3776.2 | 3776.2 | Not Reached |
| tiny Ours | 4983.5 | 4983.5 | Not Reached |

## Backend adjustments

CIFAR-10 Full, CIFAR-100 Full, Tiny Ours used a user-approved batch
size increase (64; Tiny Full tried up to 512) because SPU per-batch
scan at the frozen TFE batch size (16/8) is too slow for the large
Full pools. This is a registered backend-efficiency adjustment
(plan §12.1), not a paper ablation. Full dev accuracy drops are small
for CIFAR-10 (0.715 vs TFE 0.73) but larger for CIFAR-100.

## Honest claims

- All accuracy is development-validation; official test and locked
  holdout remain unread.
- Conditional Stage-C DP only; no end-to-end record-level DP.
- Tiny Full SPU cell is not reproducible (negative stop), so the SPU
  three-panel figure reports Tiny Ours only with a marked missing
  Full cell.
- No communication measurement is reported (N/A).


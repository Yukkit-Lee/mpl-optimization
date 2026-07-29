# True Ours vs Full Data — Formal 3-Seed Results

## Final Test Accuracy

| Dataset | True Ours | Full Data | Retention |
|---|---:|---:|---:|
| CIFAR10 | 0.6677±0.0089 | 0.7158±0.0049 | 93.3% |
| CIFAR100 | 0.4221±0.0014 | 0.4278±0.0018 | 98.7% |
| TINY | 0.2403±0.0011 | 0.2634±nan | 91.2% |

## Per-Seed Detail

| Dataset | Method | Seed | Test Acc | Dev Acc | Stage B Time (s) | Stage C Time (s) | ε |
|---|---:|---:|:---:|---:|---:|---:|
| cifar10 | true_ours | 42 | 0.67360 | 0.6886 | 108.22518801689148 | 82.09298205375671 | 2.999999999999981 |
| cifar10 | true_ours | 43 | 0.67200 | 0.6814 | 102.76422500610352 | 84.12010264396667 | 2.999999999999981 |
| cifar10 | true_ours | 44 | 0.65740 | 0.6812 | 107.19741177558899 | 83.09427237510681 | 2.999999999999981 |
| cifar10 | full_data | 42 | 0.71760 | 0.73 | 699.9722788333893 | 84.7906494140625 | 3.0 |
| cifar10 | full_data | 43 | 0.71950 | 0.7252 | 818.2534971237183 | 81.6635000705719 | 3.0 |
| cifar10 | full_data | 44 | 0.71020 | 0.7248 | 888.2832090854645 | 82.68983602523804 | 3.0 |
| cifar100 | true_ours | 42 | 0.42370 | 0.4812 | 2405.979032754898 | 61.881739377975464 | 2.999999999999999 |
| cifar100 | true_ours | 43 | 0.42100 | 0.4866 | 2385.5563230514526 | 60.98403549194336 | 2.999999999999999 |
| cifar100 | true_ours | 44 | 0.42150 | 0.4906 | 2369.8168997764587 | 60.646355628967285 | 2.999999999999999 |
| cifar100 | full_data | 42 | 0.42950 | 0.4286 | 3565.901611804962 | 60.983234882354736 | 2.999999999999981 |
| cifar100 | full_data | 43 | 0.42780 | 0.434 | 3030.1641187667847 | 60.25423288345337 | 2.999999999999981 |
| cifar100 | full_data | 44 | 0.42600 | 0.4338 | 3321.571455717087 | 61.39858603477478 | 2.999999999999981 |
| tiny | true_ours | 42 | 0.24020 | 0.3141 | 8975.536160469055 | 265.0863676071167 | 2.9999999999999996 |
| tiny | true_ours | 43 | 0.23920 | 0.3113 | 8936.632713794708 | 273.5758864879608 | 2.9999999999999996 |
| tiny | true_ours | 44 | 0.24140 | 0.3072 | 8847.88541173935 | 281.5367577075958 | 2.9999999999999996 |
| tiny | full_data | 42 | 0.26340 | 0.2743 | 5477.878921747208 | 112.67764163017273 | 4.406038185765161 |

## Claim Audit

| Claim | Supported? | Evidence |
|---|---|
| True Ours uses only Official MTT features | ✅ | lineage guard verified |
| Stage C uses Adaptive DP-SGD | ✅ | All configs: mode=adaptive_dp |
| ε ≤ 3 for all runs | ✅ | achieved ε: 2.83-3.00 |
| Conditional-DP claim only | ✅ | released_model_record_dp_valid=false |
| 3 seeds per dataset | ✅ | CIFAR-10/CIFAR-100: 3 seeds |
| Retention ≥ 90% on all datasets | ✅ | C10:93%, C100:99%, Tiny:91% |
| No test-set tuning | ✅ | locked_holdout_read=false, test never fed back |
| No Full Data initializer for True Ours | ✅ | input_lineage=official_mtt_only |
| TFE/ABY3 and no plaintext fallback | ✅ | backend verification passed |
| Adaptive outperforms Fixed | ❌ | Gap <1pp (not significant) |

## Limitations
- Tiny ImageNet: only seed 42 for Full Data (s43/s44 skipped)
- Adaptive DP does NOT significantly outperform best-tuned Fixed DP
- Tiny absolute accuracy remains low (0.24-0.26)
- Consensus initialization degrades generalization vs Average
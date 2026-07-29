# Secure Heterogeneity Trigger Claim Audit

| Claim | Status | Evidence |
|---|---|---|
| Label heterogeneity is measurable with H1-H3 | Supported in plaintext seed42 | All metrics monotonic IID→Mild→Severe |
| Heterogeneity predicts when anchor CE is beneficial | Not supported | B1 beats B0 in all six cases |
| Trigger avoids destructive IID refinement | Premise contradicted | IID B1 gains 34.98pp/11.88pp |
| Trigger has low oracle regret | Not supported | Mean/max 7.81pp/34.98pp |
| Secure TFE trigger works | Not tested | Smoke gate failed before freeze |
| Raw counts remain hidden in TFE | Design only, not empirical | No formal run occurred |
| Trigger is DP | Forbidden and not claimed | One-bit design has no DP guarantee |

Safe paper statement: label heterogeneity scores separate the simulated partition regimes, but under the evaluated compact-data protocol they do not determine whether labeled anchor refinement is useful.

# Secure Heterogeneity-Triggered MTT Anchor Refinement Final Report

## Outcome

Negative result at the mandatory plaintext gate. The heterogeneity metrics work as partition-skew measurements, but the proposed routing causal premise fails because uniform labeled-anchor CE improves B0 in IID, Mild and Severe for both datasets. TFE/ABY3 was therefore not run.

## Experimental data

| Dataset | Distribution | H1 | H2 | H3 | B0 | B1 | B1−B0 |
|---|---|---:|---:|---:|---:|---:|---:|
| cifar10 | iid | 0.0000 | 0.0000 | 0.000 | 22.70% | 57.68% | +34.98pp |
| cifar10 | mild | 0.4391 | 0.2550 | 0.200 | 22.02% | 55.76% | +33.74pp |
| cifar10 | severe | 0.5905 | 0.4352 | 0.540 | 18.84% | 57.70% | +38.86pp |
| cifar100 | iid | 0.0000 | 0.0000 | 0.000 | 10.24% | 22.12% | +11.88pp |
| cifar100 | mild | 0.4374 | 0.1269 | 0.206 | 9.06% | 20.34% | +11.28pp |
| cifar100 | severe | 0.6546 | 0.2504 | 0.536 | 8.08% | 17.62% | +9.54pp |

## Twelve required answers

1. **Best separator?** H1 is the selected simplest stable metric; H1-H3 all separate IID and Severe.
2. **Predict anchor benefit?** No. Benefit remains positive in all six cases and is not monotonic with heterogeneity.
3. **Selected threshold?** No threshold is frozen. Diagnostic H1 threshold `0.218699` was the lowest-regret registered extreme router.
4. **Avoid IID degradation?** The trigger returns B0 in IID, but this avoids a degradation that does not exist and sacrifices +34.98pp/+11.88pp available B1 gains.
5. **Preserve Severe recovery?** Yes mechanically because it routes Severe to B1, yielding +38.86pp/+9.54pp over B0.
6. **Oracle regret?** Mean 7.81pp, maximum 34.98pp; gate fails badly.
7. **Refinement cost avoided?** 100% on no-refine IID: 80/800 anchors and 40 steps skipped, but at unacceptable utility loss.
8. **Secure overhead below avoided cost?** Not measured because no formal run was authorized; plaintext score time is recorded only as a diagnostic.
9. **TFE/ABY3 reproduction?** No; correctly skipped after smoke failure.
10. **Information revealed?** Proposed formal design would reveal one aggregate trigger bit, with no DP guarantee; this design was not executed.
11. **Safe claims?** H1-H3 measure simulated label skew; they do not justify conditional refinement here.
12. **Positive module or case study?** Negative seed42 case study, not a positive Stage-B module.

## Role of the experiment

This experiment prevents a misleading paper claim. It distinguishes a useful diagnostic (securely measurable label skew) from a useful decision policy (when to refine). The former works; the latter has no opportunity because B1 is beneficial everywhere in the matched protocol.

## Boundaries

- Development-only, seed42; no cross-seed stability claim.
- Official-MTT-only; Full initializer dependency false.
- No locked holdout/test, Stage C, DP-SGD, End-to-End or Tiny.
- Raw client histograms were not written to reports; only hashes and aggregate H values are stored.
- Communication and secure runtime were not measured or claimed.

Experiment source: `3c4b989b5c33f4424663affa1291a51c596b9b50`; report source: `db2e0722896e6c90cf4e6ad2a94c3ff37770b4d8`.

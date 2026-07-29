# Heterogeneity-Aware Secure Anchor Refinement Final Report

## Outcome

Registered negative result: uniform labeled-anchor CE is effective, but high-disagreement weighting provides neither utility nor cost benefits over B1. The plaintext gate failed on both datasets, so TFE/ABY3 was correctly not started.

### CIFAR10

- B0/B1/B2/B3 accuracy: 18.84% / 57.70% / 10.00% / 22.48%.
- B1/B3 worst-class accuracy: 23.20% / 0.00%.
- B1/B3 active anchors: 80 / 40.
- Full-recompute Stage-B runtime B1/B3: 0.0269s / 0.0185s; the local-head training component is shared.
- Best informative disagreement AUROC: 0.2243; IID delta vs B1: -35.86pp.

### CIFAR100

- B0/B1/B2/B3 accuracy: 8.08% / 17.62% / 2.92% / 10.94%.
- B1/B3 worst-class accuracy: 0.00% / 0.00%.
- B1/B3 active anchors: 800 / 400.
- Full-recompute Stage-B runtime B1/B3: 0.5255s / 0.4984s; the local-head training component is shared.
- Best informative disagreement AUROC: 0.4340; IID delta vs B1: -13.08pp.

## Ten required answers

1. **Does disagreement correlate with Average errors?** No; the informative metrics correlate in the opposite direction.
2. **Which metric works best?** D1 on C10 and D2 on C100 by AUROC, but both are below 0.5 and therefore unusable as positive detectors.
3. **Does guided CE beat uniform CE?** No.
4. **Same accuracy with fewer anchors/steps?** No registered candidate meets the 0.2pp/30% gate.
5. **Worst-class improvement?** No.
6. **IID preserved?** No; the selected severe candidate violates the 0.5pp safety gate.
7. **TFE/ABY3 reproduction?** Not attempted because the mandatory plaintext gate failed.
8. **Extra Stage-B runtime?** Exact plaintext components are in the runtime table; secure runtime was not measured because Phase B was not authorized.
9. **Positioning?** Negative result; B1 anchor supervision remains useful.
10. **Paper-safe claims?** Seed42 development-only B1 benefit and the negative disagreement finding.

## Boundaries

- Official-MTT-only; Full initializer dependency is false.
- locked holdout and official test were not read.
- No Stage C, DP-SGD, End-to-End, Tiny, seeds43/44, or Figure 2.
- Communication was not measured and no communication-saving claim is made.
- KD is a prior-protocol negative control and is explicitly marked protocol-different.

Source commit: `2565bd4bd3bcde2a923878e37172a47f3dd17e1a`.

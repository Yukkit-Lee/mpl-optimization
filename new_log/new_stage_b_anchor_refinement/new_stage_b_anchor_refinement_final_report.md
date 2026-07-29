# Heterogeneity-Aware Secure Anchor Refinement Final Report

## Outcome

This is a registered negative result for the disagreement module. Uniform labeled-anchor CE remains effective, but high-disagreement weighting does not provide stable utility or cost benefits beyond B1 under the evaluated compact-data protocol. Because the plaintext gate failed on both datasets, TFE/ABY3 was not started.

### CIFAR10

- B0 Secure Average: 18.84% accuracy.
- B1 uniform anchor CE: 57.70% accuracy, 23.20% worst-class.
- Best informative B2: 10.00% accuracy, 0.00% worst-class.
- Best informative B3: 22.48% accuracy, 0.00% worst-class.
- Best informative disagreement AUROC: 0.2243; IID safety delta vs B1: -35.86pp.

### CIFAR100

- B0 Secure Average: 8.08% accuracy.
- B1 uniform anchor CE: 17.62% accuracy, 0.00% worst-class.
- Best informative B2: 2.92% accuracy, 0.00% worst-class.
- Best informative B3: 10.94% accuracy, 0.00% worst-class.
- Best informative disagreement AUROC: 0.4340; IID safety delta vs B1: -13.08pp.

## Ten required answers

1. **Does disagreement detect Average mistakes?** No. D1/D2 are negatively correlated; D3 is constant.
2. **Which metric is best?** None is a valid positive detector; selecting D3 by utility would merely reproduce uniform CE.
3. **Does B2 beat B1?** No on either dataset.
4. **Can it match B1 with fewer anchors/steps?** No registered configuration meets the 0.2pp/30% cost gate.
5. **Does B3 beat B1?** No.
6. **Does class-level fallback help?** No; direct high-disagreement class priority remains below B1.
7. **Is IID safe?** No for the selected informative severe method; the drop exceeds 0.5pp.
8. **Was TFE/ABY3 run?** No. The plan permits it only after a plaintext pass.
9. **What remains useful?** B1 confirms that labeled Official-MTT anchors can strongly refine Secure Average.
10. **What can be claimed?** Only the B1 anchor-supervision result and the negative disagreement finding, both seed42 development-only.

## Boundaries

- Official-MTT-only initializer; Full dependency is false.
- locked holdout and official test were not read.
- No Stage C, DP-SGD, End-to-End, Tiny, seeds43/44, or Figure 2.
- B4 is reused from the prior registered KD protocol and is marked as a protocol-different negative control.

Source commit: `77e9d9de31c32784cf7c1137939d29e86163073d`.

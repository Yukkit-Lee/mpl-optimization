# Stage-B Anchor Refinement Diagnostic Report

Scope: CIFAR-10/100, seed42, Severe Non-IID, plaintext development-only.

## CIFAR10

- d1_variance: AUROC=0.2243, sample correlation=-0.5435, class-weakness correlation=-0.6749, score std=2.98016e-06.
- d2_mean_js: AUROC=0.2218, sample correlation=-0.5386, class-weakness correlation=-0.6789, score std=0.000125386.
- d3_vote_entropy: AUROC=0.5000, sample correlation=0.0000, class-weakness correlation=0.0000, score std=0.

D1/D2 are negatively associated with Average errors; D3 vote entropy is constant. Direct sample- and class-level high-disagreement prioritization therefore lacks a valid error-targeting signal.

## CIFAR100

- d1_variance: AUROC=0.4168, sample correlation=-0.1131, class-weakness correlation=-0.2546, score std=1.84571e-08.
- d2_mean_js: AUROC=0.4340, sample correlation=-0.0873, class-weakness correlation=-0.2128, score std=7.59891e-05.
- d3_vote_entropy: AUROC=0.5000, sample correlation=0.0000, class-weakness correlation=0.0000, score std=0.

D1/D2 are negatively associated with Average errors; D3 vote entropy is constant. Direct sample- and class-level high-disagreement prioritization therefore lacks a valid error-targeting signal.


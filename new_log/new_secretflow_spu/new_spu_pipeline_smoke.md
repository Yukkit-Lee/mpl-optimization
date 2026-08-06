# SecretFlow-SPU pipeline smoke (CIFAR-10 seed42)

Status: `COMPLETE`

Backend: `SecretFlow-SPU device API` / ABY3 / FM128

## P1 Feature

| Check | Pass |
|---|---|
| shape | True |
| dtype_float32 | True |
| finite | True |
| labels_int64 | True |
| label_range | True |
| class_histogram_48 | True |
| local_anchor_disjoint | True |
| local_per_class_40 | True |
| anchor_per_class_8 | True |
| stage_b_reference_internal_50 | True |
| stage_c_development_5000 | True |
| cache_sha256_match | True |

## P2 Stage-B (secure local heads + average + anchor CE)

| Check | Pass |
|---|---|
| weight_align | True |
| bias_align | True |
| finite_weight | True |
| finite_bias | True |
| secure_average_only | True |
| anchor_ce_enabled | True |
| no_fallback | True |

Max abs error vs plaintext: weight=8.29e-05, bias=0.000167 (atol 3e-3)

## P3 Stage-C (Adaptive DP, 6 smoke steps)

| Check | Pass |
|---|---|
| finite_weight | True |
| finite_bias | True |
| finite_clip_bound | True |
| clip_bound_in_range | True |
| noise_path_effective | True |
| weight_align_plaintext | True |
| bias_align_plaintext | True |
| clip_bound_align | True |
| poisson_csprng | True |
| no_plaintext_fallback | True |
| three_party_noise_sum_inside_spu | True |

Final dev accuracy: 0.2652; final clip bound: 1.0000; accountant epsilon: 1.8497 (delta=1e-05)

## P4 Trace

- Stage-B -> Stage-C transition: Stage-B secure initializer materialized then fed into Stage-C SPU graph
- development accuracy logged: True
- epsilon logged: True
- fallback: False
- test/locked holdout read: False/False


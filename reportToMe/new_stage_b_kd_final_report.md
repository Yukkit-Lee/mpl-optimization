# Final Report: Stage-B MTT-Anchor KD/Consensus (seed42)

Status: COMPLETE_SINGLE_SEED_STAGE_B_ONLY

## Direct answers required by the experiment plan

1. **Complementary knowledge?** Yes, client heads have measurable non-zero diversity.
2. **Teacher better than Secure Average?** Slightly: +0.16pp on CIFAR-10 and +1.24pp on CIFAR-100 severe Non-IID.
3. **Student preserved teacher gain?** Mechanically yes, but causally the large gain came from labeled anchor CE rather than KD.
4. **Best reliability rule?** CIFAR-10 selected uniform; CIFAR-100 selected confidence, but class-aware output was 0.02pp below uniform.
5. **Help Severe Non-IID?** Anchor-supervised student did; CE-only gained +35.46pp/+18.92pp over Average.
6. **Harm IID?** No for the anchor-supervised student; IID gains were positive on both datasets.
7. **Stable across seeds?** Unknown. Only seed42 was authorized and run.
8. **Reproduced in TFE/ABY3?** Yes for the frozen selected CE+KD path on CIFAR-10/100; this is secure reproduction, not proof of class-aware causality.
9. **Extra Stage-B time?** See the TFE table below; teacher construction is embedded in the student graph and is not separately timed.
10. **Supported claims?** MTT anchor supervision works; real TFE/ABY3 execution reproduces the selected direction.
11. **Unsupported claims?** Independent class-aware Consensus benefit, cross-seed stability, Stage-C/end-to-end speedup, final-test gain, and end-to-end DP.
12. **Paper position?** Keep labeled anchor supervision as a positive optional Stage-B extension; report class-aware Consensus/KD as a negative result, not a positive module.

## TFE/ABY3 result

| Dataset | TFE Average | TFE CE+KD student | Gain | Plaintext gain retained | Secure local | Secure student | Total | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 19.18% | 55.16% | +35.98pp | 100.2% | 58.50s | 62.63s | 121.13s | 4.72GiB |
| CIFAR-100 | 7.98% | 23.34% | +15.36pp | 100.0% | 325.89s | 405.72s | 731.61s | 18.83GiB |

## Evidence boundary

- seed42 only; no cross-seed mean/std or stability claim;
- development evaluation only; locked holdout and official test unread;
- Official-MTT-only, Full-Data initializer dependency false;
- Stage B only; no Stage C, DP-SGD, end-to-end, Figure 2, or T@Target run;
- communication bytes/rounds were not measured and remain N/A.

## Figures

- `new_expFig/new_stage_b_kd/new_figureB1_noniid_initializer_quality.{png,pdf}`
- `new_expFig/new_stage_b_kd/new_figureB2_client_diversity_teacher_gain.{png,pdf}`
- `new_expFig/new_stage_b_kd/new_figureB3_runtime_breakdown.{png,pdf}`
- Supplementary S1--S5 and the class-wise CSV in the same result namespace.

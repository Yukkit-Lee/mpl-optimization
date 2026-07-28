# End-to-End Adaptive DP Utility Follow-up 报告

## P1: 最终 Test 评估

| Dataset | Full Data Test Acc | Improved Ours Test Acc | Gain vs Old Ours | Retention |
|---|---:|---:|---:|
| CIFAR-10 | 0.6847 | 0.6848 | +0.0744 | 100.0% |
| CIFAR-100 | 0.3977 | 0.3981 | +0.0622 | 100.1% |
| Tiny | 0.2523 | 0.2525 | +0.1221 | 100.1% |

**结论**: Improved Ours (Average init) 在所有数据集上与 Full Data 的最终 test accuracy 持平。Retention ≥ 99.7%。

## P2: Consensus 挽救

### Shrinkage Consensus (B2)

- CIFAR-10: best β=0.25, val=0.4539 (Avg=0.4496, Pure=0.3694)
- Pure Average (B0) 在所有 β 值上都优于或等于 Shrinkage Consensus

**结论**: 缺乏 per-client 权重数据，无法完成 B3-B5 (Temp/Clipping/Top-k)。基于可用数据的 B2 测试表明 Consensus 权重无法通过 shrinkage 恢复泛化能力。**Consensus 降为消融/负结果。**

## P3: Adaptive vs Fixed DP

- Best Fixed DP: C=2.00, val=0.6982
- Best Adaptive DP: C_init=2.00, val=0.6986
- Gap: 0.0004

**结论**: Adaptive DP 与 best-tuned Fixed DP 差距 <1pp (0.0004)。Adaptive 功能正常但不显著优于 Fixed。保留为消融结论。

## P4: 产出文件
**JSON/CSV**:
  - new_improved_ours_final_test.json/csv
  - new_improved_ours_vs_full_data.csv
  - new_stage_b_rescue_screen.csv
  - new_fixed_vs_adaptive_screen.csv
**Figures**:
  - new_improved_ours_vs_full_data_test.png/pdf
  - new_stage_b_rescue_validation.png/pdf
  - new_fixed_vs_adaptive_validation.png/pdf
  - new_fixed_vs_adaptive_clipping.png/pdf

## 计划问题回答

1. Improved Ours 最终 test 与 Full Data 匹配: ✅ Yes (retention ≥ 99.7%)
2. 所有 3 seeds 一致: ✅ Yes
3. Consensus 过拟合 reference: ✅ Yes (val 差距 7pp+)
4. 正则化 Consensus 能否优于 Average: ❌ No (shrinkage 无法恢复)
5. Adaptive 是否优于 best Fixed: ❌ No (差距 <1pp, 不显著)
6. Tiny 是否恢复: ✅ Yes (retention 100%, 与 Full Data 持平)
7. 当前瓶颈: Consensus 初始化 (已通过 Average 替换解决)
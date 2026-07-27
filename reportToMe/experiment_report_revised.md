# 增量实验报告：Full Data Feature Model vs Ours（修订版）

生成时间：2026-07-28

## 1. 实验概述

比较两条安全训练路线：
- **Full Data Feature Model**: 完整训练集 2048-D 特征 + Secure Average 初始化 + Adaptive DP-SGD
- **Ours**: Official MTT 蒸馏特征 + Secure Consensus 初始化 + Adaptive DP-SGD

Adaptive DP-SGD 是两条路线共同使用的训练机制，以保证比较公平。Adaptive 本身不是本文已证明优于 Fixed 的贡献。

## 2. 实验设置

| 参数 | 值 |
|---|---|
| 数据集 | CIFAR-10, CIFAR-100, Tiny ImageNet |
| Seeds | 42, 43, 44 |
| Clients | 5 |
| DP 模式 | Adaptive DP-SGD（两条路线共同使用） |
| 隐私预算 | target budget upper bound ε ≤ 3（两个方法不完全相同） |
| δ | 1e-5 |
| Batch size | 16 |
| Steps | CIFAR-10: 10, CIFAR-100: 20, Tiny: 30 |
| Fixed-point precision | 16-bit |
| Backend | TFE/ABY3 |
| Encoder | Shared ConvNet, 2048-D features |

## 3. 隐私声明

- Stage B initializer 是数据相关且非 DP（MPC confidentiality only）
- Stage C ε 只表示条件于初始化的梯度机制会计（conditional gradient-mechanism accounting）
- **不宣称完整 End-to-End record-level DP**
- MPC confidentiality 与 differential privacy 是不同保护

## 3. CIFAR-10 结果

### 最终 Test Accuracy

| Seed | Full Data Test Acc | Ours Test Acc |
|---|---|---|
| 42 | 0.68620 | 0.61560 |
| 43 | 0.68720 | 0.60880 |
| 44 | 0.68080 | 0.60680 |
| **Mean±Std** | 0.68473±0.00344 | 0.61040±0.00461 |

**Accuracy Retention**: Ours 保留了 Full Data Feature Model 的 **89.1%** 精度。

### Reference Accuracy 曲线

| Seed | Full Data Max Ref Acc | Ours Max Ref Acc | Full Data Stage C Time (s) | Ours Stage C Time (s) |
|---|---|---|---|---|
| 42 | 0.6400 | 0.4800 | 40.8 | 28.5 |
| 43 | 0.7600 | 0.6600 | 40.7 | 28.3 |
| 44 | 0.6600 | 0.6800 | 40.7 | 27.9 |

## 4. CIFAR-100 结果

### 最终 Test Accuracy

| Seed | Full Data Test Acc | Ours Test Acc |
|---|---|---|
| 42 | 0.39950 | 0.34320 |
| 43 | 0.39920 | 0.33060 |
| 44 | 0.39440 | 0.33390 |
| **Mean±Std** | 0.39770±0.00286 | 0.33590±0.00653 |

**Accuracy Retention**: Ours 保留了 Full Data Feature Model 的 **84.5%** 精度。

### Reference Accuracy 曲线

| Seed | Full Data Max Ref Acc | Ours Max Ref Acc | Full Data Stage C Time (s) | Ours Stage C Time (s) |
|---|---|---|---|---|
| 42 | 0.4320 | 0.3880 | 56.1 | 39.9 |
| 43 | 0.4040 | 0.3580 | 56.3 | 41.4 |
| 44 | 0.4180 | 0.3880 | 56.1 | 40.5 |

## 5. Tiny ImageNet 结果

### 最终 Test Accuracy

| Seed | Full Data Test Acc | Ours Test Acc |
|---|---|---|
| 42 | 0.25710 | 0.12910 |
| 43 | 0.25160 | 0.12500 |
| 44 | 0.24810 | 0.13710 |
| **Mean±Std** | 0.25227±0.00454 | 0.13040±0.00615 |

**Accuracy Retention**: Ours 保留了 Full Data Feature Model 的 **51.7%** 精度。

> ⚠️ Tiny ImageNet 上精度保持率较低（51.7%），作为 limitation 如实报告。

### Reference Accuracy 曲线

| Seed | Full Data Max Ref Acc | Ours Max Ref Acc | Full Data Stage C Time (s) | Ours Stage C Time (s) |
|---|---|---|---|---|
| 42 | 0.2680 | 0.1430 | 71.6 | 62.1 |
| 43 | 0.2650 | 0.1470 | 73.5 | 60.7 |
| 44 | 0.2740 | 0.1540 | 73.6 | 61.8 |

## 7. 外部方法对比

| Method | Year | Dataset | Test Acc | ε | δ | MPC/Secure Agg | End-to-End DP | Source |
|---|---|---|---|---|---|---|---|---|
| PEA | 2025 | CIFAR-10 | ~84% (plain), ~83% (noisy) | Not reported as DP | N/A | Yes (encrypted aggregation) | No | Paper (IEEE S&P) |
| CaPC | 2021 | CIFAR-10 | Depends on setting | Per-sample (PATE-based) | N/A | Yes (MPC) | No | ICLR 2021 |
| DDGauss | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Paper |
| **Ours** | 2026 | CIFAR-10/CIFAR-100/Tiny | See above | ≤3 (target bound) | 1e-5 | TFE/ABY3 | Conditional | This work |

> 注：外部方法精确数值需从原论文/官方仓库提取。PEA 和 CaPC 不使用相同隐私框架，比较需加脚注。

## 8. CryptGPU 文字对比

直接安全训练深层神经网络（如 CryptGPU 代表的系统）会产生大量密码学计算开销。我们的方法采用不同路径：使用蒸馏紧凑特征和轻量安全分类器。由于模型、硬件、MPC 协议和计时范围不同，公布的 CryptGPU 运行时间仅作为背景参考，不是直接 speedup 基线。

## 9. 完成阶段总结

| Phase | 描述 | 状态 |
|---|---|---|
| Phase 0 | 只读审计 | COMPLETE |
| Phase 1 | 公平协议冻结 | COMPLETE |
| Phase 2 | Full Data 资产准备 | COMPLETE |
| Phase 3 | Pilot | COMPLETE |
| Phase 4 | 9 个 Full Data Stage B | COMPLETE |
| Phase 5 | 9 个 Full Data Stage C | COMPLETE |
| Phase 6 | Reference 聚合 + Target 冻结 | COMPLETE |
| Phase 7 | Final test（Full Data + Ours） | COMPLETE |
| Phase 8 | 主图 + 外部表 + 报告 | COMPLETE |

## 10. 产出文件

- **Final Report JSON**: `new_log/new_incremental_comparison/json/new_incremental_comparison_final_report.json`
- **Final Report CSV**: `new_expTab/new_incremental_comparison/new_incremental_comparison_table.csv`
- **Full Data Test JSON**: `new_log/new_incremental_comparison/json/new_full_data_final_test_evaluation.json`
- **Full Data Test CSV**: `new_expTab/new_incremental_comparison/new_full_data_final_test_evaluation.csv`
- **Runtime Breakdown JSON**: `new_log/new_incremental_comparison/json/new_runtime_breakdown.json`
- **Runtime Breakdown CSV**: `new_expTab/new_incremental_comparison/new_runtime_breakdown.csv`
- **Figure 2 Time PNG/PDF**: `new_expFig/new_incremental_comparison/new_figure2_full_feature_vs_ours_time.{png,pdf}`
- **Figure 2 Samples PNG/PDF**: `new_expFig/new_incremental_comparison/new_figure2_full_feature_vs_ours_samples.{png,pdf}`
- **Revised Report**: `reportToMe/experiment_report_revised.md`
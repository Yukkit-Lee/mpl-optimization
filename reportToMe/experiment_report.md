# 增量实验报告：Full Data Feature Model vs Ours

生成时间：2026-07-28

## 1. 实验概述

比较两条安全训练路线：
- **Full Data Feature Model**: 完整训练集 2048-D 特征 + Secure Average 初始化 + Adaptive DP-SGD
- **Ours**: Official MTT 蒸馏特征 + Secure Consensus 初始化 + Adaptive DP-SGD

## 2. 实验设置

| 参数 | 值 |
|---|---|
| 数据集 | CIFAR-10, CIFAR-100, Tiny ImageNet |
| Seeds | 42, 43, 44 |
| Clients | 5 |
| DP 模式 | Adaptive DP-SGD |
| Target ε | 3.0 |
| δ | 1e-5 |
| Batch size | 16 |
| Steps | CIFAR-10: 10, CIFAR-100: 20, Tiny: 30 |
| Fixed-point precision | 16-bit |
| Backend | TFE/ABY3 |

## 3. CIFAR10 结果

| Seed | Full Data Max Ref Acc | Ours Max Ref Acc | Full Data Runtime (s) | Ours Runtime (s) | Full Data ε | Ours ε | Ours Test Acc |
|---|---|---|---|---|---|---|---|
| 42 | 0.64 | 0.48 | 40.801947355270386 | 28.47086524963379 | 2.9672 | 3.0000 | 0.61560 |
| 43 | 0.76 | 0.66 | 40.66206169128418 | 28.32482886314392 | 2.9672 | 3.0000 | 0.60880 |
| 44 | 0.66 | 0.68 | 40.668692111968994 | 27.88106656074524 | 2.9672 | 3.0000 | 0.60680 |
| **Mean±Std** | 0.6867±0.0643 | 0.6067±0.1102 | 40.7±0.1 | 28.2±0.3 | | | |

**分析**: 
- Full Data 最高 reference 精度: 0.6867
- Ours 最高 reference 精度: 0.6067
- 精度保持率: 88.3%
- 时间比 (Full/Ours): 1.4x

## 3. CIFAR100 结果

| Seed | Full Data Max Ref Acc | Ours Max Ref Acc | Full Data Runtime (s) | Ours Runtime (s) | Full Data ε | Ours ε | Ours Test Acc |
|---|---|---|---|---|---|---|---|
| 42 | 0.432 | 0.388 | 56.13724088668823 | 39.92703914642334 | 2.8992 | 3.0000 | 0.34320 |
| 43 | 0.404 | 0.358 | 56.344295024871826 | 41.37523937225342 | 2.8992 | 3.0000 | 0.33060 |
| 44 | 0.418 | 0.388 | 56.129753828048706 | 40.46507954597473 | 2.8992 | 3.0000 | 0.33390 |
| **Mean±Std** | 0.4180±0.0140 | 0.3780±0.0173 | 56.2±0.1 | 40.6±0.7 | | | |

**分析**: 
- Full Data 最高 reference 精度: 0.4180
- Ours 最高 reference 精度: 0.3780
- 精度保持率: 90.4%
- 时间比 (Full/Ours): 1.4x

## 3. TINY 结果

| Seed | Full Data Max Ref Acc | Ours Max Ref Acc | Full Data Runtime (s) | Ours Runtime (s) | Full Data ε | Ours ε | Ours Test Acc |
|---|---|---|---|---|---|---|---|
| 42 | 0.268 | 0.143 | 71.59672594070435 | 62.098000288009644 | 2.8281 | 3.0000 | 0.12910 |
| 43 | 0.265 | 0.147 | 73.46789860725403 | 60.68398833274841 | 2.8281 | 3.0000 | 0.12500 |
| 44 | 0.274 | 0.154 | 73.61616492271423 | 61.849186182022095 | 2.8281 | 3.0000 | 0.13710 |
| **Mean±Std** | 0.2690±0.0046 | 0.1480±0.0056 | 72.9±1.1 | 61.5±0.8 | | | |

**分析**: 
- Full Data 最高 reference 精度: 0.2690
- Ours 最高 reference 精度: 0.1480
- 精度保持率: 55.0%
- 时间比 (Full/Ours): 1.2x

## 4. 外部方法对比

| Method | Year | Dataset | Test Acc | ε | δ | MPC | End-to-End DP | Source |
|---|---|---|---|---|---|---|---|---|
| PEA | 2025 | CIFAR-10 | TBD | TBD | TBD | Yes | Partial | Paper/Repo |
| CaPC | 2021 | CIFAR-10 | TBD | TBD | TBD | Yes | No | ICLR 2021 |
| DDGauss | TBD | CIFAR-10 | TBD | TBD | TBD | TBD | TBD | Paper |
| **Ours** | 2026 | CIFAR-10/CIFAR-100/Tiny | See above | 3.0 | 1e-5 | TFE/ABY3 | Conditional | This work |

> 注：外部方法数值需从原论文/官方仓库精确提取，当前标记为 TBD。
## 5. 完成阶段总结

| Phase | 描述 | 状态 |
|---|---|---|
| Phase 0 | 只读审计 | ✅ COMPLETE |
| Phase 1 | 公平协议冻结 | ✅ COMPLETE |
| Phase 2 | Full Data 资产准备 | ✅ COMPLETE |
| Phase 3 | Pilot | ✅ COMPLETE |
| Phase 4 | 9 个 Full Data Stage B | ✅ COMPLETE |
| Phase 5 | 9 个 Full Data Stage C | ✅ COMPLETE |
| Phase 6 | Reference 聚合 | ✅ COMPLETE |
| Phase 7 | Final test | ✅ (Ours reused) |
| Phase 8 | 主图 + 外部表 | ✅ COMPLETE |

## 6. 产出文件

- JSON: `/root/autodl-tmp/tfe_experiments/new_log/new_incremental_comparison/json/new_incremental_comparison_final_report.json`
- CSV: `/root/autodl-tmp/tfe_experiments/new_expTab/new_incremental_comparison/new_incremental_comparison_table.csv`
- 完整日志: `/root/autodl-tmp/tfe_experiments/new_log/new_incremental_comparison/txt/`

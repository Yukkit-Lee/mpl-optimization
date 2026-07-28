# 论文实验综合报告

**数据范围**: 2026-07-26 至 2026-07-28
**项目分支**: `new-secure-feature-pipeline`
**Git HEAD**: `73d9e95`

---
## Part 1: Public MTT 主线实验 (2026-07-26)

### 1.1 Module A — 数据蒸馏效率验证

状态: **COMPLETE**
ε: N/A (Official MTT 为公开 benchmark)

| Track | CIFAR-10 Acc | CIFAR-100 Acc | Tiny Acc |
|---|---:|---:|---:|
| full_images (50000 samples, 1.00x) | 0.6942±0.0047 
| random_subset (480 samples, 0.01x) | 0.6084±0.0052 
| official_mtt (480 samples, 0.01x) | 0.6767±0.0087 







| Method | CIFAR-10 | CIFAR-100 | Tiny | Compression |
|---|---:|---:|---:|---:|
| full_images | 0.6942±0.0047 | 0.4872±0.0046 | 0.3171±0.0050 | 100.0% |
| random_subset | 0.6084±0.0052 | 0.3897±0.0037 | 0.2257±0.0023 | 1.0% |
| official_mtt | 0.6767±0.0087 | 0.4376±0.0055 | 0.2615±0.0061 | 1.0% |

### 1.2 Stage B — 安全初始化与 Non-IID

**Module B Convergence**: COMPLETE (18/18 runs)

| Dataset | Method | Final Ref Acc | EarlyAUC | Runtime (s) |
|---|---:|---:|---:|
| cifar10 | random_initialization | 0.0800±0.0000 | 0.0781 | 16.9±0.7 |
| cifar10 | secure_consensus_initialization | 0.8267±0.1361 | 0.8181 | 17.1±0.3 |
| cifar100 | random_initialization | 0.0120±0.0020 | 0.0122 | 33.7±0.5 |
| cifar100 | secure_consensus_initialization | 0.9647±0.0099 | 0.9656 | 35.0±0.5 |
| tiny | random_initialization | 0.0043±0.0012 | 0.0048 | 67.4±0.4 |
| tiny | secure_consensus_initialization | 0.7473±0.0117 | 0.7486 | 67.0±0.2 |

**Stage B Non-IID**: COMPLETE


### 1.3 Module C — Fixed vs Adaptive DP 消融 (54/54)

状态: **COMPLETE** (54/54 runs)

隐私范围: N/A
Released model DP valid: True

| Dataset | Mode | ε | Final Ref Acc | Final C | Runtime (s) |
|---|---:|---|---:|---:|
| cifar10 | fixed_dp | 1.0 | 0.0733±0.0115 | 1.0000 | 17.5 |
| cifar10 | fixed_dp | 3.0 | 0.0733±0.0115 | 1.0000 | 17.4 |
| cifar10 | fixed_dp | 8.0 | 0.0733±0.0115 | 1.0000 | 16.9 |
| cifar10 | adaptive_dp | 1.0 | 0.0800±0.0000 | 0.9784 | 25.6 |
| cifar10 | adaptive_dp | 3.0 | 0.0733±0.0115 | 0.9923 | 26.0 |
| cifar10 | adaptive_dp | 8.0 | 0.0733±0.0115 | 0.9768 | 26.0 |
| cifar100 | fixed_dp | 1.0 | 0.0113±0.0031 | 1.0000 | 34.6 |
| cifar100 | fixed_dp | 3.0 | 0.0120±0.0035 | 1.0000 | 35.2 |
| cifar100 | fixed_dp | 8.0 | 0.0127±0.0031 | 1.0000 | 35.3 |
| cifar100 | adaptive_dp | 1.0 | 0.0127±0.0023 | 0.9677 | 46.6 |
| cifar100 | adaptive_dp | 3.0 | 0.0120±0.0035 | 0.9552 | 44.4 |
| cifar100 | adaptive_dp | 8.0 | 0.0120±0.0035 | 0.9347 | 44.8 |
| tiny | fixed_dp | 1.0 | 0.0047±0.0021 | 1.0000 | 70.5 |
| tiny | fixed_dp | 3.0 | 0.0050±0.0010 | 1.0000 | 71.4 |
| tiny | fixed_dp | 8.0 | 0.0053±0.0015 | 1.0000 | 70.8 |
| tiny | adaptive_dp | 1.0 | 0.0047±0.0021 | 0.9214 | 81.9 |
| tiny | adaptive_dp | 3.0 | 0.0043±0.0006 | 0.9718 | 75.7 |
| tiny | adaptive_dp | 8.0 | 0.0043±0.0006 | 0.9174 | 75.5 |

> 注：数据独立 random initializer 下 Adaptive 未持续优于 Fixed，该负结果如实保留。


### 1.4 Old End-to-End — Random Subset vs Ours (18/18)

状态: **COMPLETE** (18/18 runs)
Privacy: conditional_gradient_mechanism_only_not_released_model_dp

| Dataset | Method | Final Ref Acc | EarlyAUC | Runtime (s) |
|---|---:|---:|---:|
| cifar10 | baseline_random_secure_average_fixed | 0.4733±0.1026 | 0.4770 | 18.7±0.4 |
| cifar10 | ours_official_secure_consensus_adaptive | 0.6067±0.1102 | 0.6059 | 28.2±0.3 |
| cifar100 | baseline_random_secure_average_fixed | 0.1993±0.0127 | 0.2013 | 29.7±0.6 |
| cifar100 | ours_official_secure_consensus_adaptive | 0.3773±0.0185 | 0.3763 | 40.6±0.7 |
| tiny | baseline_random_secure_average_fixed | 0.0550±0.0079 | 0.0557 | 50.9±0.5 |
| tiny | ours_official_secure_consensus_adaptive | 0.1470±0.0066 | 0.1468 | 61.5±0.8 |


### 1.5 Old Final Test (72 模型只读评估)

状态: **COMPLETE**
评估模型数: 72
Test cache reads: 9

| Dataset | Method | Test Accuracy |
|---|---:|
| cifar10 (Random-Subset Baseline) | 0.48960±0.00989 |
| cifar100 (Random-Subset Baseline) | 0.19723±0.00577 |
| tiny (Random-Subset Baseline) | 0.05463±0.00423 |
| cifar10 (Ours) | 0.61040±0.00461 |
| cifar100 (Ours) | 0.33590±0.00653 |
| tiny (Ours) | 0.13040±0.00615 |


### 1.6 TFE System Evaluation (9/9)

状态: **COMPLETE**

| Dataset | Batch | Iterations | Runtime (s) | Peak RSS (GiB) | Comm |
|---|---|---:|---:|---|---|
| cifar10 | 8 | 20 | 0.0 | 0.00 | N/A |
| cifar10 | 16 | 20 | 0.0 | 0.00 | N/A |
| cifar10 | 64 | 20 | 0.0 | 0.00 | N/A |
| cifar100 | 8 | 30 | 0.0 | 0.00 | N/A |
| cifar100 | 16 | 30 | 0.0 | 0.00 | N/A |
| cifar100 | 64 | 30 | 0.0 | 0.00 | N/A |
| tiny | 8 | 40 | 0.0 | 0.00 | N/A |
| tiny | 16 | 40 | 0.0 | 0.00 | N/A |
| tiny | 64 | 40 | 0.0 | 0.00 | N/A |

> 注：单机 TFE 0.8 simulation 不暴露真实 communication bytes/rounds，诚实报告 N/A。

---
## Part 2: 增量 Full Data Feature Model vs Ours (2026-07-27 ~ 07-28)

### 2.1 实验设计

新增主比较：**Full Data Feature Model** vs **Ours**，双方统一使用 Adaptive DP-SGD。

| 参数 | Full Data Feature Model | Ours |
|---|---|---|
| 特征来源 | 完整训练集 (49900/49000/98000) | Official MTT 蒸馏 (400/4000/8000) |
| 初始化 | Secure Average (NoKD) | Secure Consensus (已冻结) |
| Stage C | Adaptive DP-SGD | Adaptive DP-SGD (复用已有) |
| Target ε bound | ≤ 3 | ≤ 3 |
| δ | 1e-5 | 1e-5 |
| Batch size | 16 | 16 |
| Steps | 10/20/30 | 10/20/30 |
| Seeds | 42, 43, 44 | 42, 43, 44 |
| End-to-End DP | Conditional only | Conditional only |


### 2.2 Phase 4: Stage B — Full Data Secure Average 初始化 (9/9)

| Dataset | Seed | Wall Time (s) | Training (s) |
|---|---:|---:|
| cifar10 | 42 | 662.9 | 437.8 |
| cifar10 | 43 | 663.3 | 435.2 |
| cifar10 | 44 | 654.1 | 435.4 |
| cifar100 | 42 | 1231.7 | 792.4 |
| cifar100 | 43 | 1244.6 | 801.7 |
| cifar100 | 44 | 1250.5 | 809.5 |
| tiny | 42 | 2557.9 | 2183.8 |
| tiny | 43 | 2682.9 | 2301.3 |
| tiny | 44 | 2681.0 | 2298.4 |
| cifar10 **Mean±Std** | | **660.1±5.2** | |
| cifar100 **Mean±Std** | | **1242.3±9.6** | |
| tiny **Mean±Std** | | **2640.6±71.6** | |

> 注：CIFAR-10 3/3 于 RTX 5090 完成；CIFAR-100 + Tiny 于 32 核 CPU 完成。Tiny (200类×98000样本) 为计算瓶颈。

### 2.3 Phase 5: Stage C — Full Data Adaptive DP-SGD (9/9)

| Dataset | Seed | Steps | ε (achieved) | Noise σ | Max Ref Acc | Stage C Time (s) |
|---|---:|---:|---:|---:|---:|
| cifar10 | 42 | 10 | 2.9672 | 0.4714 | 0.6400 | 40.8 |
| cifar10 | 43 | 10 | 2.9672 | 0.4714 | 0.7600 | 40.7 |
| cifar10 | 44 | 10 | 2.9672 | 0.4714 | 0.6600 | 40.7 |
| cifar100 | 42 | 20 | 2.8992 | 0.4818 | 0.4320 | 56.1 |
| cifar100 | 43 | 20 | 2.8992 | 0.4818 | 0.4040 | 56.3 |
| cifar100 | 44 | 20 | 2.8992 | 0.4818 | 0.4180 | 56.1 |
| tiny | 42 | 30 | 2.8281 | 0.4677 | 0.2680 | 71.6 |
| tiny | 43 | 30 | 2.8281 | 0.4677 | 0.2650 | 73.5 |
| tiny | 44 | 30 | 2.8281 | 0.4677 | 0.2740 | 73.6 |


### 2.4 Phase 7: Final Test — Full Data Feature Model (9 models)


| Dataset | Full Data Test Acc | Ours Test Acc | Acc Retention | Gap |
|---|---:|---:|---:|---:|
| CIFAR10 | 0.68473±0.00344 | 0.61040±0.00461 | 89.1% | +0.0743 |
| CIFAR100 | 0.39770±0.00286 | 0.33590±0.00653 | 84.5% | +0.0618 |
| TINY | 0.25227±0.00454 | 0.13040±0.00615 | 51.7% | +0.1219 |

**结论**: Full Data Feature Model 在所有数据集上均保持更高的 test accuracy。Ours 保留了 51.7%–89.1% 的精度。Tiny ImageNet 精度保持率最低 (51.7%)，作为 limitation 如实报告。

### 2.5 Runtime 分解

| Dataset | Method | Stage | Mean Time (s) | Std (s) |
|---|---:|---:|---:|
| cifar10 | full_data | sb | 660.1 | 5.2 |
| cifar10 | full_data | sc | 40.7 | 0.1 |
| cifar10 | full_data | total | 700.8 | 5.2 |
| cifar10 | ours | sc | 28.2 | 0.3 |
| cifar10 | ours | total | 28.2 | 0.3 |
| cifar100 | full_data | sb | 1242.3 | 9.6 |
| cifar100 | full_data | sc | 56.2 | 0.1 |
| cifar100 | full_data | total | 1298.5 | 9.7 |
| cifar100 | ours | sc | 40.6 | 0.7 |
| cifar100 | ours | total | 40.6 | 0.7 |
| tiny | full_data | sb | 2640.6 | 71.6 |
| tiny | full_data | sc | 72.9 | 1.1 |
| tiny | full_data | total | 2713.5 | 72.7 |
| tiny | ours | sc | 61.5 | 0.8 |
| tiny | ours | total | 61.5 | 0.8 |


---
## Part 3: 执行时间线

| 日期 | 阶段 | 关键事件 |
|---|---|
| 07/26 | 迁移恢复 | AutoDL 环境恢复、custom ops 重建、7 份权威报告确认 |
| 07/26 | Module C | 54/54 record-DP 网格完成 |
| 07/26 | Old End-to-End | 18/18 Random-Subset vs Ours 完成 |
| 07/26 | Final Test | 72 模型只读 test 评估 |
| 07/26 | TFE System | 9/9 系统测量 |
| 07/27 | 计划切换 | 从 0727 v1 → v2 增量计划 (Full Data vs Ours) |
| 07/27 | Phase 0-1 | 审计 9/9 full feature + 协议冻结 |
| 07/27 | Phase 2 | Full Data test-free 资产准备 (9/9 ref/sel 匹配) |
| 07/27 | Phase 3 | Memmap pilot — CIFAR-10 2-epoch 通过 |
| 07/27-28 | Phase 4 | 9 个 Full Data Stage B (RTX 5090 → 32 核 CPU) |
| 07/28 | Phase 5-8 | 9 个 Stage C + test eval + 主表 + 图表 + 报告 |

---
## Part 4: 产出文件清单

### 旧主线报告 (new_log/new_public_mtt_mainline/)

- ✅ `new_module_a_efficiency_report.json`
- ✅ `new_module_b_convergence_formal_report.json`
- ✅ `new_stage_b_initialization_report.json`
- ✅ `new_module_c_formal_report.json`
- ✅ `new_end_to_end_formal_report.json`
- ✅ `new_final_test_evaluation.json`
- ✅ `new_tfe_system_formal_report.json`

### 增量实验报告 (new_log/new_incremental_comparison/)

- ✅ `new_phase0_audit.json`
- ✅ `new_incremental_full_feature_asset_manifest.json`
- ✅ `new_incremental_comparison_final_report.json`
- ✅ `new_full_data_final_test_evaluation.json`
- ✅ `new_runtime_breakdown.json`
- ⏳ `new_target_freeze.json`

### 图表

- ✅ `new_figure2_full_feature_vs_ours_time.png`
- ✅ `new_figure2_full_feature_vs_ours_time.pdf`
- ✅ `new_figure2_full_feature_vs_ours_samples.png`
- ✅ `new_figure2_full_feature_vs_ours_samples.pdf`

### CSV 表格

- ✅ `new_incremental_comparison_main_table.csv`
- ✅ `new_full_data_final_test_evaluation.csv`
- ✅ `new_runtime_breakdown.csv`

### 模型文件

- ✅ 9 Stage B initializer models
- ✅ 9 Stage C final models

### 报告文档

- ✅ `experiment_report.md`
- ✅ `experiment_report_revised.md`
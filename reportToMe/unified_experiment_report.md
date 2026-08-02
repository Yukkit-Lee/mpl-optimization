# Unified Experiment Report

更新时间：2026-08-02（UTC）
权威分支：`unified-experiments`
报告生成提交：`704755d70d2475a6a306a1b09e4e5e65195baf6c`

本报告是仓库当前唯一权威的跨实验总结。所有数字均由已跟踪的 CSV/JSON
重新读取并通过 cardinality、状态、数据边界、SHA256 与 Git lineage 审计；
本次没有启动训练、TFE 或任何正式实验。

## 1. 总结结论

1. Official MTT 在三个数据集上均显著优于等规模随机子集，但样本比例并不都是1%。
2. True Ours 的最新公平主比较为14/14 development-validation结果：三个数据集均更快达到T@90；CIFAR-100/Tiny最终development accuracy更高，CIFAR-10低4.29pp。
3. 当前最可信的正Stage-B扩展是 Secure Average 后接有标签Official-MTT anchor CE；正式TFE复现了CIFAR-10/100 Severe方向。
4. KD/Consensus的表面增益主要来自anchor标签监督；class-aware KD没有独立正贡献。
5. disagreement weighting/top-selection与heterogeneity-trigger routing均为完成的负结果，不能包装为正模块。
6. Stage-C epsilon只覆盖条件式Stage-C机制；数据依赖的Stage-B initializer非record-level DP，因此不能声称端到端DP。
7. 当前权威True Ours与Post-Aggregation均未读取official test或locked holdout；历史final-test数字不进入当前主结论。

## 2. Stage A：Official MTT样本效率

| Dataset | Full samples | Compact samples | Compact ratio | Full | Random subset | Official MTT | MTT−Random |
|---|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | 50,000 | 480 | 0.96% | 69.42% ± 0.47% | 60.84% ± 0.52% | 67.67% ± 0.87% | +6.83pp |
| cifar100 | 50,000 | 4,800 | 9.60% | 48.72% ± 0.46% | 38.97% ± 0.37% | 43.76% ± 0.55% | +4.79pp |
| tiny | 100,000 | 9,600 | 9.60% | 31.71% ± 0.50% | 22.57% ± 0.23% | 26.15% ± 0.61% | +3.57pp |

结论：MTT相对随机子集提升6.83/4.79/3.57pp；CIFAR-100与Tiny的compact ratio是9.6%，不是1%。

## 3. True Ours vs Full Data：当前主比较

状态：`COMPLETE_DEVELOPMENT_ONLY_14_OF_14`。CIFAR-10/100各3 seeds；Tiny按用户批准仅seed42。

| Dataset | Method | n | Final development accuracy | Total online secure time | T@90 |
|---|---|---:|---:|---:|---:|
| cifar10 | full_data | 3 | 72.67% ± 0.29% | 945.34 ± 36.70 s | 136.69 ± 4.02 s |
| cifar10 | true_ours | 3 | 68.37% ± 0.42% | 219.30 ± 23.38 s | 53.83 ± 17.12 s |
| cifar100 | full_data | 3 | 43.21% ± 0.31% | 3536.64 ± 125.19 s | 466.87 ± 21.11 s |
| cifar100 | true_ours | 3 | 48.61% ± 0.47% | 2463.02 ± 18.63 s | 184.34 ± 1.15 s |
| tiny | full_data | 1 | 27.20% (n=1) | 6087.04 s (n=1) | 1491.45 s (n=1) |
| tiny | true_ours | 1 | 31.41% (n=1) | 9257.16 s (n=1) | 607.48 s (n=1) |

客观解释：

- CIFAR-10：True Ours最终accuracy低约4.29pp，但总时长快约4.31×，T@90快约2.54×。
- CIFAR-100：True Ours高约5.40pp，总时长快约1.44×，T@90快约2.53×。
- Tiny：True Ours高4.21pp且T@90快约2.46×，但完整运行总时长慢约1.52×；只有n=1。
- 安全表述是“更快达到冻结目标”，不能写成“所有数据集最终都更快/更准”。

## 4. 最新正Stage-B：Post-Aggregation Anchor CE

范围：CIFAR-10/100、seed42、development-only、Official-MTT-only；40 local + 8 anchor/class；无Full initializer、test或locked holdout。

| Dataset | Dist. | C1 Random+Anchor | C3 Local-0+Anchor | C4 Average | C5 Average+Anchor | C5−C1 | C5−C3 | C5−C4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | iid | 55.48% | 57.58% | 22.70% | 57.62% | +2.14pp | +0.04pp | +34.92pp |
| cifar10 | mild | 55.48% | 22.58% | 22.02% | 55.78% | +0.30pp | +33.20pp | +33.76pp |
| cifar10 | severe | 55.48% | 21.48% | 18.84% | 57.64% | +2.16pp | +36.16pp | +38.80pp |
| cifar100 | iid | 10.18% | 21.98% | 10.24% | 22.12% | +11.94pp | +0.14pp | +11.88pp |
| cifar100 | mild | 10.18% | 3.32% | 9.06% | 20.34% | +10.16pp | +17.02pp | +11.28pp |
| cifar100 | severe | 10.18% | 6.26% | 8.08% | 17.62% | +7.44pp | +11.36pp | +9.54pp |

### 正式TFE/ABY3 Severe复现

| Dataset | TFE C1 | TFE C4 | TFE C5 | C5−C1 | Gain retention | Max accuracy gap | Secure compute | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | 55.48% | 18.84% | 57.62% | +2.14pp | 99.1% | 0.02pp | 87.37s | 2.95GiB |
| cifar100 | 10.02% | 8.02% | 17.50% | +7.48pp | 100.5% | 0.16pp | 862.18s | 8.97GiB |

结论：Secure Average对anchor CE有独立贡献，尤其在Mild/Severe下相对single-client明显；IID下C5与合法Local-0+Anchor仅近似持平，因此应定位为Non-IID扩展。

## 5. Stage-B因果与负结果

### 5.1 KD/Consensus因果分解（seed42）

| Dataset Severe | Average | Anchor CE-only | Uniform CE+KD | Class-aware CE+KD | KD−CE-only | Class-aware−Uniform |
|---|---:|---:|---:|---:|---:|---:|
| cifar10 | 19.24% | 54.70% | 55.14% | 55.14% | +0.44pp | +0.00pp |
| cifar100 | 8.08% | 27.00% | 23.46% | 23.44% | -3.54pp | -0.02pp |

结论：anchor标签监督有效；KD在CIFAR-10仅+0.44pp、CIFAR-100为−3.54pp，class-aware相对uniform为0/−0.02pp。KD/Consensus不是独立正贡献。

### 5.2 Disagreement-guided refinement

| Dataset Severe | Uniform anchor CE | Weighted CE | Top-50% CE | Best informative AUROC |
|---|---:|---:|---:|---:|
| cifar10 | 57.70% | 10.00% | 22.48% | 0.2243 |
| cifar100 | 17.62% | 2.92% | 10.94% | 0.4340 |

结论：分歧分数与错误方向相反，效用和成本门均失败；按计划未启动TFE。

### 5.3 Heterogeneity-trigger routing

H1--H3能测IID→Mild→Severe标签偏斜，但uniform anchor CE在六场均提升B0。触发器在IID关闭CE会损失CIFAR-10 34.98pp、CIFAR-100 11.88pp；mean/max oracle regret为7.81/34.98pp。因此是完成的负结果，未启动TFE。

## 6. Stage C、旧End-to-End与系统证据

### 6.1 Fixed vs Adaptive DP（历史消融）

| Dataset, ε=3 | Fixed accuracy | Adaptive accuracy | Fixed runtime | Adaptive runtime |
|---|---:|---:|---:|---:|
| cifar10 | 7.33% ± 1.15% | 7.33% ± 1.15% | 17.39s | 25.99s |
| cifar100 | 1.20% ± 0.35% | 1.20% ± 0.35% | 35.22s | 44.36s |
| tiny | 0.50% ± 0.10% | 0.43% ± 0.06% | 71.36s | 75.72s |

该历史消融中Adaptive没有持续效用优势，适合作为负/机制消融。当前True Ours Stage C使用Adaptive，但epsilon只对条件式Stage-C机制成立。

### 6.2 历史End-to-End

早期18/18 Random-subset Fixed vs Official-MTT Consensus Adaptive结果可以作为历史补充证据，但协议已被当前Full Data vs True Ours公平主比较取代，不应进入当前主表。

### 6.3 TFE系统表

旧`new_tfe_system_formal.csv`的runtime/RSS为0占位值，不是可发表系统测量。当前可用的真实系统数据来自True Ours timeline与Post-Aggregation TFE runtime/resource表；通信字节与轮数仍为N/A。

## 7. 当前论文就绪度

| Evidence | Status | Safe use |
|---|---|---|
| Stage A Official MTT efficiency | READY | 主文/补充；使用正确样本数与比例 |
| True Ours vs Full Data 14/14 | READY_WITH_SCOPE | 可作主结果，但必须标development validation；Tiny n=1 |
| Post-Aggregation Anchor CE | READY_WITH_SCOPE | 正Stage-B/Non-IID扩展；seed42、C10/C100 |
| Real TFE/ABY3 correctness/runtime | READY | 系统可行性；通信N/A |
| KD/Consensus causality | COMPLETE_NEGATIVE | 负消融；不可声称独立贡献 |
| Disagreement refinement | COMPLETE_NEGATIVE | 负消融；TFE按门控跳过 |
| Heterogeneity trigger | COMPLETE_NEGATIVE | 负消融；TFE按门控跳过 |
| Fixed vs Adaptive | SUPPLEMENTARY_NEGATIVE | Adaptive无持续显著优势 |
| Current official-test main table | NOT_AVAILABLE | 当前权威路线未读取official test |
| End-to-end record-level DP | NOT_SUPPORTED | Stage-B initializer数据依赖且非DP |
| Communication reduction | NOT_MEASURED | 不得声称 |

## 8. 安全论文表述

可以写：Official-MTT-only路线以更少compact样本提供高信息密度；真实TFE/ABY3复现安全训练；Secure Average作为跨方中间初始化，再用类别均衡的标签anchor监督修正，在受控seed42消融中优于Random+Anchor，并在异质场景优于single-client+Anchor。

不可以写：所有数据集最终都更快/更准、当前数字是official-test、整个发布模型满足epsilon=3端到端DP、class-aware KD/Consensus是正贡献、trigger能安全决定是否refine、或通信成本下降。

## 9. 权威文件

- True Ours：`new_log/new_true_ours/new_true_ours_master_results.json`
- Figure 2：`new_expFig/new_true_ours/new_figure2_pea_style_secure_time.{png,pdf,svg}`
- Stage A：`new_expTab/new_public_mtt_mainline/new_module_a_efficiency_summary.csv`
- Post-Aggregation：`new_log/new_stage_b_anchor_postagg/new_stage_b_anchor_postagg_final_report.md`
- Post-Aggregation审计：`new_log/new_stage_b_anchor_postagg/new_stage_b_anchor_postagg_completion_audit.json`
- 统一machine status：`new_log/unified_experiments/unified_artifact_status.json`
- 统一数据审计：`new_log/unified_experiments/unified_data_audit.json`

## 10. 剩余缺口

- 若投稿主表必须报告official test，需要另行设计一次冻结后、可审计的最终读取；本次未运行。
- Post-Aggregation正结果目前只有seed42；若要声称跨seed稳定，需未来补seed43/44。
- 外部方法PEA/CaPC/DDGauss精确对比仍需从原论文/官方仓库核对。
- 通信量没有测量，系统表只能报告实际secure runtime和RSS。

# 全部实验结果、证据边界与改进建议

更新时间：2026-07-29 12:57（UTC）

状态说明：最新 `new_stage_b_kd` seed42 单种子任务已完成；tiny synthetic、
CIFAR-10 和 CIFAR-100 的真实 TFE/ABY3 梯级全部通过。本文仍严格区分
“安全复现了 CE+KD 路径”和“class-aware KD 是否有独立因果贡献”。

## 0. 先给结论

当前仓库已经形成一条可解释、可复核的主实验链：

1. Official MTT 蒸馏数据确实能用远少于完整训练集的样本保留较高效用，并显著优于同样本量的随机子集。
2. TFE/ABY3 安全训练、固定点计算、Adaptive Stage C、断点恢复、谱系检查和系统资源测量均已有真实运行证据。
3. 最新且最可信的 True Ours 对比已经完成 14/14：CIFAR-10/100 各 3 seeds，Tiny-ImageNet 按已批准策略只跑 seed42。
4. 最新结果中，True Ours 在 CIFAR-100 和 Tiny-ImageNet 的开发集精度高于 Full Data，同时更快达到冻结目标；CIFAR-10 更快，但最终开发集精度低 4.29 个百分点。
5. 当前最强的效率结论是“更快达到相同冻结效用目标”，而不是“三个数据集完整训练都更快”：Tiny 上 True Ours 达到 T@90 快 2.46 倍，但跑到双方各自最终模型时反而慢 1.52 倍。
6. 当前正式结果全部是 development validation，不是新的 official-test 结果。历史 official-test 数值可以作为旧协议结果保留，但不能替代当前 True Ours 的冻结后最终评估。
7. 当前 epsilon 只描述条件式 Stage-C 梯度机制。Stage-B initializer 是数据依赖且非 DP，因此不能声称端到端 record-level DP。
8. 最新 MTT-anchor 实验进一步确认“有标签的蒸馏 anchor 数据有效”，但没有确认
   “class-aware Consensus/KD 有独立正贡献”：CIFAR-10 的 KD 在 anchor CE 之上
   仅增加 0.44pp，CIFAR-100 反而降低 3.56pp。

因此，现有材料已经足以支撑论文中的方法可行性、压缩/效用、TFE 实现和达到目标精度的效率论证；但要把最新 True Ours 写成最终主结果，仍应补齐可信的冻结后测试集评估或明确把主表标为开发集结果。

## 1. 统一实验口径

### 1.1 数据、模型和安全后端

- 数据集：CIFAR-10、CIFAR-100、Tiny-ImageNet。
- 特征维度：2048。
- 客户端数：5。
- 安全后端：TFE/ABY3，正式路径禁止 plaintext fallback。
- Stage B：在安全计算中训练并安全平均客户端线性头，当前 True Ours 只使用 Official MTT 特征。
- Stage C：Adaptive DP-SGD，正式目标为 `epsilon <= 3, delta = 1e-5`，采用 Poisson sampling、OS CSPRNG 和动态裁剪阈值。
- 当前正式计时：从 Stage-B secure graph/session 开始，到 Stage-C 最终模型结束，指标为累计 online secure seconds。

Official MTT 资产审计结果：

| 数据集 | Official MTT 样本 | 每类样本 | 相对原训练集比例 | 类别平衡 |
|---|---:|---:|---:|---|
| CIFAR-10 | 480 | 48 | 0.96% | 是 |
| CIFAR-100 | 4,800 | 48 | 9.60% | 是 |
| Tiny-ImageNet | 9,600 | 48 | 9.60% | 是 |

资产表：`new_expTab/new_true_ours/new_mtt_class_distribution.csv`

### 1.2 三种必须分开的结果类型

| 类型 | 含义 | 论文用途 |
|---|---|---|
| 当前正式 True Ours | 严格 Official-MTT-only 谱系、统一 Adaptive 协议、累计安全时间 | 当前主结果候选 |
| 历史 Public-MTT 主线 | 早期 Stage A/B/C、端到端、final test 和系统实验 | 机制证据、补充材料、历史对照 |
| 诊断/失效结果 | 混合 Full-Data initializer、旧 Tiny 错误配置、来源未验证的 direct evaluation | 只用于解释问题，不进入主结果 |

后文所有数值都注明属于哪一类，避免把不同协议拼成一张“看起来更好”的表。

## 2. 当前正式主实验：True Ours vs Full Data

### 2.1 实验想证明什么

在双方都使用真实 TFE/ABY3 Stage B、相同 Adaptive Stage C、相同隐私目标和相同计时边界时，比较：

- True Ours 是否只依赖 Official MTT，而不偷用 Full-Data initializer；
- 最终开发集效用；
- 从 Stage B 开始到 Stage C 结束的累计在线安全时间；
- 达到 Full Data 冻结平台的 80%、90%、95% 所需时间；
- 三数据集上的结论是否一致。

### 2.2 正式结果

状态：`COMPLETE_DEVELOPMENT_ONLY_14_OF_14`

| 数据集 | 方法 | n | 最终开发集精度 | 累计在线安全时间 | T@90 |
|---|---|---:|---:|---:|---:|
| CIFAR-10 | Full Data | 3 | 72.67% ± 0.29% | 945.34 ± 36.70 s | 136.69 ± 4.02 s |
| CIFAR-10 | True Ours | 3 | 68.37% ± 0.42% | 219.30 ± 23.38 s | 53.83 ± 17.12 s |
| CIFAR-100 | Full Data | 3 | 43.21% ± 0.31% | 3536.64 ± 125.19 s | 466.87 ± 21.11 s |
| CIFAR-100 | True Ours | 3 | 48.61% ± 0.47% | 2463.02 ± 18.63 s | 184.34 ± 1.15 s |
| Tiny-ImageNet | Full Data | 1 | 27.20% | 6087.04 s | 1491.45 s |
| Tiny-ImageNet | True Ours | 1 | 31.41% | 9257.16 s | 607.48 s |

Tiny 只有 seed42，所以不能报告方差或显著性。

### 2.3 结果如何解释

| 数据集 | 精度差（True−Full） | 完整流程耗时 | T@90 效率 | 客观结论 |
|---|---:|---:|---:|---|
| CIFAR-10 | −4.29 pp | True 快 4.31× | True 快 2.54× | 效率优势清楚，但有明显效用缺口 |
| CIFAR-100 | +5.40 pp | True 快 1.44× | True 快 2.53× | 效用和目标到达效率均占优 |
| Tiny-ImageNet | +4.21 pp | True 慢 1.52× | True 快 2.46× | 更早达到目标，但继续训练到最终点不划算 |

这组结果支持的最稳健表述是：

> Official-MTT-only True Ours 在三个数据集上均更快达到冻结的 90% 效用目标；它在 CIFAR-100 和 Tiny-ImageNet 上还获得更高的最终开发集精度。CIFAR-10 保留约 4.29pp 的效用缺口，Tiny 的完整训练总时长不优于 Full Data。

不应写成：

- “True Ours 在所有数据集上最终训练都更快”；
- “True Ours 在所有数据集上精度都更高”；
- “这是最终测试集精度”；
- “整个发布模型满足 epsilon=3 的端到端 DP”。

### 2.4 对应图表和原始结果

- 主 Figure 2：
  - `new_expFig/new_true_ours/new_figure2_pea_style_secure_time.png`
  - `new_expFig/new_true_ours/new_figure2_pea_style_secure_time.pdf`
  - `new_expFig/new_true_ours/new_figure2_pea_style_secure_time.svg`
- T@80/90/95 明细：`new_expTab/new_true_ours/new_figure2_time_to_target_table.csv`
- 单一权威结果：`new_log/new_true_ours/new_true_ours_master_results.json`
- 14/14 验证：`new_log/new_true_ours/new_true_ours_minimal_validation.json`
- 简化报告：`reportToMe/new_true_ours_minimal_correction_report.md`

主 Figure 2 是目前唯一适合代表最新 True Ours 时间—效用关系的图。

## 3. Stage A：Public MTT 的样本效率

### 3.1 实验想证明什么

比较完整训练集、等样本量随机子集和 Official MTT，判断蒸馏数据是否：

- 比随机抽样更有信息密度；
- 用更少样本接近完整数据的效用；
- 更快达到预设精度目标。

### 3.2 历史正式结果

| 数据集 | Full Images | Random Subset | Official MTT | MTT 相对随机提升 |
|---|---:|---:|---:|---:|
| CIFAR-10 | 69.42% ± 0.47% | 60.84% ± 0.52% | 67.67% ± 0.87% | +6.83 pp |
| CIFAR-100 | 48.72% ± 0.46% | 38.97% ± 0.37% | 43.76% ± 0.55% | +4.79 pp |
| Tiny-ImageNet | 31.71% ± 0.50% | 22.57% ± 0.23% | 26.15% ± 0.61% | +3.58 pp |

结论：Official MTT 在三个数据集上都显著优于等规模随机子集，但仍未在此组历史实验中完全达到 Full Images。

原始汇总：`new_expTab/new_public_mtt_mainline/new_module_a_efficiency_summary.csv`

对应图：

- `new_expFig/new_public_mtt_mainline/new_figureA1_accuracy_vs_steps.png/.pdf`
- `new_expFig/new_public_mtt_mainline/new_figureA2_sample_efficiency.png/.pdf`

论文用途：可以作为数据蒸馏有效性的补充证据。使用时必须采用已审计的样本数 480/4800/9600，不能沿用早期报告中“三个数据集均为 480、均为 1%”的错误标签。

## 4. 历史 Stage B：安全初始化与 Non-IID

### 4.1 收敛/初始化实验

早期 Module B convergence 共 18/18 完成，比较随机初始化与 Secure Consensus 初始化。其 `reference accuracy` 大致为：

| 数据集 | Random Init | Secure Consensus |
|---|---:|---:|
| CIFAR-10 | 8.00% ± 0.00% | 82.67% ± 13.61% |
| CIFAR-100 | 1.20% ± 0.20% | 96.47% ± 0.99% |
| Tiny-ImageNet | 0.43% ± 0.12% | 74.73% ± 1.17% |

这证明安全生成的初始化器确实包含可用训练信号，而不是随机头；但这里的 `reference accuracy` 不是下游 development/test accuracy，不能与当前 Figure 2 的最终精度横向比较。

### 4.2 Non-IID 初始化实验

历史 27/27 实验比较 NoKD、KD-Avg、KD-ReferenceWeighted，在 IID、mild 和 severe Non-IID 下评估初始准确率。

关键 severe Non-IID 结果：

| 数据集 | NoKD | KD-Avg | KD-ReferenceWeighted |
|---|---:|---:|---:|
| CIFAR-10 | 12.65% ± 4.58% | 19.11% ± 8.99% | 40.48% ± 3.90% |
| CIFAR-100 | 25.06% ± 1.32% | 31.55% ± 0.14% | 31.70% ± 0.24% |
| Tiny-ImageNet | 8.21% ± 0.61% | 12.29% ± 0.77% | 12.30% ± 0.74% |

它支持“KD/参考加权在早期严重 Non-IID 设置下能改善初始化”，尤其 CIFAR-10 最明显。

但最新严格 True Ours 复核发现：同步后的客户端头完全相同，后聚合 Consensus 权重无法产生作用；取消同步虽恢复了多样性，却没有候选通过预设门槛。因此这组旧 Non-IID 正结果只能作为特定历史协议下的机制现象，不能用来证明当前主路线中的 Consensus 必然有效。

原始表：

- `new_expTab/new_public_mtt_mainline/new_stage_b_initialization_runs.csv`
- `new_expTab/new_public_mtt_mainline/new_stage_b_initialization_summary.csv`

对应图：

- `new_expFig/new_public_mtt_mainline/new_figureB2_stage_b_noniid_initialization.png/.pdf`

## 5. 最新 Stage B 重建与消融

### 5.1 Official-MTT-only 明文筛选与真实 TFE finalist

严格 seed42 finalist：

| 数据集 | Plaintext Stage B | TFE Stage B | 公平 Full 开发集 | 结论 |
|---|---:|---:|---:|---|
| CIFAR-10 | 69.52% | 68.76% | 约 72% | 固定点损失 0.76pp，仍有约 3–4pp 总缺口 |
| CIFAR-100 | 48.04% | 48.00% | 43.08% | TFE 几乎复现明文，且高于 Full |
| Tiny-ImageNet | 31.14% | 31.48% | 27.65% | TFE 无效用损失，且高于 Full |

证明了：

- Official-MTT-only 谱系可以在真实 TFE 中工作；
- CIFAR-100/Tiny 的优势不是 plaintext-only 假象；
- CIFAR-10 的主要问题不是 TFE 固定点误差，因为 TFE 与明文只差 0.76pp，剩余差距来自数据/训练方案本身。

### 5.2 固定点精度 16→20

- precision 16：CIFAR-10 TFE development accuracy 68.76%。
- precision 20：仍为 68.76%。
- runtime 从 458.55s 附近变为 464.97s，未带来精度收益。

结论：提高固定点小数位不是修复 CIFAR-10 缺口的有效方向，正式配置保留 precision 16。

### 5.3 `secure_average_only` 图优化

在 CIFAR-10 40 epochs 的真实 TFE 等价复核中：

- 新旧 development accuracy 均为 68.76%；
- weight/bias 最大误差分别为 `4.12e-4/4.58e-5`；
- 总 runtime 从 458.55s 降为 93.89s；
- 加速 4.884×。

这是一项很强的系统优化证据：删除未使用的 teacher、KD、reference weighting 和 private selection 子图后，输出保持等价，运行时间显著下降。

### 5.4 Consensus 严格负结果

- 同步客户端头：所有头相同，Consensus 退化为 Average。
- no-sync follow-up：恢复客户端多样性，但三个数据集均 0 个候选通过门槛。
- 相对原 synced Average，最佳 severe delta 仍为：
  - CIFAR-10：−4.62pp
  - CIFAR-100：−4.08pp
  - Tiny-ImageNet：−3.37pp

结论：当前主线使用 Secure Average 是数据支持的选择；Consensus 应作为负消融如实报告，不应继续为追求正结果而在 development 上反复调参。

相关筛选表：

- `new_expTab/new_true_ours/new_true_ours_stage_b_screen.csv`
- `new_expTab/new_true_ours/new_true_ours_stage_b_fedavg_screen.csv`

### 5.5 最新 MTT-anchor KD/Consensus 因果实验（seed42）

这轮新实验与上面的旧 Consensus 筛选不同：它从每类 48 个 Official MTT
样本中严格拆出 local pool 和有标签 anchor pool，在 IID、mild 和 severe
Non-IID 下比较 Secure Average、teacher、KD student，并额外做
`anchor CE-only / KD-only / CE+KD` 因果分解。CIFAR-10/100 的
local/anchor、development split 均互斥；未读取 locked holdout 或 official test，
也未加载 Full-Data initializer。

明文 S1--S6 共登记 89 个候选。机械筛选的 severe Non-IID 结果为：

| 数据集 | Secure Average | Teacher | CE+KD Student | Student 相对 Average |
|---|---:|---:|---:|---:|
| CIFAR-10 | 19.24% | 19.40% | 55.14% | +35.90pp |
| CIFAR-100 | 8.08% | 9.32% | 23.44% | +15.36pp |

只看这张表会误以为 KD/Consensus 非常有效，因此又固定同一配置完成了
36/36 因果消融：

| 数据集（severe） | Anchor CE-only | Uniform CE+KD | Class-aware CE+KD | KD 相对 CE-only | Class-aware 相对 Uniform |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 54.70% | 55.14% | 55.14% | +0.44pp | 0.00pp |
| CIFAR-100 | 27.00% | 23.46% | 23.44% | −3.56pp | −0.02pp |

IID/mild 结论一致：CIFAR-10 的 CE-only 已达到 54.48%/54.40%，CE+KD
仅为 54.86%/54.40%；CIFAR-100 的 CE-only 为 29.78%/29.04%，高于
CE+KD 的 27.62%/26.60%。KD-only 在两数据集上均没有超过 Average。

因此这组实验实际证明的是：

- **强支持**：Official MTT anchor 的有标签监督能显著改善 Stage-B initializer；
- **不支持**：当前 class-aware teacher 相对 uniform teacher 有独立收益；
- **不支持**：KD 是主要增益来源；CIFAR-100 上 KD 还会损害 CE-only 结果；
- 当前只有 seed42，不能报告跨 seed 稳定性或显著性。

真实 TFE/ABY3 梯级最终结果：

- tiny synthetic 已通过：TFE 与 plaintext 的 Average weight/bias、teacher、
  student weight/bias 最大绝对误差均小于 `2.8e-5`，无 plaintext fallback；
- secure runtime 为 25.80s，peak RSS 为 1.53GiB；
- CIFAR-10 TFE Average/student=`19.18/55.16%`，gain=`+35.98pp`，
  保留 plaintext gain `100.22%`；secure runtime=`121.13s`，peak RSS=`4.72GiB`；
- CIFAR-100 TFE Average/student=`7.98/23.34%`，gain=`+15.36pp`，
  保留 plaintext gain `100.00%`；secure runtime=`731.61s`，peak RSS=`18.83GiB`；
- C10/C100 输出 finite、无 fallback、方向/utility/lineage 门均通过，P1/TFE
  最大参数绝对误差分别约 `2.90e-4/3.80e-4`；
- 本任务严格停留在 Stage B，没有运行 Stage C、DP 或 end-to-end。

权威结果位置：

- 89-run registry：`new_expTab/new_stage_b_kd/new_stage_b_kd_smoke_registry.csv`
- 明文确认表：`new_expTab/new_stage_b_kd/new_stage_b_kd_plaintext_confirmation.csv`
- 36-run 因果表：`new_expTab/new_stage_b_kd/new_stage_b_kd_anchor_supervision_ablation.csv`
- 因果 JSON：`new_log/new_stage_b_kd/new_stage_b_kd_anchor_supervision_ablation.json`
- split 审计：`new_log/new_stage_b_kd/new_stage_b_kd_split_audit.md`
- 当前 TFE 验证：`new_expTab/new_stage_b_kd/new_stage_b_kd_tfe_validation.csv`

对应图已经生成于 `new_expFig/new_stage_b_kd/`：Figure B1 展示 Non-IID
initializer quality，B2 明确分开 anchor CE 与 KD-over-CE，B3 展示明文和
TFE runtime；S1--S5 覆盖 anchor/temperature sweep、P1/TFE 数值误差、
class-weight 分布和逐类别变化。逐类别检查显示 C10 9/10、C100 93/100
类别改善，因此平均增益并非只来自单一多数类。

## 6. Stage C：Fixed 与 Adaptive、裁剪和隐私会计

### 6.1 历史 54-run Fixed vs Adaptive

早期 Module C 对三个数据集、三个 epsilon 和 Fixed/Adaptive 做了 54/54 个安全运行。随机初始化下的最终准确率整体接近随机猜测：

- CIFAR-10：约 7%–9%；
- CIFAR-100：约 1%；
- Tiny-ImageNet：约 0.4%–0.5%。

Adaptive 的运行时间通常略高于 Fixed，但没有表现出稳定的效用优势。这组实验最重要的价值不是“Adaptive 更准”，而是：

- 真实 Adaptive 裁剪轨迹和 composed accountant 路径能运行；
- epsilon 1/3/8 的机制比较能够完成；
- 随机初始化是效用瓶颈，说明高质量 Stage B initializer 对端到端结果是必要的。

对应图：

- `new_expFig/new_public_mtt_mainline/new_figureC_stage_c_privacy_utility.png/.pdf`
- `new_expFig/new_public_mtt_mainline/new_figureC_stage_c_clipping_trajectory.png/.pdf`

原始表：`new_expTab/new_public_mtt_mainline/new_module_c_formal_summary.csv`

### 6.2 后续 Fixed/Adaptive 小范围筛选

CIFAR-10 上：

- 最佳 Fixed（C=2 或 4）：69.82%；
- 最佳 Adaptive（C_init=2）：69.86%；
- 差距仅 +0.04pp。

这证明 Adaptive 能与调好的 Fixed 持平，但不能证明 Adaptive 显著优于 Fixed。用户后续确定主线只做 Adaptive，因此这组历史 Fixed 实验应保留为消融，不再扩大。

表：`new_expTab/new_end2end_adaptive_followup/new_fixed_vs_adaptive_screen.csv`

图：

- `new_expFig/new_end2end_adaptive_followup/new_fixed_vs_adaptive_validation.png/.pdf`
- `new_expFig/new_end2end_adaptive_followup/new_fixed_vs_adaptive_clipping.png/.pdf`

### 6.3 最新 Adaptive 配置筛选

development-only 配对筛选最终选出：

| 数据集 | Steps | LR | 初始 C | gamma | eta |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 80 | 0.002 | 1 | 0.5 | 0.02 |
| CIFAR-100 | 20 | 0.01 | 2 | 0.1 | 0.05 |
| Tiny-ImageNet | 180 | 0.002 | 1 | 0.5 | 0.1 |

真实安全 finalist 的 epsilon、C 轨迹、Poisson/CSPRNG、finite、无 fallback 等门均通过。筛选阶段 Full initializer 曾使用 metadata-complete plaintext wrapper 以避免重复高成本 Stage B；这些 finalist 只验证 Stage C 安全图，未进入正式计时。最终 Figure 2 已改用双方真实 TFE Stage B。

## 7. 历史端到端实验

### 7.1 Random+Average+Fixed vs Official-MTT+Consensus+Adaptive

历史 18/18 开发/参考实验：

| 数据集 | 历史 Baseline | 历史 Ours | 提升 | Runtime Baseline→Ours |
|---|---:|---:|---:|---:|
| CIFAR-10 | 47.33% ± 10.26% | 60.67% ± 11.02% | +13.33pp | 18.66→28.23s |
| CIFAR-100 | 19.93% ± 1.27% | 37.73% ± 1.85% | +17.80pp | 29.69→40.59s |
| Tiny-ImageNet | 5.50% ± 0.79% | 14.70% ± 0.66% | +9.20pp | 50.90→61.54s |

历史只读 final test：

| 数据集 | Random Baseline | 历史 Ours |
|---|---:|---:|
| CIFAR-10 | 48.96% ± 0.99% | 61.04% ± 0.46% |
| CIFAR-100 | 19.72% ± 0.58% | 33.59% ± 0.65% |
| Tiny-ImageNet | 5.46% ± 0.42% | 13.04% ± 0.62% |

它证明“蒸馏数据+更好的初始化/训练组合”优于弱随机基线，但 baseline 与 Ours 同时改变了数据、初始化和 DP 策略，所以不能从这张表单独归因“提升来自 Adaptive”或“提升来自 Consensus”。

表：

- `new_expTab/new_public_mtt_mainline/new_end_to_end_formal_summary.csv`
- `new_expTab/new_public_mtt_mainline/new_final_test_evaluation.csv`

图：

- `new_expFig/new_public_mtt_mainline/new_figureE_end_to_end_two_curve.png/.pdf`
- `new_expFig/new_public_mtt_mainline/new_figureF_final_test_privacy_utility.png/.pdf`

这些结果适合补充材料或方法演化叙述，不应取代最新 True Ours。

## 8. Full Data 增量对比与后续准确率诊断

### 8.1 旧 Full Data vs 旧 Ours

历史 test accuracy：

| 数据集 | Full Data | 旧 Ours |
|---|---:|---:|
| CIFAR-10 | 68.47% ± 0.34% | 61.04% ± 0.46% |
| CIFAR-100 | 39.77% ± 0.29% | 33.59% ± 0.65% |
| Tiny-ImageNet | 25.23% ± 0.45% | 13.04% ± 0.62% |

旧表中 Full Data 总时间约为 700.8/1298.5/2713.5s，而旧 Ours 表只计了 Stage C，Stage B 为 `N/A`。因此这些旧“总时间加速倍数”不是公平端到端计时，不能用于主结论。最新 Figure 2 已通过统一 Stage-B-start→Stage-C-model 边界修复这一问题。

表：

- `new_expTab/new_incremental_comparison/new_incremental_comparison_main_table.csv`
- `new_expTab/new_incremental_comparison/new_runtime_breakdown.csv`
- `new_expTab/new_incremental_comparison/new_full_data_final_test_evaluation.csv`

旧图：

- `new_expFig/new_incremental_comparison/new_figure2_full_feature_vs_ours_time.png/.pdf`
- `new_expFig/new_incremental_comparison/new_figure2_full_feature_vs_ours_samples.png/.pdf`

只适合历史诊断，不用于当前主图。

### 8.2 D1–D6 准确率诊断

CIFAR-10 seed42 的关键结果：

| 组合 | 最终验证精度 |
|---|---:|
| Full + Average + NoDP | 69.76% |
| MTT + Average + NoDP | 69.74% |
| MTT + Consensus + NoDP | 62.57% |
| Full + Average + DP | 69.76% |
| MTT + Consensus + DP | 62.44% |
| MTT + Average + DP | 69.68% |

这组诊断表明，在该历史 setup 中：

- 换成 MTT 本身只损失约 0.02pp；
- Consensus 带来约 7.17pp 的主要损失；
- DP 的额外损失很小。

但后续审计确认其中高精度的 “MTT+Average/Improved Ours” 使用了 Full-Data Stage B initializer，再用 MTT 做 Stage C，是 hybrid ablation，不是严格 True Ours。因此它只能用于定位旧问题，不能作为主方法精度。

对应图：

- `new_expFig/new_end2end_adaptive_followup/new_improved_ours_vs_full_data_test.png/.pdf`
- `new_expFig/new_end2end_adaptive_followup/new_stage_b_rescue_validation.png/.pdf`

表：

- `new_expTab/new_end2end_adaptive_followup/new_stage_b_rescue_screen.csv`

## 9. TFE 系统性能实验

### 9.1 实验想证明什么

验证真实 TFE/ABY3 在不同数据集/类别数和 batch size 下：

- 能完成多步安全训练；
- runtime 和内存不是零或占位符；
- 固定点 precision 16 和 2048-D 特征路径可运行；
- 资源需求随类别数/模型规模增大。

修复后的 9/9 结果：

| 数据集 | Batch 8/16/64 secure runtime | Batch 8/16/64 peak RSS |
|---|---|---|
| CIFAR-10 | 18.24 / 17.65 / 18.53 s | 1.59 / 1.64 / 1.93 GiB |
| CIFAR-100 | 25.68 / 24.47 / 27.20 s | 3.79 / 3.64 / 3.70 GiB |
| Tiny-ImageNet | 41.52 / 39.60 / 41.63 s | 6.85 / 6.53 / 6.72 GiB |

结果说明在本机上 batch 从 8 增至 64 并没有引起 runtime 爆炸；主要资源增长来自类别数和模型参数规模。通信字节数/轮数因当前工具未暴露，必须保持 `N/A`，不能填 0。

表：`new_expTab/new_true_ours/new_tfe_system_formal_repaired.csv`

图：`new_expFig/new_public_mtt_mainline/new_figureS_tfe_system_scaling.png/.pdf`

注意：这是一组历史 Fixed kernel 系统 scaling 实验，适合作为实现可行性/资源表；不能拿它替代最新 Adaptive Figure 2 的端到端计时。

## 10. 外部方法对比

仓库已建立来源注册表：

- `new_expTab/new_true_ours/new_external_method_source_registry.csv`
- 对应 JSON/Markdown 位于同一结果目录。

目前能安全使用的结论：

- PEA 是 IEEE S&P 2023，不是 2025。
- PEA 在其 CIFAR-10、SimCLR 特征、逻辑回归、epsilon=2 设置下报告 TFE `88.99±0.08%`。
- 表中的 CaPC `86.70±0.17%` 和 DDGauss `53.31±2.52%` 是 PEA 作者在 PEA 特征上的适配复现，不是原论文原生 CIFAR-10 结果。
- DDGauss 该适配结果使用 epsilon=5，因为 epsilon=2 不收敛。
- CryptGPU 的模型、硬件和任务与本项目线性头不同，不能用两边时间直接相除宣称严格加速。

当前没有一张完全 apples-to-apples 的外部方法主表。外部结果适合作为背景和带脚注的参考，不应与本机 True Ours 形成无条件排名。

## 11. 已淘汰或不得用于论文主结果的资产

### 11.1 旧 Tiny Full seed42

旧运行混入 LR=0.2 checkpoint，Stage C 只有 30 steps、epsilon=4.406、lineage 为 legacy，已被全新隔离的 40-epoch LR=0.1 + 180-step Adaptive 运行替换。旧链条的精度、时间和 timeline 全部失效。

### 11.2 来源未验证的 direct/final-test evaluation

`new_log/new_true_ours/direct_evaluation_16models.json` 缺少可复现生成脚本、split/cache/model hash、example count 和可靠读取账本，不能作为当前正式 test 证据。

下列图由该来源生成，已被当前主线淘汰：

- `new_expFig/new_true_ours/new_true_ours_final_test.png`
- `new_expFig/new_true_ours/new_true_ours_final_test.pdf`
- `new_expFig/new_true_ours/new_true_ours_final_test.svg`

### 11.3 Hybrid “Improved Ours”

它使用 Full-Data Stage B initializer + MTT Stage C。结果可用于说明旧 Consensus 是瓶颈，但不符合 Official-MTT-only True Ours 定义。

### 11.4 旧报告中的过强结论

以下说法必须删除或降级：

- “当前所有数据集都是 3 seeds”：Tiny 当前正式策略是 n=1。
- “所有 formal 都 epsilon<=3”：旧 Tiny Full 为 4.406，已失效。
- “当前 True Ours final test 已可信完成”：来源尚未恢复。
- “Adaptive 显著优于 Fixed”：现有差距只有约 0.04pp。
- “发布模型端到端 epsilon-DP”：Stage B 非 DP，只能写 conditional Stage-C epsilon。

## 12. 当前证据到底证明了什么

| 论文主张 | 当前状态 | 证据 |
|---|---|---|
| Official MTT 比等量随机样本更有效 | 支持 | 三数据集 Stage A 均提升 3.58–6.83pp |
| True Ours 不依赖 Full Data initializer | 支持 | strict lineage、Official-MTT-only、dependency=0 |
| 方法能在真实 TFE/ABY3 上运行 | 支持 | Stage B/C 正式 reports、无 fallback、14/14 验证 |
| Secure-average-only 保持结果且更快 | 强支持 | 等精度、误差过门、4.884× 加速 |
| True Ours 更快达到目标效用 | 支持 | 三数据集 T@90 均约 2.46–2.54× 更快 |
| True Ours 完整训练在所有数据集更快 | 不支持 | Tiny 完整流程慢 1.52× |
| True Ours 在所有数据集精度更高 | 不支持 | CIFAR-10 低 4.29pp |
| Consensus 在当前主路线有益 | 不支持 | strict screen 0 候选过门 |
| 有标签 Official-MTT anchor 对 Stage B 有效 | 单 seed 强支持 | C10/C100 severe CE-only 相对 Average +35.46/+18.92pp |
| Class-aware KD/Consensus 有独立正贡献 | 不支持 | 相对 uniform 为 0.00/−0.02pp，KD-over-CE 为 +0.44/−3.56pp |
| Adaptive 显著优于 Fixed | 不支持 | 最佳差距约 0.04pp |
| 当前主结果是最终 test accuracy | 不支持 | 当前 14/14 为 development-only |
| 发布模型满足端到端 record DP | 不支持 | 数据依赖、非 DP Stage B initializer |
| 系统通信量已量化 | 不支持 | communication bytes/rounds 为 N/A |

## 13. 图表使用建议

### 13.1 适合正文

1. `new_figure2_pea_style_secure_time`：当前最重要主图，展示真实累计安全时间、精度曲线、Stage B→C 转换和 T@Target。
2. 可从当前 master JSON 生成一张精简主表：最终 development accuracy、T@90、总时间、样本量、conditional epsilon。
3. 若论文主张数据蒸馏价值，可选 `new_figureA2_sample_efficiency`，但图注必须写正确样本数。

### 13.2 适合补充材料

- Stage A accuracy-vs-steps；
- Stage B Non-IID 历史实验及最新 Consensus 负结果；
- 最新 MTT-anchor 因果分解（待 TFE 梯级完成后再定稿作图）；
- Fixed vs Adaptive 负消融；
- Adaptive C 轨迹；
- secure-average-only 等价/加速表；
- TFE 系统 runtime/RSS scaling；
- 外部方法 provenance 表。

### 13.3 不应继续使用

- `new_true_ours_final_test.*`；
- 旧 incremental 的“不含 Ours Stage B”端到端时间图；
- Hybrid Improved Ours 作为 True Ours 主结果；
- 旧 Tiny Full 的任何点；
- 把历史 test 和当前 development 结果混在同一无说明表格中。

## 14. 可以改进的地方

### P0：投稿前最重要

1. **补齐当前 True Ours 的可信冻结后评估。**

   当前主结果只有 development validation。应使用冻结模型、冻结脚本和明确的一次性读取账本，记录 split/cache/model SHA、样本数和“结果不反馈调参”。如果不做，就必须在论文主表标题和正文中明确写 development results。

2. **解决或诚实界定 CIFAR-10 的 4.29pp 缺口。**

   precision 20 和 Consensus 已证明不是方向。若还有预算，应只在 development 上研究 MTT payload/特征质量、训练正则化、类内覆盖或更合适的轻量头；任何新配置都必须重新冻结后再跑 formal，不能用 test 选参。

3. **准确表述隐私边界。**

   目前只能说 conditional Stage-C mechanism epsilon。若要端到端 DP，需要 Stage B 使用公开/不相交 initializer，或让 Stage B 本身具有可组合 DP 保证。

4. **修正 Tiny 的效率叙述。**

   Tiny 的主要亮点是 T@90 快 2.46×，不是最终完成时间更快。可以进一步研究基于冻结 target 的早停；当前曲线已经显示继续跑到 180 steps 的边际收益与时间代价不匹配。

### P1：显著增强论文可信度

5. **Tiny 增加 seeds 43/44。**

   当前 n=1 是用户批准的算力策略，可以使用，但只能写 case study，不能做方差或显著性结论。若论文需要“三数据集均多种子”，这是最直接的补项。

6. **给 secure-average-only 单独做一张系统消融表。**

   4.884× 是非常清晰的工程贡献，应报告旧图/窄图 runtime、误差、输出精度和移除的安全子图，而不只埋在日志中。

7. **补通信统计。**

   如果论文想声称通信效率，需要在运行时或网络层记录 bytes/rounds。当前 `N/A` 只能支撑计算时间和内存结论。

8. **统一历史表格语义。**

   区分 reference accuracy、development validation、historical official test；区分 Stage-C-only runtime、Stage-B runtime 和统一 end-to-end online runtime。

9. **给负结果留正式位置。**

   Consensus、precision20、Fixed vs Adaptive 都解释了为何最终设计选择 Secure Average + Adaptive。这些负结果能减少“只挑有利配置”的质疑。

10. **按冻结计划补齐 MTT-anchor 跨 seed 复核。**

   seed42 的 tiny/C10/C100 TFE 梯级已全部通过，且 anchor 数据有效。
   已生成 `plan/0729_new_stage_b_kd_remaining_seeds_plan.md`；它只规划
   seeds43/44，不授权运行。后续必须以 CE-only 为首要基线，不能再用
   `Student−Average` 代替 KD 的净贡献。

### P2：有时间再做

11. 多机真实网络环境下复测 TFE runtime/通信，而不仅是当前服务器进程间运行。
12. 增加更强但严格可比的外部基线；统一数据、特征、模型、epsilon/delta、party 数和硬件后再比较。
13. 对 C10/C100 的时间曲线增加置信区间或 bootstrap；Tiny 必须等多 seed 才能画统计带。

## 15. 推荐的最终论文实验组织

正文建议只保留四块：

1. **样本效率：** Full / Random / Official MTT。
2. **主结果：** 当前 True Ours vs Full Data 的 Figure 2 和精简表。
3. **安全系统效率：** secure-average-only 4.884× 与 TFE runtime/RSS。
4. **关键消融：** Average vs Consensus、Fixed vs Adaptive、precision16 vs20。

补充材料放：

- 全部 Stage A/B/C 历史曲线；
- Non-IID 细表；
- Adaptive C 轨迹；
- 历史端到端/final-test 结果并明确协议；
- 外部方法 provenance；
- 所有 seed、SHA、lineage 和隐私声明审计。

这样组织后，论文主线会非常清楚：

> MTT 降低安全训练的数据规模；经过严格谱系约束的 Secure Average 在 TFE/ABY3 中保持可用效用；Adaptive Stage C 提供条件式隐私机制；系统更快达到目标精度，但仍诚实保留 CIFAR-10 效用缺口、Tiny 完整训练耗时和端到端 DP 边界。

## 16. 权威入口

- 当前主结果：`new_log/new_true_ours/new_true_ours_master_results.json`
- 当前主图：`new_expFig/new_true_ours/new_figure2_pea_style_secure_time.*`
- 当前 T@Target：`new_expTab/new_true_ours/new_figure2_time_to_target_table.csv`
- 当前验证：`new_log/new_true_ours/new_true_ours_minimal_validation.json`
- 当前简报：`reportToMe/new_true_ours_minimal_correction_report.md`
- 历史 Public-MTT 表：`new_expTab/new_public_mtt_mainline/`
- 历史增量对比表：`new_expTab/new_incremental_comparison/`
- 后续诊断表：`new_expTab/new_end2end_adaptive_followup/`
- 系统修复表：`new_expTab/new_true_ours/new_tfe_system_formal_repaired.csv`
- 最新 Stage-B anchor/KD：`new_log/new_stage_b_kd/` 与 `new_expTab/new_stage_b_kd/`
- 最新 Stage-B anchor/KD final report：`reportToMe/new_stage_b_kd_final_report.md`
- 进度与实验审计：`progressLog/project_progress.md`

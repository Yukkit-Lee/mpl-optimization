# 今日计划：Reference/Test 指标治理与 Accuracy 诊断优化

## 目标

本轮不修改论文表述，不重做外部比较，不迁移 SPU，不复现 CryptGPU。

唯一目标：

> 保留当前 Adaptive DP-SGD 主方案，通过诊断、调参和小规模消融，提高 Ours 的独立验证准确率与最终测试准确率。

主路线保持：

```text
Full Data Feature Model
→ Secure Average Initialization
→ TFE/ABY3 Adaptive DP-SGD

Ours
→ Official MTT Features
→ Secure Consensus Initialization
→ TFE/ABY3 Adaptive DP-SGD
```

不得改回 Fixed DP-SGD。已有 Fixed vs Adaptive 负结果继续保留。

---

# 1. 保留已有实验

不得删除、覆盖或无理由重跑：

- Stage A 正式结果
- Stage B convergence
- Stage B IID / Mild / Severe Non-IID
- Module C 54/54
- Old End-to-End 18/18
- 新 Full Data Feature Model 9/9
- 新 Ours 9/9
- 已有最终 test
- TFE system 9/9
- 所有 checkpoint、cache、JSON、CSV、日志和图表

新结果统一放入：

```text
new_log/new_accuracy_diagnosis/
new_expTab/new_accuracy_diagnosis/
new_expFig/new_accuracy_diagnosis/
new_checkpoints/new_accuracy_diagnosis/
```

---

# 2. 指标治理

## 2.1 Reference Accuracy

如果 reference set 参与以下任一环节：

- Stage B 权重计算
- KD 评分
- candidate selection
- secure consensus
- checkpoint selection
- early stopping
- 超参数选择

那么 reference accuracy 只能作为：

```text
internal optimization / selection metric
```

不能代替独立泛化指标。

## 2.2 Independent Validation Accuracy

必须确认或建立独立 validation set：

- 来自训练数据
- 不参与 Stage A 训练
- 不参与 Stage B 权重、KD、selection、consensus
- 不参与初始化候选筛选
- 仅用于调参和训练曲线
- 与 test 完全分离

正式调参全部依据 independent validation accuracy。

## 2.3 Test Accuracy

test 仅用于：

- 冻结最终配置后的只读评估
- 最终论文泛化结果

禁止：

- 用 test 调参
- 用 test 选 checkpoint
- 用 test 选 seed
- 根据 test 结果返回继续改参数

---

# 3. Phase 0：只读审计

先生成：

```text
new_metric_split_audit.json
new_metric_split_audit.md
```

检查：

1. reference 来源
2. reference 是否参与 Stage B
3. selection 是否与 reference 重合
4. 是否存在独立 validation
5. test 是否进入任何调参或质量门
6. Full Data 与 Ours 是否使用相同评估流程
7. 样本 ID 是否交叉
8. feature cache 是否混入 validation/test
9. Max Reference Accuracy 对应哪个 checkpoint
10. 最终 test 使用 final checkpoint 还是 selected checkpoint

若存在泄漏，先修复划分，再继续后续实验。

---

# 4. Phase 1：单 Seed 快速诊断

先用：

```text
seed = 42
```

只看 independent validation，不读 test。

优先使用：

- GPU 明文代理
- plaintext fixed-point emulation
- 现有 feature cache

目标是把差距拆成：

```text
A. MTT 特征代表性损失
B. Consensus 初始化损失
C. Adaptive DP 裁剪/噪声损失
D. 训练预算不足
E. TFE 固定点数值损失
```

---

# 5. 诊断矩阵

## 数据与初始化

```text
D1: Full Features + Average Init + No DP
D2: MTT Features + Average Init + No DP
D3: MTT Features + Consensus Init + No DP
```

用途：

- D1 vs D2：判断 MTT 数据载荷损失
- D2 vs D3：判断 Consensus 是否提高独立泛化

## DP 影响

```text
D4: MTT Features + Consensus Init + No DP
D5: MTT Features + Consensus Init + Adaptive DP
D6: Full Features + Average Init + Adaptive DP
```

用途：

- D4 vs D5：判断 Adaptive DP 造成的效用损失
- D5 vs D6：比较紧凑路线与完整特征路线

## 初始化泛化

```text
D7: MTT Features + Average Init + Adaptive DP
D8: MTT Features + Consensus Init + Adaptive DP
```

若 Consensus 只提高 reference、不提高 independent validation，则必须继续诊断权重和选择机制。

---

# 6. Phase 2：训练预算 Sweep

当前正式预算：

```text
CIFAR-10: 10 steps
CIFAR-100: 20 steps
Tiny: 30 steps
```

先检查是否明显欠训练。

## CIFAR-10

```text
10 / 20 / 40 / 80
```

## CIFAR-100

```text
20 / 40 / 80 / 120
```

## Tiny ImageNet

```text
30 / 60 / 120 / 180
```

要求：

- 每个配置重新计算 accountant
- 保持 `ε ≤ 3`
- 根据 steps 调整 noise multiplier
- 记录 validation、reference、runtime、epsilon、clip statistics
- 不读取 test

---

# 7. Phase 3：Batch Size 与 Learning Rate

在选出的训练预算上进行单 seed 小网格。

## Batch Size

```text
16 / 32 / 64
```

显存允许时可补：

```text
128
```

## Learning Rate

```text
0.5× current
1.0× current
2.0× current
```

若仍明显停滞，再补：

```text
4.0× current
```

每个配置必须：

- 重新计算 sample rate 与 accountant
- 保持 `ε ≤ 3`
- 记录吞吐、GPU利用率、RSS、validation accuracy

---

# 8. Phase 4：Adaptive Clipping 诊断

主方案继续使用 Adaptive DP-SGD，但必须记录每一步：

- clipping norm `C_t`
- gradient norm distribution
- clipped fraction
- gradient mean/std
- noise std
- update norm
- loss
- validation accuracy
- fixed-point saturation/overflow

重点检查：

1. `C_t` 是否几乎不变
2. 是否大部分梯度被裁剪
3. 是否噪声绝对幅度过高
4. clipping update 方向是否正确
5. moving average 是否更新过慢
6. batch 平均是否重复缩放
7. 分布式噪声方差是否重复累计
8. 固定点截断是否吞掉小梯度

候选参数：

```text
initial C: 0.5× / 1.0× / 2.0× current
EMA/update rate: conservative / current / aggressive
target clipped fraction: 0.1 / 0.3 / 0.5
```

不得通过放宽隐私预算换准确率。

---

# 9. Phase 5：Consensus 初始化改进

测试融合：

```text
W_init = α · W_consensus + (1-α) · W_average
```

候选：

```text
α = 0.25 / 0.50 / 0.75 / 1.00
```

同时检查：

- teacher weighting 是否过尖锐
- 是否单 client 主导
- 温度是否过低
- candidate selection 是否过拟合 reference
- 是否需要限制单 client 最大权重
- 是否需要 top-k consensus
- 是否需要 clipped-weight consensus

只用 independent validation 选 α。

---

# 10. Phase 6：Tiny ImageNet 专项

Tiny 当前差距最大，先判断是否是 MTT 预算不足。

单 seed：

```text
8000 / 12000 / 16000 distilled features
```

要求：

- 优先使用更多公开官方 MTT 资产
- 不引入 DP-MTT
- 不从 test 构造蒸馏样本
- 检查类别均衡
- 检查 encoder 对 Tiny 分辨率适配
- 检查 2048-D 标准化
- 检查 200 类线性头初始化
- 检查 30 steps 是否严重不足

目标是显著提高 independent validation accuracy，而不是先追求超过 Full Data。

---

# 11. Phase 7：固定点与 TFE 数值诊断

逐级对照：

```text
P1: PyTorch float, no DP
P2: PyTorch float, Adaptive DP
P3: Plaintext fixed-point emulation, Adaptive DP
P4: TFE secure, no DP
P5: TFE secure, Adaptive DP
```

比较：

- logits
- loss
- gradients
- clipped gradients
- noise statistics
- parameter updates
- validation predictions

若 P2 正常但 P3/P4/P5明显下降，优先修复：

- fixed-point precision
- truncation
- scaling
- overflow
- secure reduction
- division
- batch averaging

不要继续用学习率掩盖数值错误。

---

# 12. 筛选与正式复跑

## 单 Seed 筛选

所有探索配置先用：

```text
seed = 42
```

只看 independent validation。

## 入围规则

每个数据集最多选 2 个配置进入正式阶段。

必须满足：

1. validation 高于当前 Ours
2. 多个 checkpoint 持续提升
3. `ε ≤ 3`
4. 无数值异常
5. runtime 可接受
6. 不是只提高 reference

## 正式三 Seed

最终入围配置运行：

```text
42 / 43 / 44
```

配置冻结后，才允许一次最终只读 test。

---

# 13. 今日优先级

```text
Priority 1: 指标与数据划分审计
Priority 2: 训练预算 sweep
Priority 3: batch size + learning rate
Priority 4: Adaptive clipping 诊断
Priority 5: Consensus blending
Priority 6: Tiny MTT budget
Priority 7: TFE fixed-point诊断
```

若时间不足，至少完成 Priority 1–4。

---

# 14. 硬件利用

服务器：

- NVIDIA RTX 5090
- 25 vCPU Intel Xeon Platinum 8470Q

要求：

- 不人为限制 CPU 线程
- 充分使用多线程和多进程
- GPU 代理实验尽量批量并行
- TFE CPU 任务使用全部有效 vCPU
- 复用 memmap 和 feature cache
- 长任务使用 tmux
- 监控 GPU、CPU、RAM、磁盘
- 不启动会争抢内存的重复正式任务

---

# 15. 输出

必须生成：

```text
new_metric_split_audit.json
new_metric_split_audit.md
new_accuracy_diagnosis_matrix.csv
new_training_budget_sweep.csv
new_batch_lr_sweep.csv
new_adaptive_clipping_diagnostics.csv
new_consensus_blending_sweep.csv
new_tiny_budget_sweep.csv
new_numeric_consistency_report.json
new_today_accuracy_optimization_report.md
```

图表：

```text
new_validation_accuracy_vs_steps.png/pdf
new_reference_vs_validation_gap.png/pdf
new_clipping_fraction_vs_steps.png/pdf
new_gradient_noise_diagnostics.png/pdf
new_accuracy_runtime_pareto.png/pdf
```

---

# 16. 今日结束报告

必须说明：

1. reference、validation、test各自用途
2. reference 是否参与选择
3. 准确率差距主要来自哪一阶段
4. 哪些配置提高 validation
5. 是否欠训练
6. Adaptive clipping 是否正常
7. DP 噪声是否过强
8. 是否存在固定点/TFE数值损失
9. Consensus 是否只提高 reference
10. Tiny 是否主要受 MTT 预算限制
11. 最佳候选配置
12. 下一步正式三 seed 任务
13. 未读取 test 的证明
14. 新文件和 Git 状态

---

# 17. 禁止事项

禁止：

- 用 test 调参
- 根据 test 选择 checkpoint 或 seed
- 只优化 reference accuracy
- 隐藏 validation 下降
- 破坏 `ε ≤ 3`
- 擅自加入 DP-MTT
- 全面重跑正式主线
- 一次同时大改数据、初始化、优化器和 DP 参数
- 单 seed 提升后直接宣布成功

本轮目标是提高真实泛化准确率，而不是通过图表掩盖差距。

# Incremental Completion Task for Full Data Feature Model vs Ours

## Overall Goal

继续完善当前已经完成的增量实验。

当前 End-to-End 主方案保持不变：

- Full Data Feature Model + Adaptive DP-SGD
- Ours + Adaptive DP-SGD

不要把主实验改回 Fixed DP-SGD。

Adaptive DP-SGD 在本实验中的作用是作为两条路线共同使用的训练机制，以保证比较公平。

不要声称 Adaptive 优于 Fixed。
已有 Module C 中 Adaptive 与 Fixed 的结果和负结论必须保留。

本轮任务不是重新设计算法，也不是重跑整个项目，而是补齐当前报告缺失的数据、完善公平性指标并完成论文表格。

---

# 1. Preserve Existing Results

必须保留并复用现有结果：

- Stage A 正式结果
- Stage B 收敛结果
- Stage B Non-IID 结果
- Module C 54/54 Fixed/Adaptive DP 结果
- 当前 Ours End-to-End 三数据集三 seeds 结果
- 当前 Full Data Feature Model 三数据集三 seeds 结果
- 72 个已有模型的只读 test 结果
- TFE system 9/9 结果
- 所有现有 JSON、CSV、checkpoint、history 和 figure

禁止无理由重新运行：

- Stage A
- Existing Stage B
- Module C 54-run grid
- Existing Ours End-to-End
- TFE system benchmark

只有在缺失必要字段且无法从日志恢复时，才允许进行最小范围的补跑。

---

# 2. Keep Adaptive End-to-End Protocol

两条主路线保持：

## Full Data Feature Model

```text
Full Training Dataset
→ Same Shared Encoder
→ Full 2048-D Features
→ Secure Average Initialization
→ TFE/ABY3 Adaptive DP-SGD
→ Final Model
```

## Ours

```text
Official MTT Distilled Images
→ Same Shared Encoder
→ Compact 2048-D Features
→ Secure Consensus Initialization
→ TFE/ABY3 Adaptive DP-SGD
→ Final Model
```

要求两条路线保持：

- same encoder
- same feature dimension
- same classifier
- same backend
- same fixed-point precision
- same batch size
- same training-step budget
- same Adaptive DP implementation
- same accountant
- same evaluation protocol
- same seeds: 42, 43, 44

不要为了提高某一条路线的结果而单独改变超参数。

---

# 3. Highest-Priority Task: Full Data Final Test Evaluation

当前报告只有 Ours Test Accuracy，没有 Full Data Feature Model 的最终 Test Accuracy。

请对以下 9 个 Full Data 最终模型执行只读 test evaluation：

- CIFAR-10: seeds 42, 43, 44
- CIFAR-100: seeds 42, 43, 44
- Tiny ImageNet: seeds 42, 43, 44

要求：

- 不重新训练模型
- 不修改 checkpoint
- 不根据 test 结果选择 checkpoint
- 不根据 test 结果选择 seed
- 不进行 test-set tuning
- 使用与 Ours 相同的 test pipeline
- 记录每个 seed 的 test accuracy
- 计算 mean ± std

输出：

```text
new_full_data_final_test_evaluation.json
new_full_data_final_test_evaluation.csv
```

最终主表必须包含：

| Dataset | Full Data Test Acc | Ours Test Acc | Accuracy Gap | Accuracy Retention |
|---|---:|---:|---:|---:|

Accuracy Retention:

```text
Ours Test Accuracy / Full Data Test Accuracy
```

---

# 4. Rebuild the Main End-to-End Table

重新生成正式主表，包含：

- Dataset
- Method
- Seed
- Final Test Accuracy
- Max Reference Accuracy
- Total Runtime
- Stage B Runtime
- Stage C Runtime
- Number of Secure Steps
- Actual Secret-Shared Samples Processed
- Peak Process-Tree RSS
- Peak GPU Memory
- Final Epsilon
- Delta

方法只有：

```text
Full Data Feature Model
Ours
```

不要在该主表中加入 Random Subset。

Random Subset 继续作为 Stage A 或旧 End-to-End ablation 保存。

---

# 5. Privacy Reporting

当前两条路线的实际 epsilon 不完全相等。

因此论文和报告中统一写：

```text
target privacy budget upper bound ε ≤ 3
```

不要写：

```text
both methods use exactly ε = 3
```

分别保留实际 accountant 输出。

必须明确：

- Full Data 和 Ours 使用同一目标隐私预算上限；
- Stage C 使用 Adaptive DP-SGD；
- Adaptive 并不是本文已证明优于 Fixed 的贡献；
- End-to-End 初始化是数据相关且非 DP；
- End-to-End 的 ε 只表示条件于初始化的梯度机制会计；
- 不宣称完整 End-to-End record-level DP。

---

# 6. Runtime Decomposition

当前总运行时间不足以说明时间差来自哪个阶段。

请优先从已有日志提取：

- feature loading / cache preparation time
- Stage B initialization time
- Stage C secure training time
- evaluation time
- total measured time

同时统计：

- batch size
- secure steps
- samples processed per step
- total secret-shared samples processed

不要仅使用“完整数据集大小”作为安全训练工作量。

如果已有日志无法提供阶段耗时：

1. 首先检查现有 timestamp 和 profiler 日志；
2. 其次通过现有 checkpoint 做最小化 profiling；
3. 只有确实无法恢复时，才允许按完全相同配置重新运行 profiling；
4. 不要重跑所有实验来获取一个时间字段。

输出：

```text
new_runtime_breakdown.json
new_runtime_breakdown.csv
```

---

# 7. Time-to-Target Analysis

从已有完整 training history 中计算共同目标准确率。

规则：

- target 必须由 reference/validation history 决定；
- 不使用 final test 选择 target；
- 每个数据集使用同一个 target 比较两种方法；
- target 必须是两种方法都能够达到的值。

报告：

- Time@Target
- Steps@Target
- Samples@Target

如果某种方法没有达到 target，记录：

```text
Not Reached
```

不要人为选择只对 Ours 有利的 target。

---

# 8. Main Figure

生成三数据集主图：

```text
CIFAR-10
CIFAR-100
Tiny ImageNet
```

每个 subplot 只有两条曲线：

```text
Full Data Feature Model
Ours
```

主图：

```text
x-axis: Secure Training Time
y-axis: Reference/Validation Accuracy
```

辅助图：

```text
x-axis: Secret-Shared Samples Processed
y-axis: Reference/Validation Accuracy
```

最终 test accuracy 不绘制成训练曲线，只放最终表格或终点标记。

输出：

```text
new_figure2_full_feature_vs_ours_time.png
new_figure2_full_feature_vs_ours_time.pdf
new_figure2_full_feature_vs_ours_samples.png
new_figure2_full_feature_vs_ours_samples.pdf
```

---

# 9. Result Interpretation Rules

允许的表述：

- Ours reduces secure training time.
- Ours preserves part of the Full Data Feature Model utility.
- Ours provides an efficiency–utility tradeoff.
- CIFAR-10 and CIFAR-100 show stronger accuracy retention than Tiny ImageNet.
- The compact distilled-feature route reduces the secure workload.

禁止的表述，除非最终 test 结果支持：

- Ours is more accurate than Full Data Feature Model.
- Ours dominates Full Data Feature Model on all datasets.
- Adaptive DP-SGD is better than Fixed DP-SGD.
- The complete End-to-End model is record-level DP.
- Total runtime reduction is entirely caused by fewer training samples.

Tiny ImageNet 如果仍然精度保持率较低，必须作为 limitation 如实报告，不允许隐藏。

---

# 10. External Method Comparison

完成 3–4 个方法的外部对比表。

优先候选：

```text
PEA
CaPC
DDGauss
Ours
```

执行顺序：

## Priority 1

从原论文、正式补充材料和官方仓库提取公开数据。

必须记录：

- exact paper title
- publication year
- dataset
- model
- accuracy
- epsilon
- delta
- number of parties
- MPC / secure aggregation setting
- hardware
- runtime
- privacy scope
- exact table/figure/source location

## Priority 2

如果论文正文没有数字，检查：

- supplementary material
- official repository
- released logs
- official result files

## Priority 3

只有以下条件全部满足时才运行外部方法：

- 没有可用的公开结果；
- 有官方代码；
- 数据集和隐私设置可以合理对齐；
- 安装与运行成本较低；
- 该方法对主论文比较确实重要。

不要为了填一个表格单元格耗费大量时间复现外部系统。

外部表必须区分：

```text
Reported by original paper
Locally reproduced
```

外部方法表未填完前，不得将阶段状态标记为 COMPLETE。

---

# 11. CryptGPU Treatment

本轮不复现 CryptGPU。

仅进行文字对比。

从 CryptGPU 原论文或 PEA 中核对：

- model
- dataset
- target accuracy
- runtime
- hardware
- privacy setting

文字中说明：

```text
Direct secure training of deep neural networks, as represented by systems such as CryptGPU, incurs substantial cryptographic computation. Our method follows a different route by using distilled compact features and a lightweight secure classifier. Because the model, hardware, MPC protocol, and timing scope differ, the published CryptGPU runtime is used only as contextual evidence rather than a direct speedup baseline.
```

禁止：

- 用本机 Ours 时间直接除以 CryptGPU 论文时间；
- 声称严格加速多少倍；
- 把不同准确率和不同模型放在同一公平柱状图中。

---

# 12. Correct Current Report Errors

修正当前 `experiment_report.md`：

1. Full Data Test Accuracy 缺失；
2. 外部表中的 TBD 未完成；
3. Phase 8 不能在外部表仍为 TBD 时标记 COMPLETE；
4. 核对所有论文年份；
5. 区分 Max Reference Accuracy 和 Final Test Accuracy；
6. 将“same ε=3”改为“same target budget upper bound ε≤3”；
7. 明确 Adaptive 是两条路线共同设置，而不是已证明的优势。

---

# 13. Final Output

最终提交：

```text
new_incremental_comparison_final_report.json
new_incremental_comparison_final_report.csv
new_full_data_final_test_evaluation.json
new_full_data_final_test_evaluation.csv
new_runtime_breakdown.json
new_runtime_breakdown.csv
new_external_comparison_table.csv
new_experiment_report_revised.md
```

并输出：

1. 复用了哪些已有实验；
2. 新运行了哪些任务；
3. Full Data 最终 test 结果；
4. Ours 最终 test 结果；
5. accuracy retention；
6. time reduction；
7. Time@Target；
8. Samples@Target；
9. Stage B/Stage C runtime decomposition；
10. 外部数据来源；
11. CryptGPU文字对比；
12. 可以支持的论文结论；
13. 必须避免的论文结论；
14. 所有图表和文件路径；
15. Git commit 和 push 状态。

不要重新运行整个主线。
先补齐最缺失、成本最低、最影响论文完整性的结果。

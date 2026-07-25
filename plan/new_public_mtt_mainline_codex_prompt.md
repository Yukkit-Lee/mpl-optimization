# Codex 主路线切换与全部正式实验指令
## Public MTT 主线 + TFE 安全 Stage B + TFE 安全 Stage C + 最后补充 DP-MTT

> 本文档覆盖此前 `new_codex_secure_feature_full_formal_prompt.md` 中“DP-MTT 未达到质量门就阻塞后续阶段”的安排。
> 从现在开始，**Public MTT 是论文主路线，DP-MTT 是最后执行的补充实验，不得再阻塞 Module B、Module C、End-to-End 和 TFE 系统实验。**

---

# 0. 最终确定的论文主框架

## Stage A：公开 MTT 蒸馏图像与紧凑特征构建

```text
Official Public MTT Distilled Images
→ One Shared ConvNet Encoder
→ Frozen 2048-D Compact Features
```

作用：

- 降低后续安全训练需要处理的样本数量；
- 降低后续模型复杂度；
- 强调训练效率和达到目标准确率所需的迭代数量；
- 不声称 Stage A 对原始训练数据提供差分隐私。

隐私字段：

```text
stage_a_track = public_mtt_benchmark
stage_a_epsilon = N/A
stage_a_dp_protected = false
```

## Stage B：TFE/ABY3 密态共识初始化

```text
Secret-Shared 2048-D Distilled Features
+ Secret-Shared Local Knowledge
→ Secure Teacher Knowledge Aggregation
→ Secure KD-Compatible Initialization
→ Secure Average / Reference-Weighted Candidate Construction
→ Secure Candidate Evaluation and Selection
→ Secret-Shared Global Initializer
```

Stage B 的安全目标是：

- MPC 机密性；
- 隐藏单个机构的模型、logits、聚合权重和候选分数；
- 不将 MPC 机密性写成 epsilon。

隐私字段：

```text
stage_b_security = tfe_aby3_mpc_confidentiality
stage_b_epsilon = N/A
```

## Stage C：TFE/ABY3 密态 Adaptive DP 训练

```text
Secret-Shared 2048-D Distilled Features
+ Secret-Shared Global Initializer
→ Secure Fixed DP-SGD
→ Secure Adaptive DP-SGD
→ Released Differentially Private Linear Model
```

隐私字段：

```text
privacy_scope = stage_c_distilled_feature_records
stage_c_epsilon
stage_c_delta
accountant_type
noise_mechanism
```

必须明确：

> Stage-C epsilon 只保护 Stage C 所处理的蒸馏特征记录或其发布模型，不能解释为官方 MTT 图像所对应原始训练数据的端到端记录级 DP。

## DP-MTT：最后补充

```text
Raw Training Data
→ DP Expert Trajectories
→ DP-MTT Distilled Images
```

DP-MTT 不再是进入 Stage B/C 的前置质量门。必须在所有 Public MTT 主线正式实验完成后再继续调参和扩展。

---

# 1. 立即改变当前调度状态

当前状态中：

- Public MTT 三数据集、3 seeds 已完成；
- DP-MTT 仅完成 CIFAR-10 seed 42 的若干调参轮次；
- Module B、Module C、End-to-End 和 TFE 正式实验尚未开始。

现在立即执行：

1. 将 DP-MTT 状态改为：
   ```text
   DEFERRED_UNTIL_MAIN_PIPELINE_COMPLETE
   ```
2. 删除“DP-MTT 未达 Public MTT 10 个百分点内就阻塞后续阶段”的调度规则；
3. 不删除已有 DP-MTT 结果；
4. 立即从 Public MTT 特征缓存开始执行正式 Module A 效率实验、Module B、Module C、End-to-End 和 TFE 系统实验；
5. 全部主线完成后，再恢复 DP-MTT 调参。

---

# 2. 资源、线程和持久运行要求

机器有两张 RTX 4090。

## 2.1 CPU 线程严格为 1

所有 Python、PyTorch、TensorFlow、TFE、绘图和报告进程必须设置：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

Python 中：

```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

DataLoader：

```text
num_workers = 0
persistent_workers = false
```

禁止 multiprocessing pool 和隐式 CPU 并行。

## 2.2 两张 GPU

- GPU 0 和 GPU 1 各自最多运行一个正式 PyTorch 任务；
- 使用独立锁；
- 不允许同一张卡重复启动任务；
- 允许根据显存调整 microbatch，但必须保持 effective batch size；
- 记录峰值显存、microbatch、accumulation steps 和运行时间。

## 2.3 tmux 与恢复

使用 tmux session：

```text
new_public_mtt_secure_pipeline
```

窗口：

```text
gpu0
gpu1
tfe
monitor
report
```

所有长任务必须：

- 写 PID；
- 写 heartbeat；
- 写 checkpoint；
- 保存 optimizer、scheduler、accountant、adaptive clipping 和 RNG 状态；
- 断开 SSH 后继续运行；
- 异常退出后从最近 checkpoint 恢复；
- 不得因为 Codex 当前回复结束而停止后台任务。

---

# 3. 本轮所有新产物命名

本轮所有新增目录、脚本、日志、JSON、CSV、模型、图表和报告均必须以前缀：

```text
new_
```

开头。

例如：

```text
new_log/
new_expFig/
new_expTab/
new_cache/
new_init_model/
new_scripts/
new_moduleA_efficiency_*.json
new_moduleB_noniid_*.csv
new_moduleC_adaptive_*.png
new_end2end_*.json
```

不得覆盖已有旧结果。

---

# 4. 共享 Encoder 与 2048 维特征

## 4.1 编码器一致性

所有参与方、所有数据集中的同一实验必须使用同一个共享 ConvNet encoder checkpoint：

```text
X_i^syn = f_phi(D_i^syn)
```

禁止每个参与方训练独立 encoder 后直接拼接特征。

## 4.2 Encoder 规则

- 只在官方 MTT 蒸馏图像或明确允许的数据上训练；
- 训练完成后冻结；
- 提取 classifier 前的 2048-D representation；
- scaler 只能用训练/蒸馏特征统计；
- test 特征不得参与 scaler；
- 保存 encoder 和 scaler SHA256；
- 对特征做 L2 norm bound，记录上界；
- Stage B/C 必须读取相同 feature cache 和相同 encoder signature。

## 4.3 先验证特征质量

在进入安全 Stage B/C 正式实验前，验证：

```text
Official MTT images
→ Shared Encoder
→ 2048-D Features
→ Plain Linear Classifier
```

三数据集、3 seeds 的精度不得显著低于已完成的 Public MTT 参考结果：

```text
CIFAR-10: 0.6783 ± 0.0076
CIFAR-100: 0.4462 ± 0.0055
Tiny ImageNet: 0.2641 ± 0.0054
```

若新特征流程比以上均值低超过 2 个百分点：

- 检查 encoder checkpoint；
- 检查 normalization；
- 检查数据版本；
- 检查 classifier 和训练配置；
- 使用 validation/reference set 调参；
- 不得直接进入正式安全结果。

---

# 5. Module A：强调训练效率，不报告 epsilon

Stage A 不再以“DP-MTT 达到 Public MTT 精度”为主线目标，而是证明：

> 少量官方蒸馏图像可以保留完整数据集的主要训练效用，并显著减少训练样本数、优化步数或达到目标准确率所需时间。

## 5.1 正式对比

每个数据集比较：

```text
Full Images
Random Subset
Official MTT Distilled Images
```

可用官方 MTT IPC 版本时增加：

```text
Official MTT IPC=1
Official MTT IPC=10
Official MTT IPC=50
```

规则：

- 不得伪造不存在的官方 IPC 数据；
- 若只有 IPC=50，就使用已有版本；
- Random Subset 与相应 MTT 的样本数相同；
- 相同 shared encoder；
- 相同 linear classifier；
- 相同 optimizer、LR schedule 和 initialization；
- 至少 3 seeds。

## 5.2 Stage A 需要回答的两个问题

### A. 少量图像能否达到接近 Full Data 的效果？

报告：

```text
Final Accuracy
Accuracy Retention = Distilled Accuracy / Full Accuracy
Sample Compression Ratio
```

### B. 是否训练更快？

报告：

```text
Step@TargetAcc
Time@TargetAcc
EarlyAUC
Total Training Time
Processed Examples@TargetAcc
```

目标精度必须在 validation/reference set 上预定义，不能根据 test 曲线反向选择。

## 5.3 Stage A 图形

建议生成：

### Figure A1：准确率—训练步数

每个数据集一个子图：

```text
Full Images
Random Subset
Official MTT
```

### Figure A2：样本效率或时间效率

可采用柱状图：

```text
Samples
Step@TargetAcc
Time@TargetAcc
```

如果 IPC=1/10/50 都可用，再增加一张：

```text
Accuracy vs Images Per Class
```

Stage A 图中：

```text
stage_a_epsilon = N/A
```

不得给官方 MTT 强行标 epsilon。

---

# 6. Stage B：TFE/ABY3 安全实现

## 6.1 Stage B 输入

```text
secret-shared 2048-D distilled features
secret-shared labels
secret-shared local linear-head parameters or logits
secret-shared reference features/labels
```

Reference 数据来源优先：

- 官方 MTT 蒸馏特征内部独立划分；
- 公开辅助数据。

未经 DP 保护的原始私有 held-out 数据不得被称为 public。

## 6.2 安全友好 KD

优先使用：

```text
secure teacher-logit aggregation
+ secure logit-MSE distillation
```

损失：

```text
L_KD = ||z_student - z_teacher||_2^2
```

保留 plaintext reference，用于验证固定点数值误差。

## 6.3 Stage B 主候选

正式主方法至少实现：

```text
Secure KD-Avg
Secure KD-ReferenceWeighted
```

可选：

```text
Secure KD-Best
```

只有在 secure argmax 与 secure mux 真正实现后，KD-Best 才能进入主候选；否则只作为：

```text
secure_score_computation_with_revealed_index
```

并明确 reveal 边界。

## 6.4 Stage B 不能只做 smoke test

每个正式 Stage B 运行必须：

- 使用完整训练预算；
- 生成完整 history；
- 至少 3 seeds；
- 三个数据集；
- 输出 mean ± std；
- 记录 TFE runtime 和能够测得的通信指标；
- 不允许 2-step、10-step 结果进入正式表格。

---

# 7. Module B 的两类正式实验

## 7.1 Figure 5A：初始化带来的收敛速度

每个数据集主图只画两条：

```text
Random / Average Initialization
Secure Consensus Initialization
```

保持：

- 相同官方 MTT 2048-D features；
- 相同后续 Stage C 设置；
- 相同模型；
- 相同 epsilon；
- 只改变初始化方式。

指标：

```text
Step@TargetAcc
Time@TargetAcc
EarlyAUC
Final Accuracy
TFE Runtime
Communication
```

每个数据集使用各自合理的迭代终点，不得全部统一为 300 steps。

## 7.2 Non-IID 实验：必须执行

Non-IID 已明确纳入本轮正式实验，不得遗漏。

数据划分：

```text
IID
Mild Non-IID
Severe Non-IID
```

方法：

```text
NoKD
Secure KD-Avg
Secure KD-ReferenceWeighted
```

至少 3 seeds，三数据集。

输出：

```text
Final Accuracy
Accuracy Retention from IID to Severe
EarlyAUC
Step@TargetAcc
Teacher Disagreement
Reference Weight Entropy
Runtime
Communication
```

图形建议使用分组柱状图或简洁折线，不在一张图中塞入大量训练曲线。

## 7.3 Non-IID 调参要求

若 KD 在 Severe Non-IID 下不优于 NoKD：

- 检查各客户端类别覆盖；
- 检查 reference weights 是否退化为均匀；
- 调整 KD temperature；
- 调整 supervised/KD loss 权重；
- 增加 confidence gating；
- 分离 calibration set 与 selection set；
- 增加或重构 reference set；
- 使用 validation/reference 指标调参后重跑。

不得用 test set 调参。

---

# 8. Stage C：TFE 安全 Fixed/Adaptive DP-SGD

Stage C 模型：

```text
z = W x + b
x ∈ R^2048
```

## 8.1 必须秘密共享

```text
X
y
W
b
per-example gradients
gradient norm/clipping statistics
noise shares
updated parameters
```

## 8.2 解析梯度

为避免 TFE 通用 autograd 限制，使用线性模型解析梯度：

```text
e_i = W x_i + b - y_i
g_W_i = e_i x_i^T
g_b_i = e_i
```

实现：

```text
secure per-example gradient
secure norm
secure clipping
secure aggregation
distributed noise shares
secret-shared update
```

## 8.3 Secure noise

不得用单次 NumPy noise 后再 secret-share 并称为 MPC noise。

每个 MPC 计算方独立产生 noise share，总噪声由份额相加形成。记录：

```text
noise_share_provider_count
honest_party_assumption
share_variance
total_noise_variance
```

## 8.4 Adaptive clipping 顺序

必须是：

```text
用 C_t 裁剪当前 batch
→ 安全计算裁剪统计
→ 更新 C_{t+1}
→ 下一轮才使用 C_{t+1}
```

禁止明文更新 C。

---

# 9. Module C 正式实验

每个数据集主图只画：

```text
Secure Fixed DP-SGD
Secure Adaptive DP-SGD
```

Plain 结果放表格或虚线参考。

## 9.1 不同数据集不同训练预算

先按 epoch 或 processed examples 定义预算：

```text
steps = ceil(epochs × num_samples / batch_size)
```

必须通过 validation pilot 确定：

- CIFAR-10 终点；
- CIFAR-100 终点；
- Tiny ImageNet 终点。

不得全部使用同一个 step 数。

## 9.2 隐私—效用实验

每个数据集至少比较多个 Stage-C 隐私点，例如：

```text
epsilon_C ∈ {1, 3, 8}
```

如果机制会计不支持某噪声，写 `N/A`，不得套用错误 accountant。

## 9.3 Module C 输出

```text
Accuracy vs Iterations
Accuracy vs epsilon_C
C_t trajectory
Clipped Fraction
Gradient Norm Quantiles
Runtime
Communication
Mean ± Std
```

精度不理想时必须使用 validation/reference set 调：

```text
learning rate
batch size
effective batch size
epochs
C_0
C_min/C_max
gamma
eta_c
update interval
warm-up
noise multiplier
feature norm bound
global/layer-wise clipping
fixed-point precision
```

不得使用 test set 调参或只保留最好 seed。

---

# 10. End-to-End 正式实验

每个数据集主图只保留两条核心曲线。

## Baseline

```text
Random Subset
→ Shared Encoder
→ Secure Average Initialization
→ Secure Fixed DP-SGD
```

## Ours

```text
Official MTT Distilled Images
→ Shared Encoder
→ Secure KD-ReferenceWeighted Initialization
→ Secure Adaptive DP-SGD
```

公平性：

- Random subset 与 MTT 样本数相同；
- 相同 encoder；
- 相同 feature dimension；
- 相同线性模型；
- 相同 effective batch；
- 相同 target epsilon_C；
- 相同 evaluation protocol；
- dataset-specific iterations。

## 10.1 消融表

单独列出：

```text
Random Subset + Fixed DP
Official MTT + Fixed DP
Official MTT + Secure Stage B + Fixed DP
Official MTT + Secure Stage B + Adaptive DP
```

不要把所有消融曲线都塞到主图。

## 10.2 End-to-End 质量要求

目标之一：

```text
Ours Final Accuracy >= Baseline + 2 percentage points
```

或：

```text
Ours Step@TargetAcc <= 0.8 × Baseline Step@TargetAcc
```

未达到时，必须基于 validation/reference 诊断调参；不得修改或挑选 test 结果。

---

# 11. TFE 正式系统实验

不得再以 10-step sanity 作为正式结果。

每个数据集至少提供：

```text
small
medium
largest feasible
```

记录：

```text
num_samples
feature_dim=2048
num_classes
batch_size
iterations
model_parameters
fixed_point_precision
runtime
communication_bytes
communication_rounds
peak_memory
revealed_values
```

如果单机 TFE 无法暴露通信统计：

- 明确写 `N/A`；
- 说明工具限制；
- 不得伪造 LAN/WAN 数据。

所有 reveal 必须列入 JSON。

---

# 12. 正式运行数量与调度

在启动前先打印：

```text
Phase
Dataset
Seed
Split
Method
Privacy Point
GPU/TFE Assignment
Expected Output
```

然后生成完整 manifest。

至少：

- Module A：3 datasets × 3 seeds × methods；
- Module B convergence：3 datasets × 3 seeds × 2 methods；
- Module B Non-IID：3 datasets × 3 seeds × 3 splits × 3 methods；
- Module C：3 datasets × 3 seeds × privacy points × methods；
- End-to-End：3 datasets × 3 seeds × methods；
- TFE system：3 datasets × 3 scales。

所有任务在 tmux 中持续运行，完成一个阶段后自动继续，不等待人工逐项确认。

---

# 13. DP-MTT 最后补充

只有在以下全部完成后，才恢复 DP-MTT：

```text
Module A Public MTT formal complete
Module B convergence complete
Module B Non-IID complete
Module C complete
End-to-End complete
TFE system evaluation complete
```

DP-MTT 输出作为：

- 补充隐私—效用实验；
- 附录；
- 后续扩展。

不得让 DP-MTT 再阻塞 Public MTT 主线。

---

# 14. 自动检查

新增测试：

```text
new_tests/new_test_public_mtt_mainline.py
new_tests/new_test_shared_encoder_signature.py
new_tests/new_test_secure_stage_b.py
new_tests/new_test_noniid_splits.py
new_tests/new_test_secure_stage_c.py
new_tests/new_test_privacy_scope.py
new_tests/new_test_no_test_leakage.py
new_tests/new_test_no_plaintext_fallback.py
new_tests/new_test_formal_not_smoke.py
```

必须断言：

- `stage_a_epsilon=N/A`；
- `stage_b_epsilon=N/A`；
- `privacy_scope=stage_c_distilled_feature_records`；
- Module B/C backend 为 `tfe_mpc`；
- feature dimension 为 2048；
- encoder signature 一致；
- Non-IID 三种划分均存在；
- formal 配置不允许 `--quick`；
- 三数据集 steps 不得全部相同；
- test set 不进入调参；
- 无 silent plaintext fallback；
- DP-MTT 状态不能阻塞主线。

---

# 15. 最终完成标准

不能仅以代码运行或 smoke test 作为完成。

必须完成：

1. Module A Full/Random/Public MTT 正式实验；
2. Stage A 训练效率图和样本效率图；
3. Module B 收敛速度正式实验；
4. Module B IID/Mild/Severe Non-IID 正式实验；
5. Module C Fixed/Adaptive 多隐私点正式实验；
6. End-to-End 三数据集、至少 3 seeds；
7. TFE small/medium/largest-feasible 正式系统实验；
8. mean ± std；
9. 完整 JSON、CSV、PNG、PDF；
10. tmux、checkpoint、heartbeat 和断点恢复；
11. 全部新产物 `new_` 前缀；
12. 最后再补 DP-MTT。

---

# 16. 最终回复格式

全部主线完成后，按以下格式回复：

```text
1. Mainline Status
2. Branch and Git Commits
3. Public MTT Stage-A Results
4. Stage-A Training Efficiency Results
5. Shared Encoder and 2048-D Feature Validation
6. Secure Stage-B Implementation
7. Module-B Convergence Results
8. Module-B Non-IID Results
9. Secure Stage-C Fixed DP Results
10. Secure Stage-C Adaptive DP Results
11. End-to-End Two-Curve Results
12. TFE Runtime and Communication Results
13. Privacy Scope and epsilon Interpretation
14. Figures and Tables Generated
15. Exact Reproduction Commands
16. Remaining BLOCKED Items
17. DP-MTT Supplement Status
18. Supported Paper Claims
19. Claims That Must Still Be Avoided
```

---

# 17. 立即执行

现在立即：

1. 读取现有 `new_` 进度文件和 manifest；
2. 将 DP-MTT 改为 deferred，不删除已有结果；
3. 解除它对 Phase 7–10 的阻塞；
4. 核验 Public MTT shared encoder 和 2048-D feature cache；
5. 开始 Module A 完整效率实验；
6. 随后依次执行 Module B convergence、Module B Non-IID、Module C、End-to-End、TFE system；
7. 所有长任务进入 tmux；
8. CPU 线程始终为 1；
9. 使用两张 RTX 4090；
10. 不要在 smoke test 后停止；
11. 全部主线完成后再继续 DP-MTT。

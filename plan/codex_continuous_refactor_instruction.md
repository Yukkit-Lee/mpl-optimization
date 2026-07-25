# Codex 连续执行指令：依次修复四个跨阶段问题

你已经完成仓库审计。现在进入连续修改阶段。请从问题一开始，依次完成问题一、问题二、问题三、问题四。

## 一、执行原则

1. **不要再等待我逐项确认。**下面已经给出所有设计决策，请按顺序直接实施。
2. **不要在完成一个问题后停止。**每完成一个问题并通过测试后，立即继续下一个问题。
3. 如果某个子任务受 TFE 0.8.0、TensorFlow 1.15.5、缺失算子或环境兼容性阻塞：
   - 不要静默降级；
   - 在进度文件中标记 `BLOCKED` 并写明证据；
   - 实现能够诚实支持的最小方案；
   - 继续处理后续问题，不要因单个阻塞项停止整个任务。
4. 只有以下情况可以停止并向我报告：
   - 需要密码、密钥、账号或外部凭据；
   - 将执行不可逆的数据删除或大规模破坏性操作；
   - 仓库或运行环境损坏到无法继续；
   - 所有四个问题均已完成、测试并提交。
5. 禁止将 PyTorch/NumPy 运算命名为 `secure` 或 `MPC`。
6. 禁止 MPC 失败后静默回退到明文。
7. 禁止使用 test set 做候选选择、目标精度构造、超参数选择或早停。
8. 不要改动已经修复的 Stage A 独立 Figure 4 实验，除非为了共享工具和兼容性进行必要重构，且必须保证其结果逻辑不变。

---

## 二、固定设计决策

请直接采用以下决定，不要再询问。

### 决策 1：采用双轨方案

#### 算法主流程

使用完整 ConvNet 和蒸馏图像：

```text
Distilled Images
→ Local ConvNets
→ KD
→ Full-ConvNet Candidate Aggregation and Selection
→ Chosen ConvNet
→ PyTorch Adaptive DP-SGD on the Same ConvNet
→ Final ConvNet
```

算法主流程不再提取或缓存 2048 维特征。

#### 安全系统子实验

使用独立的轻量模型进行 TFE 代表性安全验证：

```text
Distilled Images
→ Flatten / Lightweight Linear or Small MLP
→ TFE Secret-Shared Training
→ Runtime and Communication Evaluation
```

TFE 子实验不得冒充完整 ConvNet 主实验。日志、表格和文件名必须明确区分：

- `algorithmic_pytorch_convnet`
- `secure_system_tfe_lightweight`

### 决策 2：Reference Set

当前 reference set 不得称为 public dataset。

请从训练集内部划分一个与以下集合互斥的 held-out reference/validation set：

- local training subsets；
- distilled training samples；
- test set。

统一命名为：

```text
heldout_reference_set
```

它只用于：

- teacher/reference predictions；
- candidate weighting；
- candidate selection；
- target accuracy 或 early-stopping 类指标。

Test set 只用于最终报告和预设间隔的只读评估，不能参与选择和调参。

由于 held-out reference set 来源于训练数据，隐私文档必须明确它属于私有数据依赖，当前不构成 public auxiliary data。

### 决策 3：候选池

主端到端候选池统一为：

1. `KD-Avg`
2. `KD-ReferenceWeighted`
3. `KD-Best`

`Prototype` 暂时移出主端到端候选池，可以保留为单独 ablation，但不得出现在主框架图、主端到端日志或主结果表中。

### 决策 4：隐私口径

短期统一采用：

```text
privacy_scope = stage_c_only
```

- Stage A：non-DP；
- Stage B：non-DP；
- Stage C：Fixed DP-SGD / Adaptive DP-SGD。

所有 epsilon 字段必须命名为：

```text
stage_c_epsilon
```

或：

```text
stage_c_epsilon_estimate
```

在正式 accountant 完成后使用前者；旧近似公式只能使用后者。

不得声称：

- end-to-end record-level DP；
- original-local-data DP；
- A+B+C total epsilon。

---

# 三、持续进度机制

立即在仓库根目录创建：

```text
CROSS_STAGE_REFACTOR_PROGRESS.md
```

包含以下 checklist：

```text
[ ] Phase 0: Baseline snapshot and branch
[ ] Phase 1: Remove 2048-D feature path from main A→B→C
[ ] Phase 2: Clarify and implement plaintext/MPC boundary
[ ] Phase 3: Add TFE secure-system Stage-C validation
[ ] Phase 4: Repair privacy accounting and test-set boundary
[ ] Phase 5: Final end-to-end regression and documentation
```

每完成一个子任务都更新该文件，记录：

- 修改文件；
- 执行命令；
- 测试结果；
- 输出文件；
- 已知限制；
- Git commit hash。

如果当前会话被中断，恢复后从第一个未勾选项继续。

创建独立分支：

```text
cross-stage-refactor
```

每个 Phase 至少一个独立 commit。不要把四个问题塞进一个巨大 commit。

---

# 四、Phase 0：基线与保护措施

1. 记录当前 Git 状态和未跟踪文件。
2. 不删除旧实验和旧缓存；将旧 2048-D 主流程标记为 legacy。
3. 保存当前关键结果、配置和日志路径。
4. 为新主流程使用新的输出目录和文件前缀，避免覆盖旧结果。
5. 运行当前 Stage A Figure 4 的最小 sanity test，保存基线。

完成后自动进入 Phase 1。

---

# 五、Phase 1：问题一——主流程改为蒸馏图像＋完整 ConvNet

## 目标

将最新主流程从：

```text
Distilled Images
→ ConvNet
→ 2048-D Features
→ LinearHead Stage B
→ Linear Stage C
```

改为：

```text
Distilled Images
→ Local ConvNets
→ KD ConvNets
→ Full-State Candidate Aggregation
→ Chosen ConvNet
→ Same ConvNet Adaptive DP-SGD
```

## 需要完成

### 1. 新增共享工具模块

新增一个官方 MTT 蒸馏图像和 ConvNet 的共享模块，例如：

```text
official_distilled_convnet_pipeline.py
```

统一提供：

- 数据加载；
- NCHW 预处理；
- held-out reference set 划分；
- ConvNetD3 / ConvNetD4 构造；
- state_dict 完整复制、平均、加权；
- 模型评估；
- checkpoint schema；
- SHA256 / state signature；
- 日志字段。

### 2. 修改最新 Stage B

修改：

- `exp_official_distilled_figure5_init.py`
- `exp_module_b_kd_noniid_official.py`

要求：

- 输入必须是蒸馏图像，不是二维 feature matrix；
- 本地模型必须是完整 ConvNet；
- KD 必须更新完整 ConvNet，或明确冻结哪些层并记录；
- `KD-Avg` 对完整 `state_dict` 做平均；
- `KD-ReferenceWeighted` 对完整 `state_dict` 做加权；
- `KD-Best` 选择完整 ConvNet；
- chosen checkpoint 保存完整模型结构和完整参数；
- Prototype 不进入主候选池。

### 3. 修改 Stage C PyTorch 主实验

修改：

- `exp_moduleC_dp_optimization.py`

要求：

- 输入为蒸馏图像；
- 加载 Stage B chosen ConvNet 的完整 checkpoint；
- Plain、Fixed-DP、Adaptive-DP 均训练同一 ConvNet；
- per-example gradient 必须覆盖所有可训练参数；
- 优先使用受控 microbatch 逐样本梯度，除非现有 PyTorch 环境可靠支持 `torch.func/vmap`；
- 记录每层梯度、总范数、裁剪比例、`C_t` 轨迹；
- 禁止读取 2048-D feature cache。

### 4. 修改端到端脚本

修改：

- `exp_end2end_framework.py`

要求：

- Stage B chosen ConvNet checkpoint 真实传递给 Stage C；
- checkpoint 加载前后 state signature 完全一致；
- 不允许重新随机初始化后只记录 `initialization_source`；
- A、B、C 使用同一 ConvNet class、同一 state-dict key schema；
- test set 不用于 target 构造和候选选择。

### 5. Legacy 隔离

旧 2048-D 脚本可以保留，但：

- 在文件头标记 `LEGACY_FEATURE_PIPELINE`；
- 不再被最新端到端脚本导入；
- 输出路径与新流程分离；
- README/进度文档说明其不属于当前主方法。

## Phase 1 测试

至少运行：

1. CIFAR-10，2 clients，每客户端 1–2 local steps；
2. KD 前后完整 ConvNet 参数变化；
3. full-state Avg/RW/Best；
4. chosen checkpoint 保存和加载；
5. Stage C Plain / Fixed / Adaptive 各 1–2 steps；
6. checkpoint SHA256 一致性；
7. no feature extraction assertion；
8. test/reference/train split 互斥断言。

Phase 1 通过后提交并自动进入 Phase 2。

---

# 六、Phase 2：问题二——明确并实现明文/MPC边界

## 最终边界

### 本地 PyTorch 明文

保留：

- local ConvNet training；
- local logits/probability generation；
- local KD；
- held-out reference predictions。

这些操作必须标记：

```text
backend = local_plaintext_pytorch
```

### 跨方 MPC 子实验

优先实现：

1. Secure Average；
2. Secure Reference-Weighted aggregation；
3. Secure candidate score aggregation；
4. Secure candidate selection，在 TFE 能力允许时实现。

不要把完整 ConvNet KD 搬入 MPC。

## 需要完成

1. 审计并重构 `secure_aggregation.py`。
2. 删除所有静默 fallback。
3. API 必须显式接收：

```text
backend = plaintext | tfe_mpc
allow_plaintext_fallback = false
```

默认不得 fallback。

4. 明文结果和 MPC 结果使用不同输出字段和目录。
5. 如果 TFE 只能在 reveal 后完成 argmax：
   - 不得命名为 `secure_selection`；
   - 命名为 `secure_score_computation_with_revealed_selection`；
   - 记录 reveal 的对象；
   - 在限制说明中明确。
6. 若完整 ConvNet 聚合在 TFE 中不可行，使用轻量模型作为代表性系统验证，但不得修改算法主流程。

## Phase 2 验收

日志必须打印：

```text
backend
protocol
secret_shared_inputs
revealed_values
fallback_used=false
selection_security_scope
```

所有带 `secure_` 前缀的函数内部不得使用 NumPy/PyTorch 完成核心跨方运算。

Phase 2 完成后提交并自动进入 Phase 3。

---

# 七、Phase 3：问题三——Stage C 的 TFE 安全系统子实验

## 目标

新增独立的 TFE 轻量模型系统实验，不替代 PyTorch ConvNet 主实验。

建议新增：

```text
exp_stageC_tfe_secure_system.py
```

## 实施顺序

### Step 1：Non-private secure sanity check

- 蒸馏图像直接 Flatten，或使用 TFE 可支持的小型 MLP；
- secret-shared inputs；
- secret-shared model parameters；
- secure forward/update；
- 验证 loss 和 accuracy 有合理变化。

### Step 2：Fixed-DP baseline

实现能够真实运行的固定裁剪安全训练。

如果精确 per-example norm 和 comparison 无法在 TFE 中实现：

- 先实现并明确命名可支持的近似或受限版本；
- 不得称为完整 secure DP-SGD；
- 记录具体缺失原语。

### Step 3：Adaptive clipping

尝试实现：

- gradient norm；
- clipping statistics；
- `C_t → C_{t+1}`；
- 当前 `C_t` 用于本轮裁剪，`C_{t+1}` 用于下一轮；
- 不允许算法顺序错误。

若 TFE 确实无法实现，新增一个独立 operator/microbenchmark，证明哪些原语可执行、哪些被阻塞，并继续后续任务。

### Step 4：Noise backend

区分：

- `numpy_binomial_simulation`
- `tfe_mpc_noise_generation`

当前 `np.random.binomial` 不得命名为 BinomialMPC。

如果安全随机 custom op 与 TF 1.15.5 不兼容：

- 记录为 BLOCKED；
- 不得伪造安全随机结果；
- Gaussian 或 binomial 的系统结果必须注明随机性来源。

### Step 5：系统指标

报告：

- runtime；
- communication bytes；
- communication rounds，如果可获取；
- input size；
- model size；
- protocol；
- LAN/WAN 配置；
- reveal 边界。

Phase 3 完成后提交并自动进入 Phase 4。

---

# 八、Phase 4：问题四——隐私会计与测试集边界

## 1. 修复 test leakage

禁止：

- 用初始 test accuracy 构造 target accuracy；
- 用 test accuracy 选择 chosen；
- 用 test accuracy 调 clipping 参数；
- 用 test accuracy 早停。

改为：

- target 基于 held-out reference/validation set；
- test 仅用于固定间隔只读曲线和最终报告；
- 最终论文指标区分 validation 和 test。

## 2. 替换 epsilon 近似公式

优先使用环境中可用的正式 accountant：

1. Opacus RDP accountant；
2. `dp-accounting`；
3. 若均不可用，实现经过单元测试的 RDP accountant。

要求：

- 记录 sample rate、steps、noise multiplier、delta；
- Fixed-DP 与 Adaptive-DP 使用一致 accountant；
- 不得把 Gaussian accountant 直接用于 binomial noise，除非有正式对应界或证明；
- 对 binomial simulation 无法正式计账时标记 `epsilon=N/A`，不得复用 Gaussian epsilon。

## 3. 隐私字段

统一输出：

```text
privacy_scope = stage_c_only
stage_c_epsilon
stage_c_delta
accountant_type
noise_mechanism
```

Stage A/B：

```text
dp_protected = false
```

## 4. 论文与日志警告

自动生成一段 limitation 文本：

> The reported privacy budget covers Stage C only. Stages A and B are data-dependent and are not included in the current record-level DP accounting; therefore, the reported epsilon does not constitute end-to-end DP for the original local datasets.

## Phase 4 测试

- accountant 单元测试；
- epsilon 随 steps 增大；
- epsilon 随 noise multiplier 增大而减小；
- Fixed/Adaptive 相同设置得到一致会计口径；
- test set 不参与选择；
- 所有结果表包含 `Privacy Scope`。

Phase 4 完成后提交并自动进入 Phase 5。

---

# 九、Phase 5：最终回归、文档与框架图数据源

1. 运行三个数据集的最小端到端 smoke test。
2. 至少运行 CIFAR-10 的完整可行实验。
3. 生成统一结果 JSON/CSV：
   - Stage A；
   - Stage B；
   - Stage C PyTorch；
   - TFE secure-system validation；
   - privacy scope；
   - runtime/communication。
4. 更新 README 或实验说明，明确双轨结构。
5. 生成最终“真实实现边界”文字框架：

```text
Algorithmic main pipeline:
Distilled Images
→ PyTorch ConvNet Stage B
→ Chosen Full ConvNet
→ PyTorch ConvNet Adaptive DP-SGD

Secure-system validation:
Distilled Images
→ TFE Lightweight Model
→ Secret-Shared Training

Privacy:
Stage-C-only DP accounting
```

6. 列出仍未完成的安全原语和论文限制。
7. 运行格式检查、导入检查和所有可运行测试。
8. 最终提交。

---

# 十、最终回复格式

所有工作结束后，只按下面格式汇报：

```text
1. Completed Phases
2. Git Commits
3. Files Added/Modified
4. Tests and Results
5. Final Data and Model Flow
6. Final Plaintext/MPC Boundary
7. Final Privacy Scope and Accountant
8. Remaining BLOCKED Items
9. Exact Commands to Reproduce
10. Paper Claims That Are Now Supported
11. Paper Claims That Must Still Be Avoided
```

现在立即开始 Phase 0。除非满足允许停止的条件，否则不要在 Phase 1、2、3 或 4 完成后停下来等待确认。

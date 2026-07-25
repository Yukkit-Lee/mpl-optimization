# 给 Codex 的跨阶段代码审计与重构任务说明

## 0. 当前进度与总体目标

我正在实现一个面向差分隐私多方学习（MPL）的三阶段训练流程：

1. **Stage A：Dataset-Distilled Compact Image Set Construction**
2. **Stage B：Consensus-Guided Global Initialization**
3. **Stage C：Adaptive Private Optimization**

目前已经完成对 **Stage A 独立实验** 的修正：

- 使用官网发布的 MTT `ipc=50` 蒸馏图像；
- 不提取 2048 维特征；
- 不训练浅层分类器；
- 比较以下三种设置：
  1. Full Images → ConvNet Training
  2. Randomly Sampled Images → Same ConvNet Training
  3. Distilled Images → Same ConvNet Training
- 三组实验使用相同 MTT ConvNet、相同 optimizer、相同 learning rate、相同随机初始化；
- 横轴为 Optimizer Steps，纵轴为 Test Accuracy；
- 主图不标注 epsilon；
- 汇总样本数、压缩率、Final Accuracy、Step@TargetAcc 和训练时间；
- 该实验只验证 Stage A，不接入 Stage B/KD，也不接入 Stage C/DP。

**重要说明：这只修复了 Stage A 的独立消融实验，并不代表端到端 A→B→C 数据流已经完成修正。**

总体希望形成的研究故事是：

> 蒸馏图像减少进入后续训练的样本量；共识初始化减少达到目标准确率所需的训练步数；自适应私有优化降低固定隐私预算下的模型效用损失。

主方法应尽量与 PEA 的“特征提取＋浅层模型”路线区分，因此不要在主流程中重新引入 2048 维特征提取，除非经过明确论证并作为单独系统实现折中，而不是方法创新。

---

# 1. 工作方式要求

请不要一次性重写全部代码。按照下面四个问题逐项处理。

对每个问题，先执行以下步骤：

1. **只审计，不修改代码；**
2. 列出相关脚本、函数、缓存文件和数据流；
3. 说明当前实现与论文目标的差异；
4. 提供两到三个可行修改方案；
5. 比较工作量、实验影响、MPC 实现难度和隐私声明；
6. 推荐最小可行方案；
7. 等我确认后再修改代码；
8. 修改后运行最小测试、单模块测试和端到端测试；
9. 不允许静默回退到明文实现；
10. 日志必须明确记录当前 backend、数据形态、模型结构、是否使用 MPC、隐私预算覆盖范围。

请优先保证：

- 代码、实验表格、框架图和论文描述一致；
- 不把普通 PyTorch/NumPy 运算称为安全计算；
- 不把 Stage C 的 epsilon 写成整个原始数据流程的端到端 epsilon；
- 不把尚未进入最终端到端脚本的候选策略画入主框架图；
- 不为了保持旧缓存而保留错误的数据流。

---

# 2. 问题一：Stage B/C 应处理蒸馏图像，还是 2048 维特征？

## 当前已知状态

Stage A 独立实验已经改成：

```text
Full / Random / Distilled Images
→ Same ConvNet
→ Accuracy comparison
```

但是原有端到端代码中可能仍存在：

```text
Distilled Images
→ Train ConvNet
→ Extract 2048-D Features
→ Stage B linear models
→ Stage C linear classifier
```

因此，需要确认 Stage A 的修改是否只影响 Figure 4 脚本，还是已经影响完整 A→B→C 数据流。

## 目标倾向

为了与 PEA 的特征提取路线明确区分，主算法流程优先考虑：

```text
Stage A:
Original / Benchmark Data
→ Distilled Images D_i^syn

Stage B:
D_i^syn
→ Local ConvNet Training
→ KD / Candidate Initializations
→ Selected ConvNet Initialization theta_B

Stage C:
Logically Global Distilled Images D^syn
+ theta_B
→ Adaptive DP-SGD on the same model
→ Final Model theta_T
```

## 请 Codex 首先审计

请搜索并报告：

- 所有 `feature_dim=2048`、`penultimate feature`、`extract_features`、`X_train_feat`、`X_test_feat`；
- 所有特征缓存 `.npy/.npz/.pt`；
- Stage B 当前模型是 ConvNet、MLP 还是 Linear；
- Stage C 当前模型是 ConvNet、MLP 还是 Linear；
- Stage B 输出的权重是否能直接被 Stage C 加载；
- A、B、C 是否使用同一种模型结构；
- 当前 bridge 脚本是否只传递线性分类器权重；
- Figure 5、KD non-IID、Module C 和 end-to-end 脚本的数据输入是否一致。

## 需要 Codex 输出

先不要改代码，只输出：

1. 当前真实数据流图；
2. 当前模型结构流图；
3. 所有需要修改的文件清单；
4. 直接使用蒸馏图像贯通 B/C 的最小方案；
5. 若 TFE 无法支持完整 ConvNet，需要明确区分：
   - PyTorch 算法主实验；
   - TFE 安全系统子实验；
6. 说明两套实验是否会造成“跨阶段模型不一致”，以及如何在论文中诚实解释。

## 验收标准

修改后至少应满足：

- Stage A 独立实验不再提取 2048 维特征；
- 主端到端实验中，B 的输出模型能作为 C 的真实初始化；
- 不允许只把准确率数字串起来，却没有真正传递模型参数；
- 框架图中的输入形式与代码一致；
- 日志打印：
  - input type；
  - input shape；
  - model class；
  - initialization source；
  - whether feature extraction is used。

---

# 3. 问题二：KD、教师概率、加权聚合和选择是否进入 MPC？

## 当前已知状态

当前以下操作由普通 PyTorch/NumPy 明文执行：

- 本地模型训练；
- 教师概率或 logits；
- KD；
- Average 聚合；
- Reference-Weighted 聚合；
- Best 候选；
- 候选评分和选择。

这意味着单一运行程序可以看到各方完整模型、预测、权重和候选分数。

## 需要明确的概念

“放入 MPC”不是把方框画到蓝色区域，而是：

- 将本地模型、logits、权重或分数转换为秘密份额；
- 使用 TFE/SPU 支持的安全加法、乘法、比较和选择；
- 任意单一计算方不能看到完整输入和中间结果。

MPC 的作用是保护跨方知识融合过程，不直接提高准确率，也不替代 DP。

## 推荐的现实边界

不要优先把完整 KD 训练全部搬入 MPC，因为安全 softmax、KL 散度和反向传播成本很高。

优先考虑：

```text
Local plaintext at each party:
- local model training
- local logits/probability generation on a public reference set
- optional local KD

MPC cross-party computation:
- secure average aggregation
- secure weighted aggregation
- secure candidate scoring
- secure candidate selection
```

## 请 Codex 首先审计

请报告：

- KD 的 teacher/student 分别是谁；
- teacher probabilities 在哪里生成；
- reference set 是公共数据还是训练数据子集；
- Reference-Weighted 权重如何计算；
- Best 如何选择；
- 哪些数据会跨参与方边界；
- 当前是否存在 `use_secure_aggregation`；
- 关闭安全聚合时是否静默回退到明文；
- 已有 `secure_aggregation.py` 实现了哪些真实安全操作，哪些只是模拟。

## 需要 Codex 输出

1. 一张“本地明文操作 / 跨方 MPC 操作”边界表；
2. 最小 MPC 化方案；
3. 需要在 TFE 中实现的安全原语：
   - sum / average；
   - weighted sum；
   - score computation；
   - comparison / argmax；
4. 哪些操作因 TFE 0.8.0 限制难以完成；
5. 是否建议先实现：
   - Secure Avg；
   - Secure Reference-Weighted；
   - Secure Best Selection；
6. 禁止将明文 fallback 结果标记为 secure。

## 验收标准

- 日志必须显示 `backend=plaintext` 或 `backend=tfe_mpc`；
- 不能用同一个字段同时代表明文和密态结果；
- MPC 失败时应报错退出，不能静默回退；
- 图中只有真实通过 MPC 执行的模块才能带锁或放在“Secret-Shared Computation”区域；
- 主候选池必须与最终端到端代码一致。

---

# 4. 问题三：Stage C 当前是 PyTorch 明文训练，如何接入 TFE？

## 当前已知状态

当前 Stage C 中以下对象均为普通 PyTorch/NumPy 明文变量：

- 模型参数 `W, b` 或 ConvNet 参数；
- per-example gradients；
- gradient norms；
- clipping bound `C_t`；
- Gaussian / binomial noise；
- model update。

因此目前 Stage C 只能证明 Adaptive DP-SGD 的算法效用，不能证明安全训练已经在 TFE 中执行。

## 需要明确的概念

TF Encrypted 是基于 TensorFlow 的 MPC 框架，但普通 PyTorch 代码不会自动变成秘密共享计算。

要接入 TFE，需要使用：

- TFE private/secret-shared tensors；
- TFE 支持的安全运算；
- 多方执行配置；
- 在秘密份额状态下完成前向、梯度、裁剪、加噪和更新。

## 请 Codex 先评估两条路线

### 路线 A：同一 ConvNet 的安全训练

```text
Distilled Images
+ Stage-B ConvNet Initialization
→ TFE Secure Adaptive DP-SGD ConvNet
```

优点：跨阶段最完整、与 PEA 区别最清晰。  
缺点：TFE 0.8.0 中安全卷积、单样本梯度和动态裁剪可能成本极高。

### 路线 B：算法主实验与安全子实验分离

```text
Algorithmic main evaluation:
Distilled Images + ConvNet in PyTorch

Secure-system validation:
Distilled Images → Flatten / lightweight model
→ TFE Fixed-DP / Adaptive-DP
```

优点：更现实，能获得真实 MPC 时间和通信结果。  
缺点：系统实验与主算法模型不完全一致，论文需要明确说明是代表性安全实现。

## 请 Codex 输出

1. 当前 TFE 0.8.0 环境可支持的模型与算子清单；
2. ConvNet 安全训练的具体阻碍；
3. 线性模型或小型 MLP 的最小安全实现方案；
4. Fixed-DP TFE baseline 的实现步骤；
5. Adaptive clipping 需要的安全原语：
   - norm；
   - comparison；
   - clipping factor；
   - clipping statistics；
   - update of `C_t`；
6. Binomial noise 后端是否真实在 MPC 中生成；
7. 预计改动文件和测试顺序；
8. 不得将 PyTorch 结果标为 TFE 结果。

## 推荐实施顺序

```text
Step 1: TFE non-private secure training sanity check
Step 2: TFE fixed-clipping DP baseline
Step 3: TFE adaptive clipping
Step 4: MPC-oriented binomial noise
Step 5: runtime and communication measurement
```

## 验收标准

- secret-shared input；
- secret-shared model parameters；
- secure forward/update；
- 明确记录实际 backend；
- 固定裁剪和自适应裁剪在相同隐私设置下可比较；
- 给出真实运行时间和通信数据；
- 不能只用函数名 `secure_*` 但内部仍是 NumPy/PyTorch。

---

# 5. 问题四：只统计 Module C 的 epsilon 是否可以？

## 当前已知状态

当前隐私会计只覆盖 Stage C。

但是：

- MTT 蒸馏图像依赖训练数据；
- Stage B 本地模型依赖蒸馏图像或原始训练数据；
- KD 教师概率依赖本地模型；
- 加权聚合和初始化选择依赖参考集及本地模型；
- 最终模型从数据相关初始化开始训练。

因此，Stage C 的 `epsilon_C` 不能自动解释为相对于原始本地数据的端到端记录级 DP。

MPC 只保护计算过程，不能替代 DP。

## 当前可以采用的三个隐私口径

### 口径 A：只声明 Stage-C DP

```text
Stage A: non-DP
Stage B: non-DP
Stage C: Adaptive DP-SGD, epsilon_C
```

论文必须明确：

> Reported epsilon covers Stage C only and does not constitute end-to-end record-level DP for the original local datasets.

这是工作量最小、最诚实的方案。

### 口径 B：Stage B + Stage C

```text
Stage B local models: DP-SGD, epsilon_B
Stage C final model: Adaptive DP-SGD, epsilon_C
```

总预算使用正式隐私会计组合，而不是只在文字中简单相加。

注意：如果 Stage A 仍由私有数据生成，仍不能直接声称完整端到端 DP。

### 口径 C：A/B/C 全部纳入 DP

```text
Stage A: DP dataset distillation, epsilon_A
Stage B: DP local knowledge, epsilon_B
Stage C: Adaptive DP-SGD, epsilon_C
```

隐私最完整，但工作量最大，不建议在未评估可行性前直接实施。

## 关于当前官方 MTT 数据

当前 Figure 4 使用官网发布的 MTT 蒸馏图像。请区分：

- **当前 benchmark 实验**：使用公开发布的蒸馏图像，不需要为 Figure 4 标 epsilon；
- **论文真实多机构场景**：若各方根据自己的私有数据生成蒸馏图像，则 Stage A 是私有数据相关机制，不能忽略其隐私边界。

## 请 Codex 首先审计

1. Stage A 输入数据来源；
2. Stage B 本地模型实际使用原始图像、蒸馏图像还是特征；
3. reference set 来源；
4. Stage C 的采样率、batch size、steps、noise multiplier、delta；
5. 当前 accountant 的实现和输出；
6. 哪些实验表中的 epsilon 实际只属于 Stage C；
7. 是否存在错误的 `epsilon_total` 命名。

## 需要 Codex 输出

1. 隐私数据依赖图；
2. 每个阶段是否访问私有数据；
3. 每个阶段当前是否有 DP；
4. 当前 epsilon 的准确含义；
5. 建议采用的隐私声明；
6. 需要修改的日志字段、表格列名和论文用语；
7. 不要擅自加入一个形式上的 epsilon，而没有对应 DP 机制和 accountant。

## 推荐短期方案

优先采用：

```text
Figure 4 / Stage A:
No epsilon, algorithmic utility only

Stage B:
暂时明确为 non-DP 或后续单独增加 DP-SGD

Stage C:
报告 Stage-C epsilon_C

Paper claim:
不声称原始数据级端到端 DP
```

完成 B 的 DP-SGD 或 DP 知识机制后，再重新计算组合预算。

## 验收标准

- 所有实验表明确写 `Privacy Scope`；
- 使用 `Stage-C epsilon`，而不是含糊的 `epsilon`；
- 非私有方法写 `infinity` 或 `N/A`，二者统一；
- Figure 4 不标 epsilon；
- 摘要中不能把 `epsilon_C` 写成整个三阶段流程对原始数据的端到端预算；
- Limitation 中说明 Stage A/B 尚未纳入隐私会计，直到对应机制完成。

---

# 6. 四个问题的处理顺序

请严格按以下顺序进行，不要并行大改：

## 第一步：问题一——统一数据形态和模型传递

先解决：

- B/C 是否直接使用蒸馏图像；
- A、B、C 是否采用同一模型；
- Stage B 输出是否真实初始化 Stage C。

这是跨阶段流程能否成立的基础。

## 第二步：问题二——划分明文与 MPC 边界

在模型和数据流稳定后，确定：

- 本地明文操作；
- 跨方秘密共享操作；
- 哪些 Stage B 操作优先迁移到 TFE。

## 第三步：问题三——Stage C 接入 TFE

优先实现真实的安全训练和运行开销，不要先花大量时间做完整安全 KD。

## 第四步：问题四——重新定义隐私预算

最后根据 A/B/C 的真实数据访问和安全实现，确定：

- epsilon_A；
- epsilon_B；
- epsilon_C；
- 论文允许声称的隐私范围。

---

# 7. Codex 第一次回复的格式

这次先不要提交代码改动。请按以下格式回复：

```text
A. Repository Audit
- Relevant files
- Current data flow
- Current model flow
- Current privacy accounting
- Current MPC/plaintext boundary

B. Problem 1 Analysis
- Current behavior
- Mismatch
- Options
- Recommended minimal plan
- Files to change
- Risks

C. Problems 2–4 Preview
- Only list dependencies and blockers
- Do not modify yet

D. Proposed Test Plan
- Unit tests
- Module tests
- End-to-end tests
- Logging changes

E. Questions Requiring My Confirmation
```

完成问题一的审计并得到确认后，再开始修改问题一。之后依次处理问题二、问题三和问题四。

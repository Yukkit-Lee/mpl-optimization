# Codex Experiment Task: MTT-Anchor Post-Aggregation Supervised Refinement

## 0. 任务目标

实现并验证一个仅作用于 Stage B 的模块：

> MTT-Anchor Post-Aggregation Supervised Refinement  
> 基于 MTT 锚点的聚合后监督修正

完整流程：

```text
Official MTT 紧凑本地特征
        ↓
5 个客户端独立训练轻量分类头
        ↓
TFE/ABY3 Secure Average
        ↓
在类别均衡、带标签、与本地训练集互斥的 MTT Anchor 上
执行少量监督式交叉熵再优化
        ↓
最终全局 Stage-B Initializer
```

本实验的核心不是证明 Secure Average 本身新颖，而是回答：

1. Secure Average 是否为 Anchor CE 提供了有效的跨客户端知识起点？
2. Secure Average + Anchor CE 是否优于 Random Initialization + Anchor CE？
3. Secure Average + Anchor CE 是否优于 Best Local Head + Anchor CE？
4. Anchor CE 的收益是否只是增加训练预算，还是依赖跨方聚合后的初始化？
5. 该方法能否在真实 TFE/ABY3 中复现，并保持合理的 Stage-B 运行成本？

本任务禁止运行 Stage C 和 End-to-End。

---

# 1. 硬性限制

不得运行或修改：

- Stage C；
- Adaptive DP-SGD；
- Fixed DP-SGD；
- End-to-End；
- Figure 2；
- T@80 / T@90 / T@95；
- official test；
- locked holdout；
- Full-Data initializer；
- seeds 43 / 44；
- Tiny-ImageNet；
- 现有 True Ours 正式主结果。

当前实验范围：

```text
datasets:
  CIFAR-10
  CIFAR-100

seed:
  42 only

clients:
  5

distributions:
  IID
  Mild Non-IID
  Severe Non-IID

stage:
  Stage B only
```

执行顺序：

```text
先做低成本 plaintext smoke test
        ↓
通过因果门槛后冻结配置
        ↓
再做正式 TFE/ABY3 Stage-B 验证
```

---

# 2. 方法定义

## 2.1 客户端本地模型

每个客户端只使用自己的 Official MTT 本地紧凑特征训练轻量分类头：

```text
local MTT features
→ local lightweight classifier head
```

所有客户端必须使用：

- 相同特征编码器；
- 相同特征维度；
- 相同模型结构；
- 相同初始参数；
- 相同优化器类型；
- 相同训练轮数；
- 相同学习率策略。

客户端之间只允许数据划分不同。

## 2.2 Secure Average

客户端本地模型参数通过 TFE/ABY3 进行安全平均：

\[
\theta_{\mathrm{avg}}
=
\frac{1}{K}
\sum_{i=1}^{K}\theta_i
\]

Secure Average 是基础算子，不得写成本文首创。

## 2.3 MTT Anchor 监督修正

从与客户端本地训练集合互斥的 Official MTT 数据中构造类别均衡 Anchor：

```text
Anchor features + ground-truth labels
```

从指定初始化模型 \(\theta_0\) 开始，优化：

\[
\mathcal{L}_{\mathrm{anchor}}
=
\frac{1}{|A|}
\sum_{(x,y)\in A}
\mathcal{L}_{CE}(f_\theta(x),y)
+
\mu\|\theta-\theta_0\|_2^2
\]

其中：

- \(\theta_0\) 可以是 Random、Best Local、Secure Average；
- \(\mu\) 用于限制修正过程偏离原初始化过远；
- 所有初始化来源必须使用完全相同的 Anchor、步数、学习率和正则预算。

最终主方法：

```text
Secure Average
→ Uniform Class-Balanced Anchor CE
→ Final Stage-B Initializer
```

---

# 3. 数据谱系与隔离

必须使用审计通过的 Official MTT 特征。

构造并验证三个互斥集合：

```text
client-local training pool
class-balanced labeled MTT anchor pool
development evaluation split
```

优先复用现有已审计划分：

```text
40 samples/class: client-local pool
8 samples/class: anchor pool
```

必须验证：

```text
client-local ∩ anchor = empty
client-local ∩ development = empty
anchor ∩ development = empty
```

禁止：

- 使用 development 样本训练；
- 读取 test 或 locked holdout；
- 加载 Full-Data initializer；
- 使用 hybrid initializer；
- 将 Anchor 同时放入客户端本地训练。

生成或复用：

```text
new_stage_b_anchor_postagg_split_manifest.json
new_stage_b_anchor_postagg_split_audit.md
```

每个结果文件必须包含：

```text
official_mtt_only = true
full_data_initializer_dependency = false
test_read = false
locked_holdout_read = false
split_sha256
config_sha256
model_sha256
source_git_commit
```

---

# 4. 分支与输出隔离

创建独立分支：

```text
new-stage-b-mtt-anchor-postagg
```

所有新文件写入：

```text
new_scripts/new_stage_b_anchor_postagg/
new_configs/new_stage_b_anchor_postagg/
new_log/new_stage_b_anchor_postagg/
new_expTab/new_stage_b_anchor_postagg/
new_expFig/new_stage_b_anchor_postagg/
new_output/new_stage_b_anchor_postagg/
new_checkpoints/new_stage_b_anchor_postagg/
```

不得覆盖：

- 既有 Secure Average；
- 既有 KD/Consensus；
- 既有 disagreement experiments；
- 既有 heterogeneity-trigger experiments；
- 既有 True Ours 主结果。

开始前记录：

```bash
git status --short --branch
git rev-parse HEAD
git diff --stat
```

---

# 5. 优先复用已有结果

先检查 COMPLETE 工件，精确复用：

- Secure Average；
- Uniform Anchor CE；
- KD 负对照；
- IID / Mild / Severe 的既有 B0/B1；
- 已有 plaintext 与 TFE Stage-B 结果。

仅当以下任一项不一致时才重跑：

- dataset；
- seed；
- split；
- client partition；
- local-head config；
- anchor config；
- evaluation split；
- backend；
- checkpoint hash；
- fixed-point precision。

生成：

```text
new_stage_b_anchor_postagg_reuse_audit.json
```

---

# 6. 核心对比组

## C0：Random Initialization

```text
Random Initialization
→ 不做 Anchor CE
```

## C1：Random + Anchor CE

```text
Random Initialization
→ Uniform Anchor CE
→ Initializer
```

用于判断 Anchor CE 是否可以完全绕过多方知识聚合。

## C2：Best Local Head

选择 development 不可见条件下的预注册本地头选择规则。

优先规则：

```text
按本地训练目标或固定客户端编号选择
```

禁止使用 development accuracy 挑选最佳客户端。

若当前代码只能通过 development 选 Best Local，则该组只能标记为 Oracle Diagnostic，不可作为正式方法。

## C3：Best Local Head + Anchor CE

```text
Best Local Head
→ Uniform Anchor CE
→ Initializer
```

用于判断单个客户端知识是否已经足够。

## C4：Secure Average

```text
Local Heads
→ Secure Average
→ Initializer
```

对应基础多方初始化。

## C5：Secure Average + Anchor CE

```text
Local Heads
→ Secure Average
→ Uniform Anchor CE
→ Final Initializer
```

这是主方法。

## C6：Centralized Anchor-Only Oracle

可选诊断组：

```text
预注册标准初始化
→ 使用完整 Anchor 预算训练
```

仅用于估计 Anchor 本身的上限，不进入正式主结论。

---

# 7. 最关键的因果比较

## 7.1 Anchor 是否有效

```text
C1 vs C0
C5 vs C4
```

## 7.2 Secure Average 是否有独立作用

```text
C5 vs C1
```

这是最关键比较。

若 C5 明显优于 C1，说明跨客户端聚合为 Anchor 修正提供了更好的起点。

## 7.3 Secure Average 是否优于单客户端知识

```text
C5 vs C3
```

## 7.4 是否只是增加了训练步数

对 C1、C3、C5 使用：

- 相同 Anchor；
- 相同步数；
- 相同 batch size；
- 相同学习率；
- 相同正则；
- 相同 wall-clock 计时边界。

另外绘制：

```text
Anchor CE step
vs
development accuracy
```

若 C5 更早达到同一精度，也可证明 Secure Average 初始化有效。

---

# 8. Phase A：Plaintext Smoke Test

## 8.1 初始范围

运行：

```text
CIFAR-10
CIFAR-100
seed42
IID
Mild Non-IID
Severe Non-IID
plaintext float
```

优先从已有检查点继续，不重复训练已完成本地头。

## 8.2 Smoke Step S0：基线复核

先确认或复用：

```text
C0 Random
C1 Random + Anchor CE
C4 Secure Average
C5 Secure Average + Anchor CE
```

记录：

- dev accuracy；
- macro-F1；
- worst-class accuracy；
- per-class accuracy；
- Anchor CE loss curve；
- refinement runtime；
- total Stage-B runtime；
- peak RSS。

## 8.3 Smoke Step S1：低成本 Anchor 预算筛选

只在 CIFAR-10 Severe 和 CIFAR-100 Severe 上筛选。

顺序调参，不做全笛卡尔积：

```text
refinement steps:
  5, 10, 20, 40

learning-rate multiplier:
  0.25, 0.5, 1.0

regularization mu:
  0, 1e-4, 1e-3
```

调参顺序：

```text
steps
→ learning rate
→ mu
```

筛选标准：

1. 优先看 C5−C1；
2. 再看 C5−C3；
3. 再看 C5 达到目标精度所需步数；
4. 不允许只根据 C5 单独最高精度选配置。

## 8.4 Smoke Step S2：初始化来源验证

使用冻结的 Anchor CE 配置运行：

```text
C1 Random + Anchor CE
C3 Best Local + Anchor CE
C5 Secure Average + Anchor CE
```

覆盖：

```text
IID
Mild
Severe
```

所有组必须使用完全一致的修正预算。

## 8.5 Smoke Step S3：步数效率

记录每一步或每若干步的：

```text
development accuracy
macro-F1
worst-class accuracy
CE loss
distance from initialization
elapsed time
```

冻结目标精度：

```text
target_80 = C5 最终精度的 80%
target_90 = C5 最终精度的 90%
target_95 = C5 最终精度的 95%
```

仅用于 Stage-B 内部比较，不得称为 End-to-End T@Target。

比较 C1、C3、C5 达到这些目标所需的：

- Anchor CE steps；
- Anchor CE seconds。

---

# 9. Smoke 成功门槛

## Gate A：Secure Average 独立贡献

至少在两个数据集中的一个满足：

```text
C5 >= C1 + 1.0pp
```

并且另一个数据集不低于：

```text
C5 >= C1 - 0.3pp
```

或者：

```text
C5 与 C1 最终精度差距 <= 0.3pp
但 C5 达到 90% 目标所需步骤减少 >= 30%
```

## Gate B：多方聚合优于单客户端

至少在两个数据集中的一个满足：

```text
C5 >= C3 + 1.0pp
```

并且另一个不出现明显下降。

## Gate C：Anchor 修正确实有效

```text
C5 > C4
C1 > C0
```

## Gate D：跨分布稳定

主结论至少在：

```text
IID / Mild / Severe 中的 2 种分布
```

方向一致。

## Gate E：类别稳健性

满足至少一项：

```text
macro-F1 提高 >= 1.0pp
worst-class accuracy 提高 >= 2.0pp
改善类别数超过总类别数的 60%
```

## Gate F：数据谱系

```text
Official-MTT-only
Full-Data dependency = false
test/holdout unread
```

若 C5 与 C1 基本等价，且无步数/时间优势，则停止，不进入 TFE，并报告：

> 主要收益来自 Anchor CE，Secure Average 未表现出独立贡献。

---

# 10. Phase A 的停止与诊断规则

## 情况 1：C1 ≈ C5

只允许继续检查：

- 更少修正步数；
- 更小学习率；
- 更强初始化正则。

目标是验证 C5 是否具有更快的早期收敛。

如果仍无优势，停止。

## 情况 2：C5 最终更好，但早期更慢

检查：

- Secure Average 与 Anchor 分布是否存在尺度不匹配；
- feature normalization；
- classifier bias initialization；
- regularization 是否过强。

每次只调整一项。

## 情况 3：C5 在 Severe 有效、IID 无效

允许将方法定位为 Non-IID 扩展，但不得写成通用模块。

## 情况 4：C5 在所有分布稳定优于 C1/C3

冻结配置，进入 TFE。

## 情况 5：Best Local 组选择不合法

若 Best Local 使用 development accuracy 选择，则：

```text
C3 = Oracle diagnostic only
```

不得作为正式比较结论。

---

# 11. Freeze Before TFE

Smoke 通过后生成：

```text
new_stage_b_anchor_postagg_config_freeze.json
new_stage_b_anchor_postagg_split_freeze.json
new_stage_b_anchor_postagg_freeze_manifest.md
```

冻结：

- dataset；
- seed42；
- client split；
- Anchor split；
- local-head config；
- Random initialization seed；
- Best Local selection rule；
- Anchor CE steps；
- Anchor CE learning rate；
- regularization；
- batch size；
- feature normalization；
- TFE fixed-point precision；
- timing boundary；
- success gates。

冻结后不得继续针对 TFE 结果调参。

---

# 12. Phase B：正式 TFE/ABY3 Stage-B 验证

仅在 Smoke 通过后运行。

## 12.1 范围

```text
datasets:
  CIFAR-10
  CIFAR-100

seed:
  42 only

distributions:
  优先 Severe Non-IID
  若耗时允许，再运行 IID

methods:
  C1 Random + Anchor CE
  C4 Secure Average
  C5 Secure Average + Anchor CE

backend:
  TFE/ABY3
```

C3 只在选择规则合法且 Smoke 有必要时运行。

## 12.2 实现阶梯

```text
P1 plaintext float
P2 plaintext fixed-point emulation
P3 TFE Secure Average
P4 TFE Anchor forward/loss
P5 TFE Anchor CE update
P6 final TFE Stage-B initializer
```

每一步记录与 plaintext 的数值差异。

## 12.3 TFE 正确性要求

必须满足：

- backend 确认为 TFE/ABY3；
- 无 plaintext fallback；
- Secure Average 确实在秘密共享环境执行；
- Anchor CE 正式路径与 frozen config 一致；
- 无 NaN；
- 无 overflow；
- 固定点权重有限；
- 模型 SHA 正确；
- split SHA 正确；
- test/locked holdout 未读取；
- Official-MTT-only；
- Full-Data dependency=false。

## 12.4 TFE 成功门槛

```text
TFE(C5−C1)
至少保留 plaintext(C5−C1) 的 80%
```

或者：

```text
C5 与 C1 最终精度接近
但 C5 达到冻结目标所需 Anchor steps/time 少 >= 25%
```

同时：

```text
TFE C5 > TFE C4
```

---

# 13. 运行时间与资源记录

每个方法记录：

- local-head training time；
- Secure Average time；
- Anchor data loading time；
- Anchor forward time；
- Anchor CE backward/update time；
- total Stage-B time；
- peak RSS；
- active Anchor samples；
- refinement steps；
- output model SHA。

生成：

```text
new_stage_b_anchor_postagg_runtime_breakdown.csv
new_stage_b_anchor_postagg_resource_summary.csv
```

不得报告：

- Stage-C 时间；
- End-to-End 时间；
- 未测量的通信字节数或轮数。

---

# 14. 必须生成的表格

## Table B1：初始化来源因果消融

```text
Dataset
Distribution
Initialization Source
Anchor CE
Dev Accuracy
Macro-F1
Worst-Class Accuracy
Improved Classes
Refinement Steps
Refinement Time
Backend
```

## Table B2：关键增益分解

```text
Dataset
Distribution
C1 Random+Anchor
C3 Local+Anchor
C4 Average
C5 Average+Anchor
C5-C1
C5-C3
C5-C4
```

## Table B3：步数效率

```text
Dataset
Distribution
Method
Steps@80
Steps@90
Steps@95
Seconds@80
Seconds@90
Seconds@95
```

注明：

```text
Stage-B internal target only
not End-to-End T@Target
```

## Table B4：Plaintext 与 TFE 一致性

```text
Dataset
Distribution
Method
Plaintext Accuracy
Fixed-Point Accuracy
TFE Accuracy
Absolute Gap
```

---

# 15. 必须生成的图

## Figure B1：初始化来源对比

X 轴：

```text
Random
Random+Anchor
BestLocal+Anchor
SecureAverage
SecureAverage+Anchor
```

Y 轴：

```text
Stage-B development accuracy
```

## Figure B2：Anchor CE 收敛曲线

比较：

```text
Random+Anchor
BestLocal+Anchor
SecureAverage+Anchor
```

X 轴：

```text
Anchor refinement steps
```

Y 轴：

```text
development accuracy
```

## Figure B3：类别级修正效果

展示 C4 与 C5 的 per-class accuracy 差异。

## Figure B4：Stage-B 时间分解

展示：

```text
local training
secure average
anchor CE
total
```

---

# 16. 强制输出文件

生成：

```text
new_stage_b_anchor_postagg_reuse_audit.json
new_stage_b_anchor_postagg_split_audit.md
new_stage_b_anchor_postagg_smoke_registry.csv
new_stage_b_anchor_postagg_smoke_summary.json
new_stage_b_anchor_postagg_init_source_ablation.csv
new_stage_b_anchor_postagg_step_efficiency.csv
new_stage_b_anchor_postagg_config_freeze.json
new_stage_b_anchor_postagg_tfe_results.csv
new_stage_b_anchor_postagg_runtime.csv
new_stage_b_anchor_postagg_main_table.csv
new_stage_b_anchor_postagg_claim_audit.md
new_stage_b_anchor_postagg_final_report.md
```

最终报告必须回答：

1. Anchor CE 是否在所有分布下有效？
2. Secure Average 是否对 Anchor CE 有独立贡献？
3. C5 是否优于 Random+Anchor？
4. C5 是否优于 Local+Anchor？
5. 优势体现为最终精度、类别稳健性，还是更少修正步骤？
6. 结果是否跨 IID/Mild/Severe 方向一致？
7. TFE 是否复现 plaintext 结论？
8. Anchor CE 增加多少 Stage-B 成本？
9. Secure Average 的额外成本是否值得？
10. 哪些论文表述安全？
11. 哪些表述不被支持？
12. 模块应作为正贡献、Non-IID 扩展，还是负结果？

---

# 17. Agent 执行循环

每轮必须：

```text
1. 检查 COMPLETE 工件；
2. 复用语义与哈希完全一致的结果；
3. 运行当前最低成本的单因素实验；
4. 同时比较 C5 vs C1、C5 vs C3、C5 vs C4；
5. 判断瓶颈属于：
   初始化无独立贡献、
   Anchor 预算不匹配、
   学习率过大、
   正则过强、
   固定点误差；
6. 每次只调整一个因素；
7. 更新 registry 和 diagnostic report；
8. 通过 smoke gate 后立即冻结；
9. 冻结后只运行 TFE；
10. 完成正式 Stage-B 后停止。
```

每次进度汇报必须写明：

- dataset；
- distribution；
- backend；
- 当前初始化来源；
- Anchor CE 配置；
- C1/C3/C4/C5 结果；
- C5−C1；
- C5−C3；
- C5−C4；
- refinement steps/time；
- 是否读取 test/holdout；
- 下一项唯一操作。

---

# 18. 正面与负面停止条件

## 正面停止

满足以下条件后停止：

- Smoke 证明 Secure Average 对 Anchor CE 有独立价值；
- 配置完成冻结；
- TFE/ABY3 复现主结论；
- 表格、图、审计和报告齐全。

## 负面停止

满足任一情况后停止：

- C5 与 C1 在所有注册预算下等价；
- C5 不优于 C3；
- Secure Average 不减少 Anchor CE 步数或时间；
- TFE 无法保留 plaintext 优势；
- 所有注册单因素诊断均已完成。

不得为了获得正结果无限扩展搜索空间。

---

# 19. 论文安全表述

若成功，可写：

> We treat secure parameter averaging as an intermediate cross-party initializer rather than the final Stage-B output. A compact, class-balanced MTT anchor set subsequently performs supervised post-aggregation refinement. Controlled initialization-source ablations show that the secure average provides a stronger or faster-to-refine starting point than random and single-client initializations.

中文：

> 本文将安全参数平均视为跨机构知识整合的中间初始化器，而非最终输出；随后利用类别均衡的 MTT 锚点对聚合模型进行监督式修正。初始化来源消融表明，安全平均相较随机初始化或单客户端初始化能够提供更有效或更易修正的全局起点。

必须保留的限制：

- 当前仅 seed42；
- 当前指标为 development Stage-B initializer accuracy；
- 不代表最终模型或 test accuracy；
- 未运行 Stage C；
- 不证明 End-to-End 收敛加速；
- Stage B 仍为数据依赖且非 record-level DP；
- Secure Average 不是本文首创；
- Official MTT 的生成/可用性属于明确实验假设。

若 C5 与 C1 等价，则写：

> The gain is primarily attributable to supervised learning on the labeled MTT anchors; secure averaging does not provide a measurable independent benefit under the evaluated protocol.

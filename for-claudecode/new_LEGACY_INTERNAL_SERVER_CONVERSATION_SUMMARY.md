# 旧内网服务器对话主线摘要

## 1. 摘要来源与隐私边界

本摘要依据：

1. `migration/CODEX_CONTEXT_HANDOFF.md`
2. `progressLog/project_progress.md`
3. 已迁移的代码、报告和Git历史

现有安全材料足以恢复研究主线，因此本轮没有读取私有原始JSONL。

原始JSONL可能包含历史凭据和敏感工具输出，禁止提交GitHub。若未来必须核查
某个具体历史问题，只能用`rg`按最小关键词定位，并且不能输出敏感字段。

## 2. 旧服务器研究路线的演变

### 2.1 最初目标：证明蒸馏数据优于随机子集

早期工作集中在Module A：

```text
Full Data
Random Subset
Distilled Data
```

目标是验证Dataset Distillation/MTT构造的紧凑训练集，是否能在相同样本数量
下优于随机抽样。

早期实验先在CIFAR-10验证，随后尝试MNIST、Tiny ImageNet和CIFAR-100。

### 2.2 自行运行MTT的探索

旧对话中曾：

1. 为MNIST生成expert trajectories；
2. 运行MNIST MTT distillation；
3. 删除低价值MNIST产物，切换Tiny ImageNet；
4. 尝试Tiny的ipc=1、ipc=10、ipc=50；
5. 多次因服务器负载、磁盘和长时间成本调整参数；
6. 为CIFAR-100准备MTT official-style流程。

这些探索说明完整自行蒸馏成本高、稳定性和资源压力大。

### 2.3 转向Official Public MTT

用户后来提供官方作者发布的高质量MTT蒸馏数据ZIP。项目逐一验证官方数据，
并用同款ConvNet在CIFAR-10、CIFAR-100、Tiny ImageNet上重新完成Module A。

研究主线最终从“自己重新蒸馏”转向：

```text
Official Public MTT distilled images
```

这是当前论文主线的来源。早期DP-MTT或自行蒸馏路线不再是当前必需项。

## 3. 从Module A到Stage B

完成Official-MTT Module A后，旧对话开始研究蒸馏数据如何帮助后续训练。

### 3.1 Figure 5风格初始化实验

先比较：

```text
Random Init
Average Init
Best Init
```

后来发现仅固定特征/线性头或不一致的预处理会导致绝对精度过低，于是排查：

- 训练步数；
- ConvNet结构；
- feature normalization；
- Figure 4与Figure 5的数据和模型一致性；
- classifier初始化映射。

### 3.2 初始化策略扩展

随后加入：

```text
Reference-Weighted Init
Distilled Prototype Init
Candidate Initialization Pool
Reference-guided Selection
```

叙事从“选一个平均模型”演变为：

```text
多个客户端候选
-> reference-guided secure selection
-> chosen initializer
-> Stage C
```

### 3.3 Non-IID KD

旧服务器完成Stage B Non-IID KD消融，覆盖：

```text
CIFAR-10
CIFAR-100
Tiny ImageNet
IID / Mild / Severe
```

目标是验证KD alignment是否缓解Non-IID客户端知识冲突。最终正式Public-MTT
Stage B使用TFE/ABY3、5 clients、test-free reference/selection和3 seeds。

## 4. Stage C的形成

旧对话逐步建立Module C DP optimization，比较过：

```text
Plain
Fixed DP
Adaptive DP
Adaptive DP + BinomialMPC
```

后来经过安全/隐私重构，正式Public-MTT Stage C聚焦：

```text
TFE/ABY3 secure DPSGD
Fixed clipping
Adaptive clipping
CSPRNG noise shares
privacy accountant
checkpoint/resume
```

重要经验：

1. Adaptive方向和clipped indicator实现曾需要审计；
2. count mechanism也消耗隐私预算，必须计入accountant；
3. Fixed模式不应无意义调用/reveal clipped-count；
4. selected initializer是data-dependent non-DP；
5. Stage-C epsilon不能自动升级为完整pipeline DP；
6. test必须在调参和候选选择之外。

## 5. End-to-End叙事的演变

早期End-to-End曾包含很多方法：

```text
Random Subset
Distilled Only
A+B Init
A+B+Fixed-DP
A+B+Adaptive-DP
A+B+Adaptive-DP+BinomialMPC
```

随后为了论文Figure 2收敛为较少曲线，例如：

```text
RandomSub-TFE
Distilled-TFE
Ours-TFE
Ours-TFE*
```

这段历史说明主图应尽量简单，复杂消融放入其他图表或补充材料。

## 6. 安全实现与算法实现的边界

旧项目中存在两类实验：

### PyTorch全ConvNet算法实验

部分旧结果标记：

```text
algorithmic_pytorch_convnet
```

它们可用于算法准确率和初始化概念验证，但不是TFE/ABY3意义下的直接安全CNN。
不能把这些结果错误标记为MPC secure ConvNet。

### TFE/ABY3安全特征实验

当前安全主线是：

```text
冻结的2048-D特征
-> 安全线性模型
-> TFE/ABY3 Stage B/Stage C
```

这种设计避免在MPC中直接训练深层CNN，显著降低安全计算压力。

## 7. 2026-07-24到07-25的重构

旧服务器后期进行大规模可迁移/安全重构：

1. 统一共享MTT ConvNet encoder和2048-D表示；
2. 区分PyTorch算法实验与TFE安全实验；
3. 增加真实TFE Stage C；
4. 修复隐私会计；
5. 加强test-set边界；
6. 增加checkpoint、heartbeat、status和恢复；
7. 所有新文件使用`new_`前缀；
8. 新输出隔离，禁止覆盖旧结果；
9. smoke test不能冒充formal。

曾建立DP-MTT全正式路线，但其trajectory、GPU时间和中断恢复成本高，后来不再
作为Public-MTT主线的前置条件。

## 8. Official Public MTT主线冻结

2026-07-26，任务切换到：

```text
plan/new_public_mtt_mainline_codex_prompt.md
```

核心决定：

1. Official Public MTT成为主线；
2. DP-MTT明确排除；
3. Stage A复用已完成结果；
4. Stage B用真实TFE/ABY3；
5. Stage C和End-to-End从selected initializer继续；
6. 三数据集、3 seeds、正式报告；
7. 不允许test调参；
8. 不允许plaintext fallback；
9. 不允许用smoke填正式图。

## 9. 旧服务器完成的Stage B正式工作

### 9.1 Test-free tuning

Stage B调参使用独立reference与selection，不加载test。Round 1暴露：

- CIFAR-10 selection过小；
- reference-weight entropy接近均匀；
- Tiny单核计算极慢。

Round 2扩大每类reference/selection并增加reference-supervised calibration。

### 9.2 数值稳定性

低temperature在16-bit fractional定点下可能指数下溢，因此secure softmax先
减去秘密共享的client mean loss，再做指数。这不改变softmax，也不泄露loss。

真实TFE低温correctness通过。

### 9.3 正式Stage B

完成：

```text
3 datasets × 3 seeds × IID/Mild/Severe = 27/27
```

候选包括：

```text
NoKD
KD-Avg
KD-ReferenceWeighted
Selected
```

test只在候选和配置全部冻结后用于最终报告。

### 9.4 结果解释

Severe Non-IID下，KD-ReferenceWeighted相对NoKD在三个数据集均提高初始模型
精度。结果支持“KD alignment缓解强异构知识冲突”的论文论点。

## 10. 迁移到AutoDL的准备

用户要求停止旧服务器训练，把项目迁移到AutoDL。主要工作：

1. 停止并禁用训练service；
2. 固化现代/legacy双环境；
3. 使用`NEW_*`路径变量；
4. 移除可执行代码和TSV中的旧绝对路径；
5. 保存MTT Git bundle、patch和未跟踪源码；
6. 收集27个portable selected initializer；
7. 打包高成本encoder/scaler/feature caches；
8. 记录SHA256；
9. 不把raw data、official tensors、secrets或DP-MTT大资产上传Git。

迁移包：

```text
autodl_new_secure_resume_20260726_150457.tar.zst
size: 8,771,438,515 bytes
SHA256: 2313a96c1352c2cb66b6262bb59f61d559fb2cd6112f87131e7ab76a53b1bae9
```

私有对话另行加密/私下迁移，不进入普通项目包或Git。

## 11. 旧对话中的资源策略已被部分覆盖

旧服务器为了防止死机，长期要求：

```text
单GPU
CPU单核/单线程
num_workers=0
串行数据集
```

这个策略在当时资源不稳定的机器上是合理的。

迁移到AutoDL后，用户明确取消“线程固定为1”，要求发挥最大性能。当前有效
规则是：

```text
不人为限制为单线程
使用可用CPU/GPU
避免无界线程爆炸
避免重复任务
长任务仍必须tmux/checkpoint/resume
```

不要因为读到旧交接第4节，就重新把正式任务绑回单核。

## 12. 必须继承的实验纪律

旧对话形成的以下规则仍然有效：

1. 不重跑已完成且完整兼容的正式实验；
2. 正式实验至少3 seeds；
3. test不参与调参、候选选择、target选择；
4. 负结果不隐藏；
5. smoke/pilot不进入正式论文图；
6. 所有privacy claim必须按真实保护范围；
7. MPC confidentiality不等于DP；
8. 所有epsilon必须由真实N、batch、steps和机制组成校准；
9. 长任务必须可恢复；
10. checkpoint/status/history/JSON/CSV/PNG/PDF必须齐全；
11. 新文件使用`new_`前缀；
12. 不覆盖旧正式资产；
13. 每次修改维护进度并push GitHub；
14. 不输出令牌、密钥或敏感日志。

## 13. 从旧对话到当前任务的最终映射

旧服务器主线提供了当前任务所需基础：

```text
Official MTT数据
共享2048-D encoder/features
Stage B Secure Consensus initializer
Stage B Non-IID证据
Stage C Fixed/Adaptive实现
TFE/ABY3 correctness
隐私会计与test冻结纪律
迁移资产与恢复文档
```

当前增量任务不推翻这些成果，只新增：

```text
Full Data Secure-Average
Full Data Adaptive Stage C
Full Data vs 已有Ours Adaptive两曲线主图
外部方法表
```

旧对话中“Adaptive standalone消融不稳定”与“selected-initializer End-to-End
Adaptive效果较好”可以同时成立，因为initializer和实验问题不同。论文必须
明确区分这两种设置，不能混为一个结论。

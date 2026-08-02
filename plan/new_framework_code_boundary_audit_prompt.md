# 服务器端 Agent 代码核验任务：论文整体框架图实现边界审计

## 0. 任务目的

我要根据真实代码重新绘制论文整体框架图。当前文字框图中有一些步骤是“方法设计层面的逻辑”，但我不确定它们在当前实验代码中究竟是客户端本地执行、预先离线构造、明文执行后再秘密共享、真正在 TFE/ABY3 秘密共享域内执行，还是仅在论文方法设定中存在、当前实验没有实现。

请你逐项检查真实代码、配置、日志和已有实验输出，**不要根据论文故事或我的描述推断**。

## 1. 核验原则

对每一项标记以下状态之一：

```text
[LOCAL]        客户端本地明文执行
[PRECOMPUTED]  预先计算/直接读取现成资产
[SECRET-SHARE] 明文结果随后秘密共享
[MPC]          真正在 TFE/ABY3 秘密共享域中计算
[HYBRID]       一部分明文、一部分 MPC
[SIMULATED]    仅模拟，没有真实 TFE/ABY3 执行
[NOT-IMPL]     当前代码未实现
[UNCLEAR]      无法从代码确认
```

每个结论尽量给出：
- 文件路径；
- 关键函数 / 类 / 脚本名；
- 关键配置项；
- 代码调用链或大致行号；
- 当前正式主线 / 历史实验 / smoke / diagnostic 的归属。

---

# 2. Stage A：本地紧凑数据构建

当前论文方法想画成：

```text
Party Pi private data Di
→ Local trajectory-matching dataset distillation
→ compact distilled dataset D̃i
→ shared/frozen encoder
→ compact feature Zi
```

### A1. 当前实验是否真的执行“每个客户端用自己的原始数据做 MTT 蒸馏”？

请确认：
- 是否存在 client-specific MTT/distillation 训练代码；
- 输入是不是每个 client 的原始 CIFAR 数据子集；
- 是否每个 client 独立优化 synthetic images；
- 是否真的产生 D̃1...D̃K；
- 还是当前代码直接读取预计算/Official MTT synthetic samples，再按客户端划分。

请明确给出：

```text
当前真实实现：
方法层面理想设计：
两者是否一致：
```

### A2. 预计算 MTT 资产在哪里进入流程？

确认：
- 读取的具体文件；
- 是 synthetic image 还是已经提取好的 feature；
- 是否有 label；
- 是否程序开始前已固定；
- 是否所有客户端从同一个 MTT pool 切分；
- 还是每客户端有独立文件。

### A3. Shared/Frozen Feature Encoder 在哪里运行？

确认：
- encoder 是什么；
- 是否 shared/frozen；
- feature extraction 是客户端本地、中央明文、TFE/ABY3，还是直接读缓存；
- 当前正式实验是否真的跑 encoder forward，还是加载预计算 2048-D features。

请明确：

```text
框架图应画：
D̃i → Encoder → Zi
还是
Precomputed Zi → 后续流程

Encoder 是否应该画在 MPC 大框之外？
```

### A4. Zi_local 与 Anchor Ai 如何划分？

确认：
- 是否从同一 compact pool 拆为 local training / anchor；
- 当前是否为 40 samples/class local + 8 samples/class anchor；
- 是否所有实验一致；
- client-local 与 anchor 是否互斥；
- 是否与 development split 互斥；
- split manifest 路径。

---

# 3. Stage B：本地头训练与秘密共享边界

### B1. 各客户端 lightweight head 到底在哪里训练？

确认：
- 明文 PyTorch/TensorFlow 本地训练，还是先 secret-share feature/label 再 TFE 训练；
- optimizer、epoch、loss；
- 是否不同客户端分别训练；
- 是否共享相同 initialization；
- 是否生成普通 tensor 参数后再 secret-share。

最终明确：

```text
Local Head Training 应画在：
[ ] MPC 大框之外
[ ] MPC 大框之内
[ ] 部分在外、部分在内
```

### B2. θi 什么时候秘密共享？

确认真实调用链是：

```text
local θi → secret share → TFE/ABY3
```

还是：

```text
local data/features → secret share → θi 从一开始就是 shared value
```

---

# 4. Stage B：Secure Average

### B3. Secure Average 平均的到底是什么？

确认：
- classifier weight + bias；
- 完整模型参数；
- logits；
- gradients；
- 其他 tensor。

公式是否准确：

```text
θavg = 1/K Σ θi
```

### B4. Secure Average 是否真实 TFE/ABY3？

确认：
- 当前正式 C4/C5 是否调用 TFE；
- 是否 3-party ABY3；
- computing parties 数量；
- 是否存在 plaintext fallback；
- fixed-point precision / fractional bits。

### B5. secure_average_only 4.884× 到底优化了什么？

确认：
- 删除哪些图：teacher / KD / reference weighting / candidate selection / secure evaluation 等；
- 保留哪些 secure ops；
- 4.884× 是 aggregation subgraph、整个 Stage B，还是某 benchmark path。

---

# 5. Stage B：Anchor 来源与构造位置

需要判断真实代码是：

```text
A1...AK → secure class balancing → global anchor A
```

还是：

```text
preconstructed class-balanced anchor A
→ secret sharing
→ secure CE refinement
```

### B6. Anchor 是否真的由每客户端分别贡献？

确认：
- 是否存在 client-specific Ai；
- 是否每客户端上传自己的 anchor；
- 还是 global anchor split 在实验开始前已确定；
- client ID 在 anchor 中是否还有语义。

### B7. “Class balancing”在哪里完成？

确认：
- 是否有真正 secure class balancing / secure sampling；
- 还是 Python 明文 split 时就保证每类 8 个；
- labels 是否进入 TFE 前已知；
- 运行时是否根据 label 重新平衡。

最终回答：

```text
框架图能否写：
[A1]...[AK] → Secure Class Balancing → [A]

如果不能，应改成什么？
```

### B8. Anchor label 是否秘密共享？

确认：
- Anchor feature 是否 secret-shared；
- Anchor labels 是否 secret-shared；
- one-hot 还是 integer class ID；
- CE loss 如何使用 label。

---

# 6. Stage B：Anchor CE 是否全程 secure

### B9. Anchor CE refinement 具体在哪里运行？

请填：

```text
Anchor CE:
forward      = [LOCAL / MPC / HYBRID]
loss         = [...]
gradient     = [...]
update       = [...]
output θB    = [...]
```

并确认：
- 是否有 reveal θavg 后明文 fine-tune；
- 是否有 plaintext fallback。

### B10. 最新 C5 TFE 的真实执行路径

根据：

```text
CIFAR-10: C1/C4/C5 = 55.48 / 18.84 / 57.62
CIFAR-100: C1/C4/C5 = 10.02 / 8.02 / 17.50
```

确认 C5 是否真的是：

```text
local heads
→ TFE Secure Average
→ TFE Anchor CE
→ θB
```

若某一步明文，必须指出。

---

# 7. Stage B 对照实现

### B11. C1 Random + Anchor 的 random initialization

确认：
- random seed；
- initialization distribution；
- 与 C5 architecture 是否完全一致；
- C1 Anchor CE 是否走同一 TFE graph。

### B12. Best Local / client-0 的选择规则

确认：
- 固定 client-0；
- development 上选 best client；
- 正式 baseline 还是 oracle diagnostic。

---

# 8. Stage B 输出 θB 的状态

### B13. θB 输出后是否仍是 secret-shared model？

确认：
- 是否一直保持 shared tensor；
- 是否 reveal/save 为 plaintext checkpoint；
- Stage C 是直接接 shared θB，还是 reload 普通 checkpoint 再 secret-share。

这决定 Stage B→Stage C 在框图中能否无缝画在同一 MPC 域。

---

# 9. Stage C 输入

### C1. Stage C 使用哪些训练样本？

确认：
- 全部 compact features；
- 仅 local compact features；
- anchor 是否进入 Stage C；
- 各客户端 feature 是否逻辑拼接；
- 是否存在变量 Zsyn；
- “logically combined compact features” 是代码事实还是概念描述。

### C2. Stage C compact features 何时 secret-share？

确认：
- Stage A 后一次 share 后复用；
- Stage B/C 分别重新 share；
- 是否存在 share reuse。

---

# 10. Stage C：Poisson Sampling

### C3. Poisson sampling 在哪里执行？

确认：
- 明文 mask；
- OS-CSPRNG；
- TFE graph 内；
- mask 是否公开；
- 是否真正逐 record Poisson；
- sample rate q 在哪里配置。

框图应该写 `Poisson Sampling` 还是 `CSPRNG Poisson Sampling`？

---

# 11. Stage C：Per-example gradients

### C4. per-example gradients 是否真实在 TFE/ABY3 计算？

确认：
- gradient 对象是不是 linear head；
- encoder 是否 frozen；
- 是否逐样本 secure gradient；
- 是否 vectorized；
- 是否存在 plaintext gradient computation。

---

# 12. Stage C：Adaptive clipping

### C5. gradient norm statistics 如何计算？

确认：
- L2 norm 是否 TFE 内；
- norm 是否 reveal；
- mean / median / quantile / EMA 哪种统计；
- 哪些 scalar public；
- 哪些 secret-shared。

### C6. clipping bound Ct 如何更新？

请给真实公式/伪代码，并确认：
- C0；
- 更新频率；
- EMA/momentum；
- upper/lower bound；
- 是否依赖当前 batch norm；
- Ct 是否公开；
- Ct 在 TFE 内还是 Python host 更新。

框图中的：

```text
Secret-Shared Gradient-Norm Statistics
→ Adaptive Clipping-Bound Update Ct
```

是否真实？

---

# 13. Stage C：Gaussian noise

### C7. 高斯噪声真实生成方式

确认：
- 是否每 computing party 生成 noise share；
- 还是一个 OS-CSPRNG 生成完整 noise；
- noise 是明文还是 shared；
- Gaussian sampler 在哪里；
- noise scale 如何计算；
- noise 在 secure aggregation 前还是后注入。

最终给出最准确的一行框图文字，不要默认写 `Distributed CSPRNG Gaussian-Noise Shares`。

---

# 14. Stage C：Secure Parameter Update

### C8. 参数更新是否全程 MPC？

确认：
- clipped/noised gradient 是否 shared；
- learning rate 是否 public；
- θ update 是否 TFE；
- 每 step 是否 reveal 参数；
- 最终 θT 何时 reveal/save。

---

# 15. 角色关系：Data Owners 与 MPC Computing Parties

当前框图是：

```text
Data owners: P1 ... PK
MPC computing parties: S1, S2, S3
```

请确认：
- P1..PK 与 S1..S3 是否不同角色；
- 是否只是实验中的逻辑角色；
- TFE/ABY3 是否固定 3 个 computing servers；
- data owners 是否直接持有 share；
- client → 3 servers share，还是单机多进程模拟。

---

# 16. Privacy Boundary

### P1. 以下对象哪些真实在 secret-sharing 域？

逐项标：

```text
compact features
anchor features
anchor labels
local model parameters
global model parameters
per-example gradients
gradient norms
clipping bound
Gaussian noise
intermediate logits
loss
```

状态：

```text
[MPC] / [PUBLIC] / [REVEALED] / [LOCAL]
```

### P2. Stage A 是否有 DP？

确认：
- 本地 MTT 是否加 DP；
- 若当前只是预计算 MTT，是否无正式 DP guarantee；
- 不要把“raw data不离开本地”等同 DP。

### P3. Stage B 是否有 record-level DP？

确认：
- Secure Average 只有 MPC confidentiality 还是另有 DP；
- Anchor CE 是否有 clipping/noise；
- 当前是否应明确写 Stage B non-DP。

### P4. Stage C accountant 覆盖什么？

确认：
- accountant 从哪个 step 开始；
- ε/δ 对应哪些 records；
- 是否覆盖 Stage B；
- 是否覆盖 anchor；
- 是否只覆盖 Stage C compact training records；
- 当前 ε≤3、δ=1e-5 是否对应正式主线。

---

# 17. 框架图中性能/隐私陈述的证据核验

逐条回答 YES / NO，并给证据路径。

### Q1
```text
Stage A reduces protected training workload.
```

### Q2
```text
Stage B produces a stronger global initializer.
```

是否由 C1/C4/C5 正式 TFE 支持？

### Q3
```text
Stage B reduces private iterations required to reach target accuracy.
```

有没有将**新 C5 initializer**接入 Stage C 后直接测过？没有就写 NO。

### Q4
```text
Stage C adaptive clipping improves utility over fixed clipping.
```

若当前只是 +0.04pp parity，请明确不能画成 improves accuracy。

### Q5
```text
The full framework has end-to-end record-level DP.
```

按 accountant/代码确认。

---

# 18. 方法设计 vs 当前实验真实实现

请输出：

| 组件 | 论文方法设计 | 当前实验真实实现 | 是否一致 | 框图建议 |
|---|---|---|---|---|
| Client-local distillation | 每客户端本地 MTT | ? | ? | ? |
| Feature encoder | shared/frozen | ? | ? | ? |
| Local head training | 本地 | ? | ? | ? |
| Secure Average | TFE/ABY3 | ? | ? | ? |
| Anchor construction | client contributions / balanced | ? | ? | ? |
| Anchor CE | secure | ? | ? | ? |
| θB → Stage C | shared handoff | ? | ? | ? |
| Poisson sampling | CSPRNG | ? | ? | ? |
| Adaptive Ct | secure statistics | ? | ? | ? |
| Gaussian noise | secure/CSPRNG | ? | ? | ? |
| Final update | secure | ? | ? | ? |

---

# 19. 最终必须输出“纯代码事实执行链”

请给一份不带论文包装、不带推测的真实链，例如：

```text
[PRECOMPUTED] MTT samples
→ [LOCAL] frozen encoder
→ [PRECOMPUTED] cached 2048-D features
→ [LOCAL] client-local head training
→ [SECRET-SHARE] W,b
→ [MPC] TFE/ABY3 Secure Average
→ [SECRET-SHARE] pre-balanced anchor features/labels
→ [MPC] secure CE refinement
→ [REVEALED] save θB
→ ...
```

注意：以上只是格式示例，不代表真实实现。

必须：
1. 每一步标状态；
2. 明确每次 reveal；
3. 明确每次 secret-sharing；
4. 明确缓存输入；
5. 明确哪些“client-local”只是模拟；
6. 不省略 MPC/明文边界。

---

# 20. 输出文件

请生成：

```text
new_framework_code_boundary_audit.md
new_framework_component_status.csv
new_framework_actual_execution_chain.md
new_framework_claim_support_matrix.md
```

其中：

- `new_framework_code_boundary_audit.md`：逐项回答 A1–P4。
- `new_framework_component_status.csv`：组件、状态、code path、backend、input、output、reveal/share boundary。
- `new_framework_actual_execution_chain.md`：纯代码事实端到端执行链。
- `new_framework_claim_support_matrix.md`：framework statement / supported? / exact evidence / safe wording / forbidden overclaim。

## 最重要要求

**不要帮我美化框架，先把真实代码事实查清。**

尤其不要默认以下已经实现：

- 每客户端真实独立 MTT 蒸馏；
- local head 一定在 MPC 里训练；
- Anchor 一定由各客户端贡献后安全平衡；
- Anchor CE 一定全程 secure；
- 新 Stage B 已经证明减少 Stage C iterations；
- Gaussian noise 一定是 distributed shares；
- Stage A/B 具有 DP；
- Stage B 与 Stage C 一定复用同一 secret shares。

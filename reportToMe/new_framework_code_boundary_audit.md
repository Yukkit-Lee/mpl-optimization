# 论文整体框架图：代码实现边界审计

日期：2026-08-02（UTC）
分支：`unified-experiments`
方式：只读代码、配置和既有输出；未运行测试、模型、TFE或实验。

## 0. 总结

仓库中不存在一条把“每客户端本地 MTT → 最新 C5 Anchor CE → Adaptive
Stage C”全部串起来的正式链。必须区分：

- True Ours 正式主线：预计算 Official-MTT 2048-D features → TFE内训练五个头
  → Secure Average → reveal/save NPZ → Stage C reload/re-share → Adaptive DP-SGD。
- 最新正Stage-B C5：预计算features → TFE内训练五个头 → Secure Average →
  TFE Anchor CE → reveal/save；明确没有运行Stage C/DP/End-to-End。

最新C5不能被画成已经接入当前Stage C。

## 1. Stage A

### A1. 每客户端独立MTT蒸馏？

**`[NOT-IMPL]`（当前正式主线）。**

- `new_stage_a_shared_encoder.py:61-81`直接读取Official MTT
  `images_best*.pt/labels_best*.pt`。
- `new_prepare_true_ours_assets.py:132-148`读取已生成的test-free feature cache。
- 没有以每个client原始子集为输入、分别优化synthetic images并产生D̃1…D̃K的正式调用链。
- `new_stage_a_dp_distill.py`是已排除的历史全局DP-MTT路径，也不是当前client-specific实现。

```text
当前真实实现：固定Official MTT公共上游资产→统一编码/缓存→再划分逻辑客户端。
理想设计：每客户端以私有Di独立执行trajectory matching产生D̃i。
是否一致：否。
```

### A2. 预计算资产入口

**`[PRECOMPUTED]`。**

- 原始根目录：`new_data/mtt_data/georgecazenavette/<dataset>_50/`，包含synthetic
  images和整数labels；没有每客户端独立文件。
- ipc=50 pool先保留reference/selection，下游`x_encoder_train`为48/class，即
  C10 480、C100 4,800、Tiny 9,600。
- True Ours输入：
  `new_cache/new_public_mtt_mainline/new_test_free_features/<dataset>/new_reference_feature_cache_<dataset>_seed<seed>.pt`。
- Stage-B/C包装：`new_cache/new_true_ours/official_mtt/{stage_b,stage_c}/...npz`
  （`new_prepare_true_ours_assets.py:245-295`）。

### A3. Shared/Frozen Encoder

**资产生成时`[LOCAL]`；正式Stage-B/C消费时`[PRECOMPUTED]`。**

- Encoder为`NewMTTConvNet`，`encode()`输出flatten后的2048-D表示
  （`new_secure_feature_models.py:48-86,104-115`）。
- `train_shared_encoder`在单个PyTorch进程中以SGD+CE训练；不在TFE内，也不是每客户端
  encoder（`new_stage_a_shared_encoder.py:170-283`）。
- `extract_features`执行encoder forward并缓存2048-D features（`:286-295`）。
- encoder在同一dataset/seed内共享并于下游冻结；不同seed有各自cache/SHA。
- 当前正式Stage-B/C直接加载feature cache，不再执行encoder forward。

框架图可画offline shared encoder，但应放在MPC框外；纯执行链应从
`[PRECOMPUTED] cached 2048-D features`开始。

### A4. local/anchor划分

**最新Post-Aggregation中为host端`[PRECOMPUTED]`划分。**

- `mtt_anchor_split`按已知标签逐类shuffle，前8条为全局anchor，余40条为local总池
  （`new_stage_b_kd/new_stage_b_kd_core.py:211-228`）。
- 40/class是五客户端共享的总池，不是每客户端40/class；随后由
  `client_partitions`分成五个逻辑客户端（`:231-278`）。
- local∩anchor=0且并集覆盖48/class；split freeze位于
  `new_log/new_stage_b_anchor_postagg/new_stage_b_anchor_postagg_split_freeze.json`，
  并登记development交集为0。
- 40/8并非所有历史实验一致。True Ours正式`secure_average_only`无Anchor CE；
  Stage C读取完整48/class cache；Tiny没有最新C5正式TFE。

## 2. Stage B：本地头与sharing

### B1. lightweight head在哪里训练？

**正式路径`[MPC]`；另有plaintext筛选/参考代码。**

- 最新C5 worker把五份client features、host one-hot labels、W、b定义为private
  variables（`new_anchor_postagg_tfe_worker.py:74-104`），再在ABY3内执行full-batch
  softmax CE+SGD（`:105-144`）。
- C10为5 epochs/LR.1，C100为40 epochs/LR.2，weight decay=1e-4；五个头共享
  相同initial W/b。
- True Ours主线也在MPC内训练五个线性头（`new_stage_b_secure_init.py:480-565`），
  但用logits-MSE与另一套epochs/sync配置，不是最新C5的CE实现。
- NumPy plaintext runner只作筛选、因果消融和TFE正确性参考。

`Local Head Training`若对应当前正式代码，应画在MPC大框内。这与“客户端明文训练后
上传θi”的设计故事不一致。

### B2. θi何时share？

**真实链是features/labels先share，θi从训练开始就是`[MPC]` state。**

不存在正式`plaintext θi→share`。True Ours动态batch经
`define_private_input("server0",...)`进入；最新C5直接从冻结NPZ构造private variables。

## 3. Secure Average

### B3. 平均对象

**`[MPC]`平均线性classifier W(2048×C)+b(C)，不是encoder、logits或gradients。**

最新C5为均匀1/K平均（worker `:146-156`），公式`Wavg=ΣWi/K, bavg=Σbi/K`准确。
True Ours旧路径支持公开uniform/sample-count系数，因此不总是严格1/K
（`new_stage_b_secure_init.py:567-592`）。

### B4. 是否真实TFE/ABY3？

**`[MPC]`，但为单机三逻辑server模拟。**

- `make_protocol`使用`tfe.protocol.ABY3()`（`new_stage_b_secure_init.py:407-420`）。
- share更新明确遍历3 servers×2 replicas（`:451-477`）。
- 最新C4/C5报告为TFE/ABY3、fractional bits=16、chunk=64、无plaintext fallback。
- 没有三物理机或LAN/WAN部署证据。

### B5. `secure_average_only` 4.884×

**这是True Ours旧CIFAR-10 seed42完整Stage-B窄图路径，不是average算子本身，也不是
最新C5。**

- 保留secure local training、周期FedAvg、NoKD average snapshot、最终release和可选
  checkpoint development-logit评价。
- 不构建teacher/KD/reference labels/selection/reference-weighted candidate/MSE/private
  selection子图（`new_stage_b_secure_init.py:506-528,604-644`）。
- 审计为458.55s→93.89s、accuracy均68.76%、speedup=4.884×，见
  `new_log/new_true_ours/new_stage_b_average_only_equivalence_audit.json:44-54`。

## 4. Anchor

### B6. 每客户端分别贡献Ai？

**client-specific Ai为`[NOT-IMPL]`；实际是`[PRECOMPUTED]`全局anchor。**

host在任何client训练前从同一Official-MTT pool选8/class。只有全局
`x_anchor/y_anchor`，client ID无anchor贡献语义。

### B7. Class balancing

**`[PRECOMPUTED]` host Python明文完成，不是secure balancing。**

labels在进入TFE前已知，运行时不重新采样/平衡。不能画
`A1...AK→Secure Class Balancing→A`；应画：

```text
[PRECOMPUTED] class-balanced global Official-MTT anchor
→ secret-share features/one-hot labels → Secure Anchor CE
```

### B8. Anchor labels

**`[SECRET-SHARE]`后用于`[MPC]`。** 输入NPZ保存整数class ID；host先转换成
N×C one-hot float，再与features一起`define_private_variable`
（worker `:158-164`）。

## 5. Anchor CE

### B9. 具体边界

```text
forward=[MPC]; error/loss=[MPC]; gradient=[MPC]; update=[MPC];
output θB=[REVEALED] final W,b NPZ。
```

实现见worker `:191-246`。C5从shared average expression直接赋值给private C5 W/b，
没有先reveal θavg再明文fine-tune。正式worker无fallback。输入准备脚本的plaintext
C1/C5只用于P1/P2正确性参考，不代替TFE训练。

### B10. 最新C5执行路径

题目所列数字确实来自：

```text
[PRECOMPUTED] client-partitioned features/labels
→ [SECRET-SHARE]
→ [MPC] five local linear heads
→ [MPC] uniform Secure Average
→ [MPC] 40-step Anchor CE + μ regularization
→ [REVEALED] C1/C4/C5 W,b
→ [LOCAL] development correctness evaluation
```

验证见`new_stage_b_anchor_postagg_tfe_validation.json:7-80`。无Stage C。

## 6. 对照与输出

### B11. C1 Random+Anchor

**初始化`[PRECOMPUTED]`，训练`[MPC]`。** `RandomState(seed+991)`生成
W~N(0,.01)，b=0（`new_stage_b_kd_core.py:310-318`）；冻结seed42。C1/C5架构、
anchor、steps、LR、μ一致并复用同一个secure `anchor_update`。

### B12. Best Local/client-0

正式C3固定client-0，不读development：`select_registered_local_client()`硬返回0
（`new_stage_b_anchor_postagg_core.py:222-224`）。development best-client只能作oracle
diagnostic。正式TFE只跑C1/C4/C5；C3来自plaintext冻结消融。

### B13. θB是否保持shared？

**否，最终`[REVEALED]`。**

- 最新worker最终`.reveal()` C1/C4/C5 W,b并保存普通NPZ（`:363-390`）。
- Stage C从普通NPZ加载initializer，再在新图中`define_private_variable`
  （`new_stage_c_secure_dpsgd.py:47-51,171-190,832-840`）。
- Stage B/C是不同进程、图和Session，没有share复用。
- 最新C5没有交给Stage C；Stage C用旧True Ours Secure-Average-only initializer。

框图必须画`reveal/save→reload→re-share`断点。

## 7. Stage C输入与Poisson

### C1. 训练样本

**`[PRECOMPUTED]`完整Official-MTT encoder-train cache。** True Ours Stage C使用全部
48/class features；不保留40/8运行时语义。没有变量`Zsyn`，也没有运行时把K份client
features拼接的函数；“logically combined compact features”只是概念描述。

### C2. 何时share？

**Stage B/C分别重新`[SECRET-SHARE]`，无复用。** Host每step从plaintext cache选取并
填充batch，再通过server0 provider分享（`new_stage_c_secure_dpsgd.py:469-510`）。

### C3. Poisson sampling

**`[HYBRID]`：host OS-CSPRNG选择，batch/mask再share。**

- `poisson_batch_indices`在Python调用`os.urandom`，逐record Bernoulli(q)
  （`:61-78`），不是TFE内采样。
- q=`expected_batch_size/N`；正式expected=16、graph capacity=64。True Ours q为
  16/480、16/4800、16/9600。
- host知道selected IDs和batch size；代码构造padded batch/one-hot labels/mask，再将
  三者private-share（`:469-510`）。

框图写：`[LOCAL] OS-CSPRNG per-record Poisson selection → [SECRET-SHARE] padded batch+mask`。

## 8. Stage C安全计算

### C4. per-example gradients

**`[MPC]`向量化线性头per-example clipping math。** Encoder已冻结。代码通过
`||error||²||x||²+||error||²`计算每样本W+b梯度norm，先求clip factor、缩放error，
再安全矩阵乘法聚合（`:247-285`）。不显式materialize N×2048×C tensor；正式路径
无plaintext gradient计算。

### C5. norm statistics

**raw norms`[MPC]`；DP-noised fraction/Ct轨迹`[REVEALED]`。** 统计不是
mean/median/quantile/EMA，而是batch中`norm<=Ct`的比例；加count Gaussian noise后
clamp到[0,1]（`:295-311`）。Adaptive step reveal noisy fraction和proposed C；current C
实际每stepreveal，log interval还reveal development logits/loss（`:517-588`）。

### C6. Ct公式

```text
b_t=(1/Bexpected)Σ1[||g_i||<=C_t]mask_i
b~_t=clip(b_t+N(0,σcount²)/Bexpected,0,1)
C_{t+1}=clip(C_t exp(-η_C(b~_t-γ)),Cmin,Cmax)
```

C0/γ/η/bounds/warmup/interval均public；Ct是private variable但轨迹揭示。True Ours冻结：
C10 C0=1,γ=.5,η=.02；C100 C0=2,γ=.1,η=.05；Tiny C0=1,γ=.5,η=.1；
全为warmup=5、每5 steps更新、范围[.05,4]。无EMA/momentum。

### C7. Gaussian noise

**`[HYBRID]`：每逻辑server本地CSPRNG part+MPC求和。**

- `_provider_gaussian`以secure-random custom op+Box-Muller生成Gaussian（`:81-100`）。
- server0/1/2各生成N(0,1/3) part，分别private-input并在ABY3求和为N(0,1)
  （`:102-131`）；seed/counter分别checkpoint（`:195-245`）。
- scale=`noise_multiplier*Ct/Bexpected`，在clipped gradient聚合后加入（`:277-293`）。
- 这是单机三逻辑server，不是已验证的三物理服务器。

准确文字：`[SIMULATED 3-server/HYBRID] per-server checkpointed CSPRNG Gaussian parts
→ [MPC] ABY3 sum → add to aggregated clipped gradient`。

### C8. 参数更新

**更新`[MPC]`，最终模型`[REVEALED]`。** gradient/W/b shared，LR public，
`secure_simultaneous_assign`原子更新；不逐step reveal W/b。每stepreveal Ct、adaptive
stepreveal noisy count/Cnext、log intervalreveal evaluation logits。完成后reveal W/b/Ct并
保存NPZ（`:608-633,889-906`）。

## 9. Data Owners与Computing Parties

**`[SIMULATED]`角色，未实现K owners→三物理servers网络。**

K=5只是单进程逻辑partitions/variables；ABY3为S0/S1/S2三逻辑servers。True Ours动态
batch统一由server0 provider输入，Post-Aggregation从host NPZ构造private variables。
不存在五个独立owner进程。报告明确`single-machine process simulation`和至少一方诚实
假设；无LAN/WAN通信证据。

## 10. Privacy Boundary

### P1. 对象状态

| 对象 | 状态 | 边界 |
|---|---|---|
| compact features | `[PRECOMPUTED]` at rest；运行时`[MPC]` | 每阶段从host cache重新share |
| anchor features/labels | `[MPC]` | host预构造、integer→one-hot后share |
| local parameters | `[MPC]` | shared data内训练，不单独release θi |
| global parameters | `[MPC]→[REVEALED]` | Stage B/C结束save NPZ |
| per-example gradients | `[MPC]` derived/vectorized | 不显式materialize/reveal完整tensor |
| gradient norms | `[MPC]` | raw norms不reveal |
| clipping bound | `[MPC]→[REVEALED]` | current/proposed Ct作为诊断揭示 |
| Gaussian noise | `[LOCAL]` per-server part→`[MPC]` | 总noise不reveal，单机模拟 |
| intermediate logits | `[MPC]→部分[REVEALED]` | development checkpoint/log intervals |
| loss | training error`[MPC]`；eval loss`[REVEALED]` | 诊断边界揭示 |

### P2. Stage A DP

**否。** Official-MTT cache metadata明确`stage_a_dp_protected=false`、epsilon=N/A
（`new_prepare_public_mtt_reference_caches.py:38-63`）。raw data不离开不等于DP。

### P3. Stage B record-level DP

**否。** Secure Average/Anchor CE提供MPC confidentiality，无per-record clipping/noise；
initializer为data-dependent non-DP。

### P4. Stage C accountant

仅覆盖Stage-C cache feature records上的sampled Gaussian gradient机制，以及adaptive
count steps。`new_privacy_accountant.py:9-31`用Opacus RDPAccountant组合。它不覆盖
Stage A、Stage B、anchor或initializer。当前冻结目标ε≤3、δ=1e-5，但代码强制
`initializer_privacy_status=data_dependent_non_dp`，所以只是conditional Stage-C
mechanism claim，不是最终模型end-to-end record DP。

## 11. Q1-Q5

| 问题 | 结论 | 安全表述 |
|---|---|---|
| Q1 Stage A reduces protected workload | **YES_WITH_SCOPE** | 固定compact cache把下游secure-training records降到480/4,800/9,600；不能声称client-local MTT或Stage-A DP。 |
| Q2 Stage B stronger initializer | **YES_WITH_SCOPE** | seed42 Severe TFE：C10 C5=57.62>C1=55.48>C4=18.84；C100 C5=17.50>C1=10.02>C4=8.02。只限该development协议。 |
| Q3 new C5 reduces Stage-C iterations | **NO** | 最新C5未接Stage C；Stage-B内部轨迹不能冒充Stage-C iterations。 |
| Q4 Adaptive improves utility | **NO** | 历史ε=3消融C10/C100持平、Tiny更差且Adaptive更慢；只能说机制已实现/部分parity。 |
| Q5 end-to-end record DP | **NO** | Stage A无DP、Stage B非DP、accountant仅条件式Stage C。 |

## 12. 方法设计与当前实现

| 组件 | 设计 | 当前实现 | 一致？ | 框图建议 |
|---|---|---|---|---|
| Client-local distillation | 每客户端本地MTT | global Official-MTT asset | 否 | 画precomputed asset；client MTT标future/design |
| Feature encoder | shared/frozen | offline dataset/seed ConvNet，formal读cache | 部分 | MPC框外画offline encoder |
| Local head training | 本地 | formal在TFE内训练logical heads | 否 | 放MPC框内 |
| Secure Average | TFE/ABY3 | secure average linear W+b | 有限一致 | 标明只平均head parameters |
| Anchor construction | client contributions/balanced | host global 8/class | 否 | pre-balanced global anchor |
| Anchor CE | secure | latest C5全secure至final reveal | 是 | MPC框内，末端标reveal |
| θB→Stage C | shared handoff | reveal/save/reload/re-share；C5未接 | 否 | 显式断点/虚线 |
| Poisson sampling | CSPRNG | host OS-CSPRNG | 部分 | MPC框外sampling，随后share |
| Adaptive Ct | secure stats | secure raw stats/update，诊断reveal | 部分 | 标DP-noised reveal |
| Gaussian noise | secure/CSPRNG | 三逻辑server parts→ABY3 sum | 单机范围内一致 | 标single-machine simulation |
| Final update | secure | MPC update，final reveal | 至release一致 | 标final reveal/save |

框架图必须明确：`Latest C5 has not been connected to Stage C`。

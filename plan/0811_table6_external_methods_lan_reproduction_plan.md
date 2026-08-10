# Table 6 外部方法受控 LAN 复现实验操作文档

更新时间：2026-08-11 UTC

仓库：`/root/autodl-tmp/tfe_experiments`

规范分支：`unified-experiments`

状态：**EXECUTION PLAN；尚未安装网络工具、尚未下载外部代码、尚未启动实验。**

## 0. 目标与不可突破的边界

本任务为论文 Table 6 设计统一、可审计的 training-time 复现实验。目标不是预先制造
“本文一定最快”的结论，而是在相同硬件和受控 LAN 下测量后，让优势或负结果由真实
数据决定。

当前 Table 6 混用了：

- 本文方法：相对 matched Full Data 的 T@90；
- PEA：LAN 下达到 88% 的 7/55 分钟；
- CaPC/DDGauss：只有 PEA 作者适配后的最终 accuracy，没有 training time；
- CryptGPU：PEA 转述的“达到约 88% 超过 16 小时”，并非 CryptGPU 原论文直接给出的
  accuracy-time trajectory。

不能通过改写单位把这些数字变成严格排名。必须将 Table 6 分成两个层级：

### Block A：Controlled LAN reproduction

只有同时满足以下条件的方法才能进入严格时间比较区：

1. 同一 CIFAR-10 数据版本和冻结 split；
2. 同一三方物理 LAN，或同一三实例受控 emulated LAN；
3. 同一机器配额、网络校准和计时边界；
4. 真实方法后端，无 plaintext fallback；
5. 逐 checkpoint development trajectory；
6. 同一固定绝对 accuracy target；
7. 至少 seeds 42/43/44；
8. 输入、代码、配置、环境、网络和结果均有 SHA/manifest。

### Block B：Published contextual operating points

无法在 Block A 中忠实复现的方法继续使用公开数字，并明确标注：

```text
Published / not locally reproduced / not hardware-normalized
```

禁止将 Block B 的论文时间除以本地时间生成 speedup。

## 1. 当前 Table 6 的权威输入

执行者必须先完整读取：

1. `初稿_真实密态环境.md` §5.6；
2. `new_expTab/new_true_ours/new_external_method_source_registry.csv`；
3. `new_log/new_true_ours/new_external_method_source_registry.md`；
4. `reportToMe/unified_experiment_report.md`；
5. `reportToMe/paper_readiness_assessment.md`；
6. `plan/0810_networked_secure_training_efficiency_experiment_plan.md`；
7. 本文件。

本地 `goodPaper/` 只读且不提交，其中 `PEA.pdf` 是主要方法/硬件来源。外部来源优先
使用原论文和官方仓库；下载页面或二手博客不能作为配置依据。

当前已核验事实：

- PEA：IEEE S&P 2023，CIFAR-10，SimCLR features + logistic regression，三方，
  `epsilon=2, delta=1/(10n)`；论文报告 TFE `88.99±0.08%`、Queqiao
  `88.18±0.33%`，LAN 下分别在 7/55 分钟内超过 88%。
- CaPC 原文：ICLR 2021，50 effective parties、3 querying parties、VGG-5、
  `epsilon=20`，原文报告的是协作 gain `+4.09pp`，不是统一绝对 accuracy。
- Table 6 的 CaPC `86.70±0.17%, epsilon=2` 是 PEA 作者在 PEA features 上的
  适配复现，不是 CaPC 原始结果。
- DDGauss 原文没有 CIFAR-10；Table 6 的 `53.31±2.52%, epsilon=5` 是 PEA 作者
  的适配复现，且 PEA 说明 epsilon=2 不收敛。
- CryptGPU 原论文在三台 AWS p3.2xlarge（每台 V100 16GB）、约 1.25GB/s、0.2ms
  LAN 上报告 batch32 单次私有训练 iteration `12.14s / 7.55GB`，独立 accuracy 表的
  average-pooling VGG-16 为 `83.17%`。`>16h to 88%` 必须归因于 PEA。

## 2. 方法可行性分级

| Method | Block A优先级 | 当前可行性 | 允许标签 |
|---|---:|---|---|
| Full Data + TFE | P0 | 当前代码可改造成网络三方 | Locally reproduced |
| Ours + TFE | P0 | 当前代码可改造成网络三方 | Locally reproduced |
| PEA-TFE | P0 | 需官方代码或独立忠实实现门 | Official reproduction / Independent faithful reimplementation |
| PEA-Queqiao | P1 | 取决于框架源码、编译链和远程三方支持 | Locally reproduced 或 Published only |
| CaPC-PEA adaptation | P2 | 必须取得PEA适配代码；原CaPC代码不能自动等价 | PEA-style adaptation |
| DDGauss-PEA adaptation | P2 | 官方原代码无CIFAR-10，必须取得/重建PEA适配 | PEA-style adaptation |
| CryptGPU | P3 | 需要三台GPU节点及官方框架；当前单机CPU不可替代 | Locally reproduced 或 Published only |

“官方代码能运行”不代表设置一致。每个外部方法必须先生成
`method_reproduction_contract.json`，登记：

- 原论文身份与版本；
- 官方仓库 URL、commit、license；
- 本地修改 patch SHA；
- 原始设置和本地设置的逐字段差异；
- 数据、模型、参与方、DP 单位、epsilon/delta；
- 是否 MPC、是否真实网络、是否公开模型；
- 允许的论文标签；
- 是否有资格进入 Block A。

无公开代码或缺少关键配置时，状态写
`PUBLIC_ARTIFACT_INSUFFICIENT_DO_NOT_REIMPLEMENT_SILENTLY`，保留在 Block B。

## 3. 推荐的最小可投稿集合

优先完成：

```text
CIFAR-10
Methods: Full Data TFE, Ours TFE, PEA-TFE
Deployment: one controlled LAN mode
Seeds: 42, 43, 44
Total: 9 formal runs
```

这个集合能够回答：在同一 TFE/ABY3 类型后端、同一网络和计时协议下，本文紧凑训练
路线相对 Full Data 与 PEA 风格实现的达到固定效用时间如何。

Queqiao、CaPC、DDGauss、CryptGPU 只有通过各自 compatibility gate 后才扩展。不要因
Table 6 单元格为空而用简化 NumPy 版本代替。

## 4. 部署选择

### 4.1 首选：三台独立服务器的真实 LAN

```text
Node A -> MPC party 0
Node B -> MPC party 1
Node C -> MPC party 2
```

最低要求：

- 相同 CPU 型号、vCPU 配额、内存和 Linux/Python 环境；
- 各方独立 IP、端口和实验网卡；
- NTP/chrony 时钟同步；
- 实测吞吐 >=900Mbps，median RTT <=1ms，loss=0；
- 每方独立流量计数；
- 运行期间无其他训练或大流量任务。

此模式可写 `three-server LAN`。

### 4.2 当前单机可做：three-party emulated LAN

单机必须创建三个独立 network namespace/容器，每方独立 veth/interface，并通过
Linux `tc` 控制网络。三方同进程或共享 loopback 不算 LAN。

此模式只能写：

```text
three-party emulated LAN on one 32-vCPU host
```

当前服务器审计：32 vCPU、约503GiB内存，但 `ip/tc/iperf3/docker/podman` 当前均未
安装。正式实施前需安装 `iproute2` 和 `iperf3`。国内 apt 镜像不使用用户 9999
代理；只有 GitHub/海外官方源确实过慢时才临时使用代理，且不得把代理写进 freeze。

建议CPU隔离：

```text
party0: 10 vCPU
party1: 10 vCPU
party2: 10 vCPU
orchestrator/counter: 2 vCPU
```

所有比较方法使用相同映射。不得给 Ours 更多 CPU 或把外部方法与其他任务并跑。

### 4.3 Emulated LAN 网络目标

```text
rate: 1000mbit
pairwise median RTT: 0.2--1.0ms
loss: 0%
```

注意 `tc netem delay` 是单向出口延迟；链路两侧各加 0.1ms，RTT约增加0.2ms。不得
两侧各加1ms后仍写成1ms RTT。

## 5. 当前TFE代码必须先完成的架构门

当前 `new_stage_b_secure_init.py` / `new_stage_c_secure_dpsgd.py` 使用 TFE `ABY3()`
在一个 TensorFlow 图中创建逻辑 server0/1/2，正式报告明确是
`single-machine process simulation`。简单把进程放进一个 namespace 不会变成三方
LAN。

进入 Block A 前必须实现真正的三方进程部署：

1. 三个 party server 各自独立进程/namespace/IP；
2. orchestrator 只负责启动、barrier 和收集非秘密 metadata；
3. TFE remote config 明确映射 server0/1/2 地址；
4. private input provider 只连接其拥有方；
5. ABY3 shares 在网络方之间传输；
6. custom aux/secure_random ops 在三方都加载；
7. 任一方缺少 custom op 必须硬失败，不允许 fallback；
8. checkpoint 按方保存 shares，恢复后不重复 DP noise；
9. 训练结束后只执行协议允许的评估释放。

若 TFE 0.8.0 当前 fork 无法可靠远程部署，不得伪造 LAN。可选择：

- 修复/扩展 TFE remote config；或
- 在 SecretFlow-SPU 上实现三方网络版 Ours/Full/PEA-compatible 方法；
- 但不能把 SPU 数字标成 PEA-TFE。

## 6. 统一数据与模型协议

### 6.1 CIFAR-10 split

所有 Controlled LAN 方法共用一份预先冻结的 CIFAR-10 划分：

```text
training pool
development-validation pool（轨迹与T@Target）
locked official-test pool（最终一次性评估）
```

split 文件、索引和原始数据 SHA 必须一致。外部方法如果原生 split 不同，另外保留
`native reproduction`，不能混入 controlled block。

### 6.2 特征与模型

不能强迫所有方法使用同一种 feature/model，否则方法身份会被抹掉。必须同时记录：

- Ours：当前正式 MTT/compact features + secure linear head；
- Full Data：当前完整 frozen features + secure linear head；
- PEA：论文要求的 data-independent SimCLR features + logistic regression；
- CaPC：teacher/query/student 流程和 PATE-style privacy；
- DDGauss：federated optimizer + discrete Gaussian aggregation；
- CryptGPU：modified VGG-16 secure GPU training。

Table 6 的严格结论只能说“在统一部署/计时下的系统工作点”，不能说模型结构完全匹配。

## 7. 统一隐私登记

每条 run 必须登记：

```text
privacy_unit
adjacency_definition
epsilon
delta
accountant
accounted_steps
sampling_rule
noise_mechanism
released_objects
end_to_end_dp_claimed
```

保持原工作点的主复现：

- Ours：epsilon=3，conditional Stage-C DP；Stage B 非DP但MPC保密；
- PEA：epsilon=2，delta=1/(10n)；
- CaPC-PEA：epsilon=2，但PATE/collaborative-labeling单位不同；
- DDGauss-PEA：epsilon=5；
- CryptGPU：无DP。

这些 epsilon 不能直接横向解释。若需要 matched-budget sensitivity，另建表运行 Ours
epsilon=2；不得覆盖 Ours epsilon=3 主结果，也不得假设改 epsilon 不影响 T@Target。

## 8. 统一 Training Time 指标

### 8.1 固定绝对效用目标

当前本文 CIFAR-10 frozen target 为：

```text
A* = 0.65394 development accuracy
```

它来自 matched Full Data target freeze，必须在任何外部正式 run 前写入 LAN freeze。
Controlled Block 的主时间统一写：

```text
T@65.394% = first observed cumulative online training time whose
development accuracy >= 0.65394
```

未达到写 `N/R`，禁止插值。这样所有受控方法达到的是同一个绝对 accuracy，而不是
每种方法各自不同的“90%最终值”。

为了衔接本文原术语，可在表注注明 `65.394% is the pre-frozen T@90-Full target`，但
外部行的列名统一用 `T@65.394%`，不要误写成各自的 T@90。

### 8.2 补充指标

同时保存：

- final development accuracy；
- 一次性 official-test accuracy；
- total online training time；
- raw wall time；
- offline feature extraction/teacher training time；
- compile/setup/share preparation/evaluation time；
- communication TX/RX GB；
- peak RSS/GPU memory；
- processed protected examples at target。

### 8.3 在线计时边界

主 online timer：

```text
start: all computation parties ready and immediately before the first
       secret/private training input is consumed
end:   the secure/private update corresponding to the target checkpoint
       has materialized on all required parties
```

不包含：数据下载、encoder预训练、feature extraction、环境启动、图编译、秘密分享准备、
development evaluation和结果落盘。这些必须单独计时，不能删除。

对 CaPC/DDGauss 等不同流程，额外给出 phase decomposition；不能把昂贵 teacher/local
training 移到“offline”后只比较最后一小段。Table 6 的 `End-to-end wall` 必须包含形成
最终模型所必需的所有计算，`Online private time` 才按上述安全边界单列。

## 9. 网络校准与通信计数

### 9.1 安装后只读确认

```bash
ip -V
tc -V
iperf3 --version
ip netns list
```

### 9.2 每个正式实验块前后

三对 party 逐一执行：

- 20次以上 ping；
- 30秒 iperf3 single stream；
- 记录 median/p95 RTT、吞吐、retransmits；
- 训练后重复，检查漂移。

LAN gate：

```text
throughput >=900Mbps
median RTT <=1ms
loss=0
```

### 9.3 流量统计

每方在 barrier 前记录独立 interface TX/RX bytes，online end 后立即记录 delta。必须
排除 iperf、Git、SSH 和日志上传流量。若使用 pcap/eBPF，过滤固定 party ports，并将
filter/config SHA 写入报告。

不得由 tensor shape 推算通信量，不得用 packet 数冒充 communication rounds。

## 10. 统一结果 Schema

每条 run 保存 JSON：

```json
{
  "schema": "new_table6_lan_reproduction_v1",
  "status": "COMPLETE",
  "method": "ours|full_data|pea_tfe|pea_queqiao|capc_pea|ddgauss_pea|cryptgpu",
  "provenance": "official|faithful_reimplementation|pea_adaptation|published_only",
  "dataset": "cifar10",
  "seed": 42,
  "backend": "...",
  "deployment": "three_server_lan|three_party_emulated_lan",
  "privacy_unit": "...",
  "epsilon": 3.0,
  "delta": 1e-5,
  "accuracy_target": 0.65394,
  "time_to_target_s": null,
  "final_development_accuracy": null,
  "official_test_accuracy": null,
  "online_private_training_s": null,
  "end_to_end_wall_s": null,
  "offline_preprocessing_s": null,
  "compile_s": null,
  "evaluation_s": null,
  "tx_bytes": null,
  "rx_bytes": null,
  "network_calibration_pass": false,
  "plaintext_fallback": false,
  "input_manifest_sha256": "...",
  "config_sha256": "...",
  "runner_sha256": "...",
  "git_commit": "...",
  "dirty_worktree": false
}
```

`published_only` 行不伪造本地 schema 数值，另存 published registry，并在 controlled
aggregator 中硬拒绝。

## 11. 实施 Phase

### Phase 0：源码和许可 gate

为 PEA、Queqiao、CaPC、DDGauss、CryptGPU 分别生成：

```text
new_log/new_table6_lan/source_audit/<method>.json
new_log/new_table6_lan/source_audit/<method>.md
```

记录官方 URL、commit、license、依赖、数据脚本、论文配置位置和缺失项。GitHub 等
海外官方源只有直连确实过慢时才临时用用户9999代理；禁止输出token/代理凭据。

### Phase 1：受控LAN基础设施

1. 安装 `iproute2/iperf3`；
2. 创建三个 namespace/veth 和独立 party port；
3. 固定 CPU affinity；
4. 实现网络 apply/show/clear；
5. 实现 pre/post calibration 和 per-interface counters；
6. 断电/异常路径也必须 clear netem；
7. 所有基础设施先用 dummy TCP traffic 测试。

### Phase 2：三方远程后端 smoke

顺序：

1. TFE三方固定点加法/乘法；
2. TFE private input + reveal allowed aggregate；
3. C10两epochStage-B + 五步Stage-C；
4. checkpoint中断/恢复；
5. nonzero TX/RX与计时边界；
6. 无fallback/custom-op缺失。

若TFE远程三方不可实现，停止 Block A 的 TFE 项，不能继续用LocalConfig冒充LAN。

### Phase 3：CIFAR-10 seed42 pilot

严格顺序：

```text
Full Data TFE seed42
Ours TFE seed42
PEA-TFE seed42
```

每条先跑短gate再跑完整pilot。pilot只用于验证完整轨迹、target、通信和资源，不进入
最终 mean±std。三条必须在同一 network calibration block 和空闲主机下执行。

通过条件：

- final accuracy 与对应单机/论文语义在预注册容差内；
- timeline 单调且 target 无插值；
- online/private/wall/eval分段完整；
- TX/RX非零且三方收支差异可解释；
- official test只在freeze后一次读取；
- 网络化结果没有未解释的 >2pp accuracy drift。

### Phase 4：P0正式矩阵

在 pilot 代码/配置 freeze 后运行9条：

```text
3 methods x seeds 42/43/44
```

每次只运行一个方法/seed，避免32vCPU和1Gbps链路争用。每条完成立即validate、更新
progress、commit、push；checkpoint不提交。

### Phase 5：P1/P2兼容性扩展

按顺序：

1. PEA-Queqiao；
2. CaPC-PEA adaptation；
3. DDGauss-PEA adaptation。

每种先做source contract和seed42 pilot。任何方法的流程、epsilon或模型与PEA表不一致
时，不得沿用Table 6当前名字。

### Phase 6：CryptGPU决策门

只有具备三台兼容GPU服务器、官方代码可构建、同一LAN校准和至少48小时独占窗口时
才启动。当前单机32-vCPU环境不能用CPU模拟时间代替。

若不满足，保留 published context：

- 原论文：12.14s/iteration、7.55GB/iteration、83.17%独立accuracy；
- PEA转述：>16h to about 88%；
- 两者不能合并成一个“原论文轨迹”。

### Phase 7：聚合与Table 6更新

aggregator只接受：

- COMPLETE；
- exact freeze/input/code hash；
- calibration PASS；
- 无fallback；
- seeds 42/43/44齐全；
- target=0.65394；
- 同一deployment/hardware；
- development trajectory和official-test边界合格。

## 12. 新 Table 6 格式

建议正文 controlled block：

| Method | Backend / deployment | Privacy | Accuracy | T@65.394% | Online private time | End-to-end wall |
|---|---|---|---:|---:|---:|---:|
| Full Data | TFE / emulated LAN | conditional ε=3 | mean±std | mean±std | mean±std | mean±std |
| **Ours** | TFE / emulated LAN | conditional ε=3 | mean±std | **mean±std** | mean±std | mean±std |
| PEA-TFE | TFE / emulated LAN | ε=2 | mean±std | mean±std或N/R | mean±std | mean±std |

只有这些相同部署的行可以计算：

```text
T@Target speedup = baseline paired-seed T@Target / Ours paired-seed T@Target
time reduction = 1 - Ours / baseline
```

published block继续列：PEA-Queqiao、CaPC-PEA、DDGauss-PEA、CryptGPU，并在training
time列写原文口径和`Published, not normalized`，不能补估计mean±std。

如果最终使用真实三服务器，将 deployment 全部替换为 `three-server LAN`；不得把
emulated LAN 简写为 LAN。

## 13. 图表交付

建议新增：

```text
new_expTab/new_table6_lan/new_table6_controlled_lan.csv
new_expTab/new_table6_lan/new_table6_published_context.csv
new_expTab/new_table6_lan/new_table6_runtime_breakdown.csv
new_expTab/new_table6_lan/new_table6_network_summary.csv
new_expFig/new_table6_lan/new_table6_accuracy_time.{png,pdf,svg}
new_log/new_table6_lan/new_table6_hash_manifest.json
reportToMe/new_table6_lan_reproduction_report.md
```

图只画 controlled runs；published points 不用虚构轨迹。曲线无插值，n=3才有误差带，
caption明确development target、网络类型、隐私差异和方法模型不同。

## 14. 时间和资源预算

### 最小P0集合

| 工作 | 预计时间 |
|---|---:|
| 网络工具/namespace/counter/calibration | 0.5--1天 |
| TFE真正三方远程执行与恢复改造 | 1--3天 |
| PEA-TFE源码审计与忠实实现 | 1--3天 |
| seed42 pilots与9条正式运行 | 0.5--1天 |
| 聚合、验收、Table 6/报告 | 0.5天 |
| **总计** | **约3.5--8.5天** |

### 全部外部方法

- Queqiao：额外2--7天，且可能因旧框架/源码不可用停止；
- CaPC-PEA：额外2--5天；
- DDGauss-PEA：额外2--5天；
- CryptGPU：额外3--7天工程时间和至少16--48小时三GPU运行。

完整本地复现保守需要2--4周，并依赖外部代码与GPU服务器。建议先完成P0并更新Table 6，
再依据论文收益决定P1/P2；不要一次性启动全部路线。

## 15. 硬停止条件

出现任一条件立即停止该方法并保存诊断：

- 只有单进程loopback，无法证明网络三方；
- 官方源码/关键配置不可得；
- 需要未经说明的大规模重写，方法身份改变；
- plaintext fallback；
- 网络校准失败；
- DP accountant/单位无法复核；
- official test被用于调参或逐checkpoint曲线；
- 输入/split/hash不一致；
- 同一方法不同seed使用不同commit/config；
- T@Target通过插值、估算或日志尾数补造；
- GPU方法使用CPU时间替代；
- 外推总耗时超过用户批准预算。

失败状态必须写入Table 6 source audit，但不得为了“突出优势”把失败方法的时间写成
无穷大；只有完整运行真实未达到target时才可写 `N/R within frozen budget`。

## 16. Git与协作规则

每次修改都：

1. 更新 `progressLog/project_progress.md`；
2. 精确 `git add`，不提交 `new_checkpoints/`、外部大仓库、数据集或`goodPaper/`；
3. commit并push `unified-experiments`；
4. 外部仓库以URL+commit+patch SHA登记，不把整个vendor历史塞进本仓库；
5. token只通过临时askpass读取`/root/.config/githubToken.txt`，不得输出；
6. 正式实验使用tmux并记录精确命令；
7. 不清理用户现有dirty worktree。

## 17. 完成定义

P0只有在以下条件全部满足时才可标记COMPLETE：

1. 三方LAN/emulated-LAN隔离和网络校准可审计；
2. Full/Ours/PEA-TFE各3 seeds完整；
3. 同一target、计时、硬件和网络schema；
4. 所有输入/代码/config/result SHA通过；
5. 无fallback、无test调参、无不合格恢复；
6. Table 6 controlled/published两个block明确分开；
7. 只有controlled行计算speedup；
8. 结果不论正负均保留；
9. 报告、表格、图、progress已提交并推送；
10. 论文表述不再把本地单进程模拟写成LAN。

# Networked Secure-Training Efficiency 具体实验执行文档

> 文档状态：待执行（EXECUTION PLAN）  
> 目标分支：`unified-experiments`  
> 实验主题：真实或受控仿真的网络化密态训练效率  
> 主比较：同一后端、同一网络、同一数据集下的 `Full Data` 与 `Ours`  
> 主要后端：TF Encrypted（TFE）与 SecretFlow-SPU  
> 主要网络：LAN；WAN 为第二优先级敏感性实验  
> 重要声明：第 17 节中的数值是验收目标，不是实验结果，禁止直接写入论文。

---

## 0. 这项实验最终要回答什么

该实验不是普通的“脚本运行时间测试”，而是要回答一个可被审稿人直接检查的问题：

> 在给定隐私预算与目标准确率的条件下，紧凑训练载体和跨阶段初始化是否能够减少网络化 MPC 训练达到目标效用所需的在线密态时间与通信量？

必须同时提供以下四类证据：

1. **效用证据**：最终 development-validation accuracy，以及固定目标下是否成功达到目标。
2. **时间证据**：Stage B、Stage C、端到端在线密态时间，以及 `T@Target`。
3. **网络证据**：每个计算方发送/接收字节数、总发送字节数、实际 RTT 与实际吞吐率。
4. **可复现证据**：硬件、软件、网络配置、代码提交、输入资产哈希、随机种子和逐运行状态。

主结论只允许进行**后端内部比较**：

```text
TFE-LAN: Full Data vs Ours
SPU-LAN: Full Data vs Ours
TFE-WAN: Full Data vs Ours
SPU-WAN: Full Data vs Ours
```

禁止用 `TFE runtime / SPU runtime` 声称某个后端优于另一个后端。两个后端的实现、图编译、定点数语义和运行时调度不同，只用于证明趋势具有跨后端可复现性。

---

## 1. 当前证据与本实验要补的缺口

### 1.1 已有、可作为配置参照的证据

当前仓库已经具有单机/单物理节点条件下的在线密态时间证据：

| Backend | Dataset | Full Data T@90 (s) | Ours T@90 (s) | Existing speedup |
|---|---|---:|---:|---:|
| TFE | CIFAR-10 | 136.686 | 53.833 | 2.54x |
| TFE | CIFAR-100 | 466.872 | 184.336 | 2.53x |
| TFE | Tiny-ImageNet | 1491.452 | 607.477 | 2.46x |
| SPU | CIFAR-10 | 1063.8 | 113.6 | 9.36x |
| SPU | CIFAR-100 | 6172.2 | 2570.0 | 2.40x |
| SPU | Tiny-ImageNet | 21154.7 | 6067.6 | 3.49x |

来源：

```text
new_expTab/new_true_ours/new_figure2_time_to_target_table.csv
new_log/new_secretflow_spu/corrected_v2/new_spu_corrected_v2_final_report.md
```

这些数据可用于冻结实验入口、估计工时和设置异常检测区间，但**不能替代网络化结果**。

### 1.2 当前缺口

当前正式产物中的 `communication_bytes` 和 `communication_rounds` 为 `N/A`，且计算方主要运行在同一物理节点。缺少：

- 三个独立计算方之间的真实或受控仿真网络；
- 可审计的实际 LAN/WAN 带宽与 RTT；
- Stage B、Stage C 和全流程的通信字节数；
- 同一后端、同一网络条件下 Full Data 与 Ours 的配对比较；
- 网络约束对 `T@Target` 加速比的影响。

因此，本实验完成后才能把“secure runtime improvement”提升为“networked secure-training efficiency”。

---

## 2. 科学口径与不可变边界

### 2.1 主要假设

在固定后端、隐私预算、数据划分和目标准确率下，Ours 应通过两条路径降低网络化安全训练成本：

1. 紧凑训练载体减少进入安全训练的数据量和 Stage B 的批次数；
2. 更有效的初始化减少达到目标效用所需的累计在线密态训练时间。

### 2.2 冻结指标

主指标：

```text
T@90 = 首次达到冻结的 90% Full-Data development plateau 时，
       已累计消耗的在线密态训练时间。
```

冻结目标必须直接读取：

```text
new_configs/new_true_ours/new_true_ours_target_freeze.json
```

| Dataset | T@80 | T@90（主指标） | T@95 |
|---|---:|---:|---:|
| CIFAR-10 | 0.58128 | 0.65394 | 0.69027 |
| CIFAR-100 | 0.34464 | 0.38772 | 0.40926 |
| Tiny-ImageNet | 0.22120 | 0.24885 | 0.262675 |

规则：

- 取第一个 `accuracy >= target` 的归档检查点；
- 不进行线性插值；
- 未达到时写 `Not Reached`，不得以最后一个时间点代替；
- development-validation evaluation 不计入在线密态训练时间，但其起止必须单独记录；
- test/locked holdout 不得用于调参、目标定义或中途选择。

### 2.3 隐私与协议冻结

正式运行保持当前论文配置：

```text
epsilon = 3.0
delta   = 1e-5
Stage C accountant = Opacus RDP, including adaptive-count releases
SPU protocol = ABY3 / FM128 / fxp16
```

不得因为 LAN/WAN 结果不理想而更改：

- 数据划分；
- 目标阈值；
- 模型结构；
- 每个方法的冻结训练配置；
- 隐私预算；
- 评估间隔；
- Ours 或 Full Data 的输入资产。

网络实验只改变部署位置和网络条件，不重新做模型调参。

---

## 3. 部署模式

### 3.1 首选：三台独立 Linux 计算节点

每台计算节点承载一个 MPC 计算方：

```text
Node A -> Party 0
Node B -> Party 1
Node C -> Party 2
```

最低要求：

- 三台节点 CPU 型号、核数配额和内存尽量一致；
- 使用同一 Linux 发行版和相同 Python/依赖版本；
- 使用专用实验网卡或可按端口计数的网络；
- 三台机器时钟通过 NTP/chrony 同步；
- 实验期间不得运行其他高流量或高 CPU 任务；
- 数据持有方仍保持当前五方语义，三台服务器是 MPC 计算方，不得把“三台服务器”误写成“三个数据客户端”。

这种结果可在论文中写为 **three-server LAN/WAN deployment**。

### 3.2 可接受：单机三隔离实例的受控网络仿真

若没有三台物理服务器，必须使用三个独立容器、虚拟机或 network namespace，并为每个计算方分配独立 veth/interface。不得只在同一 Python 进程中设置一个 `network=LAN` 标签。

这种结果只能写为：

```text
three-party emulated LAN
three-party emulated WAN
```

不得写成真实三服务器部署。

### 3.3 禁止的部署

以下运行不能作为 Networked Secure-Training Efficiency 正式结果：

- 三方全部在同一进程或共享 loopback，且没有网络隔离；
- 只在配置文件中填写 `1 Gbps` 或 `20 ms`，没有实测；
- 将 wall time 当成 online secure time；
- 用操作系统总网卡流量代替实验进程流量，而网卡上同时有其他任务；
- 用 TCP packet 数推断“通信轮数”；
- 一方执行真实 MPC，另一方法退化为 NumPy/PyTorch plaintext fallback。

---

## 4. 网络条件

### 4.1 LAN（主结果）

目标条件：

```text
nominal bandwidth: 1 Gbps
pairwise RTT: <= 1 ms
packet loss: 0%
```

真实 LAN 使用实测值，不强制添加延迟。仿真 LAN 可在每个计算方出接口配置：

```bash
sudo tc qdisc replace dev ${IFACE} root netem rate 1000mbit delay 0.1ms loss 0%
```

### 4.2 WAN（第二优先级）

目标条件：

```text
nominal bandwidth: 100 Mbps
pairwise RTT: approximately 20 ms
packet loss: 0%
```

若三个计算方的每个出接口均施加延迟，则每侧使用约 10 ms 单向延迟：

```bash
sudo tc qdisc replace dev ${IFACE} root netem rate 100mbit delay 10ms loss 0%
```

不得同时在链路两端各配置 20 ms 后仍将结果标记为 20 ms RTT。

### 4.3 网络校准

每个正式实验块开始前执行：

1. 三对节点分别执行至少 20 次 `ping`；
2. 三对节点分别执行 30 秒 `iperf3` 单流测试；
3. 记录 median RTT、p95 RTT、实测吞吐和重传次数；
4. 校准完成后停止 `iperf3`，再开始训练；
5. 一个实验块结束后再次执行同样校准，检查网络漂移。

LAN 验收：

```text
measured throughput >= 900 Mbps
median RTT <= 1 ms
```

WAN 验收：

```text
90 Mbps <= measured throughput <= 110 Mbps
18 ms <= median RTT <= 22 ms
```

若超出区间，该实验块标记 `NETWORK_CALIBRATION_FAIL` 并重跑，不能直接进入汇总。

---

## 5. 正式实验矩阵

### 5.1 主矩阵

```text
Datasets: CIFAR-10, CIFAR-100, Tiny-ImageNet
Methods:  Full Data, Ours
Backends: TFE, SecretFlow-SPU
Networks: LAN, WAN
Seeds:    42, 43, 44
```

完整正式矩阵：

```text
3 datasets x 2 methods x 2 backends x 2 networks x 3 seeds = 72 runs
```

### 5.2 优先级

按以下顺序执行，不能为了获得更好结果改变顺序：

1. `LAN + CIFAR-10 + TFE/SPU + seed42` smoke-to-formal transition；
2. LAN 的 CIFAR-10 全部 3 seeds；
3. LAN 的 CIFAR-100 全部 3 seeds；
4. LAN 的 Tiny-ImageNet 全部 3 seeds；
5. WAN 的 CIFAR-10 全部 3 seeds；
6. WAN 的 CIFAR-100 全部 3 seeds；
7. WAN 的 Tiny-ImageNet 全部 3 seeds。

最低可投稿集合是完整 LAN 36 runs。WAN 是强烈建议的补充，不得把只完成 seed42 的 WAN 数据与三 seed LAN 均值混在同一统计列中。

### 5.3 可选 Random Subset 控制

Random Subset 不是主网络矩阵，因为它与 Ours 的训练载体规模相同，主要作用是区分“数据变少”与“蒸馏载体更有效”。若论文篇幅允许，增加：

```text
TFE + LAN + 3 datasets + Random Subset + seeds 42/43/44 = 9 runs
```

Random Subset 的样本数必须从 Ours 输入资产自动读取并严格相等。该控制只进入补充表或 Stage-A 分析，不替代 Full Data vs Ours 主比较。

---

## 6. 方法与输入资产冻结

### 6.1 Full Data

Full Data 使用当前正式 Full-Data 特征资产、数据划分、Stage-B 初始化训练和 Stage-C 配置。不得重新抽样或降低 Full Data 的输入规模以缩短运行时间。

### 6.2 Ours

Ours 使用当前正式紧凑表示、secure lightweight-head training、secure parameter averaging、labeled-anchor CE refinement 和 adaptive private optimization 语义。

### 6.3 必须自动登记的输入信息

每条运行必须从实际加载的文件中记录，禁止手填：

- 训练样本数；
- 特征维度；
- 类别数；
- 每个数据持有方的样本数；
- compact/full ratio；
- 特征文件路径和 SHA256；
- 标签文件路径和 SHA256；
- split 文件路径和 SHA256；
- initializer/config SHA256；
- runner Git commit 与 dirty status。

若输入哈希与 formal freeze 不一致，状态写 `INPUT_LINEAGE_FAIL`，不得开始正式运行。

---

## 7. 计时边界

### 7.1 必须分开的时间

每条运行至少输出：

| Field | 定义 | 是否进入主 online time |
|---|---|---|
| `offline_distillation_s` | 本地数据蒸馏时间 | No |
| `offline_encoding_s` | shared frozen encoder 本地编码时间 | No |
| `share_preparation_s` | 序列化/秘密分享准备 | No，单独报告 |
| `backend_startup_s` | 进程、Ray/TFE server、session 启动 | No，单独报告 |
| `compile_s` | 图/JAX/XLA/协议编译 | No，单独报告 |
| `stage_b_online_s` | Stage B 第一个密态输入被消费至 initializer shares 生成 | Yes |
| `stage_c_online_s` | Stage C 第一个私有 batch 至 final model shares 生成 | Yes |
| `total_online_s` | Stage B online + Stage C online | Yes |
| `evaluation_s` | development-validation 评估 | No，单独报告 |
| `raw_wall_s` | 进程启动至全部产物落盘 | No，诊断字段 |

### 7.2 在线时间的精确定义

```text
online_start:
  三个计算方均 ready，且第一份 secret-shared training input
  即将进入第一个安全训练算子之前的 barrier。

online_end:
  最后一次 secure model update 完成，最终模型 shares 已在三个计算方
  materialized，并通过 barrier，但尚未进行授权重构之前。
```

统一使用 `time.monotonic_ns()`。三方各自计时，并以：

```text
phase_time = max(party_end) - min(party_start)
```

作为保守的阶段 wall time，同时保留每方本地时间用于审计。

### 7.3 T@Target 计时

每个预先冻结的 checkpoint：

1. 完成安全更新并停止累计在线计时；
2. 保存模型 shares 和 `cumulative_online_s`；
3. 在审计评估路径上计算 development-validation accuracy；
4. 评估时间不加入 `cumulative_online_s`；
5. 从同一模型 shares 恢复下一训练段；
6. 首次满足目标的 checkpoint 给出 `T@80/T@90/T@95`。

评估频率必须在运行前冻结，Full Data 与 Ours 对同一数据集使用相同检查点规则。

---

## 8. 通信量采集

### 8.1 主要通信指标

主通信量定义为：

```text
total_communication_bytes = sum(TX bytes of Party 0, Party 1, Party 2)
```

只累加计算方之间与该次实验端口相关的流量。不要把发送和接收相加后再次除二，也不要把 SSH、Git、监控或数据下载流量计入。

分别报告：

- Stage B TX/RX；
- Stage C TX/RX；
- total online TX/RX；
- 每方 TX/RX；
- 最大单方通信量；
- packets、TCP retransmissions（诊断）；
- communication rounds（仅在可精确审计时报告）。

### 8.2 推荐采集优先级

1. **首选**：后端原生 send/recv byte counters；
2. **次选**：按计算方 peer IP 与实验端口配置 `nftables` 独立计数器；
3. **可接受**：专用容器 veth/interface 的 `/sys/class/net/<iface>/statistics/{tx,rx}_bytes`；
4. **审计辅助**：`tcpdump`/pcap，只保留摘要和 SHA256，原始大文件不提交 Git。

不得使用共享物理网卡总计数，除非实验期间该网卡完全隔离且有证据证明无背景流量。

### 8.3 通信审计门

每条正式运行必须满足：

```text
sum_tx_bytes > 0
sum_rx_bytes > 0
abs(sum_tx_bytes - sum_rx_bytes) / sum_tx_bytes <= 0.02
all party counters are monotonic
counter timestamps cover exactly the online phase
```

超过 2% 差异时标记 `COMMUNICATION_AUDIT_FAIL`。不得手动补齐或按样本数估算通信量。

### 8.4 通信轮数

通信轮数必须来自后端协议层消息屏障或经审计的 send/recv round instrumentation。TCP packet 数、函数调用数、训练 step 数都不是通信轮数。

若当前后端无法可靠获取，正式表中写 `N/A`，同时仍报告字节数。禁止推算一个看起来合理的轮数。

---

## 9. 需要新增的代码与目录

所有新工作放入独立 namespace：

```text
new_scripts/new_network_efficiency/
new_configs/new_network_efficiency/
new_log/new_network_efficiency/
new_expTab/new_network_efficiency/
new_expFig/new_network_efficiency/
new_output/new_network_efficiency/
new_checkpoints/new_network_efficiency/   # 永不提交
```

建议新增文件：

```text
new_scripts/new_network_efficiency/
  new_network_preflight.py
  new_network_calibrate.py
  new_network_counters.py
  new_network_run_tfe.py
  new_network_run_spu.py
  new_network_run_matrix.py
  new_network_validate_run.py
  new_network_aggregate.py
  new_network_plot.py

new_configs/new_network_efficiency/
  new_network_topology.example.yaml
  new_network_conditions.json
  new_network_formal_matrix.json
  new_network_formal_freeze.json
```

要求：

- 复用当前正式 TFE/SPU 计算内核，网络 wrapper 不得复制后悄悄改变算法；
- TFE 与 SPU 分开入口，避免依赖环境互相污染；
- 所有 runner 支持 `--check-only`、`--run-id`、`--resume` 和 `--output-root`；
- smoke 与 formal 使用不同输出目录；
- 每条运行开始前写 `RUNNING` 原子状态，成功后写 `COMPLETE`；
- 异常退出写 `FAILED` 与 traceback，不得留下伪 `COMPLETE`；
- checkpoint 包含阶段、step、optimizer、accountant、clip state 和噪声 counter；
- resume 后不得重复使用 DP noise，也不得重复累计通信/时间。

---

## 10. 逐运行 JSON Schema

每条正式运行生成：

```text
new_log/new_network_efficiency/formal/<run_id>/run_report.json
new_log/new_network_efficiency/formal/<run_id>/timeline.csv
new_log/new_network_efficiency/formal/<run_id>/network_before.json
new_log/new_network_efficiency/formal/<run_id>/network_after.json
new_log/new_network_efficiency/formal/<run_id>/stdout.log
```

`run_report.json` 至少包含：

```json
{
  "schema": "networked_secure_training_efficiency_v1",
  "run_id": "lan_spu_cifar10_true_ours_seed42",
  "status": "COMPLETE",
  "git_commit": "...",
  "git_dirty": false,
  "dataset": "cifar10",
  "method": "true_ours",
  "backend": "secretflow_spu",
  "protocol": "ABY3",
  "network_mode": "lan",
  "deployment_kind": "three_server_real_or_three_instance_emulated",
  "seed": 42,
  "epsilon": 3.0,
  "delta": 1e-5,
  "training_sample_count": 0,
  "feature_dim": 0,
  "num_classes": 0,
  "target_t90": 0.0,
  "target_reached": false,
  "final_development_accuracy": 0.0,
  "t80_online_s": null,
  "t90_online_s": null,
  "t95_online_s": null,
  "offline_distillation_s": null,
  "offline_encoding_s": null,
  "share_preparation_s": 0.0,
  "backend_startup_s": 0.0,
  "compile_s": 0.0,
  "stage_b_online_s": 0.0,
  "stage_c_online_s": 0.0,
  "total_online_s": 0.0,
  "evaluation_s": 0.0,
  "raw_wall_s": 0.0,
  "stage_b_tx_bytes": 0,
  "stage_c_tx_bytes": 0,
  "total_tx_bytes": 0,
  "total_rx_bytes": 0,
  "communication_rounds": null,
  "peak_rss_bytes_by_party": {},
  "measured_rtt_median_ms": 0.0,
  "measured_rtt_p95_ms": 0.0,
  "measured_throughput_mbps": 0.0,
  "input_hashes": {},
  "security_gates": {},
  "validation_gates": {}
}
```

数值 `0` 只是 schema 占位，不得在真实报告中保留为“未测量”的替代值；未测量字段必须为 `null` 并说明原因。

---

## 11. 汇总 CSV Schema

生成：

```text
new_expTab/new_network_efficiency/new_network_efficiency_per_run.csv
new_expTab/new_network_efficiency/new_network_efficiency_summary.csv
new_expTab/new_network_efficiency/new_network_efficiency_speedup.csv
new_expTab/new_network_efficiency/new_network_efficiency_paper_table.csv
```

`per_run.csv` 字段：

```text
run_id,status,dataset,method,backend,protocol,network_mode,deployment_kind,
seed,epsilon,delta,training_sample_count,feature_dim,num_classes,
final_development_accuracy,target_t90,target_reached,t80_online_s,
t90_online_s,t95_online_s,stage_b_online_s,stage_c_online_s,total_online_s,
raw_wall_s,stage_b_tx_bytes,stage_c_tx_bytes,total_tx_bytes,total_rx_bytes,
communication_rounds,peak_rss_bytes,measured_rtt_median_ms,
measured_rtt_p95_ms,measured_throughput_mbps,git_commit,input_hash,
network_gate,security_gate,privacy_gate,acceptance_gate
```

`summary.csv` 按 `dataset + method + backend + network_mode` 聚合，输出：

- `n_complete`；
- accuracy mean/std；
- T@90 mean/std；
- Stage B/C/total online mean/std；
- communication mean/std；
- peak memory mean/std；
- target-reached count。

样本标准差使用 `ddof=1`。若 `n < 2`，std 写 `N/A`，禁止伪造误差条。

---

## 12. 计算公式

### 12.1 时间加速

```text
Speedup_T90 = mean(T90_Full) / mean(T90_Ours)
Speedup_Online = mean(TotalOnline_Full) / mean(TotalOnline_Ours)
```

只有 Full 和 Ours 都达到目标时才计算 `Speedup_T90`。任一方 `Not Reached` 时报告原始状态，不用最后时间代替。

### 12.2 通信下降

```text
Communication Reduction (%)
= (1 - mean(Bytes_Ours) / mean(Bytes_Full)) x 100
```

同时给出通信比：

```text
Communication Ratio = mean(Bytes_Ours) / mean(Bytes_Full)
```

### 12.3 效用变化

```text
Accuracy Delta (pp)
= 100 x (mean(Acc_Ours) - mean(Acc_Full))
```

不得把百分点差写成百分比提升。

### 12.4 每样本与每步诊断

仅用于解释，不作为主结论：

```text
bytes_per_secure_sample = total_tx_bytes / protected_sample_visits
seconds_per_secure_step = total_online_s / secure_update_steps
```

这两个指标可判断加速来自处理数据量减少，还是底层单步实现异常。

---

## 13. 执行阶段

### Phase 0：只读审计与冻结

执行前：

```bash
cd /root/autodl-tmp/tfe_experiments
git status --short --branch
git rev-parse HEAD
git fetch origin unified-experiments
df -h
free -h
lscpu
nvidia-smi || true
```

完成：

- 读取 `AGENTS.md`、统一实验报告和最新 progress log；
- 确认没有活动正式实验；
- 确认 TFE 与 SPU 当前正式入口；
- 创建 network formal matrix；
- 记录三节点主机、IP、网卡、CPU、内存和软件版本；
- 生成 `new_network_formal_freeze.json`；
- freeze 必须在 smoke 后、formal 前提交并推送。

### Phase 1：网络与计数器实现

必须先通过单元测试：

- counter start/stop 只覆盖指定端口；
- TX/RX delta 正确；
- 进程失败时仍保存 counter；
- resume 不重复累计已完成阶段；
- LAN/WAN 配置可应用、读取、清除；
- 未使用 root 权限时立即失败，不静默跳过 `tc`；
- communication rounds 不可得时输出 `null`，不推算。

### Phase 2：网络校准

为 LAN/WAN 各生成：

```text
new_log/new_network_efficiency/calibration/lan_pre.json
new_log/new_network_efficiency/calibration/lan_post.json
new_log/new_network_efficiency/calibration/wan_pre.json
new_log/new_network_efficiency/calibration/wan_post.json
```

### Phase 3：Smoke

Smoke 只使用：

```text
Dataset: CIFAR-10
Method: Ours
Seed: 42
Backends: TFE, SPU
Networks: LAN, WAN
Training: 2 Stage-B epochs + 5 Stage-C steps
```

Smoke 验收：

- 三方进程位于不同 node/container/namespace；
- 无 plaintext fallback；
- 能产生非零且平衡的 TX/RX；
- Stage B/C 计时边界可区分；
- 模型 share checkpoint 可恢复；
- DP noise counter 恢复后不重复；
- smoke 产物与 formal 目录隔离；
- smoke accuracy 不进入论文。

### Phase 4：LAN 正式实验

使用 `tmux` 或调度器，先生成 registry，再逐条执行。建议每个后端同时只运行一条正式任务，避免 CPU、内存和网络争用。

命令接口应统一为：

```bash
python new_scripts/new_network_efficiency/new_network_run_matrix.py \
  --matrix new_configs/new_network_efficiency/new_network_formal_matrix.json \
  --backend tfe \
  --network lan \
  --resume

python new_scripts/new_network_efficiency/new_network_run_matrix.py \
  --matrix new_configs/new_network_efficiency/new_network_formal_matrix.json \
  --backend secretflow_spu \
  --network lan \
  --resume
```

每完成一条立即运行 validator。不要等 36 条全部跑完后才发现计时或通信计数无效。

### Phase 5：WAN 正式实验

LAN 全部通过后才能执行 WAN。WAN 必须使用同一 formal freeze，仅切换 `network_mode`。

### Phase 6：聚合与作图

```bash
python new_scripts/new_network_efficiency/new_network_aggregate.py \
  --input new_log/new_network_efficiency/formal \
  --output new_expTab/new_network_efficiency

python new_scripts/new_network_efficiency/new_network_plot.py \
  --summary new_expTab/new_network_efficiency/new_network_efficiency_summary.csv \
  --output new_expFig/new_network_efficiency
```

聚合器必须拒绝：

- 非 COMPLETE；
- network gate 失败；
- 输入哈希失败；
- epsilon/delta 不一致；
- test/holdout 被读取；
- plaintext fallback；
- TX/RX 审计失败；
- 同一 cell seed 重复；
- 使用不同 Git commit 且没有批准的恢复说明。

---

## 14. 论文图表设计

### 14.1 主表：Networked secure-training efficiency

建议主表字段：

| Dataset | Backend | Network | Method | Accuracy (%) | T@90 (s) | Online Time (s) | Communication (GB) | T@90 Speedup |
|---|---|---|---|---:|---:|---:|---:|---:|

排版规则：

- 每个 backend/network block 中仅比较 Full Data 与 Ours；
- `Ours` 可加粗；
- 未达到目标写 `N/R`；
- 时间、通信和准确率为 mean ± std；
- 加速比只在 Ours 行填写；
- LAN 进入正文主表，WAN 可放正文次表或 appendix。

### 14.2 主图：Accuracy vs cumulative online secure time

文件：

```text
new_expFig/new_network_efficiency/new_network_accuracy_time.png
new_expFig/new_network_efficiency/new_network_accuracy_time.pdf
```

建议 2 x 3 面板：

```text
Row 1: TFE-LAN, CIFAR-10 / CIFAR-100 / Tiny-ImageNet
Row 2: SPU-LAN, CIFAR-10 / CIFAR-100 / Tiny-ImageNet
```

图中：

- x 轴为 cumulative online secure time；
- y 轴为 development-validation accuracy；
- Full Data 与 Ours 使用固定且跨面板一致的颜色；
- T@90 以水平虚线表示；
- 三 seed 曲线先按共同时间网格聚合；
- n=3 才绘制误差带；
- 不在图底部添加额外小字说明，科学边界写入 caption。

### 14.3 通信图

文件：

```text
new_expFig/new_network_efficiency/new_network_communication.png
new_expFig/new_network_efficiency/new_network_communication.pdf
```

每个数据集显示 Full Data 与 Ours 的 total TX GB，按 TFE/SPU 分组。LAN 与 WAN 的协议消息字节理论上应接近；若差异很大，优先检查重传、压缩、连接重建或计数边界，而不是直接解释为算法现象。

### 14.4 阶段分解图（可选但推荐）

堆叠柱：

```text
Stage B online time + Stage C online time
Stage B communication + Stage C communication
```

作用：验证主要效率收益是否来自紧凑训练载体进入 Stage B 后减少受保护工作量，而不是 Stage C 偶然缩短。

---

## 15. 硬验收门

单条运行只有同时满足以下条件才是 `COMPLETE`：

1. Git commit、环境、硬件和输入哈希完整；
2. 三方隔离成立；
3. TFE/SPU 真实安全后端被调用；
4. 无 NumPy/PyTorch plaintext fallback；
5. 网络校准通过；
6. Stage B/C 计时边界完整；
7. TX/RX 非零且差异不超过 2%；
8. epsilon=3.0、delta=1e-5；
9. accountant 和 adaptive-count releases 完整；
10. test/locked holdout 未读；
11. timeline 单调，checkpoint 可恢复；
12. final accuracy 有限且在 `[0,1]`；
13. 运行没有 OOM、actor crash 或未解释 retry；
14. validator 输出 `PASS`。

整个实验只有同时满足以下条件才可生成论文主表：

- LAN 36/36 runs COMPLETE；
- 每个 cell 恰有 seeds 42/43/44；
- Full Data 与 Ours 使用同一后端、网络、目标与数据划分；
- 所有通信字段均有真实数值；
- 没有把历史单机结果混入网络均值；
- 没有把 smoke、调参或失败恢复的局部日志当正式结果。

---

## 16. 失败与恢复策略

### 16.1 网络失败

- 网络校准失败：停止该 block，清除 `tc` 后重新配置；
- 节点断连：该 run 标记 FAILED，从最近完整 share checkpoint 恢复；
- 恢复前重新采集 network calibration；
- 恢复段单独记录时间和流量，聚合时与已完成段相加；
- 不得重复计算已完成段。

### 16.2 DP 状态恢复

checkpoint 必须保存：

- global step；
- optimizer state；
- model shares；
- clipping bound `C_t`；
- accountant state；
- adaptive count release index；
- noise share counter/commitment。

无法证明恢复后噪声不重复时，该 run 作废并从头重跑。

### 16.3 结果异常

以下情况先做实现审计，禁止直接调参：

- 网络化准确率相对同配置单机下降超过 2 pp；
- LAN 运行时间超过历史单机同 cell 的 3 倍；
- Ours 的通信量高于 Full Data；
- LAN 与 WAN 的逻辑消息字节差异超过 10%；
- 三个 seed 中单个 runtime 偏离中位数超过 30%；
- SPU/TFE 出现不同输入样本数。

只有确定是实现错误后才能修复并重跑；修复必须形成新 commit，并在 run report 中登记 superseded run。

---

## 17. 理想目标与最低成功门槛

### 17.1 重要说明

下面数值根据当前单机结果和预期网络放大效应制定，仅用于：

- 判断实验是否值得进入论文主表；
- 识别实现或计数异常；
- 安排资源优先级。

它们不是已测结果，不能复制到摘要、正文、图表或 Git 实验报告的“Results”字段。

### 17.2 LAN 理想目标

| Dataset | Backend | Full T@90 target | Ours T@90 target | Desired speedup | Desired communication ratio | Utility gate |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | TFE | ~137 s | <= 50 s | >= 2.7x | <= 0.35 | Ours loss <= 2 pp |
| CIFAR-100 | TFE | ~467 s | <= 170 s | >= 2.7x | <= 0.45 | Ours >= Full |
| Tiny-ImageNet | TFE | ~1491 s | <= 500 s | >= 3.0x | <= 0.45 | Ours >= Full |
| CIFAR-10 | SPU | ~1064 s | <= 110 s | >= 9.5x | <= 0.25 | Ours loss <= 4 pp |
| CIFAR-100 | SPU | ~6172 s | <= 2400 s | >= 2.5x | <= 0.45 | Ours >= Full |
| Tiny-ImageNet | SPU | ~21155 s | <= 5800 s | >= 3.5x | <= 0.40 | Ours >= Full |

说明：网络化实测时间通常不会恰好等于历史单机时间，因此表中的时间只用于数量级检查。真正要守住的是**配对加速比、通信下降和效用**。

### 17.3 最低成功门槛

建议论文主张的最低门槛：

```text
1. 三个数据集、两个后端的 LAN T@90 配对加速均 >= 2.0x；
2. Ours 的 total communication bytes 相对 Full Data 平均下降 >= 50%；
3. CIFAR-10 的 Ours 最终效用下降不超过 4 pp；
4. CIFAR-100 与 Tiny-ImageNet 的 Ours 不低于 Full Data，或至少在误差范围内持平；
5. WAN 下 Ours 相对 Full Data 的加速比不低于对应 LAN 加速比的 80%；
6. 所有主要结论在 TFE 和 SPU 上方向一致。
```

若某个 cell 未达到门槛，应如实报告并分析，不得通过删除 seed、修改目标或更换时间边界来制造优势。

---

## 18. 资源与预计工时

根据当前 SPU 归档，Tiny-ImageNet Full 单条约 5.9 小时在线时间，网络约束可能进一步延长。粗略预算：

| Scope | Runs | Estimated wall time | Priority |
|---|---:|---:|---|
| Smoke | 4 | < 2 h | P0 |
| Full LAN matrix | 36 | 3-6 days serial; 需以实测修正 | P0 |
| Full WAN matrix | 36 | 5-10 days serial; 需以实测修正 | P1 |
| Random Subset control | 9 | 0.5-1.5 days | P2 |

并行原则：

- 同一三节点集群一次只跑一条网络正式任务；
- 不允许两个任务共享同一实验网卡计数窗口；
- 若有两套完全独立的三节点集群，可按 backend 并行；
- 先完成 LAN 全矩阵，再决定是否完整跑 WAN Tiny。

---

## 19. 最终交付物

必须生成：

```text
new_configs/new_network_efficiency/new_network_formal_freeze.json
new_log/new_network_efficiency/new_network_run_registry.csv
new_log/new_network_efficiency/new_network_validation_report.json
new_log/new_network_efficiency/new_network_validation_report.md
new_log/new_network_efficiency/new_network_final_report.md
new_expTab/new_network_efficiency/new_network_efficiency_per_run.csv
new_expTab/new_network_efficiency/new_network_efficiency_summary.csv
new_expTab/new_network_efficiency/new_network_efficiency_speedup.csv
new_expTab/new_network_efficiency/new_network_efficiency_paper_table.csv
new_expFig/new_network_efficiency/new_network_accuracy_time.png
new_expFig/new_network_efficiency/new_network_accuracy_time.pdf
new_expFig/new_network_efficiency/new_network_communication.png
new_expFig/new_network_efficiency/new_network_communication.pdf
```

最终报告必须明确列出：

- 真实三服务器还是三实例仿真；
- 每个网络条件的实测 RTT/吞吐；
- 72 条 formal run 的完成状态；
- 每个 cell 的 accuracy、T@90、online time、communication；
- Full Data vs Ours 的 speedup 与 communication reduction；
- 未达到目标和失败项；
- 是否读取 test/locked holdout；
- 是否存在 plaintext fallback；
- 可写入论文的结论与不能写入论文的边界。

---

## 20. Git 与运行纪律

1. 所有新代码、配置、计划和小型审计产物提交到 `unified-experiments`；
2. 每次仓库修改同步更新 `progressLog/project_progress.md`；
3. 正式运行必须从 clean commit 启动，并在报告中记录 commit；
4. `new_checkpoints/`、pcap 大文件和私有 runtime state 不提交；
5. formal freeze 提交并推送后才能启动正式矩阵；
6. 每完成一个数据集/后端/network block 就提交一次结果和进度；
7. 不覆盖历史单机 TFE/SPU 结果；
8. 新结果只写入 `new_network_efficiency` namespace；
9. 网络或 Git 中断时使用 `tmux`/调度器恢复，不凭终端显示判断任务仍在运行；
10. 最后执行 JSON 解析、CSV schema、Python syntax、`git diff --check` 和图像非空检查。

---

## 21. 最终验收清单

```text
[ ] 三方部署边界有截图/manifest/进程证据
[ ] LAN/WAN 实际 RTT 与吞吐已测量
[ ] TFE 和 SPU 均无 plaintext fallback
[ ] 输入资产、配置和 commit 已冻结
[ ] epsilon=3, delta=1e-5 全部通过
[ ] test/locked holdout 未读
[ ] LAN 36/36 COMPLETE
[ ] WAN 36/36 COMPLETE，或明确标记未完成范围
[ ] 每条运行 Stage B/C/total online time 可审计
[ ] 每条运行 TX/RX 非零且差异 <= 2%
[ ] communication rounds 无法精确获取时保持 N/A
[ ] T@80/T@90/T@95 按冻结目标计算且不插值
[ ] 三 seed mean/std 无伪造误差条
[ ] Full Data vs Ours 只做同后端同网络比较
[ ] 历史单机结果未混入网络汇总
[ ] 理想目标值未被写成真实结果
[ ] 主表、时间曲线、通信图均由正式 CSV 自动生成
[ ] progressLog、最终报告和 Git 提交完整
```

完成上述清单后，这项实验才能支持论文中的以下表述：

> The compact cross-stage pipeline reduces both the online secure-training time and network communication required to reach a fixed utility target under controlled LAN and WAN deployments.

在清单未完成前，只能称为单机 backend-internal runtime evidence，不能称为 networked secure-training result。

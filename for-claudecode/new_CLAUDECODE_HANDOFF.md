# Claude Code 详细交接：0727 Incremental Full-Feature 主线

> 更新时间：2026-07-27（Asia/Shanghai）
>
> 项目目录：`/root/autodl-tmp/tfe_experiments`
>
> Git 分支：`new-secure-feature-pipeline`
>
> 当前任务计划：`plan/0727_new_cross_stage_incremental_full_feature_plan.md`

## 0. 你接手后的第一原则

这是一个已经完成大量正式实验的论文项目，不是从零开始的工程。

接下来只做“增量实验”：

1. 保留已有 Stage A、Stage B、Stage C、旧 End-to-End、final-test 和 TFE system 全部结果；
2. 新增 `Full Data Feature Model`；
3. Full Data 与 Ours 都使用相同的 Adaptive DP-SGD；
4. 优先复用已有且效果较好的 Ours Adaptive End-to-End 结果；
5. 生成新的 `Full Data Feature Model vs Ours` 两曲线主图；
6. 整理 3–4 个外部方法的可靠文献对比；
7. 不做 DP-MTT；
8. 不复现 CryptGPU；
9. 不重跑已经完整且兼容的实验。

任何长任务都必须在 tmux 或现有持久队列机制中运行。任何代码、配置、报告或文档修改都必须同步更新 `progressLog/project_progress.md`，分阶段 commit 并 push GitHub。

不要为了快而减少 seed、epoch、模型、数据量或正式评估质量。不要重新设置单线程限制；要使用正常可用的 CPU/GPU 性能，同时防止无界进程或线程爆炸。

## 1. 必读顺序

开始修改前完整阅读：

1. `plan/0727_new_cross_stage_incremental_full_feature_plan.md`
2. 本文件
3. `for-claudecode/new_CONVERSATION_CONTEXT_INDEX.md`
4. `for-claudecode/new_CURRENT_AUTODL_WINDOW_SUMMARY.md`
5. `for-claudecode/new_LEGACY_INTERNAL_SERVER_CONVERSATION_SUMMARY.md`
6. `progressLog/project_progress.md`，重点看 2026-07-26 和 2026-07-27 的最后几段
7. `migration/CODEX_CONTEXT_HANDOFF.md`
8. `AUTODL_MIGRATION.md`
9. `new_configs/new_public_mtt_mainline/new_mainline_config.json`
10. `new_prepare_end_to_end_assets.py`
11. `new_prepare_end_to_end_baseline_stage_b.py`
12. `new_prepare_end_to_end_formal.py`
13. `new_stage_b_secure_init.py`
14. `new_stage_c_secure_dpsgd.py`
15. `new_report_end_to_end_formal.py`
16. `new_final_test_evaluation.py`
17. `new_scripts/new_run_task_queue.sh`
18. `new_scripts/new_single_gpu_supervisor.sh`

不要一次性读取私有历史 JSONL。交接文档和项目日志足以回答当前任务。任何令牌、密钥、代理认证字段都不得写入日志、文档、命令输出或 Git。

## 2. 当前机器存在硬阻塞，先不要跑实验

最后一次检查结果：

```text
nvidia-smi -L -> No devices found
/sys/fs/cgroup/cpu.max -> 50000 100000
nproc -> 1
```

这代表当前容器没有可见 GPU，CPU 实际配额约为 0.5 核，而计划声明的硬件是 RTX 5090 + 25 vCPU。

在以下条件全部满足前，不允许启动 pilot 或 formal：

```text
[ ] nvidia-smi 能看到 RTX 5090
[ ] CUDA 在现代环境中可用
[ ] CPU cgroup 配额恢复到约 25 vCPU，而不是 0.5 core
[ ] /root/autodl-tmp 至少还有 80 GiB 可用
[ ] 没有旧训练进程或旧活动队列在运行
[ ] Git 工作树差异已审计
```

建议的只读检查：

```bash
cd /root/autodl-tmp/tfe_experiments
source /root/.config/new-secure-feature-pipeline.env
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse origin/new-secure-feature-pipeline
nvidia-smi
nvidia-smi -L
lscpu
cat /sys/fs/cgroup/cpu.max
nproc
free -h
df -h /root/autodl-tmp
tmux ls
pgrep -af 'new_run_task_queue|new_stage_b_secure_init|new_stage_c_secure_dpsgd' || true
```

硬件恢复后，先按照 `AUTODL_MIGRATION.md` 的验证口径确认现代环境、legacy TFE 环境和 custom ops。不要重建环境，除非验证确实失败。

## 3. 用通俗语言理解接下来的核心实验

论文现在要回答一个简单问题：

> 在安全训练环境里，是把完整训练集的所有 2048 维特征都送进去训练更好，还是先用 MTT 把数据压缩成很小的蒸馏集，再使用更强的安全共识初始化更划算？

需要比较两条路线。

### 3.1 Full Data Feature Model

```text
完整训练图像
-> 已有的同一个共享 ConvNet encoder
-> 完整 2048-D 特征
-> 多方本地安全训练后做 Secure Average 初始化
-> TFE/ABY3 Secure Adaptive DP-SGD
-> 线性分类模型
```

它代表常规做法：不进行数据蒸馏，安全系统处理大量真实训练特征。

### 3.2 Ours

```text
Official MTT 蒸馏图像
-> 同一个共享 ConvNet encoder
-> 很小的 2048-D 蒸馏特征集
-> 已完成的 Secure Consensus 初始化
-> TFE/ABY3 Secure Adaptive DP-SGD
-> 线性分类模型
```

它代表论文方法：用少量蒸馏特征减少安全计算，再通过更好的初始化维持精度。

### 3.3 论文真正要比较什么

不要求 Ours 的最终最高精度一定超过 Full Data。

主要看：

1. 在双方都能达到的共同准确率目标上，Ours 是否更快；
2. Ours 到达目标时处理了多少 secret-shared examples；
3. Ours 的安全工作量减少多少；
4. Ours 最终保留了 Full Data 多少准确率；
5. 这种代价—精度折中在 CIFAR-10、CIFAR-100、Tiny ImageNet 上是否稳定。

## 4. 已完成成果：严禁无理由重跑

权威报告位于：

```text
new_log/new_public_mtt_mainline/json/
new_log/new_public_mtt_mainline/txt/
new_expFig/new_public_mtt_mainline/
new_expTab/new_public_mtt_mainline/
new_output/new_public_mtt_mainline/
```

已完成：

| 模块 | 状态 | 处理方式 |
|---|---|---|
| Stage A / Module A | COMPLETE | 复用 |
| Stage B convergence | COMPLETE | 复用 |
| Stage B IID/Mild/Severe | 27/27 | 复用 |
| Module C Fixed/Adaptive | 54/54 | 复用，Adaptive 负结果不能隐藏 |
| 旧 End-to-End | 18/18 | 保留为消融/旧比较 |
| 旧 final test | 72 models | 保留，不覆盖 |
| TFE system | 9/9 | 复用 |

关键权威文件包括：

```text
new_log/new_public_mtt_mainline/json/new_module_a_efficiency_report.json
new_log/new_public_mtt_mainline/json/new_module_b_convergence_formal_report.json
new_log/new_public_mtt_mainline/json/new_stage_b_initialization_report.json
new_log/new_public_mtt_mainline/json/new_module_c_formal_report.json
new_log/new_public_mtt_mainline/json/new_end_to_end_formal_report.json
new_log/new_public_mtt_mainline/json/new_final_test_evaluation.json
new_log/new_public_mtt_mainline/json/new_tfe_system_formal_report.json
```

不要修改、删除或覆盖这些文件。新实验必须写到新的 incremental 路径。

## 5. 已有资产：完整特征不需要重新抽取

完整训练特征已经存在：

```text
new_cache/new_public_mtt_mainline/new_full_features/
```

形状：

| 数据集 | 每个 seed 的形状 | seeds |
|---|---:|---|
| CIFAR-10 | 50000 x 2048 | 42, 43, 44 |
| CIFAR-100 | 50000 x 2048 | 42, 43, 44 |
| Tiny ImageNet | 100000 x 2048 | 42, 43, 44 |

9 份 manifest 均声明：

```text
schema = new_full_feature_cache_v1
status = COMPLETE
test_loaded = false
```

总大小约 4.6–4.9 GiB。优先 mmap 使用，不要重复复制成多个巨型缓存。

正式使用前必须逐份验证：

1. 文件存在；
2. shape 与 dtype 正确；
3. feature/label SHA256 与 manifest 一致；
4. encoder/scaler signature 与同 seed 的 Official-MTT cache 一致；
5. 每个特征向量满足既有 L2 bound；
6. 不包含 test；
7. manifest 里的旧服务器绝对路径只作为历史字段，实际路径必须由当前 `NEW_PROJECT_ROOT` 重建。

不要因为 manifest 中仍显示 `/mnt/code/lixujie/...` 就重新抽特征。先按当前相对目录解析并验证哈希。

## 6. 最容易把实验做错的八个问题

### 6.1 不能把 reference/selection 样本拿去训练 Full Data

新计划要求先用 validation/reference 确定共同 target，再读取最终 test。

因此 Full Data 的“完整训练集”应解释为：

```text
原训练集减去共享 reference/selection holdout
```

否则 Full Data 会在训练中见过用于选 target 的样本，比较失效。

现有 `new_prepare_end_to_end_assets.py` 使用确定性分层抽样产生 shared reference/selection。新资产脚本必须复现相同 seed 和 class 规则，并验证新生成的 reference/selection 与旧 End-to-End cache 逐元素相等。Full Data 训练 mask 只排除 reference/selection；旧 Random-Subset train 部分仍可包含在 Full Data 训练池中。

必须把训练、reference、selection 的原始索引或不可歧义的哈希写入新 manifest。

### 6.2 现有 End-to-End prepare 脚本不能原样复用

`new_prepare_end_to_end_formal.py` 把 DP 样本数写死为：

```python
{"cifar10": 400, "cifar100": 4000, "tiny": 8000}
```

这是旧 compact Random/MTT 比较的口径。

Full Data 必须使用去除 holdout 后的真实 `N` 计算：

```text
sample_rate = batch_size / actual_train_size
```

绝不能用 400/4000/8000 代替 Full Data 的样本数，否则 epsilon 会计错误。

### 6.3 旧 Ours 正是 Adaptive，应优先复用

旧 End-to-End 方法名是：

```text
ours_official_secure_consensus_adaptive
```

新版主图要求：

```text
Full Data -> Adaptive DP
Ours -> Adaptive DP
```

因此已有 Ours 的 3 datasets × 3 seeds Adaptive 曲线、模型、history 和
read-only final-test 是首选复用资产。先审计它们是否与新 Full Data 使用相同
的 encoder policy、Adaptive 配置、epsilon/delta、batch、steps、evaluation
schedule 和计时边界。全部匹配时，不补跑 Ours Stage C。

只有出现明确的不兼容或需要严格同机 total-pipeline timing 时，才做最小化
对齐重跑。不要重跑 Official MTT Stage A。是否重跑 Ours Stage B 只由
“完整流水线计时是否必须同硬件对齐”决定，见第 8 节。

### 6.4 旧 final-test 脚本不能直接运行

`new_final_test_evaluation.py`：

1. 硬编码旧的两个 End-to-End 方法；
2. 要求旧报告恰好 18 runs；
3. 会写入旧 `new_final_test_evaluation.*`；
4. 会覆盖旧 final-test 图表/表格。

必须新建 incremental final-test evaluator，使用新路径和新方法名。新流程
只需要对 9 个 Full Data 新模型开启 final-test 门；Ours 的 test accuracy
优先从已冻结的旧 final-test 报告复用，并记录来源。禁止修改旧 evaluator
来覆盖旧成果。

### 6.5 “相同 epsilon”不等于“相同 noise multiplier”

Full Data 与 Ours 的训练样本数不同，所以 sample rate 不同。

公平要求是：

```text
相同 target epsilon
相同 delta
各自使用真实 N、真实 steps、真实 sample rate 校准 noise
```

noise multiplier 可以不同，因为它是为了实现相同隐私预算而校准出来的。每个 run 必须保存 target epsilon、achieved epsilon、delta、N、batch、steps、sample rate 和 noise multiplier。

### 6.6 主比较应复用已有 Ours 的训练预算，并诚实拆分工作量

用户明确要求复用已有且效果较好的 Ours Adaptive End-to-End，以节省时间。
因此主比较优先冻结到已有 Ours 的 `10/20/30 steps`、batch、evaluation
schedule 和 Adaptive 配置，再让 Full Data 使用完全相同的 Stage-C 配置。

必须明确：

```text
Full Data 的完整数据主要进入 Stage-B Secure-Average 初始化
Stage C 从完整训练池中按真实 N 采样，但 10/20/30 steps 不等于遍历完整数据
Ours 与 Full Data 的 Stage-C steps/batch/optimizer/Adaptive规则必须相同
Stage-B、Stage-C 和 total processed examples 必须分开报告
```

不能声称 Full Data 的 Stage C 在短预算内“看过全部训练样本”。论文中的安全
工作量差异应主要由 Full Data Stage-B 初始化和 Ours compact Stage-B 初始化
支撑，同时单独报告 Stage-C 相同预算下的曲线。

如果未来需要“双方相同 epoch、完整遍历数据”的扩展实验，它将迫使 Ours
重新运行，必须作为独立扩展计划，不能悄悄混入当前节省时间的增量任务。

### 6.7 安全训练时间必须报告两个边界

为了避免不公平，建议同时报告：

1. `stage_c_secure_time`：从冻结 initializer 开始到 Stage C 对应点；
2. `end_to_end_secure_time`：Stage B secure initialization + Stage C。

旧 Official-MTT Stage B 在旧服务器上完成。如果 Full Data Stage B 在当前 AutoDL 上运行，却直接拿旧服务器的 Ours Stage B runtime 做严格总时间比较，就违反“同硬件、同计时边界”。

处理方式：

- 新主图优先使用 Full Data 新跑 Adaptive 与已有 Ours Adaptive 的同口径 Stage-C 时间；
- 完整流水线时间若要做严格数值比较，应在硬件恢复后对 Ours Stage B 做最小化同配置计时重跑；
- 如果不重跑 Ours Stage B，只能把 total pipeline time 标为跨机器参考，不能声称严格 speedup；
- 不需要重新做 Stage A 或重新选择 initializer。

### 6.8 TFE secure training 是 CPU 路线

当前 TFE 0.8/TF 1.15.5 的安全线性训练主要运行在 CPU。RTX 5090 主要用于 PyTorch encoder，但完整特征已缓存。

因此：

- secure-stage peak GPU memory 很可能为 0 或 N/A；
- 应如实区分 encoder GPU memory 与 TFE secure-training GPU memory；
- 不要为了图表好看伪造 GPU 利用率；
- TFE 单机 simulation 不暴露真实 communication bytes/rounds，继续写 `N/A`，不要推算冒充测量。

## 7. 推荐的新文件与隔离路径

不要改旧正式输出。建议新增：

```text
new_configs/new_incremental_comparison/new_incremental_full_feature_config.json
new_configs/new_incremental_comparison/new_tfe_full_feature_stage_b.tsv
new_configs/new_incremental_comparison/new_tfe_full_feature_adaptive_stage_c.tsv

new_prepare_incremental_full_feature_assets.py
new_prepare_incremental_full_feature_stage_b.py
new_prepare_incremental_full_feature_stage_c.py
new_report_incremental_full_feature.py
new_final_test_incremental_full_feature.py

new_log/new_incremental_comparison/
new_expFig/new_incremental_comparison/
new_expTab/new_incremental_comparison/
new_cache/new_incremental_comparison/
new_checkpoint/new_incremental_comparison/
new_init_model/new_incremental_comparison/
new_output/new_incremental_comparison/
```

所有 incremental 文件名使用 `new_` 前缀。不要复用旧 task_id；否则 status 文件会误判旧任务 COMPLETE。

建议方法名固定为：

```text
full_data_secure_average_adaptive
ours_official_secure_consensus_adaptive
```

不要在方法名中写 `end_to_end_dp`，因为两条路线的 initializer 都是 data-dependent non-DP。

## 8. 推荐执行顺序

### Phase 0：只读审计

输出一个新的 audit JSON，至少记录：

```text
Git commit
硬件
现代/legacy Python路径与版本
TFE/TF版本
custom ops loaded状态
磁盘
9份full feature哈希、shape、dtype
encoder/scaler signatures
旧权威报告状态
旧活动队列/训练进程状态
```

审计不读取 test tensor。

### Phase 1：冻结公平协议

创建独立 incremental config，明确：

```text
datasets = cifar10/cifar100/tiny
seeds = 42/43/44
num_clients = 5
feature_dim = 2048
mode = adaptive_dp
target_epsilon = 3.0
delta = 1e-5
batch_size
steps（优先与已有Ours的10/20/30完全一致）
evaluation cadence
optimizer
learning_rate
initial clip C
count noise multiplier
warmup steps
update interval
fixed-point precision = 16
matmul chunk size
checkpoint interval
reference/selection规则
target选择规则
计时边界
processed-example计数边界
privacy interpretation
```

不要直接修改 `new_mainline_config.json` 中旧 End-to-End 配置，因为它是旧结果的配置证据。

### Phase 2：准备 Full Data test-free 资产

必须做到：

1. mmap 读取现有 full `.npy`；
2. 重建 shared reference/selection indices；
3. 验证与旧 shared reference/selection 完全一致；
4. Full Data train 排除这些 holdout；
5. 不读取 test；
6. 保存 manifest 和索引哈希；
7. 尽量不复制 4.6 GiB 特征。

`new_stage_b_secure_init.py` 当前主要接受 NPZ。如果直接生成巨型 NPZ：

- 会额外占用约 4–5 GiB；
- 可能产生大内存峰值；
- 可能使 TFE graph/checkpoint 序列化变慢。

优先评估为 Stage B 增加兼容的 NPY/memmap provider，但不要破坏现有 NPZ 路径。新增接口必须向后兼容，并用小规模真实 TFE correctness 对比 NPZ 与 memmap 的输出误差。

### Phase 3：小规模可行性 pilot

pilot 只能用于：

- 验证数据流；
- 测量吞吐/内存；
- 判断是否需要 streaming；
- 检查 checkpoint/resume；
- 检查 epsilon accountant；
- 冻结合理的正式资源参数。

pilot 不能进入论文正式图表。

先跑最便宜的 CIFAR-10 seed42 小规模任务，再跑能够暴露内存风险的 Tiny seed42。检查：

```text
真实 TFE/ABY3
secure_random_loaded
silent_plaintext_fallback = false
test_loaded = false
无 NaN/Inf
checkpoint 可恢复
进程树 peak RSS
实际线程/进程数
吞吐
磁盘增长
候选模型 shape
TFE 与 NumPy reference 数值误差
```

不要把线程设为 1。若 TensorFlow 创建数百个线程导致抖动，只允许把线程上限调整为实际分配的 CPU 数量附近，并必须先做吞吐对比，把理由和结果写入日志。禁止 CPU affinity 固定到单核。

### Phase 4：Full Data Secure Average initializer，9 runs

基于 `new_prepare_end_to_end_baseline_stage_b.py` 的 NoKD-Secure-Average 逻辑新建 incremental prepare 脚本。

保持：

```text
num_clients = 5
split = iid
distill_epochs = 0
lambda_kd = 0
reference_supervised_weight = 0
formal_candidate_release
```

使用 Full Data train，而不是旧 Random Subset cache。

每项保存：

```text
NoKD secure-average candidate
完整 history
local batches/epoch
processed examples
secure compute time
wall time
peak RSS
checkpoint
feature/label/index hashes
test_loaded=false
```

正式队列应串行或按 pilot 证明安全的有限并发运行。不能因为有 25 vCPU 就启动多个各自创建大量 TFE worker 的任务。

### Phase 5：Full Data Adaptive Stage C，并复用 Ours Adaptive

需要：

```text
Full Data Adaptive Stage C: 3 datasets x 3 seeds（9个新run）
Ours Adaptive Stage C: 优先复用已有3 datasets x 3 seeds
```

Ours 复用已有 selected initializer、曲线、模型和报告。initializer 位于：

```text
new_init_model/new_public_mtt_mainline/stage_b/<dataset>/seed<seed>/iid/
new_secure_stage_b_selected_<dataset>_iid_seed<seed>.npz
```

Full Data 使用 Phase 4 新候选。两种方法的 Adaptive clipping 配置必须一致。

每种方法根据自己的真实 N 和相同的固定 steps 校准 gradient noise
multiplier；双方 target epsilon/delta、count-noise policy、warmup 和 update
interval 相同。Adaptive count mechanism 必须计入 composed accountant。

新 Full Data run 和被复用 Ours run 必须断言：

```text
mode == adaptive_dp
adaptive_count_steps 与配置一致且大于0（若配置产生更新）
count mechanism已计入epsilon
test_loaded == false
accounted_epsilon 与 target 的误差 < 1e-4
initializer_privacy_status == data_dependent_non_dp
released_model_record_dp_valid == false
privacy_interpretation == conditional_gradient_mechanism_only_not_released_model_dp
```

如果审计确认旧 Ours 全部匹配，本阶段总共只新增 9 个 Stage-C run，不新增
Ours run。

### Phase 5B：Standalone Stage C Adaptive 消融的后续调整

旧 54-run standalone Fixed/Adaptive 负结果必须保留。它使用
data-independent initializer，和 selected-initializer End-to-End 回答的不是
同一个问题。

用户允许后续继续调整 Adaptive，但必须：

1. 使用 test-free reference/selection；
2. 在独立 tuning/pilot 路径比较 clip 初值、warmup、update interval、
   count-noise 和训练预算；
3. 不使用 End-to-End final test 反向调参；
4. 不覆盖或重新命名旧 54-run 报告；
5. 只有冻结的新配置确实值得正式验证时，才创建新的 3-seed formal ablation；
6. 该调整不是本轮 Full Data vs Ours 主图的前置阻塞。

### Phase 6：reference-only 聚合与共同 target

在任何新 final test 读取前：

1. 9/9 新 Full Data Adaptive 状态必须 COMPLETE；
2. 9/9 旧 Ours Adaptive 兼容性审计必须通过；
3. 所有 checkpoint/model/history 必须存在；
4. encoder/scaler signatures 必须匹配；
5. epsilon gate 必须通过；
6. 只能读取 shared reference；
7. 从两条 reference 曲线选择双方可达到的同一个 target；
8. target 规则、数值和选择时间写入冻结 JSON；
9. 如果没有可靠共同 target，Time-to-Target 必须报告 N/A，并改报固定预算下精度与accuracy retention，不能降低 target 去美化结果。

Target 不能按 seed 分别选择，也不能给两种方法不同 target。

### Phase 7：新 final-test，只读一次

新建 evaluator，不修改旧 evaluator。

硬门：

```text
9/9新Full Data模型 COMPLETE
9/9既有Ours模型及旧test结果来源验证通过
reference report COMPLETE
target JSON frozen
全部模型哈希固定
全部配置哈希固定
test 尚未被新流程读取
```

对每个 dataset/seed 的 test feature cache 只加载一次，评估新的 Full Data
模型。Ours 优先复用旧 `new_final_test_evaluation.json` 中已经冻结的
read-only test 结果；不要为了组合新图而重复运行整个旧 evaluator。test
accuracy 不能反馈给 target、steps、seed、超参数或图形选择。

保存：

```text
new JSON
new CSV
test read audit
model hashes
mean ± sample std
feedback_to_tuning_or_quality_gates=false
```

### Phase 8：主图和processed-example图

必须输出计划指定的四个文件：

```text
new_figure2_full_feature_vs_ours.png
new_figure2_full_feature_vs_ours.pdf
new_figure2_processed_examples_vs_accuracy.png
new_figure2_processed_examples_vs_accuracy.pdf
```

建议图中只放两条曲线：

```text
Full Data Feature Model
Ours
```

旧 Random-Subset 不进入新主图，但保留在旧图和补充材料。

## 9. 指标必须如何计算

每个 dataset/method/seed 至少保存：

```text
final_reference_accuracy
final_test_accuracy
secure_wall_seconds
stage_c_secure_seconds
end_to_end_secure_seconds（仅同硬件可比时）
step_at_target
time_at_target_seconds
examples_at_target
total_processed_examples
peak_process_tree_rss_bytes
peak_gpu_memory_bytes 或 N/A/0 + scope
target_epsilon
achieved_epsilon
delta
sample_rate
noise_multiplier
train_N
batch_size
steps
mean/std aggregation
```

Processed examples 不能只给一个含糊数字。至少拆成：

```text
stage_b_local_examples_processed
stage_b_consensus_or_kd_examples_processed
stage_c_examples_processed
total_secure_examples_processed
```

Stage C：

```text
examples_at_step = batch_size * step
```

Stage B 应从实际 batches 和实际 batch 长度累计，不能简单用 `epochs*N` 假设最后一个 batch 总是满的。

Speedup 只在双方都达到同一 target 时计算：

```text
time_speedup = full_time_at_target / ours_time_at_target
workload_reduction = 1 - ours_examples_at_target / full_examples_at_target
accuracy_retention = ours_final_accuracy / full_final_accuracy
```

若分母方法未达到 target，保持 N/A，不做外推。

## 10. 隐私声明必须保持准确

Stage A：

```text
epsilon = N/A
Official MTT 为公开 benchmark artifact
```

Stage B：

```text
MPC confidentiality only
epsilon = N/A
```

Standalone Module C 使用 data-independent initializer 时，已有结果可以具有有效 record-level DP。

新 End-to-End 两条路线的 initializer 都依赖训练数据且本身不是 DP，所以：

```text
Stage C epsilon = conditional gradient-mechanism accounting
released_model_record_dp_valid = false
不能声称完整 end-to-end record-level DP
```

不要把“使用 MPC”写成“获得 DP”；MPC confidentiality 与 differential privacy 是不同保护。

## 11. 外部方法表

目标是 3–4 个相关方法，优先：

```text
PEA
CaPC
DDGauss
Ours
```

只使用：

1. 原论文；
2. 官方 supplementary；
3. 官方 repository/released logs。

每个外部数值记录：

```text
method
year
paper title
paper URL/DOI
official repo URL
dataset
model
test accuracy
epsilon
delta
number of parties
threat model
MPC/secure aggregation
end-to-end record DP?
Non-IID?
runtime
hardware
communication
source location（table/figure/page/log）
reported_by_original_paper 或 locally_reproduced
comparability note
```

如果设置不一致，写 N/A 或脚注。不能把不同模型、epsilon、硬件、协议的数字做严格排名。

本轮不复现 CryptGPU。它只用于文字说明直接安全 CNN 的成本高。不要把 CryptGPU 的论文 runtime 除以本地 runtime 来声称 speedup。

外部资料核验可以和 TFE 正式队列并行进行，但不能为了补一个表格单元格投入数天复现。

## 12. 队列、tmux、恢复和监控

长任务 session 建议：

```text
new_incremental_full_feature
```

可以复用现有 queue/recovery 机制的设计，但必须使用新的 queue 和 task_id。

队列启动前：

```text
[ ] queue TSV 可解析
[ ] command 中无 --quick
[ ] command 中无 test 路径
[ ] command 中无 DP-MTT
[ ] command 中无单线程环境注入
[ ] task_id 全部唯一
[ ] checkpoint/output/status 路径全部 incremental
[ ] 预计磁盘增长已计算
[ ] 先生成 plan JSON，默认不 activate
```

每个任务都应有：

```text
heartbeat
status JSON
attempt number
checkpoint/resume
最多有限次数 retry
失败后停止后续队列
process-tree RSS monitor
GPU/CPU/RAM/disk periodic sample
```

正式队列不能因为单个任务失败就悄悄跳过。失败必须进入 `NEEDS_ATTENTION`。

恢复时先看 checkpoint 和 status，不要删除失败日志，不要从头重跑已 COMPLETE 的 task。

## 13. 测试和质量门

代码提交前至少运行：

```text
modern Python py_compile
legacy Python py_compile（所有 TFE 脚本必须兼容 Python 3.6）
bash -n
JSON parse
queue parser
no-test static audit
no-plaintext-fallback audit
no-DP-MTT audit
no-single-thread injection audit
privacy accountant unit tests
incremental path isolation test
```

注意 Python 3.6 不支持部分新语法。传给 legacy 环境执行或导入的代码不能使用：

```text
f"{x=}"
dataclasses（未安装时）
pathlib 新版本专属 API
typing 的新语法
match/case
```

Pilot 通过不代表 formal 完成。正式报告必须严格检查 18/18，而不是“存在若干 JSON 就算完成”。

## 14. Git 与文档规则

任何修改都要：

1. 更新 `progressLog/project_progress.md`；
2. `git diff --check`；
3. 只 add 本阶段文件；
4. commit；
5. push `new-secure-feature-pipeline`；
6. 验证本地 HEAD 与 `origin/new-secure-feature-pipeline` 一致。

建议分阶段提交：

```text
Add incremental full-feature audit and protocol
Add full-data secure-average initializer pipeline
Record full-data Stage B formal results
Add aligned adaptive-DP end-to-end pipeline
Record aligned full-feature vs ours results
Add frozen incremental final-test evaluation
Add external comparison table and final figures
```

GitHub token 存在本机配置文件中。只允许 credential helper 在 push 时读取；不要 `cat`、`echo` 或打印 token。用户的 9999 代理非必要不要使用。国内源优先直连或使用 AutoDL 提供的国内网络；只有官方海外源确实需要时再选择代理。

不要提交：

```text
token/secret
私有 JSONL
原始数据集
Official MTT 大 tensor
full feature .npy
checkpoint
大型日志
临时文件
```

## 15. 时间和资源预估

硬件恢复到 RTX 5090 + 25 vCPU 后：

| 工作 | 预计时间 |
|---|---:|
| 资产/SHA/协议审计 | 0.25–0.5 天 |
| 新脚本、隔离路径、测试 | 0.5–1 天 |
| Full Data TFE pilot | 0.25–0.5 天 |
| 9 个 Full Data Stage B | 12–24 小时墙钟 |
| 9 个 Full Data Adaptive Stage C | 0.25–1 小时 |
| Ours Adaptive | 优先复用，0个新run |
| reference聚合、target冻结、final-test、图表 | 0.5–1 天 |
| 外部方法核验与表格 | 0.5–1 天，可与队列重叠 |

现实总工期：

```text
1.5–3 天
```

如果 Full Data 被嵌入 TFE graph/checkpoint，触发严重内存或序列化瓶颈，需要改成 streaming provider：

```text
5–7 天
```

预计资源：

```text
150–350 vCPU-hours
新增 5–15 GiB checkpoint/history/report
通常新增 GPU 计算接近 0
缓存不兼容需重抽特征时额外 2–6 GPU-hours
```

当前 0.5 CPU core / 无 GPU 状态不应启动。当前状态下不仅会慢，而且无法满足论文的同硬件计时要求。

## 16. 完成标准

只有以下全部满足，才能报告新版增量任务完成：

```text
[ ] 旧成果未删除、未覆盖
[ ] 9份Full Data feature资产验证通过
[ ] holdout与Full Data训练严格隔离
[ ] Full Data Stage B 9/9 COMPLETE
[ ] Full Data Adaptive Stage C 9/9 COMPLETE
[ ] 既有 Ours Adaptive 9/9 兼容性审计通过
[ ] 18条组合曲线的accountant字段全部验证
[ ] 两方法均为同一Adaptive DP口径
[ ] shared reference协议一致
[ ] target在final test前冻结
[ ] 新final test只读门通过
[ ] 两套Figure 2 PNG/PDF完成
[ ] CSV/JSON/完整history/checkpoint完成
[ ] processed examples与计时边界可审计
[ ] mean ± sample std完成
[ ] PEA/CaPC/DDGauss/Ours表格来源可追溯
[ ] CryptGPU只作有边界文字背景
[ ] 隐私声明没有升级为end-to-end DP
[ ] progress log更新
[ ] 所有阶段commit并push
[ ] 工作树干净
```

## 17. 明确禁止

```text
禁止重跑Stage A
禁止重跑已有54个Module C网格
禁止删除Adaptive负结果
禁止复现DP-MTT
禁止本轮复现CryptGPU
禁止使用test调target/epoch/超参数
禁止把shared reference混入Full Data训练
禁止把Full Data的N写成400/4000/8000
禁止无理由重跑已经兼容的Ours Adaptive
禁止用Ours测试结果反向调整Adaptive消融
禁止覆盖旧End-to-End/final-test文件
禁止单seed结论
禁止smoke结果进入论文
禁止伪造GPU memory或communication
禁止将conditional epsilon写成完整end-to-end DP
禁止人工限制为单线程
禁止无界并发多个TFE任务
禁止未进入tmux/恢复队列就启动长任务
禁止输出任何token、密钥或代理认证
```

## 18. 你接手后的第一个可执行阶段

如果硬件仍未恢复：

1. 只做静态审计；
2. 设计 incremental config；
3. 实现资产 validator；
4. 实现新的隔离 prepare/report/evaluator；
5. 编写测试；
6. 更新 progress log；
7. commit/push；
8. 停止，不启动 pilot。

如果硬件已经恢复：

1. 仍先完成上述静态阶段并提交；
2. 再启动 CIFAR-10 seed42 test-free pilot；
3. 检查 Tiny seed42 内存 pilot；
4. 冻结协议；
5. 生成但不自动激活正式队列；
6. 审计队列；
7. 用 tmux/持久恢复机制激活；
8. 每完成主要阶段立即报告、更新进度、commit/push。

遇到口径不明确时，优先保护：

```text
公平性
test隔离
隐私声明
旧结果不可变
可恢复性
可审计性
```

不要用“先跑起来再说”的方式处理正式实验。

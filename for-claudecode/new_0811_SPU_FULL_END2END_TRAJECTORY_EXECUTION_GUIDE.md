# Claude Code 操作指南：完整 SPU End-to-End 轨迹重建（Stage B + Stage C）

更新时间：2026-08-11 UTC

仓库：`/root/autodl-tmp/tfe_experiments`

规范分支：`unified-experiments`

任务状态：**仅完成设计与交接；尚未修改 runner、尚未启动训练。**

## 0. 本任务最终要交付什么

在真实 SecretFlow-SPU 三方 ABY3 后端上，重新运行 seed42 的六个正式 cell，获得
从 Stage B 训练早期一直延续到 Stage C 结束的真实轨迹：

| Dataset | Method | Seed |
|---|---|---:|
| CIFAR-10 | True Ours | 42 |
| CIFAR-10 | Full Data | 42 |
| CIFAR-100 | True Ours | 42 |
| CIFAR-100 | Full Data | 42 |
| Tiny-ImageNet | True Ours | 42 |
| Tiny-ImageNet | Full Data | 42 |

本任务绝不把 Stage B 改成 NumPy/TFE/明文模拟。Stage B 客户端模型、聚合模型和
checkpoint 必须继续保持 SPU secret shares；Stage C 继续使用三方 PYU 私有 OS
CSPRNG noise shares 和原 Opacus RDP accountant。

新增的唯一实验语义是：像已验收 TFE 正式实验一样，在预注册 checkpoint 上通过
SPU 计算 development-validation logits，只公开**聚合开发集 logits**用于画曲线。
禁止公开任一客户端 head，禁止在训练中读取 official test 或 locked holdout。

最终交付至少包括：

1. 新的隔离 runner、CPU/evaluation gate、结果 builder 和绘图脚本；
2. 六个 cell 的 report、effective config、Stage-B/Stage-C trace、完整 log；
3. 六个 cell 的 SHA256 manifest；
4. 新的完整 SPU Figure 2 候选图（PNG/PDF/SVG），不先覆盖当前论文图；
5. 重算后的 T@80/T@90/T@95 和最终 accuracy 表；
6. 一份验收报告，明确 `seed42 / n=1 / development-only / conditional Stage-C DP`；
7. 每次修改同步更新 `progressLog/project_progress.md`，逐阶段 commit 并 push
   `origin/unified-experiments`。

## 1. 接手后先读，不要立即运行

完整阅读：

1. `AGENTS.md`
2. `reportToMe/unified_experiment_report.md`
3. `reportToMe/new_spu_corrected_v2_acceptance_and_paper_figure_inventory.md`
4. `new_configs/new_secretflow_spu/new_spu_corrected_v2_formal_freeze.json`
5. `new_configs/new_secretflow_spu/new_spu_final_config_freeze.json`
6. `new_log/new_secretflow_spu/new_spu_cpu_freeze.json`
7. `new_log/new_secretflow_spu/new_spu_cpu_tuning_report.md`
8. `new_scripts/new_secretflow_spu/corrected_v2/new_spu_corrected_formal.py`
9. `new_scripts/new_true_ours/new_run_true_ours_formal_three_seed.py` 中
   `write_timeline()` 和 Stage-B checkpoint evaluation 的实现；
10. 本文件。

随后只读检查：

```bash
cd /root/autodl-tmp/tfe_experiments
git status --short --branch
git rev-parse HEAD
git rev-parse origin/unified-experiments
git log -5 --oneline
tmux ls || true
ps -eo pid,ppid,etimes,%cpu,%mem,cmd --sort=-%cpu | head -30
nproc
lscpu | rg 'Model name|CPU\(s\)|Thread|Core|Socket|NUMA'
cat /proc/meminfo | head
df -h /root/autodl-tmp
cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null || \
  cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true
```

预期硬件是 32 vCPU、AMD EPYC 7T83。必须读取实际 cpuset，不能仅凭型号假定可用
64 核。正式运行至少需要 32 GiB 内存，推荐 48--64 GiB；现有 corrected-v2 的
真实峰值为 13.4--23.3 GiB。若 cgroup 只有 2 GiB，立即停止，不得尝试启动 SPU。

仓库目前可能存在用户未提交的 Figure 2、Figure 6、Tiny extra-seed 等修改。它们
不属于本任务，不得清理、覆盖、stash 或加入提交。所有 `git add` 都使用精确路径。

## 2. 为什么旧 SPU 曲线必须重跑

现有 corrected-v2 SPU 数据本身通过 report/trace SHA 核验，但 Stage B 没有记录
accuracy。现有 trace 只在 Stage C 每 10 steps 记录一次，所以第一点已经完成整段
Stage B，造成“一开始精度很高、曲线很平”的视觉结果。

现有点数：

- CIFAR-10：每方法 8 个 Stage-C 点；
- CIFAR-100：每方法只有 2 个 Stage-C 点；
- Tiny：每方法 18 个 Stage-C 点。

不能从最终 Stage-B checkpoint 倒推历史 epoch，也不能用插值、平滑或明文模拟点
补齐。因此必须从 epoch 0 重新运行 Stage B，并在运行当时记录 checkpoint 指标。

旧 SPU `9.4x / 2.4x / 3.5x` 是“Stage B 结束后首次观察到 T@90”的粗粒度结果。
完整轨迹完成前，CIFAR-10 的精确 `9.4x` 必须标为 provisional，不得写“加速
900%”。最终应报告 speedup ratio 和 time reduction percentage 两种无歧义口径。

## 3. 冻结训练语义：禁止借本任务重新调参

完整轨迹使用 corrected-v2 已正式运行过的参数，不能改成 TFE 的小 batch。TFE 和
SPU Stage C 参数基本一致，但 corrected-v2 为减少 SPU 图调用，已经为五个 Stage B
cell 冻结了更大的 batch/LR。若原样改回 TFE batch，SPU 工作量预计增加 4--8 倍，
可能把约 14 小时任务扩大到 60 小时以上。

正式六 cell 的 Stage B 参数：

| Cell | batch | chunk batches | LR | epochs |
|---|---:|---:|---:|---:|
| C10 Ours | 8 | 10 | 0.25 | 40 |
| C10 Full | 64 | 10 | 0.8 | 40 |
| C100 Ours | 32 | 25 | 1.0 | 80 |
| C100 Full | 64 | 16（tail 11） | 0.8 | 40 |
| Tiny Ours | 64 | 25 | 0.8 | 80 |
| Tiny Full | 64 | 16（tail 5） | 0.4 | 40 |

其他配置（初始化共享策略、weight decay、client weighting、FedAvg interval、Anchor
CE、Stage C steps/LR/noise/clip/adaptive schedule）必须继续由
`new_spu_final_config_freeze.json` 读取并逐项写入 effective config。禁止事后根据
正式结果改变参数。

## 4. 新命名空间：绝对不要覆盖 corrected-v2

新增隔离命名空间，建议统一命名：

```text
runner schema:  new_spu_full_trajectory_v1
run prefix:     full_trajectory_v1_formal_<dataset>_<method>_seed42
output:         new_output/new_secretflow_spu/full_trajectory_v1/
log/report:     new_log/new_secretflow_spu/full_trajectory_v1/
checkpoint:     new_checkpoints/new_secretflow_spu/full_trajectory_v1/
figures:        new_expFig/paper_experiment/spu_full_trajectory_v1/
tables:         new_expTab/new_secretflow_spu/full_trajectory_v1/
```

`new_checkpoints/` 永远不提交。旧 corrected-v2 report、trace、figure 和 checkpoint
只读保留，不能覆盖或删除。

建议新增而不是直接大改旧 runner：

```text
new_scripts/new_secretflow_spu/full_trajectory_v1/
  new_spu_full_trajectory_formal.py
  new_spu_stage_b_eval_core.py
  new_spu_stage_b_eval_benchmark.py
  new_build_spu_full_trajectory_results.py
  new_plot_spu_full_trajectory.py
  new_audit_spu_full_trajectory.py
```

允许从 corrected-v2 import 已审计的 packer、Stage-B/Stage-C secure functions、
RevealLedger、三方 noise providers、checkpoint 和 accountant；不要复制后再静默改变
这些安全关键实现。

## 5. Stage B 完整轨迹的精确定义

### 5.1 预注册记录点

每个 cell 固定记录：

- epoch 0（训练前初始化模型）；
- epoch 5、10、15……直到最后一个 Stage-B epoch；
- True Ours 在最终 Anchor CE 前、后各记录一次，后者单独标记
  `stage_b_post_anchor`；
- Stage C 延续 corrected-v2 的每 10 steps 记录一次。

禁止看到结果后增加、删除或移动 checkpoint。`eval_interval_epochs=5` 必须写入正式
freeze 和 effective config。

### 5.2 与 TFE 一致的评估释放

TFE 正式代码在 checkpoint epoch 做的是：

```text
秘密客户端 heads
  -> SPU/MPC 内临时加权平均
  -> SPU/MPC 内 development features @ averaged head
  -> 公开 aggregate development logits
  -> driver 使用 development labels 计算 accuracy
```

SPU 新 runner 必须复制该语义：

1. 使用 `make_secure_stage_b_average()` 对当前五个 secret client heads 做临时平均；
2. 不改变训练中的 `client_w_s/client_b_s`，评估必须是 side-effect free；
3. development features 由 Alice PYU 分块送入 SPU，并在 SPU 内计算 logits；
4. 只允许 RevealLedger 公开 aggregate development logits；
5. development labels 只用于 driver 计算 aggregate accuracy；
6. 不公开 averaged weight/bias，不公开任一客户端 head，不公开逐客户端 logits；
7. checkpoint logits 不参与梯度、模型选择、早停或调参。

为了限制 Tiny 的内存，development logits 必须分块计算。建议初始候选
`eval_chunk_rows=250` 和 `500`，只能通过第 7 节的性能/内存 gate 选择，不能依据
accuracy 选择。full chunk 和 tail shape 都在计时前预编译。所有 chunk 拼接后的
行数、类别数、顺序和 SHA 必须核对；禁止漏行、重复行或重排标签。

### 5.3 安全声明必须同步改变

新报告仍可声明：

- Stage-B client-local/global model state remains secret-shared；
- Stage-B client model public reveals = 0；
- 公开内容包含预注册 checkpoint 的 aggregate development logits；
- Stage B 非 DP；
- conditional Stage-C epsilon=3, delta=1e-5；
- initializer data-dependent and non-DP；
- 不声明 released model end-to-end record-level DP；
- official test/locked holdout 未读。

RevealLedger 中每次释放必须包含：epoch、split、rows、classes、chunk count、释放原因和
`used_for_training=false`。

### 5.4 两套时间字段都要保存

每个轨迹点同时记录：

```text
cumulative_training_online_secure_seconds
cumulative_instrumented_wall_seconds
evaluation_online_secure_seconds
evaluation_reveal_seconds
stage
epoch_or_step
development_accuracy
```

`cumulative_training_online_secure_seconds` 只累计真正的 Stage-B/Stage-C 安全训练，
不把为了画图新增的评估开销算作算法训练时间；它是 T@Target 主横轴。
`cumulative_instrumented_wall_seconds` 包含 compile/infeed/checkpoint/evaluation，作为
复现成本补充报告。不得把两者混用。

T@Target 是在预注册离散 checkpoint 上“首次观察到达到目标”的时间，不是两个点间
插值出的精确 crossing。禁止插值。

## 6. 32 vCPU 最大性能规则

“最大性能”不是给三个 SPU party 各开 32 线程。三方共享同一个 32-vCPU cgroup，
过度订阅会更慢。旧实测吞吐：

| OMP threads / party | total party threads | secure steps/s |
|---:|---:|---:|
| 4 | 12 | 5.422 |
| 6 | 18 | 5.235 |
| **8** | **24** | **5.585（最佳）** |
| 10 | 30 | 5.121 |
| 12 | 36 | 2.284 |
| 16 | 48 | 1.406 |

旧 CPU utilization 字段实现有缺陷，不能引用其 `1.0`；但各候选的同机 wall time 和
steps/s 排序仍显示 OMP8 最快。默认正式配置继续使用：

```bash
export RAY_USAGE_STATS_ENABLED=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export JAX_PLATFORMS=cpu
export PYTHONPATH=/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/full_trajectory_v1:/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/corrected_v2:/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu:/root/autodl-tmp/tfe_experiments
```

并保持：

```text
sf.init(num_cpus=30, omp_num_threads=8)
三方 party threads 合计 24
剩余 8 vCPU 给 Ray、driver、infeed、checkpoint 和系统调度
taskset 使用实际检查到的完整 32-vCPU cpuset
```

在新的 evaluation microbenchmark 中只允许比较 OMP8 与 OMP10；如果 OMP10 在
CIFAR-10 和 Tiny exact-shape evaluation 上都稳定快至少 5%，且不增加峰值内存/错误，
才可在任何正式 run 前把 formal CPU freeze 改为 OMP10。否则继续 OMP8。禁止测试
12/16（已证明过度订阅），禁止在正式矩阵中途切换线程配置。

六个正式 cell 必须串行。并跑虽然可能缩短表面墙钟等待，但会争抢 CPU/内存并污染
T@Target，不能用于正式计时。运行时不要同时执行绘图、压缩、大规模 hash 或其他实验。

## 7. 正式运行前的硬门：先 CIFAR-10，再 Tiny exact-shape

### Phase G0：静态和最小安全 smoke

先实现 runner 和 audit，但不启动正式矩阵。检查：

- Stage-B eval function 输入/输出均为 SPUObject，只有 logits 通过 RevealLedger；
- evaluation 不修改 client secret state；
- epoch checkpoint/resume 后不会重复或遗漏轨迹点；
- trace 使用原子写入；
- effective config 包含全部训练、评估、CPU 参数和源码 SHA；
- official test/locked holdout 路径没有被打开；
- 旧 corrected-v2 runner/hash/result 不变。

使用小型 synthetic ABY3 smoke 验证两轮训练、一次 evaluation、secret checkpoint
恢复和相同训练终态。smoke 不能登记为正式结果。

### Phase G1：CIFAR-10 评估开销检查（用户明确要求）

使用 CIFAR-10 True Ours、冻结 batch/LR，运行两个隔离 development gate：

```text
Gate A: Stage B 5 epochs，evaluation disabled，Stage C 不运行
Gate B: Stage B 5 epochs，epoch0+epoch5 evaluation enabled，Stage C 不运行
```

两个 gate 使用相同 seed/data/config/CPU，分别使用唯一 run tag 和 checkpoint；Gate B
不能 resume Gate A。通过 aggregate logits/accuracy 和不透明 secret-state 等价门确认
evaluation 没有改变训练结果。

同时对 evaluation-only exact shape 使用独立新进程比较 OMP8/OMP10 和
`eval_chunk_rows=250/500`。性能选择只能依据 wall、training/eval phase time、cgroup
CPU delta 和 run-local memory peak，不能依据 accuracy。

G1 必须满足：

1. Gate A/B 训练终态对应的开发 logits/accuracy一致（仅允许固定点舍入误差）；
2. Gate B 只有 aggregate logits public reconstruction；
3. client-state public reveals=0；
4. epoch0、epoch5 两点存在且 x 单调；
5. evaluation 不进入 training-online timer；
6. 无 fallback、NaN、Ray ActorDied、OOM；
7. 估算六 cell evaluation 总开销不超过 4 小时。

若不通过，只能优化 evaluation chunk/compile/reuse；禁止先启动正式矩阵。

### Phase G2：Tiny exact-shape evaluation smoke

CIFAR-10 类别数只有 10，不能单独证明 Tiny 的 200 类评估成本可控。因此 G1 PASS
以后，必须用 synthetic 或 development-only Tiny exact shape 做 **evaluation-only**
smoke：`D=2048, K=200`，development rows 与真实 Tiny development pool 一致，只做
一次预热和两次计时，不做训练、不输出论文 accuracy。

G2 必须满足：

- run-local peak memory < 28 GiB；
- 单次 checkpoint evaluation 外推到 Tiny 两方法共 24 次后仍在 3 小时以内；
- 无 ActorDied/OOM/fallback；
- chunk 拼接 shape、顺序、有限值检查全部 PASS。

若峰值 28--32 GiB，先减小 eval chunk；若仍超过 32 GiB 或外推超过 3 小时，停止并
向用户报告，不能浪费约 14 小时后才发现 Tiny 不可行。

### Phase G3：冻结

G0--G2 全部 PASS 后，在任何 formal run 前生成：

```text
new_configs/new_secretflow_spu/new_spu_full_trajectory_v1_freeze.json
new_log/new_secretflow_spu/full_trajectory_v1/new_spu_full_trajectory_preflight.md
new_log/new_secretflow_spu/full_trajectory_v1/new_spu_stage_b_eval_benchmark.csv
new_log/new_secretflow_spu/full_trajectory_v1/new_spu_stage_b_eval_benchmark.json
```

freeze 包含训练参数、eval interval/chunk、CPU/cpuset、runner SHA、依赖源码 SHA、数据
资产 SHA、seed、目标阈值和明确科学边界。提交并 push 后才允许启动 formal。

## 8. 正式运行方式

所有运行必须使用 tmux、`set -o pipefail`、唯一日志，并在启动前记录精确命令。
下面的脚本名/参数是本任务要求 Claude 实现的稳定 CLI；实现后不得临时手改脚本变量。

共同变量：

```bash
cd /root/autodl-tmp/tfe_experiments
PY=/root/autodl-tmp/new_envs/spu_e2e/bin/python
RUN=/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/full_trajectory_v1/new_spu_full_trajectory_formal.py
CPUSET=$(cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null || cat /sys/fs/cgroup/cpuset.cpus.effective)
export RAY_USAGE_STATS_ENABLED=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 JAX_PLATFORMS=cpu
export PYTHONPATH=/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/full_trajectory_v1:/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/corrected_v2:/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu:/root/autodl-tmp/tfe_experiments
```

每条命令都追加：

```text
--seed 42 --unroll 1 --block_steps 10
--stage_b_eval_interval 5 --stage_b_eval_chunk_rows <G2冻结值>
```

运行顺序和参数：

```text
1. --dataset cifar10  --method true_ours --batch_size 8  --batch_chunk_batches 10 --local_lr 0.25
2. --dataset cifar10  --method full_data --batch_size 64 --batch_chunk_batches 10 --local_lr 0.8
3. --dataset cifar100 --method true_ours --batch_size 32 --batch_chunk_batches 25 --local_lr 1.0
4. --dataset cifar100 --method full_data --batch_size 64 --batch_chunk_batches 16 --local_lr 0.8
5. --dataset tiny     --method true_ours --batch_size 64 --batch_chunk_batches 25 --local_lr 0.8
6. --dataset tiny     --method full_data --batch_size 64 --batch_chunk_batches 16 --local_lr 0.4
```

tmux 示例：

```bash
tmux new-session -d -s spu_traj_c10_ours "bash -lc '
set -o pipefail
cd /root/autodl-tmp/tfe_experiments
export RAY_USAGE_STATS_ENABLED=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 JAX_PLATFORMS=cpu
export PYTHONPATH=/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/full_trajectory_v1:/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/corrected_v2:/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu:/root/autodl-tmp/tfe_experiments
taskset -c <实际32-vCPU-cpuset> /root/autodl-tmp/new_envs/spu_e2e/bin/python \
  new_scripts/new_secretflow_spu/full_trajectory_v1/new_spu_full_trajectory_formal.py \
  --dataset cifar10 --method true_ours --seed 42 \
  --batch_size 8 --batch_chunk_batches 10 --local_lr 0.25 \
  --unroll 1 --block_steps 10 --stage_b_eval_interval 5 \
  --stage_b_eval_chunk_rows <G2冻结值> \
  2>&1 | tee new_log/new_secretflow_spu/full_trajectory_v1/new_spu_full_trajectory_cifar10_true_ours_seed42.log
'"
```

不要直接复制 `<...>` 占位符运行。Claude 必须先替换为 G2 freeze 和实际 cpuset。

每 30--60 秒低频监控：

```bash
tmux capture-pane -pt spu_traj_c10_ours:0.0 -S -80 | tail -80
rg 'StageB|StageC|evaluation|Traceback|ERROR|ActorDied|OOM|SYSTEM_ERROR' \
  new_log/new_secretflow_spu/full_trajectory_v1/*.log | tail -40
```

断联后先检查原进程，不得重复启动。同一 cell 中断时只有确认原进程已死，才使用
同参数加 `--resume`。resume 必须从三方 shares 恢复，并保持 Stage-B/Stage-C trace
无重复、无缺口、累计时间单调。禁止删除 checkpoint 从头跑来掩盖恢复错误。

## 9. 每个 cell 的验收门

单条完成后必须运行 audit，至少验证：

```text
status == COMPLETE
run_kind == formal
schema == new_spu_full_trajectory_v1
effective config == pre-formal freeze
seed == 42
backend == SecretFlow-SPU device API
protocol == ABY3 / FM128 / fxp16
Stage-B expected points 全部存在
Stage-C steps 严格为 10,20,...,total_steps
epoch/step/time 严格递增
epoch0 + 每5epoch + pre/post-anchor（若适用）无缺失
Stage-B client-state public reveals == 0
公开 Stage-B 内容仅为预注册 aggregate development logits
evaluation used_for_training == false
三类噪声的三方 provider counters/commitments 正常
conditional Stage-C epsilon <= 3.000001
delta == 1e-5
final report accuracy == trace 最后一行
official_test_read == false
locked_holdout_read == false
所有 accuracy/time/clip/epsilon finite
report、trace、effective config、log SHA256 已登记
```

每个 cell 完成后：

1. 更新 `progressLog/project_progress.md`；
2. 只添加该 cell 的源码/报告/trace/log/effective config，不添加 checkpoint；
3. 精确 commit；
4. 使用 `/root/.config/codex-secrets/githubToken` 的安全 askpass 方式 push；
5. 确认 `HEAD == origin/unified-experiments`；
6. 才启动下一个 cell。

不要在命令、日志、commit message 或文档中输出 token。临时 askpass 用完立即删除。

## 10. 结果合并与画图

六条全部通过后，由新 builder 从新 trace 生成，不允许手工编辑 CSV：

```text
new_expTab/new_secretflow_spu/full_trajectory_v1/new_spu_full_trajectory_main_table.csv
new_expTab/new_secretflow_spu/full_trajectory_v1/new_spu_full_trajectory_time_to_target.csv
new_expTab/new_secretflow_spu/full_trajectory_v1/new_spu_full_trajectory_runtime_breakdown.csv
new_log/new_secretflow_spu/full_trajectory_v1/new_spu_full_trajectory_hash_manifest.json
new_log/new_secretflow_spu/full_trajectory_v1/new_spu_full_trajectory_final_report.md
```

绘图要求：

- 每个点必须来自 trace，禁止插值、平滑、人工抖动或补点；
- `n=1`，不画误差带；
- Stage B 与 Stage C 使用连续真实轨迹，但可用竖线标阶段边界；
- 主横轴为 `cumulative_training_online_secure_seconds`；
- 图注说明 evaluation overhead 单独计量且未进入训练时间；
- True Ours/Full 颜色遵循当前 PEA 风格；
- 不把 SPU 与 TFE 当作硬件归一化速度比较；
- 不覆盖现图，先输出到 `spu_full_trajectory_v1/` 并实际打开 PNG 检查；
- CIFAR-100 不再只有两个点，应能看到 Stage B 的 8/16 个 checkpoint；
- T@90 用新完整轨迹重算，旧 `9.4x` 不自动继承。

若要更新论文，必须先由用户验收候选图和 final report，再改 LaTeX 与当前 Figure 2。

## 11. 时间预算与动态停止线

旧 corrected-v2 六条真实 wall time合计约 13.6 小时，其中 Stage C 合计约 22 分钟；
绝大多数成本是 Stage B。新增 checkpoint evaluation 的 TFE 实测开销很低，但 SPU
必须通过 G1/G2 实测后才能定值。

当前预估：

| 阶段 | 预计时间 |
|---|---:|
| runner/eval/audit 实现与静态检查 | 1.5--3 小时 |
| synthetic smoke + C10 A/B gate + OMP8/10 | 0.5--1.5 小时 |
| Tiny exact-shape evaluation smoke + freeze | 0.5--1 小时 |
| 六条正式 SPU seed42 串行运行 | 14--18 小时 |
| 汇总、hash、绘图、验收、提交 | 1--2 小时 |
| **总计** | **约 18--25 小时** |

如果 G1/G2 显示 evaluation 总开销低于 1 小时，则总时间更可能是 17--21 小时；
如果外推超过 4 小时或 Tiny 单次评估异常慢，则不要启动正式矩阵，先向用户报告新的
实测估时。

硬停止条件：

- 内存不足 32 GiB；
- Tiny exact-shape peak >= 32 GiB；
- 六 cell 外推总时间 > 30 小时；
- evaluation 改变训练终态；
- client model/global weights 被公开；
- official test/locked holdout 被读取；
- epsilon、trace、resume、hash 或 freeze 任一硬门失败。

遇到硬停止条件时保留失败证据、更新 progress、commit/push 诊断，不得用明文结果、
插值点或旧 Stage C 轨迹冒充完成。

## 12. 完成定义

只有同时满足以下条件才可向用户说“任务完成”：

1. G0--G3 全部 PASS，CPU/eval 参数在 formal 前冻结；
2. 六条 seed42 全部从 epoch0 连续运行至 Stage C 终点；
3. 所有 Stage-B/Stage-C 预注册点齐全且来自真实 SPU；
4. 六条 report/effective config/trace/log/hash 全部验收；
5. 新 T@Target 表不再依赖 Stage-C 第一观察点作为 Stage-B 代理；
6. PNG/PDF/SVG 候选图已实际打开检查；
7. 科学边界、n=1、development-only 和 conditional DP 表述准确；
8. progress、最终报告和本 handoff 的完成状态已更新；
9. 所有修改已提交并推送 `unified-experiments`；
10. 无残留实验进程，工作区只剩接手前已有的用户修改。

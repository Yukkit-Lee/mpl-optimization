# Claude Code 实验操作指南：corrected-v2 SPU 正式矩阵与 PEA Figure 2

更新时间：2026-08-07（交接时 shell 显示 11:30 UTC；SecretFlow/Ray 日志时间显示 19:30）  
仓库：`/root/autodl-tmp/tfe_experiments`  
分支：`new-secretflow-spu-end2end`  
交接基线提交：`c99122f4e5bd9645fe3a11b46846ecdb9f0ca8cd`  
状态：**实验正在运行，禁止停止、禁止重复启动。**

## 0. 你接手后第一分钟要做什么

先完整阅读以下文件，不要先改代码：

1. `AGENTS.md`
2. `plan/0807_spu_formal_correction_and_pea_rebuild.md`
3. `plan/0807_new_secretflow_spu_end2end_figure2_codex_prompt_v2.md`
4. `new_log/new_secretflow_spu/corrected_v2/new_spu_correction_preflight.md`
5. `new_log/new_secretflow_spu/corrected_v2/new_spu_corrected_v2_formal_freeze.md`
6. `new_configs/new_secretflow_spu/new_spu_corrected_v2_formal_freeze.json`
7. 本文件。

然后只读检查：

```bash
cd /root/autodl-tmp/tfe_experiments
git status --short --branch
git rev-parse HEAD
git rev-parse origin/new-secretflow-spu-end2end
tmux ls
tmux capture-pane -pt spu_formal_c10_full_v2:0.0 -S -80 | tail -80
tail -30 new_log/new_secretflow_spu/corrected_v2/new_spu_corrected_formal_cifar10_full_data_seed42.log
df -h /root/autodl-tmp
cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || true
```

交接时 `HEAD == origin == c99122f`。当前允许存在的未跟踪运行态文件是：

```text
new_log/new_secretflow_spu/corrected_v2/new_spu_corrected_formal_cifar10_full_data_seed42.log
new_output/new_secretflow_spu/corrected_v2/corrected_v2_formal_cifar10_full_data_seed42/
spu.log
```

不要因为它们未跟踪就清理或覆盖。`new_checkpoints/` 永远不要提交。

## 1. 当前实时实验：不要重启

tmux 会话：

```text
spu_formal_c10_full_v2
```

交接时进度：CIFAR-10 / Full Data，Stage B `9/40` epoch；内存约
`13.4 GB`，磁盘剩余约 `30 GiB`。三方 SPURuntime 都在持续计算。
Ray 的 metrics exporter warning 不是实验失败；只要 SPURuntime CPU 在动、
epoch 继续增长，就不要干预。

当前精确命令为：

```bash
set -o pipefail
export RAY_USAGE_STATS_ENABLED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8 JAX_PLATFORMS=cpu
export PYTHONPATH=/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/corrected_v2:/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu:/root/autodl-tmp/tfe_experiments
taskset -c 96-127 /root/autodl-tmp/new_envs/spu_e2e/bin/python \
  /root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/corrected_v2/new_spu_corrected_formal.py \
  --dataset cifar10 --method full_data --seed 42 \
  --batch_size 64 --batch_chunk_batches 10 --local_lr 0.8 \
  --unroll 1 --block_steps 10
```

原命令通过 `2>&1 | tee` 写入上面的 log。不要启动第二份相同 runner。

推荐每 30--60 秒观察一次，但不要用高频轮询拖慢机器：

```bash
rg '\[StageB corrected\]|\[StageC corrected\]|Traceback|ERROR|SYSTEM_ERROR' \
  new_log/new_secretflow_spu/corrected_v2/new_spu_corrected_formal_cifar10_full_data_seed42.log | tail -20
```

不要擅自终止服务器上用户自己的 Claude Code 后台进程。它会污染
cgroup CPU utilization，所以该字段只可写成诊断值；每个 runner 自己记录的
分段 wall timer 仍是作图的协议时间来源。

## 2. 已经完成且绝对不要重跑的工作

以下工作已经完成、审计、提交和推送：

- 旧 Claude SPU 五条结果及旧图已隔离为 `QUARANTINED_DIAGNOSTIC`；不得用于论文。
- Stage B 五客户端状态全程保持为 SPUObject；公开客户端状态次数为 0。
- Stage B/Stage C checkpoint 是 Alice/Bob/Carol 三方 SPU shares。
- Stage C 三方分别在自己的 PYU 中以 OS CSPRNG 生成噪声 share；driver 不知道
  seed、entropy 或 share。
- Opacus RDP accountant 覆盖 gradient steps 与 adaptive-count steps，目标
  `epsilon=3, delta=1e-5`；只允许表述为 conditional Stage-C DP，Stage B 非 DP。
- 真实安全 smoke、真实数据门、计划中断/恢复门、跨进程累计计时门全部 PASS。
- 修复 MTT local-subset index 被错误套到 compact source 数组的问题。
- 修复 `shared_initialization=false` 被忽略的问题。
- 修复 cgroup v1/v2 CPU/内存统计，并新增 run-local 内存轮询。
- Stage B 改成严格保序 chunk：密文客户端状态跨 chunk 延续，FedAvg 只在原
  epoch 边界发生；full/tail 两种 shape 都在 online 前预编译。
- 完成 15 个预注册明文 semantic-twin 窄调参，不能再扩张搜索。
- 六 cell 参数已经在任何正式 run 之前冻结，禁止事后改参数追逐更好结果。

关键提交按时间顺序包括：

```text
8701ccd  corrected-v2 security gate evidence
9af07c7  first real-data corrected gate
34a8eb9  Stage-B chunking validation
cff7c2c  bounded batch/LR tuning
417d421  six-cell corrected-v2 formal freeze
c99122f  corrected CIFAR-10 True Ours formal result
```

## 3. 已完成的正式单元 1/6

CIFAR-10 / True Ours / seed42 已 COMPLETE 并在 `c99122f` 推送：

```text
final development accuracy       0.6864 (68.64%)
plaintext semantic-twin selection 0.6872 (差 0.08pp)
conditional Stage-C epsilon      3.0
delta                            1e-5
Stage-B client public reveals    0
Stage-B online secure            110.3574 s
Stage-C online secure            25.4962 s
total online secure              135.8536 s
Stage-C trace                    steps 10,20,...,80（8/8 完整）
official test / locked holdout   false / false
```

正式证据：

```text
new_log/new_secretflow_spu/corrected_v2/corrected_v2_formal_cifar10_true_ours_seed42_report.json
new_log/new_secretflow_spu/corrected_v2/new_spu_corrected_formal_cifar10_true_ours_seed42.log
new_output/new_secretflow_spu/corrected_v2/corrected_v2_formal_cifar10_true_ours_seed42/effective_config.json
new_output/new_secretflow_spu/corrected_v2/corrected_v2_formal_cifar10_true_ours_seed42/new_spu_corrected_trace.csv
```

这条结果与 TFE C10 Ours 约 68.86% 接近，但不要把 “SPU 必然比 TFE 精度高”
写成结论。SPU 是执行后端，不会天然提高统计精度；必须如实报告实际结果。

## 4. 当前 C10 Full 完成后的验收、提交和恢复

期望输出：

```text
new_log/new_secretflow_spu/corrected_v2/corrected_v2_formal_cifar10_full_data_seed42_report.json
new_output/new_secretflow_spu/corrected_v2/corrected_v2_formal_cifar10_full_data_seed42/effective_config.json
new_output/new_secretflow_spu/corrected_v2/corrected_v2_formal_cifar10_full_data_seed42/new_spu_corrected_trace.csv
```

报告必须同时满足：

```text
status == COMPLETE
run_kind == formal
effective_config.formal_run == true
effective config 与 freeze 完全一致
conditional_stage_c_epsilon <= 3.000001
stage_b_client_state_public_reveals == 0
official_test_read == false
locked_holdout_read == false
所有 accuracy/time/clip/epsilon 都 finite
trace steps 严格递增，终点等于配置 Stage-C steps
report reveal ledger 中没有 client-local public reconstruction
provider 三方 counters 正确且 commitments 互异
```

如果 tmux 因 VS Code 断联仍存在，继续等待即可。如果进程确实消失且报告不是
COMPLETE，先确认没有残留 Ray/SPU 进程，再用**同一参数加 `--resume`**恢复；
不要删除 checkpoint，不要从 epoch 0 重跑：

```bash
# 只在确认原进程已死后执行
tmux new-session -d -s spu_formal_c10_full_v2_resume "bash -lc '
set -o pipefail
export RAY_USAGE_STATS_ENABLED=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 JAX_PLATFORMS=cpu
export PYTHONPATH=/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/corrected_v2:/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu:/root/autodl-tmp/tfe_experiments
taskset -c 96-127 /root/autodl-tmp/new_envs/spu_e2e/bin/python /root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/corrected_v2/new_spu_corrected_formal.py --dataset cifar10 --method full_data --seed 42 --batch_size 64 --batch_chunk_batches 10 --local_lr 0.8 --unroll 1 --block_steps 10 --resume 2>&1 | tee -a /root/autodl-tmp/tfe_experiments/new_log/new_secretflow_spu/corrected_v2/new_spu_corrected_formal_cifar10_full_data_seed42.log
'"
```

完成后：删除仓库根目录临时 `spu.log`（不要删正式 log），更新
`plan/0807_spu_formal_correction_and_pea_rebuild.md` 的正式计数为 2/6，向
`progressLog/project_progress.md` 追加精度、epsilon、分段计时、峰值内存、
验收结论和下一任务。只添加该 cell 的 report/log/effective config/trace，
不要添加 checkpoint。JSON report 被通用 ignore 规则挡住，必须精确
`git add -f <report>`。提交并推送成功后，才能启动下一 cell。

## 5. 剩余正式矩阵：严格串行，禁止并跑

每次启动前必须：

1. 上一个 cell 已 COMPLETE、验收、更新 progress、commit、push；
2. `git status` 干净，`HEAD == origin`；
3. 没有另一个实验/Ray/SPURuntime；
4. 检查 cpuset、cgroup 内存和磁盘；
5. 使用 tmux，日志记录精确命令；
6. 不把 CPU/BLAS 线程限制为 1。

所有命令共同前缀：

```bash
export RAY_USAGE_STATS_ENABLED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8 JAX_PLATFORMS=cpu
export PYTHONPATH=/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/corrected_v2:/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu:/root/autodl-tmp/tfe_experiments
PY=/root/autodl-tmp/new_envs/spu_e2e/bin/python
RUN=/root/autodl-tmp/tfe_experiments/new_scripts/new_secretflow_spu/corrected_v2/new_spu_corrected_formal.py
```

在当前 C10 Full 之后，按下面顺序逐个运行：

```text
3/6 CIFAR-100 True Ours
--dataset cifar100 --method true_ours --seed 42
--batch_size 32 --batch_chunk_batches 25 --local_lr 1.0
--unroll 1 --block_steps 10

4/6 CIFAR-100 Full Data
--dataset cifar100 --method full_data --seed 42
--batch_size 64 --batch_chunk_batches 16 --local_lr 0.8
--unroll 1 --block_steps 10
# 每客户端 107 batches，tail=11；runner 会预热 full/tail 两种图。

5/6 Tiny-ImageNet True Ours
--dataset tiny --method true_ours --seed 42
--batch_size 64 --batch_chunk_batches 25 --local_lr 0.8
--unroll 1 --block_steps 10

6/6 Tiny-ImageNet Full Data
--dataset tiny --method full_data --seed 42
--batch_size 64 --batch_chunk_batches 16 --local_lr 0.4
--unroll 1 --block_steps 10
# 每客户端 213 batches，tail=5；这是修复旧 OOM/ActorDied 的关键配置。
```

调用形式统一为：

```bash
taskset -c 96-127 "$PY" "$RUN" <上面的对应参数>
```

用各自唯一 tmux 名称和 log 名称，例如：

```text
spu_formal_c100_ours_v2
spu_formal_c100_full_v2
spu_formal_tiny_ours_v2
spu_formal_tiny_full_v2

new_spu_corrected_formal_cifar100_true_ours_seed42.log
new_spu_corrected_formal_cifar100_full_data_seed42.log
new_spu_corrected_formal_tiny_true_ours_seed42.log
new_spu_corrected_formal_tiny_full_data_seed42.log
```

配置文件决定完整 epoch/step 数；不要使用 `--development_gate`、
`--max_stage_b_epochs`、`--max_stage_c_steps` 或 `--planned_stop_after_steps`。
预期 C100 Stage C 为 20 steps，Tiny Stage C 为 180 steps。每个 cell 完成后
都执行第 4 节的全部验收和单独 commit/push。

## 6. Tiny Full 的特殊故障处理

旧 runner 把 68,000 条全量数据按整 epoch 塞进单个 SPU 图，旧 `NEGATIVE_STOP`
只有泛化 Ray ActorDied，不能证明 OOM。corrected-v2 已通过 16 batches/chunk +
5 tail 限制图和输入规模，因此必须先按冻结命令正式尝试，不能沿用旧负面结论。

若 Tiny Full 仍失败：

- 保留完整 log/checkpoint，不覆盖、不伪造 COMPLETE；
- 记录失败前最后 epoch/chunk、退出码/信号、Ray worker stderr；
- 记录 cgroup `memory.usage_in_bytes`、`memory.max_usage_in_bytes`、
  `memory.failcnt`（v1）或 `memory.events`（v2）；
- 记录 driver、三 SPURuntime、PYU worker 的 RSS；
- 记录 effective config、full/tail shape、每客户端 batches；
- 先尝试同参数 `--resume`；
- 不允许 plaintext fallback，不允许把 generic Ray error 直接写成 OOM；
- 不允许改冻结 batch/LR/chunk 追逐成功，除非先写明原因、更新计划、重新冻结并
  在任何替代正式 run 前提交推送。优先诊断代码/恢复，而不是扩张调参。

## 7. 最终结果构建：禁止复用旧 Claude 数据源

旧脚本：

```text
new_scripts/new_secretflow_spu/new_build_spu_results.py
new_scripts/new_secretflow_spu/new_plot_figure2_spu.py
new_scripts/new_secretflow_spu/new_plot_dual_backend.py
```

读取的是被隔离的旧 report/trace，不能直接运行后当作修复完成。新建
`corrected_v2` 专用 builder，必须只按**六个精确 report 路径**读取：

```text
new_log/new_secretflow_spu/corrected_v2/corrected_v2_formal_{dataset}_{method}_seed42_report.json
```

以及每个对应的：

```text
new_output/new_secretflow_spu/corrected_v2/corrected_v2_formal_{dataset}_{method}_seed42/new_spu_corrected_trace.csv
```

builder 必须硬断言 6/6 COMPLETE、run_kind=formal、effective config 等于 freeze、
epsilon/无 local reveal/test/holdout/reveal ledger/provider metadata 全部合格。
输出中登记每个源文件 SHA256，并在 plot data 中登记或伴随 manifest 登记这些
hash；这样图不能静默吃到旧数据。

时间轴定义：

```text
x = Stage-B online secure 累计时间 + 当前 Stage-C online secure 累计时间
```

严禁把 compile、private infeed、checkpoint、development reveal/evaluation、
accounting 或 raw wall 混入 “online secure” 横轴。另做 runtime breakdown 表，
完整呈现这些分段。T@80/T@90/T@95 使用已冻结目标：

```text
CIFAR-10   0.58128 / 0.65394 / 0.69027
CIFAR-100  0.34464 / 0.38772 / 0.40926
Tiny       0.22120 / 0.24885 / 0.262675
```

T@Target 是轨迹中首次达到目标的累计 online secure 秒数；没达到就明确写
`Not Reached`，不能插值制造命中。

## 8. 重新作图：必须复刻原 PEA 图的视觉语法

视觉基准：

```text
new_expFig/new_true_ours/new_figure2_pea_style_secure_time.png
```

原图代码可直接复用其绘图结构：

```text
new_scripts/new_true_ours/new_build_minimal_results_and_figure2.py
```

尤其参考其中 `draw_figure()`，不要沿用旧 Claude SPU 图的布局。SPU 新图要求：

- 1×3 对齐面板，标题严格为 CIFAR-10 / CIFAR-100 / Tiny-ImageNet；
- y 轴百分比，x 轴 seconds 的 log scale；
- Full Data 蓝色 `#2C7BB6`，True Ours 红色 `#D7191C`；
- 两条曲线均标终点；
- 两条同色竖向点线表示各自 Stage B→C transition；
- 横向灰虚线和文字标 T@80/T@90/T@95；
- 每个面板顶部左右两个同色边框 summary box：方法、最终开发精度、
  secure feature count、T@90、conditional epsilon；
- 图例为 `Full Data (n=1)` / `True Ours (n=1)`；
- SPU 只有 seed42，严禁方差阴影、严禁伪造 ±；
- footer 明确：横轴是 cumulative online secure；点线是 Stage B→C；epsilon
  只覆盖 conditional Stage C，不是 end-to-end DP；
- 输出 PNG/PDF/SVG 到 `new_expFig/new_secretflow_spu/corrected_v2/`；
- PNG 分辨率至少达到原图等级，保存后必须实际打开做视觉检查。

还需生成 TFE/SPU 2×3 dual-backend 图，但只有在逐字段审计两个后端 timing
boundary 可比后才可并排声称速度；否则图注必须明确 “backend-specific measured
online boundaries，not a strict hardware-normalized speedup comparison”。TFE 原结果
不可修改，只读加载。

## 9. 科学边界和禁止事项

- 所有精度均为 development validation；official test 和 locked holdout 未读。
- `epsilon=3` 只覆盖 Stage C；Stage B initializer 是 data-dependent non-DP。
- MPC confidentiality 不等于 DP。
- SPU 更现代不代表精度天然高于 TFE；只能报告实测，不能围绕预期结论选结果。
- 不做额外 seeds、不做 SPU ablation、不重跑已完成 TFE/Stage A/旧 Stage B。
- 不删除或改写旧负面结果；只把它们标为隔离证据。
- 不提交 `new_checkpoints/`、Ray 临时状态或私有 token。
- 不输出 `/root/.config/githubToken.txt` 内容。
- GitHub push 使用该 token 时创建临时 `GIT_ASKPASS`，push 后立即删除脚本。
- 每次修改都同步更新 `progressLog/project_progress.md`，然后 commit/push。
- 每个正式 cell 严格串行，不能为了总时长并跑，否则在线时间和内存证据失真。

## 10. 每个 cell 的提交清单

每条完成后只提交：

```text
对应 COMPLETE report（通常需 git add -f 精确路径）
对应完整 log
对应 effective_config.json
对应 new_spu_corrected_trace.csv
plan/0807_spu_formal_correction_and_pea_rebuild.md
progressLog/project_progress.md
```

检查 `git diff --check`，确认没有 checkpoint/stale partial/`spu.log` 被加入。
push 成功并确认本地/远端 HEAD 相同后，才启动下一条。

最终 builder、表、报告、图也要同样更新 progress、commit、push。最终报告必须
列出六条的精度、TFE参考差异、Stage-B/Stage-C online time、raw wall、峰值内存、
epsilon、T@80/90/95、是否达到，以及共享 Claude 进程使 cgroup CPU utilization
不具备论文基准资格这一限制。

## 11. 任务真正完成的判定

只有以下全部成立才能向用户说“完成”：

1. corrected-v2 六条正式 cell 全部 COMPLETE；若 Tiny Full 失败，只能以主计划
   要求的直接测量证据形成受审计 negative stop；
2. 六条 report/effective config/trace/log 均已提交推送；
3. security、DP accountant、reveal、freeze、timing、test/holdout 门全部通过；
4. corrected-v2 专用结果表与 hash manifest 已生成；
5. 新 SPU 1×3 图和 dual-backend 2×3 图按原 PEA 风格生成；
6. 实际打开 PNG 检查，无错位、遮挡、错误颜色、错误 n、假阴影或旧数据；
7. progress、计划状态、最终 `reportToMe` 报告、Claude handoff 均更新；
8. Git 工作区干净，所有修改已在 `new-secretflow-spu-end2end` 推送。

当前 Codex 交接时只完成 1/6，第二条仍在 tmux 运行。不要把旧的
“5 COMPLETE + 1 NEGATIVE_STOP”当作当前状态，那批证据已经被正式否定和隔离。

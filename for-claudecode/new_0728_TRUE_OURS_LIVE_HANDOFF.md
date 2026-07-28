# 0728 True Ours / PEA Figure 2 实时接力文档

## 置顶：2026-07-28 19:20 Asia/Shanghai 人工停止快照

> 用户因 Codex 额度即将耗尽，明确要求停止当前任务并完成交接。
>
> **当前没有 tmux 会话，也没有 Stage B/Stage C/正式 runner 进程。**
>
> 本节优先级高于本文后面 18:08 的旧实时快照。

### A. 当前 Git 与冻结状态

```text
branch: new-true-ours-pea-figure2
freeze-containing HEAD before this handoff commit:
  6973d2528af611c98b60176538e9f6139f7477ab
remote before this handoff commit:
  6973d2528af611c98b60176538e9f6139f7477ab
freeze commit subject:
  Freeze True Ours formal protocol
```

正式配置、target、代码 SHA、split SHA、seeds 42/43/44 和公平计时边界已经冻结并
先于正式实验推送。不得修改冻结配置后继续把结果称为同一批 formal；如确需修改，
必须另开协议版本并重新 freeze。

### B. 正式 18-task 队列的精确停止点

正式任务顺序为：

```text
dataset: cifar10 -> cifar100 -> tiny
method within dataset: true_ours -> full_data
seed within method: 42 -> 43 -> 44
total: 18 end-to-end tasks, each task = Stage B + Adaptive Stage C
```

当前状态：

```text
completed and fully validated: 1/18
completed task: cifar10 / true_ours / seed42
interrupted task: cifar10 / true_ours / seed43 / Stage B
safe checkpoint: local_epochs_completed = 5/40
teacher_initialized = false
distill_epochs_completed = 0
locked holdout read = false
official test read = false
```

注意：seed43 日志已经打印到 epoch 10，但 `Ctrl-C` 正发生在 epoch 10 的
TensorFlow checkpoint 保存过程中。状态 JSON 仍为 `local_epochs_completed=5`，
因此 **5/40 才是最后一个完整可恢复点**。恢复后重算 epoch 6--10 是正确行为，
不得手工把 state 改成 10。

断点路径：

```text
new_checkpoints/new_true_ours/stage_b/true_ours_formal/cifar10/seed43/
  checkpoint
  new_stage_b.data-00000-of-00001
  new_stage_b.index
  new_stage_b.meta
  new_stage_b_new_state.json
```

停止时 state SHA256：

```text
new_stage_b_new_state.json
  3e1729575c982e95159fe48ad6c94d0750be7d63d5895993fa4600ddc57161fb
new_stage_b.index
  08b2d1697d9428de5c1f927de1e66b82abfbacfe88a8a564ca775ebd3e081b81
new_stage_b.data-00000-of-00001
  1f63da19bae6e03363cc15b2b02dec33537b463f6d345052b7d93b3af03571a7
```

队列状态文件：

```text
new_log/new_true_ours/new_true_ours_formal_queue_status.json
```

它仍显示 `status=RUNNING, completed=1`，因为用户要求在第 2 项中途停止，runner
来不及写一个 STOPPED 状态。它是**停止时快照，不代表当前存在运行进程**。恢复
前必须同时检查 `tmux ls` 和 `ps`；当前两者均确认没有残留实验。

### C. 已完成的第 1/18 正式结果

```text
dataset/method/seed: cifar10 / true_ours / 42
Stage B development accuracy at final checkpoint: 0.6880
Stage C final development accuracy: 0.6886
conditional Stage-C epsilon: 2.999999999999981
final adaptive C: 0.915557861328125
Stage B online secure seconds: 104.50239992141724
Stage C online secure seconds: 101.36699604988098
total online secure seconds: 205.86939597129822
```

硬门全部通过：

```text
Official-MTT-only lineage
Full-Data initializer dependency = false
TFE/ABY3 and no plaintext fallback
Poisson sampling + os-CSPRNG
epsilon <= 3.000001
conditional-DP claim only; no end-to-end record-DP claim
finite non-boundary adaptive C
development-only evaluation
locked holdout/test unread
```

权威产物：

```text
new_output/new_true_ours/stage_b/true_ours_formal/cifar10/seed42/
new_output/new_true_ours/stage_c/true_ours_formal/cifar10/seed42/
new_log/new_true_ours/formal_timelines/cifar10/true_ours/
```

不要删除、覆盖或重跑该任务。正式 runner 会检测 COMPLETE report，复核后跳过
训练并把该行重新写回队列状态。

### D. 唯一推荐恢复命令

先检查分支、freeze SHA、硬件、磁盘和无残留进程，再运行 preflight：

```bash
cd /root/autodl-tmp/tfe_experiments
git status --short --branch
git rev-parse HEAD
tmux ls
ps -eo pid,ppid,etime,pcpu,pmem,args \
  | rg 'new_run_true_ours_formal_three_seed|new_stage_[bc]_secure'
/root/miniconda3/bin/python \
  new_scripts/new_true_ours/new_run_true_ours_formal_three_seed.py \
  --preflight_only
```

确认无异常后，用追加日志方式恢复：

```bash
tmux new-session -d -s true_ours_formal_three_seed \
  "env OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32 \
  NUMEXPR_NUM_THREADS=32 /root/miniconda3/bin/python \
  new_scripts/new_true_ours/new_run_true_ours_formal_three_seed.py \
  >> new_log/new_true_ours/txt/new_true_ours_formal_three_seed_runner.log \
  2>&1"
```

恢复逻辑会：

1. 复核并跳过已完成的 CIFAR-10 True Ours seed42；
2. 从 seed43 Stage-B 完整 checkpoint 5/40 恢复；
3. 后续逐项运行，失败即停；
4. 每个完整 B+C 任务后更新 queue status 与 timeline。

不要并行启动第二个 formal runner，也不要单独启动 seed43 命令；否则可能争用同一
checkpoint/output。

### E. 剩余耗时和额度/断联预案

基于同一 TFE 实现的历史 Full-Data 实测，当前冻结配置的剩余墙钟估计为
**14--17 小时**，机器抖动时保守按 20 小时。Full-Data Stage B 是瓶颈，单 seed
粗估：

```text
CIFAR-10:  ~44 min
CIFAR-100: ~55 min
Tiny:      ~176 min
```

tmux、每 5 epochs/steps checkpoint、COMPLETE gate 和独立输出命名空间已经构成
断联预案。`new_checkpoints/` 是本地恢复资产，未纳入 Git，**绝对不能执行
git clean、删除工作树或清理该目录**。

停止时资源：

```text
disk available: ~95 GiB
RAM available: ~291 GiB
container intended CPU parallelism: 32
```

### F. 18/18 之后仍未完成的工作

当前仅完成 Phase 8 的 1/18，以下均尚未执行：

1. 18/18 正式队列完成与统一验证；
2. freeze 后 locked holdout 一次性只读评估；
3. official test 一次性补充评估；
4. 三 seeds mean ± sample std 与 retention；
5. T@80/90/95（未达到必须写 `Not Reached`）；
6. PEA 风格 Figure 2 的 PNG/PDF/SVG；
7. 主结果 CSV/JSON、论文表格、claim audit 与最终 readiness report；
8. 最终 Git 审计和推送。

本次停止前**尚未实现新的 freeze 后自动收尾器**。旧
`new_final_test_evaluation.py` 属于旧 Public-MTT 主线，不能直接拿来评价本轮
True Ours formal。接手者必须先实现并审查本轮专用 evaluator，且设置硬门：

```text
queue status == COMPLETE
completed == total == 18
all 18 Stage-B/Stage-C reports and model SHA pass
freeze/code/split SHA pass
```

只有全部门通过后才允许读取 locked holdout/test。禁止在当前 1/18 状态下打开
holdout 特征、计算 holdout accuracy 或加载 official-test tensor；也禁止根据
freeze 后结果返回调参。

> 快照时间：2026-07-28 18:08 Asia/Shanghai
> 项目：`/root/autodl-tmp/tfe_experiments`  
> 分支：`new-true-ours-pea-figure2`  
> 快照基线提交：`a88eaea`
> 当前唯一有效计划：`plan/0728_new_agent_plan_true_ours_pea_figure2_rebuild.md`

## 0. 接手原则

本文件是额度耗尽、VS Code 断联或 agent 更换时的实时恢复点。旧的
`new_CLAUDECODE_HANDOFF.md` 记录 0727 历史，不能覆盖本文件和 0728 计划。

必须继续执行的用户指令：

1. 完整完成 0728 计划，不因单次负结果停止；
2. 先查现有 COMPLETE/report/checkpoint，绝不重复有效工作；
3. 双方 Stage C 只跑 Adaptive DP，不新增 Fixed DP-SGD；
4. True Ours Stage B 只读 Official MTT，禁止 Full-Data initializer；
5. 调参只读 development validation；freeze 前禁止 locked holdout/test；
6. 正式实验为 Full Data 与 True Ours、3 datasets、seeds 42/43/44；
7. 不控制线程为 1，使用机器最大合理性能；
8. 每个长任务用 tmux、checkpoint、幂等 COMPLETE gate；
9. 每次代码/文档/正式结果修改都更新
   `progressLog/project_progress.md`，commit 并 push；
10. 不输出 GitHub token、代理认证或私有 JSONL 内容。

私有历史 JSONL 只有交接文档无法回答具体历史问题时才允许用 `rg` 最小定位，
禁止整文件读取。

## 1. 当前实时状态

当前没有训练进程或 tmux 会话。刚完成：

```text
Tiny seed42 True Ours Stage B: COMPLETE
development accuracy: 0.3148
runtime/training elapsed: 4449.56 / 3862.08 s

CIFAR-10 secure_average_only equivalence: COMPLETE/PASS
development accuracy: 0.6876 (old full graph also 0.6876)
runtime: 93.89 s vs old 458.55 s
speedup: 4.884x
formal enable flag: true

Regularized Consensus synced + no-sync gates: COMPLETE
passing candidates: 0/3 datasets
decision: no seed expansion; formal mainline uses Secure Average

Paired Adaptive float development screen: COMPLETE
C10 config: steps80, LR0.002, C0=1, gamma0.5, eta0.02
C100 config: steps20, LR0.01, C0=2, gamma0.1, eta0.05
Tiny config: steps180, LR0.002, C0=1, gamma0.5, eta0.1
all six Full/True trajectories: epsilon<=3, finite, no C boundary hit

Adaptive TFE/ABY3 seed42 finalists: COMPLETE 6/6
C10 Full/True dev: 0.7274 / 0.6876
C100 Full/True dev: 0.4300 / 0.4806
Tiny Full/True dev: 0.2733 / 0.3135
all secure-finalist checks: PASS; locked/test unread

Formal config/target/manifest freeze: GENERATED, PREFLIGHT PASS
formal tasks: 18 end-to-end rows (3 datasets x 2 methods x 3 seeds)
next hard gate: push the containing freeze commit before starting tmux
```

关键路径：

```text
checkpoint:
new_checkpoints/new_true_ours/stage_b/true_ours_finalist/tiny/seed42/
  new_stage_b_new_state.json

training log:
new_log/new_true_ours/txt/new_stage_b_tfe_finalist_tiny_seed42.log

runner log:
new_log/new_true_ours/txt/new_stage_b_tfe_finalist_tiny_runner.log

expected report:
new_output/new_true_ours/stage_b/true_ours_finalist/tiny/seed42/
  new_secure_stage_b_report_tiny_balanced_seed42.json

expected NoKD initializer:
new_output/new_true_ours/stage_b/true_ours_finalist/tiny/seed42/
  new_secure_stage_b_candidate_nokd_tiny_balanced_seed42.npz

expected development evaluation:
new_log/new_true_ours/
  new_true_ours_stage_b_development_tiny_seed42.json
```

如果状态与本快照不一致，先检查：

```bash
tmux ls
ps -eo pid,ppid,etimes,%cpu,%mem,rss,stat,cmd \
  | rg 'new_stage_b_secure_init|new_run_stage_b_secure_finalists'
python - <<'PY'
import json
p = ("new_checkpoints/new_true_ours/stage_b/true_ours_finalist/"
     "tiny/seed42/new_stage_b_new_state.json")
d = json.load(open(p))
print(d["local_epochs_completed"], d["elapsed_seconds_completed"])
PY
```

原 `true_ours_tfe_tiny_wait` 和 `true_ours_avgonly_equiv_wait` 已正常退出，不要
重复启动。

输出：

```text
new_output/new_true_ours/stage_b/true_ours_avgonly_equivalence/cifar10/seed42/
new_log/new_true_ours/txt/
  new_stage_b_tfe_avgonly_equivalence_cifar10_seed42.log
new_log/new_true_ours/
  new_true_ours_stage_b_development_cifar10_seed42_avgonly.json
```

## 2. 已完成且不得重跑

### 2.1 基础审计和资产

- Phase 0 lineage pre-run gate；
- Phase 1 三数据集 × 3 seeds 严格互斥 split；
- Phase 2 Official MTT 样本数/类别/签名审计；
- Stage A 和旧 Stage B 全部历史有效结果；
- 旧 54-run Fixed/Adaptive 消融，只保留，不新增 Fixed；
- repaired TFE system 9/9；
- 外部方法 PEA/CaPC/DDGauss/CryptGPU provenance registry。

### 2.2 True Ours seed42 Stage B

```text
CIFAR-10:
secure development acc = 0.6876
plaintext matched config = 0.6952
fair Full-Data development = 0.7266
precision20 result = 0.6876 (no gain; rejected)

CIFAR-100:
secure development acc = 0.4800
plaintext matched config = 0.4804
fair Full-Data development = 0.4308

Tiny:
plaintext matched config = 0.3114
fair Full-Data development = 0.2765
secure result currently running
```

所有 True Ours 结果必须继续保持：

```text
initializer_source = official_mtt_only
full_data_initializer_dependency = 0
locked_holdout_read = false
official_test_read = false
backend = tfe_mpc
protocol = ABY3
silent_plaintext_fallback = false
```

### 2.3 最近 Git 状态

最近已推送提交：

```text
5f10e4d Resolve current True Ours initializer lineage
4036a3a Separate DDGauss original and CIFAR reproduction
3d22421 Separate CaPC original and reproduced results
6e69d42 Gate average-only Stage B optimization
7e433cb Verify CryptGPU source boundaries
cf574a5 Audit external method source provenance
```

训练产生的 `new_checkpoints/` 和未完成日志保持 untracked 是正常的，不要把大型
checkpoint 加入 Git。

## 3. Tiny 完成后的强制核验

先验证两个 JSON 的 `status=COMPLETE`，再检查：

1. report config 与命令一致：epochs80、batch8、LR0.2、wd1e-4、
   uniform、sync1、precision16；
2. `backend=tfe_mpc`、`protocol=ABY3`；
3. `official_mtt_only`、Full dependency=0；
4. 只发布 NoKD candidate，且其 metadata 完整；
5. development evaluator 读取 `development_validation`；
6. locked/test 两个 read flag 都是 false；
7. report/candidate SHA256 相符；
8. 记录 development accuracy、总 runtime 和 secure-training elapsed；
9. 更新 progress/audit，提交 report、候选、evaluation 和完整日志；
10. checkpoint 不提交。

如果进程消失但 report 未 COMPLETE，不要从零重跑。使用同一个命令和同一个
`checkpoint_prefix` 恢复；Stage B 会读取 `_new_state.json`。

## 4. average-only 等价门

该门已完成，不要重跑。审计输出：

```bash
/root/miniconda3/bin/python \
  new_scripts/new_true_ours/new_audit_stage_b_average_only_equivalence.py
```

审计器检查：

- 旧/新 report COMPLETE；
- Exact config、Official MTT lineage、NoKD-only；
- 无 plaintext fallback；
- weight max abs error <= 5e-4；
- bias max abs error <= 1e-4；
- development accuracy 差 <= 0.1pp；
- runtime speedup >= 1.2x；
- locked/test 未读。

输出已为 `enable_secure_average_only_for_formal=true`，正式 Stage B 应使用该路径。

正式 True Ours 输出命名空间必须用：

```text
new_output/new_true_ours/stage_b/true_ours_formal/
```

不要用 `stage_b/formal/`。Stage C lineage guard 要求路径包含
`stage_b/true_ours`，而 `true_ours_formal` 同时满足隔离和硬断言。

## 5. average-only 后的执行顺序

### 5.1 Full-Data metadata 定向升级

已完成，不要重跑。权威结果：

```text
new_log/new_true_ours/new_full_data_lineage_metadata_upgrade.json
```

必须放入独立 tmux，输出日志。它只应修改 18 个 Full-Data metadata：

```text
9 × stage_b
9 × stage_c
```

强制门：

- 27 个 frozen split/Official-MTT 文件升级前后 SHA256 全部不变；
- metadata source=`full_data_disjoint`；
- dependency=1；
- split manifest path/SHA 有效；
- `test_loaded=false`。

结果为 18/18 metadata semantic gate 通过、27/27 preserved SHA 不变。

### 5.2 Regularized Consensus development gate

该门已完成，不要重跑。首轮 synced screen 因 final FedAvg 使5个client heads
完全相同；严格单因素 no-sync follow-up 已恢复非零client-loss spread，但三个
数据集均没有候选通过预设双门：

```text
CIFAR-10 best severe delta: -4.62pp
CIFAR-100 best severe delta: -4.08pp
Tiny best severe delta: -3.37pp
```

权威输出：

```text
new_log/new_true_ours/new_regularized_consensus_development_screen.json
new_log/new_true_ours/new_regularized_consensus_no_sync_followup.json
```

所有 `selected_candidate=null`、`expand_to_three_seeds=false`。不要扩43/44，
不要继续针对development搜Consensus；主路线固定MTT-only Secure Average，
Consensus作为负结果。

### 5.3 Adaptive float development screen

已完成，不要重跑。权威输出：

```text
new_log/new_true_ours/new_adaptive_stage_c_development_screen.json
```

它按单因素顺序筛 steps、LR、initial C、gamma、eta。每个候选同时运行Full Data
与True Ours，两边分别按真实N和composed accountant校准到epsilon=3；共享配置
按最差initializer retention优先、再按平均retention/accuracy和runtime选择，禁止
单边为True Ours调参。只读development，不是正式安全结果。

本轮已修复两个正式性问题：

1. formal lineage 强制 CSPRNG Poisson sampling；TFE 图使用固定 capacity、私有
   mask 和 expected-batch normalization，真实 ABY3 非空/空批次测试已通过；
2. TFE `select(bit,x,y)` 在 bit=1 时选第三项，旧代码误把
   `fraction_below` 方向写反。现已恢复 `select(norm<=C,0,1)` 并由真实安全测试
   锁定。旧 Adaptive 消融不得作为本轮正式结果。

正式 Stage C 命令必须显式包含：

```text
--sampling_mode poisson
--batch_size 64
--expected_batch_size 16
--initializer_privacy_status data_dependent_non_dp
--evaluation_split development_validation
```

### 5.4 TFE/ABY3 Adaptive finalist

已完成，不要重跑。权威聚合审计：

```text
new_log/new_true_ours/new_adaptive_stage_c_secure_finalists.json
```

选出的双方共同 clipping rule 已用真实 TFE/ABY3、seed42、
`--evaluation_split development_validation` 验证：

- epsilon <= 3；
- composed count mechanism 已计入；
- C 轨迹不持续撞 `c_max`；
- 无 NaN/overflow/fixed-point anomaly；
- Full 与 True Ours 使用相同 batch/steps/LR/optimizer/clipping/update schedule；
- 双方仅真实 N 和 Stage-B lineage 不同。

任何数值错误先修正确性，不用 LR 掩盖。

可恢复顺序runner（只在聚合审计非COMPLETE时使用）：

```bash
/root/miniconda3/bin/python \
  new_scripts/new_true_ours/new_run_adaptive_stage_c_secure_finalists.py
```

必须放入 `tmux true_ours_adaptive_tfe_finalists`，日志位于
`new_log/new_true_ours/txt/new_adaptive_stage_c_finalist_*`。共6条：
3 datasets×Full/True，已有COMPLETE report会跳过并重新硬校验。

Full development finalist为避免freeze前后重复Full Stage-B，使用已完成plaintext
tuned权重的metadata wrapper，强制`formal_secure_result=0`；它只验证Stage-C
安全图/会计/C轨迹，不能进入论文或正式计时。freeze后formal Stage C必须读取
双方真实TFE Stage-B initializer。

## 6. Freeze 和正式三 seeds

freeze文件已生成并通过preflight：

```text
new_configs/new_true_ours/new_true_ours_config_freeze.json
new_configs/new_true_ours/new_true_ours_target_freeze.json
new_configs/new_true_ours/new_true_ours_freeze_manifest.md
```

它们已包含source Git commit、精确代码SHA、split hash、双方每数据集config、
真实N、accountant、epsilon/delta、timing boundary、seeds、checkpoint schedule
和80/90/95% targets。不要手工修改freeze；如SHA不匹配必须停止。

只有包含freeze的commit已push才可启动正式18条end-to-end任务。启动命令：

```bash
tmux new-session -d -s true_ours_formal_three_seed \
  "env OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32 \
  NUMEXPR_NUM_THREADS=32 /root/miniconda3/bin/python \
  new_scripts/new_true_ours/new_run_true_ours_formal_three_seed.py \
  > new_log/new_true_ours/txt/new_true_ours_formal_three_seed_runner.log 2>&1"
```

runner对每个dataset/method/seed顺序执行Stage B→Stage C，共18条；每条有独立
checkpoint/log、COMPLETE skip和严格report validator。为避免内存带宽争用，
不要同时启动第二个Full-Data/Tiny TFE graph。

Full Data output 使用：

```text
new_output/new_true_ours/stage_b/full_data_formal/
```

其路径必须包含 `stage_b/full_data`，满足 Stage C lineage guard。

## 7. 冻结后唯一评估

18 条完整且审计通过后：

1. locked holdout 只读一次；
2. official test 作为补充只读一次；
3. 不把结果反馈回超参数、seed 或方法选择；
4. 聚合 mean ± sample std；
5. 生成从 Stage B 起算的统一 cumulative online secure timeline；
6. 生成 T@80/T@90/T@95，未达到写 `Not Reached`；
7. 生成 Figure 2 PNG/PDF/SVG；
8. 生成 master results、paper tables/report、claim audit；
9. 外部方法表只使用已登记 provenance，不制造不可比 speedup。

最终目标：

```text
locked-holdout retention >= 95%
T@90 speedup >= 5x
three-seed stable
epsilon <= 3
no Full-Data initializer dependency for True Ours
```

若精度/速度未达目标，诚实报告负结果和 limitation；freeze 后不得回到 holdout
调参。

## 8. 每个接力点的 Git 操作

先确认 diff 只含本任务文件：

```bash
git status --short
git diff --check
git diff --stat
```

更新 `progressLog/project_progress.md` 后提交。推送使用当前机器已配置的安全
askpass，不读取或打印 token：

```bash
GIT_ASKPASS=/tmp/codex_git_askpass_true_ours.sh \
GIT_TERMINAL_PROMPT=0 \
git push origin new-true-ours-pea-figure2
```

验证：

```bash
git rev-parse HEAD
git rev-parse origin/new-true-ours-pea-figure2
```

## 9. 额度/连接中断恢复清单

接手 agent 先做：

```text
[ ] 阅读本文件、0728 plan、progress 最后一个 0728 block
[ ] git status / branch / HEAD / remote HEAD
[ ] tmux ls
[ ] 查 Stage B/C 真实进程树，防止重复
[ ] 确认三个seed42 Stage-B report、average-only audit均COMPLETE，不重跑
[ ] 确认两个Consensus JSON均COMPLETE且no-sync三数据集selected=null，不扩seed
[ ] 查 locked/test read flags 仍为 false
[ ] 查磁盘剩余空间
[ ] 确认paired Adaptive float JSON COMPLETE，不重跑
[ ] 确认6条TFE/ABY3 finalist与聚合审计COMPLETE，不重跑
[ ] 确认formal freeze三文件存在且runner --preflight_only通过
[ ] 确认包含freeze的commit已push，远端HEAD=本地HEAD
[ ] 当前第一个未满足门是18条formal end-to-end queue
[ ] 查`true_ours_formal_three_seed` tmux和formal queue status，防止重复
```

长任务断线后先查tmux与COMPLETE JSON；正式TFE任务每5 epochs保存checkpoint。
不要因为会话消失就从epoch 0重跑。

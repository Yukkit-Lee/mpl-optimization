# 0729 True Ours 最小可信整改执行计划

状态：`COMPLETE`

创建日期：2026-07-29（UTC）

执行分支：`new-true-ours-pea-figure2`

完成摘要：

- Phase 0–6全部完成；
- 14/14 formal development-only任务通过validator；
- 只补跑了Tiny Full Data seed42，未重跑其他有效任务；
- locked holdout和official test未读取；
- 三面板Figure 2、master JSON、T@Target CSV和简化报告已生成；
- 大型checkpoint本地保留并通过SHA256 manifest验证；
- 全部相关文本日志、reports、timelines、最终小型模型和图表提交Git。

上游交接：

- `for-claudecode/new_0728_TRUE_OURS_LIVE_HANDOFF.md`
- `for-claudecode/new_0729_TRUE_OURS_CLAUDECODE_CORRECTION_GUIDE.md`
- `plan/0728_new_agent_plan_true_ours_pea_figure2_rebuild.md`

## 1. 本轮目标

在不重跑已通过实验、不扩大实验范围、不读取 locked holdout/official
test 的前提下，纠正 Tiny-ImageNet Full Data seed42 的无效正式运行，
重建可信的 14-task development-only 结果集合，重新计算累计 online
secure runtime 与 T@Target，并生成符合原计划语义的三面板 Figure 2。

本轮是“最小可信整改”，不是新一轮调参或完整论文审计。

## 2. 固定任务集合

正式聚合集合固定为：

- CIFAR-10：True Ours / Full Data，各 seed42/43/44；
- CIFAR-100：True Ours / Full Data，各 seed42/43/44；
- Tiny-ImageNet：True Ours / Full Data，各 seed42；
- 合计 14 条 Stage B→Adaptive Stage C end-to-end 任务。

用户已批准 Tiny 只使用 seed42。已有 Tiny True Ours seed43/44 仅保留为
补充诊断，不进入主统计、Figure 2 或双方对比。

## 3. 明确复用与禁止重跑

以下正式任务必须复用，不得重跑：

- CIFAR-10 双方 seed42/43/44；
- CIFAR-100 双方 seed42/43/44；
- Tiny True Ours seed42；
- 已完成 Stage A、development tuning、Consensus、Adaptive finalists。

唯一允许补跑：

- Tiny Full Data seed42 Stage B；
- 由新 Stage B initializer 驱动的 Tiny Full Data seed42 Adaptive Stage C。

禁止：

- Fixed DP-SGD；
- DP-MTT；
- 新增调参；
- 重跑 Tiny True Ours；
- 读取 locked holdout；
- 读取 official test；
- 用来源未验证的 direct evaluation accuracy 指导选择；
- 覆盖或删除旧错误产物。

## 4. 已确认的失效运行

当前旧 Tiny Full Data seed42 不具备正式资格：

- Stage B `local_lr=0.2`，冻结值为0.1；
- Stage B 从不兼容 checkpoint 恢复，形成混合配置；
- Stage C `steps=30`，冻结值为180；
- Stage C noise multiplier=`0.4677076681`，冻结值为
  `0.5685304635176568`；
- Stage C lineage=`legacy_unspecified`；
- Stage C epsilon=`4.406038185765161`，超过3。

旧路径只作为 superseded evidence 保留。纠正运行使用独立
`corrected_v2` 命名空间，不复用旧 checkpoint，也不覆盖旧 report、
model、timeline 或 log。

## 5. Git 与断联恢复政策

- 每个 Phase 完成后更新 `progressLog/project_progress.md`；
- 每个可恢复节点 commit 并 push GitHub；
- 本轮全部相关文本日志、reports、timelines、validator outputs、master
  JSON、CSV、图和文档必须提交 Git；
- 大型 TensorFlow checkpoint 按用户最终指令不上传 Git；
- checkpoint 在本地保留，并记录 path、size、SHA256、COMPLETE state 和
  恢复命令；
- 不使用 `git add .`；
- 不删除已有未跟踪 runtime artifacts；
- 长任务只在独立 tmux 中运行；
- 最大化当前容器可用 CPU 性能，不人为限制为单线程；
- 每次 push 后记录 local/remote HEAD；
- 长任务运行期间按关键 checkpoint 或约30–60分钟间隔向用户汇报。

## 6. Phase 0：现状固化与计划提交

动作：

1. 检查 tmux/process，确认没有重复 runner；
2. 检查 Git branch/status、磁盘、RAM、CPU；
3. 记录旧无效 Tiny Full 产物和 checkpoint；
4. 创建并推送本计划；
5. 在任何正式实验前完成该 Phase。

退出条件：

- 本计划、progress log 已 commit/push；
- 无活动的重复 formal/Tiny runner；
- 磁盘足够容纳新的隔离 checkpoint；
- locked/test 未读取。

## 7. Phase 1：最小协议补充与失效产物清单

新增：

- Tiny seed42-only policy amendment；
- superseded artifact manifest；
- correction namespace manifest；
- 简短 execution patch note。

manifest 至少包含旧文件 path、size、SHA256、失效原因和
`paper_eligible=false`。

退出条件：

- 原 freeze 文件未被回写；
- 原始 `6973d25` freeze lineage 保留；
- 14-task policy 明确；
- 新旧 Tiny Full 路径不会被聚合器混淆。

## 8. Phase 2：14-task 最小 validator 与纠正 runner

实现目标：

1. validator 从 policy 读取期望 seed，不硬编码18；
2. 校验 Stage B/Stage C 精确冻结参数；
3. 校验 TFE/ABY3、lineage、无 plaintext fallback；
4. 校验 Poisson+CSPRNG 和 conditional DP；
5. 校验 `epsilon<=3.000001`；
6. 校验 development-only 和 timeline 单调性；
7. 明确拒绝旧 Tiny Full；
8. 通过 correction manifest 显式选择新 Tiny Full；
9. 只为 Tiny Full seed42 提供隔离 runner；
10. runner 先通过 preflight，再允许启动 tmux。

本 Phase 只做代码、静态验证和轻量 smoke/preflight，不运行正式训练。

退出条件：

- 现有13条可复用任务通过或给出精确缺项；
- 旧 Tiny Full 被 validator 拒绝；
- correction runner 打印冻结参数并通过 preflight；
- 代码、日志和progress已 commit/push。

## 9. Phase 3：Tiny Full seed42 干净纠正运行

### Stage B

- dataset=`tiny`；
- method=`full_data`；
- seed=42；
- split=`balanced`；
- batch size=16；
- local epochs=40；
- local LR=0.1；
- secure average only；
- development checkpoint evaluation；
- checkpoint interval=5 epochs；
- initializer lineage=`full_data_disjoint`；
- backend=TFE；
- protocol=ABY3；
- 无 plaintext fallback。

### Adaptive Stage C

- steps=180；
- learning rate=0.002；
- clip C=1；
- gamma=0.5；
- eta C=0.1；
- noise multiplier=`0.5685304635176568`；
- Poisson sampling；
- OS CSPRNG；
- expected batch=16；
- graph capacity=64；
- initializer lineage=`full_data_disjoint`；
- `epsilon<=3.000001`；
- conditional Stage-C mechanism privacy；
- `end_to_end_record_dp_claimed=false`。

运行规则：

- 新 checkpoint、新 output、新 log、新 timeline；
- 绝不读取旧 Tiny Full checkpoint；
- tmux session 名、PID、日志、checkpoint 写入 progress；
- checkpoint 完成后记录 SHA256；
- 失败只从同一纠正配置的完整 checkpoint 恢复；
- 不因 development accuracy 调整任何参数。

退出条件：

- Stage B report=`COMPLETE` 且 validator通过；
- Stage C report=`COMPLETE` 且 validator通过；
- timeline 覆盖 Stage B→Stage C；
- epsilon、lineage、backend和split flags全部通过；
- checkpoint和日志可用于另一 agent 恢复。

## 10. Phase 4：14-task 汇总与 T@Target

主时间字段唯一使用：

`timeline.total_online_secure_seconds`

边界：

`Stage B secure start → Stage C final model materialization`

T@Target：

- 使用冻结 target；
- 从 timeline `points` 查找首次达到 target 的累计时间；
- 未达到写 `Not Reached`；
- CIFAR-10/100 报每 seed、mean、sample std（ddof=1）；
- Tiny 双方只用 seed42，写 `n=1, std=N/A`；
- 只有双方达到同一 target 时才计算 speedup。

退出条件：

- 14/14 validator通过；
- per-seed timing和T@Target CSV生成；
- 不再使用进程 runtime 或局部 epoch duration；
- 简化 master results JSON 成为本轮图表唯一输入。

## 11. Phase 5：Figure 2 与简化报告

精确输出：

- `new_expFig/new_true_ours/new_figure2_pea_style_secure_time.png`
- `new_expFig/new_true_ours/new_figure2_pea_style_secure_time.pdf`
- `new_expFig/new_true_ours/new_figure2_pea_style_secure_time.svg`
- `new_expTab/new_true_ours/new_figure2_time_to_target_table.csv`

Figure 2：

- CIFAR-10/CIFAR-100/Tiny 三 panel；
- Full Data 与 True Ours Stage B→Stage C 曲线；
- x轴累计 online secure seconds，log scale；
- y轴 development accuracy；
- C10/C100 mean curve + sample-std band；
- Tiny 双方 seed42，明确 `n=1`，不画虚假 std band；
- 标出 Stage B→C transition；
- 标出 T@80/T@90/T@95 或 Not Reached；
- endpoint annotation 包含 accuracy、实际样本数、T@90和 conditional
  Stage-C epsilon；
- PNG/PDF/SVG 从同一 master JSON 和同一绘图入口生成。

报告只保留本轮 development-only 可信结论，并删除：

- 所有数据集3 seeds；
- 严格一次性 Final Test；
- end-to-end DP；
- 来源未验证的 official-test claim；
- 旧 Tiny Full accuracy/epsilon。

退出条件：

- 三种图形格式均可打开；
- 人工检查无标签重叠、log轴错误或图例遮挡；
- 图、CSV、报告数值与 master JSON 一致。

## 12. Phase 6：最终 Git 与交接验收

必须提交：

- 本轮所有相关日志；
- 14-task validator output；
- corrected Tiny Full reports/timeline/final model；
- 简化 master JSON；
- T@Target CSV；
- Figure 2 PNG/PDF/SVG；
- 修正报告；
- progress/handoff；
- checkpoint SHA256 inventory。

大型 checkpoint 保留本地、不进入 Git。

最终命令级门：

- `git diff --check`；
- validator 14/14；
- no locked/test read；
- local HEAD=remote branch HEAD；
- 无活动 tmux；
- checkpoint恢复信息完整。

## 13. 暂不执行

- 不恢复 `direct_evaluation_16models.json` 的 holdout/test provenance；
- 不再次读取 holdout/test；
- 不修复旧最终 accuracy 柱状图；
- 不整理 PEA/CaPC/DDGauss 外部方法表；
- 不制作完整版 claim audit；
- 不开发通用未来 evaluator；
- 不做完整历史 freeze commit 审计；
- 不上传大型 checkpoint；
- 不启动任何范围外实验。

## 14. 时间预算

- Phase 0–2：约1.5–2.5小时；
- Phase 3 Stage B：约4–6小时；
- Phase 3 Stage C：约15–40分钟；
- Phase 4–6：约2–3小时；
- 可在 Stage B 运行时并行准备聚合/绘图代码；
- 预计墙钟时间6–9小时，保守上限约12小时。

若正式 Tiny Full Stage B 出现 TFE 超时，只允许从同配置 checkpoint 恢复，
不得切换LR、epochs、数据或隐私参数。

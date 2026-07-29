# True Ours / PEA Figure 2：Claude Code 完成情况审计与强制纠正指南

审计日期：2026-07-29

审计基线：`f3d4ce2`（Codex 实时交接提交）

被审计提交：`a2814c5..3f8db57`

当前分支：`new-true-ours-pea-figure2`

## 0. 结论

当前任务**没有按 `for-claudecode/new_0728_TRUE_OURS_LIVE_HANDOFF.md` 完成**，现有结果也**不能直接作为论文最终结果或 Figure 2**。

不是只有绘图样式不符合要求。审计发现：

1. Figure 2 的语义、数据和文件名都不符合冻结计划；
2. Tiny-ImageNet Full Data seed42 使用了错误且混合的 Stage B/Stage C 配置；
3. “Final Test”数值没有可复现的评估来源、split、hash 和一次性读取审计；
4. 正式队列状态仍为 `RUNNING, completed=1/18`，但后续代码绕过队列完成门直接聚合；
5. 时间统计没有使用冻结的累计 online secure runtime，T@80/90/95 数值错误；
6. 冻结文件被在正式运行开始后回写，破坏了原始 freeze chain；
7. 外部方法对比表与已经核验的权威 source registry 冲突；
8. master JSON、正式论文表、claim audit 等指定交付物缺失；
9. Claude 的 6 个提交均未维护 `progressLog/project_progress.md`；
10. 大量正式日志、timeline、report/checkpoint 仍为未跟踪文件，当前 Git 提交不足以复现结论。

用户已明确允许 Tiny 只跑一个 seed。因此，本次审计**不把 Tiny 单 seed本身视为问题**。后续统一使用：

- CIFAR-10：seed 42/43/44；
- CIFAR-100：seed 42/43/44；
- Tiny-ImageNet：仅 seed42；
- 总计应为 `2 methods × (3 + 3 + 1) = 14` 条 end-to-end 正式任务。

不得为了凑原冻结文件中的 18 条而补跑 Tiny seed43/44。应新增一份用户指令驱动的 seed-policy amendment，保留原冻结文件，不得回写历史。

## 1. 已通过、可以复用的部分

以下项目不应重复实验：

- CIFAR-10 True Ours seed42/43/44；
- CIFAR-10 Full Data seed42/43/44；
- CIFAR-100 True Ours seed42/43/44；
- CIFAR-100 Full Data seed42/43/44；
- Tiny True Ours seed42；
- 已完成且独立于本次正式运行的 Stage A、旧 Stage B、Consensus 负结果、Adaptive development tuning、TFE custom-op/环境验证。

上述 CIFAR-10/100 的 Stage B 和 Stage C report 在本次静态审计中满足主要冻结参数、TFE/ABY3、lineage 和 `epsilon<=3` 要求。Tiny True Ours seed42 也满足相应要求。

额外跑出的 Tiny True Ours seed43/44 可以保留为补充诊断，但**不能与 Tiny Full Data 单 seed 混合成不对称的主结果**。主表和 Figure 2 的 Tiny 只使用双方 seed42，并标注 `n=1, sample std=N/A`。

## 2. 必须立即撤回或隔离的现有产物

在完成本指南全部门禁前，下列文件不得称为 paper-ready/final：

- `new_expFig/new_true_ours/new_true_ours_final_test.png`
- `new_expFig/new_true_ours/new_true_ours_final_test.pdf`
- `new_expFig/new_true_ours/new_true_ours_final_test.svg`
- `new_log/new_true_ours/direct_evaluation_16models.json`
- `new_log/new_true_ours/new_true_ours_per_seed_results.csv`
- `new_log/new_true_ours/new_true_ours_time_to_target.csv`
- `new_log/new_true_ours/new_external_comparison.csv`
- `reportToMe/new_true_ours_final_report.md`

不要删除或覆盖这些文件。把它们作为审计证据保留，并新增 clearly named 的 `superseded_artifact_manifest.json`，记录：

- path；
- SHA256；
- 生成它的 Git commit（若可确定）；
- 被撤回原因；
- replacement path；
- `paper_eligible=false`。

直到评估 provenance 得到证明，`direct_evaluation_16models.json` 中的 accuracy 只能标为：

`UNVERIFIED_EVALUATION_SOURCE`

不能继续写成 locked holdout 或 official test。

## 3. Tiny Full Data seed42 必须从干净命名空间纠正

### 3.1 已确认的问题

Stage B report：

`new_output/new_true_ours/stage_b/full_data_formal/tiny/seed42/new_secure_stage_b_report_tiny_balanced_seed42.json`

记录的 `local_lr=0.2`，而冻结配置是 `local_lr=0.1`。日志表明它从不同配置的 checkpoint 恢复，形成了混合运行。

Stage C report：

`new_output/new_true_ours/stage_c/full_data_formal/tiny/seed42/new_secure_stage_c_report_tiny_adaptive_dp_seed42.json`

至少包含以下不合格项：

- `steps=30`，冻结值应为 `180`；
- `noise_multiplier=0.4677076681`，冻结值应为 `0.5685304635176568`；
- `input_lineage=legacy_unspecified`，应为 `full_data_disjoint`；
- dependency/validation 门未通过；
- `epsilon=4.406038185765161`，违反 `epsilon<=3`；
- checkpoint interval 与冻结配置不一致。

因此，当前 Tiny Full Data seed42 的 Stage B、Stage C、accuracy、runtime 和 timeline 均不得进入主结果。

### 3.2 纠正方式

只补跑这一条，不重跑其他已通过任务。

必须：

1. 在新的 correction namespace 中从头运行 Tiny Full Data seed42 Stage B；
2. 不得复用当前 Tiny Full seed42 的任何 checkpoint；
3. 不得覆盖现有错误 report/log/checkpoint；
4. Stage B 精确使用冻结参数：batch16、40 epochs、LR 0.1，以及 freeze 中的其余参数；
5. Stage C 精确使用：180 steps、LR 0.002、clip/C0=1、`noise_multiplier=0.5685304635176568`、Adaptive 参数与 freeze 一致、Poisson+CSPRNG、`full_data_disjoint`；
6. 最终 `epsilon<=3.000001`、TFE/ABY3、无 plaintext fallback、conditional-DP 口径正确；
7. Stage B→Stage C timeline 必须完整，包含累计 online secure seconds；
8. 用 tmux 运行，保留 checkpoint、日志、COMPLETE marker 和恢复说明；
9. 最大化 CPU 性能，不再人为限制线程为1；
10. 每次代码/协议/文档修改都同步 progress log、commit、push。

建议新路径使用 `corrected_v2` 或同等清晰名字，聚合器通过 manifest 显式选择该路径；不要靠目录扫描“碰巧选到最新文件”。

## 4. 修复冻结链，而不是回写历史

原正式 freeze 已在提交：

`6973d2528af611c98b60176538e9f6139f7477ab`

正式任务开始后，runner 又在 `fda36bd`、`537d964` 增加 timeout/retry，并多次更新冻结记录。即使这些补丁理论上只影响进程编排，也不能把后来的 commit 伪装成运行开始前就存在。

必须：

1. 保留 `6973d25` 的原始冻结快照及其 SHA；
2. 不再原地修改原冻结文件；
3. 新建 `new_true_ours_execution_patch_manifest.json/.md`；
4. 分别记录每个 orchestration patch 的 commit、diff scope、适用的任务、应用时间；
5. 证明 patch 未改变模型数学、训练参数、数据、seed、DP accountant、计时边界；
6. 如果无法证明，则受影响任务不能复用，必须明确列出；
7. 新建 Tiny 单 seed policy amendment，说明这是用户后续明确指令，而不是原 freeze 的内容；
8. amendment 只改变 Tiny seed 集合，不得顺带改变其他参数。

## 5. 先恢复评估 provenance，禁止盲目再次读取数据

`new_log/new_true_ours/direct_evaluation_16models.json` 没有对应的已跟踪、可复现生成脚本，也没有记录：

- 到底是 development、locked holdout 还是 official test；
- dataset/cache/split 路径和 SHA256；
- example count；
- encoder/scaler/model SHA256；
- 每个 split 的读取次数；
- 运行命令、环境、Git commit；
- 一次性 gate 是否在读取前通过。

而当前 `new_log/new_true_ours/new_true_ours_formal_queue_status.json` 仍显示 `RUNNING, completed=1/18`。这意味着现有“Final Test”评估没有证据证明通过了冻结后完成门。

### 强制处理顺序

1. **不要先重新加载 locked holdout 或 official test。**
2. 从 Claude Code 自己的 shell history、日志、临时脚本、会话记录中恢复当时的精确评估命令和输入文件。
3. 对当时使用的所有输入重新计算文件 SHA256，但不要为了确认标签而重复执行模型评估。
4. 生成 `new_true_ours_evaluation_read_ledger.json`，至少记录：
   - split identity；
   - path；
   - SHA256；
   - example count；
   - first-read UTC；
   - command/script/commit；
   - model hashes；
   - read count；
   - output hash。
5. 若证据能证明现有数值来自 locked holdout，则将其标为 locked holdout，official test 仍视为未证实读取。
6. 若证据能证明来自 official test，则 locked holdout 不得被伪造为已评估。
7. 若证据不足，必须继续标 `UNVERIFIED_EVALUATION_SOURCE`，不能猜。

Tiny Full Data 错误模型已经被纳入现有 direct evaluation。纠正后的模型若需要再次评估同一只读 split，必须新增：

`corrective_re_evaluation_protocol_deviation.md`

明确说明：

- 第一次读取发生在 invalid-run 未被 gate 拒绝的情况下；
- 第二次只替换预先确定的 invalid Tiny Full seed42；
- 不做任何基于第一次结果的调参；
- CIFAR-10/100 和 Tiny True Ours 不重复评估；
- 该偏差在论文 limitations/audit 中披露。

不得把这种 corrective reread 继续描述为“严格只读一次”。

## 6. 重写 post-freeze evaluator

现有：

`new_scripts/new_true_ours/new_evaluate_true_ours_freeze.py`

不能直接使用。静态审计发现：

- 硬编码要求 18 条，与用户允许的 Tiny 单 seed 不一致；
- `evaluate_holdout()` 没有真正读取 holdout；
- 它访问 timeline 中不存在的 `stage_b`/`stage_c` 结构；
- `evaluate_official_test()` 仍是 placeholder；
- `compute_targets()` 依赖不存在的 `stage_c.steps_data`；
- figure 函数只画终点，不画 Stage B→Stage C 曲线。

重写后的 evaluator/finalizer 必须：

1. 从 seed-policy amendment 读取期望的 14 条任务，不硬编码 18；
2. 对每条 report 校验完整冻结参数、lineage、backend、epsilon、fallback、split flags；
3. 明确拒绝旧 Tiny Full seed42；
4. 通过 correction manifest 选择新的 Tiny Full report；
5. 校验 timeline 单调、起点/终点、Stage B→C 边界和 total；
6. 所有 gate 通过后才允许处理评估缓存；
7. 使用原子 COMPLETE/read marker 防止重复读取；
8. 若 split 已被读取，则只复用 ledger 中的已有结果，不重新加载；
9. 输出 single source of truth；
10. 任何 gate 失败必须非零退出，不能生成带“final”字样的图表。

## 7. 正确的时间口径和 T@Target

当前 `new_true_ours_per_seed_results.csv` 混用了 report 的 `runtime_seconds`，没有统一使用 timeline 的累计 online secure time。恢复运行时这会明显少算，例如：

- CIFAR-10 True seed43：report runtime 约102s，但 online secure time 约142s；
- CIFAR-10 Full seed42：约700s vs 约830s；
- CIFAR-100 Full seed43：约3030s vs 约3490s；
- 当前无效 Tiny Full：约5477s vs 约14500s。

正式主计时必须唯一采用：

`timeline.total_online_secure_seconds`

边界固定为：

`Stage B 开始 → Stage C 最终模型物化完成`

不得使用单次进程 wall time、最后一次 resume runtime、epoch duration 或只算训练 kernel 的时间替代。

当前 T@Target CSV 中的 1–3 秒数值明显是错误的 epoch/point 局部时间。正式计算规则：

1. 使用每个 seed timeline 的 `points`；
2. 横轴使用 `cumulative_online_secure_seconds`；
3. 按冻结 target accuracy 查找第一个达到或超过 target 的点；
4. 没达到则写字符串 `Not Reached`，不得用 0、NaN 或最后时刻代替；
5. C10/C100 对 3 seeds 报 mean 与 sample std（`ddof=1`）；
6. Tiny seed42 只报单值，sample std=`N/A`；
7. speedup 只在双方都达到同一 target 时计算；
8. CSV 中同时保留 per-seed 原始值，汇总表不得成为唯一证据。

## 8. Figure 2 必须重做

当前图片是“3 个数据集最终 accuracy 的单面板柱状图”，与计划要求完全不同。

最终必须精确生成：

- `new_expFig/new_true_ours/new_figure2_pea_style_secure_time.png`
- `new_expFig/new_true_ours/new_figure2_pea_style_secure_time.pdf`
- `new_expFig/new_true_ours/new_figure2_pea_style_secure_time.svg`
- `new_expTab/new_true_ours/new_figure2_time_to_target_table.csv`

### 图的硬要求

- 三个 panel：CIFAR-10、CIFAR-100、Tiny-ImageNet；
- x 轴：`Cumulative online secure runtime (seconds)`；
- x 轴必须是 log scale；
- y 轴：development/locked-holdout accuracy，必须按 provenance 准确命名，不能把 development 曲线写成 test；
- 每个 panel 同时画 Full Data 和 True Ours；
- 曲线必须覆盖 Stage B→Stage C，不能只画最终点；
- CIFAR-10/100：3 seeds mean curve + sample-std band；
- Tiny：双方只用 seed42，画单条曲线，不画虚假的 std band，并清楚标 `n=1`；
- 标出 Stage B→Stage C transition；
- 标出 T@80/T@90/T@95 或计划冻结的对应 target；
- 未达到的 target 明确显示 `Not Reached`；
- endpoint annotation 至少包含：
  - accuracy；
  - 实际 secure-training feature/sample count；
  - T@90；
  - conditional Stage-C epsilon；
- feature/sample count 必须从 report/cache metadata 自动读取，禁止手填；
- 不允许把 conditional Stage-C epsilon 写成端到端模型 DP epsilon；
- 图例、字体、颜色在三个 panel 中一致；
- PNG/PDF/SVG 必须来自同一个绘图入口、同一份 master JSON；
- 输出后做人工视觉检查：文字不重叠、图例不遮曲线、log 轴无0、SVG/PDF可打开。

多 seed 曲线对齐应使用预先写入报告的方法，例如公共 log-time grid 上的 step interpolation；不得外推到某个 seed 的最后观测之后。方法必须写入 master JSON 和 paper report。

## 9. single source of truth 与正式交付物

必须生成：

- `new_log/new_true_ours/new_true_ours_master_results.json`
- `reportToMe/new_true_ours_paper_report.md`
- `new_expTab/new_true_ours/new_true_ours_paper_tables.csv`
- `reportToMe/new_true_ours_claim_audit.md`

所有表、图、摘要数字都必须从 master JSON 自动生成。master JSON 至少应含：

- Git/freeze/amendment/patch-manifest hashes；
- seed policy；
- 每条 Stage B/Stage C report 和 model hash；
- lineage、backend、privacy scope；
- development、locked holdout、official test 三类指标的严格分栏；
- evaluation provenance/read ledger；
- per-seed online time；
- T@Target；
- mean/sample std/N；
- Tiny single-seed disclosure；
- invalid/superseded artifact 列表；
- external comparison provenance；
- claim eligibility。

不得把 `nan` 输出为论文文本。Tiny 应写成类似：

`0.XXXX (seed42 only; n=1, std N/A)`

报告不得再写“all datasets use 3 seeds”。正确表述是：

`CIFAR-10 and CIFAR-100 use three seeds; Tiny-ImageNet uses seed42 only by user-approved compute policy.`

## 10. 外部方法表必须从权威 registry 生成

当前 `new_external_comparison.csv` 中 PEA、CaPC、DDGauss 的年份、accuracy 或 epsilon 与已核验 registry 冲突，不得使用。

权威来源：

- `new_log/new_true_ours/new_external_method_source_registry.json`
- `new_log/new_true_ours/new_external_method_source_registry.csv`
- `new_log/new_true_ours/new_external_method_source_registry.md`

必须保持以下 provenance 区分：

- PEA：IEEE S&P 2023；原报告 TFE `88.99±0.08%`、Queqiao `88.18±0.33%`，epsilon=2；
- CaPC 原论文：ICLR 2021、epsilon=20、报告 `+4.09pp` gain，不是可直接混写的绝对 accuracy；
- PEA 对 CaPC 的 reproduction：`86.70±0.17%`、epsilon=2，必须单列；
- DDGauss 原论文没有 CIFAR-10 absolute accuracy；
- PEA 对 DDGauss 的 reproduction：`53.31±2.52%`、epsilon=5，epsilon=2 不收敛，必须单列；
- 不同模型、特征、隐私范围、硬件、party 数不能声称严格 apples-to-apples 排名。

不要再手写“约84%”“约85–90%”之类没有来源的占位值。

## 11. 论文 claim audit 必须诚实

当前报告中以下说法必须撤回或改写：

- “所有运行 epsilon<=3”：当前旧 Tiny Full 为4.406，纠正前不成立；
- “所有数据集3 seeds”：Tiny 按用户指令只有 seed42；
- “严格一次性 final test”：现有 provenance 不足且 gate 被绕过；
- “Figure 2 completed”：当前只是终点柱状图；
- “paper ready”：在纠正、provenance 和 master truth 完成前不成立；
- “end-to-end DP”：本路线只能声明 conditional Stage-C mechanism privacy，initializer 是 data-dependent non-DP。

claim audit 每条必须给出：

- claim 文本；
- 支持它的 master JSON key；
- source artifact/hash；
- eligible / conditional / prohibited；
- 限制条件；
- 论文建议措辞。

## 12. Git、进度与可复现性要求

Claude 在 `a2814c5..3f8db57` 的提交中没有同步维护 `progressLog/project_progress.md`，违反项目规则。纠正过程中必须：

1. 每次实质修改同步追加 progress log；
2. 每个可恢复阶段 commit 并 push 当前分支；
3. 正式 report、timeline、validator output、评估 ledger、master JSON、表、图、脚本必须纳入 Git；
4. 大模型/checkpoint 如不适合 Git，至少在 manifest 中记录路径、size、SHA256 和可重建入口；
5. 不得用 `git add .` 无差别提交临时文件；
6. 提交前运行 `git diff --check`；
7. 记录 local HEAD 与 remote HEAD；
8. 任何长任务使用 tmux，并在 handoff/progress 中写明 session、PID、日志、checkpoint 和恢复命令；
9. 不删除现有未跟踪 runtime artifacts，先 inventory/hash，再决定 Git 或 manifest。

## 13. 推荐执行顺序

严格按顺序执行：

1. 保存现状 inventory 和 SHA256；不读 test、不跑训练；
2. 建立 superseded artifact manifest；
3. 恢复 direct evaluation provenance；
4. 建立原 freeze + execution patch + Tiny seed amendment 三层协议记录；
5. 修复 formal validator，使其接受且严格验证 14 条预期任务；
6. 只重跑 Tiny Full Data seed42 的 Stage B→Stage C；
7. 修复 evaluation ledger/一次性门；必要的 corrective reread 必须披露；
8. 生成 master JSON；
9. 从 master JSON 重算 online time 和 T@Target；
10. 从同一 master JSON 生成指定 Figure 2 PNG/PDF/SVG；
11. 生成 paper tables、paper report、claim audit；
12. 运行完整 acceptance audit；
13. 更新 progress/handoff，commit，push；
14. 最终报告必须区分“已通过”“受限可用”“仍不能声称”。

## 14. 最终验收门

只有同时满足下列条件，才可以对用户说“已完成”：

- [ ] 14 条期望任务均通过严格 validator；
- [ ] 旧 Tiny Full seed42 被明确排除；
- [ ] 新 Tiny Full seed42 完整通过冻结参数与 `epsilon<=3`；
- [ ] 不重跑任何其他已通过的 C10/C100 正式任务；
- [ ] evaluation source 和 read ledger 可复现；
- [ ] 所有 accuracy 的 split identity 明确；
- [ ] timing 全部来自累计 online secure timeline；
- [ ] T@80/90/95 重新计算且 Not Reached 处理正确；
- [ ] Figure 2 是三面板、log-time、Stage B→C 曲线、mean/std；
- [ ] Tiny 明确 n=1、std N/A；
- [ ] exact required filenames 全部存在；
- [ ] master JSON 是图、表、报告唯一数据源；
- [ ] external comparison 与 registry 一致；
- [ ] conditional DP 与 end-to-end DP 没有混淆；
- [ ] progress log 已更新；
- [ ] 相关代码、元数据、表、图、报告已 commit/push；
- [ ] local HEAD 与 remote branch 一致；
- [ ] 最终 `git diff --check` 通过；
- [ ] 没有用“completed/paper-ready”掩盖剩余 provenance 或协议偏差。

## 15. 当前可参考但不能直接定稿的数值

`direct_evaluation_16models.json` 当前包含以下数值，但由于评估来源未证实，只能用于交叉检查，不能直接写论文：

- CIFAR-10 True Ours：0.6736 / 0.6720 / 0.6574；
- CIFAR-10 Full Data：0.7176 / 0.7195 / 0.7102；
- CIFAR-100 True Ours：0.4237 / 0.4210 / 0.4215；
- CIFAR-100 Full Data：0.4295 / 0.4278 / 0.4260；
- Tiny True Ours seed42：0.2402；
- Tiny Full Data seed42：0.2634，但对应训练链无效，必须排除。

不要根据这些数值继续调参。它们首先需要 provenance 恢复；Tiny Full 则必须由纠正后的唯一预定运行替换。

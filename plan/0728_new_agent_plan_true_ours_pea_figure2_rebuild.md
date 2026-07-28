# Agent Plan：重建 True Ours、修复评估协议并生成 PEA 风格 Figure 2

## 0. 总目标

本轮不是继续叠加零散实验，而是修复当前主结论中的方法混入、数据泄漏和计时口径问题，重新得到一个可以用于论文主文的 **True Ours End-to-End** 结果。

最终目标：

1. **True Ours 不得加载任何 Full Data Stage B initializer**；
2. 从 Stage B 开始使用严格互斥的数据划分重新训练；
3. 在严格验证协议下，通过持续诊断和调参，使 True Ours 尽可能接近 Full Data 的最终泛化精度；
4. 在相同计时边界下证明 True Ours 显著减少安全计算时间；
5. 生成类似 PEA Figure 2 的“准确率—安全时间”主图；
6. Adaptive DP 只在严格对齐、充分调参后才能声明效用优势；
7. 修复所有报告、CSV、系统表和 Git 冻结问题。

本轮禁止继续使用以下错误结果作为论文主结果：

```text
Full-Data Stage B initializer
+
MTT Stage C fine-tuning
```

该路线只能保留为：

```text
Full-Data-init + MTT fine-tuning hybrid ablation
```

不得继续命名为 Improved Ours。

---

# 1. 本轮方法定义

## 1.1 Full Data Feature Model

```text
Full Training Features
→ Full-Data Secure Average Initialization
→ TFE/ABY3 Adaptive DP-SGD
→ Final Model
```

## 1.2 True Ours：主效率路线

```text
Official MTT Compact Features
→ MTT-Only Secure Average Initialization
→ TFE/ABY3 Adaptive DP-SGD
→ Final Model
```

要求：

- Stage B 只能读取 Official MTT/紧凑客户端特征；
- 不得加载 `new_output/new_incremental_comparison/stage_b/...` 中的 Full Data initializer；
- 不得通过路径复用、软链接、默认参数或 fallback 间接加载 Full Data 模型；
- 模型报告必须记录 initializer lineage；
- lineage 必须能证明 initializer 仅由 MTT 紧凑数据生成。

## 1.3 True Ours-Robust：Non-IID 鲁棒路线

```text
Official MTT Compact Features
→ Regularized Secure Consensus / KD Alignment
→ TFE/ABY3 Adaptive DP-SGD
→ Final Model
```

该路线只在严格独立 validation 上证明有效后，才能作为主方法的一部分。

如果 Regularized Consensus 无法在不损伤 IID 泛化的情况下改善 Severe Non-IID，则：

- 主 End-to-End 使用 MTT-Only Secure Average；
- Consensus/KD 保留为独立 Non-IID 模块或负结果；
- 不得把 Average 重新命名为 Consensus。

---

# 2. 文件与分支管理

创建新分支：

```text
new-true-ours-pea-figure2
```

所有新产出进入：

```text
new_log/new_true_ours/
new_expTab/new_true_ours/
new_expFig/new_true_ours/
new_checkpoints/new_true_ours/
new_cache/new_true_ours/
new_output/new_true_ours/
new_scripts/new_true_ours/
```

禁止覆盖：

- 旧主线实验；
- Full Data 增量实验；
- Accuracy diagnosis 探索实验；
- End-to-End follow-up 探索实验。

启动前执行：

```bash
git status
git rev-parse HEAD
git diff --stat
```

将当前未提交内容分类：

1. 正式代码；
2. 探索代码；
3. 日志；
4. 临时文件。

完成清理后提交一次：

```text
Freeze pre-rebuild audit state
```

---

# 3. Phase 0：只读审计与禁止 Full Data 混入

生成：

```text
new_true_ours_lineage_audit.json
new_true_ours_lineage_audit.md
```

必须扫描：

- 所有 initializer 路径；
- checkpoint provenance；
- feature cache 路径；
- manifest；
- 默认参数；
- fallback；
- 软链接；
- 环境变量；
- report 中的 source path。

对每个 dataset/seed 输出：

| Dataset | Seed | Stage B Input | Initializer Path | Full Data Dependency | Valid True Ours |
|---|---:|---|---|---|---|

正式 True Ours 必须满足：

```text
Full Data Dependency = False
Valid True Ours = True
```

在正式运行前增加硬断言：

```python
assert initializer_source == "official_mtt_only"
assert full_data_initializer_path not in resolved_dependencies
```

若断言失败，立即停止任务。

---

# 4. Phase 1：重新建立严格互斥的数据划分

当前旧 validation 参与过 Full Data Stage B 训练，不能继续作为正式独立验证集。

为每个 dataset/seed 创建新的互斥划分：

```text
stage_b_train
stage_c_train
reference_internal
selection_internal
development_validation
locked_holdout
official_test
```

要求：

- `development_validation` 不得进入 Stage B/Stage C 训练；
- `locked_holdout` 不得用于任何调参、早停或配置选择；
- `reference_internal` 与 `selection_internal` 互斥；
- `official_test` 保留为历史可比结果，但必须披露此前已读取；
- 正式主结论优先使用新的 `locked_holdout`；
- official test 作为冻结后补充评估，不得再影响参数选择。

建议划分原则：

```text
Development validation: 原训练集的 10%
Locked holdout: 原训练集的 10%
Remaining train pool: 80%
```

要求按类别分层抽样，三个 seeds 分别生成固定 split manifest。

生成：

```text
new_true_ours_split_manifest.json
new_true_ours_split_audit.md
```

必须验证：

- 所有 split 样本 ID 两两无交集；
- feature cache 不包含其他 split；
- Full Data 和 True Ours 使用完全相同的 development/holdout/test；
- 所有 split manifest 在调参前提交 Git。

提交：

```text
Freeze disjoint evaluation splits
```

---

# 5. Phase 2：修复 Module A 样本数与资产口径

正式使用真实样本数：

```text
CIFAR-10 Official MTT: 480 samples, 0.96%
CIFAR-100 Official MTT: 4800 samples, 9.6%
Tiny ImageNet Official MTT: 9600 samples, 9.6%
```

检查：

- 实际 feature cache 样本数；
- 每类样本数；
- 类别均衡；
- encoder/scaler signature；
- 是否误用了 400/4000/8000；
- 是否存在缺类或重复样本；
- MTT 特征是否基于当前正确共享 encoder 重新生成。

输出：

```text
new_mtt_asset_audit.json
new_mtt_class_distribution.csv
```

若样本数或特征缓存不一致，先修复资产，再进入 Stage B。

---

# 6. Phase 3：True Ours Stage B 从零构造

## 6.1 MTT-Only Secure Average Baseline

从 Official MTT 紧凑特征重新训练本地客户端模型，并进行 Secure Average。

不得复用 Full Data 本地模型。

需要记录：

- 每个客户端样本数；
- 每类样本数；
- local epochs；
- local batch size；
- local learning rate；
- local validation；
- secure aggregation time；
- total Stage B time；
- initializer hash；
- initializer lineage。

## 6.2 客户端 MTT 数据分配诊断

比较：

```text
Class-balanced partition
Dirichlet Mild Non-IID
Dirichlet Severe Non-IID
```

IID 主 Figure 2 优先采用 class-balanced compact split。

Non-IID 实验单独报告。

## 6.3 Stage B 调参空间

先用 development validation 和 seed 42 进行筛选。

候选：

```text
local epochs: 5 / 10 / 20 / 40
local batch: 8 / 16 / 32
local LR: 0.25× / 0.5× / 1× / 2× current
weight decay: 0 / 1e-5 / 1e-4 / 1e-3
client weighting: uniform / sample-count / class-balanced
```

诊断指标：

- local train loss；
- local development validation accuracy；
- aggregated initializer validation accuracy；
- inter-client parameter distance；
- class-wise accuracy；
- calibration；
- Stage B secure time。

每次只改变一个因素，禁止一次同时大改全部参数。

## 6.4 True Ours Stage B 成功标准

主效率路线入围标准：

```text
MTT-Only Secure Average validation accuracy
≥ Full Data Secure Average validation accuracy - 1.0pp
```

且：

- 三个 seeds 中至少两个满足；
- 不使用 Full Data initializer；
- Stage B 时间显著低于 Full Data；
- locked holdout 尚未读取。

若未达到，继续执行 Phase 4 诊断。

---

# 7. Phase 4：True Ours 精度差距诊断与持续优化

按以下顺序定位瓶颈。

## 7.1 数据载荷损失

对比：

```text
Full Features + Average + NoDP
MTT Features + MTT-Only Average + NoDP
```

若差距明显，检查：

- MTT feature cache；
- 类别覆盖；
- scaler；
- feature normalization；
- local training budget；
- class weighting；
- synthetic sample imbalance。

## 7.2 初始化损失

对比：

```text
MTT centralized plaintext classifier
MTT client-local average
MTT secure average
```

若 centralized 正常、secure average 下降，检查：

- aggregation scaling；
- fixed-point truncation；
- client weighting；
- parameter serialization；
- bias aggregation；
- optimizer state错误复用。

## 7.3 Stage C 损失

对比：

```text
MTT-Only Average + NoDP
MTT-Only Average + Adaptive DP
```

若 DP 后下降明显，只进入 Adaptive clipping、accountant 与训练预算诊断；本轮不新增 Fixed DP-SGD 实验。

## 7.4 数值正确性

逐级对照：

```text
PyTorch float
Plaintext fixed-point emulation
TFE secure no-DP
TFE secure Adaptive DP
```

比较：

- logits；
- loss；
- gradient；
- clipped gradient；
- noise statistics；
- parameter update；
- validation prediction。

若误差超阈值，优先修复：

- scale；
- truncation；
- overflow；
- secure division；
- batch averaging；
- duplicated noise variance；
- bias update。

不得用学习率掩盖数值错误。

---

# 8. Phase 5：Regularized Consensus 挽救

目标：

> 在不牺牲 IID 泛化的前提下，改善 Severe Non-IID。

候选方法：

```text
Average
Original Consensus
Shrinkage Consensus
Temperature-Smoothed Consensus
Weight-Clipped Consensus
Top-k Consensus
Class-Balanced Consensus
```

## 8.1 Shrinkage

```text
W_init = (1 - β) · W_average + β · W_consensus
β = 0.05 / 0.10 / 0.25 / 0.50
```

## 8.2 Temperature

```text
T = 2 / 4 / 8
```

## 8.3 Weight cap

```text
max client weight = 0.30 / 0.40 / 0.50
```

## 8.4 成功标准

Regularized Consensus 只有满足以下条件才可进入主方法：

1. IID development validation 不低于 Average 0.5pp以上；
2. Severe Non-IID validation 至少提高 1pp；
3. 三 seeds 中至少两个方向一致；
4. locked holdout 未参与选择。

否则：

- End-to-End 主路线使用 MTT-Only Secure Average；
- Consensus/KD 作为 Non-IID 独立模块或负结果；
- 不强行纳入 Figure 2。

---

# 9. Phase 6：Adaptive DP 对齐、调参与真实效用验证

当前已有 follow-up 仅为探索性，不可作为正式 Adaptive 贡献。

## 9.1 正式对比

比较：

```text
Full Data Feature Model + Tuned Adaptive DP-SGD
True Ours MTT-Only + Tuned Adaptive DP-SGD
```

两者保持一致：

- data split；
- batch；
- steps；
- LR；
- optimizer；
- ε upper bound；
- δ；
- backend；
- fixed-point precision；
- seeds；
- checkpoint schedule。

两者只允许 Stage-B 输入规模与 initializer lineage 按方法定义不同；Adaptive clipping 规则、隐私目标和评估日程必须一致。

## 9.2 Adaptive clipping 调参

```text
initial C = 0.50 / 1.00 / 2.00
target clipped fraction = 0.10 / 0.30 / 0.50
EMA/update rate = slow / current / fast
```

所有筛选只读取 development validation；每次只改变一个因素。旧 54-run Fixed/Adaptive 负结果保留为历史消融，不删除、不覆盖，也不触发新的 Fixed DP-SGD 运行。

为避免单边调参，每个候选配置必须同时在 Full Data 与 True Ours 上运行；
两边分别按真实样本数校准 accountant。共享配置按“两边中较差的
NoDP-initializer accuracy retention”优先选择，再依次比较平均 retention、
平均 development accuracy 与较低 runtime。不得先只为 True Ours 选参后再套用
到 Full Data。

冻结前的真实 TFE/ABY3 Stage-C correctness finalist 可为 Full Data 使用已完成
plaintext tuned Stage-B 权重的 metadata-complete development wrapper，但必须明确
标记 `formal_secure_result=0`，不得计入论文结果或安全 Stage-B 时间。冻结后的
formal Stage C 必须改为读取双方真实 TFE Stage-B initializer；不得复用该 wrapper。

## 9.3 训练预算

```text
CIFAR-10: 10 / 20 / 40 / 80
CIFAR-100: 20 / 40 / 80 / 120
Tiny: 30 / 60 / 120 / 180
```

每个设置重新计算 accountant，保持：

```text
ε ≤ 3
δ = 1e-5
```

## 9.4 重点场景

先在以下场景筛选：

```text
CIFAR-10 IID
CIFAR-100 IID
CIFAR-100 Severe Non-IID
```

Severe Non-IID 仅用于诊断 Adaptive clipping 轨迹和稳定性；不得为了得到正结果增加 Fixed 对照或读取 locked holdout/test。

## 9.5 成功标准

只有同时满足以下条件，才可把 Adaptive 路线用于正式主图：

1. Full Data 与 True Ours 使用相同 Adaptive clipping 规则；
2. composed accountant 对两种方法分别按真实样本数校准，且 `ε ≤ 3, δ = 1e-5`；
3. 相对各自 NoDP initializer/trajectory 的效用损失可解释，且无数值错误；
4. 三 seeds 结果稳定，不依赖单个异常 seed；
5. locked holdout/test 在冻结前未读取；
6. 主结论只比较 Full Data 与 True Ours 的准确率—安全时间权衡，不声称 Adaptive 优于 Fixed。

若 Adaptive 仍造成明显精度损失，正式结论必须如实降级为：

```text
Adaptive clipping satisfies the matched privacy protocol, but its utility loss remains a limitation.
```

不得为了得到正结果反复读取 locked holdout 或 official test。

---

# 10. Phase 7：冻结主实验配置

调参完成后，生成：

```text
new_true_ours_config_freeze.json
new_true_ours_target_freeze.json
new_true_ours_freeze_manifest.md
```

文件必须包含：

- dataset；
- split manifest hash；
- method；
- initializer lineage；
- batch；
- LR；
- steps；
- optimizer；
- clipping；
- ε/δ；
- checkpoint schedule；
- target accuracy；
- timing boundary；
- seeds；
- Git commit。

冻结文件必须先提交 Git，再运行正式实验。

提交：

```text
Freeze True Ours formal protocol
```

---

# 11. Phase 8：正式三 Seeds 重跑

正式方法：

```text
Full Data Feature Model
True Ours
```

可选第三条仅用于补充：

```text
True Ours-Robust
```

Seeds：

```text
42 / 43 / 44
```

正式运行要求：

- 从 Stage B 开始重跑；
- 使用新互斥 split；
- 全程记录时间点；
- 不读取 locked holdout；
- 不读取 official test；
- 生成逐 checkpoint validation accuracy；
- 保存 Stage B 和 Stage C 的累计时间。

完成后：

1. 只读一次 locked holdout；
2. 生成正式主结果；
3. official test 仅作为补充历史可比结果；
4. 不根据 holdout/test 返回调参。

---

# 12. Phase 9：统一公平计时

## 12.1 主计时边界

定义：

```text
Online Secure Runtime
=
从 Stage B 第一份秘密共享/安全训练输入开始
到 Stage C 最终模型输出完成
```

Full Data 和 True Ours 必须完全相同。

必须包含：

- Stage B local secure workload；
- secure aggregation；
- initializer output；
- Stage C secure DP training；
- final model materialization。

不得出现：

```text
Full Data total = Stage B + Stage C
Ours total = Stage C only
```

## 12.2 单独报告的时间

另表报告：

```text
Public/Offline preprocessing
Feature extraction
Cache construction
Online secure Stage B
Online secure Stage C
Total online secure runtime
```

主效率结论使用 `Total online secure runtime`。

## 12.3 计时仪器

每个 checkpoint 保存：

```text
wall_clock_since_stage_b_start
stage
step
validation_accuracy
epsilon_so_far
processed_secure_examples
```

输出：

```text
new_true_ours_timeline_seedXX.json
```

---

# 13. Phase 10：PEA 风格 Figure 2

## 13.1 图的核心逻辑

Figure 2 不再画单纯最终accuracy柱状图。

采用 PEA 风格的训练过程图：

```text
Accuracy vs Training Progress
```

但为了突出本文效率，横轴使用：

```text
Cumulative Online Secure Runtime (seconds, log scale)
```

纵轴使用：

```text
Development Validation Accuracy
```

最终 locked holdout/test 只作为终点注释，不参与曲线选择。

## 13.2 主布局

三个子图：

```text
(a) CIFAR-10
(b) CIFAR-100
(c) Tiny ImageNet
```

每个子图至少两条曲线：

```text
Full-Feature + Adaptive DP
True Ours-MTT + Adaptive DP
```

若正式 NoDP 三seed结果完成，可加入浅色虚线：

```text
Full-Feature NoDP
True Ours-MTT NoDP
```

## 13.3 曲线构造

曲线必须覆盖：

```text
Stage B training progress
→ initializer output
→ Stage C secure DP steps
```

不能只从 Stage C 开始。

每个时间点：

- 三 seeds 取均值；
- 阴影为 ±1 std；
- 横轴为累计在线安全时间；
- 使用 log scale；
- 不使用不同计时起点；
- 不用缺失 Stage B 的旧数据填充。

## 13.4 多目标 Time-to-Target

不要只选一个可能有利的 target。

根据 development validation，在正式运行前冻结：

```text
80% / 90% / 95% of Full Data development plateau
```

报告：

| Dataset | Method | T@80% | T@90% | T@95% |
|---|---|---:|---:|---:|

若未达到，写：

```text
Not Reached
```

## 13.5 子图注释

每个子图右下角显示：

```text
Locked-holdout accuracy
MTT vs Full feature count
T@90% speedup
ε ≤ 3
```

示例格式：

```text
Holdout Acc: 68.1% vs 68.5%
Secure Samples: 480 vs 49,900
T@90%: 12.4× faster
ε ≤ 3
```

实际数字必须来自正式冻结实验。

## 13.6 Figure 2 输出

```text
new_figure2_pea_style_secure_time.png
new_figure2_pea_style_secure_time.pdf
new_figure2_pea_style_secure_time.svg
new_figure2_time_to_target_table.csv
```

---

# 14. Phase 11：系统表修复

修复 TFE system 汇总脚本，恢复真实结果：

- CIFAR-10 runtime 约17.7–18.5s；
- CIFAR-100约24.5–27.2s；
- Tiny约39.6–41.6s；
- Peak RSS约1.6–6.8GiB；
- communication bytes/rounds继续写N/A。

增加聚合单元测试：

```text
assert runtime > 0
assert peak_rss > 0
assert source_rows == 9
```

输出：

```text
new_tfe_system_formal_repaired.csv
new_tfe_system_formal_repaired.json
```

---

# 15. Phase 12：统一所有报告

建立单一事实源：

```text
new_true_ours_master_results.json
```

所有表格和报告只能从该文件生成。

修复：

- Full Data 三seed均值；
- retention版本不一致；
- Stage C表头错位；
- Module A样本数；
- TFE全零；
- Git commit；
- 方法名称；
- privacy scope；
- initializer lineage；
- timing boundary。

生成：

```text
new_true_ours_paper_report.md
new_true_ours_paper_tables.csv
new_true_ours_claim_audit.md
```

---

# 16. Phase 13：外部方法对比

主实验稳定后，补3–4个外部方法。

优先：

```text
PEA
CaPC
DDGauss
Ours
```

优先使用：

1. 原论文；
2. supplementary；
3. 官方仓库；
4. 官方日志。

只有没有可用数据、代码可运行且设置可对齐时才本地复现。

必须记录：

- exact source；
- dataset；
- model；
- accuracy；
- ε/δ；
- privacy scope；
- MPC setting；
- runtime；
- hardware；
- reported/reproduced。

CryptGPU只做文字背景对比，不做本地复现。

---

# 17. 每日迭代循环

Agent每天按以下循环工作：

```text
Audit
→ Run one-factor diagnostic
→ Analyze development validation
→ Keep/reject configuration
→ Update experiment registry
→ Commit valid code/results
→ Continue next bottleneck
```

每轮必须回答：

1. 本轮改了什么；
2. 为什么改；
3. 哪个指标改善；
4. 是否只改善reference；
5. 是否影响ε；
6. 是否引入Full Data依赖；
7. 是否值得进入三seed；
8. 下一轮只改哪个因素。

---

# 18. 停止条件

True Ours 主实验理想目标：

```text
Locked-holdout accuracy retention ≥ 95%
T@90% speedup ≥ 5×
Three-seed result stable
No Full Data initializer dependency
ε ≤ 3
```

最低可接受目标：

```text
CIFAR-10/CIFAR-100 retention ≥ 90%
Tiny retention ≥ 80%
T@90% speedup ≥ 3×
```

若多轮调参后仍无法达到：

- 停止盲目调参；
- 输出瓶颈归因；
- 诚实报告效率—效用权衡；
- 不篡改图轴或删除不利seed。

Adaptive DP 理想目标：

```text
双方 matched Adaptive protocol
ε ≤ 3，accountant 完整
三 seeds 稳定且效用损失可接受
```

本轮不新增 Fixed DP-SGD；旧 Fixed/Adaptive 结果仅保留为历史消融。

---

# 19. 禁止事项

禁止：

- 继续使用 Full Data initializer 伪装成 MTT-only Ours；
- validation样本进入训练；
- 用 official test 调参；
- 用 locked holdout 调参；
- 不同方法使用不同计时边界；
- Ours缺失Stage B却报告total time；
- target freeze失败后继续生成time-to-target；
- 根据结果挑seed；
- 将Average命名为Consensus；
- 将探索性plaintext结果当正式TFE DP结果；
- 使用旧 Fixed 结果或新增 Fixed run 声称 Adaptive 更优；
- 为了曲线好看截断关键区间或隐藏失败结果；
- 未提交Git就宣称正式冻结完成。

---

# 20. 最终交付

必须输出：

```text
new_true_ours_lineage_audit.json
new_true_ours_split_manifest.json
new_mtt_asset_audit.json
new_true_ours_stage_b_screen.csv
new_true_ours_stage_b_formal.csv
new_consensus_rescue_formal.csv
new_adaptive_aligned_formal.csv
new_true_ours_config_freeze.json
new_true_ours_target_freeze.json
new_true_ours_master_results.json
new_figure2_pea_style_secure_time.png/pdf/svg
new_figure2_time_to_target_table.csv
new_tfe_system_formal_repaired.csv
new_true_ours_paper_report.md
new_true_ours_claim_audit.md
```

最终报告必须回答：

1. True Ours是否完全没有Full Data initializer依赖；
2. 数据划分是否严格互斥；
3. True Ours最终holdout/test精度；
4. 与Full Data的accuracy retention；
5. Full Data与True Ours完整Stage B+Stage C时间；
6. T@80%/90%/95%；
7. Figure 2是否来自同口径正式三seed；
8. Consensus是否真正改善Non-IID；
9. Adaptive是否在双方对齐、ε≤3的协议下稳定且效用损失可接受；
10. 哪些结论可进入论文主文；
11. 哪些只能作为补充或负结果；
12. Git commit、branch与复现命令。

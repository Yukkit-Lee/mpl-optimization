# Corrected-v2 SPU 验收与论文图表清单

更新时间：2026-08-08（UTC）  
验收方式：无卡、2 GiB cgroup；只读取 Git、源码、JSON/CSV/Markdown 和已有图片，
未启动训练、TFE、SPU、模型评价或测试套件。  
验收源分支：`new-secretflow-spu-end2end`，源 HEAD `39214a2`。

## 1. 总结结论

Claude Code 交付的六条 corrected-v2 SPU 正式运行**确实全部存在且为 COMPLETE**。
独立回读六份原始 report、六份 trace、六份 effective config、冻结 JSON 和 SHA256
manifest 后，以下核心数字成立：

| Dataset | Method | Dev accuracy | Stage-B online | Stage-C online | Total online | T@90 | Peak memory | ε |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | True Ours | 68.64% | 110.36s | 25.50s | 135.85s | 113.6s | 13.41GiB | 3.0 |
| CIFAR-10 | Full Data | 72.54% | 1060.62s | 25.75s | 1086.37s | 1063.8s | 14.86GiB | 3.0 |
| CIFAR-100 | True Ours | 47.66% | 2565.84s | 8.27s | 2574.11s | 2570.0s | 14.35GiB | 3.0 |
| CIFAR-100 | Full Data | 42.52% | 6167.61s | 9.03s | 6176.64s | 6172.2s | 17.46GiB | 3.0 |
| Tiny-ImageNet | True Ours | 30.84% | 6061.03s | 223.63s | 6284.66s | 6067.6s | 22.83GiB | 3.0 |
| Tiny-ImageNet | Full Data | 27.68% | 21148.71s | 223.98s | 21372.69s | 21154.7s | 23.26GiB | 3.0 |

六条均为 seed42、development-validation；official test 与 locked holdout 均未读。
`ε=3, δ=1e-5` 只覆盖条件式 Stage C，不能写成端到端 record-level DP。

True Ours 相对 Full 在同一 SPU 后端首次观察到 T@90 的时间分别缩短约
`9.4x / 2.4x / 3.5x`。这是**同后端、同数据集、seed42 的轨迹比较**，不是
跨硬件归一化加速比。CIFAR-10 最终精度仍比 Full 低 3.90pp；CIFAR-100 和
Tiny 分别高 5.14pp 和 3.16pp，不能写成所有数据集都更准。

## 2. 独立通过的验收项

- 六份 report 均为 `status=COMPLETE`、`run_kind=formal`。
- 六份 effective config 的 dataset/method/seed、batch、LR、chunk、shared-init、
  epoch、Stage-C steps、unroll 和 block size 均逐项匹配 pre-formal freeze。
- 当前 runner SHA256 与 freeze 登记值一致：
  `eac3d9c1703fbecbc065f00c2bcc508e85348815d9f0d01b5f95cf407875a1c2`。
- 六份 trace steps 严格递增，并分别以 80/20/180 的冻结总步数结束；末行累计
  online time 与 report 一致。
- Stage-B client-state public reveal 为 0；reveal ledger 中没有任何
  `client_local_secret_state/public_reconstruction=true`。
- 公开重建仅出现于条件式 DP global trajectory release 和非秘密 provider
  metadata；weight/bias/count 三条流的三方末次 commitments 互异。
- Stage-B online + Stage-C online 精确等于 report 的 total online。
- manifest 中 6 report + 6 trace 的 SHA256 与当前文件全部一致。
- 所有最终 accuracy、epsilon、online time、raw wall 为有限数。
- Tiny Full 在 corrected chunk 配置下正常 COMPLETE，证明旧运行崩溃不再阻塞
  该 cell；没有使用 plaintext fallback。

## 3. Claude 完成声明中仍有的问题

### 3.1 结果构建器的 freeze 硬门不完整

`new_build_corrected_v2_results.py` 中名为 `effective_matches_freeze` 的硬门只检查
seed、unroll 和 Stage-C block size，并未在代码中检查 batch、LR、chunk、
shared initialization、epochs 和总 steps。Claude 的“构建器已硬断言 effective
config 与冻结完全一致”因此不成立。

本次验收已独立逐项比较并确认六条实际配置确实匹配，所以这属于**审计器实现
缺口，不是否定现有六条结果**。以后重新生成结果前应修复构建器。

### 3.2 accountant 报告级 provenance 未完全达到交接要求

正式 runner 每个 release block 确实调用与 TFE 路线相同的 Opacus RDP accountant，
并用 gradient steps、adaptive-count steps、sample rate、noise multipliers 和 delta
计算 epsilon；effective config 也保留主要输入。但最终 report 只保存 epsilon 和
accounting 耗时，没有显式保存：

- accountant/Opacus 版本；
- sample rate 与 count steps；
- noise/count-noise multiplier 的扁平审计记录；
- accountant state/config hash。

因此“epsilon 数值已计算且可由冻结代码/配置复核”成立，但“交接要求的完整
accountant provenance 全部进入正式 report”不成立。论文可以使用 conditional
Stage-C epsilon，但应引用冻结配置和 accountant 源码 hash，不应声称 report 已
包含完整 accountant state ledger。

### 3.3 SPU 1×3 图仍有可见占位符

`new_figure2_spu_corrected_v2_secure_time` 的六个顶部 summary box 都写成
`features=…`，没有显示实际 secure feature count。该图数据曲线、颜色、T@线、
Stage-B→C 点线、n=1 和条件式 DP footer 基本正确，但**当前版本不建议直接提交
论文**。这是纯绘图修正，不需要重跑实验；在资源恢复后用已有 CSV 重画即可。

### 3.4 dual-backend 图只是候选补充图

corrected dual-backend 图只读取 TFE seed42，并把上下两行都标为 n=1；没有呈现
当前 TFE 主结果的 CIFAR-10/100 三 seed 均值/标准差，也没有复刻 1×3 图的
summary box、T@Target 横线和 Stage transition。footer 已正确声明不是严格的
hardware-normalized speedup comparison。

因此它可用于内部核对或补充材料中的“seed42 后端轨迹示意”，不应替代当前
TFE 三 seed 主 Figure 2，也不应用于声称 SPU 相对 TFE 的严格加速比。

### 3.5 Tiny Full 的旧故障原因不可改写成已证明 OOM

旧证据只足以标记 `UNRESOLVED_RUNTIME_CRASH`：当时没有直接 cgroup OOM event、
exit signal 和 worker RSS 证明。新 chunk runner 完成 Tiny Full，说明 chunking
解决了旧路径的运行阻塞并显著降低图/输入规模，但不能反向证明旧崩溃一定是
OOM。论文安全表述应为“chunked execution enabled completion of the previously
failing Tiny Full cell”，不要写“proved/fixed OOM”。

### 3.6 单 seed 与时间分辨率

SPU 六条均只有 seed42，不能画误差带或做显著性结论。CIFAR-100 只有 2 个、Tiny
只有 18 个 Stage-C release 点；T@Target 是第一个公开轨迹点达到阈值的累计时间。
CIFAR-10/CIFAR-100/Tiny 多个 T@值落在同一个首个 release 点是预期结果，但应写
“first observed release time”，不能解释为精确越过阈值的瞬时值。

## 4. 当前可用于论文主文的图表

### 4.1 主结果与样本效率

1. `new_expFig/new_true_ours/new_figure2_pea_style_secure_time.{png,pdf,svg}`  
   当前最适合作为主 Figure 2：Full Data 与 True Ours 的 development accuracy--
   cumulative online secure time。C10/C100 为 n=3，Tiny 为用户批准的 n=1；必须
   保留 conditional Stage-C DP 和非 official-test 口径。

2. `new_expFig/new_public_mtt_mainline/new_figureA1_accuracy_vs_steps.{png,pdf}`  
   Stage A 的训练步数/精度曲线，用于说明 Official MTT 表示的收敛行为。

3. `new_expFig/new_public_mtt_mainline/new_figureA2_sample_efficiency.{png,pdf}`  
   Stage A Full/Random subset/Official MTT 样本效率。正文必须写实际 compact size
   480/4,800/9,600，即 0.96%/9.6%/9.6%，不能统一写成 1%。

对应主表：

- `new_expTab/new_public_mtt_mainline/new_module_a_efficiency_summary.csv`
- `new_expTab/new_true_ours/new_figure2_time_to_target_table.csv`
- `new_log/new_true_ours/new_true_ours_master_results.json`

### 4.2 正 Stage-B / Non-IID 扩展

4. `new_expFig/new_stage_b_anchor_postagg/new_figure_b1_initialization_sources.{png,pdf}`  
   Random+Anchor、Local-0+Anchor、Secure Average、Average+Anchor 的初始化源比较。

5. `new_expFig/new_stage_b_anchor_postagg/new_figure_b2_anchor_convergence.{png,pdf}`  
   Anchor CE 收敛轨迹。

6. `new_expFig/new_stage_b_anchor_postagg/new_figure_b3_classwise_correction.{png,pdf}`  
   类别级修正，可作为主文分析图或补充图。

7. `new_expFig/new_stage_b_anchor_postagg/new_figure_b4_stage_b_runtime.{png,pdf}`  
   Stage-B plaintext/TFE runtime。当前只有 C10/C100 seed42，通信量仍为 N/A。

对应表：`new_expTab/new_stage_b_anchor_postagg/` 下的 main table、init-source、
step-efficiency、runtime 和 TFE results CSV。该路线是 Official-MTT-only、
development-only、seed42；适合表述为 Non-IID 正扩展，不能声称跨 seed 稳定。

## 5. 可用于补充材料的图表

### 5.1 SPU 后端可行性/单 seed 系统补充

8. `new_expFig/new_secretflow_spu/corrected_v2/new_figure2_spu_corrected_v2_secure_time.{png,pdf,svg}`  
   **数据可用，图片需先把 `features=…` 改为真实计数后再投稿。**用于说明
   corrected SecretFlow-SPU/ABY3 后端复现相同定性趋势；所有曲线 n=1。

9. `new_expFig/new_secretflow_spu/corrected_v2/new_figure2_dual_backend_corrected_v2_2x3.{png,pdf}`  
   只适合作为 seed42 后端轨迹示意或内部图。若放补充材料，必须保留“不作严格
   hardware-normalized speedup”图注；不能替代主 Figure 2。

对应表：

- `new_output/new_secretflow_spu/corrected_v2/new_spu_corrected_v2_end2end_main_table.csv`
- `new_output/new_secretflow_spu/corrected_v2/new_spu_corrected_v2_runtime_breakdown.csv`
- `new_log/new_secretflow_spu/corrected_v2/new_spu_corrected_v2_hash_manifest.json`

### 5.2 负结果/机制消融

以下图片可以进入补充材料，但必须明确标为负结果，不能包装成正模块：

- `new_expFig/new_end2end_adaptive_followup/new_fixed_vs_adaptive_validation.{png,pdf}`
- `new_expFig/new_end2end_adaptive_followup/new_fixed_vs_adaptive_clipping.{png,pdf}`  
  Fixed vs Adaptive 历史消融；Adaptive 没有持续 utility 优势。
- `new_expFig/new_stage_b_kd/new_figureB1_noniid_initializer_quality.{png,pdf}`
- `new_expFig/new_stage_b_kd/new_figureB2_client_diversity_teacher_gain.{png,pdf}`
- `new_expFig/new_stage_b_kd/new_figureB3_runtime_breakdown.{png,pdf}`
- `new_expFig/new_stage_b_kd/new_figureS1_anchor_size_sweep.{png,pdf}`
- `new_expFig/new_stage_b_kd/new_figureS2_temperature_sweeps.{png,pdf}`
- `new_expFig/new_stage_b_kd/new_figureS3_plaintext_tfe_numerics.{png,pdf}`
- `new_expFig/new_stage_b_kd/new_figureS4_class_weight_distributions.{png,pdf}`
- `new_expFig/new_stage_b_kd/new_figureS5_classwise_accuracy_delta.{png,pdf}`  
  KD/Consensus 因果分解：主要增益来自有标签 anchor CE，KD 独立正贡献不成立。
- `new_expFig/new_stage_b_anchor_refinement/new_figure_b1_initializer_quality.{png,pdf}`
- `new_expFig/new_stage_b_anchor_refinement/new_figure_b2_disagreement_vs_error.{png,pdf}`
- `new_expFig/new_stage_b_anchor_refinement/new_figure_b3_utility_cost.{png,pdf}`
- `new_expFig/new_stage_b_anchor_refinement/new_figure_b4_classwise.{png,pdf}`  
  disagreement-guided weighting/selection 的完成负结果。
- `new_expFig/new_stage_b_heterogeneity_trigger/new_figure_b1_heterogeneity_vs_anchor_benefit.{png,pdf}`
- `new_expFig/new_stage_b_heterogeneity_trigger/new_figure_b2_routed_stage_b_accuracy.{png,pdf}`
- `new_expFig/new_stage_b_heterogeneity_trigger/new_figure_b3_stage_b_runtime.{png,pdf}`
- `new_expFig/new_stage_b_heterogeneity_trigger/new_figure_b4_trigger_decision_diagram.{png,pdf}`  
  heterogeneity-trigger routing 的完成负结果；trigger 关闭 CE 会产生较大 regret。

对应 CSV 位于同名 `new_expTab/new_stage_b_kd/`、
`new_expTab/new_stage_b_anchor_refinement/` 和
`new_expTab/new_stage_b_heterogeneity_trigger/` 目录。

## 6. 当前不要放入论文正式结果的图

- `new_expFig/new_secretflow_spu/new_figure2_spu_seed42_secure_time.*` 和
  `new_figure2_dual_backend_2x3_seed42.*`：旧 SPU 证据已隔离，存在 client reveal、
  deterministic driver noise、无正式 accountant 和错误 timing boundary。
- `new_expFig/new_true_ours/new_true_ours_final_test.*`、
  `new_expFig/new_public_mtt_mainline/new_figureF_final_test_privacy_utility.*`、
  `new_expFig/new_end2end_adaptive_followup/new_improved_ours_vs_full_data_test.*`：
  历史 final-test 路线，不属于当前权威 development-only 主协议。
- `new_expFig/new_public_mtt_mainline/new_figureS_tfe_system_scaling.*`：旧系统表包含
  0 秒/RSS 占位值，不能作为当前系统测量。
- `new_expFig/new_incremental_comparison/*`：旧 incremental 协议，已被当前公平
  Full Data vs True Ours 主比较取代。
- `new_expFig/new_public_mtt_mainline/new_figureE_end_to_end_two_curve.*`：旧端到端
  协议，只能作为历史 provenance，不进入当前主结论。
- `new_expFig/new_public_mtt_mainline/new_figureB2_stage_b_noniid_initialization.*`、
  `new_figureC_stage_c_privacy_utility.*`、`new_figureC_stage_c_clipping_trajectory.*`
  以及 `new_expFig/new_secure_feature/new_phase6_public_mtt_reference_curves.*`：属于
  较早或诊断协议。若确需引用，必须先逐图回溯当时 config/seed/data scope，当前不
  建议用它们替代最新 Post-Aggregation、True Ours 或 corrected-v2 结果。

## 7. 论文实验章节推荐结构

1. **Experimental setup and claim boundary**：数据集、2048-D frozen features、
   parties、ABY3、development-only、conditional Stage-C DP。
2. **Compact-data efficiency**：Figure A1/A2 + Stage-A efficiency table。
3. **End-to-end utility/time trade-off**：TFE PEA-style Figure 2 + T@Target table。
4. **Stage-B Non-IID mechanism**：Post-Aggregation Figure B1/B2，B3/B4 放补充。
5. **Backend feasibility**：修正占位符后的 SPU Figure 2 + corrected-v2 runtime table；
   明确 n=1、非硬件归一化。
6. **Ablations and negative findings**：Adaptive、KD、disagreement、trigger 图放补充，
   主文用一张紧凑表总结“不支持正贡献”。
7. **Limitations**：无当前 official-test 主表、无 end-to-end record DP、无通信量、
   Post-Aggregation 和 SPU 主要为 seed42。

## 8. 当前论文就绪判断

- corrected-v2 六条实验结果：`ACCEPTED_WITH_SCOPE`。
- SPU runtime/main table：可进入系统补充材料。
- SPU 1×3 图片：`MINOR_FIGURE_FIX_REQUIRED`（替换 feature count 占位符）。
- dual-backend 图片：`SUPPLEMENTARY_CANDIDATE_ONLY`。
- “所有交接门全部满足”：不成立，主要缺 accountant provenance 和 builder 完整
  freeze hard gate；不需要重跑现有六条即可修复未来审计/呈现。


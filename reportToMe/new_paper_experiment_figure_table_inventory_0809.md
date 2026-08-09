# 论文实验章节图表清单（2026-08-09）

## 主文当前可用

| 编号 | 文件/位置 | 主要证明 | 关键边界 |
|---|---|---|---|
| Figure 2(a) | `new_expFig/paper_experiment/new_figure2a_tfe_pea_i_errorbars.{png,pdf}` | 原始TFE准确率—时间轨迹 | development；累计秒数对数轴；C10/C100 n=3有I型误差条，Tiny n=1无误差条 |
| Figure 2(b) | `new_expFig/paper_experiment/new_figure2b_spu_short_axis_deltas.{png,pdf}` | SPU短轴以首发布为基准展开真实微小准确率变化 | development；分钟短窗、纵轴为变化pp且标绝对端点、带明确断口；全部n=1；T@Target仍按原始绝对时间 |
| Figure 3 | `new_expFig/paper_experiment/new_figure3_compact_carrier_fast_convergence.{png,pdf}` | 蒸馏载体更快达到冻结目标，Ours按目标处提前停止 | n=3；原冻结协议的定点只读official-test |
| Figure 4 | `new_expFig/paper_experiment/new_figure4_initializer_source_bars.{png,pdf}` | 异构条件下Secure Average+Anchor优于主要初始化对照 | development，seed42，n=1 |
| Figure 5 | `new_expFig/paper_experiment/new_figure5_anchor_convergence_pea.{png,pdf}` | Anchor修正轨迹及初始化来源差异 | development，seed42，n=1，因此不画误差条 |
| Figure 6 | `new_expFig/paper_experiment/new_figure6_user_tuned_three_datasets.{png,pdf}` | 三数据集上独立配置Fixed/Adaptive的准确率轨迹及Adaptive C_t行为 | development，n=3，epsilon=3；同数据集内共享输入但两方法优化参数不同；准确率有I型样本标准差，灰色Adaptive C_t无误差条；横轴为更新步而非时间 |
| Table 4 | `初稿_真实密态环境.md` §5.4 | Figure 4的精确数值 | 与Figure 4重复；等待用户决定移到补充材料或保留 |
| Table 5 | `初稿_真实密态环境.md` §5.5 | 三数据集独立配置Fixed/Adaptive的最终准确率、AUC和最终C | development，n=3，epsilon=3；Adaptive最终增益为C10 +11.600、C100 +6.447、Tiny +4.123pp；属于完整配置比较而非裁剪单因素因果消融 |
| Table 6 | `初稿_真实密态环境.md` §5.6 | 本地matched control与公开文献背景 | 只有同后端本地成对行可直接比较；外部行不是统一排行榜 |
| Table 7 | `初稿_真实密态环境.md` §5.7 | Plaintext与TFE数值一致性 | Severe、seed42；只报告accuracy/gap |
| Table 8 | `初稿_真实密态环境.md` §5.7 | corrected SPU时间与内存分解 | n=1；conditional Stage-C DP；无统一通信字节计数 |

## 推荐放补充材料

- `new_expFig/new_stage_b_anchor_postagg/new_figure_b3_classwise_correction.{png,pdf}`：类别级Anchor修正。
- `new_expFig/new_stage_b_anchor_postagg/new_figure_b4_stage_b_runtime.{png,pdf}`：Stage-B plaintext/TFE运行时间，C10/C100 seed42。
- `new_expFig/new_public_mtt_mainline/new_figureA2_sample_efficiency.{png,pdf}`：Full、Random subset、Official MTT样本效率；必须写480/4800/9600的真实样本数。
- KD、disagreement selection与heterogeneity trigger目录中的图可作为负结果补充，但必须标记`COMPLETE_NEGATIVE`，不能声称正贡献。

## 本轮明确省略

- Figure 7：缺少owner count × dimension × LAN/WAN的匹配矩阵，也没有统一通信计数器；现阶段补齐代价高且不能由已有数据可靠推导。
- 完整八单元Adaptive析因表：这些匹配实验没有运行，已删除占位值。

## 禁止使用

- 旧SPU seed42图：存在client reveal、driver noise、accountant和timing边界问题。
- 历史final-test图：不属于当前development-only权威协议。
- 旧TFE system scaling图：含0秒/RSS占位值。
- 任何将conditional Stage-C epsilon=3写成端到端record-level DP的图注或表述。

完整来源与不确定性映射见
`reportToMe/new_paper_figure_source_map_0809.md`；SPU逐条验收见
`reportToMe/new_spu_corrected_v2_acceptance_and_paper_figure_inventory.md`。

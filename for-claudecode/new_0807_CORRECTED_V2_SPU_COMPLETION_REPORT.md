# Corrected-v2 SPU 正式矩阵完成报告

更新时间：2026-08-08（Asia/Shanghai）
分支：`new-secretflow-spu-end2end`
最后提交：`f0af78a`（全部推送远端）

## 1. 总体状态

`plan/0807_spu_formal_correction_and_pea_rebuild.md` 的 corrected-v2 正式矩阵
**6/6 全部 COMPLETE**。安全、DP accountant、reveal、freeze、timing、
test/holdout 全部验收门通过。PEA 风格 Figure 2 与 dual-backend 图已生成并视觉检查。

## 2. 六条正式 cell 结果

| 数据集 | 方法 | SPU dev acc | TFE dev acc | Δ | StageB online(s) | StageC online(s) | Total online(s) | Raw wall(s) | Mem(GiB) | ε | T@90(s) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | True Ours | 0.6864 | 0.6886 | -0.002 | 110.4 | 25.5 | 135.9 | 213.3 | 13.4 | 3.0 | 113.6 |
| CIFAR-10 | Full Data | 0.7254 | 0.7300 | -0.005 | 1060.6 | 25.8 | 1086.4 | 2622.3 | 14.9 | 3.0 | 1063.8 |
| CIFAR-100 | True Ours | 0.4766 | 0.4866 | -0.010 | 2565.8 | 8.3 | 2574.1 | 3402.1 | 14.4 | 3.0 | 2570.0 |
| CIFAR-100 | Full Data | 0.4252 | 0.4340 | -0.009 | 6167.6 | 9.0 | 6176.6 | 8384.3 | 17.5 | 3.0 | 6172.2 |
| Tiny | True Ours | 0.3084 | 0.3141 | -0.006 | 6061.0 | 223.6 | 6284.7 | 8302.6 | 22.8 | 3.0 | 6067.6 |
| Tiny | Full Data | 0.2768 | 0.2720 | +0.005 | 21148.7 | 224.0 | 21372.7 | 26070.6 | 23.3 | 3.0 | 21154.7 |

**主趋势跨后端复现**：Ours 在三个数据集都更快达 T@90（CIFAR-10 快 9.4×、
CIFAR-100 快 2.4×、Tiny 快 3.5×）。

## 3. 关键修复验证

corrected-v2 修复了旧隔离 Claude 实现的所有缺陷，并被六条正式 run 证实：
- Stage-B 五客户端状态全程 SPUObject，client-state public reveals = 0（全部 cell）
- DP 噪声由三方 PYU 各自 CSPRNG 生成，driver 无 seed/share；provider commitments 互异
- Opacus RDP accountant 覆盖 gradient + adaptive-count steps，ε=3.0/δ=1e-5
- effective config 与冻结完全一致（无事后调参）
- timing 分离 compile/infeed/online/checkpoint/eval/accounting
- **Tiny Full 从旧 NEGATIVE_STOP 变为 COMPLETE（0.2768）**，chunk 图成功修复 OOM

## 4. 结果文件索引

- main table：`new_output/new_secretflow_spu/corrected_v2/new_spu_corrected_v2_end2end_main_table.csv`
- hash manifest：`new_log/new_secretflow_spu/corrected_v2/new_spu_corrected_v2_hash_manifest.json`
- plot data：`new_output/new_secretflow_spu/new_figure2_spu_corrected_v2_plot_data.csv`
- runtime breakdown：`new_output/new_secretflow_spu/corrected_v2/new_spu_corrected_v2_runtime_breakdown.csv`
- final report：`new_log/new_secretflow_spu/corrected_v2/new_spu_corrected_v2_final_report.md`
- SPU Figure 2：`new_expFig/new_secretflow_spu/corrected_v2/new_figure2_spu_corrected_v2_secure_time.{png,pdf,svg}`
- dual-backend：`new_expFig/new_secretflow_spu/corrected_v2/new_figure2_dual_backend_corrected_v2_2x3.{png,pdf}`
- 六条 report/log/effective config/trace：`new_log/new_secretflow_spu/corrected_v2/` 与 `new_output/new_secretflow_spu/corrected_v2/`

## 5. 科学边界（严格保持）

- 全部 accuracy 为 development-validation；**official test 与 locked holdout 未读**
- ε=3 只覆盖 conditional Stage C；Stage-B initializer 是 data-dependent non-DP
- cgroup CPU utilization 仅诊断（容器内有用户 Claude 进程），不作论文基准
- 通信量 N/A；无 plaintext fallback
- 未覆盖旧 TFE Figure 2；旧隔离 SPU 证据保留为 QUARANTINED_DIAGNOSTIC

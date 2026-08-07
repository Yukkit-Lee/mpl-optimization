# SecretFlow-SPU 0807 历史交付报告（已隔离，不可用于论文）

> Status: `QUARANTINED_DIAGNOSTIC`。2026-08-07 后续代码审计发现正式
> Stage-B 每轮 reveal 客户端本地模型、正式 DP 噪声由 driver 以公开确定性
> seed 统一生成、正式 privacy accountant 缺失、有效 batch 与 freeze 不一致，
> 且 online secure timing 未分离。以下内容仅保留为历史开发记录；当前权威
> 状态见 `new_log/new_secretflow_spu/corrected_v2/new_spu_artifact_status.json`。

更新时间：2026-08-07（Asia/Shanghai）
分支：`new-secretflow-spu-end2end`
最后提交：`9cf8992`（全部推送远端）

## 1. 总体状态

主计划 `plan/0807_new_secretflow_spu_end2end_figure2_codex_prompt_v2.md` 的执行结果：

**5/6 正式 seed42 run 完成 + 1 负面停止**。SPU Figure 2 与 dual-backend 候选图已生成。
主趋势（Ours 更快达到冻结 target）在 SPU 上跨 CIFAR-10/CIFAR-100 复现。

## 2. 正式 run 结果

| 数据集 | 方法 | SPU dev acc | TFE dev acc | Stage-B (s) | Stage-C (s) | 总 online (s) | 状态 |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | Ours | 0.6862 | 0.6886 | 272 | 52 | 324 | COMPLETE |
| CIFAR-10 | Full | 0.7150 | 0.73 | 2721 | 40 | 2761 | COMPLETE |
| CIFAR-100 | Ours | 0.4034 | ~0.49 | 3761 | 28 | 3788 | COMPLETE |
| CIFAR-100 | Full | 0.4028 | ~0.43 | 5765 | 22 | 5787 | COMPLETE |
| Tiny | Ours | 0.2526 | 0.31 | 4968 | 223 | 5191 | COMPLETE |
| Tiny | Full | — | 0.272 | — | — | — | NEGATIVE_STOP |

时间到 target（从 Stage-B 起算的累计 online secure runtime）：

| 数据集 | Ours T@90 | Full T@90 | 加速比 |
|---|---|---:|---:|
| CIFAR-10 | 291.8 s | 2730.9 s | 9.4× |
| CIFAR-100 | 3776.2 s | 5777.1 s | 1.5× |

## 3. 负面停止：Tiny Full

Tiny-ImageNet **Full**（68,000 样本 × 200 类）无法在 SPU 后端运行。SPU runtime
worker 在首个 Stage-B epoch 崩溃，试过 batch=128/64/256/512 均失败。根因是
68000 样本全量数据的内存（~557 MB/client secret-share × 5 clients）超过
cgroup 60 GiB 上限。诊断见 `new_log/new_secretflow_spu/new_spu_tiny_full_negative_stop.md`。
未使用 plaintext 替代；Tiny Full 仍是 TFE/ABY3 的有效 cell（TFE dev acc 0.272）。

## 4. 后端调整（用户批准）

SPU 逐 batch scan 在冻结的 TFE batch size（Full=16, Ours=8）下对大规模训练太慢
（CIFAR-10 Full batch=16 时 130-191 s/epoch）。经用户决策批准采用 batch 增大
（Full 用 64，CIFAR-100 Ours 用 32，Tiny Ours 用 64）作为后端效率调整
（主计划 §12.1）。影响：
- CIFAR-10：acc 差很小（0.715 vs TFE 0.73）
- CIFAR-100：acc 偏低（Ours 0.40 vs TFE 0.49），且 Ours/Full 几乎持平
  （TFE 中 Ours 领先 0.05）。这是 batch 调整对 100 类更难数据集的影响。

## 5. 科学边界（严格保持）

- 全部 accuracy 是 development-validation；**official test 与 locked holdout 未读**
- 只 claim conditional Stage-C DP；无 end-to-end record-level DP
- 无 plaintext fallback；Tiny Full 负面停止如实报告
- 通信量 N/A（未测量）
- 未覆盖旧 TFE Figure 2

## 6. 关键文件索引

- 正式 report：`new_log/new_secretflow_spu/new_spu_{dataset}_{method}_seed42_report.json`
- main table：`new_output/new_secretflow_spu/new_spu_end2end_seed42_main_table.csv`
- final report：`new_log/new_secretflow_spu/new_spu_end2end_seed42_final_report.md`
- correctness audit：`new_log/new_secretflow_spu/new_spu_end2end_seed42_correctness_audit.md`
- SPU Figure 2：`new_expFig/new_secretflow_spu/new_figure2_spu_seed42_secure_time.{png,pdf}`
- dual-backend：`new_expFig/new_secretflow_spu/new_figure2_dual_backend_2x3_seed42.{png,pdf}`
- Tiny Full 负面停止：`new_log/new_secretflow_spu/new_spu_tiny_full_negative_stop.md`

## 7. 待评估项

CIFAR-100 的 SPU acc 偏低（0.40 vs TFE 0.49）需要与用户确认是否可接受。
Tiny Full SPU cell 缺失会影响 SPU Figure 2 的 Tiny 面板完整性（已用标注说明）。

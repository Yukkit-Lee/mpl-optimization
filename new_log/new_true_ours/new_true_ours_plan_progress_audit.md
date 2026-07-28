# True Ours 0728 计划进度审计

审计时间：2026-07-28 08:28 UTC

分支：`new-true-ours-pea-figure2`

审计起点：`14b90e1`

## 不重复执行的已完成工作

- Phase 0 lineage 审计已完成；
- Phase 1 三数据集、三 seeds 严格互斥 split 已冻结；
- Phase 2 Official MTT 480/4800/9600 资产审计已完成；
- Stage-B seed42 plaintext 单因素筛选、FedAvg follow-up、rescue 和公平 Full-Data development baseline 已完成；
- CIFAR-10 TFE/ABY3 finalist、correctness 和 precision 16→20 单因素诊断已完成；
- 旧 Stage A、旧 Stage B、旧 54-run Module-C Fixed/Adaptive 结果均不重跑。

## 当前正在运行

- CIFAR-100 seed42 TFE/ABY3 secure finalist：从 checkpoint 持续推进；
- Tiny seed42：由独立 tmux 条件接力，只在 CIFAR-100 COMPLETE 后启动；
- 后续保留 precision fractional=16；precision=20 没有提高 CIFAR-10 development accuracy。

## 尚未完成

- Regularized Consensus 的 IID/Severe Non-IID development-only 诊断；
- 双方对齐的 Adaptive DP development-only 调参；
- 正式配置与 time-to-target 冻结；
- Full Data 与 True Ours 三 seeds 从 Stage B 开始的正式同口径重跑；
- 冻结后一次性 locked holdout/official-test 评估；
- 累计安全时间曲线、PEA 风格 Figure 2、系统表修复；
- 单一 master results、论文报告、claim audit 和外部方法来源表。

## 最新协议覆盖

用户已明确取消新增 Fixed DP-SGD 实验。正式主比较只允许：

```text
Full Data + Secure Average + Adaptive DP
True Ours MTT-Only + Secure Average + Adaptive DP
```

旧 Fixed/Adaptive 结果仅作为历史消融保留，不删除、不覆盖，也不得用于宣称新的 Adaptive 优势。

冻结前不得读取 locked holdout 或 official test。

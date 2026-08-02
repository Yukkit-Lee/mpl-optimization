# Current Paper Readiness Assessment

更新时间：2026-08-02（UTC）
状态：**READY WITH EXPLICIT SCOPE LIMITATIONS**
来源提交：`704755d70d2475a6a306a1b09e4e5e65195baf6c`

本文件已取代2026-07-28版本。完整数据、负结果和证据路径见
[`unified_experiment_report.md`](unified_experiment_report.md)。

## 可进入论文

- Stage A Official MTT样本效率：三数据集3 seeds；正确compact规模为
  480/4,800/9,600，不可统一写成480或1%。
- True Ours vs Full Data：14/14 development-validation结果。C10/C100各3
  seeds，Tiny仅seed42。三个数据集均更快达到T@90；最终accuracy并非全胜。
- 真实TFE/ABY3可行性与runtime/RSS：使用True Ours timeline和最新
  Post-Aggregation正式结果，不使用旧0秒占位System表。
- Secure Average + Official-MTT Anchor CE：正Stage-B/Non-IID扩展，当前为
  CIFAR-10/100 seed42 development-only。

## 只适合作为负/补充实验

- KD/Consensus因果分解：独立KD贡献不成立。
- Disagreement-guided refinement：效用和成本门失败。
- Heterogeneity-trigger routing：oracle regret门失败。
- Fixed vs Adaptive：Adaptive没有持续显著效用优势。
- 旧End-to-End：协议不同且已被当前公平主比较取代。

## 当前不能声称

- 当前主结果是official-test或locked-holdout结果；两者在权威链中均未读取。
- 整个端到端发布模型是record-level DP；epsilon只覆盖条件式Stage-C机制。
- 三个数据集最终都更快、更准。
- class-aware KD、disagreement weighting或trigger routing是正贡献。
- 通信字节或轮数降低；没有测量。

## 投稿前最值得补的内容

1. 若审稿要求test主表，执行一次冻结后、带读取ledger的最终评估。
2. Post-Aggregation补seed43/44以支持跨seed稳定性。
3. 核对外部方法的官方数字与协议。

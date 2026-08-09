# Figure 6用户独立配置三数据集扩展计划（2026-08-10）

状态：`COMPLETE`

## 目标

保留已完成的CIFAR-10用户独立配置结果，不重复运行；使用完全相同的Fixed和
Adaptive方法参数，新增CIFAR-100与Tiny ImageNet各三seed结果，最终生成三个
数据集的准确率—private-update-step面板，并在每个面板右轴只画Adaptive平均
`C_t`灰线。

## 冻结参数

- Fixed：steps280，LR0.013，expected batch16，C=2；
- Adaptive：steps160，LR0.002，expected batch32，C0=2，gamma0.5，eta0.1，
  warmup0，update interval5，count-noise16，C范围[0.05,4]；
- 两方法：epsilon3，delta1e-5，seeds42/43/44；
- 每个候选按自身steps/sample-rate/count-releases重新校准Opacus噪声。

## 输入边界

- CIFAR-10：复用`new_user_tuned_fixed_vs_adaptive_formal.json`，不重跑；
- CIFAR-100：Anchor-post-aggregation C5 seed42初始化与Official-MTT Stage-C
  seed42缓存；
- Tiny：仓库没有同类型C5正式模型，使用Official-MTT-only True-Ours Stage-B
  selected seed42初始化与Official-MTT Stage-C seed42缓存；
- 两种方法在同一数据集内共享完全相同的初始化器、训练数据和development划分；
- 禁止读取official test和locked holdout。

## 阶段

1. `COMPLETE`：资源、输入、依赖、配置与SHA预检；
2. `COMPLETE`：提交并推送runner/plan/preflight；
3. `COMPLETE`：在tmux运行12个新增任务；
4. `COMPLETE`：核验epsilon、三seed完整性、输入边界和结果；
5. `COMPLETE`：按用户版式重画Figure 6、更新Table 5与5.5；
6. `COMPLETE`：视觉检查、文档同步、提交推送。

## 图形语义

- 三面板：CIFAR-10、CIFAR-100、Tiny ImageNet；
- 左轴：Fixed/Adaptive准确率均值与三seed样本标准差I型误差条；
- 右轴：仅Adaptive平均`C_t`灰线，不画Fixed C、不画C误差条；
- 横轴为private-update step；更少步数只有在达到稳定且有竞争力准确率时才表示
  迭代收敛更快，不代表绝对运行时间更短。

## 完成结果

- 12条新增任务全部COMPLETE；CIFAR-10六条从SHA锁定源结果复用；
- 全部18条均为epsilon3，official test与locked holdout均未读；
- Adaptive相对Fixed最终增益：CIFAR-10 `+11.600 ± 1.752pp`、CIFAR-100
  `+6.447 ± 0.163pp`、Tiny `+4.123 ± 0.417pp`；
- 正式图：`new_figure6_user_tuned_three_datasets.{png,pdf}`；
- 科学边界：两方法steps/LR/batch不同，结果是完整配置对比，不是Adaptive
  clipping的单因素因果证据，也不证明真实时间加速。

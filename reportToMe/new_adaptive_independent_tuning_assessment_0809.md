# Adaptive独立调优消融验收（2026-08-09）

## 结论

在用户与导师确认“仅隐私预算等核心条件相同、优化参数允许分别调优”的口径下，
Fixed-DP旧三seed结果保持完全冻结，Adaptive-DP经过400个development-only候选
筛选后，在formal seeds 42/43/44上均获得更高最终准确率。

| 方法 | 参数 | Final development accuracy | Early AUC | Final C | epsilon | Count releases |
|---|---|---:|---:|---:|---:|---:|
| Fixed DP（冻结） | steps320, LR.001, batch16, C2 | 58.673% ± 0.214pp | 58.321% ± 0.149pp | 2.000 ± 0.000 | 3.0 | 0 |
| Adaptive DP（独立调优） | steps160, LR.003, batch32, C0=2, gamma=.5, eta=.1 | 59.073% ± 0.081pp | 58.419% ± 0.305pp | 1.254 ± 0.016 | 3.0 | 32 |

逐seed Adaptive−Fixed为`+0.14/+0.62/+0.44pp`，均值为
`+0.40 ± 0.24pp`；Early AUC均值差为`+0.098pp`。因此新结果支持：在相同
epsilon、数据、initializer和development评价边界下，独立调优的Adaptive在三个
formal seed上稳定优于冻结Fixed，但当前增益仍小于1个百分点，不应写成大幅提升。

## 调优与边界

- Fixed未重新运行，也未用于选择Adaptive候选；旧Fixed JSON SHA256为
  `3709c4d...858`，三条配置均硬核验steps320/LR.001/C2。
- Adaptive screen固定RNG17，按112项预算/LR/batch、240项控制器、48项更新日程
  顺序共运行400项；所有epsilon=3最大误差约`7.1e-13`。
- 最终Adaptive配置在formal前冻结，之后只运行seeds42/43/44，不做结果后扩网格。
- official test与locked holdout均未读取；clipped fraction是非私有诊断，不是协议
  对外release；epsilon仍只覆盖conditional Stage-C。
- 两方法steps/batch/LR不同，因此这是`independently tuned / same-epsilon`
  比较，不是matched compute-budget或纯粹的单因素因果消融。

## 产物

- 配置：`new_adaptive_independent_tuning_freeze.json`、
  `new_adaptive_independent_formal_freeze.json`
- Screen：`new_adaptive_independent_screen.{json,csv}`
- Formal：`new_adaptive_independent_formal.{json,csv}`
- 绘图脚本：`new_scripts/new_paper_figures/new_build_figure6.py`
- Figure 6：`new_figure6_adaptive_independently_tuned_pea.{png,pdf}`

# True Ours 0728 计划进度审计

审计时间：2026-07-28 10:08 UTC

分支：`new-true-ours-pea-figure2`

审计提交：`a88eaea`

## 不重复执行的已完成工作

- Phase 0 lineage 审计已完成；
- Phase 1 三数据集、三 seeds 严格互斥 split 已冻结；
- Phase 2 Official MTT 480/4800/9600 资产审计已完成；
- Stage-B seed42 plaintext 单因素筛选、FedAvg follow-up、rescue 和公平 Full-Data development baseline 已完成；
- CIFAR-10 TFE/ABY3 finalist、correctness 和 precision 16→20 单因素诊断已完成；
- 旧 Stage A、旧 Stage B、旧 54-run Module-C Fixed/Adaptive 结果均不重跑。

## Stage B seed42 已完成

- CIFAR-10/CIFAR-100/Tiny 的 TFE/ABY3 secure finalist 均 COMPLETE；
- development accuracy 分别为 `0.6876 / 0.4800 / 0.3148`；
- 后续保留 precision fractional=16；precision=20 没有提高 CIFAR-10；
- `secure_average_only` 的 CIFAR-10 等价门通过：accuracy delta=0，
  weight/bias max error=`4.12e-4/4.58e-5`，runtime
  `458.55s→93.89s`，加速 `4.88×`，正式 Stage B 可启用。

## 尚未完成

- Regularized Consensus 的 IID/Severe Non-IID development-only 诊断；
- 双方对齐的 Adaptive DP development-only 调参；
- 正式配置与 time-to-target 冻结；
- Full Data 与 True Ours 三 seeds 从 Stage B 开始的正式同口径重跑；
- 冻结后一次性 locked holdout/official-test 评估；
- 累计安全时间曲线与 PEA 风格 Figure 2；
- 单一 master results、论文报告、claim audit 和外部方法来源表。

## 已完成的独立后续项

- Phase 11 TFE system 表字段修复完成：9/9 权威 source rows 均通过
  `secure_runtime_seconds>0`、`peak_rss_bytes>0` 硬断言；
- repaired JSON/CSV 只用于历史 kernel scaling 补充表，Adaptive 主图仍等待本轮
  实际 Stage-B + Stage-C 累计时间。
- Adaptive Stage-C sampled-Gaussian accountant 已与 CSPRNG Poisson sampling 对齐；
  真实 TFE/ABY3 masked-batch/空批次/expected-batch normalization 测试通过。
- 纠正 TFE `select(bit,x,y)` 的参数语义，修复旧 Adaptive clipping
  `fraction_below` 反向 bug；旧 Adaptive 结果不得进入本轮 formal。
- PEA/CaPC/DDGauss/CryptGPU 外部来源 provenance 已完成，Ours 数值等待 formal。

## 最新协议覆盖

用户已明确取消新增 Fixed DP-SGD 实验。正式主比较只允许：

```text
Full Data + Secure Average + Adaptive DP
True Ours MTT-Only + Secure Average + Adaptive DP
```

旧 Fixed/Adaptive 结果仅作为历史消融保留，不删除、不覆盖，也不得用于宣称新的 Adaptive 优势。

冻结前不得读取 locked holdout 或 official test。

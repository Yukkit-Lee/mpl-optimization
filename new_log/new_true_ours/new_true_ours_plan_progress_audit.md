# True Ours 0728 计划进度审计

审计时间：2026-07-28 11:20 UTC

分支：`new-true-ours-pea-figure2`

审计基线提交：`2138fa9`

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

## Consensus 决策

- synced screen 因最终 FedAvg 使 client heads 完全相同，属于结构退化；
- no-sync 单因素 follow-up 恢复了 client diversity，但三数据集均无候选通过
  `IID delta >= -0.5pp` 且 `Severe delta >= +1pp`；
- 不扩 seeds 43/44，主路线使用 MTT-only Secure Average；
- Consensus 保留为负结果，不继续对 development 过拟合。

## 尚未完成

- 双方对齐的 Adaptive DP float调参与6条真实TFE/ABY3 seed42 finalists已完成；
- formal freeze文件已生成并通过18任务preflight；必须先提交推送；
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
- Adaptive float screen 已改为每个候选同时评价 Full Data 与 True Ours，两边
  分别按真实 N 校准epsilon，并按最差 initializer retention 选择共享配置；
  完整三数据集screen已通过，六条最终轨迹epsilon≤3、finite且未撞C边界。
- 6/6 Stage-C finalists均为TFE/ABY3、Poisson os-CSPRNG、无fallback、conditional
  DP声明正确，locked holdout/test未读；下一门是提交formal freeze文件。
- Stage-B checkpoint development曲线与跨重启online-secure计时已实现并通过真实
  ABY3小测试；formal config/target/manifest已冻结精确代码与split SHA。
- PEA/CaPC/DDGauss/CryptGPU 外部来源 provenance 已完成，Ours 数值等待 formal。

## 最新协议覆盖

用户已明确取消新增 Fixed DP-SGD 实验。正式主比较只允许：

```text
Full Data + Secure Average + Adaptive DP
True Ours MTT-Only + Secure Average + Adaptive DP
```

旧 Fixed/Adaptive 结果仅作为历史消融保留，不删除、不覆盖，也不得用于宣称新的 Adaptive 优势。

冻结前不得读取 locked holdout 或 official test。

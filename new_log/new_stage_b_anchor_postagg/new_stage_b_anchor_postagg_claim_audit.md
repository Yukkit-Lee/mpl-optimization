# Stage-B Post-Aggregation Claim Audit

## 审计结论

状态：**PASS_WITH_LIMITATIONS**。冻结的 seed42 Stage-B development 实验和正式
TFE/ABY3 均支持“Secure Average 是有效的跨方初始化，随后使用类别均衡的
Official-MTT anchor 做监督修正”的有限主张。

## 有证据支持

- 同预算下，C5 相对 C1 在 CIFAR-10/CIFAR-100 Severe 分别提升
  2.16/7.44 pp。
- C5 相对固定 client-0 的 C3 在 Severe 分别提升
  36.16/11.36 pp。
- C5 相对未修正 C4 在 Severe 分别提升
  38.80/9.54 pp。
- 正式 TFE 保留 plaintext C5−C1 增益的
  99.1%/100.5%。
- TFE 类别级 C4→C5 改善类数为 CIFAR-10 10/10、
  CIFAR-100 84/100。

## 必须限定

- 当前只有 seed42，指标是 development Stage-B initializer accuracy。
- IID 下 C5 与固定 local-0 的 C3 仅相差 0.04/0.14 pp；跨机构聚合相对单客户端
  的主要价值出现在 Mild/Severe 异质场景。
- 未运行 Stage C、End-to-End、DP-SGD、official test 或 locked holdout。
- Stage-B 使用标签 anchor，因此不是 record-level DP；Secure Average 也不是新算法贡献。

## 不被支持

- 不可声称最终测试精度、End-to-End 收敛或最终模型性能提高。
- 不可声称多 seed 稳健性或统计显著性。
- 不可声称 Secure Average 在 IID 下显著优于任意单客户端初始化。
- 不可声称通信量下降；本轮未测通信字节或轮数。

Lineage: report source `58352813ad6261486d225c1e3eebdb73b973849f`; experiment source `d87b223527f13254b92f30529105177c88641adb`.

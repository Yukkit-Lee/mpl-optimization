# Stage-B MTT-Anchor Post-Aggregation Final Report

## 最终结论

本轮任务完成并达到正面停止条件。明文因果消融 Gate A--F 全部通过，冻结配置为
CIFAR-10/CIFAR-100 各 40 个 full-batch Anchor CE steps，LR=0.1/0.2，
mu=1e-3；正式 TFE/ABY3 成功复现方向。没有启动 Stage C、DP、End-to-End、
Tiny 或额外 seed。

## 核心数据

| Dataset | Dist. | C1 Random+Anchor | C3 Local-0+Anchor | C4 Average | C5 Average+Anchor | C5−C1 | C5−C3 | C5−C4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cifar10 | iid | 55.48% | 57.58% | 22.70% | 57.62% | +2.14% | +0.04% | +34.92% |
| cifar10 | mild | 55.48% | 22.58% | 22.02% | 55.78% | +0.30% | +33.20% | +33.76% |
| cifar10 | severe | 55.48% | 21.48% | 18.84% | 57.64% | +2.16% | +36.16% | +38.80% |
| cifar100 | iid | 10.18% | 21.98% | 10.24% | 22.12% | +11.94% | +0.14% | +11.88% |
| cifar100 | mild | 10.18% | 3.32% | 9.06% | 20.34% | +10.16% | +17.02% | +11.28% |
| cifar100 | severe | 10.18% | 6.26% | 8.08% | 17.62% | +7.44% | +11.36% | +9.54% |


正式 TFE Severe：CIFAR-10 C1/C4/C5 =
55.48%/18.84%/57.62%；
CIFAR-100 =
10.02%/8.02%/17.50%。

## 计划要求的 12 个回答

1. **Anchor CE 是否在所有分布下有效？** 是。C1 相对 C0、C3 相对 C2、C5
   相对 C4 的主要方向均为正；尤其 C5−C4 在六场均显著为正。
2. **Secure Average 是否对 Anchor CE 有独立贡献？** 是。相同 anchor、步数、
   LR 和 mu 下，六场 C5 均高于 C1，提升 0.30--11.94 pp。
3. **C5 是否优于 Random+Anchor？** 是，六场方向一致；Severe 为
   +2.16/+7.44 pp。
4. **C5 是否优于 Local+Anchor？** IID 仅近似持平（+0.04/+0.14 pp），但
   Mild/Severe 明显更优；因此结论应限定为异质场景优势。
5. **优势体现在哪里？** 主要是最终精度和类别修正；并非统一的更少步骤优势。
   步数效率仅是 Stage-B 内部目标，不能称作 End-to-End 加速。
6. **是否跨 IID/Mild/Severe 一致？** C5−C1、C5−C3、C5−C4 方向均一致为正，
   但相对 C3 的 IID 幅度很小。
7. **TFE 是否复现 plaintext？** 是。增益保留率为
   99.1%/100.5%，
   最大精度误差 0.02/0.16 pp。
8. **Anchor CE 增加多少 Stage-B 成本？** 正式 TFE 中 C5 anchor 分支为
   13.15s/
   120.73s；详见 runtime breakdown。
9. **Secure Average 额外成本是否值得？** Severe 下安全平均自身仅
   5.43s/
   4.69s，并带来相对 C1 的正增益；
   在本 Stage-B development 协议下值得，但不外推通信或 E2E 成本。
10. **哪些表述安全？** 可表述安全平均提供跨方中间初始化，Official-MTT
    anchor 再监督修正，且受控消融/TFE 支持其在异质场景的价值。
11. **哪些表述不被支持？** 最终 test/E2E 提升、多 seed 显著性、record-level
    DP、通信降低、以及 IID 下显著优于任意 local 初始化均不支持。
12. **模块定位？** 可作为 Stage-B 正贡献及 Non-IID 扩展；证据仍属于
    seed42 development-only，后续论文定稿最好补独立 seed 确认。

## 运行与资源

- CIFAR-10 TFE：graph build 45.56s，
  multi-branch secure compute 87.37s，
  peak RSS 2.95 GiB。
- CIFAR-100 TFE：graph build 50.83s，
  multi-branch secure compute 862.18s，
  peak RSS 8.97 GiB。
- 通信字节/轮数未测，不报告。

## Lineage 与边界

Report source commit: `58352813ad6261486d225c1e3eebdb73b973849f`。Experiment source commit:
`d87b223527f13254b92f30529105177c88641adb`。Official-MTT-only=true，
Full-data initializer dependency=false，test/locked holdout read=false。

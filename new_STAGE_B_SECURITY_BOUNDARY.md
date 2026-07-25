# New Secure Stage B Security Boundary

## 用途

说明新 Stage B 中哪些值在 TFE/ABY3 replicated secret sharing 上计算，
哪些值会被 reveal，以及 correctness 模式与正式模式的边界。

## Secret-Shared 输入与中间量

- 每个客户端的 2048 维 synthetic features 与 labels；
- local linear-head 参数；
- reference 与 selection features/labels；
- teacher logits；
- logit-MSE KD 梯度；
- KD-Avg 与 KD-ReferenceWeighted 候选参数；
- reference reliability weights；
- candidate MSE scores 与 secure selection bit；
- 最终被选中的初始化参数（在释放前）。

输入拥有方通过原生 TensorFlow provider placeholder 向
`tfe.define_private_input` 提供数据。没有使用旧版
`ABY3PrivatePlaceholder.feed()` 的 NumPy 随机拆分。

## 默认 Reveal

正式模式只 reveal：

- `final_selected_stage_b_weight`
- `final_selected_stage_b_bias`

correctness 模式还可显式 reveal teacher、local heads、两个候选、候选分数、
reference weights 和 selection bit。所有额外 reveal 均写入 JSON，且 correctness
输出不得作为正式安全结果。

## 固定点与原子更新

2048 维内积按 64 维分块，每块独立执行 ABY3 乘法与截断后再求和。
TFE 0.8.0 的多个 `assign` 之间缺少跨变量同步；W/b 同时更新会产生读取旧值与
写入新值交错的竞态。`secure_simultaneous_assign` 先物化所有新 replicated
shares，再统一提交底层 share 赋值。该修复不 reconstruct、不 reveal。

三次独立多头更新测试的最大权重误差均低于 `3.1e-5`；完整 local/KD/候选选择
链路相对 NumPy correctness reference 的最大误差低于 `9.6e-5`。

## 声明限制

- 当前是单机多逻辑参与方模拟，不等同于 LAN/WAN 部署；
- 当前未测得真实网络通信 bytes；
- 默认会释放最终 Stage-B initializer；
- 必须成功加载针对 TF 1.15.5 编译的安全随机自定义算子，否则硬失败；
- 不存在静默明文 fallback。

# New Secure Stage C Privacy Definition

## 用途

定义新 Stage C 的 DP 记录单位、隐私会计范围、分布式噪声和 reveal 边界。

## 两种隐私范围

- `stage_a_original_record_privacy`：继承自 DP expert trajectory / DP-MTT；
- `stage_c_distilled_feature_record_privacy`：Stage C 对蒸馏特征记录执行
  DP-SGD 得到的 `(epsilon_C, delta_C)`。

Stage B/C 只读取 Stage A 的 DP 输出时，相对于 Stage A 原始记录属于后处理。
代码不会无依据地报告 `epsilon_total = epsilon_A + epsilon_C`。

## Stage C 机制

- 模型：2048 维 linear classifier；
- loss：安全友好的 squared loss；
- per-example gradient：由线性模型解析计算；
- clipping：norm、factor、gradient sum 均保持 ABY3 secret-shared；
- gradient query：Gaussian noise multiplier；
- clipping-count query：独立 Gaussian noise multiplier；
- accountant：Opacus `RDPAccountant` 顺序组合两个 subsampled Gaussian query；
- record：一条 distilled 2048-D feature；
- sample rate：`batch_size / distilled_feature_count`；
- delta：配置文件显式给出。

## 分布式噪声

三个逻辑计算方分别持有 checkpointed CSPRNG seed 和 counter，各自生成
`N(0, 1/3)` Gaussian share。三份输入进入 ABY3 后求和为 `N(0,1)`，
再在秘密共享域内乘以 `noise_multiplier * C_t / batch_size`。

没有使用“单次 NumPy 总噪声采样后再 secret-share”的伪 MPC 实现。安全假设是
三个计算方中至少一个诚实且不泄露自己的 seed/share。

## Reveal

- 正式训练释放最终 DP model；
- 调参/correctness 可释放 DP-noised clipping fraction 和聚合 reference metric；
- 不释放单样本 gradient、norm、clip factor 或单方 noise share；
- correctness 模式的额外 reveal 必须写入 JSON，不能作为正式安全结果。

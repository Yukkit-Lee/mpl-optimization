# New Secure Adaptive DP-SGD Validation

## 用途

记录新 Stage C adaptive clipping 的算法顺序、稳定性保护和正式调参边界。

## 更新顺序

每一步严格执行：

1. 读取当前 secret-shared `C_t`；
2. 使用 `C_t` 裁剪当前 per-example gradients；
3. 聚合裁剪梯度并加入三方 Gaussian noise shares；
4. 计算 `b_t = fraction(norm <= C_t)`；
5. 对 `b_t` 加入独立 Gaussian count noise；
6. 计算 `C_{t+1} = C_t * exp(-eta_c * (b_t_tilde - gamma))`；
7. 原子提交新 W、b、`C_{t+1}`；
8. 下一步才使用 `C_{t+1}`。

一步 correctness 中，使用旧 C 的 factor 误差为 `0.00153`，安全更新并提交后的
`C_{t+1}` 误差为 `5.1e-5`。

## 稳定性保护

- `C_min` 与 `C_max`；
- warm-up steps；
- update interval；
- 连续撞击 `C_max` 的 exploding-C hard stop；
- checkpoint 保存当前 C、history、连续撞界次数、模型、CSPRNG seed/counter；
- global clipping；
- parameter-block clipping（W 与 b 分别计算 factor）。

Adaptive 参数只能在 synthetic reference set 上调节。test set 不能进入 pilot、
候选选择、early stop 或超参数搜索。

## 近似与会计

- `inv_sqrt`、comparison、maximum 和 exp 均在 TFE/ABY3 中执行；
- exp 使用已编译并验证的 TFE aux custom op；
- gradient query 与 clipping-count query 在 Opacus RDP accountant 中顺序组合；
- Fixed/Adaptive 正式公平比较必须固定 sample rate、steps、delta，并匹配 epsilon；
- 未达到 Adaptive +1 percentage point 与 EarlyAUC 门槛时标记
  `NOT_READY_FOR_PAPER`，不得隐藏负结果。

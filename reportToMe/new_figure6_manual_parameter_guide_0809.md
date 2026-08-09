# Figure 6 Fixed/Adaptive 手动调参指南

## 1. 只需要关注的三个代码文件

1. 参数入口：`new_scripts/new_paper_adaptive/new_user_editable_figure6_config.py`
2. 实验入口：`new_scripts/new_paper_adaptive/new_run_user_configurable_fixed_vs_adaptive.py`
3. 绘图入口：`new_scripts/new_paper_figures/new_build_figure6.py`

参数入口已经用中文逐项解释。Fixed 和 Adaptive 是两套完全独立的字典；代码不再
锁定或复用旧 Fixed 结果。唯一强制相同的是 `target_epsilon` 和 `delta`，两边都
读取同一数据、initializer、development-validation split 和同一组 RNG seed。

修改后可以先做零训练配置检查：

```bash
/root/autodl-tmp/new_envs/distillation/bin/python \
  new_scripts/new_paper_adaptive/new_run_user_configurable_fixed_vs_adaptive.py \
  --check-only
```

底层单候选训练和隐私会计分别位于
`new_screen_latest_initializer_fixed_vs_adaptive.py` 的 `run_candidate()` 与
`new_stage_c_privacy_calibration.py`。一般不需要修改这两个底层文件；所有应由你
调整的实验参数都已集中到参数入口，避免把算法实现和超参数混在一起。

## 2. 建议的手动调参顺序

先把 `RNG_SEEDS` 临时改成 `[17]`，并给每次输出换一个不重复的文件名。分别调整：

1. Fixed：先试 `steps × learning_rate × expected_batch_size`，再试 `clip_c`；
2. Adaptive：先试 `steps × learning_rate × expected_batch_size`；
3. 再试 `clip_c(C0) × gamma × eta_c`；
4. 最后试 `warmup_steps × update_interval × count_noise_multiplier`。

每次改变 steps、batch 或 Adaptive 的 count 发布次数后，训练器都会重新计算噪声
倍数，使最终 epsilon 仍为目标值。不要直接比较不同 epsilon 的结果。

## 3. 快速单 seed 诊断

先检查资源并在 tmux 中启动：

```bash
cd /root/autodl-tmp/tfe_experiments
df -h /root/autodl-tmp
free -h
tmux new -s figure6_manual_tune
/root/autodl-tmp/new_envs/distillation/bin/python \
  new_scripts/new_paper_adaptive/new_run_user_configurable_fixed_vs_adaptive.py \
  --workers 2 2>&1 | tee new_log/new_paper_adaptive/txt/new_user_tuned_attempt.log
```

单 seed 结果只能用于选参数，不能在论文里报告方差或稳定提升。

## 4. 正式三 seed

选定参数后，将 `RNG_SEEDS` 改回 `[42, 43, 44]`，更换输出 JSON/CSV 文件名，
再运行：

```bash
cd /root/autodl-tmp/tfe_experiments
tmux new -s figure6_formal
/root/autodl-tmp/new_envs/distillation/bin/python \
  new_scripts/new_paper_adaptive/new_run_user_configurable_fixed_vs_adaptive.py \
  --workers 6 2>&1 | tee new_log/new_paper_adaptive/txt/new_user_tuned_formal.log
```

## 5. 绘制你的 Figure 6

将 `--input` 改成配置文件中 `OUTPUT_JSON` 的值：

```bash
cd /root/autodl-tmp/tfe_experiments
/root/autodl-tmp/new_envs/distillation/bin/python \
  new_scripts/new_paper_figures/new_build_figure6.py \
  --input new_log/new_paper_adaptive/new_user_tuned_fixed_vs_adaptive_formal.json \
  --output-stem new_figure6_user_tuned \
  --table new_expTab/new_paper_adaptive/new_figure6_user_tuned_summary.csv
```

生成：

- `new_expFig/paper_experiment/new_figure6_user_tuned.png`
- `new_expFig/paper_experiment/new_figure6_user_tuned.pdf`
- `new_expTab/new_paper_adaptive/new_figure6_user_tuned_summary.csv`

绘图不验证 JSON 哈希，也不要求 Fixed 冻结。它只检查 JSON 可解析、两方法 seed
一致、epsilon 一致、没有读取 test/holdout。n=1 时不会伪造误差条，n>=2 时使用
真实 seed 样本标准差。

## 6. 参数含义和注意事项

| 参数 | 含义 | 常见影响 |
|---|---|---|
| `steps` | 私有更新次数 | 更多步不一定更准；相同 epsilon 下通常要求更大噪声 |
| `learning_rate` | 学习率 | 太小欠拟合，太大可能震荡 |
| `expected_batch_size` | Poisson 期望 batch | 同时改变训练动态、采样率和隐私会计 |
| `clip_c` | Fixed 的恒定 C，或 Adaptive 的初始 C0 | 太小会严重裁剪，太大使噪声绝对尺度增大 |
| `gamma` | Adaptive 的目标 noisy fraction | 决定 C 上下调整的目标点 |
| `eta_c` | Adaptive 的 C 更新速度 | 太大会震荡，太小可能来不及适应 |
| `warmup_steps` | 开始更新 C 前的等待步数 | 防止训练初期统计不稳定 |
| `update_interval` | 私有更新 C 的间隔 | 越小发布越频繁、count 隐私开销越高 |
| `count_noise_multiplier` | 私有 count 的噪声倍数 | 越大 count 更隐私但控制信号更嘈杂 |
| `c_min/c_max` | Adaptive C 的允许范围 | 防止 C 数值塌缩或爆炸 |

最终论文必须说明这是“相同隐私预算、分别调参”的 best-configuration 比较，而非
matched steps/batch/LR 的单因素因果消融。调参只看 development validation；禁止
使用 official test 或 locked holdout 选择参数。

如果看到 `ModuleNotFoundError: No module named 'opacus'`，说明命令误用了
`/root/miniconda3/bin/python`。不要在base环境重复安装依赖，直接使用上面明确写出的
`/root/autodl-tmp/new_envs/distillation/bin/python`；该环境已验证包含
`opacus==1.5.4`。

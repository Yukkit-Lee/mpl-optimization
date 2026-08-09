# CIFAR-100 Figure 6调参与Tiny End-to-End补seed运行指南

更新时间：2026-08-10。以下命令均从仓库根目录
`/root/autodl-tmp/tfe_experiments`执行。本轮只准备入口，没有启动训练。

## 1. CIFAR-100 Fixed/Adaptive自定义参数

只修改这个文件：

`new_scripts/new_paper_adaptive/new_user_editable_figure6_cifar100_config.py`

Fixed与Adaptive参数完全分开，可以分别调整：

- `steps`：私有更新步数；
- `learning_rate`：学习率；
- `expected_batch_size`：Poisson期望batch；
- `clip_c`：Fixed的固定C，或Adaptive的初始C0；
- Adaptive的`gamma`、`eta_c`、`warmup_steps`、`update_interval`、
  `count_noise_multiplier`、`c_min/c_max`。

两种方法的`target_epsilon`和`delta`应保持一致。每次改变steps、batch或count
发布次数后，runner都会重新校准噪声，不需要手算noise multiplier。

### 1.1 零训练配置检查

```bash
cd /root/autodl-tmp/tfe_experiments
/root/autodl-tmp/new_envs/distillation/bin/python \
  new_scripts/new_paper_adaptive/new_run_user_configurable_fixed_vs_adaptive.py \
  --dataset-config cifar100 --check-only
```

如果误用`/root/miniconda3/bin/python`，会因缺少Opacus而失败；不要在base环境重复
安装，直接使用上述distillation解释器。

### 1.2 快速单seed调参

先在CIFAR-100配置文件中把`RNG_SEEDS`临时改成`[17]`，并修改
`OUTPUT_JSON/OUTPUT_CSV`，避免覆盖其他尝试。随后执行：

```bash
cd /root/autodl-tmp/tfe_experiments
df -h /root/autodl-tmp
free -h
tmux new-session -d -s figure6_c100_tune -c /root/autodl-tmp/tfe_experiments \
  '/root/autodl-tmp/new_envs/distillation/bin/python new_scripts/new_paper_adaptive/new_run_user_configurable_fixed_vs_adaptive.py --dataset-config cifar100 --workers 2 2>&1 | tee new_log/new_paper_adaptive/txt/new_user_tuned_cifar100_attempt.log'
tmux attach -t figure6_c100_tune
```

单seed只用于挑参数，不能报告稳定性或绘制误差条。

### 1.3 正式三seed

选定参数后，把`RNG_SEEDS`恢复为`[42, 43, 44]`，输出路径建议保持默认的
`...cifar100_formal.json/csv`，然后执行：

```bash
cd /root/autodl-tmp/tfe_experiments
df -h /root/autodl-tmp
free -h
tmux new-session -d -s figure6_c100_formal -c /root/autodl-tmp/tfe_experiments \
  '/root/autodl-tmp/new_envs/distillation/bin/python new_scripts/new_paper_adaptive/new_run_user_configurable_fixed_vs_adaptive.py --dataset-config cifar100 --workers 6 2>&1 | tee new_log/new_paper_adaptive/txt/new_user_tuned_cifar100_formal.log'
tmux attach -t figure6_c100_formal
```

正式输出：

- `new_log/new_paper_adaptive/new_user_tuned_fixed_vs_adaptive_cifar100_formal.json`
- `new_expTab/new_paper_adaptive/new_user_tuned_fixed_vs_adaptive_cifar100_formal.csv`

### 1.4 用新CIFAR-100结果生成候选Figure 6

先生成独立候选图，不覆盖当前论文图：

```bash
cd /root/autodl-tmp/tfe_experiments
/root/autodl-tmp/new_envs/distillation/bin/python \
  new_scripts/new_paper_figures/new_build_figure6.py \
  --cifar100-override new_log/new_paper_adaptive/new_user_tuned_fixed_vs_adaptive_cifar100_formal.json \
  --output-stem new_figure6_user_tuned_three_datasets_c100_candidate \
  --table new_expTab/new_paper_adaptive/new_figure6_user_tuned_three_datasets_c100_candidate.csv
```

确认候选图后，如需替换当前论文Figure 6，再把`--output-stem`改为
`new_figure6_user_tuned_three_datasets`，并把`--table`改回当前正式summary路径。
绘图入口不校验JSON哈希，但仍检查COMPLETE、epsilon、seed和test/holdout边界。

## 2. Tiny End-to-End seed43/44

只读审计结果：Tiny True Ours seed43/44的Stage B、Stage C和timeline已经存在；
缺失的是Full Data seed43/44。因此不要重跑旧18-cell总队列，也不要重跑True Ours。

专用入口：

`new_scripts/new_true_ours/new_run_tiny_end2end_extra_seeds.py`

它会先验证已有True Ours两条链，再顺序补Full Data seed43、44。两个大型TFE任务
不并行，以避免TF1图和缓存的内存峰值相互叠加；断联后执行同一命令会根据完整报告
跳过已完成阶段并继续。

### 2.1 正式运行前检查

```bash
cd /root/autodl-tmp/tfe_experiments
df -h /root/autodl-tmp
free -h
/root/autodl-tmp/new_envs/distillation/bin/python \
  new_scripts/new_true_ours/new_run_tiny_end2end_extra_seeds.py \
  --preflight-only
```

### 2.2 正式补齐两个seed

```bash
cd /root/autodl-tmp/tfe_experiments
tmux new-session -d -s tiny_end2end_extra_seeds -c /root/autodl-tmp/tfe_experiments \
  '/root/autodl-tmp/new_envs/distillation/bin/python new_scripts/new_true_ours/new_run_tiny_end2end_extra_seeds.py --seeds 43 44 2>&1 | tee new_log/new_true_ours/txt/formal_corrected_v2/new_tiny_end2end_extra_seeds_runner.log'
tmux attach -t tiny_end2end_extra_seeds
```

查看进度而不进入tmux：

```bash
tmux capture-pane -pt tiny_end2end_extra_seeds -S -120
tail -n 80 new_log/new_true_ours/txt/formal_corrected_v2/new_tiny_end2end_extra_seeds_runner.log
```

只跑其中一个seed时，可将正式命令改成`--seeds 43`或`--seeds 44`。完整结果写入
`full_data_formal_corrected_v2/tiny/seed43`和`seed44`独立目录，checkpoint继续遵守
`new_checkpoints/`本地保存、不提交Git的规则。

完成后需要重新聚合Tiny三seed到TFE master，再重画Figure 2(a)；不能只把新点手工
塞进PNG。聚合时必须同时纳入已有True Ours seed43/44与新Full Data seed43/44，使用
三seed样本标准差，并把旧`14_OF_14`状态更新为真实18-cell状态。

# Module C Fixed-DP / Adaptive-DP 消融参数与运行指南

## 1. 原消融是否使用相同参数

是。旧正式 Module C 消融对同一个数据集的 Fixed-DP 与 Adaptive-DP 使用了
相同的更新步数、batch、学习率和初始/固定裁剪阈值：

| 数据集 | Fixed steps | Adaptive steps | batch | Fixed LR / C | Adaptive LR / C0 |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 10 | 10 | 16 | 0.002 / 1.0 | 0.002 / 1.0 |
| CIFAR-100 | 20 | 20 | 16 | 0.002 / 1.0 | 0.002 / 1.0 |
| Tiny ImageNet | 30 | 30 | 16 | 0.002 / 1.0 | 0.002 / 1.0 |

两者共享 seed 42/43/44、epsilon={1,3,8} 和 delta=1e-5。Adaptive 额外使用
count-noise=8、warmup=5、update-interval=5；它的 count release 会被隐私会计
组合，因此相同 epsilon 下两种方法的梯度 noise multiplier 本来就会略有不同。

旧设计回答的是“同预算下Adaptive更新C是否优于固定C”，不是“分别调到最佳后
谁的最高准确率更高”。

## 2. 参数在哪里修改

主参数文件：

`new_configs/new_public_mtt_mainline/new_mainline_config.json`

进入 `module_c_formal.mode_parameters`：

- `fixed_dp.datasets.<dataset>`：分别修改 Fixed 的 `steps`、`batch_size`、
  `learning_rate`、`clip_c`；
- `adaptive_dp.datasets.<dataset>`：分别修改 Adaptive 的 `steps`、`batch_size`、
  `learning_rate`、`clip_c`（这里是初始C0）；
- `adaptive_dp` 下还可修改 `count_noise_multiplier`、`warmup_steps`、
  `update_interval`、`gamma`、`eta_c`、`c_min`、`c_max`。

不要手工填写 `noise_multiplier`。队列生成器会根据每种方法自己的 steps、batch、
count release次数以及目标epsilon重新运行RDP校准。

## 3. 实际执行代码

1. 队列与隐私参数生成器：`new_prepare_module_c_formal.py`；
2. 单条 TFE/ABY3 Fixed/Adaptive 训练：`new_stage_c_secure_dpsgd.py`；
3. 防断联串行任务执行器：`new_scripts/new_run_task_queue.sh`；
4. 结果聚合与图表：`new_report_module_c_formal.py`。

以上Python文件的入口、参数含义和可修改范围已添加中文注释。

## 4. 运行方式

每一套新参数都使用一个全新英文 `run-tag`。先生成队列，不加 `--activate`：

```bash
/root/autodl-tmp/new_envs/distillation/bin/python \
  new_prepare_module_c_formal.py --run-tag user_ablation_v1
```

这里使用现代环境是因为生成器需要 Opacus 重新校准隐私噪声；普通系统Python
可以查看 `--help`，但若没有安装Opacus则不能生成正式队列。

生成文件：

```text
new_configs/new_public_mtt_mainline/new_tfe_module_c_user_ablation_v1.tsv
new_log/new_public_mtt_mainline/json/new_module_c_user_ablation_v1_plan.json
```

确认plan中的参数后，在tmux运行：

```bash
tmux new -s module_c_user_ablation
bash new_scripts/new_run_task_queue.sh tfe \
  new_configs/new_public_mtt_mainline/new_tfe_module_c_user_ablation_v1.tsv
```

按 `Ctrl-b` 再按 `d` 离开tmux但保留进程。重新进入：

```bash
tmux attach -t module_c_user_ablation
```

全部任务完成后聚合：

```bash
/root/autodl-tmp/new_envs/distillation/bin/python \
  new_report_module_c_formal.py --run-tag user_ablation_v1
```

## 5. 如何解释不同参数

- 若要做严格“同预算消融”，应保持两者 epsilon、delta、数据、初始化、seed、
  steps和batch相同；可以分别预调 Fixed-C、Adaptive C0/gamma/eta，学习率也可
  分别预调，但必须使用同一预注册搜索范围和选择规则。
- 若为了展示各自最高效果而令 steps/batch 不同，可以执行，但应称为
  “separately tuned / best-config comparison”，同时报告各自steps、时间和采样率，
  不能再写成同训练预算下Adaptive优于Fixed。
- official test和locked holdout不能参与参数选择；原Module C使用test-free
  reference split。epsilon只覆盖该Module C协议声明的范围。

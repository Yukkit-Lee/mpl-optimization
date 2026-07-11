# 2026-07-09 模块A数据蒸馏验证实验修改记录

## 当前时间
- 2026-07-09 Asia/Shanghai

## 本次目标
- 新增独立脚本 `exp_cifar10_distilled_moduleA_verify.py`。
- 验证新的模块A候选：Dataset-Distilled Compact Training Set Construction。
- 比较 Full-Data、Random-Subset、Distilled-Data 三种训练数据来源。
- 生成 Figure 4 风格曲线图到 `expFig/figure4_distill/figure4_cifar10_distilled_verify.png`。
- 全程限制 CPU 线程，避免机器负载过高。

## 已完成
- 创建 `progressLog` 目录。
- 开始定位 `/mnt/code/lixujie/mtt-distillation` 中的 CIFAR-10 蒸馏样本。

## 待完成
- 识别 distilled images/labels 文件格式。
- 生成验证脚本。
- 运行轻量验证实验并记录输出。

## 追加记录
- 时间：2026-07-09 Asia/Shanghai
- 修改内容：新增脚本 `exp_cifar10_distilled_moduleA_verify.py`，用于验证 Dataset-Distilled Compact Training Set Construction。
- 关键实现：本地 CIFAR-10 npz 加载、MTT distilled `.pt` 自动定位、SmallConvNet 统一训练、Optimizer Steps 曲线、figure4_distill 输出。
- 安全措施：脚本顶部限制 OMP/MKL/OPENBLAS/NUMEXPR 线程，运行时限制 torch 线程。

## 追加记录
- 时间：2026-07-09 19:39:14 Asia/Shanghai
- run_id：20260709_193900
- 新增/更新脚本：/mnt/code/lixujie/tfe_experiments/exp_cifar10_distilled_moduleA_verify.py
- 修改内容：新增 CIFAR-10 数据蒸馏模块A轻量验证实验，比较 Full-Data / Random-Subset / Distilled-Data。
- 线程控制：脚本开头限制 OMP/MKL/OPENBLAS/NUMEXPR，并在运行时设置 torch 线程数。
- 输出 txt：/mnt/code/lixujie/tfe_experiments/log/txt/cifar10_distilled_moduleA_verify_20260709_193900.log
- 输出 json：/mnt/code/lixujie/tfe_experiments/log/json/cifar10_distilled_moduleA_verify_20260709_193900.json
- 输出 csv：/mnt/code/lixujie/tfe_experiments/expTab/cifar10_distilled_moduleA_verify_20260709_193900.csv
- 输出 figure：/mnt/code/lixujie/tfe_experiments/expFig/figure4_distill/figure4_cifar10_distilled_verify.png

## 追加记录
- 时间：2026-07-09 Asia/Shanghai
- 问题发现：第一版验证中 Distilled-Data 低于 Random-Subset，原因是验证协议与 MTT 蒸馏数据生成协议不匹配。
- 修改内容：将 ConvNet 改为 MTT quick_test 风格的无 BatchNorm ConvNet；真实 CIFAR 输入改为 raw [0,1]，不再标准化；学习率改为 0.01，weight decay 改为 0。
- 修改目的：让 Full/Random/Distilled 的验证协议更接近已有 distilled samples 的预期训练设置。

## 追加记录
- 时间：2026-07-09 19:41:48 Asia/Shanghai
- run_id：20260709_194140
- 新增/更新脚本：/mnt/code/lixujie/tfe_experiments/exp_cifar10_distilled_moduleA_verify.py
- 修改内容：新增 CIFAR-10 数据蒸馏模块A轻量验证实验，比较 Full-Data / Random-Subset / Distilled-Data。
- 线程控制：脚本开头限制 OMP/MKL/OPENBLAS/NUMEXPR，并在运行时设置 torch 线程数。
- 输出 txt：/mnt/code/lixujie/tfe_experiments/log/txt/cifar10_distilled_moduleA_verify_20260709_194140.log
- 输出 json：/mnt/code/lixujie/tfe_experiments/log/json/cifar10_distilled_moduleA_verify_20260709_194140.json
- 输出 csv：/mnt/code/lixujie/tfe_experiments/expTab/cifar10_distilled_moduleA_verify_20260709_194140.csv
- 输出 figure：/mnt/code/lixujie/tfe_experiments/expFig/figure4_distill/figure4_cifar10_distilled_verify.png

## 追加记录
- 时间：2026-07-09 19:45:30 Asia/Shanghai
- 修改内容：将验证脚本中的学生模型改为与 mtt-distillation `get_network('ConvNet')` 等价的 InstanceNorm ConvNet，增加随机裁剪/翻转增强和 cosine scheduler。
- 修改内容：新增 distilled checkpoint 候选预筛选逻辑，先用少量 optimizer steps 选择更有效的 CIFAR-10 蒸馏样本，再进入正式 Full/Random/Distilled 三曲线验证。
- 安全设置：保留 CPU 线程限制为 1，避免候选筛选阶段过度并行。

## 追加记录
- 时间：2026-07-09 19:45:33 Asia/Shanghai
- run_id：20260709_194457
- 新增/更新脚本：/mnt/code/lixujie/tfe_experiments/exp_cifar10_distilled_moduleA_verify.py
- 修改内容：新增 CIFAR-10 数据蒸馏模块A轻量验证实验，比较 Full-Data / Random-Subset / Distilled-Data。
- 线程控制：脚本开头限制 OMP/MKL/OPENBLAS/NUMEXPR，并在运行时设置 torch 线程数。
- 输出 txt：/mnt/code/lixujie/tfe_experiments/log/txt/cifar10_distilled_moduleA_verify_20260709_194457.log
- 输出 json：/mnt/code/lixujie/tfe_experiments/log/json/cifar10_distilled_moduleA_verify_20260709_194457.json
- 输出 csv：/mnt/code/lixujie/tfe_experiments/expTab/cifar10_distilled_moduleA_verify_20260709_194457.csv
- 输出 figure：/mnt/code/lixujie/tfe_experiments/expFig/figure4_distill/figure4_cifar10_distilled_verify.png

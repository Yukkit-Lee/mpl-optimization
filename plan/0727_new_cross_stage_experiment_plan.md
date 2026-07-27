# Formal Experimental Plan for Cross-Stage DP-MPL Framework

## Hardware Environment

Server: - GPU: NVIDIA RTX 5090 - CPU: 25 vCPU Intel(R) Xeon(R) Platinum
8470Q

要求： - 充分发挥 GPU 性能； - 支持多 GPU 时优先使用； -
根据显存自动调整 batch size； - 监控 GPU 利用率和显存占用； -
不人为降低模型规模、epoch 或实验次数。

CPU： - 不限制线程数量； -
允许充分利用多线程、多进程进行数据加载和预处理； -
避免异常线程爆炸导致系统不稳定。

长时间任务必须使用 tmux/screen 保持运行。

------------------------------------------------------------------------

# File Rule

所有新增文件添加：

`new_`

前缀。

不要覆盖旧实验。

------------------------------------------------------------------------

# Experiment 1: End-to-End Main Experiment

目标： 证明整体跨阶段优化框架优于传统安全训练路线。

比较：

1.  Full Data ConvNet-DP

Full Dataset → ConvNet → Secure DP Training

2.  Full Feature-DP

Full Dataset → Shared Encoder → 2048 Feature → Secure DP Training

3.  Ours

Official MTT Distilled Dataset → Shared Encoder → 2048 Feature → Secure
Consensus Initialization → Secure DP Training

数据集： - CIFAR-10 - CIFAR-100 - Tiny ImageNet

至少 3 seeds。

指标： - Accuracy - Training Time - Steps@Target Accuracy - Memory -
Processed Samples

生成 Figure 2 风格曲线。

------------------------------------------------------------------------

# Experiment 2: Stage A Dataset Distillation

目标： 证明蒸馏数据降低安全训练数据量。

方法：

-   Full Dataset
-   Random Subset
-   Official MTT Distilled Dataset

指标：

-   Accuracy
-   Compression Ratio
-   Steps@Target Accuracy
-   Training Time

生成 Accuracy vs Iteration 曲线。

------------------------------------------------------------------------

# Experiment 3: Stage B Initialization

目标： 证明初始化策略提升收敛速度。

方法：

-   Random Initialization
-   Average Initialization
-   Secure Consensus Initialization

指标：

-   Accuracy curve
-   Early AUC
-   Steps@Target Accuracy

------------------------------------------------------------------------

# Experiment 4: Stage B Non-IID

划分：

-   IID
-   Mild Non-IID
-   Moderate Non-IID
-   Severe Non-IID

方法：

-   NoKD
-   KD-Average
-   KD-Consensus

输出准确率表格。

------------------------------------------------------------------------

# Experiment 5: Stage C DP Evaluation

测试：

epsilon: - 1 - 3 - 5 - 8

记录：

-   Accuracy
-   epsilon
-   Runtime
-   Convergence

------------------------------------------------------------------------

# Experiment 6: External Comparison

生成论文表：

-   SecureML
-   SecureNN
-   CryptGPU
-   CrypTen
-   PEA
-   Ours

比较： MPC、DP、模型类型、Non-IID支持、隐私保证。

------------------------------------------------------------------------

# Experiment 7: CryptGPU-style Efficiency

比较：

Direct Secure CNN Training

vs

Ours Compact Feature Training

指标：

-   Time
-   Iterations
-   Computation Cost

目标： 证明减少安全计算成本。

------------------------------------------------------------------------

# Privacy Scope

Stage A: epsilon=N/A

Stage B: MPC confidentiality

Stage C: 报告 epsilon、delta、accountant。

不要声称完整pipeline record-level DP，除非所有阶段满足。

------------------------------------------------------------------------

# Quality Requirements

禁止： - smoke test作为正式结果； - 单seed； - 使用test set调参。

必须： - 保存checkpoint； - 保存history； - 保存csv/json； -
自动生成论文图表。

结果不好时： 检查模型、学习率、batch
size、optimizer、epoch，不要直接降低实验规模。

------------------------------------------------------------------------

# Git

每完成主要实验：

git add . git commit git push

------------------------------------------------------------------------

# Final Report

输出：

1.  实验状态；
2.  图表路径；
3.  CSV/JSON路径；
4.  支持论文claim；
5.  当前限制。

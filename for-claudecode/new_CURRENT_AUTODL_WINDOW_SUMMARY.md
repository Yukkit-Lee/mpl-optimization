# 当前 AutoDL 窗口对话摘要

## 1. 本窗口从哪里开始

本窗口开始时，论文项目刚从旧内网服务器迁移到AutoDL。用户要求先恢复
环境和资产，不允许启动Stage C或正式长实验，并要求：

1. 完整读取迁移交接、项目进度、AutoDL迁移指南和任务计划；
2. 检查硬件、Git分支、迁移包SHA256和磁盘；
3. 严格执行`AUTODL_MIGRATION.md`第1–7节；
4. 不重跑已完成的Stage A和Stage B；
5. 每次修改更新进度日志并提交推送GitHub。

私有旧对话JSONL被明确限定为“只有交接文档不足时，才用`rg`按关键词定位”，
禁止整文件读取，禁止泄露敏感信息。

## 2. 代理与下载源的连续指令

用户多次校正网络策略：

1. 配环境需要代理；
2. 用户个人代理端口为`9999`，可以通过环境变量确认；
3. 官方海外源可能使用个人代理更快；
4. 阿里或其他国内源使用个人代理反而可能更慢；
5. 下载慢时优先尝试阿里源；
6. 后续用户明确要求个人`9999`代理非必要不要使用。

最终形成的策略：

```text
国内镜像/国内资源：优先直连或AutoDL国内网络
官方海外源：确有需要时才考虑个人代理
不要输出代理认证信息
```

## 3. 用户提供的本地资产

用户说明AutoDL本地已经存在：

1. Official MTT蒸馏数据ZIP；
2. CIFAR-10原始档案；
3. CIFAR-100原始档案；
4. Tiny ImageNet原始档案。

因此停止重复网络下载，删除重复下载的副本/临时文件以节省空间，并使用用户
已上传原件完成哈希和资产验证。

关键结果：

```text
Official MTT non-ZCA tensors: 20/20通过
Raw archives: 3/3通过
重复网络下载和临时文件已清理
```

没有删除用户上传的规范原件。

## 4. GitHub认证和推送

用户把GitHub token放入本机配置文件，允许用于push。安全边界：

1. 只允许Git credential helper在push时读取；
2. 禁止`cat`或打印token；
3. token不得进入命令输出、日志、文档或Git。

当前分支持续为：

```text
new-secure-feature-pipeline
```

本窗口所有阶段性文档和实验提交均已推送。

## 5. AutoDL环境恢复结果

环境恢复阶段完成：

### 现代环境

```text
Python 3.9
PyTorch 2.8.0+cu128
torchvision 0.23.0+cu128
Opacus 1.5.4
```

### legacy TFE环境

```text
Python 3.6.13
TensorFlow 1.15.5
TF-Encrypted 0.8.0
NumPy 1.18.5
```

### TFE custom ops

`aux`与`secure_random`在AutoDL重新构建并通过真实加载/correctness验证。
新主机编译产物SHA与旧服务器不同属于正常现象，以当前主机加载和数值验证为准。

### MTT源码

从迁移bundle恢复MTT基线提交、tracked patch和未跟踪源码。路径改为AutoDL
可配置`NEW_*`变量，不能把旧服务器绝对路径重新写入可执行代码或队列。

恢复阶段没有重跑Stage A或Stage B。

## 6. 用户对正式主线的改变

环境恢复后，用户指定继续以：

```text
plan/new_public_mtt_mainline_codex_prompt.md
```

为任务主线，并给出两个重要变化：

1. 不做DP-MTT实验；
2. 不再把线程固定为1，要发挥最大合理性能。

这覆盖了旧迁移交接里的单核/单线程安全策略。当前规则是：

```text
不人为限制为1线程
允许合理多线程/多进程
避免线程爆炸或重复任务
长任务必须tmux/持久恢复
```

## 7. 本窗口完成的Public-MTT正式主线

在持久队列、断点恢复、heartbeat和严格冻结门下完成：

### Module A

已有正式结果复用并完成统一审计。

### Module B convergence

完成Random Initialization与Secure Consensus Initialization的3数据集×3 seeds
收敛对比。

### Stage B Non-IID

既有27/27正式结果复用：

```text
3 datasets × 3 seeds × IID/Mild/Severe
```

### Module C

完成54/54：

```text
3 datasets
× Fixed/Adaptive
× epsilon {1,3,8}
× seeds {42,43,44}
```

privacy accountant、模型状态、reference-only报告和最终test冻结门均通过。

Standalone Module C的Adaptive并未稳定优于Fixed，这个负结果被如实保留。

### End-to-End baseline initializer

完成9/9 Random-Subset + NoKD Secure-Average初始化。

过程中发现并修复：

1. 旧单线程环境变量残留；
2. checkpoint每epoch保存导致严重开销；
3. checkpoint interval改为5，同时保留恢复能力。

### End-to-End

完成18/18：

```text
Baseline:
Random Subset + Secure Average + Fixed

Ours:
Official MTT + Secure Consensus + Adaptive
```

Ours在三个数据集的reference与final test上均明显优于旧Random-Subset baseline。

Final test accuracy，baseline -> ours：

```text
CIFAR-10:  0.48960 -> 0.61040
CIFAR-100: 0.19723 -> 0.33590
Tiny:      0.05463 -> 0.13040
```

这些Ours Adaptive结果是当前增量计划要优先复用的关键资产。

### Final test

在Module C、End-to-End报告和全部模型冻结后进行只读test评估。测试结果没有
反馈给调参、质量门或seed选择。

### TFE system

完成9/9 small/medium/largest-feasible测量，记录secure runtime与process-tree
peak RSS。单机TFE simulation不暴露真实communication bytes/rounds，因此继续
诚实报告`N/A`。

## 8. 0727计划的两次变化

### 第一版0727计划

最初上传的计划范围较大，包括：

- 三条End-to-End路线；
- Average/Consensus初始化；
- Moderate Non-IID；
- epsilon=5；
- 外部系统表；
- CryptGPU-style直接安全CNN。

审计发现真正MPC secure CNN会成为高风险长任务，因此当时估计从数天到数周。

### 第二版增量计划

用户随后用：

```text
plan/0727_new_cross_stage_incremental_full_feature_plan.md
```

替换第一版。

范围收敛为：

1. 保留全部旧结果；
2. 主图只比较Full Data Feature Model与Ours；
3. 外部方法只做3–4个可靠来源；
4. 不复现CryptGPU；
5. 尽量复用已有Ours。

同时确认9份full feature缓存已存在：

```text
CIFAR-10:  3 × 50000 × 2048
CIFAR-100: 3 × 50000 × 2048
Tiny:      3 × 100000 × 2048
```

因此不需要重跑encoder。

## 9. Fixed到Adaptive的最新更正

在创建Claude Code初版交接后，用户明确纠正：

> 不使用Fixed DP-SGD，继续使用Adaptive DP；已有End-to-End效果不错，
> Standalone Stage C消融不理想可以后续调整。

所以当前有效主比较为：

```text
Full Data + Secure Average + Adaptive
vs
Official MTT + Secure Consensus + Adaptive
```

已有Ours Adaptive 9条曲线、模型、history和final-test优先复用。新增任务原则上：

```text
9个Full-Data Secure-Average Stage-B initializer
9个Full-Data Adaptive Stage-C run
```

不再补跑Ours Fixed。

Standalone 54-run Fixed/Adaptive负结果继续保留；如果未来调整Adaptive，必须走
独立test-free tuning/pilot、新配置和新报告，不能覆盖旧结果，也不能使用
End-to-End test反向调参。

## 10. 当前最重要的防错结论

1. Full Data训练必须排除shared reference/selection holdout；
2. Full Data accountant必须使用真实N，不能沿用旧`400/4000/8000`；
3. Full Data和Ours双方必须使用同一Adaptive规则；
4. gradient noise可因真实sample rate不同而不同，但target epsilon/delta相同；
5. Adaptive clipped-count mechanism必须计入composed accountant；
6. 旧final-test evaluator会覆盖旧路径，必须新建incremental evaluator；
7. Ours旧test结果优先复用，新流程只对Full Data新模型开启test门；
8. Stage-C time与包含Stage-B的total secure time必须分开报告；
9. 跨服务器Stage-B runtime不能声称严格同硬件speedup；
10. TFE secure training主要是CPU路径，GPU memory可能为0/N/A；
11. 不得把MPC confidentiality写成DP；
12. 两条End-to-End initializer均data-dependent non-DP，因此epsilon只是conditional
    gradient-mechanism accounting，不能声称完整end-to-end record DP。

## 11. 当前状态与下一步

当前文档和Git均已整理，但硬件检查显示：

```text
No devices found
CPU约0.5 core
```

硬件恢复前：

```text
只做静态审计、配置、validator、隔离脚本和测试
不启动pilot/formal
```

硬件恢复后：

1. 验证环境/custom ops/full feature资产；
2. 冻结与已有Ours完全匹配的Adaptive协议；
3. CIFAR-10 seed42与Tiny seed42 pilot；
4. 9个Full Data Stage-B；
5. 9个Full Data Adaptive Stage-C；
6. reference-only共同target；
7. 只读Full Data final test；
8. 合并复用Ours结果；
9. 生成两套Figure 2；
10. 整理外部方法表；
11. 每阶段更新进度、commit、push。

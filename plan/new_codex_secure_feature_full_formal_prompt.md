# Codex 连续执行任务：重建 DP 蒸馏特征 + 安全 Stage B + 安全 Stage C，并完成全部正式实验

## 0. 总目标

请在保留当前 `cross-stage-refactor` 分支和完整 ConvNet 结果的前提下，新建独立分支：

```bash
new-secure-feature-pipeline
```

在该分支中实现并验证下面的新主流程：

```text
Stage A:
Private Local Data
→ Differentially Private Expert Trajectory Generation
→ MTT Distilled Images
→ Shared/Frozen ConvNet Encoder
→ 2048-D Compact Distilled Features

Stage B:
Secret-Shared Distilled Features and Labels
→ Secure Teacher/Knowledge Aggregation
→ Secure KD-Compatible Linear Initialization
→ Secure Average / Reference-Weighted Candidate Construction
→ Secure Candidate Evaluation and Selection
→ Secret-Shared Global Initializer

Stage C:
Secret-Shared 2048-D Distilled Features
+ Secret-Shared Global Initializer
→ Secure Fixed DP-SGD / Secure Adaptive DP-SGD
→ Released Differentially Private Linear Model
```

本任务不是 smoke test。最终必须完成三数据集、多随机种子、完整训练曲线、均值与标准差、隐私会计、运行时间和可复现实验脚本。

---

# 1. 必须遵守的原则

1. **不要覆盖当前完整 ConvNet 主流程。**
   - 当前实现保留在原分支；
   - 新路线只在 `new-secure-feature-pipeline` 分支实现；
   - 旧结果与新结果使用不同目录和文件前缀。

2. **不要把 epsilon 作为标签硬加到 Stage A。**
   - Stage A 只有在真正执行了具有正式会计的 DP 机制后才能报告 epsilon；
   - 使用官网 MTT ipc=50 蒸馏图像的实验只能标为 `public_mtt_benchmark`，其 `stage_a_epsilon=N/A`；
   - 新的私有 Stage A 必须通过 DP expert trajectories 或其他可证明机制生成蒸馏图像。

3. **不得将 PyTorch/NumPy 计算称为 MPC。**
   - 只有秘密份额上的 TFE/ABY3 计算才能标记 `tfe_mpc`；
   - 任何 reveal 都必须记录；
   - MPC 失败不得静默回退到明文。

4. **不得使用 test set 调参。**
   - 所有超参数、early stopping、target accuracy、候选选择都使用 reference/validation 数据；
   - test set 只用于固定间隔只读评估和最终报告。

5. **不得把所有数据集设置成相同 steps。**
   - 训练预算使用 epochs 或 processed examples 定义；
   - CIFAR-10、CIFAR-100、Tiny ImageNet 必须根据样本数和收敛情况设置不同 iteration 终点。

6. **单个子任务受阻时不要停止整个任务。**
   - 记录 `BLOCKED`、原因、复现命令和替代实现；
   - 继续完成其余可执行任务；
   - 不得用不安全实现冒充已完成。

7. **本轮所有新生成的目录、文件、脚本、日志、checkpoint、图表、表格、manifest 和 tmux session 名称都必须以前缀 `new_` 开头。**
   - 新分支使用 `new-secure-feature-pipeline`；
   - 新文件示例：`new_exp_stage_a_dp_mtt.py`；
   - 新目录示例：`new_log/`、`new_expFig/`；
   - 新输出示例：`new_secureFeature_cifar10_seed42.json`；
   - 不得生成无 `new_` 前缀的新实验产物；
   - 修改已有源码时可保留原文件名，但优先通过新的 `new_` 入口调用，避免污染旧路线。

8. 每个 Phase 完成后：
   - 更新进度文件；
   - 运行单元测试和正式子实验；
   - 独立 Git commit；
   - 自动进入下一个 Phase，不等待确认。

---

# 2. 两张 RTX 4090 的资源策略与防崩溃运行要求

机器有两张 RTX 4090。首先探测：

```bash
nvidia-smi
nproc
free -h
df -h
```

然后创建：

```text
new_scripts/new_run_dual_gpu_formal_experiments.sh
new_scripts/new_create_tmux_formal_session.sh
new_scripts/new_resume_incomplete_runs.sh
new_scripts/new_monitor_resources.sh
```

## 2.1 CPU 线程必须严格为 1

无论使用一张还是两张 GPU，每个实验进程、TFE 进程和报告进程都必须限制为单 CPU 线程，避免机器死机。

每个命令执行前强制设置：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

Python 入口中同时调用：

```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

数据加载器统一：

```text
num_workers=0
persistent_workers=false
prefetch_factor=None
```

OpenCV 如被导入，调用：

```python
cv2.setNumThreads(0)
```

禁止创建 multiprocessing pool，禁止隐式并行数据预处理，禁止根据 `nproc` 自动增加线程。

## 2.2 GPU 调度

- GPU 0 和 GPU 1 各运行一个独立 PyTorch 训练进程；
- 使用：
  ```bash
  CUDA_VISIBLE_DEVICES=0
  CUDA_VISIBLE_DEVICES=1
  ```
- 按 `dataset × seed` 分配任务；
- 一张 GPU 同时最多一个正式训练进程；
- 使用独立锁：
  ```text
  /tmp/new_secure_feature_gpu0.lock
  /tmp/new_secure_feature_gpu1.lock
  ```
- 每个任务启动前检查显存、系统内存和磁盘；
- 显存不足时等待，不杀死其他进程；
- 可自动降低 microbatch，但必须通过 gradient accumulation 保持 effective batch size；
- 每个任务记录 GPU 型号、峰值显存、effective batch size、microbatch、accumulation steps 和运行时间；
- 建议对每个进程设置不超过约 90% 的显存使用上限，预留系统空间；
- 禁止两张卡上的任务共同启动额外 CPU worker。

## 2.3 tmux 持久运行与自动恢复

所有长实验必须在 tmux 中运行，不能依赖当前终端会话。

创建固定 session：

```text
new_secure_feature_formal
```

至少包含窗口：

```text
gpu0
gpu1
tfe
monitor
report
```

要求：

1. `new_create_tmux_formal_session.sh` 可重复执行；
2. session 已存在时不得重复启动相同任务；
3. 每个窗口执行独立队列；
4. 每个正式任务写：
   - PID 文件；
   - heartbeat 文件；
   - stdout/stderr 日志；
   - checkpoint；
   - manifest 状态；
5. 每 60 秒更新 heartbeat；
6. 进程异常退出时，自动从最近 checkpoint 恢复；
7. 同一任务最多自动重试 2 次；
8. tmux 不可用时才允许使用 `nohup` 作为后备，并记录该事实；
9. 不得因为 SSH 断开、Codex 会话结束或终端关闭而中断实验。

## 2.4 系统安全监控

`new_monitor_resources.sh` 每 60 秒记录：

```text
timestamp
GPU utilization
GPU memory
GPU temperature
CPU load
RAM usage
swap usage
disk free space
active PIDs
heartbeat age
```

启动新任务前必须满足：

- 可用内存不少于预设安全阈值；
- 磁盘剩余空间足以保存 checkpoint 和日志；
- GPU 温度和显存处于安全范围；
- 没有同名任务正在运行。

如系统内存持续高于 90%、swap 快速增长、磁盘接近满载或 heartbeat 超时：

- 暂停启动新任务；
- 保存当前 checkpoint；
- 优雅停止风险任务；
- 不得直接 `kill -9`，除非正常终止完全失败；
- 在 manifest 中记录原因。

## 2.5 TFE 任务

- TFE/ABY3 老环境若无法稳定使用 GPU，则使用 CPU；
- TFE 进程同样严格限制 CPU 线程为 1；
- 使用独立锁：
  ```text
  /tmp/new_secure_feature_tfe.lock
  ```
- 为避免 CPU/内存冲突，TFE 正式任务默认不与高内存 Tiny ImageNet GPU 任务同时启动；
- 是否并行由资源监控脚本决定，而不是盲目并发。

## 2.6 断点恢复

每个正式运行必须支持：

- epoch/step checkpoint；
- optimizer state；
- scheduler state；
- privacy accountant state；
- adaptive clipping state；
- random-number-generator state；
- run manifest；
- 已完成任务自动跳过；
- 中断后精确恢复，而不是从头重跑；
- 恢复后在日志中记录 checkpoint 来源和恢复 step。

---

# 3. 进度文件和结果目录

创建：

```text
new_SECURE_FEATURE_PIPELINE_PROGRESS.md
new_SECURE_FEATURE_EXPERIMENT_MANIFEST.json
```

Checklist：

```text
[ ] Phase 0: Preserve baseline and create branch
[ ] Phase 1: Implement real Stage-A privacy mechanism
[ ] Phase 2: Build shared 2048-D representation pipeline
[ ] Phase 3: Implement secure Stage-B computation
[ ] Phase 4: Implement secure Stage-C Fixed DP-SGD
[ ] Phase 5: Implement secure Stage-C Adaptive DP-SGD
[ ] Phase 6: Complete Module-A formal experiments
[ ] Phase 7: Complete Module-B formal experiments
[ ] Phase 8: Complete Module-C formal experiments
[ ] Phase 9: Complete end-to-end formal experiments
[ ] Phase 10: TFE system evaluation and final reporting
```

所有新输出使用前缀：

```text
new_secureFeature_
```

目录建议：

```text
new_log/new_secure_feature/json/
new_log/new_secure_feature/txt/
new_expTab/new_secure_feature/
new_expFig/new_secure_feature/
new_init_model/new_secure_feature/
new_cache/new_secure_feature/
```

---

# 4. Phase 0：保存当前结果并建立新路线

1. 确认当前分支与远端同步；
2. 保存当前正式 JSON/CSV/图表路径；
3. 创建 `new-secure-feature-pipeline`；
4. 将当前 feature pipeline 旧代码复制为参考，但不要直接恢复旧 bug；
5. 新实现必须使用共享模块，避免 Stage A/B/C 各自重新训练不同 encoder；
6. 记录环境版本：
   - Python；
   - PyTorch；
   - TensorFlow；
   - TF Encrypted；
   - CUDA；
   - cuDNN；
   - Opacus。

完成后自动进入 Phase 1。

---

# 5. Phase 1：为 Stage A 实现真正可会计的 DP 蒸馏

## 5.1 两条 Stage-A 轨道

必须同时保留：

### Track A：公开 MTT benchmark

```text
Official MTT ipc=50 images
→ Shared ConvNet Encoder
→ 2048-D Features
```

标记：

```text
stage_a_track=public_mtt_benchmark
stage_a_epsilon=N/A
```

作用：复现数据集蒸馏效用，不声称原始数据 DP。

### Track B：DP-MTT 路线

优先实现：

```text
Private Local Data
→ DP-SGD Expert Training
→ DP Expert Trajectories
→ MTT Optimization Using Only DP Trajectories
→ DP Distilled Images
```

关键要求：

1. Expert trajectory 生成使用 Opacus 或经过测试的 RDP accountant；
2. per-example clipping、noise multiplier、sample rate、steps、delta 全部记录；
3. MTT 蒸馏阶段不得再次访问原始训练数据；
4. MTT 只能读取已生成的 DP trajectory checkpoints；
5. 蒸馏图像作为 DP trajectories 的后处理；
6. 多个 expert trajectories 对同一机构数据使用 RDP composition；
7. 不同机构数据互斥时，机构之间使用 parallel composition；
8. 最终报告：
   ```text
   original_record_privacy_scope=stage_a_dp_trajectory
   stage_a_epsilon
   stage_a_delta
   accountant_type
   num_expert_trajectories
   ```
9. 编写隐私说明，明确该保证依赖：
   - 蒸馏优化不再访问 raw data；
   - 所有 trajectory 均通过 DP-SGD 生成；
   - 未发布非 DP trajectory。

## 5.2 可行性失败时的规则

若完整 DP-MTT 无法在当前代码上实现：

- 不得伪造 epsilon；
- 先实现单数据集、单机构、单 expert trajectory 的最小正确版本；
- 扩展到多 trajectory 并正式会计；
- 若 Tiny ImageNet 成本过高，允许先使用较少 trajectory，但必须运行到完整收敛，不是 2-step smoke；
- 将限制写入论文边界；
- 继续其他 Phase。

## 5.3 Stage A 隐私单元测试

- epsilon 随 steps 增加；
- epsilon 随 noise multiplier 增大而降低；
- 多 trajectory 的 composed epsilon 不小于单 trajectory；
- distillation 阶段代码断言不加载 raw dataset；
- DP trajectory 与非 DP trajectory 输出路径严格分离。

---

# 6. Phase 2：共享 ConvNet 编码器和 2048-D 特征

## 6.1 统一表示空间

所有参与方必须使用同一个共享编码器：

```text
X_i^syn = f_phi(D_i^syn)
```

禁止：

```text
X_i^syn = f_phi_i(D_i^syn)
```

后直接拼接。

## 6.2 编码器来源

按优先级实现：

1. 使用 DP 蒸馏图像的逻辑全局集合训练共享 encoder；
2. 训练过程只访问 DP 蒸馏图像，属于 Stage A 输出的后处理；
3. 训练后冻结 encoder；
4. 所有机构用同一 checkpoint 提取 2048-D 特征。

公开 benchmark 轨道也使用同一 encoder 规则。

## 6.3 特征处理

- 提取 classifier 前的 2048-D representation；
- 使用训练/蒸馏特征统计量做 normalization；
- 不允许使用 test feature 计算 scaler；
- 对每条特征实施固定 L2 范数上界，例如：
  ```text
  ||x||_2 <= 1
  ```
  便于 Stage C 安全裁剪；
- 保存：
  ```text
  feature_dim
  encoder_sha256
  scaler_sha256
  input_norm_bound
  feature_cache_sha256
  ```

## 6.4 Stage A 图与指标

Module A 正式实验比较：

```text
Full Images
→ Same Shared Encoder
→ Same Linear Classifier

Random Subset
→ Same Shared Encoder
→ Same Linear Classifier

Official MTT Distilled Images
→ Same Shared Encoder
→ Same Linear Classifier

DP-MTT Distilled Images
→ Same Shared Encoder
→ Same Linear Classifier
```

要求：

- Random subset 与 distilled 的样本数量相同；
- 相同 optimizer、learning rate、epochs 和 classifier；
- 三数据集均至少 3 seeds；
- 报告：
  - sample count；
  - compression ratio；
  - Final Accuracy；
  - Step@TargetAcc；
  - EarlyAUC；
  - runtime；
  - stage_a_epsilon（仅 DP-MTT）；
  - encoder training cost。

---

# 7. Phase 3：将 Stage B 的跨方操作放入 TFE/ABY3

## 7.1 输入

Stage B 输入：

```text
secret-shared X_i^syn
secret-shared y_i
secret-shared local linear-head parameters
secret-shared reference features/labels
```

Reference 数据只能来自：

- DP distilled feature split；或
- 公开辅助数据。

不得使用未经 DP 保护的私有 held-out raw training set。

## 7.2 安全友好的 KD

标准 softmax + KL 在 TFE 中成本高。主安全实现优先使用：

```text
teacher logits aggregation
+ logit MSE distillation
```

即：

```text
L_KD = ||z_student - z_teacher||_2^2
```

必须同时保留 plaintext reference implementation，验证数值误差。

## 7.3 Secure Stage-B 主方法

实现：

1. secure teacher-logit aggregation；
2. secure average model candidate；
3. secure reference-weighted model candidate；
4. secure candidate score computation；
5. secure selection 或安全加权融合。

主候选池优先使用：

```text
Secure KD-Avg
Secure KD-ReferenceWeighted
```

`KD-Best` 仅在安全 argmax 和 secure mux 真正实现后进入主候选池，否则只保留 plaintext ablation。

## 7.4 Reveal 边界

默认只允许 reveal：

```text
final selected model after Stage B
```

如因 TFE 限制必须 reveal candidate scores 或 selected index：

- 明确记录；
- 方法名不得写成 fully secure selection；
- 使用：
  ```text
  secure_score_computation_with_revealed_index
  ```

## 7.5 安全正确性测试

- TFE secure output 与 plaintext output 的最大误差；
- 不同 fixed-point precision 下的误差；
- 不同参与方数量；
- Avg 和 ReferenceWeighted 不能数值完全相同，除非权重确实均匀；
- 无 silent fallback；
- 所有 reveal 都进入 JSON。

---

# 8. Phase 4：安全 Fixed DP-SGD Stage C

Stage C 模型：

```text
linear classifier: z = W x + b
x ∈ R^2048
```

## 8.1 秘密共享对象

必须秘密共享：

```text
X^syn
y
W
b
per-example gradients
clipping statistics
noise shares
updated model
```

## 8.2 损失函数

优先采用安全友好的 squared loss：

```text
L_i = ||W x_i + b - y_i||_2^2
```

同时保留 plaintext softmax-cross-entropy 作为算法参考，但安全主实现必须清楚标注损失函数差异。

## 8.3 Per-example gradient

利用线性模型解析梯度，避免通用安全 autograd：

```text
e_i = W x_i + b - y_i
g_W_i = e_i x_i^T
g_b_i = e_i
```

实现并验证：

- per-example gradient；
- squared norm；
- fixed clipping；
- secure sum/mean；
- secret-shared update。

## 8.4 Secure noise

不要依赖缺失的 TF 1.15.5 secure-random custom op。

实现分布式噪声份额：

```text
each MPC compute party samples an independent noise share
total noise = sum of shares
```

要求：

- 至少一个诚实计算方即可隐藏总噪声；
- Gaussian shares 的方差配置正确；
- 或实现离散 binomial/Rademacher noise shares；
- 每个计算方只知道自己的 share；
- 本地单进程模拟必须逻辑隔离 provider；
- 不得用单次 NumPy 采样后直接 secret-share 并称为 MPC noise。

## 8.5 Privacy accountant

Stage C 需要报告两种范围：

```text
distilled_feature_record_privacy = (epsilon_C, delta_C)
original_record_privacy = inherited from Stage A DP output
```

禁止无依据地写：

```text
epsilon_total = epsilon_A + epsilon_C
```

如果 B/C 只访问 Stage A 的 DP 输出，那么相对于原始数据，B/C 是后处理；在论文中分别报告：

- Stage-A original-record DP；
- Stage-C distilled-feature-record DP。

---

# 9. Phase 5：安全 Adaptive DP-SGD Stage C

## 9.1 算法顺序

必须严格实现：

```text
use current C_t to clip current gradients
compute clipping statistics
update C_{t+1}
use C_{t+1} in next iteration
```

禁止先更新 C 再裁剪同一批梯度。

## 9.2 自适应统计

实现：

```text
b_t = fraction of per-example norms <= C_t
```

对统计量增加噪声或使用安全机制，更新：

```text
C_{t+1} = C_t * exp(-eta_c * (b_t_tilde - gamma))
```

TFE 不支持 exp 时：

- 使用经过误差验证的低阶多项式或分段近似；
- 给出近似区间和最大误差；
- 与 plaintext reference 对比；
- 不能偷偷在明文更新 C。

## 9.3 稳定性保护

必须提供：

- `C_min`；
- `C_max`；
- log-domain update；
- warm-up；
- update interval；
- layer-wise 或 block-wise clipping 可选项；
- exploding-C detection。

调参只使用 reference set。

## 9.4 消融

至少比较：

```text
Fixed global clipping
Adaptive global clipping
Adaptive layer-wise/block-wise clipping
```

并输出：

- C_t curve；
- clipped fraction；
- gradient norm quantiles；
- utility；
- epsilon_C；
- runtime；
- communication。

---

# 10. 精度质量门槛与自动调参协议

本轮目标不是“代码跑通”，而是得到达到论文使用标准的精度。不得以低精度结果结束整个任务，也不得把不收敛结果包装为正式结果。

但所有调参必须遵守科学规范：

- 只能使用训练集和独立 validation/reference set；
- test set 在配置冻结前不可用于调参、选模型、选 epoch 或比较超参数；
- 不得删除失败实验；
- 不得只保留最好 seed；
- 不得手工修改结果；
- 不得通过降低公平性来换取精度；
- 所有 trial 必须写入 `new_` 前缀的 tuning manifest。

## 10.1 三阶段调参顺序

必须按以下顺序调优，上一阶段未达到门槛时，不得直接依靠后续阶段掩盖问题：

```text
Stage A representation quality
→ Stage B initialization quality
→ Stage C non-private convergence
→ Fixed DP utility
→ Adaptive DP utility
→ End-to-end
```

## 10.2 Stage A 精度门槛

先运行官方 MTT 原始评估代码，得到每个数据集、每个 seed 的官方配置参考准确率，记为：

```text
official_mtt_reference_accuracy
```

新 Stage A `public_mtt_benchmark` 轨道必须满足：

```text
new_pipeline_accuracy >= 0.95 × official_mtt_reference_accuracy
```

或与官方参考的绝对差不超过 2 个百分点，取更严格者。

DP-MTT 轨道目标：

- 明确绘制 privacy–utility curve；
- 在选定 Stage-A epsilon 下尽可能接近 public MTT；
- 若低于 public MTT 超过 10 个百分点，不得直接进入最终端到端主结果，必须继续调：
  - expert training；
  - trajectory count；
  - trajectory length；
  - synthetic learning rate；
  - MTT outer/inner iterations；
  - image initialization；
  - encoder schedule。

## 10.3 Stage B 精度和收敛门槛

在相同 Stage C 后续训练配置下，Secure Consensus Initialization 必须优先达到：

```text
FinalAccuracy >= RandomInitialization - 0.5 percentage point
Step@TargetAcc <= 0.8 × RandomInitialization Step@TargetAcc
```

Non-IID 实验的目标是：

```text
Secure KD-ReferenceWeighted severe-non-IID accuracy
>= NoKD severe-non-IID accuracy + 1 percentage point
```

若没有达到：

1. 检查 reference weights 是否退化为均匀；
2. 调整 KD temperature；
3. 调整 supervised/KD loss weight；
4. 增加 confidence gating；
5. 调整 reference-set size；
6. 分离 calibration set 与 selection set；
7. 检查 teacher disagreement；
8. 重新运行 validation trials。

没有达到门槛时，不得宣称 Stage B 有效，也不得停止调参。

## 10.4 Stage C 精度门槛

首先要求安全/对应明文 Plain 模型在各数据集上充分收敛。DP 对比只在 Plain 已稳定后进行。

在相同 target epsilon、delta、样本、模型和训练预算下，Adaptive DP 的目标是：

```text
Adaptive FinalAccuracy >= Fixed FinalAccuracy + 1 percentage point
```

并且：

```text
Adaptive EarlyAUC >= Fixed EarlyAUC
```

同时避免：

- `C_t` 持续撞击 `C_max`；
- 平均裁剪比例长期接近 0% 或 100%；
- 噪声尺度因 C 失控而爆炸；
- DP 精度接近随机猜测。

未达门槛时自动调节：

```text
learning rate
optimizer
weight decay
batch size/effective batch size
epochs
C_0
C_min/C_max
gamma
eta_c
update interval
warm-up
noise multiplier for the target epsilon
global/layer-wise/block-wise clipping
feature norm bound
loss scaling
fixed-point precision
```

调参只依据 validation/reference 指标。

## 10.5 End-to-End 门槛

在相同样本数、encoder、模型、Stage-C target epsilon 和训练预算下，Ours 相比 Baseline 的目标为：

```text
FinalAccuracy improvement >= 2 percentage points
```

或：

```text
Step@TargetAcc reduction >= 20%
```

至少满足一个，且另一个指标不得显著恶化。

## 10.6 自动调参流程

每个数据集采用分阶段搜索：

### 粗搜索

- 随机搜索或有限网格；
- 每个模块先运行短但足以区分配置的 validation budget；
- 不允许使用 2-step/10-step smoke 作为精度依据。

### 精搜索

- 选 validation 前 3–5 个配置；
- 使用完整正式预算；
- 至少 2 个 tuning seeds；
- 冻结最终配置后再运行正式 seeds。

### 正式测试

- 最终超参数冻结；
- test set 只运行正式 seeds；
- 不根据 test 结果再次调整同一实验并覆盖原结果；
- 如确需新一轮方法修正，创建新的 `new_tuning_round_*` 版本并保留旧轮次。

## 10.7 不允许伪造“理想精度”

“调到理想”为目标，不代表可以无限制挑结果。若经过系统调参与诊断仍未达到门槛：

- 保持任务状态为 `NOT_READY_FOR_PAPER`；
- 输出最佳 validation 配置、失败原因和下一步；
- 不得将 Phase 标记为完全通过；
- 不得用 test set 反复试参；
- 不得隐藏负结果。

只有达到上述质量门槛，或有充分证据说明方法本身不支持该结论并如实降级论文主张，才可结束对应正式实验阶段。

---

# 11. 正式实验预算：不得再统一 steps

所有实验使用 epochs 或 processed examples 定义，并将其转换为 dataset-specific steps。

## 11.1 Pilot 只用于确定正式终点

每个数据集先运行 1 个 seed 的 pilot：

- 使用 reference accuracy；
- 当连续 5 个 eval intervals 提升小于 0.2 percentage points 时视为接近平台；
- 正式 max epochs 取 pilot plateau epoch 的 1.25 倍；
- 设置合理上下限；
- pilot 结果不得作为正式均值。

## 11.2 推荐初始搜索范围

### CIFAR-10

- Stage B：50–150 epochs；
- Stage C：50–150 epochs。

### CIFAR-100

- Stage B：100–300 epochs；
- Stage C：100–300 epochs。

### Tiny ImageNet

- Stage B：150–400 epochs；
- Stage C：150–400 epochs。

最终 steps 自动计算：

```text
steps = ceil(epochs * num_samples / batch_size)
```

## 11.3 正式 seeds

至少：

```text
42, 43, 44
```

关键最终表建议增加：

```text
45, 46
```

若时间过长：

- 三数据集主结果至少 3 seeds；
- CIFAR-10 关键消融 5 seeds；
- 不得退回 1 seed 后宣称正式结果。

---

# 12. Phase 6：Module A 正式实验

每个数据集：

```text
Full Images
Random Subset
Official MTT
DP-MTT
```

三数据集、至少 3 seeds。

图形：

- 每个数据集一个子图；
- 横轴用各自 iteration 终点；
- 不强行统一；
- 主图最多 4 条曲线；
- DP-MTT 图例标注 Stage-A epsilon；
- 输出 mean ± std 阴影或误差条。

---

# 13. Phase 7：Module B 正式实验

Stage B 分成两组图，不混在一起。

## Figure B1：收敛速度

每个数据集只画两条主曲线：

```text
Random Initialization
Secure Consensus Initialization
```

条件：

- 相同 2048-D features；
- 相同安全 Stage C 后续训练；
- 相同 optimizer 和 privacy setting；
- 只改变初始化；
- 横轴 dataset-specific steps。

指标：

- Step@TargetAcc；
- EarlyAUC；
- Final Accuracy；
- Secure runtime；
- communication。

## Figure B2：Non-IID 缓解

数据划分：

```text
IID
Mild Non-IID
Severe Non-IID
```

方法：

```text
NoKD
Secure KD-Avg
Secure KD-ReferenceWeighted
```

使用分组柱状图或折线，不画大量训练曲线。

指标：

- Final Accuracy；
- accuracy retention from IID to Severe；
- teacher disagreement；
- reference weight entropy；
- runtime。

必须验证：

- ReferenceWeighted 与 Avg 实际不同；
- KD 在 severe non-IID 下至少有稳定趋势；
- 若无改善，如实报告并调整 KD 监督权重、温度和置信度门控后重跑。

---

# 14. Phase 8：Module C 正式实验

每个数据集主图只画：

```text
Secure Fixed DP-SGD
Secure Adaptive DP-SGD
```

Plain secure/nonprivate 结果放表格或虚线参考，不要让主图过载。

## 隐私点

每个数据集至少比较 3 个 Stage-C 目标隐私强度，例如：

```text
epsilon_C ∈ {1, 3, 8}
```

若 accountant/机制不适用于某个离散噪声，标记 N/A，不得复用错误会计。

## 输出

- accuracy vs iterations；
- accuracy vs epsilon_C；
- C_t；
- clipped fraction；
- runtime；
- communication；
- approximation error；
- mean ± std。

---

# 15. Phase 9：End-to-End 正式实验

每个数据集 Figure 2 风格只画两条主曲线：

## Baseline

```text
Random Subset
→ Shared Encoder
→ Secure Average Initialization
→ Secure Fixed DP-SGD
```

## Ours

```text
DP-MTT Distilled Images
→ Shared Encoder
→ Secure KD-ReferenceWeighted Initialization
→ Secure Adaptive DP-SGD
```

公平性要求：

- 相同 2048-D encoder；
- 相同样本数；
- 相同模型；
- 相同 batch size；
- 相同 Stage-C target epsilon_C；
- 同一 original-record Stage-A privacy setting；
- dataset-specific iteration endpoint。

消融表另外列出：

```text
Random subset
Official MTT only
DP-MTT only
DP-MTT + Secure Stage B
DP-MTT + Secure Stage B + Fixed DP
DP-MTT + Secure Stage B + Adaptive DP
```

主图不画全部曲线。

---

# 16. Phase 10：TFE 系统评估

必须是真实正式规模，而不是 10-step sanity。

## 16.1 正式系统规模

至少提供：

- small；
- medium；
- largest feasible。

每个规模记录：

```text
num_samples
feature_dim=2048
num_classes
batch_size
iterations
model_parameters
fixed-point precision
runtime
communication bytes
communication rounds
peak memory
revealed values
```

## 16.2 网络环境

如果只能单机模拟：

- 明确标记 `single-machine process simulation`；
- 不声称 LAN/WAN；
- 尽可能通过进程通信统计 bytes；
- 无法统计时说明工具限制。

## 16.3 安全边界

最终报告必须逐项说明：

- 哪些值始终 secret-shared；
- 哪些值 reveal；
- 何时 reveal；
- 候选选择是否完全安全；
- noise 是否由分布式份额生成；
- 至少一个 honest party 假设。

---

# 17. 正式结果最低验收标准

任务不能以“代码能运行”作为完成标准，也不能以精度明显不收敛或接近随机猜测作为正式完成结果。

必须满足：

1. 三数据集均完成正式 Module A/B/C/end-to-end；
2. 每个主实验至少 3 seeds；
3. 每个数据集使用不同合理 iteration 终点；
4. 生成完整 JSON、CSV、PNG/PDF；
5. 结果包含 mean ± std；
6. Stage A epsilon 来自真实 DP mechanism；
7. Stage B 跨方聚合在 TFE 中执行；
8. Stage C 固定与自适应训练在 TFE 中执行；
9. 无 silent plaintext fallback；
10. smoke/pilot 与 formal 结果目录完全分离；
11. 失败结果不能删除；
12. 论文声明与真实实现一致。

---

# 18. 自动检查规则

创建测试：

```text
tests/test_stage_a_dp_trajectory.py
tests/test_shared_encoder_alignment.py
tests/test_secure_stage_b.py
tests/test_secure_stage_c_fixed.py
tests/test_secure_stage_c_adaptive.py
tests/test_privacy_scope.py
tests/test_no_test_leakage.py
tests/test_no_plaintext_fallback.py
```

自动断言：

- 主 formal 配置不能包含 `--quick`；
- formal steps 不得全部相同；
- Stage B/C backend 必须是 `tfe_mpc`；
- feature dimension 必须是 2048；
- encoder signature 跨参与方一致；
- test set 不进入调参 API；
- DP-MTT 生成阶段不重新加载 raw data；
- Gaussian/binomial accountant 不混用；
- reveal 列表非空时自动写 limitation；
- 任何 BLOCKED 项不得被标记 PASS。

---

# 19. 最终输出格式

全部完成后，请按以下格式回复：

```text
1. Branch and Git Commits
2. Implemented Stage-A DP Mechanism
3. Shared Encoder and Feature Pipeline
4. Secure Stage-B Implementation
5. Secure Stage-C Fixed DP-SGD
6. Secure Stage-C Adaptive DP-SGD
7. Privacy Definitions and Accounting
8. GPU/CPU Resource Utilization
9. Module-A Formal Results
10. Module-B Convergence Results
11. Module-B Non-IID Results
12. Module-C Formal Results
13. End-to-End Two-Curve Results
14. TFE Runtime and Communication Results
15. Files Added/Modified
16. Exact Reproduction Commands
17. Remaining BLOCKED Items
18. Supported Paper Claims
19. Claims That Must Still Be Avoided
```

---

# 20. 立即执行

现在立即：

1. 检查当前仓库和远端状态；
2. 新建 `new-secure-feature-pipeline`；
3. 创建进度文件和双 GPU 调度器；
4. 开始 Phase 0；
5. 完成一个 Phase 后自动进入下一个；
6. 不要在 smoke test 后停止；
7. 除非需要外部凭据、存在不可逆破坏操作，或全部 Phase 完成，否则不要等待我逐项确认。


---

# 最终追加约束

- 精度不理想时，不得仅生成图表后结束；必须执行诊断、validation 调参、重新训练和重新评估，直到达到质量门槛或明确标记 `NOT_READY_FOR_PAPER`。
- 所有调参过程必须可追踪，不得使用 test leakage 或 cherry-picking。
- CPU 线程始终严格为 1，包括两张 GPU 同时运行时。
- 所有长任务必须通过 tmux 持久执行并支持 checkpoint 恢复。
- 本轮新生成的一切实验文件和目录必须以 `new_` 开头。

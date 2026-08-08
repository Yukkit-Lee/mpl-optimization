# SecretFlow-SPU 0807 任务现场交接（给 Claude Code）

更新时间：2026-08-07 01:40 UTC 之后  
项目目录：`/root/autodl-tmp/tfe_experiments`  
当前分支：`new-secretflow-spu-end2end`  
任务主文档：`plan/0807_new_secretflow_spu_end2end_figure2_codex_prompt_v2.md`  
交接前已推送基线：`61153988ab6ad714f8ba5cc58086758d7d272b26`  
交接提交：请以 `git log -1 --oneline` 查询；它应包含本文件、设备 smoke PASS 产物和最新进度。

## 1. 当前现场结论

用户要求 Codex 立即停下并交接，因此目前没有运行中的训练、Ray、SPU、GCS 或 tmux 会话。不要把“任务暂停”误解为“计划已经完成”。

已经完成并验证：

1. Phase A 硬件审计；
2. Phase B 隔离环境安装与冻结；
3. Phase C 官方 bare-SPU 三方 quickstart 的 S1--S4；
4. Phase D 六组既有特征/划分/lineage 复用审计；
5. 官方源码 SFL 调度层安装与兼容适配；
6. 真实 `secretflow.SPU` 三方设备 API 的最小线性更新 smoke。

尚未开始：

1. 论文管线 P1--P4 smoke 的实现和运行；
2. Stage B / Adaptive Stage C 的 SecretFlow-SPU 正式实现；
3. CPU 代表性微基准与 CPU freeze；
4. Tiny Anchor CE 有界诊断和最终算法配置 freeze；
5. 六条正式 seed42 任务；
6. SPU Figure 2、dual-backend 图、正式表格和最终报告。

因此绝对不能对用户说“SPU 迁移/实验已经完成”或“可以直接画论文图”。当前只证明环境、底层算子、资产复用和真实 SecretFlow 设备链可用。

## 2. 接手后的强制阅读顺序

在修改或运行任何内容之前，完整读取：

1. `AGENTS.md`
2. `plan/0807_new_secretflow_spu_end2end_figure2_codex_prompt_v2.md`
3. 本交接文件
4. `progressLog/project_progress.md` 中最后一个 `2026-08-07` 区块
5. `new_log/new_secretflow_spu/new_spu_install_report.md`
6. `new_log/new_secretflow_spu/new_spu_hardware_audit.md`
7. `new_log/new_secretflow_spu/new_spu_feature_conversion_audit.md`
8. `new_log/new_secretflow_spu/new_spu_secretflow_device_smoke.md`
9. `reportToMe/unified_experiment_report.md`
10. `reportToMe/paper_readiness_assessment.md`

不要读取旧私有 JSONL，除非这些交接/进度材料无法回答具体历史问题；若必须读取，只能用 `rg` 精确定位，不能整文件输出，也不能泄露 token、密钥或敏感内容。

## 3. Git、文档和运行纪律

- 只在 `new-secretflow-spu-end2end` 分支继续。
- 任何仓库修改都必须同步更新 `progressLog/project_progress.md`，随后 commit 并 push。
- 实验代码必须先提交推送，再从该提交在 tmux 中运行；结果再单独提交推送。
- 所有长任务或正式任务必须使用 tmux，并在启动前检查 CPU、cgroup 内存、磁盘、残留进程，记录精确命令。
- 不要把 CPU/BLAS 固定为 1；但三方 SPU 不应无脑各占 32 线程，必须用短微基准选择总吞吐最好的分配。
- `new_checkpoints/` 永远不提交。大型恢复状态也只留本机。
- 当前 TFE 代码、配置、日志、checkpoint、旧 Figure 2 全部只读。需要共享逻辑时复制到 `new_secretflow_spu` 命名空间。
- 不得覆盖 `new_expFig/new_true_ours/new_figure2_pea_style_secure_time.png` 或其 plot data。
- GitHub token 位于 `/root/.config/githubToken.txt`。只能通过临时 `GIT_ASKPASS` 脚本读取，禁止 echo/cat/日志输出 token；push 后删除临时脚本。

## 4. 代理和下载源策略

用户明确要求 9999 代理非必要不要使用：

- 阿里、清华等国内源：清除代理或用 `NO_PROXY` 直连。
- GitHub、官方 PyPI 等境外官方源：必要时使用 `HTTP_PROXY=http://127.0.0.1:9999` 和 `HTTPS_PROXY=http://127.0.0.1:9999`。
- pip 已采用阿里主源、官方 PyPI fallback；只有 fallback 流量应走 9999。
- 不要重新下载三套原始数据集或 MTT public data；资产已经在 `/root/autodl-tmp`/现有缓存中。

## 5. 硬件与资源事实

不要相信 `free` 展示的全部宿主资源，必须以 cgroup/cpuset 为准：

- 宿主：2 socket × 32 physical cores，128 logical CPUs。
- 当前容器 CPU affinity/cpuset：`96-127`，配额约 32 CPU。
- cgroup memory limit：60 GiB，无 swap。
- 交接时项目盘：约 30 GiB 可用。
- `numactl` 未安装。
- CPU 支持 AVX-512；OpenBLAS/libgomp 可用。
- `new_log/new_secretflow_spu/new_spu_hardware_audit.{json,md}` 是权威审计。

启动前建议只读检查：

```bash
git status --short --branch
git rev-parse HEAD
nproc
grep Cpus_allowed_list /proc/self/status
cat /sys/fs/cgroup/cpu.max
cat /sys/fs/cgroup/memory.max
df -h /root/autodl-tmp
free -h
tmux list-sessions 2>/dev/null || true
pgrep -a -x raylet || true
pgrep -a -x gcs_server || true
```

## 6. 环境现状

环境位置：`/root/autodl-tmp/new_envs/spu_e2e`

核心版本：

```text
Python                  3.10.20
SecretFlow              1.14.0b0
SPU                     0.9.4.dev20250618
JAX/JAXLIB              0.4.26 / 0.4.26
NumPy                   1.26.4
protobuf                4.25.9
Ray                     2.52.0
SecretFlow-RayFed       0.2.1a2
SFL                     1.0.0.dev20260807
interconnection         0.2.0.dev0
```

官方 SFL 源码位于 `/root/autodl-tmp/secretflow_sfl_source`，固定 commit：

```text
c383e40f665063d7f7d87e437a73e015f87c435c
```

SFL 是 `--no-deps` 源码构建，原因是其单体 metadata 会拉取数 GiB、与本 SPU device pipeline 无关的 TensorFlow/PyTorch/ONNX/audio/vision 栈，并锁定可能冲突的 `secretflow-lite==1.13.0b0`。不要为了让 `pip check` 表面全绿而安装这些无关 extras。当前实际 device imports 已通过真实运行门禁。完整冻结见：

- `new_log/new_secretflow_spu/new_spu_environment_manifest.txt`
- `new_log/new_secretflow_spu/new_spu_pip_freeze.txt`
- `new_log/new_secretflow_spu/new_spu_install_report.md`

## 7. 已完成门禁及精确结果

### 7.1 Bare-SPU 官方 quickstart（只证明底层运行时）

文件：

- `new_scripts/new_secretflow_spu/new_spu_quickstart_smoke.py`
- `new_log/new_secretflow_spu/new_spu_quickstart_smoke.json`
- `new_log/new_secretflow_spu/new_spu_quickstart_smoke.md`

结果：三方 ABY3/FM128，加法、比较、矩阵乘、线性 logits/MSE/update 共 7/7 校验通过，最大绝对误差约 `2.98e-8`，elapsed `1.84s`，driver peak RSS `0.415GiB`。它使用 `spu.utils.distributed`，只允许称为底层 quickstart gate，不能冒充论文正式 SecretFlow 后端。

### 7.2 真实 SecretFlow-SPU device smoke（交接前刚通过）

文件：

- `new_scripts/new_secretflow_spu/new_secretflow_spu_backend.py`
- `new_scripts/new_secretflow_spu/new_secretflow_device_smoke.py`
- `new_log/new_secretflow_spu/new_spu_secretflow_device_smoke.json`
- `new_log/new_secretflow_spu/new_spu_secretflow_device_smoke.md`
- `new_log/new_secretflow_spu/txt/new_spu_secretflow_device_smoke.log`

运行边界：真实 `secretflow.SPU`，源码 SFL `SimulationStrategy`，Ray 承载 alice/bob/carol 三方，ABY3/FM128、`fxp_fraction_bits=16`；X 和 y 来自不同 PYU；只 reveal 最终 toy 输出；没有 debug/PYU 模拟、bare helper 或 plaintext fallback。

PASS 结果：

```text
logits max abs error  2.7466565e-05
loss max abs error    3.0398369e-06
weight max abs error  3.6627054e-05
bias max abs error    3.8459897e-05
elapsed               13.214894 s
driver peak RSS       0.458412 GiB
test read             false
locked holdout read   false
```

运行结束后 Ray/GCS 全部退出。日志里出现 JAX 在 Ray fork 前已多线程初始化的 RuntimeWarning，但本次没有死锁且成功完成。正式管线应避免在 `sf.init()` 前触发 JAX backend 初始化；可以将 `jax.numpy` 的实际加载/调用推迟到 Ray 初始化后，并单独验证，但不要修改 site-packages。

注意：SFL 固定源码仍从旧位置导入 `ray._private.ray_option_utils`，而 Ray 2.52 已移动到 `ray._common`。当前 adapter 只在本进程中给同一个官方 Ray 模块设置兼容 alias，再通过 SecretFlow 公共 `add_distribution_modes` 注册官方 SFL strategy；没有 patch site-packages，也没有自制明文 scheduler。

历史失败必须保留：

- `a0f769a` 前：Ray API 路径漂移；
- `e344a1e`：SFL 顶层无条件导入 Interconnection 链但缺包；
- `6115398`：补齐 `interconnection==0.2.0.dev0` 并冻结环境；
- 之后本次 smoke 才 PASS。

## 8. 资产复用与方法语义

权威审计：

- `new_log/new_secretflow_spu/new_spu_feature_reuse_audit.json`
- `new_log/new_secretflow_spu/new_spu_feature_conversion_audit.md`

三个数据集均为 2048-D float32、finite，现有 Stage A/feature extraction 不得重跑。

样本数：

| Dataset | Full Stage B | Full Stage C | Ours Stage B | Ours Stage C |
|---|---:|---:|---:|---:|
| CIFAR-10 | 34,900 | 5,000 | 480 | 480 |
| CIFAR-100 | 34,000 | 5,000 | 4,800 | 4,800 |
| Tiny-ImageNet | 68,000 | 10,000 | 9,600 | 9,600 |

重要角色语义：Stage-C cache 中的 `x_reference/y_reference` 是 development evaluation，不是受保护训练池；真正 Stage-C 训练数据是 `x_encoder_train/y_encoder_train`。Full 使用原始 Stage-C split，Ours 继续使用 Official-MTT compact。Full/Ours development 逐值一致，test/locked holdout 未读。

Ours 每类 48 条 MTT，确定性划分为 40 local + 8 labeled anchor。新 SPU Ours 必须采用：

```text
secret-shared compact client features
→ secure lightweight-head training
→ Secure Average
→ class-balanced labeled compact-anchor CE refinement
→ refined global initializer
```

旧 TFE Figure 2 的 Ours 仍是 `secure_average_only`，不包含最新 Anchor CE。不要把旧图口径写成新方法，也不要修改旧图。

Anchor CE 的正向正式 freeze 只验证了 CIFAR-10/100：

```text
CIFAR-10   steps=40, LR=0.1, mu=1e-3
CIFAR-100  steps=40, LR=0.2, mu=1e-3
```

Tiny 在旧任务中被明确排除，不能称为“已经验证最佳超参”。Tiny 的预注册起点是从 C100 拷贝的 `steps=40, LR=0.2, mu=1e-3`；如有必要，只允许 development-only、一次改变一个因素的有界诊断，登记 tuning registry 后 freeze。禁止大网格和结果导向搜索。

冻结 T@80/90/95 target，绝对不能看 SPU 结果后移动：

```text
CIFAR-10:     0.58128 / 0.65394 / 0.69027
CIFAR-100:    0.34464 / 0.38772 / 0.40926
Tiny-ImageNet 0.22120 / 0.24885 / 0.262675
```

## 9. 下一阶段应如何做（不要跳步）

### Step 1：只实现隔离的论文 SPU pipeline

所有新代码只放在：

```text
new_scripts/new_secretflow_spu/
new_configs/new_secretflow_spu/
new_log/new_secretflow_spu/
new_output/new_secretflow_spu/
new_expTab/new_secretflow_spu/
new_expFig/new_secretflow_spu/
new_checkpoints/new_secretflow_spu/   # local only
```

先从 TFE/reference 代码复制最少必要语义，不要编辑旧实现。为 Stage B、Stage C、计时、accountant、checkpoint 明确写 correctness assertions 和 claim flags。

### Step 2：CIFAR-10 P1--P4 pipeline smoke

严格按主计划：

- P1 feature：shape、dtype、label、split、hash；
- P2 Stage B：极短 secure local heads → average → anchor refinement，与 plaintext reference 对齐；
- P3 Stage C：仅几步 Adaptive DP，验证 finite weight/accuracy/clip、noise path、privacy accountant、无 fallback；
- P4 trace：验证 Stage-B 时间、B→C、累计 online secure seconds、development accuracy、epsilon 均有正确日志。

输出：

```text
new_spu_pipeline_smoke.json
new_spu_pipeline_smoke.md
```

P1--P4 全部通过前不得启动正式 run。

### Step 3：Stage C 必须忠实保持 Adaptive DP

概念链必须仍是：

```text
Poisson sampling
→ per-example linear-head gradients
→ adaptive clipping
→ clipped-gradient aggregation
→ Gaussian perturbation
→ secure parameter update
```

必须审计：host CSPRNG 只产生 sampling IDs；per-example gradient、clip、aggregation 和 update 位于 SPU；Gaussian noise 不能在单一明文端合成后再送入 SPU，应使用可审计的分方 noise shares/安全求和设计；accountant 与 noisy count/adaptive clip 语义要对应现有论文实现。

论文隐私 claim 永远只是：`conditional Stage-C DP`，不是 end-to-end record-level DP。Stage A 无 DP，Stage B 只有 MPC confidentiality。

### Step 4：CPU 短微基准和 freeze

用代表性固定 shape secure step 测，不使用模型准确率选择 CPU 配置。至少登记 secure steps/s、wall、CPU utilization、peak RSS、context switching（若可靠）。候选应围绕 32 CPU 总配额设计，避免三方线程严重 oversubscribe。生成：

```text
new_spu_cpu_tuning.csv
new_spu_cpu_freeze.json
new_spu_cpu_tuning_report.md
```

正式六条运行前必须 freeze。

### Step 5：算法 tuning 和 freeze

从已验证参数开始。只有 correctness 已通过但 utility/稳定性明显下降时，才允许 development-only 的 one-factor-at-a-time 小诊断；每次写清原因、结果、接受/拒绝。生成：

```text
new_spu_tuning_registry.csv
new_spu_tuning_report.md
new_spu_final_config_freeze.json
```

### Step 6：六条正式运行

只能 seed42，禁止额外 seed 和所有 SPU ablation。推荐顺序：

```text
1. CIFAR-10 Ours
2. CIFAR-10 Full
3. CIFAR-100 Ours
4. CIFAR-100 Full
5. Tiny-ImageNet Ours
6. Tiny-ImageNet Full
```

在 Tiny 前根据前四条估算内存和工期。Tiny 仍是必做；只有用实测证明资源不可行时才能负面停止，绝不能用 plaintext 替代。

每条必须使用唯一 run ID 和可恢复 checkpoint。恢复不能重复 DP noise；要保存 RNG/accountant/step/optimizer/clip 状态。checkpoint 不上传 Git。

### Step 7：计时、结果和作图

Figure 2 横轴必须是累计 online secure runtime。分别记录环境、数据加载、编译、secret sharing/infeed、Stage B online、Stage C online、evaluation/reveal、raw wall；不隐藏 compile。如果 plot 排除 compile，报告中单列。

最终只 reveal 计划允许的模型/诊断边界。communication 若没有可靠 API 就填 `N/A`，禁止估算伪造。

生成主计划第 24 节的全部文件，尤其六个 trace、main table、runtime breakdown、correctness audit、final report、SPU 3-panel Figure 2 和可选 dual-backend 2×3。旧 Figure 2 永远不覆盖。

## 10. 正式实验硬范围

正式矩阵严格只有：

```text
CIFAR-10      Full seed42
CIFAR-10      Ours seed42
CIFAR-100     Full seed42
CIFAR-100     Ours seed42
Tiny-ImageNet Full seed42
Tiny-ImageNet Ours seed42
```

禁止：额外 seeds、SPU KD/Consensus、disagreement、heterogeneity trigger、Fixed-vs-Adaptive、任意 Stage-B ablation。precision 变化只有在修复后端数值稳定性时才可作为工程诊断，不能包装成论文 ablation。

## 11. 科学结论边界

- 已有 TFE 详细消融仍是主要消融后端；本任务 SPU 只做跨后端主趋势复现。
- 不得声称 official test/locked holdout；当前全是 development-only。
- 不得声称多 seed 或画 n=1 variance band。
- 不得声称端到端 record-level DP。
- 不得用不可比计时边界声称 SPU 快于 TFE。
- 不得声称通信量，除非 SecretFlow/SPU 暴露可靠可审计值。
- KD/consensus causality、disagreement-guided anchor、heterogeneity-trigger 都是旧协议下负结果，不能作为正模块。

## 12. 接手后的第一个唯一动作

不要马上跑实验。先执行：

1. 拉取并确认本交接提交；
2. 确认工作区 clean、HEAD=远端；
3. 确认无 Ray/SPU/tmux 残留；
4. 完整读主计划和本交接；
5. 只读定位现有 TFE Adaptive Stage-C、privacy accountant、checkpoint 与 timing 语义；
6. 在聊天里报告拟复制的代码边界和将新建的 SPU 文件；
7. 更新 progress，先提交推送实现骨架，再运行 CIFAR-10 P1--P4 smoke。

不要重复 Phase A--D、bare quickstart、feature audit 或本次真实 device smoke，除非文件/hash/环境被更改并有明确原因。

## 13. 暂停时现场核验

交接生成时：

```text
active tmux sessions       none
raylet                     none
gcs_server                 none
formal runs completed      0 / 6
pipeline smoke             not implemented / not run
fallback                   no
test read                  false
locked holdout read        false
```

这就是当前真实进度。

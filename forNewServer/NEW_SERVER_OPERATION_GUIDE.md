# 新服务器恢复与论文改稿操作指南

- 更新时间：2026-08-28（UTC）
- 源服务器项目：`/root/autodl-tmp/tfe_experiments`
- GitHub：`https://github.com/Yukkit-Lee/mpl-optimization.git`
- 权威分支：`unified-experiments`
- 编写本指南时的仓库 HEAD：`5e9bcf4ae4d4224a3d3401514c4a6608d41c8629`
- 最后一个 SPU 正式实验提交：`ca2ced9a4fec43b06793c8da8626462f8ed2fb0b`

> 本文面向更换服务器后的用户或新 Agent。论文已经投稿，默认任务是恢复、核验、
> 重画论文图表和准备改稿；没有用户明确授权时，不启动正式训练、参数搜索或长实验。
> 本文不包含任何 token、SSH 私钥、代理认证信息或其他凭据。

## 1. 接手后的阅读顺序

进入仓库后依次完整阅读：

1. `AGENTS.md`：当前仓库必须遵守的操作规则。
2. 本文件：新服务器路径、环境、数据和常用入口。
3. `progressLog/project_progress.md` 的末尾：最新实际执行记录。
4. `reportToMe/unified_experiment_report.md`：跨实验结论和科学边界。
5. `reportToMe/new_paper_figure_source_map_0809.md`：Figures 2--6 的数据源。
6. `AUTODL_MIGRATION.md`：早期环境恢复细节；注意其中部分分支名和任务状态已经过时。
7. `plan/` 中与当前改稿问题直接相关的计划，不要把所有旧计划都当成待办任务。

状态冲突时采用以下优先级：

```text
当前正式 report/audit/effective_config
  > progressLog/project_progress.md 的最新记录
  > reportToMe/unified_experiment_report.md
  > 旧 plan / handoff / 历史日志
```

不要把文件名中带有 `FAIL`、`RUNNING`、`gate`、`pilot`、`smoke`、
`INTERRUPTED_PARALLEL_DO_NOT_USE` 或 `superseded` 的产物当作当前正式结果。

## 2. 当前项目状态摘要

### 2.1 Git 状态

- 权威分支为 `unified-experiments`。
- 正式 SPU CIFAR-100 Full Data seed44 已在提交 `ca2ced9a` 中完成、审计并推送；其后的
  `5e9bcf4a` 是论文文字、图例和版面调整，不是新的正式实验运行。
- 源服务器在编写本文前仍有用户本地修改和未跟踪文件。因此，新服务器不能只执行
  `git clone` 就认为状态完整；必须同时恢复迁移包中的工作树文件。
- 尤其注意以下本地修改：
  - `new_scripts/new_paper_figures/new_build_figure2.py`
  - `new_scripts/new_paper_adaptive/new_user_editable_figure6_cifar100_config.py`
  - `new_log/new_true_ours/new_true_ours_master_results.json`
  - `new_expFig/paper_experiment/new_figure2b_spu_short_axis_deltas.pdf`
- `goodPaper/` 是本地参考论文目录，默认不上传 GitHub。
- `new_checkpoints/` 是运行状态和大 checkpoint，永远不要提交 Git。

迁移后先执行：

```bash
cd /root/autodl-tmp/tfe_experiments
git branch --show-current
git rev-parse HEAD
git rev-parse origin/unified-experiments
git status --short --branch
```

在没有确认迁移包里的本地改动都已恢复前，不要运行 `git reset --hard`、
`git checkout -- <file>`、`git clean` 或覆盖式同步。

### 2.2 当前科学结论边界

- 当前主结果主要是 development-validation，不是 official-test 或 locked-holdout。
- 条件式 Stage-C DP 的 `epsilon=3, delta=1e-5` 不等于完整端到端 record-level DP。
- Stage-B initializer 是数据依赖的，论文中不能把整个 pipeline 宣称为端到端 DP。
- TFE 与 SPU 之间不是硬件归一化速度比较；安全结论是各自后端内部的 Ours vs Full。
- KD/consensus、disagreement-guided anchor 和 heterogeneity-trigger routing 是已完成负结果，
  不得包装成正模块。
- Figure 6 是相同 epsilon 下、Fixed/Adaptive 各自配置的完整方法比较；双方学习率、
  steps 和 batch 可不同，因此不能宣称它隔离证明了 adaptive clipping 单因素因果。

## 3. 推荐的新服务器目录布局

保持绝对路径不变能减少配置修改：

```text
/root/autodl-tmp/
├── tfe_experiments/                 # 本仓库和全部论文证据
├── new_envs/
│   ├── distillation/                # 现代 PyTorch/Opacus/绘图环境
│   ├── tfe/                         # Python 3.6 + TF 1.15 + TF-Encrypted
│   └── spu_e2e/                     # SecretFlow/SPU 环境
├── new_data/                        # 实际使用的数据副本
├── mtt-distillation/                # MTT 源码树
├── georgecazenavette.zip            # 官方 MTT 蒸馏数据归档
├── cifar-10-python.tar.gz
├── cifar-100-python.tar.gz
└── tiny-imagenet-200.zip
```

如果新服务器必须使用其他根目录，先修改路径配置，不要批量直接替换 JSON 中的证据字段：

```bash
cp config/new_paths.autodl.example.env ~/.config/new-secure-feature-pipeline.env
set -a
source ~/.config/new-secure-feature-pipeline.env
set +a
```

配置文件不得写入 GitHub token、SSH 私钥或代理密码。

## 4. 仓库核心目录地图

| 目录 | 含义 | 是否必须迁移 |
|---|---|---|
| `new_scripts/` | 当前实验、审计、绘图脚本 | 必须 |
| `new_configs/` | 冻结参数与正式配置 | 必须 |
| `new_log/` | 正式 report、audit、JSON、运行日志 | 必须 |
| `new_output/` | 轨迹 CSV、输入输出和绘图数据 | 必须 |
| `new_expTab/` | 论文表格 CSV | 必须 |
| `new_expFig/` | 论文图 PNG/PDF/SVG | 必须 |
| `reportToMe/` | 论文结论、验收与图源说明 | 必须 |
| `progressLog/` | 真实执行时间线 | 必须 |
| `plan/` | 历史计划与实验协议 | 必须 |
| `portable_artifacts/` | 可复现的小型冻结资产 | 必须 |
| `new_init_model/` | 初始化器和便携模型 | 必须 |
| `new_cache/` | 高成本特征缓存，约 12GB | 强烈建议 |
| `new_checkpoints/` | 新 TFE/SPU checkpoint，约 26GB | 选择性但建议保留正式项 |
| `new_checkpoint/` | 早期 Public-MTT checkpoint，约 15GB | 空间足够再保留 |
| `migration/` | 环境快照、MTT/TFE 恢复资产 | 必须 |
| `goodPaper/` | 本地参考论文 | 可选，不上传 Git |

## 5. 三套环境及用途

严禁把现代 PyTorch、legacy TFE 和 SPU 依赖混装在一个环境中。

### 5.1 现代训练与绘图环境

```bash
MODERN_PY=/root/autodl-tmp/new_envs/distillation/bin/python
"$MODERN_PY" --version
"$MODERN_PY" -c 'import torch,opacus,numpy,matplotlib; print(torch.__version__, opacus.__version__)'
```

主要用途：

- 明文诊断和特征提取
- Adaptive/Fixed DP 消融
- Figures 2--6 绘制
- 大部分 JSON/CSV 构建和审计

如果未迁移整个环境，优先参考：

```text
environment.yml
requirements.txt
migration/env_snapshots/new_distillation-*
```

### 5.2 legacy TFE 环境

```bash
TFE_PY=/root/autodl-tmp/new_envs/tfe/bin/python
"$TFE_PY" --version
"$TFE_PY" -c 'import tensorflow as tf, tf_encrypted as tfe; print(tf.__version__)'
```

主要用途：TFE/ABY3 下的 Stage B、Stage C 和端到端正式实验。

关键版本是 Python 3.6、TensorFlow 1.15.5、TF-Encrypted 0.8.0。最容易丢失的是
secure-random custom ops：

```text
/root/autodl-tmp/new_envs/tfe/lib/python3.6/site-packages/
  tf_encrypted/operations/secure_random/
    secure_random_module_tf_1.15.2.so
    secure_random_module_tf_1.15.5.so
```

如果导入失败，按 `AUTODL_MIGRATION.md` 第 3--4 节及
`scripts/new_build_tfe_custom_ops.sh` 重建。不要用系统 Python 运行 TFE。

### 5.3 SecretFlow/SPU 环境

```bash
SPU_PY=/root/autodl-tmp/new_envs/spu_e2e/bin/python
"$SPU_PY" --version
"$SPU_PY" -c 'import secretflow,spu,ray; print("SPU environment OK")'
```

主要用途：SecretFlow-SPU/ABY3/FM128/fxp16 完整轨迹实验和审计。

环境恢复后先做 import/只读审计，不要直接启动正式完整轨迹。

## 6. 数据、缓存和 checkpoint 放置

### 6.1 官方 MTT 蒸馏数据

正式数据布局应与 manifest 一致：

```text
new_data/mtt_data/georgecazenavette/
├── cifar10_50/
├── cifar100_50/
└── tiny_50/
```

原始归档与校验依据：

```text
/root/autodl-tmp/georgecazenavette.zip
migration/new_official_data_manifest.json
migration/mtt/SHA256SUMS
```

迁移后不要重新随机划分数据；优先恢复既有 split NPZ 和 manifest。路径主要在：

```text
new_cache/new_true_ours/splits/
new_log/new_true_ours/new_true_ours_split_manifest.json
new_configs/new_true_ours/new_true_ours_config_freeze.json
```

### 6.2 高成本缓存

优先原样恢复整个 `new_cache/`。其中关键资产包括：

```text
new_cache/new_public_mtt_mainline/new_full_features/
new_cache/new_true_ours/
new_cache/new_secure_feature/
```

缓存可以重建，但可能耗费数小时或更久。迁移后先核对文件大小与 SHA256，再决定重建。

### 6.3 checkpoint

至少恢复：

```text
new_checkpoints/new_secretflow_spu/full_trajectory_v1/
new_checkpoints/new_secretflow_spu/corrected_v2/
new_checkpoints/new_true_ours/stage_b/
new_checkpoints/new_true_ours/stage_c/
```

checkpoint 只用于恢复、重新评估或继续训练，不是论文数字的唯一证据。论文数字应追溯到
`report + audit + trace + effective_config + CSV/JSON`。永远不要提交 `new_checkpoints/`。

## 7. 新服务器首次恢复流程

### Phase 0：只读硬件和磁盘检查

```bash
cd /root/autodl-tmp/tfe_experiments
date -u
lscpu
free -h
df -h /root/autodl-tmp /root
nvidia-smi || true
cat /proc/meminfo | head
git status --short --branch
```

若是 AutoDL 无卡/2GB 内存模式，只允许读代码、检查 JSON/CSV 和轻量绘图；禁止启动 TFE、
SPU、Ray 或载入大 checkpoint。

### Phase 1：校验迁移文件

迁移包应附带 `SHA256SUMS`。在目标目录执行：

```bash
sha256sum -c SHA256SUMS
```

如果是分层迁移，代码证据包、数据包、缓存包、checkpoint 包和环境包分别校验。任一正式
report、CSV 或配置哈希不符时先停止，不要用新运行覆盖旧文件。

### Phase 2：恢复 Git 和本地工作树

新建服务器时可以先 clone：

```bash
cd /root/autodl-tmp
git clone -b unified-experiments \
  https://github.com/Yukkit-Lee/mpl-optimization.git tfe_experiments
cd tfe_experiments
git rev-parse HEAD
```

然后把迁移包中的未提交/未跟踪文件恢复到工作树。恢复采用非破坏式方式，遇到同名文件先
比较 SHA256 或使用 `rsync --dry-run`；不要直接覆盖用户在新服务器产生的改动。

### Phase 3：环境导入验证

依次验证 modern、TFE、SPU 三套环境。检查失败时只修复对应环境，不要交叉安装依赖。

```bash
/root/autodl-tmp/new_envs/distillation/bin/python -m pip check
/root/autodl-tmp/new_envs/tfe/bin/python -m pip check || true
/root/autodl-tmp/new_envs/spu_e2e/bin/python -m pip check
```

legacy 包可能因为老依赖产生已知的 `pip check` 警告；最终以真实 import 和现有验证脚本为准。

### Phase 4：代码和资产轻量验证

```bash
MODERN_PY=/root/autodl-tmp/new_envs/distillation/bin/python
"$MODERN_PY" scripts/check_autodl_env.py
"$MODERN_PY" scripts/new_verify_official_mtt_assets.py
bash -n new_scripts/*.sh scripts/new_*.sh
"$MODERN_PY" -m py_compile new_scripts/new_paper_figures/*.py
```

如果某个 glob 在新 shell 中没有匹配项，应缩小检查范围，不要把 shell glob 字符串误传给
Python。

### Phase 5：只读验证正式 SPU 证据

正式矩阵核心是：

```text
CIFAR-10:  True Ours / Full Data × seeds 42,43,44
CIFAR-100: True Ours / Full Data × seeds 42,43,44
Tiny:      True Ours / Full Data × seed 42
```

CIFAR-10/100 共 12 个 n=3 cell 已完成并审计；Tiny 按用户决定保持 n=1。对应文件位于：

```text
new_log/new_secretflow_spu/full_trajectory_v1/
new_output/new_secretflow_spu/full_trajectory_v1/
```

使用受保护的只读验收入口：

```bash
bash new_scripts/new_secretflow_spu/full_trajectory_v1/\
new_spu_final_verify_and_shutdown.sh --check
```

注意：该脚本还会检查运行进程、Git 同步及指定产物状态。在新服务器恢复未完成或存在迁移
中的工作树差异时，它可能失败；先读清每条 FAIL，不要为了变绿而删除用户文件或放宽门禁。
绝对不要在新服务器执行其 `--shutdown --confirm-shutdown` 参数，除非用户再次明确授权关机。

## 8. 当前正式 SPU 数字速查

以下是 formal report 中的最终 development accuracy，用于识别迁移是否拿错文件，不代替论文
表格或重新聚合：

| Dataset | Method | seed42 | seed43 | seed44 |
|---|---|---:|---:|---:|
| CIFAR-10 | True Ours | 68.58% | 67.88% | 67.60% |
| CIFAR-10 | Full Data | 72.48% | 72.42% | 72.34% |
| CIFAR-100 | True Ours | 47.74% | 47.84% | 48.46% |
| CIFAR-100 | Full Data | 42.60% | 43.28% | 44.00% |
| Tiny | True Ours | 30.75% | N/A | N/A |
| Tiny | Full Data | 27.56% | N/A | N/A |

每条正式证据还必须满足：

- `status=COMPLETE`
- `run_kind=formal`
- `official_test_read=false`
- `locked_holdout_read=false`
- audit `passed=true`
- Stage-B client-local public reconstruction 为 0
- ABY3/FM128/fxp16 与冻结配置一致
- conditional Stage-C epsilon 不超过 3

若精度吻合但 audit/effective config 缺失，仍不能视为完整迁移。

## 9. 论文 Figures 2--6 的重绘入口

### 9.1 当前论文统一风格输出

从仓库根目录运行：

```bash
cd /root/autodl-tmp/tfe_experiments
/root/autodl-tmp/new_envs/distillation/bin/python \
  new_scripts/new_paper_figures/build_no_title_bold_figures.py
```

输出目录：

```text
new_expFig/paper_experiment/no_title_bold/
```

该入口生成无整图标题、粗体排版的 Figures 2--6，不覆盖旧图。

### 9.2 分图独立运行

```bash
MODERN_PY=/root/autodl-tmp/new_envs/distillation/bin/python
"$MODERN_PY" new_scripts/new_paper_figures/new_build_figure2.py
"$MODERN_PY" new_scripts/new_paper_figures/new_build_figure3.py
"$MODERN_PY" new_scripts/new_paper_figures/new_build_figure4.py
"$MODERN_PY" new_scripts/new_paper_figures/new_build_figure5.py
"$MODERN_PY" new_scripts/new_paper_figures/new_build_figure6.py
```

### 9.3 图与数据源

| Figure | 绘图脚本 | 主要数据源 |
|---|---|---|
| 2(a) TFE | `new_build_figure2.py` | `new_log/new_true_ours/new_true_ours_master_results.json` |
| 2(b) SPU旧短窗 | `new_build_figure2.py` | `new_output/new_secretflow_spu/new_figure2_spu_corrected_v2_plot_data.csv` |
| SPU完整轨迹候选 | `new_plot_spu_full_trajectory.py` | `new_output/new_secretflow_spu/full_trajectory_v1/` 下的聚合数据 |
| 3 | `new_build_figure3.py` | 冻结的 Module-A 九条 JSON |
| 4 | `new_build_figure4.py` | Stage-B initialization-source ablation CSV |
| 5 | `new_build_figure5.py` | Stage-B trajectory JSON |
| 6 | `new_build_figure6.py` | `new_user_tuned_fixed_vs_adaptive_multidataset_formal.json` |

重要版本提示：

- 当前 `new_build_figure2.py` 的 SPU Figure 2(b) 仍读取 corrected-v2 seed42 数据，代码中
  `EXPECTED_SPU_RELEASES` 也是 n=1 口径。
- 后续完成的 CIFAR-10/CIFAR-100 seed43/44 位于 `full_trajectory_v1`，尚不能仅通过运行旧
  Figure 2 builder 自动进入误差条。
- 如果改稿要采用 SPU n=3，应先写/核验独立聚合器：只读取 12 条 formal report/trace，按
  相同归档点对齐，计算 sample SD，并保留 Tiny n=1；不得把 gate 或 corrected-v2 混入。
- Figure 2(a) 的 master JSON 和 builder 在源服务器存在本地改动，迁移时必须保留。
- 绘图脚本已取消输入 JSON/CSV 的 hash hard gate，但保留字段、数量和 scope 验证。不要为
  通过绘图而删除这些科学门禁。

## 10. Table 与报告定位

优先从以下目录查找论文表格：

```text
new_expTab/paper_experiment/
new_expTab/new_paper_adaptive/
new_expTab/new_true_ours/
new_log/unified_experiments/
reportToMe/
```

任何表格数字都应能够追溯到 JSON/CSV。不要从 PNG 像素、终端截图或 Markdown 手工抄数。
重新聚合时记录：数据源列表、seed 数、均值定义、sample SD (`ddof=1`)、单位和 scope。

## 11. 如确需补实验的标准流程

论文已投稿，只有用户明确要求补实验时才执行本节。

### 11.1 开始前

```bash
cd /root/autodl-tmp/tfe_experiments
git status --short --branch
df -h /root/autodl-tmp
free -h
lscpu
nvidia-smi || true
tmux ls || true
pgrep -af 'python|ray|SPURuntime|tensorflow' || true
```

必须先：

1. 确认目标 cell 没有 `COMPLETE + audit PASS`，避免重复运行。
2. 读取冻结配置和原始正式 report 的 effective config。
3. 创建 tmux，记录精确命令、环境路径、seed、CPU/GPU 和数据路径。
4. 检查磁盘与内存；SPU/TFE 长实验必须有 checkpoint/恢复方案。
5. 不再把 CPU/BLAS 线程固定为 1；在正确性允许时发挥机器性能。
6. 正式实验不得读取 official test 或 locked holdout，除非协议明确授权并在报告中记录。

### 11.2 运行中

- 每次只按用户批准的并行度执行。
- 监控日志、checkpoint 时间戳、CPU/RSS、cgroup `memory.events` 和磁盘。
- Ray metrics exporter warning 通常不等于训练失败；以 worker 活性、checkpoint 和 report 为准。
- 不从失败、部分或中断运行复制 timing/accuracy 到正式结果。

### 11.3 完成后

必须依次完成：

```text
report COMPLETE
→ audit PASS
→ effective config / trace / scope 检查
→ 图表或表格聚合
→ 更新 progressLog/project_progress.md
→ 精确 git add 本次小型产物（排除 checkpoint/private runtime）
→ commit
→ push origin unified-experiments
```

仓库中已有用户无关的脏文件时，只暂存本次明确修改的文件，禁止 `git add -A`。

## 12. 常见故障与处理

### 12.1 `ModuleNotFoundError: opacus`

说明使用了系统 Python 或错误环境。改用：

```bash
/root/autodl-tmp/new_envs/distillation/bin/python <script.py>
```

### 12.2 TFE custom op 无法加载

检查 Python/TF 版本和 `.so` 是否位于当前 TFE 环境的 site-packages。不要把为 TF 1.15.2
构建的文件无审计地当作 TF 1.15.5 正式证据；按构建脚本重新校验 SHA256 和 import。

### 12.3 SPU/Ray 日志出现 metrics exporter 错误

先检查：

```bash
ps -eo pid,stat,etime,pcpu,pmem,args --sort=-pcpu | head -40
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp
```

若三个 SPU runtime worker 仍持续计算、checkpoint 继续更新且 `oom_kill=0`，不要只因 telemetry
warning 杀掉正式实验。

### 12.4 Git 仓库很大

源仓库 `.git` 曾约 19GB，因为历史中存在大对象。新服务器可以从远端 clone，但迁移前必须
先确保源服务器的本地未提交文件已另行保存。不要擅自重写 Git 历史或强推。

### 12.5 磁盘不足

优先保留：正式 JSON/CSV/配置/图表、原始数据、`new_cache/`、最终 SPU checkpoint、最终
TFE checkpoint。可删除项只限已确认的 Ray 临时目录、包缓存和隔离的失败运行；删除前先列出
精确路径并取得用户许可。

## 13. 安全与隐私

以下内容不得进入仓库、公开云盘或共享迁移包：

- `/root/.config/codex-secrets/`
- GitHub token、SSH 私钥、代理密码
- shell history 中的凭据
- 私有原始对话 JSONL
- 任何未脱敏的内部数据

如需新服务器 Git push，在新机器重新配置凭据，不要从 Markdown 或日志恢复 token。

## 14. 新服务器接管完成判定

只有以下各项全部满足，才可以说迁移完成：

- [ ] 当前分支为 `unified-experiments`，正式 HEAD 可追溯到 `ca2ced9a` 或其后继提交。
- [ ] 迁移包 SHA256 全部通过。
- [ ] 源服务器的本地修改和未跟踪正式文件没有丢失。
- [ ] MTT 蒸馏数据、原始数据归档和固定 split 可读取。
- [ ] `new_cache/` 中论文依赖的特征缓存存在。
- [ ] modern、TFE、SPU 三套 Python 环境各自 import 成功。
- [ ] TFE secure-random custom ops 能正确载入。
- [ ] 12 条 CIFAR-10/CIFAR-100 SPU formal report 与 audit 齐全。
- [ ] Tiny 两条 seed42 formal report 与 audit 齐全。
- [ ] Figures 2--6 可以从保存的 JSON/CSV 重画，不需要读取 checkpoint 才能得到原数字。
- [ ] 当前论文源文件、投稿 PDF、补充材料和 `.bib` 已独立备份。
- [ ] 没有 token、私钥或私有 JSONL 混入迁移包。
- [ ] 至少在两个独立位置保留迁移备份，并实际抽查解压和读取。

## 15. 给新 Agent 的首条任务模板

可以把下面文字直接发送给新服务器上的 Agent：

```text
这是论文投稿后的改稿环境。先完整阅读 AGENTS.md、
forNewServer/NEW_SERVER_OPERATION_GUIDE.md、progressLog/project_progress.md 末尾、
reportToMe/unified_experiment_report.md 和
reportToMe/new_paper_figure_source_map_0809.md。

当前权威分支是 unified-experiments。先只读检查 Git、硬件、磁盘、三套环境、数据、
缓存、formal report/audit 和迁移 SHA256；不要启动实验，不要删除文件，不要覆盖本地改动，
不要读取 official test/locked holdout。报告恢复缺口和可重画的论文图表后等待我的下一步指示。
```

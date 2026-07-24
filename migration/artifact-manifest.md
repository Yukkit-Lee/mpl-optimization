# 迁移资产清单

盘点时间：2026-07-24 23:44（Asia/Shanghai）

## 当前主线必需资产

| 资产 | 当前路径 | 规模 | 用途 | 可重新下载 | AutoDL 建议路径 | 迁移包 |
|---|---|---:|---|---|---|---|
| Official CIFAR-10 ipc50 | `/home/lixujie/tmpWork/mtt_data/georgecazenavette/cifar10_50` | 约 5.9 MiB | 模块 A/B/C 与 end-to-end | 官方来源，但不依赖其持续可用 | `$OFFICIAL_MTT_ROOT/cifar10_50` | 包含 |
| Official CIFAR-100 ipc50 | `/home/lixujie/tmpWork/mtt_data/georgecazenavette/cifar100_50` | 约 58.6 MiB | 模块 A/B/C 与 end-to-end | 官方来源，但不依赖其持续可用 | `$OFFICIAL_MTT_ROOT/cifar100_50` | 包含 |
| Official Tiny ipc50 | `/home/lixujie/tmpWork/mtt_data/georgecazenavette/tiny_50` | 约 469 MiB | 模块 A/B/C 与 end-to-end | 官方来源，但不依赖其持续可用 | `$OFFICIAL_MTT_ROOT/tiny_50` | 包含 |
| CIFAR-10 原始归档 | `/home/lixujie/tmpWork/mtt_cifar10_work/data/cifar-10-python.tar.gz` | 163 MiB | 训练/测试集 | 可从公开镜像下载 | `$DATA_ROOT/mtt_cifar10_work/data/` | 包含 |
| CIFAR-100 原始归档 | `/home/lixujie/tmpWork/mtt_cifar100_work/data/cifar-100-python.tar.gz` | 162 MiB | 训练/测试集 | 可从公开镜像下载 | `$DATA_ROOT/mtt_cifar100_work/data/` | 包含 |
| Tiny ImageNet 原始 zip | `/home/lixujie/tmpWork/mtt_tiny_work/data/tiny-imagenet-200.zip` | 237 MiB | 训练/测试集 | 下载源不稳定，建议从旧服务器/内网取 | `$DATA_ROOT/mtt_tiny_work/data/` | 包含 |
| MTT 真实源码快照 | `migration/mtt/` | 约 40 MiB | 恢复本地 MTT 修改 | 不可仅靠公开仓库重建 | `$PROJECT_ROOT/migration/mtt` | 包含 |

上述文件的 SHA256 记录在 `migration/artifact-sha256.txt`；迁移包根目录另有相对路径
`SHA256SUMS`。

## 已盘点但不进入最小包

| 资产 | 当前路径 | 规模 | 用途/原因 | 是否可重建 | AutoDL 建议 |
|---|---|---:|---|---|---|
| Tiny expert buffers | `/home/lixujie/tmpWork/mtt_tiny_work/buffers_tiny` | 1.7 GiB | 重新做 Tiny MTT 蒸馏 | 可重建但耗时高 | 若继续蒸馏，从旧服务器单独传到 `$DATA_ROOT/mtt_tiny_work/buffers_tiny` |
| CIFAR-100 expert buffers | `/home/lixujie/tmpWork/mtt_cifar100_work/buffers_cifar100` | 984 MiB | 重新做 CIFAR-100 MTT 蒸馏 | 可重建但耗时高 | 从旧服务器单独传到 `$DATA_ROOT/mtt_cifar100_work/buffers_cifar100` |
| MTT 仓库内 CIFAR-10 buffers | `/mnt/code/lixujie/mtt-distillation/buffers` | 约 6.1 GiB | 旧蒸馏实验 | 可重建但耗时高 | 仅在复现旧实验时内网传输 |
| 手工蒸馏数据 | `/mnt/code/lixujie/tfe_experiments/distilledData` | 261 MiB | 历史 ipc1/ipc10 结果 | 不可低成本重建 | 已记录 SHA256；如需旧消融，从旧服务器传到 `$PROJECT_ROOT/distilledData` |
| 旧特征缓存 | `/mnt/code/lixujie/tfe_experiments/featureData` | 17 MiB | HOG/RoBERTa/DINO 旧模块 A | 可重新提取 | 已记录 SHA256；按需传到 `$PROJECT_ROOT/featureData` |
| 模块 B init 模型 | `/mnt/code/lixujie/tfe_experiments/init_model` | 151 KiB | 旧 feature-based B→C 流程 | 可重训 | 已记录 SHA256；按需传到 `$PROJECT_ROOT/init_model` |
| 模块 C final 模型 | `/mnt/code/lixujie/tfe_experiments/log/finalModel` | 112 KiB | 旧实验最终权重 | 可重训 | 已记录 SHA256；按需传到 `$OUTPUT_ROOT/log/finalModel` |
| MTT 旧 checkpoints/蒸馏快照 | `/mnt/code/lixujie/mtt-distillation` 中 223 个 `.pt/.pth` | 合计约 6.9 GiB | 旧 MPC/STL10/CIFAR10 实验 | 部分不可替代 | 文件列表见 `migration/mtt/untracked-all.txt`；如需复现旧分支必须从旧服务器/内网传输 |

## 必须从旧服务器或内网补取的资产

最小包足够继续当前 official distilled CIFAR-10/CIFAR-100/Tiny 的 A+B+C 实验。以下内容
仅在恢复历史/重新蒸馏时必须从旧服务器或内网单独下载：

1. Tiny 与 CIFAR-100 expert buffers。
2. MTT 仓库内 223 个历史 `.pt/.pth` checkpoints 和旧 logged files。
3. `distilledData/` 中手工生成的 ipc1/ipc10 数据（若论文仍需引用这些旧消融）。
4. 完整 `featureData/`（若继续旧 HOG/RoBERTa/DINO 路线且不想重新提取）。

未删除、未移动上述任何原始文件。


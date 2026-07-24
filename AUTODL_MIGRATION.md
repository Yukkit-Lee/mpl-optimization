# AutoDL 迁移与恢复

本文档对应 `autodl-migration` 分支。现代蒸馏/GPU 环境与旧 Python 3.6/TFE 环境必须分开。

## 1. 克隆项目

```bash
cd /root/autodl-tmp
git clone -b autodl-migration https://github.com/Yukkit-Lee/mpl-optimization.git mpl-optimization
cd /root/autodl-tmp/mpl-optimization
```

## 2. 创建现代 GPU 环境

推荐先使用可迁移配置：

```bash
conda env create -f environment.yml
conda activate mpl-distillation
python scripts/check_autodl_env.py
```

如果镜像中的 Conda 求解失败，可先创建 Python 3.9，再使用 pip 核心依赖：

```bash
conda create -n mpl-distillation python=3.9 pip -y
conda activate mpl-distillation
pip install -r requirements.txt
```

当前实测关键版本：

- Ubuntu 24.04.1，Linux 6.17.0-40。
- Python 3.9.7。
- NVIDIA RTX 4090 24 GiB，driver 580.159.03。
- PyTorch 2.8.0+cu128，torchvision 0.23.0+cu128，CUDA runtime 12.8。
- NumPy 1.22.4，SciPy 1.7.3，scikit-learn 1.6.1，Matplotlib 3.5.3。

完整审计快照位于 `migration/env_snapshots/`，其中包括 pip freeze、Conda export、
`conda list --explicit`、系统和 GPU 信息。文件已移除本机 prefix 和 `file:///` 路径。

## 3. 配置路径

```bash
cp config/paths.autodl.example.env config/paths.autodl.env
source config/paths.autodl.env
mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT"
python -c "from project_paths import print_resolved_paths; print_resolved_paths()"
```

所有最新入口统一支持：

- `PROJECT_ROOT`
- `MTT_ROOT`
- `DATA_ROOT`
- `OFFICIAL_MTT_ROOT`
- `OUTPUT_ROOT`

## 4. 恢复真实 MTT 修改

不要只从公开 MTT 仓库下载。项目保存了当前服务器的完整 Git history、tracked patch 和
未跟踪源码：

```bash
source config/paths.autodl.env
bash scripts/restore_mtt_snapshot.sh
```

恢复后检查：

```bash
git -C "$MTT_ROOT" status --short --branch
(cd "$PROJECT_ROOT" && sha256sum -c migration/mtt/SHA256SUMS)
```

## 5. 恢复迁移包资产

将 `autodl_migration_bundle_时间戳` 上传到 `/root/autodl-tmp/`，然后执行：

```bash
bundle=/root/autodl-tmp/autodl_migration_bundle_时间戳
source "$PROJECT_ROOT/config/paths.autodl.env"

mkdir -p "$DATA_ROOT/mtt_data" \
  "$DATA_ROOT/mtt_cifar10_work/data" \
  "$DATA_ROOT/mtt_cifar100_work/data" \
  "$DATA_ROOT/mtt_tiny_work/data"

cp -a "$bundle/assets/official_mtt/georgecazenavette" "$DATA_ROOT/mtt_data/"
cp "$bundle/assets/dataset_archives/cifar-10-python.tar.gz" "$DATA_ROOT/mtt_cifar10_work/data/"
cp "$bundle/assets/dataset_archives/cifar-100-python.tar.gz" "$DATA_ROOT/mtt_cifar100_work/data/"
cp "$bundle/assets/dataset_archives/tiny-imagenet-200.zip" "$DATA_ROOT/mtt_tiny_work/data/"

tar -xzf "$DATA_ROOT/mtt_cifar10_work/data/cifar-10-python.tar.gz" \
  -C "$DATA_ROOT/mtt_cifar10_work/data"
tar -xzf "$DATA_ROOT/mtt_cifar100_work/data/cifar-100-python.tar.gz" \
  -C "$DATA_ROOT/mtt_cifar100_work/data"
unzip -q "$DATA_ROOT/mtt_tiny_work/data/tiny-imagenet-200.zip" \
  -d "$DATA_ROOT/mtt_tiny_work/data"

python "$PROJECT_ROOT/scripts/verify_artifacts.py" \
  "$bundle/SHA256SUMS" --root "$bundle"
python "$PROJECT_ROOT/scripts/check_autodl_env.py"
```

## 6. 运行最新实验

所有命令默认单线程 CPU 辅助、GPU 训练，不改变原随机种子和实验超参数。

CIFAR-10：

```bash
conda activate mpl-distillation
source config/paths.autodl.env
python exp_official_distilled_moduleA_verify.py --dataset cifar10
python exp_module_b_kd_noniid_official.py --dataset cifar10 --seeds 42,43,44
python exp_moduleC_dp_optimization.py --dataset cifar10 --seeds 42,43,44
python exp_end2end_framework.py --dataset cifar10 --seeds 42,43,44
```

CIFAR-100：

```bash
python exp_official_distilled_moduleA_verify.py --dataset cifar100
python exp_module_b_kd_noniid_official.py --dataset cifar100 --seeds 42,43,44
python exp_moduleC_dp_optimization.py --dataset cifar100 --seeds 42,43,44
python exp_end2end_framework.py --dataset cifar100 --seeds 42,43,44
```

Tiny ImageNet：

```bash
python exp_official_distilled_moduleA_verify.py --dataset tiny
python exp_module_b_kd_noniid_official.py --dataset tiny --seeds 42,43,44
python exp_moduleC_dp_optimization.py --dataset tiny --seeds 42,43,44
python exp_end2end_framework.py --dataset tiny --seeds 42,43,44
```

## 7. 旧 TFE 环境（隔离）

旧环境实测为 Python 3.6.13、TensorFlow 1.15.5、TF-Encrypted 0.8.0、PyTorch 1.10.1。
它不能与 RTX 4090 的现代 PyTorch 环境合并。

```bash
conda env create -f environment-tfe-legacy.yml
conda activate mpl-tfe-legacy
pip install -r requirements-tfe-legacy.txt
```

如需最大程度复刻原服务器，可先用
`migration/env_snapshots/tfe-legacy-conda-explicit.txt` 恢复 Conda 包，再安装 legacy
requirements。该环境目前缺 TF-Encrypted `aux` 和 `secure_random` 自定义 `.so`，会回退到
不安全随机数；在声称“生产级密态安全”前必须针对 AutoDL 的 TF 1.15.5 重新编译 custom ops。

## 8. 安全与限制

- 不把 token、`.env`、数据、权重、迁移包提交到 Git。
- 不在 TFE 环境运行 RTX 4090 深度模型。
- 正式长实验前先运行 `scripts/check_autodl_env.py`。
- 当前迁移过程没有删除、移动或覆盖原始数据，也没有执行正式长实验。

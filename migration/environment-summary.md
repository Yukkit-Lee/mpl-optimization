# 实际环境摘要

采集时间：2026-07-24（Asia/Shanghai）

## 现代 distillation 环境

- Python：3.9.7
- PyTorch：2.8.0+cu128
- torchvision：0.23.0+cu128
- PyTorch CUDA runtime：12.8
- CUDA 可用：是
- GPU：NVIDIA GeForce RTX 4090，24,564 MiB，compute capability 8.9
- NVIDIA driver：580.159.03
- NumPy：1.22.4
- SciPy：1.7.3
- scikit-learn：1.6.1
- Matplotlib：3.5.3

## 旧 TFE 环境

- Python：3.6.13
- TensorFlow：1.15.5
- TF-Encrypted：0.8.0
- NumPy（实际 import）：1.19.2
- PyTorch：1.10.1
- torchvision：0.11.2
- timm：0.4.12

旧环境导入 TF-Encrypted 时确认缺少：

- `operations/aux/aux_module_tf_1.15.5.so`
- `operations/secure_random/secure_random_module_tf_1.15.5.so`

因此当前会回退到 insecure randomness。该问题不阻止旧原型运行，但在将实验描述为生产级
安全 MPC 前必须重新编译 custom ops。

## 系统

- Ubuntu 24.04.1 LTS
- Kernel：6.17.0-40-generic
- Conda：22.9.0

详细输出见 `migration/env_snapshots/`。


# New TFE Custom Ops Build Record

## 用途

记录新安全特征路线在旧版 TensorFlow 1.15.5 / TF Encrypted 0.8.0
环境中实际使用的 `aux` 与 `secure_random` 自定义算子，避免把安全随机回退
误写成 MPC 安全实现。

## 构建来源

- TFE 源码目录：`/home/lixujie/tmpWork/new_tf_encrypted_src`
- 源码 commit：`51de98f9c2208d5d4bd716bf7316d3cd4e0bf434`
- TensorFlow：`1.15.5`
- TF Encrypted：`0.8.0`
- 安装环境：`/home/lixujie/.conda/envs/tfe`

## 安装产物

- `aux_module_tf_1.15.5.so`
  - SHA256：`d0522709c817c63e770327fc69e5dc4e422a9f3a7cd3e0af19b0a4375a485c73`
- `secure_random_module_tf_1.15.5.so`
  - SHA256：`7f3ab2d053761f1f60eb511547d0839285ffc341788f7f5b179c8e52dc24d340`

二进制文件只安装在旧 TFE 环境中，不提交 Git。迁移到新机器时必须针对目标
TensorFlow ABI 重新构建，并重新运行
`new_tests/new_secure_feature/new_test_tfe_custom_ops.py` 和
`new_tests/new_secure_feature/new_test_tfe_zero_sharing.py`。
## 已验证性质

- `aux` comparison、exp 和 inverse-sqrt 均可调用；
- `secure_random` 相同 8×`int32` seed 产生完全相同的输出；
- 非种子安全随机输出不是常量；
- 16 组 `2048×3` ABY3 算术零共享在模 `2^64` 下严格求和为零；
- 未触发 insecure randomness fallback。

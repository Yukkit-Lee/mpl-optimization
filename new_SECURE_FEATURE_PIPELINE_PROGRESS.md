# New Secure Feature Pipeline Progress

## Isolation

- Branch: `new-secure-feature-pipeline`
- Baseline commit: `0a7a254b6313b39dc8d367d28825028286a0750e`
- Previous complete-ConvNet results remain on `cross-stage-refactor`.
- Every artifact produced by this route must use a `new_` prefix.
- Test data is readonly and cannot participate in tuning or selection.

## Resource Contract

- Two RTX 4090 devices may be used, one process per GPU.
- Every process fixes CPU, BLAS and PyTorch threads to 1.
- `DataLoader(num_workers=0, persistent_workers=False)`.
- No multiprocessing pool or implicit parallel preprocessing.
- Persistent runtime: tmux 3.7b from `/home/lixujie/tmpWork/new_tmux_runtime`.
- Session: `new_secure_feature_formal`.
- Heartbeat interval: 60 seconds.
- Checkpoint/resume and at most two automatic retries are mandatory.
- Disk safety threshold: at least 25 GiB free on both project and home filesystems.

## Checklist

- [x] Phase 0: Preserve baseline and create branch
- [x] Phase 1: Implement real Stage-A privacy mechanism (secure RNG blocker recorded)
- [x] Phase 2: Build shared 2048-D representation pipeline
- [x] Phase 3: Implement secure Stage-B computation
- [x] Phase 4: Implement secure Stage-C Fixed DP-SGD
- [ ] Phase 5: Implement secure Stage-C Adaptive DP-SGD
- [ ] Phase 6: Complete Module-A formal experiments
- [ ] Phase 7: Complete Module-B formal experiments
- [ ] Phase 8: Complete Module-C formal experiments
- [ ] Phase 9: Complete end-to-end formal experiments
- [ ] Phase 10: TFE system evaluation and final reporting

## Phase 0 Evidence

- GPUs: 2 x NVIDIA GeForce RTX 4090, 24,564 MiB each.
- RAM: 188 GiB total, 183 GiB initially available.
- Project filesystem: 110 GiB initially available.
- Home filesystem: 61 GiB initially available.
- Modern environment: Python 3.9.7, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, cuDNN 91002, Opacus 1.5.4.
- Legacy environment: Python 3.6.13, TensorFlow 1.15.5, TF Encrypted 0.8.0.
- TFE aux 与 secure-random custom ops 已针对 TF 1.15.5 编译、安装并通过测试。
- tmux was absent from the host and installed into isolated `new_tmux_runtime`; the training environment was not modified.
- Baseline artifact inventory: `new_log/new_secure_feature/json/new_phase0_baseline_artifacts.json`.
- Environment snapshot: `new_log/new_secure_feature/json/new_phase0_environment.json`.
- Long-running session `new_secure_feature_formal` contains independent `gpu0`, `gpu1`, `tfe`, `monitor`, and `report` windows.
- Queue workers use per-resource file locks, 60-second heartbeats, checkpoint paths, two retries, and resume markers.

## Quality Gates

- Stage A public benchmark: at least 95% of official MTT reference accuracy and within the stricter 2 percentage-point rule.
- Stage A DP-MTT: privacy-utility curve; more than 10 points below public MTT remains `NOT_READY_FOR_PAPER`.
- Stage B: within 0.5 point of random final accuracy and at least 20% faster Step@TargetAcc.
- Stage B severe Non-IID: KD-ReferenceWeighted at least 1 point above NoKD.
- Stage C: Adaptive at least 1 point above Fixed and EarlyAUC not lower at matched privacy settings.
- End-to-End: at least 2-point final improvement or 20% Step@TargetAcc reduction without material regression.

## Current Status

- Phase 0 infrastructure is complete and is being committed independently.
- Phase 1 implementation and minimal trajectory-to-distillation correctness chain are complete.
- Phase 1 formal secure-RNG execution is blocked by the CUDA 10.2-only `torchcsprng 0.2.1` wheel; no silent fallback is permitted.
- Phase 2 shared 2048-D encoder implementation is in progress.
- No smoke result will be promoted into a formal result.

## Phase 1 Evidence

- DP expert diagnostic: CIFAR-10, 1,000 disjoint records, 8 optimizer steps, one epoch, epsilon=1.761323 at delta=1e-5.
- DP-MTT diagnostic: two trajectory-matching iterations, random-noise image initialization, no raw-data loader, inherited epsilon=1.761323.
- Unit tests: epsilon increases with steps, decreases with noise, sequential composition is monotonic, raw-data isolation passes, and DP/public output namespaces are separate.
- All three dataset models produce exactly 2048 classifier-input features and pass the Opacus module validator.
- These are correctness diagnostics only and are not formal accuracy results.

## Phase 2 Evidence

- Public CIFAR-10 correctness source contains 500 official ipc=50 samples.
- Class-stratified split is 400 encoder-train / 50 reference / 50 selection with no overlap.
- One shared encoder checkpoint produced SHA256 `da49c86e20e1cb8e3720d7816f1f1d642c33345c325c94c8c68faa1a06aa6ead`.
- Train/reference/selection feature tensors have shapes 400x2048, 50x2048 and 50x2048.
- Scaler scope is `synthetic_encoder_train_only`; every output feature has L2 norm at most 1.
- Test data was not loaded during correctness or tuning.
- The one-epoch reference accuracy of 0.14 is diagnostic only and is not a formal quality result.

## Phase 3 Evidence

- TFE source commit: `51de98f9c2208d5d4bd716bf7316d3cd4e0bf434`.
- `aux_module_tf_1.15.5.so` SHA256: `d0522709c817c63e770327fc69e5dc4e422a9f3a7cd3e0af19b0a4375a485c73`.
- `secure_random_module_tf_1.15.5.so` SHA256: `7f3ab2d053761f1f60eb511547d0839285ffc341788f7f5b179c8e52dc24d340`.
- 16 组 `2048x3` ABY3 zero sharing 在模 `2^64` 下严格抵消。
- 2048 维 private matmul、单头更新和重复更新 correctness 通过。
- 已定位并修复旧 TFE 多变量 assign 竞态；三次独立三头测试最大误差均低于 `3.1e-5`。
- 完整 local MSE、teacher aggregation、logit-MSE KD、Avg/RW 构造、secure score comparison 与 mux correctness 通过。
- 完整链路相对 NumPy reference 最大误差低于 `9.6e-5`。
- 默认只 reveal 最终 selected initializer；correctness 额外 reveal 均写入 JSON。
- 当前执行是单机 ABY3 模拟；网络通信 bytes 尚未测量。

## Phase 4 Evidence

- 2048-D linear squared-loss per-example gradient 使用解析公式在 ABY3 中计算。
- norm、clip factor、clipped gradient、noise 与 updated model 均保持 secret-shared。
- 三个逻辑方各自用 checkpointed CSPRNG seed/counter 生成 `N(0,1/3)` Gaussian share。
- 三方总噪声 correctness 样本均值 `-0.0089`、标准差 `1.0113`。
- logits、norm、factor、clipped gradient 与最终参数均通过 NumPy correctness reference。
- 最终参数最大绝对误差低于 `2.3e-5`；三个 provider counter 均从 0 推进到 1。
- Stage-C epsilon 使用 Opacus RDP accountant，组合 gradient 与 clipping-count queries。
- Stage-A original-record DP 与 Stage-C distilled-feature-record DP 分开报告，不相加。

# New Secure Feature Pipeline Progress

## Isolation

- Branch: `new-secure-feature-pipeline`
- Baseline commit: `0a7a254b6313b39dc8d367d28825028286a0750e`
- Previous complete-ConvNet results remain on `cross-stage-refactor`.
- Every artifact produced by this route must use a `new_` prefix.
- Test data is readonly and cannot participate in tuning or selection.

## Resource Contract

- Public MTT 主线允许 GPU 0/1 各运行一个相互独立的 PyTorch 正式任务；
  每张卡最多一个进程并使用独立锁。首个 feature-quality 队列仅使用 GPU 0。
- Every process fixes CPU, BLAS and PyTorch threads to 1.
- `DataLoader(num_workers=0, persistent_workers=False)`.
- No multiprocessing pool or implicit parallel preprocessing.
- Persistent runtime: tmux 3.7b from `/home/lixujie/tmpWork/new_tmux_runtime`.
- Session: `new_secure_feature_single_gpu_formal`.
- `systemd --user` supervisor 依据磁盘活动队列恢复重启后丢失的 tmux。
- Heartbeat interval: 60 seconds.
- Checkpoint/resume and at most two automatic retries are mandatory.
- Disk safety threshold: at least 25 GiB free on both project and home filesystems.

## Checklist

- [x] Phase 0: Preserve baseline and create branch
- [x] Phase 1: Implement real Stage-A privacy mechanism
- [x] Phase 2: Build shared 2048-D representation pipeline
- [x] Phase 3: Implement secure Stage-B computation
- [x] Phase 4: Implement secure Stage-C Fixed DP-SGD
- [x] Phase 5: Implement secure Stage-C Adaptive DP-SGD
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
- Phase 1 secure-RNG blocker is resolved by the tested project-owned
  `chacha20_box_muller_4sample` backend; no ordinary RNG fallback is permitted.
- Phase 2 shared 2048-D encoder implementation and correctness are complete.
- Phase 6 已完成 CIFAR-10 public MTT 三个正式 seed；CIFAR-100/Tiny
  validation pilot 与其余正式矩阵仍在进行。
- No smoke result will be promoted into a formal result.
- `plan/new_public_mtt_mainline_codex_prompt.md` 已将 Public MTT 设为论文
  主线；DP-MTT 状态为 `DEFERRED_UNTIL_MAIN_PIPELINE_COMPLETE`，不再阻塞
  Phase 7--10。当前先验证 frozen 2048-D features 的 plain linear quality。

## Resume Safety Evidence

- 主机在 2026-07-25 17:48 发生系统重启；未发现 OOM 或 GPU Xid 证据。
- 重启前 CIFAR-10 public Round 3 的三份 checkpoint/status 均完整，恢复时跳过。
- 状态、heartbeat 和活动队列均持久写盘；状态 JSON 使用原子替换并记录 boot ID。
- 单 GPU supervisor 会在活动队列完成时保持 `QUEUE_COMPLETE`，遇到正式任务
  `FAILED` 时转为 `NEEDS_ATTENTION`，不会无限重复失败任务。

## Phase 1 Evidence

- DP expert diagnostic: CIFAR-10, 1,000 disjoint records, 8 optimizer steps, one epoch, epsilon=1.761323 at delta=1e-5.
- DP-MTT diagnostic: two trajectory-matching iterations, random-noise image initialization, no raw-data loader, inherited epsilon=1.761323.
- Unit tests: epsilon increases with steps, decreases with noise, sequential composition is monotonic, raw-data isolation passes, and DP/public output namespaces are separate.
- All three dataset models produce exactly 2048 classifier-input features and pass the Opacus module validator.
- These are correctness diagnostics only and are not formal accuracy results.
- Secure-RNG diagnostic v2 used 1,000 disjoint records, 8 optimizer steps and
  completed with epsilon `1.761323` at delta `1e-5`.
- A 200,000-sample statistical test at target std `1.7` produced mean
  `-0.012484` and std `1.701208`; checkpoint replay was bitwise identical.
- Normal checkpoints contain no CSPRNG key. The external key sidecar has mode
  `0600`; trajectory output exposes only its SHA256 fingerprint.

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

## Phase 5 Evidence

- 当前 `C_t` 裁剪当前梯度，`C_{t+1}` 只在同一步末尾原子提交。
- 一步顺序测试中旧 C 的 clip-factor 最大误差为 `0.00153`。
- 加噪比例 `0.3093` 时，C 从 `0.699997` 更新到 `0.706711`，误差低于 `7.4e-6`。
- 支持 C_min/C_max、warm-up、update interval 和连续撞击 C_max 的 hard stop。
- checkpoint 包含模型、C_t、history、撞界计数以及每方 CSPRNG seed/counter。
- 支持 global 与 parameter-block clipping；正式 Adaptive utility 与消融仍在 Phase 8 验证。
- 隐私范围、test leakage 与 no plaintext fallback 共 6 项自动检查通过。

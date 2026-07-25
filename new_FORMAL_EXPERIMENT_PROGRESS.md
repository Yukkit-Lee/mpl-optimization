# New Secure Feature Formal Experiment Progress

## Completion Rule

本文件只记录 `new_secure_feature_pipeline` 的正式实验。correctness、smoke 和 pilot
不能标记正式完成；所有超参数必须先在 reference/validation set 冻结，test set
只允许在正式配置冻结后只读评估。

## Resolved Resource Policy

- Branch: `new-secure-feature-pipeline`
- CPU threads per process: 1
- DataLoader workers: 0
- GPU processes: exactly one formal process on GPU 0; GPU 1 is intentionally unused
- TFE processes: serial, protected by `/tmp/new_secure_feature_tfe.lock`
- Persistent session: `new_secure_feature_single_gpu_formal`
- Reboot recovery: `systemd --user` supervisor reloads the active queue from disk
- Heartbeat: every 60 seconds
- Automatic retries: at most 2
- Checkpoint/resume: mandatory
- Test tuning: forbidden

## Planned Formal Runs

| Phase | Scope | Minimum runs | Status |
|---|---:|---:|---|
| Stage-A DP experts | 3 datasets x 3 seeds x 5 parties x 2 experts | 90 | PENDING |
| Stage-A DP-MTT | 3 datasets x 3 seeds | 9 | PENDING |
| Module-A comparison | 3 datasets x 3 seeds x 4 sources | 36 | IN_PROGRESS (official MTT 9/9 complete) |
| Module-B formal | convergence + IID/Mild/Severe | 99 | PENDING |
| Module-C formal | Plain + Fixed/Adaptive at 3 privacy points | 72 | PENDING |
| End-to-End | 3 datasets x 3 seeds x 6 ablations | 54 | PENDING |
| TFE system scales | 3 datasets x small/medium/largest-feasible | 9 | PENDING |
| **Minimum total** | tuning trials excluded | **369** | PENDING |

## Phase Status

- [x] Phase 0: runtime isolation and audit infrastructure
- [x] Phase 1 implementation: accountant-backed DP experts and CSPRNG
- [x] Phase 2 implementation: shared frozen 2048-D encoder
- [x] Phase 3 implementation: TFE/ABY3 Stage B
- [x] Phase 4 implementation: secure Fixed DP-SGD
- [x] Phase 5 implementation: secure Adaptive DP-SGD
- [ ] Phase 6 formal Module-A
- [ ] Phase 7 formal Module-B
- [ ] Phase 8 formal Module-C
- [ ] Phase 9 formal End-to-End
- [ ] Phase 10 formal TFE system report

## Open Quality Gates

- Public MTT representation must reach the stricter of 95% official accuracy or
  an absolute gap no larger than 2 percentage points.
- DP-MTT more than 10 points below public MTT remains `NOT_READY_FOR_PAPER`.
- Secure Stage-B chosen initialization must reduce Step@TargetAcc by at least
  20% without losing more than 0.5 accuracy point at convergence.
- Severe Non-IID Secure KD-ReferenceWeighted target gain is at least 1 point.
- Adaptive Stage-C target gain is at least 1 point over Fixed at matched privacy,
  with no lower EarlyAUC and no exploding clipping bound.
- End-to-End must improve final accuracy by at least 2 points or reduce
  Step@TargetAcc by at least 20% without material regression.

## Current Evidence

- Secure RNG diagnostic: CIFAR-10, 1,000 records, 8 steps,
  epsilon=1.761323 at delta=1e-5.
- Backend: `chacha20_box_muller_4sample`; no `torch.normal` fallback.
- Key storage: external mode-0600 sidecar; project checkpoint stores only
  counter and key fingerprint.
- CIFAR-10 public MTT Round 3 已完成正式 seeds 42/43/44，test accuracy
  分别为 0.6788/0.6856/0.6704，mean=0.6783、sample std=0.0076；
  三个 seed 均通过严格门槛 0.6684，后续恢复时跳过。
- CIFAR-10 DP-MTT validation 当前 test accuracy=0.3020，落后 public
  约 37.6 个百分点，因此保持 `NOT_READY_FOR_PAPER` 并继续 validation 调参。
- CIFAR-100 与 Tiny ImageNet public MTT reference-only plateau pilots 已完成；
  两者均在 reference 上冻结 300 个正式 epoch，pilot 不读取 test。
- Public MTT 三数据集正式 9 seeds 已全部完成且通过严格质量门：
  CIFAR-10 `0.6783 +/- 0.0076`、CIFAR-100 `0.4462 +/- 0.0055`、
  Tiny ImageNet `0.2641 +/- 0.0054`。逐 seed 的 test 只在 pilot 冻结配置后
  进行一次只读评估。
- 正式结果已写入
  `new_expTab/new_secure_feature/new_phase6_public_mtt_{results,summary}.csv`
  和 `new_expFig/new_secure_feature/new_phase6_public_mtt_reference_curves.*`。
  正式 encoder 使用全部 synthetic 样本，因此图中 reference 曲线只作为
  冻结后拟合诊断，不能解释为独立 validation 泛化曲线；质量门以只读 test
  结果为准。
- CIFAR-10 完整 DP-MTT validation 已在单 GPU 队列中启动：
  5 parties x 2 experts x 50 epochs -> 5000 MTT iterations ->
  reference-only encoder pilot。当前仍是 tuning/validation，不计入正式均值。
- CIFAR-10 DP-MTT round 2 已完成并在冻结后只读 test 得到 `0.3520`，
  相对 public mean 低 `0.3263`，状态为 `NOT_READY_FOR_PAPER`。encoder
  reference pilot 的最佳值为 `0.90`，继续增加 encoder epoch 不能解决差距。
- 下一轮 tuning 将复用相同 DP trajectories 和相同 Stage-A epsilon，优先搜索
  MTT 的轨迹起点/目标跨度；新增 external public-MTT reference evaluator，
  该入口不加载 raw private data 或 benchmark test，避免继续用 test 调参。
- Round 3 coarse plan 固定 2500 MTT iterations，对比 `late3`
  (`max_start=45, expert_epochs=3, syn_steps=20`) 与 `late10`
  (`max_start=40, expert_epochs=10, syn_steps=30`)。两者复用相同的 10 条
  DP trajectories，选择指标仅为 external public-MTT reference accuracy；
  round2 reference baseline 为 `0.2920`。
- 2026-07-25 21:09 CST：round3 已由持久化 single-GPU supervisor 启动。
  运行约束为 CPU 单核、线程数 1、`num_workers=0`、仅 GPU0；每 100 steps
  保存 checkpoint、每 60 秒写 heartbeat，异常时最多自动重试 2 次。
- 2026-07-25 21:31 CST：round3 完成。`late3` 与 `late10` 的 external
  public-reference accuracy 分别为 `0.2940` 和 `0.3280`；二者均未加载
  benchmark test/raw private data。`late10` 相对 round2 baseline 提升
  3.6pp，进入 round4：从 checkpoint 续跑 2500→5000 MTT iterations。
- Round4 首次恢复诊断：`map_location=cuda` 会把 checkpoint 内 CPU RNG
  state 一并移动到 GPU，导致 PyTorch 拒绝恢复。已在共享恢复函数中将
  CPU/CUDA RNG state 显式规范为 CPU ByteTensor；仅修复断点恢复，不改变
  算法、seed、隐私会计或训练超参数。

## Resume Audit 2026-07-25 19:36 CST

- 上一轮结束后主机于 2026-07-25 17:48 重启；可读取的内核日志中没有 OOM、
  GPU Xid、watchdog 或 kernel panic 证据，不能归因于训练内存爆炸。
- 重启导致旧 tmux socket 消失，但磁盘 checkpoint 和 COMPLETE 状态均保留。
- 当前安全策略固定 CPU core 0、GPU 0、所有线程为 1、`num_workers=0`；
  状态 JSON 改为同目录原子替换，并记录 boot ID。
- 活动队列保存在
  `new_configs/new_secure_feature/new_active_single_gpu_queue.txt`，supervisor
  只恢复该队列中未完成且未 FAILED 的任务。

## Formal Resume 2026-07-25 20:29 CST

- Public MTT formal queue completed without retry: CIFAR-100 seeds 42/43/44
  test accuracy `0.4488/0.4498/0.4399`; Tiny ImageNet seeds 42/43/44
  test accuracy `0.2671/0.2673/0.2579`.
- `new_expTab` and `new_expFig` are no longer empty. All generated public
  artifacts use the required `new_` prefix and the report status is
  `QUALITY_GATE_PASS`.
- The active persisted queue is now
  `new_configs/new_secure_feature/new_gpu0_phase6_dp_validation.tsv`.
- Runtime remains pinned to CPU core 0 with one training process on GPU 0;
  GPU 1 remains unused, DataLoader workers remain zero, and swap usage remains
  zero.

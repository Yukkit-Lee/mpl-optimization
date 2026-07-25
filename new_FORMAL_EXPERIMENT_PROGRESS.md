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
| Module-A comparison | 3 datasets x 3 seeds x 4 sources | 36 | IN_PROGRESS |
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
- CIFAR-100 与 Tiny ImageNet public MTT reference-only plateau pilots 尚未完成；
  已生成单 GPU 串行恢复队列，pilot 不读取 test。

## Resume Audit 2026-07-25 19:36 CST

- 上一轮结束后主机于 2026-07-25 17:48 重启；可读取的内核日志中没有 OOM、
  GPU Xid、watchdog 或 kernel panic 证据，不能归因于训练内存爆炸。
- 重启导致旧 tmux socket 消失，但磁盘 checkpoint 和 COMPLETE 状态均保留。
- 当前安全策略固定 CPU core 0、GPU 0、所有线程为 1、`num_workers=0`；
  状态 JSON 改为同目录原子替换，并记录 boot ID。
- 活动队列保存在
  `new_configs/new_secure_feature/new_active_single_gpu_queue.txt`，supervisor
  只恢复该队列中未完成且未 FAILED 的任务。

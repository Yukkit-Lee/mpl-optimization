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
- [ ] Phase 1: Implement real Stage-A privacy mechanism
- [ ] Phase 2: Build shared 2048-D representation pipeline
- [ ] Phase 3: Implement secure Stage-B computation
- [ ] Phase 4: Implement secure Stage-C Fixed DP-SGD
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
- Legacy limitation: TFE secure-random and aux custom ops for TF 1.15.5 are missing.
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
- Phase 1 implementation is in progress; no Stage-A privacy claim is active yet.
- No smoke result will be promoted into a formal result.

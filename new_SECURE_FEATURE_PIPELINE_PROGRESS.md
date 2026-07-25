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

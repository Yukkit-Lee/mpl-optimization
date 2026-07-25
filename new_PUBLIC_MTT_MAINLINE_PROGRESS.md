# Public MTT Mainline Formal Progress

## Mainline

```text
Official Public MTT Images
-> Shared Frozen ConvNet
-> L2-bounded 2048-D Features
-> TFE/ABY3 Stage B
-> TFE/ABY3 Fixed/Adaptive Stage C
```

`DP-MTT` is retained as
`DEFERRED_UNTIL_MAIN_PIPELINE_COMPLETE` and does not block this mainline.

## Resource Contract

- CPU/BLAS/PyTorch/TensorFlow threads per process: 1.
- `DataLoader num_workers=0`, no multiprocessing pool.
- GPU0/GPU1 each permit at most one process under independent locks.
- TFE tasks are serial under the TFE lock.
- Long tasks use tmux, heartbeat, atomic status, checkpoint and bounded retry.
- Test data is forbidden during tuning and configuration selection.

## Status

- [x] Public MTT ConvNet formal reference: 3 datasets x 3 seeds.
- [x] Frozen 2048-D linear-probe quality gate.
- [x] Module A Full/Random/Public efficiency.
- [ ] Secure Module B convergence.
- [ ] Secure Module B IID/Mild/Severe.
- [ ] Secure Module C multi-privacy comparison.
- [ ] End-to-End comparison.
- [ ] TFE small/medium/largest-feasible report.
- [ ] DP-MTT supplement after the complete mainline.

## Active Step

The first mainline gate performs reference-only optimizer selection for a
plain 2048-D linear probe. Formal test evaluation starts only after the
optimizer configuration is frozen. The required test thresholds are:

- CIFAR-10: at least `0.6583`.
- CIFAR-100: at least `0.4262`.
- Tiny ImageNet: at least `0.2441`.

## Frozen Feature Quality Result

- CIFAR-10: `0.6782 +/- 0.0071`, required `0.6583`.
- CIFAR-100: `0.4384 +/- 0.0046`, required `0.4262`.
- Tiny ImageNet: `0.2619 +/- 0.0061`, required `0.2441`.
- Tuning used physically test-free caches and seeds 42/43 only.
- Formal test evaluation used seeds 42/43/44 after optimizer freeze.
- Status: `QUALITY_GATE_PASS`.

## Module-A Formal Efficiency Result

- CIFAR-10 final accuracy: Full `0.6942 +/- 0.0047`, Random
  `0.6084 +/- 0.0052`, Official MTT `0.6767 +/- 0.0087`.
- CIFAR-100 final accuracy: Full `0.4872 +/- 0.0046`, Random
  `0.3897 +/- 0.0037`, Official MTT `0.4376 +/- 0.0055`.
- Tiny ImageNet final accuracy: Full `0.3171 +/- 0.0050`, Random
  `0.2257 +/- 0.0023`, Official MTT `0.2615 +/- 0.0061`.
- Mean Step@Target for Full versus Official is `426.7 vs 133.3`,
  `480 vs 280`, and `400 vs 300`, respectively.
- The equal-size Random subset did not reach the declared target on
  CIFAR-10 or CIFAR-100; this is reported as `N/R`, not omitted.
- All 27 method runs are complete and use three formal seeds.

The active phase now moves to secure Module B convergence and heterogeneity
experiments using the same encoder/scaler signatures.

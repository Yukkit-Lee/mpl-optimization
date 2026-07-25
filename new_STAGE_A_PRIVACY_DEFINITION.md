# New Stage-A Privacy Definition

## Track A: Public MTT Benchmark

- Input: official MTT ipc=50 images.
- Privacy claim: none.
- `stage_a_track=public_mtt_benchmark`
- `stage_a_epsilon=N/A`

## Track B: DP-MTT

1. Raw private records are accessed only by `new_stage_a_dp_experts.py`.
2. Each expert is trained with Opacus per-example clipping and Gaussian DP-SGD.
3. Gaussian noise is generated from a ChaCha20 CSPRNG byte stream and mapped by
   Box-Muller. Four independent Gaussian draws are summed and divided by two,
   following the floating-point hardening structure used by Opacus secure mode.
4. The RDP accountant records noise multiplier, sample rate, optimizer steps and delta.
5. Multiple experts on one party use sequential RDP composition.
6. Parties use disjoint record indices, so the released union uses parallel composition and reports the maximum party epsilon.
7. `new_stage_a_dp_distill.py` accepts only SHA256-verified DP trajectory files, initializes images from random noise and never imports a raw dataset loader.
8. Distilled images are post-processing of DP trajectories and inherit their epsilon and delta.

The claim depends on never releasing a non-DP trajectory and never allowing the distillation process to access raw data.

## Secure RNG Backend and Recovery Boundary

The available `torchcsprng 0.2.1` wheel links against CUDA 10.2 and cannot load
with PyTorch 2.8/CUDA 12.8. `new_cryptographic_gaussian.py` therefore replaces
only Opacus' module-level noise sampler with a tested ChaCha20 implementation;
Opacus still performs per-sample clipping and RDP accounting.

The CSPRNG key is stored outside the project tree in a mode-0600 sidecar. Normal
checkpoints contain only its SHA256 fingerprint and request counter. This allows
exact resume without committing or logging a secret key. A formal run with
`--secure_rng` never falls back to `torch.normal`.

The backend is an audited project implementation, not the official
`torchcsprng` extension. Paper claims and manifests must use the exact name
`chacha20_box_muller_4sample` rather than claiming that Opacus secure mode was
enabled.

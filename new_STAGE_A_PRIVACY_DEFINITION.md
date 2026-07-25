# New Stage-A Privacy Definition

## Track A: Public MTT Benchmark

- Input: official MTT ipc=50 images.
- Privacy claim: none.
- `stage_a_track=public_mtt_benchmark`
- `stage_a_epsilon=N/A`

## Track B: DP-MTT

1. Raw private records are accessed only by `new_stage_a_dp_experts.py`.
2. Each expert is trained with Opacus per-example clipping and Gaussian DP-SGD.
3. The RDP accountant records noise multiplier, sample rate, optimizer steps and delta.
4. Multiple experts on one party use sequential RDP composition.
5. Parties use disjoint record indices, so the released union uses parallel composition and reports the maximum party epsilon.
6. `new_stage_a_dp_distill.py` accepts only SHA256-verified DP trajectory files, initializes images from random noise and never imports a raw dataset loader.
7. Distilled images are post-processing of DP trajectories and inherit their epsilon and delta.

The claim depends on never releasing a non-DP trajectory and never allowing the distillation process to access raw data.

## Current Limitation

The correctness run used `PrivacyEngine(secure_mode=False)`. The only available
`torchcsprng 0.2.1` Python 3.9 wheel links against CUDA 10.2 and cannot load in
the current PyTorch 2.8/CUDA 12.8 environment. Formal runs request
`--secure_rng` and fail closed while this incompatibility remains. Therefore the
correctness run demonstrates DP-SGD clipping/noise/accounting semantics but must
not be described as production-grade cryptographic randomness.

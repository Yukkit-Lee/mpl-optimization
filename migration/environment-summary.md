# Environment provenance snapshot

Captured on 2026-07-26 (Asia/Shanghai).

## Host

- Ubuntu 24.04.1 LTS, Linux `7.0.0-28-generic`, x86_64.
- 2 x NVIDIA GeForce RTX 4090, 24,564 MiB each.
- NVIDIA driver 580.173.02; `nvidia-smi` reports CUDA compatibility 13.0.
- New-pipeline default: GPU 0 only, one process, one CPU thread and
  `DataLoader num_workers=0`.

## Modern distillation environment

- Source-host prefix `/home/lixujie/.conda/envs/distillation` is provenance only.
- Python 3.9.7.
- PyTorch 2.8.0+cu128; torchvision 0.23.0+cu128.
- NumPy 1.22.4, SciPy 1.7.3, scikit-learn 1.6.1,
  Matplotlib 3.5.3, Opacus 1.5.4, cryptography 36.0.0.
- Portable specs: `environment.yml`, `requirements.txt`.

## Legacy TF-Encrypted environment

- Source-host prefix `/home/lixujie/.conda/envs/tfe` is provenance only.
- Python 3.6.13, TensorFlow 1.15.5, TF-Encrypted 0.8.0.
- NumPy 1.19.2, SciPy 1.5.2.
- Portable specs: `environment-tfe-legacy.yml` and
  `requirements-tfe-legacy.txt`.
- TFE source commit and custom-op hashes: `migration/tfe/`.

Full current `pip freeze`, Conda package lists, sanitized exports and explicit
specs are in `migration/env_snapshots/new_*`. They contain no Conda `prefix:`
and strip local `file://` dependency paths.

Resolver dry-runs are recorded in
`migration/environment-install-validation.md`. Target hardware must still run
the CUDA and TFE custom-op import checks after installation.


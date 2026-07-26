# AutoDL recovery: `new-secure-feature-pipeline`

This guide restores the current Public-MTT mainline without uploading official
MTT tensors, raw CIFAR/Tiny data, secrets, or large checkpoints to GitHub.
Modern GPU training and legacy Python 3.6/TF-Encrypted remain separate.

## 1. Clone the exact branch

```bash
cd /root/autodl-tmp
git clone -b new-secure-feature-pipeline \
  https://github.com/Yukkit-Lee/mpl-optimization.git tfe_experiments
cd /root/autodl-tmp/tfe_experiments
cp config/new_paths.autodl.example.env ~/.config/new-secure-feature-pipeline.env
set -a; source ~/.config/new-secure-feature-pipeline.env; set +a
```

The clone defaults to one GPU (`NEW_SINGLE_GPU=0`), one CPU thread, and
`DataLoader num_workers=0`.

## 2. Restore the real MTT source tree

Do not assume the locally used MTT tree can be recreated from GitHub alone.
The repository contains its Git history, tracked patch, and untracked source:

```bash
source ~/.config/new-secure-feature-pipeline.env
bash scripts/restore_mtt_snapshot.sh
git -C "$NEW_MTT_ROOT" status --short --branch
(cd migration/mtt && sha256sum -c SHA256SUMS)
```

The MTT base is commit `50dccea47ba48b9cfb3b9c59375d7da92e4e603b`.
Its tracked working-tree differences are file-mode changes; all local untracked
source is preserved in `migration/mtt/mtt-untracked-source.tar.gz`.

## 3. Create two independent environments

```bash
mkdir -p "$NEW_ENV_ROOT"
conda env create -p "$NEW_ENV_ROOT/distillation" -f environment.yml
conda env create -p "$NEW_ENV_ROOT/tfe" -f environment-tfe-legacy.yml
```

If the environment YAML solver is unavailable, use the requirements files only
inside separate Python 3.9 and Python 3.6 environments. Never install modern
PyTorch and TF-Encrypted into one environment.

```bash
"$NEW_MODERN_PYTHON" -m pip install -r requirements.txt
"$NEW_TFE_PYTHON" -m pip install -r requirements-tfe-legacy.txt
```

Host provenance and full frozen package lists are in
`migration/env_snapshots/new_*`. The current modern core is Python 3.9.7,
PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128 and Opacus 1.5.4. The legacy core
is Python 3.6.13, TensorFlow 1.15.5 and TF-Encrypted 0.8.0.

## 4. Build TF-Encrypted custom ops

Install system build prerequisites first:

```bash
apt-get update
apt-get install -y git make g++ curl tar autoconf automake libtool
source ~/.config/new-secure-feature-pipeline.env
bash scripts/new_build_tfe_custom_ops.sh
```

The script checks out exact TFE commit
`51de98f9c2208d5d4bd716bf7316d3cd4e0bf434`, builds libsodium with one job,
compiles `aux` and `secure_random` for TensorFlow 1.15.5, installs both `.so`
files, prints their SHA256, and verifies import. Original-host checksums are in
`migration/tfe/SHA256SUMS`; the ordinary migration archive also contains the
original `.so` files as an emergency backup, not as the preferred install.

## 5. Download public data instead of migrating it

```bash
source ~/.config/new-secure-feature-pipeline.env
bash scripts/new_download_public_assets.sh
```

Download the official **non-ZCA** distilled tensors from the authors' page:

```text
https://georgecazenavette.github.io/mtt-distillation/tensors/index.html
```

Place the tensors exactly as follows:

```text
$NEW_OFFICIAL_MTT_ROOT/
  cifar10_50/images_best.pt, labels_best.pt
  cifar100_50/images_best.pt, labels_best.pt
  tiny_50/images_best_0.pt ... images_best_4.pt
  tiny_50/labels_best_0.pt ... labels_best_4.pt
```

Then verify all official tensors and raw archives:

```bash
"$NEW_MODERN_PYTHON" scripts/new_verify_official_mtt_assets.py
```

Exact local SHA256 values and raw download URLs are frozen in
`migration/new_official_data_manifest.json`.

## 6. Restore high-cost caches and selected Stage-B models

Upload `autodl_new_secure_resume_TIMESTAMP.tar.zst` to `/root/autodl-tmp/` and:

```bash
source ~/.config/new-secure-feature-pipeline.env
tar --zstd -xf /root/autodl-tmp/autodl_new_secure_resume_TIMESTAMP.tar.zst \
  -C "$NEW_PROJECT_ROOT"
"$NEW_MODERN_PYTHON" scripts/new_restore_portable_artifacts.py
```

The archive restores the expensive encoder/scaler/feature caches. The 27 final
Stage-B selected initializers are already tracked in Git under
`portable_artifacts/stage_b_selected/`; the restore helper recreates the runtime
`new_init_model/new_public_mtt_mainline/stage_b/` layout. Candidate models and
completed Stage-B training checkpoints are intentionally excluded.

## 7. Validate before continuing

```bash
source ~/.config/new-secure-feature-pipeline.env
"$NEW_MODERN_PYTHON" scripts/check_autodl_env.py
bash -n new_scripts/*.sh scripts/new_*.sh
"$NEW_MODERN_PYTHON" -m py_compile new_*.py scripts/new_*.py
```

Regenerate queues on AutoDL. TSV files are runtime products and are no longer
committed because they contain resolved absolute paths:

```bash
"$NEW_MODERN_PYTHON" new_prepare_public_mtt_mainline.py
"$NEW_MODERN_PYTHON" new_prepare_module_a_efficiency.py
"$NEW_MODERN_PYTHON" new_prepare_stage_b_reference_tuning.py
"$NEW_TFE_PYTHON" new_prepare_stage_b_formal.py
```

Add `--activate` only when intentionally starting a persistent queue.

## 8. Continue Stage C serially

Stage B is complete for CIFAR-10, CIFAR-100 and Tiny with seeds 42/43/44 and
IID/Mild/Severe splits. Stage C had not produced a formal checkpoint before the
host interruption. A safe seed-42 adaptive example is:

```bash
source ~/.config/new-secure-feature-pipeline.env
dataset=cifar100; seed=42; split=severe
cache="$NEW_PROJECT_ROOT/new_cache/new_public_mtt_mainline/new_stage_b_npz_round2/$dataset/new_stage_b_features_${dataset}_seed${seed}.npz"
init="$NEW_PROJECT_ROOT/new_init_model/new_public_mtt_mainline/stage_b/$dataset/seed${seed}/$split/new_secure_stage_b_selected_${dataset}_${split}_seed${seed}.npz"
out="$NEW_OUTPUT_ROOT/new_log/new_public_mtt_mainline/stage_c/$dataset/seed${seed}/$split"
ckpt="$NEW_OUTPUT_ROOT/new_checkpoint/new_public_mtt_mainline/stage_c/$dataset/seed${seed}/$split/new_stage_c"
taskset -c "$NEW_CPU_CORE" "$NEW_TFE_PYTHON" new_stage_c_secure_dpsgd.py \
  --feature_cache "$cache" --initializer "$init" --dataset "$dataset" \
  --mode adaptive_dp --output_dir "$out" --checkpoint_prefix "$ckpt" \
  --seed "$seed" --batch_size 16 --steps 100 --learning_rate 0.02 \
  --noise_multiplier 1.2 --count_noise_multiplier 8.0 --clip_c 1.0 \
  --gamma 0.5 --eta_c 0.05 --c_min 0.05 --c_max 4.0 \
  --warmup_steps 10 --update_interval 5 --checkpoint_interval 10
```

Repeat with `dataset=cifar10` or `dataset=tiny`; keep datasets serial. This is a
continuation template, not a claim that Stage-C formal hyperparameters were
already frozen.

## 9. DP-MTT exclusion

DP-MTT is no longer part of the migration target. Its secrets, trajectories and
tuning checkpoints remain untouched on the source host but are not uploaded or
packaged. Public MTT Stage B/C does not depend on them.

## 10. Optional persistent single-GPU supervisor

```bash
mkdir -p ~/.config/systemd/user
cp new_scripts/new_secure_feature_single_gpu_resume.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now new_secure_feature_single_gpu_resume.service
```

The source host's supervisor was deliberately disabled before migration so no
formal task restarts unexpectedly.

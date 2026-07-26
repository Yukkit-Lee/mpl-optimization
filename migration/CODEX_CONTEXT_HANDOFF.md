# Codex Project Context Handoff

This file is the safe, repository-tracked context entry point for continuing the
project with a new Codex installation on AutoDL. It intentionally contains no
credentials, CSPRNG keys, raw datasets, or private conversation content.

## 1. Repository State

- Repository: `https://github.com/Yukkit-Lee/mpl-optimization.git`
- Branch: `new-secure-feature-pipeline`
- Migration baseline commit: `57f2544ab778dc42d861adefbef37b059a59c95f`
- Main project root on the source server: `/mnt/code/lixujie/tfe_experiments`
- AutoDL setup guide: `AUTODL_MIGRATION.md`
- Full chronological project record: `progressLog/project_progress.md`

Always inspect the current branch head after cloning:

```bash
git status --short --branch
git log -1 --oneline
```

## 2. Current Research Mainline

The active paper path is the **Official Public MTT** secure-feature pipeline.
The earlier DP-MTT path has been abandoned and is not required for migration.

The current stages are:

```text
Stage A:
Official public MTT distilled images
  -> one shared MTT ConvNet encoder
  -> frozen 2048-D representations

Stage B:
TFE/ABY3 local pretraining
  -> NoKD/KD candidate construction
  -> KD-Avg / KD-ReferenceWeighted
  -> reference-guided secure selection
  -> selected initializer

Stage C:
TFE/ABY3 secure DPSGD
  -> fixed or adaptive clipping
  -> checkpointed three-party CSPRNG noise shares
  -> Stage-C-only privacy accounting
```

Privacy accounting applies only to Stage C in the Public MTT mainline. Stage A
and Stage B are not included in the reported Stage-C epsilon.

## 3. Completed Work

- Public MTT Stage A formal encoder runs are complete for CIFAR-10, CIFAR-100,
  and Tiny ImageNet with seeds 42, 43, and 44.
- Stage B formal TFE/ABY3 runs are complete: 3 datasets x 3 seeds x
  IID/Mild/Severe = 27 runs.
- Frozen Stage B selected initializers are tracked in
  `portable_artifacts/stage_b_selected/`.
- Stage B reports, CSV/JSON/TXT, and paper figures are tracked in the repository.
- TFE 1.15.5 `aux` and `secure_random` custom ops were successfully built and
  tested on the source host.
- Stage C formal training had not started when migration preparation began.
- All persistent training services and workers were deliberately stopped and
  disabled before migration.

Do not rerun completed Stage A or Stage B jobs unless an integrity check fails.

## 4. Runtime Safety Contract

All future runs must preserve these limits unless explicitly revised after a
resource audit:

- one GPU only, default `CUDA_VISIBLE_DEVICES=0`;
- one CPU core and one CPU thread;
- `DataLoader num_workers=0`;
- no multiprocessing or implicit BLAS/OpenMP pools;
- datasets run serially;
- long jobs use tmux or the persistent single-GPU supervisor;
- checkpoints, heartbeat, atomic status writes, and resume-before-restart.

Before running any experiment, print the resolved configuration, run count,
output paths, and resource estimate.

## 5. Environment and Paths

Modern and legacy environments must remain separate:

- modern distillation environment: Python 3.9, PyTorch 2.8.0+cu128;
- legacy TFE environment: Python 3.6, TensorFlow 1.15.5, TF-Encrypted 0.8.0.

Use `config/new_paths.autodl.example.env` and the `NEW_*` environment variables.
Do not reintroduce source-host absolute paths into executable code or queue
files. Runtime TSV queues must be regenerated on the target machine.

## 6. Assets to Restore

The following large Public-MTT cache archive must be transferred manually:

```text
/home/lixujie/tmpWork/autodl_new_secure_resume_20260726_150457.tar.zst
size:   8,771,438,515 bytes
SHA256: 2313a96c1352c2cb66b6262bb59f61d559fb2cd6112f87131e7ab76a53b1bae9
```

It contains high-cost encoder/scaler/feature caches and a backup of the TFE
custom-op shared libraries. It does not contain official MTT tensors, raw
datasets, DP-MTT assets, or secrets.

Official non-ZCA MTT tensors and CIFAR/Tiny raw data must be downloaded again as
described in `AUTODL_MIGRATION.md`, then verified against
`migration/new_official_data_manifest.json`.

## 7. Private Conversation Export

The source host contains a private raw Codex JSONL export:

```text
/home/lixujie/tmpWork/codex_conversation_export_20260726_1522.tar.zst
size:   12,154,981 bytes
SHA256: 3b6420f7aa7c3a0c17c0b17b8e797ebe3224d6e73c49b519eaf5154ce679601a
```

The archive can contain historical credentials and sensitive tool output. It
must be transferred privately and must never be committed to GitHub.

The JSONL is useful for audit and search, but Codex clients do not guarantee
that a raw session can be imported as a resumable UI conversation. This handoff
file plus `progressLog/project_progress.md` is the supported working-context
restoration path.

## 8. First Prompt on AutoDL

After cloning and restoring assets, start a new Codex session from the project
root with this instruction:

```text
请先完整读取 migration/CODEX_CONTEXT_HANDOFF.md、
progressLog/project_progress.md、AUTODL_MIGRATION.md，以及当前任务对应的
plan/new_*.md。先审计 git、环境、资产校验和运行状态，不要重跑已经完成的
Stage A/Stage B；继续任务前严格使用单 GPU、CPU 线程 1、num_workers=0，
并维护 project_progress.md 后提交推送。
```

## 9. Next Work

The next research task is to resume Public-MTT Stage C from the selected Stage B
initializer, first with validation/reference-only diagnostics and then with
frozen formal settings. Do not claim adaptive-clipping improvement until the
fixed/adaptive comparison is fair and clipping-bound stability is diagnosed.
Test data must remain outside tuning and candidate-selection paths.

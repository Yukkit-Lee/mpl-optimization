# Large artifact migration manifest

The generated bundle paths, compressed sizes and SHA256 values are written here
after packaging. The ordinary archive intentionally excludes official MTT,
raw datasets, private secrets, candidate models, and completed Stage-B training
checkpoints.

## Ordinary resume archive contents

| Relative path | Source size | Purpose | Required to continue Stage C/end-to-end | Recomputable | GitHub |
|---|---:|---|---|---|---|
| `new_cache/new_secure_feature/new_formal_phase6/` | 3,244,529,333 bytes | Frozen encoder/scaler/formal features | Yes | Yes, expensive | No |
| `new_cache/new_public_mtt_mainline/` | 6,292,113,097 bytes | Public mainline full/test-free/Stage-B caches | Yes | Yes, expensive | No |
| `legacy_tfe_custom_ops/` | 207,208 bytes | Original-host `.so` backup | No; rebuild preferred | Yes | No |

## Explicitly excluded

- `new_checkpoint/new_public_mtt_mainline/stage_b/` (7,346,882,061 bytes):
  all 27 Stage-B runs are complete; selected initializers and formal reports are
  sufficient. Original checkpoints remain untouched on the source host.
- Official MTT and raw CIFAR/Tiny data: downloadable and verified separately.
- DP-MTT secrets/trajectories: excluded because DP-MTT is no longer part of the migration target.

## Generated ordinary archive

- Path: `/home/lixujie/tmpWork/autodl_new_secure_resume_20260726_150457.tar.zst`
- Compressed size: `8,771,438,515` bytes (about 8.17 GiB)
- SHA256: `2313a96c1352c2cb66b6262bb59f61d559fb2cd6112f87131e7ab76a53b1bae9`
- AutoDL extraction root: `$NEW_PROJECT_ROOT`

## DP-MTT assets

DP-MTT experiments are discontinued. Existing trajectories, tuning caches,
checkpoints and CSPRNG sidecars remain untouched on the source host, but are not
in Git or any migration archive. The Public MTT mainline does not depend on them.

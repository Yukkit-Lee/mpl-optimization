# AutoDL artifact inventory for `new-secure-feature-pipeline`

| Asset | Current size | Purpose | Re-download/recompute | AutoDL destination | Migration policy |
|---|---:|---|---|---|---|
| Official non-ZCA MTT tensors | 641 MiB | Public Stage A input | Download from the MTT authors' tensor page | `$NEW_OFFICIAL_MTT_ROOT` | Not Git, not bundled |
| CIFAR-10 archive/work tree | 341 MiB | Raw benchmark test/train data | Official Toronto URL | `$NEW_DATA_ROOT/mtt_cifar10_work` | Not Git, not bundled |
| CIFAR-100 archive/work tree | 1.6 GiB | Raw benchmark test/train data | Official Toronto URL | `$NEW_DATA_ROOT/mtt_cifar100_work` | Not Git, not bundled |
| Tiny ImageNet archive/work tree | 7.5 GiB | Raw benchmark test/train data | Stanford archive URL | `$NEW_DATA_ROOT/mtt_tiny_work` | Not Git, not bundled |
| `new_cache/new_secure_feature/new_formal_phase6` | 3.1 GiB | Frozen public-MTT encoders, scalers and 2048-D feature caches | Expensive but reproducible | Same relative project path | Ordinary large bundle |
| `new_cache/new_public_mtt_mainline` | 5.9 GiB | Full/test-free features and Stage-B NPZ caches | Expensive but reproducible | Same relative project path | Ordinary large bundle |
| Stage-B selected initializers | 11 MiB / 27 files | Final selected model for each dataset/seed/split | Recomputable but costly TFE | `portable_artifacts/stage_b_selected` then runtime restore | GitHub |
| Stage-B candidate models | Remaining part of local 43 MiB init tree | Paper ablation candidates only | Recomputable | None | Excluded |
| Stage-B completed checkpoints | 6.9 GiB | Resume already-complete Stage B training | Not needed after selected model/report exists | None | Excluded, originals retained |
| Legacy TFE custom-op `.so` | 203 KiB | Emergency backup of TF 1.15.5 aux/secure_random | Preferred action is rebuild | `legacy_tfe_custom_ops/` in archive | Ordinary large bundle only |
| DP-MTT trajectories/tuning caches | ~1.43 GiB | Discontinued private Stage-A research | Not required by Public MTT mainline | None | Excluded; originals retained |
| DP CSPRNG secret sidecars | 16 KiB | Discontinued private DP-MTT state | Not required by Public MTT mainline | None | Excluded; originals retained |
| Legacy `featureData` / `distilledData` | 17 MiB / 261 MiB | Older pre-refactor experiments | Recomputable/not required by new mainline | Optional | Excluded |
| MTT expert `buffers/` | 6.1 GiB | Older manual distillation work | Recomputable; Public MTT mainline does not depend on it | Optional | Excluded |

Exact official tensor and archive checksums are in
`migration/new_official_data_manifest.json`. Exact bundle contents and hashes
are recorded in `LARGE_ARTIFACT_MANIFEST.md` and `SHA256SUMS` after packaging.

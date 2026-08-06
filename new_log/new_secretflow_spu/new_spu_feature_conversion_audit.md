# SecretFlow-SPU feature conversion audit

Status: `PASS`

No feature conversion was needed. Full uses the existing read-only float32
2048-D NumPy memmap plus frozen row indices; Ours uses the existing float32
Official-MTT NPZ arrays directly. Stage A and feature extraction runs are zero.

| Dataset | Full Stage-B rows | Ours rows | Stage-C rows | Dev rows | Feature dim |
|---|---:|---:|---:|---:|---:|
| cifar10 | 34900 | 480 | 5000 | 5000 | 2048 |
| cifar100 | 34000 | 4800 | 5000 | 5000 | 2048 |
| tiny | 68000 | 9600 | 10000 | 10000 | 2048 |

All audited training/reference/selection/development members are pairwise
disjoint. The locked-holdout member and official test were not read.

CIFAR-10/100 Anchor CE parameters come from the completed positive freeze.
Tiny-ImageNet was excluded from that ablation, so its identical method
semantics use the CIFAR-100 parameters only as a registered smoke/tuning
starting point; this is not mislabeled as prior Tiny validation.

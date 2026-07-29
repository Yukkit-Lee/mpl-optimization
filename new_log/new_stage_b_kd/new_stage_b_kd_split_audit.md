# Stage-B MTT-anchor split audit

- Status: COMPLETE
- Current task seed: 42 only
- Official-MTT local and anchor IDs are mutually exclusive and exhaustive.
- Development validation comes from the frozen original-training split and is never loaded into local training or anchor KD.
- Locked holdout and official test were not read.

| Dataset | Anchor/class | Local | Anchor | Intersection | Union | Dev |
|---|---:|---:|---:|---:|---:|---:|
| cifar10 | 4 | 440 | 40 | 0 | 480 | 5000 |
| cifar10 | 8 | 400 | 80 | 0 | 480 | 5000 |
| cifar10 | 12 | 360 | 120 | 0 | 480 | 5000 |
| cifar100 | 4 | 4400 | 400 | 0 | 4800 | 5000 |
| cifar100 | 8 | 4000 | 800 | 0 | 4800 | 5000 |
| cifar100 | 12 | 3600 | 1200 | 0 | 4800 | 5000 |

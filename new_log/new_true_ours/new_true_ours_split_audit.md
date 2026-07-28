# True Ours disjoint split audit

- Status: FROZEN_PRE_TUNING
- All original-training-set splits are stratified, pairwise disjoint, and exhaustive.
- Locked holdout was not read for metrics.
- Official test was not read by this rebuild.
- Official MTT is a fixed public upstream Stage-A asset; disjointness claims apply from Stage B onward.

| Dataset | Seed | Stage B | Stage C | Ref | Sel | Dev | Locked | Disjoint |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cifar10 | 42 | 34900 | 5000 | 50 | 50 | 5000 | 5000 | YES |
| cifar10 | 43 | 34900 | 5000 | 50 | 50 | 5000 | 5000 | YES |
| cifar10 | 44 | 34900 | 5000 | 50 | 50 | 5000 | 5000 | YES |
| cifar100 | 42 | 34000 | 5000 | 500 | 500 | 5000 | 5000 | YES |
| cifar100 | 43 | 34000 | 5000 | 500 | 500 | 5000 | 5000 | YES |
| cifar100 | 44 | 34000 | 5000 | 500 | 500 | 5000 | 5000 | YES |
| tiny | 42 | 68000 | 10000 | 1000 | 1000 | 10000 | 10000 | YES |
| tiny | 43 | 68000 | 10000 | 1000 | 1000 | 10000 | 10000 | YES |
| tiny | 44 | 68000 | 10000 | 1000 | 1000 | 10000 | 10000 | YES |

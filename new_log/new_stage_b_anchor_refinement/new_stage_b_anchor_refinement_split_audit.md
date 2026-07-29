# Stage-B Anchor Refinement Split Audit

Official-MTT-only 40 local + 8 anchor samples/class; seed42.

| Dataset | Local/class | Anchor/class | Intersection | Union | Checkpoint SHA256 |
|---|---:|---:|---:|---:|---|
| cifar10 | 40 | 8 | 0 | 480 | `6403042e7593eb18f090f7d930a977d5746c5ac33c597e39ac374257e1ece42c` |
| cifar100 | 40 | 8 | 0 | 4800 | `85bf4b51ab4c5028bf8ef81aeb19155b07bb777f97dbbf4cf0329ac3e9e7202c` |

Development examples come from the disjoint frozen training split; locked holdout and official test were not read.

Source commit: `2565bd4bd3bcde2a923878e37172a47f3dd17e1a`.

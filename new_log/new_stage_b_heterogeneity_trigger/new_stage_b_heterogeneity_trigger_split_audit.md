# Heterogeneity Trigger Split Audit

| Dataset | Distribution | Local∩Anchor | Union | Checkpoint SHA256 |
|---|---|---:|---:|---|
| cifar10 | iid | 0 | 480 | `3b74b7e8e07365c40275ea232554a0cf64112ece0eca5d41f9300e011ef50f55` |
| cifar10 | mild | 0 | 480 | `9aa6b9830f702a9292260fefb48c37ff9a92c4d991d8c7ff7d834cf1cacebe9f` |
| cifar10 | severe | 0 | 480 | `6403042e7593eb18f090f7d930a977d5746c5ac33c597e39ac374257e1ece42c` |
| cifar100 | iid | 0 | 4800 | `ea9a794d858cbf3f130c4df2a0cadf45e16aada9532f228d615179401d5f5ee1` |
| cifar100 | mild | 0 | 4800 | `e6f9edf8aa318836eeecc34b9cea2a26d98882047581ca0813e36ce165c75010` |
| cifar100 | severe | 0 | 4800 | `85bf4b51ab4c5028bf8ef81aeb19155b07bb777f97dbbf4cf0329ac3e9e7202c` |

All cases use Official-MTT 40 local + 8 anchor samples/class. Development is disjoint; locked holdout/test were not read.

Source commit: `db2e0722896e6c90cf4e6ad2a94c3ff37770b4d8`.

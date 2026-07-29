# Stage-B MTT-Anchor Post-Aggregation Split Audit

- Status: **PASS**
- Report source commit: `58352813ad6261486d225c1e3eebdb73b973849f`
- Experiment source commit: `d87b223527f13254b92f30529105177c88641adb`
- Scenarios audited: 6
- Official-MTT-only: `True`
- Full-data initializer dependency: `False`
- Official test read: `False`
- Locked holdout read: `False`

| Dataset | Distribution | Local/class | Anchor/class | Local∩Anchor | Dev∩Train | Dev∩Anchor |
|---|---:|---:|---:|---:|---:|---:|
| cifar10 | iid | 40 | 8 | 0 | 0 | 0 |
| cifar10 | mild | 40 | 8 | 0 | 0 | 0 |
| cifar10 | severe | 40 | 8 | 0 | 0 | 0 |
| cifar100 | iid | 40 | 8 | 0 | 0 | 0 |
| cifar100 | mild | 40 | 8 | 0 | 0 | 0 |
| cifar100 | severe | 40 | 8 | 0 | 0 | 0 |

All six scenarios use the frozen 40-local + 8-anchor per class Official-MTT split. Development IDs are disjoint from training and anchor pools; the locked holdout key was deliberately not accessed.

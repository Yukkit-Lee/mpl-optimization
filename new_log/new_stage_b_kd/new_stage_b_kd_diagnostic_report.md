# Stage-B MTT-Anchor KD/Consensus Diagnostic Report

Scope: seed42 only; development-only; Official-MTT-only; Stage B only.

## Search and diagnosis

- S1--S6 completed with 89 registered plaintext candidates.
- Client heads showed non-zero parameter/JS diversity and maximal hard prediction disagreement.
- The mechanical teacher/student gates passed, but student gain was much larger than teacher gain.
- A pre-method causal decomposition therefore compared anchor CE-only, KD-only, and CE+KD under the frozen selected budget.

## Selected plaintext results

| Dataset | Average | Teacher | CE+KD student | Teacher gain | Student gain |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 19.24% | 19.40% | 55.14% | +0.16pp | +35.90pp |
| CIFAR-100 | 8.08% | 9.32% | 23.44% | +1.24pp | +15.36pp |

## Causal finding

| Dataset | CE-only gain | KD increment over CE | Class-aware vs uniform student | Interpretation |
|---|---:|---:|---:|---|
| CIFAR-10 | +35.46pp | +0.44pp | +0.00pp | anchor supervision dominated |
| CIFAR-100 | +18.92pp | -3.56pp | -0.02pp | anchor supervision dominated |

The registered evidence supports labeled anchor supervision, not an independent
class-aware Consensus/KD contribution. KD-only never exceeded Average.

## Security and lineage

- TFE/ABY3, fixed-point precision 16, custom ops required, no plaintext fallback.
- CIFAR-10/100 secure rows passed finite, direction/utility, lineage, and unread-test gates.
- Locked holdout and official test were not read; no Full-Data initializer was loaded.

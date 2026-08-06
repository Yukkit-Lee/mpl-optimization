# SecretFlow-SPU quickstart smoke

Status: `COMPLETE`

This gate follows the installed SPU official distributed quickstart: three
local parties, ABY3, FM128, and JAX-authored secure functions. It is a
backend smoke only; formal paper code uses the SecretFlow SPU device API.

## Checks

| Check | Pass | Maximum absolute error |
|---|---|---:|
| addition | True | 0 |
| comparison | True | exact |
| matrix multiplication | True | 0 |
| linear logits | True | 2.98e-08 |
| linear loss | True | 0 |
| linear weight update | True | 1.49e-08 |
| linear bias update | True | 1.49e-08 |

Only final values needed for correctness were revealed. No dataset, test
split, locked holdout, Stage A artifact, or TFE artifact was read.

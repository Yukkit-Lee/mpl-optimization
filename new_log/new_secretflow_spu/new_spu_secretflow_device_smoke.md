# SecretFlow-SPU device smoke

Status: `COMPLETE`

This gate used the real `secretflow.SPU` device API with the official
source-SFL SimulationStrategy, three Ray-hosted parties, ABY3/FM128,
and 16 fractional bits. Debug/PYU emulation and the bare SPU helper
were both disabled.

| Check | Pass | Maximum absolute error |
|---|---|---:|
| logits | True | 2.75e-05 |
| loss | True | 3.04e-06 |
| weight | True | 3.66e-05 |
| bias | True | 3.85e-05 |

Private X and y originated on distinct PYUs. Only the final smoke outputs were revealed.

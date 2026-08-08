# SecretFlow-SPU end-to-end correctness audit (seed42)

Status: `5 COMPLETE + 1 NEGATIVE_STOP`

Every completed run used the real `secretflow.SPU` device API with
three-party ABY3/FM128/fxp16 simulation. Audit items:

| Dataset | Method | Status | Finite w/b/c | Adaptive DP | Poisson CSPRNG | 3-party noise | No fallback | test read | lockout read |
|---|---|---|---|---|---|---|---|---|---|
| cifar10 | true_ours | COMPLETE | yes | adaptive_dp | os_csprng | yes | yes | false | false |
| cifar10 | full_data | COMPLETE | yes | adaptive_dp | os_csprng | yes | yes | false | false |
| cifar100 | true_ours | COMPLETE | yes | adaptive_dp | os_csprng | yes | yes | false | false |
| cifar100 | full_data | COMPLETE | yes | adaptive_dp | os_csprng | yes | yes | false | false |
| tiny | true_ours | COMPLETE | yes | adaptive_dp | os_csprng | yes | yes | false | false |
| tiny | full_data | NEGATIVE_STOP | - | - | - | - | - | false | false |

The Tiny-ImageNet Full cell is a documented negative stop: the SPU
runtime worker dies in the first Stage-B epoch under every batch
configuration (see `new_spu_tiny_full_negative_stop.md`). No
plaintext fallback was used. TFE/ABY3 retains the valid Tiny Full cell.


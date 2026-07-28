# True Ours formal freeze manifest

Status: `FROZEN_BEFORE_FORMAL`

Source Git commit: `fda36bdcc959981f5ff3637c9301730d37a99608`

Frozen methods:

- Full Data + Secure Average + Adaptive DP
- Official MTT True Ours + Secure Average + Adaptive DP

Frozen seeds: `42 / 43 / 44`

All Stage B runs use TFE/ABY3, `secure_average_only`, balanced five-client
partitions, precision fractional 16, checkpoint every 5 epochs, and aggregate
development-logit evaluation only at checkpoint epochs. All Stage C runs use
TFE/ABY3, CSPRNG Poisson sampling, graph capacity 64, expected batch 16,
global Adaptive clipping, composed accountant target epsilon 3 and delta
1e-5, checkpoint every 5 steps.

The timing boundary starts immediately before Stage-B secure graph/session
construction and ends after Stage-C final noisy-model materialization. It is
identical for both methods.

The Stage-B initializer is data-dependent and non-DP. Stage-C epsilon is a
conditional mechanism claim only; no end-to-end record-DP claim is allowed.

Targets are frozen at 80/90/95 percent of the tuned Full-Data development
plateau. A target that is not reached must be reported as `Not Reached`.

Locked holdout and official test remain unread by this rebuild. They may be
read only once after every formal Stage-B and Stage-C report is COMPLETE and
the queue audit passes.

Exact protocol:

- `new_configs/new_true_ours/new_true_ours_config_freeze.json`
- `new_configs/new_true_ours/new_true_ours_target_freeze.json`

Development evidence:

- `new_log/new_true_ours/new_adaptive_stage_c_development_screen.json` (`31d4404c4d320163b62d67be99e71e6b24b056fd16bb6e34e6cbf5b9db7cdb37`)
- `new_log/new_true_ours/new_adaptive_stage_c_secure_finalists.json` (`58d0e7e207ac5546794d2a0abf8990279e0588bc5cdd39b42e5002954f944f4c`)
- `new_log/new_true_ours/new_true_ours_split_manifest.json` (`873bfa126363b0edc5ce40c54379192005396e4964f3376433a53d3a6e1148d1`)

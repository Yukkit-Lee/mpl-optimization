# True Ours minimal-correction execution note

This note does not rewrite the original formal freeze at commit
`6973d2528af611c98b60176538e9f6139f7477ab`.

The 2026-07-29 correction makes only these execution-policy changes:

1. Tiny-ImageNet uses seed42 only, as explicitly directed by the user.
2. The invalid Tiny Full Data seed42 run is replaced in a new
   `full_data_formal_corrected_v2` namespace.
3. Timeout/retry remains an orchestration-only recovery wrapper. A retry may
   resume only a checkpoint written by the same corrected configuration.
4. CIFAR-10, CIFAR-100, and Tiny True Ours valid formal runs are reused without
   retraining.
5. Evaluation remains development-only. Locked holdout and official test are
   not read in this correction.

The correction does not change the frozen Tiny Full Data model mathematics,
data split, Stage-B training configuration, Adaptive Stage-C configuration,
privacy accountant, target accuracies, or online-secure timing boundary.

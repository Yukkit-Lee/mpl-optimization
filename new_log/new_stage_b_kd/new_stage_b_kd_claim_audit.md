# Stage-B KD Claim Audit

| Claim | Status | Evidence boundary |
|---|---|---|
| Clients contain measurable diversity | SUPPORTED (seed42) | non-zero cosine/JS diversity |
| Labeled MTT anchors improve the initializer | SUPPORTED (seed42) | CE-only gains on both datasets |
| Class-aware teacher beats uniform teacher | UNSUPPORTED | 0.00/-0.02pp student delta |
| KD is the main source of the student gain | UNSUPPORTED | KD-over-CE +0.44/-3.56pp |
| The secure implementation reproduces the selected direction | SUPPORTED (seed42) | C10/C100 TFE gates passed |
| The effect is stable across seeds | UNSUPPORTED | only seed42 was run |
| Stage C or end-to-end convergence improves | UNSUPPORTED/NOT TESTED | forbidden by task scope |
| Released model has end-to-end record DP | UNSUPPORTED/NOT TESTED | Stage B is non-DP |
| Communication cost is measured | UNSUPPORTED | bytes/rounds remain N/A |

All claims are single-seed development results. No mean/std, paired-seed
stability, final-test, Stage-C, end-to-end, or DP-convergence claim is made.

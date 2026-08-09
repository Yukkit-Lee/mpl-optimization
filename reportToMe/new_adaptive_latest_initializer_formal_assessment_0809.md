# Adaptive clipping formal assessment (2026-08-09)

## Outcome

The frozen three-seed plaintext gate completed. It does **not** authorize a TFE
follow-up and does not support a positive utility claim for Adaptive DP.

| Mode | Final development accuracy | Early AUC | Final clipping bound | Accounted epsilon | Count releases |
|---|---:|---:|---:|---:|---:|
| Fixed DP | 58.673% +/- 0.214 pp | 58.321% +/- 0.149 pp | 2.000 +/- 0.000 | 3.0 | 0 |
| Adaptive DP | 58.807% +/- 0.023 pp | 58.328% +/- 0.145 pp | 1.388 +/- 0.091 | 3.0 | 63 |

The Adaptive-minus-Fixed final gain is `+0.133 +/- 0.192 pp`. Per-seed gains
are -0.040, +0.340, and +0.100 pp for seeds 42, 43, and 44. The mean is below
the frozen +1.0 pp threshold and the sign is not positive for every seed.
Early AUC is effectively tied (+0.0067 pp on average). The epsilon check passes
for all six runs, including the Adaptive count mechanism.

## Scope and decision

- Dataset: CIFAR-10 only.
- Initializer: latest audited C5 anchor-refined initializer.
- Budget: 320 updates, epsilon=3, delta=1e-5.
- Evaluation: development validation only; official test and locked holdout are
  not read.
- Fixed and Adaptive gradient mechanisms use common random numbers per seed.
- The clipped-example fraction in Figure 6 is explicitly a non-private
  diagnostic, not a protocol release.
- Because the positive gate failed, no TFE experiment was launched. Expanding
  the tuning grid after seeing this result would violate the frozen protocol.

## Reproducible artifacts

- Freeze: `new_configs/new_paper_adaptive/new_adaptive_latest_initializer_formal_freeze.json`
- Runner: `new_scripts/new_paper_adaptive/new_run_latest_initializer_formal_plaintext.py`
- Full result: `new_log/new_paper_adaptive/new_adaptive_latest_initializer_formal_plaintext.json`
- Per-run table: `new_expTab/new_paper_adaptive/new_adaptive_latest_initializer_formal_plaintext.csv`
- Summary table: `new_expTab/new_paper_adaptive/new_adaptive_latest_initializer_formal_summary.csv`
- Figure builder: `new_scripts/new_paper_adaptive/new_build_adaptive_figure6.py`
- Figure: `new_expFig/paper_experiment/new_figure6_adaptive_clipping_matched_pea.{png,pdf}`

The manuscript Table 5 now reports this matched pair instead of an unexecuted
eight-cell factorial table. This preserves the negative result and prevents
placeholder combinations from being mistaken for experimental evidence.

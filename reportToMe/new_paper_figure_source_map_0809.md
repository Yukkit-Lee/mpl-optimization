# Paper figure source map and evidence boundaries (2026-08-09)

## Rebuilt figures

| Figure | Output | Frozen sources | Repeats / uncertainty | Audited boundary |
|---|---|---|---|---|
| 2 | `new_expFig/paper_experiment/new_figure2_dual_backend_pea_i_errorbars.{png,pdf}` | `new_true_ours_master_results.json`; corrected-v2 SPU plot CSV | TFE CIFAR: n=3 with sample-SD I-bars; TFE Tiny/SPU: n=1, no bars | Development validation; conditional Stage-C epsilon; rows not hardware-normalized |
| 3 | `new_expFig/paper_experiment/new_figure3_compact_carrier_fast_convergence.{png,pdf}` | Nine frozen Module-A JSON runs | n=3 with sample-SD I-bars | Official-test values from fixed-interval read-only formal reporting; Ours truncated at last seed's first frozen-target release |
| 4 | `new_expFig/paper_experiment/new_figure4_initializer_source_lollipop.{png,pdf}` | Stage-B initialization-source ablation CSV | seed42, n=1, no bars | Development validation; no test or locked holdout read |
| 5 | `new_expFig/paper_experiment/new_figure5_anchor_convergence_pea.{png,pdf}` | Stage-B trajectory JSON | seed42, n=1, no bars | Development validation; no test or locked holdout read |

The builder is `new_scripts/new_paper_figures/new_build_figures_2_to_5.py`.
It aborts on missing files, incomplete master status, an unexpected evaluation
scope, or a Stage-B test/holdout flag.

## Important limitations

1. Figure 5 cannot honestly contain I-shaped error bars because the archived
   convergence trajectories contain only seed 42. Error bars were omitted and
   `n=1` is explicit rather than converting within-run points into fake repeats.
2. Figure 3 is not development validation. Its original frozen run explicitly
   records `test_usage=fixed_interval_readonly_formal_reporting`; the revised
   caption now states this instead of leaving the split ambiguous.
3. Figure 2 SPU curves are corrected-v2 seed42 results. They demonstrate backend
   reproduction, not a hardware-normalized TFE-versus-SPU comparison.
4. Figure 4 and Table 4 currently contain the same six cells and four methods.
   The redesigned plot makes the heterogeneity pattern easier to read, while the
   table preserves exact values. Table 4 should be removed only after the user
   decides whether exact in-body values are worth the duplication.
5. Figure 6 and Table 5 remain pending a new matched Adaptive experiment. The
   historical Fixed-vs-Adaptive grid is negative and cannot be relabeled as a
   positive contribution.
6. Figure 7 is not yet supported by a matched owner-count × dimension × network
   matrix. It should be skipped unless the evidentiary gain justifies that cost.

# Paper figure source map and evidence boundaries (2026-08-09)

## Rebuilt figures

| Figure | Output | Frozen sources | Repeats / uncertainty | Audited boundary |
|---|---|---|---|---|
| 2(a) | `new_expFig/paper_experiment/new_figure2a_tfe_pea_i_errorbars.{png,pdf}` | `new_true_ours_master_results.json` | TFE CIFAR-10/100: n=3 with sample-SD I-bars; Tiny: n=1, no bar | Development validation; original TFE log-seconds trajectories and Stage-B-to-C dotted lines |
| 2(b) | `new_expFig/paper_experiment/new_figure2b_spu_short_axis_deltas.{png,pdf}` | corrected-v2 SPU plot CSV | SPU: n=1, no uncertainty bars | Development validation; paired short windows with y=accuracy change from each method's first release (pp), absolute endpoints printed, explicit break marks and all 8/2/18 releases; T@Target uses original cumulative time; not hardware-normalized against TFE |
| 3 | `new_expFig/paper_experiment/new_figure3_compact_carrier_fast_convergence.{png,pdf}` | Nine frozen Module-A JSON runs | n=3 with sample-SD I-bars | Official-test values from fixed-interval read-only formal reporting; per-seed convergence requires all later points within terminal +/-0.5pp, then uses latest seed cut |
| 4 | `new_expFig/paper_experiment/new_figure4_initializer_source_bars.{png,pdf}` | Stage-B initialization-source ablation CSV | seed42, n=1, no error bars | Development validation; 2x3 grouped bars; no test or locked holdout read |
| 5 | `new_expFig/paper_experiment/new_figure5_anchor_convergence_pea.{png,pdf}` | Stage-B trajectory JSON | seed42, n=1, no bars | Development validation; no test or locked holdout read |
| 6 | `new_expFig/paper_experiment/new_figure6_user_tuned_plaintext.{png,pdf}` | `new_user_tuned_fixed_vs_adaptive_formal.json` | n=3 with sample-SD I-bars | Plaintext CIFAR-10 development validation; same epsilon/data/initializer/seeds but independently configured optimizer/compute parameters; clipped fraction is a non-private diagnostic; not TFE |

Each figure has an independent documented builder:

| Figure | Builder | Run from repository root |
|---|---|---|
| 2(a)/(b) | `new_scripts/new_paper_figures/new_build_figure2.py` | `python new_scripts/new_paper_figures/new_build_figure2.py` |
| 3 | `new_scripts/new_paper_figures/new_build_figure3.py` | `python new_scripts/new_paper_figures/new_build_figure3.py` |
| 4 | `new_scripts/new_paper_figures/new_build_figure4.py` | `python new_scripts/new_paper_figures/new_build_figure4.py` |
| 5 | `new_scripts/new_paper_figures/new_build_figure5.py` | `python new_scripts/new_paper_figures/new_build_figure5.py` |
| 6 | `new_scripts/new_paper_figures/new_build_figure6.py` | `/root/autodl-tmp/new_envs/distillation/bin/python new_scripts/new_paper_figures/new_build_figure6.py --input new_log/new_paper_adaptive/new_user_tuned_fixed_vs_adaptive_formal.json --output-stem new_figure6_user_tuned_plaintext --table new_expTab/new_paper_adaptive/new_figure6_user_tuned_plaintext_summary.csv` |

Shared paths, colors, saving and I-bar helpers are in
`new_scripts/new_paper_figures/new_paper_figure_common.py`; it contains no
Figure-specific data selection. The former
`new_build_figures_2_to_5.py` remains only as a thin compatibility wrapper that
calls the four independent builders. Each builder documents safe visual
parameters versus evidence-defining parameters. Plotting does not compute or
compare the SHA256 of any input JSON/CSV and does not read a hash manifest or
completion-audit file. A user may therefore edit a source JSON and redraw
immediately. File parsing, required-field/cardinality checks, and evaluation-
scope gates remain active because they are not hash checks.

## Important limitations

1. Figure 5 cannot honestly contain I-shaped error bars because the archived
   convergence trajectories contain only seed 42. Error bars were omitted and
   `n=1` is explicit rather than converting within-run points into fake repeats.
2. Figure 3 is not development validation. Its original frozen run explicitly
   records `test_usage=fixed_interval_readonly_formal_reporting`; the revised
   caption now states this instead of leaving the split ambiguous.
3. Figure 2 SPU short-window curves are corrected-v2 seed42 results. The left
   and right windows retain the original cumulative-time coordinates after a
   seconds-to-minutes unit conversion, while diagonal marks disclose the empty
   interval omitted between Ours and Full. The SPU y coordinate is the exact
   percentage-point difference from that method's first archived accuracy; the
   absolute start/final values are printed, and no smoothing is applied.
   T@Target is computed from the original absolute times, not the displayed
   window width. This demonstrates backend reproduction, not a
   hardware-normalized TFE-versus-SPU comparison.
4. The former combined `new_figure2_dual_backend_pea_i_errorbars` is retained
   only as a superseded historical artifact. The manuscript uses the split
   2(a)/2(b) files so unlike y-axis semantics are never placed in one figure.
5. Figure 4 and Table 4 currently contain the same six cells and four methods.
   The redesigned plot makes the heterogeneity pattern easier to read, while the
   table preserves exact values. Table 4 should be removed only after the user
   decides whether exact in-body values are worth the duplication.
6. The current Figure 6 uses the latest user-configured plaintext run for both
   methods. Adaptive gains `+11.600 +/- 1.752 pp`, with all three seed gains
   positive. Fixed uses steps280/LR.013/batch16 and falls from a mean best
   archived accuracy of 56.853% to 47.333% at the final checkpoint; Adaptive
   uses steps160/LR.002/batch32 and finishes at 58.933%. Because steps, batch,
   learning rate and terminal behavior differ, this is a same-epsilon complete-
   configuration comparison rather than a causal isolation of clipping. It is
   plaintext and cannot be described as TFE evidence. Older matched and
   Adaptive-only-tuned Figure 6 files are retained as historical evidence.
7. Figure 7 is not yet supported by a matched owner-count × dimension × network
   matrix. It should be skipped unless the evidentiary gain justifies that cost.

# SubLevel Detect

## Integrated research: from fitting to hypothesis validation

The physics-structured differentiable model describes the archive and proposes multicomponent candidates under specified priors and selectors. The repository now also contains collision-history modeling, retarding-response analysis and the diagnostic checks that establish their physical interpretation and limits. Follow [the research guide](docs/research_overview.md), [reproduction instructions](docs/research_reproduction.md), and [the detailed Chinese storyline](docs/research_storyline.zh-CN.md). New minimal numerical evidence is in [research_evidence](source_data_package/research_evidence/README.md); manuscript PDFs and new figure archives remain local.

Free K=4 selection is a conditional candidate, distinct from the fixed-spacing H4s test. True H4s holdout has 60 completed units and a 2.36% mean NRMSE improvement with endpoint reversal. Legacy `cv_rmse_mean` in free-K exports is fitting error. The selected-model residual bootstrap never selects K=4 in 30 trials (K=5/6 each 10); the later A5 span-selector pathology is a separate check. A4's fair cross-section comparison is not executed.

[中文 README](README.zh-CN.md)

Source project for testing a four-level Ar I 4s interpretation in an argon Franck-Hertz teaching experiment.

This repository is code-first for normal reproduction: it contains source code, one input spreadsheet, tests, and documentation, while fresh runtime outputs remain under `output/` and are not committed. For manuscript checking, the curated `source_data_package/` directory is committed separately and contains the figure assets, manuscript-facing CSV/JSON tables, retained K=1/K=4 run records, and checksums used by the current paper draft.

## Technical Reports

Implementation details are recorded in:

- `Techique_Report.md`: English technical report.
- `Techique_Report_zh.md`: Chinese technical report.

The reports document the full workflow, loss terms, physical kernel, optimizer split, hyperparameter search, K-neutral selector, forward-prior logic, perturbation structure, ablation design, robustness design, and source-data package validation.

## What This Project Reproduces

The code fits and evaluates a physics-based Frank-Hertz argon response model. The paper-facing question is whether the four lowest NIST Ar I 4s levels improve prediction of a completely held-out retarding-voltage curve relative to one effective threshold. It is organized around the following reproducible workflows:

- Main baseline: forward-evidence preparation, optional hyperparameter optimization, training over candidate level counts, and automatic post-evaluation.
- Ablation baseline: selector-only ablations plus a no-forward-anchor-gap retraining run to test how much the final decision depends on forward anchors.
- Robustness baseline: selector weight perturbation plus leave-one-retarding-voltage-out retraining with the main baseline hyperparameters fixed.
- Sensitivity supplement: forward-anchor prior-strength scan and two K=4 uncertainty summaries under seed jitter, residual bootstrap, noise perturbation, and peak-window-radius perturbation. `conditional_k4_all_fits` is a stress-test drift summary over all fitted K=4 channels; `production_anchor_matched_k4` matches perturbed K=4 channels back to the four production K=4 anchors.
- Supplementary validation suite: selector decontamination audit, circular moving-block residual bootstrap, leave-one-retarding-voltage-out prediction, and a targeted four-hypothesis comparison of the NIST Ar I 4s manifold.

The formal H4s comparison is complete: 60/60 units finished with zero failures (five held-out retarding voltages, four hypotheses, and three optimization restarts). `H4s` lowers fold-level median NRMSE relative to `H1` in four of five conditions, mainly at 4, 6, and 8 V; the direction reverses at 10 V. This supports joint participation of the specified 4s manifold as an interpretable response hypothesis. It does not establish four individually resolved levels or exclude alternative multichannel models with comparable complexity.

The retained physical response audit uses two gates: late-bias and high-retarding-voltage valley-depth.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Quick Reproduction

Run the targeted H4s comparison smoke check on CPU:

```powershell
$smoke = Join-Path $env:TEMP 'fh_validation_smoke'
python run.py --mode smoke --exclude hpopt --validation-only h4s --output $smoke --device cpu
Remove-Item -LiteralPath $smoke -Recurse -Force
```

Run the preregistered formal validation suite with the audited CPU configuration:

```powershell
python run.py --mode fullscan --validation --device cpu
```

The formal command runs `selector`, `bootstrap`, `holdout`, and `h4s` in dependency order. To reproduce only the targeted physical comparison, use:

```powershell
python run.py --mode fullscan --validation-only h4s --device cpu
```

`--validation-only` enables validation automatically and resolves required earlier stages. A compatible completed main baseline under the selected output root is reused; otherwise the runner uses the committed `source_data_package/` as a read-only baseline. Input, configuration, checkpoint, source, and baseline hashes are recorded so incompatible results are not silently mixed or overwritten.

Formal validation can be computationally expensive. Runtime products are written under `output/validation/`, are ignored by Git, and are not part of the source-code release.

## If You Are an AI Agent

Use this sequence to reproduce the study without guessing project layout:

1. Inspect `README.md`, `README.zh-CN.md`, `docs/reproduction.md`, and `docs/outputs.md`.
2. Confirm that `data/argon/FHdata.xlsx` exists. For an audit recomputation, archive any old runtime output first, then write the formal run to `output/`.
3. Run a functional smoke check:

```powershell
python run.py --mode smoke --exclude hpopt --ablation
```

4. Treat smoke outputs as runtime validation only. Delete or ignore smoke outputs before paper-facing analysis.
5. Run the main paper baseline into the recomputation directory:

```powershell
python run.py --mode fullscan --output output
```

6. Run the full baseline with ablation and robustness evidence:

```powershell
python run.py --mode fullscan --output output --ablation --robustness
```

7. Run the forward-prior and uncertainty sensitivity supplement against the recomputed main baseline:

```powershell
python run.py --mode fullscan --output output --exclude hpopt --sensitivity --device cpu
```

8. `output/` is now the formal default output root. Do not mix archived legacy evidence with current `output/` evidence when comparing results.

9. For a faster but non-hyperoptimized reproduction, use:

```powershell
python run.py --mode fullscan --exclude hpopt --ablation
```

10. Summarize conclusions only from the selected output root's `main/fullscan/decision.json`, `main/fullscan/model_selection_table.csv`, `main/paper_summary.json`, `ablation/ablation_summary.csv`, `robustness/robustness_summary.json`, and `sensitivity/` CSV/JSON files.

11. For uncertainty language, use `sensitivity/uncertainty/channel_uncertainty_anchor_matched.csv` for production-anchor matched K=4 intervals. Treat `channel_uncertainty_conditional_k4.csv` and the legacy alias `channel_uncertainty_summary.csv` as conditional stress-test drift, not production energy confidence intervals.

## Audit Recompute Commands

The current recommended full audit sequence writes new results without overwriting the legacy `output/` directory:

```powershell
python run.py --mode fullscan --output output
python run.py --mode fullscan --output output --ablation --robustness
python run.py --mode fullscan --output output --exclude hpopt --sensitivity --device cpu
```

The smoke command remains a functional check only. Remove smoke output directories after use unless they are the explicit subject of a test.

## Experiment Design

Input data:

- Default file: `data/argon/FHdata.xlsx`
- Expected content: Frank-Hertz argon current-voltage curves with accelerating voltage, retarding voltage, curve identifier, and measured current columns.
- Loader behavior: the code resolves common column labels and can read Excel directly; it also includes a fallback reader for `.xlsx` files when the usual Excel dependency is unavailable.

Main baseline workflow:

- The command `python run.py --mode fullscan` builds forward evidence from the observed oscillatory structure.
- Hyperparameter optimization is enabled by default for full scans and can be skipped with `--exclude hpopt`.
- Candidate level counts are scanned across the configured K range.
- Each candidate writes per-level metrics, checkpoints, scorecards, scan tables, and selector diagnostics under `output/main/fullscan/`.
- Automatic post-evaluation writes paper-facing summaries under `output/main/`.

Ablation workflow:

- `--ablation` runs the main baseline first, then launches ablation analysis.
- Selector ablations recompute decisions while removing individual selector components.
- The no-forward-anchor-gap condition retrains with forward anchor priors disabled.
- Ablation outputs are written under `output/ablation/`.

Robustness workflow:

- `--robustness` runs after the main baseline has produced a sweep table.
- Selector perturbation recomputes decisions under a fixed rank-weight grid without retraining.
- Leave-one-Vr-out retrains K candidates after excluding one retarding-voltage curve at a time, using the main baseline hyperparameter configuration rather than running hyperopt again.
- Robustness outputs are written under `output/robustness/`.

Sensitivity workflow:

- `--sensitivity` reuses the existing main fullscan when available and does not rerun hyperparameter optimization when paired with `--exclude hpopt`.
- Prior-strength scanning evaluates forward-anchor strength factors `0`, `0.25`, `0.5`, `1`, `2`, and `4` for K=1..8. The `1x` setting is defined as the actual production prior weights recorded in `main/fullscan/config_used.json`; higher and lower factors scale those recorded weights directly.
- The main workflow exports `main/k_selected_full/prediction_points.csv` with `curve_id,Vr,Va,observed,predicted,residual`. Residual bootstrap samples from these selected-model residuals. If the table is missing, sensitivity fails and asks for the main workflow to be rerun.
- Uncertainty scans evaluate seed-dependent initialization jitter, residual bootstrap, curve-level noise perturbation, and peak-window-radius perturbation while retaining K=1..8 scorecards.
- `channel_uncertainty_conditional_k4.csv` summarizes all fitted K=4 channels and is a stress-test drift table.
- `channel_uncertainty_anchor_matched.csv` matches perturbed K=4 fits to the four production K=4 channel anchors and is the candidate table for production K=4 uncertainty language.
- Sensitivity outputs are written under `output/sensitivity/`.

Preregistered validation workflow:

- Selector audit keeps the production selector frozen while removing individual and grouped diagnostics, adding a fit-complexity-only control, and reporting rank correlations and selected-K distributions.
- Circular moving-block residual bootstrap resamples centered residuals within each curve. The formal design uses a 7-point main block with 5- and 9-point sensitivity checks and reports the selected-K distribution and Wilson interval for the K=4 rate.
- Leave-one-retarding-voltage-out prediction evaluates K=1..8 and the training-fold selected model on unseen curves. Its primary endpoints are zero-shot RMSE, MAE, and range-normalized RMSE; a secondary 21-point calibration fits only gain, bias, and accelerating-voltage offset.
- The targeted H4s comparison evaluates `H1`, `H1+B`, `H4s`, and `H4s+B` on the same five held-out retarding-voltage curves. `H4s` uses the four NIST Ar I 4s energies with one bounded common energy adjustment; `B` is the existing smooth high-energy loss term, not a fifth excitation channel.
- The H4s stage reports paired zero-shot RMSE, MAE, and NRMSE differences without automatically declaring support or rejection. Seeds are optimization restarts rather than independent samples; formal results use five fold-level medians, and smoke output is never claim-evaluable.
- H4s seed aggregation first forms within-seed paired differences and then takes the fold-level median. Failed seeds and folds remain in the expected denominator. A fixed runtime audit selected CPU with four single-thread workers; see `docs/h4s_performance_audit.md`.
- Every stage writes a manifest, progress state, unit-level records, stage status, and `stage_result.json`. Failed units remain in the registered denominator, and an incomplete or failed validation run returns a non-zero exit code.

Device and path controls:

```powershell
python run.py --mode fullscan --input data/argon/FHdata.xlsx --output output --device cpu
```

`--device` accepts `cpu`, `cuda`, or `auto`. The project defaults to CPU scheduling; `auto` is treated as CPU, and CUDA is used only when `--device cuda` is explicitly requested.

## Data Analysis

Primary decision files:

- `output/main/fullscan/decision.json`: selected level count and decision diagnostics.
- `output/main/fullscan/model_selection_table.csv`: scored candidate K table used for model selection.
- `output/main/fullscan/scan_summary.csv`: fit, retained seed-summary diagnostic, structure, and physical-response metrics by candidate.
- `output/main/k_selected_full/prediction_points.csv`: selected-model point predictions and residuals used by residual-bootstrap sensitivity.
- `output/main/paper_summary.json`: compact paper-facing summary of the selected result.

Structure analysis:

- `structure_metrics.csv` summarizes peak-valley preservation and flatline guarding.
- `peak_valley_segments.csv`, `curve_structure_summary.csv`, and `class_structure_summary.csv` provide segment-level and grouped diagnostics.

Physical response analysis:

- `vr_physical_response.csv` contains per-curve physical-response diagnostics.
- `vr_physical_response.json` stores physical-response summaries and grouped rows.
- The reported physical caveats are late-bias and high-retarding-voltage valley-depth.

Ablation analysis:

- `output/ablation/ablation_summary.csv` compares the main baseline, no-forward-anchor-gap retrain, and selector-only variants.
- `output/ablation/selector_ablation_decision.json` stores detailed selector decisions for each ablation group.
- `output/ablation/ablation_report.md` is the human-readable ablation report.

Robustness analysis:

- `output/robustness/selector_weight_perturbation.csv` reports each rank-weight perturbation scenario and its selected K.
- `output/robustness/selector_weight_perturbation_summary.csv` summarizes the selected-K distribution across perturbations.
- `output/robustness/leave_one_vr_out_summary.csv` reports the selected K and key metrics for each excluded retarding-voltage curve.
- `output/robustness/robustness_summary.json` is the compact paper-facing robustness summary.

Supplementary validation analysis:

- `output/validation/experiment_manifest.json` records the requested stages and hashes of all identity-bearing inputs.
- `output/validation/progress.json` exposes resumable unit progress without changing the registered denominator.
- `output/validation/<stage>/stage_manifest.json` and `stage_result.json` bind each result to its exact stage identity.
- `output/validation/validation_summary.json` and `validation_summary.md` summarize completed evidence, inference boundaries, and limitations after all requested stages finish.

For manuscript writing, cite the generated JSON/CSV files rather than intermediate checkpoints. The committed `source_data_package/` is a curated manuscript archive; newly generated checkpoints and runtime outputs outside that directory remain ignored by git.

## Source Data Package

The `source_data_package/` directory contains the current manuscript source-data package:

- `manuscript_source_tables/`: 29 CSV/JSON files used by manuscript figures, model-selection tables, physical-response audits, ablation discussion, robustness checks, and corrected sensitivity/uncertainty checks.
- `figures/main/` and `figures/supplementary/`: 71 generated figure files, including the R/GGPlot2 sensitivity PNG/PDF/SVG/TIFF exports.
- `run_records/k1_full/` and `run_records/k_selected_full/`: 16 retained run-record files for K=1 and the selected K=4 run, including metrics, parameters, prediction points, logs, scorecards, status files, and checkpoints.
- `output_results/`: selected `output/` decision tables, summaries, prediction residual points, robustness outputs, and sensitivity outputs that support the manuscript claims.
- `FILE_INDEX.csv`, `SHA256SUMS.txt`, `validation_report.json`, and `source_data_package_manifest.md`: package paths, source paths, sizes, SHA256 checksums, validation results, and the readable manifest.

The package is rebuilt from an existing formal output root with:

```powershell
$env:SUBLEVEL_OUTPUT='output'
python scripts/build_source_data_package.py
```

If manuscript visualization source tables or figures are stored outside this repository, provide them explicitly with `SUBLEVEL_SOURCE_TABLES` and `SUBLEVEL_FIGURES`.

The current package contains 309 files. Validation passes for the 29 source tables, 71 figure assets, `prior_strength=1x selected_k=4`, the four production anchor energies, and the residual-bootstrap scale check.

## Commands

Smoke check:

```powershell
python run.py --mode smoke --exclude hpopt
python run.py --mode smoke --exclude hpopt --ablation
```

Full main baseline:

```powershell
python run.py --mode fullscan
```

Full main baseline without hyperparameter optimization:

```powershell
python run.py --mode fullscan --exclude hpopt
```

Full main baseline plus ablation:

```powershell
python run.py --mode fullscan --ablation
```

Full main baseline plus ablation and robustness:

```powershell
python run.py --mode fullscan --ablation --robustness
```

Forward-prior and uncertainty sensitivity supplement:

```powershell
python run.py --mode fullscan --exclude hpopt --sensitivity --device cpu
```

Preregistered supplementary validation:

```powershell
python run.py --mode fullscan --validation --device cpu
python run.py --mode fullscan --validation-only selector --device cpu
python run.py --mode fullscan --validation-only bootstrap --device cuda
python run.py --mode fullscan --validation-only holdout --device cuda
python run.py --mode fullscan --validation-only h4s --device cpu
```

## Outputs

Main baseline evidence:

- `output/main/fullscan/decision.json`
- `output/main/fullscan/model_selection_table.csv`
- `output/main/fullscan/scan_summary.csv`
- `output/main/fullscan/vr_physical_response.csv`
- `output/main/paper_summary.json`
- `output/main/paper_summary.md`

Ablation baseline evidence:

- `output/ablation/ablation_summary.csv`
- `output/ablation/ablation_report.md`
- `output/ablation/selector_ablation_decision.json`

Robustness baseline evidence:

- `output/robustness/robustness_summary.json`
- `output/robustness/selector_weight_perturbation.csv`
- `output/robustness/selector_weight_perturbation_summary.csv`
- `output/robustness/leave_one_vr_out_summary.csv`

Sensitivity supplement evidence:

- `output/sensitivity/prior_strength/prior_strength_selection.csv`
- `output/sensitivity/prior_strength/prior_strength_channel_drift.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_samples.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_conditional_k4.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_anchor_matched.csv`
- `output/sensitivity/uncertainty/channel_uncertainty_summary.csv`
- `output/sensitivity/uncertainty/uncertainty_selection_summary.csv`
- `output/sensitivity/sensitivity_summary.json`

Supplementary validation evidence:

- `output/validation/experiment_manifest.json`
- `output/validation/progress.json`
- `output/validation/selector_audit/stage_result.json`
- `output/validation/block_bootstrap/stage_result.json`
- `output/validation/holdout/stage_result.json`
- `output/validation/h4s_comparison/stage_result.json`
- `output/validation/h4s_comparison/h4s_comparison_summary.json`
- `output/validation/h4s_comparison/paired_contrasts.csv`
- `output/validation/h4s_comparison/fold_seed_medians.csv`
- `output/validation/h4s_comparison/fold_median_contrasts.csv`
- `output/validation/validation_summary.json`
- `output/validation/validation_summary.md`

## Tests

```powershell
python -m compileall -q run.py src/sublevel_detect
python -m pytest
```

The smoke test creates a temporary output directory inside this repository and removes it before finishing.


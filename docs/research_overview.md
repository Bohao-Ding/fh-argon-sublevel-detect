# Complete-curve modeling and effective energy inference

The current `spectrum-revised` method estimates a nonnegative effective response
distribution without anchoring four atomic states. Neural optimization estimates
the physical response parameters; the original free-K=4 work remains its foundation.
The completed revision uses 644 points, nested whole-condition prediction, matched
phase controls and known-truth recovery. Mean H1/G1/C NRMSE is
0.09348/0.09467/0.04925. Better simple-model solutions lie near 11.73 eV;
the C mode is 10.355 eV (start range 10.285–10.885) and is not a stable
concentration in the independently known 4s range. The corrected 10 V C stress
score is 0.06610, weakening a measurement-error inference from earlier failures.

Known-response, no-added-noise controls recover a 0.15 eV Gaussian width as
0.150004 eV without curvature and 0.658523 eV at the real-data-selected 1e-4
coefficient. Prediction and spectral recovery are separate results.
See [protocol](SPECTRAL_REVISION.md), [result discussion](SPECTRAL_REVISION_RESULTS.zh-CN.md),
[reproduction](spectral_reproduction.md), and [245-file revised evidence](../source_data_package/spectral_revision_evidence/README.md).
The 1049-fit primary matrix passes every 0.005/0.0025 eV quadrature check;
896 fits meet local stationarity and 153 do not. These retained statuses do
not establish a global solution. Local discrete exploration is not run.

## Historical 805-point studies and earlier spectrum implementations

The research starts with a physics-structured differentiable model trained by neural-network optimization. It describes the archived argon Franck–Hertz curves and supplies multicomponent candidates under specified priors and selectors. Subsequent checks ask what those candidates establish: cross-condition prediction, explanatory necessity, macroscopic response structure, and level specificity.

The measurement archive contains five retarding-voltage curves, 161 points each (805 measured points). Every measured analysis reuses it. Resampling, dataset splits, injection and virtual scans add computational checks, not independent measurements.

| Research question | Implementation | Verified evidence and boundary |
|---|---|---|
| Can the physical response model describe curves and propose a candidate? | Existing `src/sublevel_detect`, `run.py` | Free-K fitting RMSE: K=1 0.08020, K=4 0.07735 μA; production selector K=4. Legacy `cv_rmse_mean` is fitting error. |
| Does a fixed spacing predict held-out conditions? | Existing H1/H4s validation | Five folds × four hypotheses × three restarts = 60 completed units; H1/H4s mean NRMSE 0.12918/0.12613, 2.36% improvement, endpoint reversal. |
| Are multiple nearby energies necessary for the main peaks? | `NewModel-TEST` | v2l H1 passes 15/15 shape checks; H4s 12/15, unconverged, lowest-channel share 91.36%. A4 is not executed. |
| Is there information beyond amplitude rescaling? | `FULL-REtry`, `EXPERIMENT_V2` | Four steady intervals improve about 41–55%; tested variants bound the claim. A15 rewrites the same predictions as band-specific amplitudes and shifts. |
| Does this structure identify NIST 4s spacing? | Template alternatives and calibrated injections | 74.8%/69.7% alternatives fit at least as well; best prediction separation below about 5.8×10⁻⁴σ in the tested prediction space. Detection of existence does not establish spacing identity. |

The free K=4 candidate and the preset-relative-spacing H4s hypothesis are different models. Removing the complexity group selects 8, removing the prior anchor selects 7, and omitting the 10 V curve selects 1. The preserved selected-model residual bootstrap selects 5/6/7/8 in 10/10/5/5 of 30 trials, never 4; 5 and 6 tie. This is a different protocol from the moving-block validation stage. The later A5 span selector is also different from the early K selector; its null argmin bias must not be assigned retrospectively to K selection.

The integrated repository preserves the original neural entrypoint and adds companion modules without changing their numerical model definitions. New A7 paths read packaged H4s and v2l receipts and fail if required evidence is missing. The neural source self-check is scoped to the formal neural package; version names in preserved companion research are retained.

[Detailed Chinese argument and source links](research_storyline.zh-CN.md) · [Standalone reproduction](research_reproduction.md) · [Minimal evidence](../source_data_package/research_evidence/README.md)

Current article sources and new figure/PDF assets remain local. Existing remote neural source-data assets and Git history are retained. Module documents are historical records; corrected interpretation of old injection recovery is given by A5/A14 and the storyline. Apparatus metadata and DOI archiving remain unresolved.

The current primary interpretation locates energy concentration before any optional discrete hypothesis. `concentration.csv` reports the connected main half-height interval and effective mass in the known 4s range; recompute with `python scripts/summarize_spectral_concentration.py source_data_package/spectral_evidence_v1`. These density-derived summaries do not alter training or model selection.

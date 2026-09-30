# From curve description to physical-hypothesis validation

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

# Minimal research evidence

This additive package connects preserved physics-structured fits to the subsequent hypothesis checks. It contains numerical tables and receipts, not manuscript PDFs, figure archives, training histories or checkpoints.

- `h4s_holdout/`: formal 60-unit comparison and derived manuscript-facing tables. The original run identity is retained; optimizer seeds are restarts, not independent measurements.
- `neural_selector/`: deterministic group-removal recalculation from the preserved fitted candidates; no new model training. This audit is separate from the original 23 weight-perturbation scenarios.
- `collision_history/`: v2l H1/H4s results, predictions and morphology audit. H4s did not plateau; A4's matched cross-section experiment was not executed.
- `full_retry/`: compact frozen foundational results and virtual sensitivities. Historical recovery interpretations are superseded by A5/A14.
- `diagnostics_v2/`: A-series CSV/JSON aggregates, including failed gates and A15's numerical identity.
- `SOURCE_MAP.json` / `FILE_INDEX.csv`: original workspace-relative sources and file provenance. Historical paths inside receipts describe the original run; reproduction reads packaged files.
- `SHA256SUMS.txt`: hashes of actual published bytes, independent of historical DataFrame hashes. `.gitattributes` preserves these bytes on checkout.

Earlier neural-model selection, ablation and sensitivity evidence remains in `../output_results/`. Free K=4 components differ from the fixed-relative-spacing H4s hypothesis. Legacy `cv_rmse_mean` equals fitting error in the retained free-K export and is not held-out cross-validation.

All measured evidence uses the same five-curve, 805-point archive. Resampling, splits, injection and virtual scans do not create new measurements. Read [the research guide](../../docs/research_overview.md) and [reproduction instructions](../../docs/research_reproduction.md), then run `python scripts/verify_research_evidence.py` from the repository root.

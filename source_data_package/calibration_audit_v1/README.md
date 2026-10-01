# Frozen common affine calibration audit

This second analysis uses the existing 644-point physical responses. It estimates only training-curve readout coefficients by equal-curve, range-normalized least squares. Four outer C inner decisions and the final decision choose identity; no new common correction is adopted and no neural kernel or spectrum is changed. The Vr-dependent readout is exploratory and is not a verified instrument calibration.

`summary.json` records choices, score means, unchanged mode, hypothetical voltage-scale range, input and source hashes. `scores.csv`, `outer_fold_medians.csv`, `selection.csv`, `readout_parameters.json`, and `prediction_points.csv` expose all comparisons. `features.csv` and `feature_regressions.csv` describe visible extrema and condition shifts on the original 0.5 V grid, without absolute collision-order assignment. `existing_nuisance.csv` reports the original fitted per-curve current terms. No 10 V observation enters selection or calibration coefficients.

The frozen spectral package supplies prior parameters; twelve additional selected-inner C receipts are in `../calibration_reference_v1`. Run from the repository root: `python -B scripts/audit_affine_calibration.py --output output/calibration_audit_replay` with an empty output directory. All numeric tables and parameter coefficients reproduce without checkpoints or new training. See `../../docs/AFFINE_CALIBRATION_AUDIT.md` for interpretation and sources.

`SOURCE_MAP.json` and `FILE_INDEX.csv` map original bytes. The SHA-256 list covers every released file except itself. Formal derived outputs remain local.

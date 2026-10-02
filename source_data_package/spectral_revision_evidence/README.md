# Revised effective-spectrum evidence

Independent release; historical spectrum and calibration evidence are unchanged. One 805-point archive; 644 points (0/4/6/8 V) train and select; unchanged 10 V is scored only after freezing.

`claim_summary.json` and `outer_seed_medians.csv` summarize actual complete-condition prediction. Starts are optimizer variation, not new measurements. `selection.csv` retains nested one-SE decisions. `phase_scores.csv` compares fixed phase-on/off candidates at each training-selected C penalty; these are matched diagnostics, not reselection on outer outcomes. `concentration.csv` and `peak_threshold_sensitivity.csv` distinguish a peak, diffuse/boundary spectra, and agreement across starts. Atomic quantiles and integrated continuous CDF quantiles use different definitions.

`bootstrap_seed_medians.csv` reports 30 conditional block resamples after median over three starts. `bootstrap_density_band.csv` uses those replicate-centered densities. Kernel, domain and smoothing are fixed; these intervals do not replace condition holdouts or sensitivity fits.

`recovery_seed_medians.csv` uses three-curve synthetic recovery; `matched_recovery_seed_medians.csv` compares free/fixed kernels on the same four curves and same noise realization. Synthetic controls are not new measurements or a formal power calculation. Effective response masses are not atomic populations.

Parameter-only receipts reproduce real predictions and distributions without checkpoints:

```powershell
python scripts/replay_spectral_revision.py
```

`fit_status.csv` retains all stationary/iteration-limit judgments; stationarity is local. `quadrature.csv` tests integration, not experimental resolution. `SOURCE_MAP.json` records original byte hashes and transformations. `FILE_INDEX.csv` omits itself/checksum list; `SHA256SUMS.txt` covers the index and every other release file except itself. Full outputs and plots remain local.

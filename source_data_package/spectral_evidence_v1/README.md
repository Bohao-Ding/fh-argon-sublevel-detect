# Minimal spectral inference evidence

One archived dataset; 0/4/6/8 V training (644 points), unchanged 10 V final known-anomaly stress. Effective excitation response density is conditional on the kernel; it is not an atomic continuum.

Completed: four nested outer holdouts, three outer/final starts, 30 conditional residual resamples, 20 synthetic datasets. Final grid: 0.01 eV. Fit statuses (unique units): {'plateau': 944, 'epoch_limit': 399, 'ineligible': 16}. Epoch-limit fits are not convergence.

`concentration.csv` locates the connected main half-height lobe and its mass, and compares the known Ar I 4s range (11.54835442–11.82807116 eV). Continuous inversion locates effective energy concentration; it does not naturally imply four separated states. Local discrete comparisons are supplementary exploration. Missing inner local choices remain missing; diagnostic candidate scores are not substituted for a nested selected-model score.

`summary.json` and `outer_seed_medians.csv` give complete-curve prediction scores. `matched_comparisons.csv` restricts contrasts to conditions available for both candidates; `claim_summary.json` excludes incomplete candidates from four-condition means. `selection.csv` contains every one-SE decision; `windows.json` records eligible windows. `bootstrap.csv` uses conditional training-objective candidate ranking, not holdout selection. `synthetic_recovery_counts.csv` gives best held-out candidates and false-splitting controls; these are synthetic checks, not new observations or sufficient statistical power.

Parameter-only unit receipts allow replay without checkpoint files. From the repository root:

```powershell
python scripts/replay_spectral_evidence.py
```

Old 805-point evidence and A1–A15 retain their original scope; A4 remains unexecuted. `SOURCE_MAP.json` maps original bytes and transformations. `FILE_INDEX.csv` indexes published files excluding itself and the checksum list; `SHA256SUMS.txt` covers all files except itself, including the index.

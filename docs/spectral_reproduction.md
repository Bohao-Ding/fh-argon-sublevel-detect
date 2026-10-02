# Revised spectrum: standalone reproduction

Use the existing Python requirements. CPU execution needs no checkpoint, external
work folder, cross-section table, or pretrained model. The current protocol and
actual results are in [SPECTRAL_REVISION.md](SPECTRAL_REVISION.md) and
[the discussion](SPECTRAL_REVISION_RESULTS.zh-CN.md).

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -B run.py --experiment spectrum-revised --mode smoke --device cpu --output output/revision_smoke
python -B scripts/replay_spectral_revision.py
```

Smoke is engineering-only and its disposable output should be removed after checks.
The committed parameter-only release supports a lightweight replay without any
local fullscan output. Replay verifies source and input bytes, all 244 release
hashes, original observations, 24,633 point predictions, subset invariance, scores,
densities and conditional bootstrap bands. The recorded maximum prediction error
is 8.88e-16 microampere in the reference environment. It does not retrain models.

For a fresh scientific run, use the fullscan and companion commands in the protocol.
Each CPU worker uses one numerical-library thread. The accepted primary run uses
0.005 eV quadrature, checks every fit at 0.0025 eV, and freezes selection before
10 V. The initial 0.01 eV revision failed this gate and is retained as a failure.
The builder refuses replacing a populated release: pass a new empty destination
for a new export, using `--help` for arguments. Figures can be regenerated locally:

```powershell
Rscript scripts/plot_spectral_revision.R source_data_package/spectral_revision_evidence <local-figure-directory>
```

R dependencies are ggplot2, patchwork, svglite and ragg. Editable PDF/SVG and
600 dpi PNG/TIFF stay local with manuscripts and complete optimizer histories.

## Historical spectral_v1 reproduction (preserved scope)

# Spectrum experiment: execution and parameter-only replay

Install existing Python requirements. No pretrained model, external work folder, GPU
service or cross-section table is required. See [the frozen protocol](SPECTRAL_PROTOCOL.md)
for scientific choices and the definition of effective excitation response density.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -B run.py --experiment spectrum --mode smoke --device cpu --output output/spectral_smoke
python -B run.py --experiment spectrum --mode fullscan --device cpu --output output/spectral_v1
```

Four CPU workers each use one numerical-library thread. Smoke reduces epochs, starts
and replicate counts and has `scientific_run=false`. Inspect and remove only its new
output directory afterward. Fullscan completes the approved matrix; formal receipts
and numerical refinement attempts stay local. Omitting `--experiment` preserves the
old CLI behavior.

Unit identities include training data, settings and source bytes; checkpoints have
exact hashes. The identical command reuses compatible completed units. Changed input,
source bytes or settings refuse reuse. Legacy checkpoints and the interrupted window-
integration attempt cannot initialize this schema. A stopped unit without a final
receipt trains afresh. Do not edit identities to force recovery.

The released formal run rejects `grid_0.02` and uses fresh fits in `grid_0.01`, checked against 0.005 eV. All 1343 fitted measures pass, with maximum prediction difference 7.18e-5 microampere. Numerical grid spacing is not experimental resolution. The 1359 slots comprise 944 plateau, 399 epoch-limit and 16 ineligible states. All nine final stage-1 fits reach the epoch limit.

Outputs are in `grid_0.02` or a finer accepted grid. `fit_status.csv` distinguishes
plateau, epoch limit and ineligible hypotheses. `outer_scores.csv` contains whole-curve
predictions; `inner_scores.csv`, `selection.csv` and `windows.json` document decisions.
Densities, point predictions, bootstrap and synthetic recovery tables are retained.
`quadrature.csv` checks every fitted measure; refinement is automatic if a prediction
difference reaches 1e-4 microampere. Training diagnostics are not validation scores.
Bootstrap winners are conditional training-objective rankings. Synthetic results do
not imply experimental replication or sufficient statistical power.

The committed release is already populated: use replay directly for a lightweight check. To create another release after a complete formal run, pass a new empty destination to the builder (see `--help`); it refuses overwriting the frozen release:

```powershell
python scripts/replay_spectral_evidence.py
```

The original publication commands were:

```powershell
python scripts/build_spectral_evidence.py
python scripts/replay_spectral_evidence.py
```

The builder rejects smoke/incomplete matrices and keeps minimal tables, real point
predictions, density references and parameter-only receipts. Replay verifies every
published byte, source hashes and the shared spreadsheet, then reconstructs real
predictions without checkpoints or local training outputs. Maximum discrepancy must
be below 1e-6 microampere. The separate source map records transformations; old frozen
research evidence remains unchanged. The 252-file release contains 225 parameter receipts; the audited replay rebuilds 43,470 prediction points from 150 real model states, with maximum discrepancy 2.38e-7 microampere.

The selected regularizer is 1e-4 and final local family is d1. Four-condition mean outer NRMSE is 0.10833/0.10824/0.12242 for H1/G1/C. C does not improve overall prediction. NIST is better than equal spacing in two of four conditions; synthetic controls do not recover four-point template identity. The 10 V stress score cannot change these selections. Outer 6/8 V inner local choices are missing; do not substitute a diagnostic candidate score as a completed nested selected-model estimate.

Figures use R packages ggplot2, patchwork, svglite and ragg:

```powershell
Rscript scripts/plot_spectral_manuscript.R source_data_package/spectral_evidence_v1 <local-figure-directory>
```

SVG/PDF and 600 dpi PNG/TIFF remain local. The script uses published tables and
distinguishes fitting, whole-curve prediction and frozen 10 V stress.

The historical spectral_v1 interpretation locates energy concentration before any optional discrete hypothesis. `concentration.csv` reports the connected main half-height interval and effective mass in the known 4s range; recompute with `python scripts/summarize_spectral_concentration.py source_data_package/spectral_evidence_v1`. These density-derived summaries do not alter training or model selection.

The second common affine readout audit also uses frozen kernel parameters, with twelve
additional inner C references and separate byte manifests. Run
`python -B scripts/audit_affine_calibration.py --output output/calibration_audit_replay`
in an empty directory; it requires no local checkpoints or new training. All four
outer inner decisions and the final decision retain identity readout. The optional
Vr-dependent diagnostic is not selected, and spectra stay unchanged. See
[the calibration audit](AFFINE_CALIBRATION_AUDIT.md). Remove disposable replay output
after comparing its numerical tables with `source_data_package/calibration_audit_v1`.

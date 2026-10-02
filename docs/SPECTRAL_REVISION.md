# Effective spectrum correction and retraining protocol

This is a new, exploratory analysis of the same archived data after an implementation
and design audit. Historical sources, results and byte manifests remain unchanged.
The primary question is the position and stability of an effective energy concentration.
Local discrete candidates are not required or run by this protocol.

## Data and isolation

The 644 points at Vr = 0, 4, 6, 8 V enter training and selection. Initialization,
range normalization and residual generation read only each unit's training curves.
The 10 V curve is scored only after choices and numerical checks are frozen. It is
a known-anomaly stress condition; measurement error or overheating has not been proven.
The voltage normalization scale is the maximum training Va (80 V), saved as a model
buffer and specification field. It never depends on a prediction query's extent.

## Revised representation and kernel

H1 is a delta response energy. G1 is a Gaussian normalized on the energy domain.
C is a nonnegative cubic B-spline density with default 0.1 eV knots on 9–16 eV.
Quadrature uses 0.005 eV; every fitted measure is compared with 0.0025 eV predictions
at a maximum absolute tolerance of 1e-4 microampere before the stress curve is scored.
Integration resolution is not experimental or representation resolution.

The first corrected 0.01 eV run retained 1049 units but failed this gate in 108
(94 G1 and 14 C), with maximum difference 3.29e-4 microampere. It stopped before
scoring 10 V and remains in `output/spectral_revision_grid_0.01_failed`, including
its source snapshot and failure receipt. Halving the grid in frozen-parameter
diagnostics passed all 108 failed units (maximum 8.18e-5 microampere); this is only
a refinement diagnostic. The accepted matrix is independently refitted from new
initializations at 0.005 eV, then checked on 0.0025 eV.

The original emission, collection and periodic response equations are retained,
with a fixed voltage scale and an optional common condition shift:
the periodic argument is Va - beta * (Vr - 6 V), beta bounded to -1 to +1 V/V.
Background and collection are evaluated at the original Va. This is a phenomenological
condition-response comparison, not a claim of instrumental voltage calibration.
No NIST energy or spacing anchor enters the loss or model selection.

The shared penalty is 1e-4 times squared, scale-normalized physical corrections
around zero: offset, slope, contrast terms, late baseline, high-energy loss and beta.
Curve gain/bias use their neutral values and existing bounds, at quarter weight.
Unknown amplitude, kernel width, collection parameters and emission scale have no
arbitrary midpoint penalty. This remains an explicit modeling prior, not a calibration.
C curvature candidates are 0, 1e-8, 1e-6, 1e-4; its affine null space is retained and
reported, rather than suppressed with an energy-location prior.

## Optimization and nested prediction

AdamW: lr 0.002, weight decay 0, gradient clipping 5, at most 1500 epochs,
at least 300, patience 150, relative improvement 2e-5, seven-value smoothing.
Then strong-Wolfe L-BFGS (400 iterations) and, for C, a joint SLSQP solve of
physical raw parameters and nonnegative unit-simplex coefficients (600 iterations).
The SQP objective and analytic gradient are multiplied by 1000 to condition its
initial Hessian; reported losses and model predictions retain their original units.
Pilot comparisons found that separate blocks left a nonstationary shared gradient.
The best training objective state is saved. Receipts record optimizer histories,
SLSQP outcomes, the shared raw-coordinate gradient and a direct simplex KKT residual.
"stationary" requires shared gradient <= 1e-5 and simplex residual <= 1e-4;
other units are "iteration_limit". Stationarity is local, not a global optimum proof.
The latter field means the stationarity gate failed within the protocol; solver
termination messages distinguish an actual iteration cap from an earlier stop.

Four outer whole-Vr holdouts each use three training curves. Inner whole-curve
holdouts select phase response and, for C, curvature separately for each family.
The one-standard-error rule first favors no added phase term, then the largest
eligible curvature coefficient. Inner units use start 0. Outer/final units use
three different starts: uniform, narrow and broad coefficient profiles for C;
shifted initial energy, Gaussian width and response-kernel width for H1/G1.
All held-out predictions use gain 1 and bias 0. Computational starts are not
experimental replicates. Scores are median across starts per condition, then
mean across conditions; individual values and ranges are retained.

A matched diagnostic also fits phase off/on for every family, condition and start,
at the outer-training-selected C penalty. It reports separate held-out scores and
does not alter the nested choices. The final four-curve phase pairs are training
diagnostics only. This separates response-phase effects from representation effects;
it does not estimate an instrument voltage gain or use NIST as a calibration anchor.

## Concentration and sensitivity

A continuous concentration must have an interior maximum, peak/mean density >= 1.5,
and a connected half-height interval that does not cover the full domain. Boundary
and diffuse spectra receive explicit statuses. Across starts, all spectra must qualify
and mode range must be <= 0.5 eV for "stable_concentration". These are descriptive
criteria frozen before retraining; threshold sensitivity at ratios 1.25 and 2 is reported.
Quantiles invert the integral CDF of a piecewise-linear sampled density. Atomic
quantiles invert cumulative mass without interpolating across energy gaps.

Final C is freshly refitted with kernel widths 1, 2, 3 V, knots 0.25 and 0.05 eV,
and an expanded 8–17 eV domain, using the selected primary settings. These are
sensitivity fits, not extra selection on the held-out curves or the 10 V stress.
Reported quantities include peak status, mode, half-height interval/mass, mean,
median, standard deviation, quantiles, and response mass below/in the Ar I 4s group.
Masses are effective model contributions, not atom populations or cross sections.

## Resampling and recovery

Thirty centered circular residual block resamples (length 9) refit C with three
starts each and frozen real-data settings. Their intervals are conditional on the
kernel, domain, smoothing choice and residual model. Condition holdouts and
sensitivity fits report other sources of variation separately.

Four synthetic truths each have five block-noise realizations: single energy
11.65 eV; Gaussian mean 11.65 eV and sigma 0.15 eV; broad Gaussian sigma 0.8 eV;
and asymmetric mixture 0.7 N(11.5,0.25^2) + 0.3 N(12.3,0.55^2).
They use the final real-data kernel, neutral readout and training-derived residuals.
Every synthetic dataset uses four whole-curve holdouts and all three starts.
C also receives a four-curve recovery fit with the generating kernel/readout fixed.
The companion `run_spectral_matched_recovery.py` adds free-kernel four-curve fits
with identical synthetic observations. Only this matched comparison isolates the
kernel constraint; three-curve holdout recovery and four-curve fixed recovery
also differ in their training conditions. Mode, mean and width error are reported
separately; these are synthetic controls, not new observations or statistical power.
Sixteen additional noiseless, generating-kernel-fixed fits (four truths by four
curvature coefficients, start 0) diagnose regularization broadening without noise
or free-response compensation. They are mechanism checks, not model selection.

## Commands and preservation

```powershell
python run.py --experiment spectrum-revised --mode smoke --device cpu --output <temporary-directory>
python scripts/run_spectral_revision_pilot.py --output output/spectral_revision_pilot
python run.py --experiment spectrum-revised --mode fullscan --device cpu --output output/spectral_revision
python scripts/run_spectral_matched_recovery.py
python scripts/run_spectral_phase_comparison.py
python scripts/build_spectral_revision_evidence.py
python scripts/replay_spectral_revision.py
```

Four CPU processes each use one numerical-library thread. The pilot is an engineering
and matched-fit diagnostic on training data, not a predictive validation result.
Run/checkpoint identities include source bytes, input, split, initialization, grid,
domain, knots, regularization and fixed-parameter controls; incompatible reuse fails.
Historical `--experiment spectrum` and the original neural-network path remain available.
Fresh science outputs stay local. A separately hashed evidence package will contain
source-derived aggregate tables, real predictions and parameter receipts for replay.

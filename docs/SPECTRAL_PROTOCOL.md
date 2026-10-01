# Effective energy inversion: frozen protocol, release 1

This protocol implements the approved two-stage analysis of one archived argon
Franck–Hertz dataset. The inferred measure is an effective excitation response
distribution conditional on the phenomenological periodic response kernel. It is
neither an atomic continuum nor the collected-electron kinetic-energy spectrum.

## Data and decisions

Only Vr = 0, 4, 6, 8 V (644 points) enters training, initialization, selection,
residual noise estimation, and synthetic controls. The original 10 V observations
are retained unchanged for a final known-anomaly stress test. A possible drift in
emission or collection is a hypothesis, not an established measurement error.

Four outer whole-curve holdouts each contain three training curves. Three inner
whole-curve holdouts select the curvature penalty and local representation using
seed 0. Outer and final refits use optimization starts 0, 1, 2. These starts are
computational replicates. All held-out predictions have gain 1, bias 0, and
voltage shift 0. Training gains and biases retain the existing bounded nuisance
parameterization; every voltage shift is fixed at zero.

## Stage 1

H1 is one effective energy; G1 is a truncated, normalized Gaussian with mean in
9–16 eV and standard deviation 0.05–2 eV; C is a normalized nonnegative cubic
B-spline distribution on 9–16 eV with 0.25 eV knots (31 basis functions).
Trapezoidal integration starts at 0.02 eV. Density coefficients use a simplex
over individually normalized basis functions. No energy anchor, separation prior,
or entropy score is used. Continuous coefficients start near the uniform density;
H1/G1 use peak spacing estimated only from their training curves.

The objective is the mean of per-curve range-normalized MSE, plus 1e-4 times
the sum of squared raw shared parameters (and 0.25 times squared raw gain/bias),
plus lambda times the integral of squared density curvature for C only.
Lambda candidates are 1e-6, 1e-4, 1e-2. Spectral parameters are excluded from the
shared penalty. AdamW uses lr=0.002 and weight_decay=0 (the explicit penalty is
the only common regularizer), gradient norm clipping 5, maximum 3500 epochs,
minimum 300, patience 100, relative improvement 2e-4, smoothing window 7.
The best unsmoothed training-objective state is saved. Plateau termination and
epoch-limit termination are recorded separately.

The one-standard-error rule uses the standard error across inner held-out curves
and prefers the largest eligible lambda. Reports include all candidates, not just
the selected model. Fixed kernel widths 1, 2, 3 V are refitted from fresh starts.

## Stage 2

Each training unit defines its window from the median continuous-density mode
across its starts, plus/minus 0.5 eV clipped to the energy domain. An interior
maximum is required; boundary modes are explicitly recorded as no eligible
interior main peak. For a single-start inner unit its own mode defines the window.
Across-start mode range is reported; comparisons remain conditional if unstable.

The energy measure outside the window (nodes and exact quadrature masses), the
total mass inside the window, and the first-stage kernel width are frozen for
each start. Only the inside representation is replaced. All remaining common
and training nuisance parameters can refit from that start's stage-1 state.
Candidates: local delta, local Gaussian, free ordered K=2/3/4 with numerical
minimum gap 0.02 eV, NIST 4s spacings, and equal-spacing four points with the
same endpoint span. The last two share four spectral degrees of freedom
(one shift, three independent weights) and identical shift/window constraints.
The NIST group is (11.54835442, 11.62359272, 11.72316039, 11.82807116) eV;
its common shift is limited to +/-0.25 eV. An infeasible group is recorded as
outside the current main-peak candidate set. It is never projected into the window.
The local one-SE rule prefers fewer spectral parameters, retains ties, and treats
NIST and equal-spacing candidates symmetrically.

Window integrals are split at both exact endpoints, with trapezoidal half weights
on each segment; the boundary is not counted as an entire inside grid cell.
Fine-grid checks keep the local inside mass frozen and renormalize the outside
quadrature to its frozen mass. All real, profile, bootstrap, and recovery units
must pass the 1e-4 uA prediction tolerance. An interrupted initial engineering
attempt exposed and corrected the endpoint-cell error; its partial receipts stay
local and are excluded from the formal release.

## Conditional resampling and recovery controls

Thirty circular moving-block residual resamples (block length 9) are performed
per curve, using the median-start final continuous fit and centered per-curve
residuals. Each resample refits C and every eligible local candidate with seed 0,
using the frozen real-data hyperparameters. Candidate counts here use training
objective ranking only and are labeled conditional fitting stability, not held-out
support or a new model-selection validation. Distribution quantiles are conditional
on this fitted kernel and residual model.

Four synthetic truth families have five noise realizations each: delta at
11.65 eV; Gaussian mean 11.65 eV, sigma 0.15 eV; NIST four energies; and
equal-spacing four energies with the same endpoints. Both four-point truths use
weights (0.4, 0.3, 0.2, 0.1). Shared response parameters come from the final
continuous fit; synthetic predictions use neutral nuisance corrections. Noise is
the same per-curve residual-block mechanism. Seeds are fixed at 20000–20019.
Each synthetic dataset runs four whole-curve holdouts and the same two stages,
with the real selected hyperparameters frozen. These are algorithmic recovery
controls, not new measurements or a claim of sufficient statistical power.

Evidence for internal splitting requires held-out improvement over local delta
and local Gaussian and absence of the same false splitting in those synthetic
controls. NIST specificity additionally requires stable superiority to the equal-
spacing template. A compatible candidate is reported when specificity fails.

## Numerical and operational checks

Quadrature predictions at 0.02 versus 0.01 eV must differ by less than 1e-4 uA.
If not, refine to 0.01 and then 0.005 eV and record the final grid. Formal output
includes fit checkpoints locally, densities/masses, point predictions, inner and
outer scores, peak/trough positions and missing/extra counts, parameter states,
windows, convergence states, bootstrap and recovery tables. Input, training split,
model, grid, regularization, seed, and source-byte hashes identify every unit;
incompatible checkpoints are refused. Four CPU workers each use one numerical
library thread. Smoke is an engineering check with shortened epochs/replicate
counts and is never used as a scientific result.

Existing 805-point research evidence, A1–A15 (including A4 not executed), source
history, and the raw spreadsheet remain unchanged. Formal PDFs and full figures
stay local. A separate minimal spectral evidence package receives its own byte
hashes and source mapping.

## Interpretation emphasis, October 2, 2026

The frozen computational protocol and results above are retained. The current article first asks where effective energy response concentrates and whether its connected main half-height region overlaps the first Ar I 4s range. Continuous inversion is not expected to produce four separated states. Stage-2 discrete tests are completed supplementary exploration, not a prerequisite for describing concentration. `concentration.csv` uses normalized sampled densities, interpolated half-height crossings around the global mode, and exact-endpoint trapezoidal interval masses; these are response contributions, not atomic populations.

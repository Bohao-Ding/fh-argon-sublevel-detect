# v2k H1 experiment plan

## Question

Can an energy-conserving collector selection gate remove the artificial first
troughs at high retarding voltage while preserving the v2j improvements in
oscillation alignment and late-voltage fit?

## Frozen change

- Keep the v2j collision-history ensemble, state space, supply law, angular
  aperture, loss energies, training loss, and early-stop policy unchanged.
- Replace only the ordinary Gaussian collector CDF by
  `erf(max(E_axial - E_barrier, 0) / (sqrt(2) * sigma_E))`.
- Add no trainable parameters; initialize compatible tensors from the formal
  v2j H1 checkpoint.

## Pass rule

The H1 model must pass all 15 pre-registered v2j single-level checks, including
`trough_precision = 1.0`, `max_cutoff_error_V <= 1.5`, and
`early_high_retarding_mae_uA <= 0.075`.  H4s is not started unless H1 passes.

## Diagnostics

Inspect all per-curve cutoffs and peak/trough lists, the 15-check gate, fitted
parameter bounds, the dynamic early-stop record, and the rendered fit.  A lower
aggregate error cannot compensate for a failed morphology check.

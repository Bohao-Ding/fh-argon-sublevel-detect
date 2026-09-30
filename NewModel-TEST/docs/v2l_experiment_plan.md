# v2l H1 experiment plan

## Decision basis

The v2k subthreshold-truncated gate failed 4 of 15 frozen H1 checks and retained
the artificial first trough.  A read-only scan of the v2j collision-field
dispersion instead found 21 feasible points among 63 local candidates.  The
lowest-NRMSE feasible point was `sigma0 = 10.5 eV`, `decay = 32 V`.

## Frozen change

- Return to the v2j Gaussian collector energy gate.
- Fix only the collision-history dispersion at `10.5 eV * exp(-Va / 32 V)`.
- Treat this pair as an apparatus closure shared by H1 and H4s, not as universal
  argon constants or evidence for multiple levels.
- Initialize all compatible trainable tensors from the formal v2j H1 checkpoint.
- Use no additional trainable parameters or morphology-specific loss term.

## Protocol amendment after local trajectory diagnosis

A no-output 200-epoch trajectory showed that the initialization passed 15/15,
whereas lower-loss epochs 25--200 reintroduced an unmatched first trough.  The
formal H1 result therefore uses a two-stage constrained fit: the source v2j
continuous fit must have stopped on a loss plateau, then the frozen 63-point
apparatus grid selects the lowest-NRMSE candidate among those satisfying all 15
constraints.  Gradient loss is not used to rank infeasible candidates above a
feasible morphology.

## Formal fit and pass rule

Inherit the remaining 19 parameters from the plateau-stopped v2j fit and perform
the deterministic coordinate calibration above.  H1 must pass all 15 already
frozen v2j checks.  Inspect the rendered curves and the per-curve feature lists
before starting H4s.  H4s is a secondary model-comparison fit only after H1
passes and retains dynamic plateau stopping for its continuous optimization.

## H4s numerical convergence amendment

The first H4s run with 8 collision counts had 495 states and was stopped before
producing artifacts.  H1 convergence checks established that 8 to 6 collisions
changed the maximum current by only `3.58e-7 uA`, whereas 6 to 5 changed it by
`0.662 uA`; therefore six is the smallest admissible collision cap.  Reducing
thermal/angular quadrature from six to four changed the H1 current by at most
`0.00211 uA` and retained 15/15 checks.  The fair H1/H4s comparison therefore
uses 32 layers, six collisions, and four quadrature points for both hypotheses.

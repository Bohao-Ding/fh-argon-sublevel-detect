# Calibration audit verification, 2026-10-02

- Independent checkout of the candidate commits: **129 passed, 1 skipped** in 19.65 s.
  The skipped test uses an external manuscript figure script outside the source
  repository. The initial clean-checkout test invocation omitted the temporary
  parent directory; the corrected invocation uses an existing `output` parent.
  The new schema label was also adjusted to the existing repository naming check.
- The same independent checkout reconstructs **39,123 prediction points**, seven
  numeric tables, all readout coefficients and the summary **exactly**. No local
  checkpoint, formal training directory or manuscript file is present in it.
- The new audit/reference packages verify 12/14 hashed files respectively (each
  checksum list excludes itself). They contain 13/15 released files. The older
  251-file spectral checksum manifest and the five frozen kernel source hashes
  are verified by the audit; old data and reference bytes are unchanged.
- Known-answer/isolated-data tests cover shared affine recovery, unchanged extrema,
  the voltage-period equivalence, exclusion of heldout and 10 V observations,
  one-SE retention/rejection, and distinction from the exploratory Vr trend.
- All four outer C inner decisions and the final decision retain identity. The
  fixed-kernel reanalysis is not a new joint kernel/calibration model search and
  does not establish instrument error or change the inferred density.
- Local multi-file RevTeX builds succeed: main 7 pages, supplement 16 pages,
  no overfull boxes or undefined references/citations. Poppler-rendered changed
  pages and neighboring transitions were inspected. Existing RevTeX float-placement
  warnings remain in historical material; the new table and referenced historical
  tables/figures are present. PDF author metadata is blank.

Tests and disposable independent-checkout/replay/render directories are removed
after verification. Local scientific fits, derived calibration outputs, manuscript
backup and formal PDFs remain. Existing user source-package/figure changes are
excluded from the new commits.

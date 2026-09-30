# Integration verification — 2026-10-01

Verification used a Git-tree export containing only intended publication files, with no external work directories, formal training outputs or local manuscript assets. Python 3.13.7, NumPy 2.3.3, pandas 2.3.2, SciPy 1.16.2 and PyTorch 2.8.0 were used; numerical-library threads were limited to one per process.

| Suite | Result |
|---|---|
| Neural pipeline | 87 passed, 1 skipped |
| Collision-history transport | 44 passed |
| Foundational analysis | 18 passed |
| Diagnostic program | 13 passed |

The single skip concerns an external legacy manuscript-figure script. Numerical research checks ran. Writable parent directories must exist before pytest's `--basetemp` is used; the reproduction guide includes this setup. The evidence verifier's two tests were rerun after adding the selector recalculation receipt.

The documented CPU smoke command completed all four validation stages (`selector`, `bootstrap`, `holdout`, `h4s`) with engineering smoke status. This is an execution check, not a new formal experiment. `EXPERIMENT_V2/run_all.py` completed all 14 registered modules. A7 read five H4s folds from packaged files and the complete v2l receipt (`source_available=true`, `fold_source=fresh_read`), reproducing 15/15 versus 12/15 morphology gates and 91.36% channel concentration. Missing reference files cause explicit failures; rounded narrative values are not substitutes.

Numeric columns in regenerated diagnostic CSVs were compared to the frozen references: no absolute difference exceeded 1e-10. A15 reproduced a maximum prediction difference of 1.0658141036401503e-14 against its 1e-10 tolerance. File serialization, timings and environment records can differ across library versions; the frozen receipts were not rewritten.

The additive evidence verifier checks 159 actual file hashes, three identical measured inputs, the complete 60-unit H4s matrix, free-K fitting RMSE, baseline K=4 and complexity-group removal K=8, H4s averages, v2l gates, bootstrap counts and A15 identity. The 15 selector group-removal scenarios are deterministic recalculations on retained fitted candidates and differ from the original 23 weight perturbations.

Computational completion preserves scientific failures. A1's all-settings smoothing gate, both A2 gates, A3's representation gate, A6's frozen fine-spacing gate and A13's wide-span per-template gate remain false. A6 detects additional structure and does not assign its spacing; A14's nested test addresses existence. A4 remains NOT_EXECUTED, and apparatus metadata and independent measurements remain unavailable. Generated test, smoke and diagnostic products are temporary; published reference evidence remains frozen.

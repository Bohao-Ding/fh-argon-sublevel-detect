# Standalone research reproduction

Use Python 3.11 or later. From the repository root, install `requirements.txt`; companion requirements are already covered by its NumPy, pandas, SciPy, PyTorch, matplotlib, openpyxl and pytest dependencies. The foundational module uses NumPy ≥2.3, pandas ≥2.3 and SciPy ≥1.16. Preserved receipts record the original computing environments; newly serialized outputs can differ by library version even when reported results agree.

```powershell
python -m pip install -r requirements.txt
python scripts/verify_research_evidence.py
```

The verifier checks actual evidence-file bytes, the three identical input copies, the complete formal H4s matrix, key averages, v2l gates, residual-bootstrap counts and A15 identity. Historical `table_sha256` fields can hash a DataFrame serialization rather than the saved CSV; use the additive package's `SHA256SUMS.txt` for file integrity. Reindexing is only appropriate for an intentionally reviewed evidence update, not for accepting a mismatch.

Run each test suite in its own module directory. Keep temporary test directories writable and distinct when suites run in parallel:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
New-Item -ItemType Directory -Path .test-tmp -Force | Out-Null
python -B -m pytest -p no:cacheprovider tests -o addopts='' -q --basetemp .test-tmp/neural
Push-Location NewModel-TEST
python -B -m pytest -p no:cacheprovider tests -o addopts='' -q --basetemp ../.test-tmp/transport
Pop-Location
Push-Location FULL-REtry
python -B -m pytest -p no:cacheprovider tests -o addopts='' -q --basetemp ../.test-tmp/foundation
Pop-Location
Push-Location EXPERIMENT_V2
python -B -m pytest -p no:cacheprovider tests -o addopts='' -q --basetemp ../.test-tmp/diagnostics
Pop-Location
```

The existing manuscript-figure test deliberately skips when its external manuscript script is absent. This does not skip research numerical checks. Remove only these newly created test directories after inspecting their results; preserve pre-existing user files.

Run the lightweight diagnostic program from its directory:

```powershell
Push-Location EXPERIMENT_V2
python -B run_all.py
Pop-Location
```

It writes fresh CSV/JSON and an A7 figure under ignored `EXPERIMENT_V2/results/`. A7 reads its reference H4s and v2l evidence from `source_data_package/research_evidence`, not another checkout or ignored training directory. The frozen reference aggregates remain separate and are never overwritten by this command. Several scientific gates intentionally report FAIL; process success means computations completed, not that every scientific hypothesis passed.

For an engineering-only neural check, use an isolated output directory:

```powershell
python -B run.py --mode smoke --exclude hpopt --validation --device cpu --output .test-tmp/neural_smoke
```

Smoke products are not scientific evidence. Delete the new smoke output after checking completion. Full neural or transport training is intentionally separate from this integration task; see the existing module instructions when a formal new run is requested. v2m requires a complete verified four-channel cross-section table and provenance; A4 is NOT_EXECUTED and no substitute scientific table is supplied.

The measurement SHA-256 is `FF4B316B7D11C84B99E07B9AE0E32AFD03EEFD33698CF9DE3D02062D7F1902F3`. Figures can be reconstructed from numeric outputs; local manuscript/PDF packages and historical training products are not required by the published diagnostic pipeline.

# Franck–Hertz Supplementary Validation Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible, failure-transparent validation layer implementing the frozen selector audit, block bootstrap, held-out prediction, synthetic recovery, and matched-budget MLP experiments without changing production results or manuscript artifacts.

**Architecture:** Keep `run.py` as the only executable entry point and extend the existing flat `sublevel_detect` pipeline style. `validation_pipeline.py` owns orchestration, baseline identity, manifests, resumability, and summaries; five focused stage modules own scientific calculations. Shared statistical and artifact helpers live in `validation_common.py`, while training continues to reuse the frozen `model.Config`, `model.run_level_scan`, checkpoint schema, and production selector.

**Tech Stack:** Python 3.13, PyTorch 2.8, NumPy, pandas, SciPy, pytest, JSON/CSV artifacts, SHA-256 manifests.

---

## File map

- Modify `src/sublevel_detect/cli.py`: expose `--validation` and `--validation-only`, preserve legacy routing, propagate validation failures as a non-zero exit.
- Modify `src/sublevel_detect/paths.py`: add only the validation output path helper.
- Create `src/sublevel_detect/validation_common.py`: canonical JSON hashing, baseline resolution, status/progress records, Wilson intervals, metrics, artifact writers, and immutable run identity.
- Create `src/sublevel_detect/validation_pipeline.py`: stage dependency routing, manifest construction, resume checks, stage execution, and integrated summary.
- Create `src/sublevel_detect/validation_selector.py`: frozen 15-scenario selector audit and rank correlations.
- Create `src/sublevel_detect/validation_bootstrap.py`: centered circular moving-block residual resampling and replicate runner.
- Create `src/sublevel_detect/validation_holdout.py`: leakage-safe leave-one-`Vr` training, checkpoint loading, neutral prediction, and three-parameter sparse nuisance calibration.
- Create `src/sublevel_detect/validation_synthetic.py`: truth matrices, fixed-kernel data generation, epoch gate, selection accounting, and assignment metrics.
- Create `src/sublevel_detect/validation_benchmark.py`: exact 45-parameter MLP and fold-matched training/evaluation.
- Create `tests/test_validation_cli.py`: CLI, routing, dependency, return-code, and legacy-compatibility contracts.
- Create `tests/test_validation_common.py`: baseline identity, hashing, resume refusal, Wilson, metric, and failure-denominator tests.
- Create `tests/test_validation_selector.py`: 15 scenarios, immutable weights, competition ranks, ties, and correlation schema.
- Create `tests/test_validation_bootstrap.py`: determinism, curve isolation, centered residuals, circular blocks, scale, and row preservation.
- Create `tests/test_validation_holdout.py`: fold isolation, checkpoint selection, calibration split, nuisance-only optimization, and NRMSE denominator.
- Create `tests/test_validation_synthetic.py`: truth construction, matrix counts, epoch gate, block noise, assignment, Wilson recovery, and failed-fit denominator.
- Create `tests/test_validation_benchmark.py`: parameter count, input contract, train-only standardization/early stop, and fold pairing.
- Modify `tests/test_smoke_pipeline.py`: integrated validation smoke output contract and cleanup assertion.
- Append only `docs/supplementary_experiments.md`: stage status and evidence after each verified stage.

## Task 1: CLI and validation orchestration contract

**Files:**
- Modify: `src/sublevel_detect/cli.py`
- Modify: `src/sublevel_detect/paths.py`
- Create: `src/sublevel_detect/validation_pipeline.py`
- Test: `tests/test_validation_cli.py`

- [ ] **Step 1: Write failing parser and routing tests**

```python
def test_validation_defaults_off_and_only_enables_validation() -> None:
    default = cli.build_parser().parse_args([])
    staged = cli.build_parser().parse_args(["--validation-only", "selector"])
    assert default.validation is False
    assert default.validation_only is None
    assert staged.validation is False
    assert cli.validation_enabled(staged) is True

def test_validation_only_routes_one_stage_without_legacy_optional_stages(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(cli.validation_pipeline, "run", lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(cli.main_pipeline, "run", lambda **kw: {"sweep_dir": "unused"})
    assert cli.main(["--mode", "smoke", "--validation-only", "selector"]) == 0
    assert calls[0]["requested_stage"] == "selector"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_validation_cli.py -q`

Expected: collection or assertion failure because validation arguments and module do not exist.

- [ ] **Step 3: Implement the minimal public routing API**

```python
VALIDATION_STAGES = ("selector", "bootstrap", "holdout", "synthetic", "benchmark")

def validation_enabled(args: argparse.Namespace) -> bool:
    return bool(args.validation or args.validation_only is not None)

def run(*, mode: str, input_path: str | Path, output_root: str | Path,
        device: str, requested_stage: str | None) -> dict[str, Any]:
    stages = [requested_stage] if requested_stage else list(VALIDATION_STAGES)
    return {"ok": True, "requested_stages": stages}
```

Add parser arguments with the frozen choices, invoke validation without requiring a new production baseline run, and return `1` when its result has `ok=False`. Add `paths.validation_dir(output_root)` returning `<output_root>/validation`.

- [ ] **Step 4: Verify GREEN and legacy compatibility**

Run: `python -m pytest tests/test_validation_cli.py tests/test_cli_contract.py -q`

Expected: all tests pass; existing no-validation calls still route exactly as before.

- [ ] **Step 5: Commit**

```powershell
git add src/sublevel_detect/cli.py src/sublevel_detect/paths.py src/sublevel_detect/validation_pipeline.py tests/test_validation_cli.py
git commit -m "feat: add validation CLI routing"
```

## Task 2: Baseline resolver, immutable run identity, and failure-transparent state

**Files:**
- Create: `src/sublevel_detect/validation_common.py`
- Modify: `src/sublevel_detect/validation_pipeline.py`
- Test: `tests/test_validation_common.py`

- [ ] **Step 1: Write failing baseline priority and same-root tests**

```python
def test_resolver_prefers_complete_same_output_baseline(tmp_path: Path) -> None:
    local = write_complete_baseline(tmp_path / "main")
    package = write_complete_package(tmp_path / "source_data_package")
    baseline = resolve_baseline(tmp_path, package_root=package)
    assert baseline.kind == "local"
    assert {path.parents[1] for path in baseline.files.values()} == {local}

def test_incomplete_local_baseline_falls_back_as_a_whole(tmp_path: Path) -> None:
    write_incomplete_baseline(tmp_path / "main")
    package = write_complete_package(tmp_path / "source_data_package")
    baseline = resolve_baseline(tmp_path, package_root=package)
    assert baseline.kind == "package"
    assert all(str(path).startswith(str(package)) for path in baseline.files.values())
```

Also test required-file absence, uppercase SHA-256 values, canonical configuration hash, existing manifest mismatch refusal, allowed status vocabulary, atomic progress writes, and failures retained in the total denominator.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_validation_common.py -q`

Expected: import failure for `validation_common`.

- [ ] **Step 3: Implement focused immutable types and helpers**

```python
@dataclass(frozen=True)
class Baseline:
    kind: str
    root: Path
    files: dict[str, Path]
    hashes: dict[str, str]

@dataclass(frozen=True)
class RunIdentity:
    code_commit: str
    code_dirty: bool
    baseline_hash: str
    matrix_hash: str
    seed_hash: str

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()

def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()

def write_stage_status(path: Path, status: str, **details: Any) -> None:
    if status not in ALLOWED_STAGE_STATUSES:
        raise ValueError(f"Unsupported stage status: {status}")
    atomic_json_dump({"status": status, **details}, path)

def write_progress(path: Path, completed: int, failed: int,
                   total: int, **details: Any) -> None:
    if completed + failed > total:
        raise ValueError("completed + failed exceeds preregistered total")
    atomic_json_dump({"completed": completed, "failed": failed,
                      "total": total, **details}, path)
```

The packaged root must map the frozen main scan artifacts plus `run_records/k_selected_full/{checkpoint_best.pt,scorecard.json,prediction_points.csv}` and `manuscript_source_tables/channel_parameters.csv`. No resolver branch may mix roots.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_validation_common.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/sublevel_detect/validation_common.py src/sublevel_detect/validation_pipeline.py tests/test_validation_common.py
git commit -m "feat: add immutable validation manifests"
```

## Task 3: Selector decontamination audit

**Files:**
- Create: `src/sublevel_detect/validation_selector.py`
- Modify: `src/sublevel_detect/validation_pipeline.py`
- Test: `tests/test_validation_selector.py`

- [ ] **Step 1: Write failing scenario, ranking, and immutability tests**

```python
def test_selector_scenarios_are_exactly_preregistered() -> None:
    assert [s.name for s in selector_scenarios()] == [
        "production", "leave_one_rmse_out", "leave_one_legacy_summary_out",
        "leave_one_d1_out", "leave_one_d2_out", "leave_one_structure_out",
        "leave_one_physical_out", "leave_one_bic_out", "leave_one_aic_out",
        "leave_one_degeneracy_out", "remove_fit_group", "remove_shape_group",
        "remove_physical_group", "remove_complexity_group", "fit_complexity_only",
    ]

def test_scenarios_do_not_mutate_or_renormalize_production_weights() -> None:
    original = dict(model.DEFAULT_SELECTOR_RANK_WEIGHTS)
    scenarios = selector_scenarios()
    assert model.DEFAULT_SELECTOR_RANK_WEIGHTS == original
    assert scenarios[1].weights["rmse"] == 0.0
    assert scenarios[1].weights["structure"] == 1.25
```

Add tests for `1e-12` competition-rank ties, lower-K composite ties, 9-by-9 symmetric Spearman output, 15-row decision table, and selected-K frequency totals.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_validation_selector.py -q`

Expected: import failure for `validation_selector`.

- [ ] **Step 3: Implement the selector audit without retraining**

```python
@dataclass(frozen=True)
class SelectorScenario:
    name: str
    weights: dict[str, float]

SCENARIO_NAMES = (
    "production", "leave_one_rmse_out", "leave_one_legacy_summary_out",
    "leave_one_d1_out", "leave_one_d2_out", "leave_one_structure_out",
    "leave_one_physical_out", "leave_one_bic_out", "leave_one_aic_out",
    "leave_one_degeneracy_out", "remove_fit_group", "remove_shape_group",
    "remove_physical_group", "remove_complexity_group", "fit_complexity_only",
)
```

Read only the frozen model-selection table; call `model.select_k_neutral_level_decision` with copied weight dictionaries. Write `selector_scenarios.csv`, `rank_correlation.csv`, `selected_k_distribution.csv`, `selector_summary.json`, and `stage_status.json`.

- [ ] **Step 4: Verify GREEN, stage smoke, and formal package audit**

Run: `python -m pytest tests/test_validation_selector.py tests/test_validation_common.py -q`

Run: `python run.py --mode smoke --exclude hpopt --validation-only selector --output C:\tmp\fh_selector_smoke --device cpu`

Expected: exit 0 and all selector artifacts present. Inspect, then remove `C:\tmp\fh_selector_smoke`.

Run: `python run.py --mode fullscan --validation-only selector --device cuda`

Expected: no training is launched; formal audit completes from a single resolved baseline.

- [ ] **Step 5: Append evidence, verify, and commit**

Run: `python -m pytest -q` and `git diff --check`.

Append only the stage-2 status and evidence block in `docs/supplementary_experiments.md`, then commit:

```powershell
git add src/sublevel_detect/validation_selector.py src/sublevel_detect/validation_pipeline.py tests/test_validation_selector.py docs/supplementary_experiments.md
git commit -m "feat: audit selector decontamination"
```

## Task 4: Circular moving-block residual bootstrap

**Files:**
- Create: `src/sublevel_detect/validation_bootstrap.py`
- Modify: `src/sublevel_detect/validation_pipeline.py`
- Test: `tests/test_validation_bootstrap.py`

- [ ] **Step 1: Write failing resampling and matrix tests**

```python
def test_circular_blocks_preserve_contiguous_order_and_length() -> None:
    pool = np.arange(6, dtype=float)
    draw = circular_moving_block_sample(pool, size=10, block_length=4,
                                        rng=np.random.default_rng(3))
    assert len(draw) == 10
    for block in draw.reshape(-1, 4)[:-1]:
        assert np.all(np.diff(block) % len(pool) == 1)

def test_centered_noise_never_crosses_curve_boundaries() -> None:
    residuals = {1: np.array([10., 12., 14.]), 2: np.array([-9., -6., -3.])}
    sampled = sample_residuals_by_curve(residuals, {1: 12, 2: 12}, 3, seed=7)
    assert abs(sampled[1].mean()) < 4.0
    assert sampled[1].min() >= -2.0
    assert sampled[2].max() <= 3.0
```

Add exact preregistered unit-count tests, deterministic repeated-seed tests, row/order preservation, scale multiplication, Wilson interval, seed sensitivity, and failed fits remaining in denominators.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_validation_bootstrap.py -q`

Expected: import failure for `validation_bootstrap`.

- [ ] **Step 3: Implement resampling and the replicate executor**

```python
@dataclass(frozen=True)
class BootstrapUnit:
    block_length: int
    replicate: int
    train_seed: int

def circular_moving_block_sample(pool: np.ndarray, size: int,
                                 block_length: int,
                                 rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, len(pool), size=math.ceil(size / block_length))
    blocks = [pool[(start + np.arange(block_length)) % len(pool)] for start in starts]
    return np.concatenate(blocks)[:size]
```

Center each curve pool before drawing; generate inputs atomically; reuse production hyperparameters with hpopt disabled; record every unit before training; continue after fit failures; summarize block-7 K=4 Wilson 95% and block/seed sensitivity separately from legacy pointwise bootstrap.

- [ ] **Step 4: Verify GREEN and smoke**

Run: `python -m pytest tests/test_validation_bootstrap.py -q`

Run: `python run.py --mode smoke --exclude hpopt --validation-only bootstrap --output C:\tmp\fh_bootstrap_smoke --device cpu`

Expected: smoke matrix completes with manifests and summaries. Inspect and remove the temporary directory.

- [ ] **Step 5: Run formal stage, append evidence, and commit**

Run: `python run.py --mode fullscan --validation-only bootstrap --device cuda`

Run: `python -m pytest -q` and `git diff --check`.

Append the stage-3 evidence block and commit all stage files.

## Task 5: Leakage-safe leave-one-`Vr`-out prediction

**Files:**
- Create: `src/sublevel_detect/validation_holdout.py`
- Modify: `src/sublevel_detect/validation_pipeline.py`
- Test: `tests/test_validation_holdout.py`

- [ ] **Step 1: Write failing fold, split, and nuisance tests**

```python
def test_fold_training_excludes_heldout_vr() -> None:
    curves = fake_curves([0., 4., 6., 8., 10.])
    train, heldout = split_vr_fold(curves, heldout_vr=6.0)
    assert {float(c["Vr"]) for c in train} == {0., 4., 8., 10.}
    assert {float(c["Vr"]) for c in heldout} == {6.}

def test_calibration_indices_are_fixed_and_disjoint() -> None:
    calibration, evaluation = calibration_split(161)
    assert calibration.tolist() == list(range(0, 161, 8))
    assert len(calibration) == 21 and len(evaluation) == 140
    assert not set(calibration).intersection(evaluation)
```

Add tests that training-fold selection consumes only training scorecards, K=1/4/8 plus selected-K are retained, neutral prediction never indexes held-out nuisance, calibration changes exactly `gain/bias/delta_va`, the physical state dict remains bitwise unchanged, and both metric modes use the full 161-point observed range.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_validation_holdout.py -q`

Expected: import failure for `validation_holdout`.

- [ ] **Step 3: Implement fold training and evaluation APIs**

```python
@dataclass(frozen=True)
class HoldoutUnit:
    heldout_vr: float
    n_levels: int
    seed: int

def calibration_split(n_points: int = 161) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.arange(0, n_points, 8, dtype=int)
    evaluation = np.setdiff1d(np.arange(n_points, dtype=int), calibration)
    return calibration, evaluation

def prediction_metrics(observed: np.ndarray, predicted: np.ndarray,
                       denominator_observed: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    rmse = float(np.sqrt(np.mean(np.square(error))))
    scale = float(np.max(denominator_observed) - np.min(denominator_observed))
    if scale <= 0.0:
        raise ValueError("Held-out observed range must be positive")
    return {"rmse": rmse, "mae": float(np.mean(np.abs(error))),
            "nrmse": rmse / scale}
```

Construct each scan from the packaged production config with `exclude_vr_values`, `hyperopt_enabled=False`, K=1..8, and seeds 0/1/2 in formal mode. Evaluate only after the training-fold decision is finalized. Write fold configs, per-point predictions, zero-shot/calibrated metrics, and fold summaries.

- [ ] **Step 4: Verify GREEN and smoke**

Run: `python -m pytest tests/test_validation_holdout.py -q`

Run the dedicated smoke command to a temporary root, verify its fold schema, and remove the root.

- [ ] **Step 5: Run formal stage, append evidence, and commit**

Run: `python run.py --mode fullscan --validation-only holdout --device cuda`

Then run the full test suite, append stage-4 evidence, and commit.

## Task 6: Semi-synthetic recovery and detectability

**Files:**
- Create: `src/sublevel_detect/validation_synthetic.py`
- Modify: `src/sublevel_detect/validation_pipeline.py`
- Test: `tests/test_validation_synthetic.py`

- [ ] **Step 1: Write failing truth, matrix, gate, and matching tests**

```python
def test_k4_delta_changes_only_second_energy() -> None:
    truth = truth_channels(4, delta=0.5, channel_table=fake_k8_table())
    assert truth.energies == pytest.approx([11.5, 12.0, 12.594, 13.965])
    assert sum(truth.weights) == pytest.approx(1.0)

def test_epoch_gate_requires_all_k_under_one_percent_and_same_selection() -> None:
    good = gate_rows(max_relative_rmse=0.009, selected=(4, 4))
    bad = gate_rows(max_relative_rmse=0.011, selected=(4, 4))
    changed = gate_rows(max_relative_rmse=0.001, selected=(4, 5))
    assert choose_epochs(good) == 1600
    assert choose_epochs(bad) == 3500
    assert choose_epochs(changed) == 3500
```

Add exact screening/confirmation/prior-off matrix-count tests, K=1/2/4/8 truth tests, fixed-kernel synthetic predictions, block-noise scale, assignment energy RMSE and weight MAE, near-pair recovery, Wilson lower-bound reliability, and failed-fit denominator tests.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_validation_synthetic.py -q`

Expected: import failure for `validation_synthetic`.

- [ ] **Step 3: Implement truth generation and recovery accounting**

```python
@dataclass(frozen=True)
class ChannelTruth:
    n_levels: int
    energies: tuple
    weights: tuple

@dataclass(frozen=True)
class SyntheticUnit:
    truth_k: int
    delta: float | None
    noise_scale: float
    replicate: int
    train_seed: int
    prior_enabled: bool

def choose_epochs(rows: Sequence[dict[str, Any]]) -> int:
    comparisons = [row for row in rows if int(row["epochs"]) == 1600]
    same_selection = all(int(row["selected_k"]) == int(row["selected_k_3500"])
                         for row in comparisons)
    rmse_close = all(abs(float(row["rmse"]) - float(row["rmse_3500"]))
                     / max(abs(float(row["rmse_3500"])), 1e-12) < 0.01
                     for row in comparisons)
    return 1600 if comparisons and same_selection and rmse_close else 3500
```

Load the K=4 production checkpoint once per device, freeze shared physics and nuisance, replace only channel energies/weights for generation, and add centered block-7 residual noise. Hash every generated dataset. Run the epoch gate before constructing the formal fit configs. Keep all fit failures in exact-K denominators.

- [ ] **Step 4: Verify GREEN and smoke**

Run: `python -m pytest tests/test_validation_synthetic.py -q`

Run the dedicated smoke command, inspect generated-data hashes and summaries, then remove the temporary root.

- [ ] **Step 5: Run formal stage, append evidence, and commit**

Run: `python run.py --mode fullscan --validation-only synthetic --device cuda`

Then run the full suite, append stage-5 evidence, and commit.

## Task 7: Fixed 45-parameter MLP benchmark

**Files:**
- Create: `src/sublevel_detect/validation_benchmark.py`
- Modify: `src/sublevel_detect/validation_pipeline.py`
- Test: `tests/test_validation_benchmark.py`

- [ ] **Step 1: Write failing architecture and leakage tests**

```python
def test_budget_mlp_has_exactly_45_parameters_and_two_inputs() -> None:
    network = BudgetMLP()
    assert sum(p.numel() for p in network.parameters()) == 45
    assert tuple(network(torch.zeros(3, 2)).shape) == (3, 1)

def test_standardizer_uses_training_fold_only() -> None:
    train = np.array([[0., 0.], [2., 4.]])
    test = np.array([[100., 100.]])
    scaler = fit_standardizer(train)
    assert scaler.mean == pytest.approx([1., 2.])
    assert transform(test, scaler)[0, 0] > 90
```

Add tests that inputs are exactly Va/Vr, Softplus output is non-negative, fold/seed matrix matches holdout, early stopping consumes training loss only, test curves never enter optimization, and paired-comparison keys match physical zero-shot rows.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_validation_benchmark.py -q`

Expected: import failure for `validation_benchmark`.

- [ ] **Step 3: Implement the fixed benchmark**

```python
class BudgetMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2, 11), torch.nn.Tanh(),
            torch.nn.Linear(11, 1), torch.nn.Softplus(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

BENCHMARK_OPTIMIZER = {
    "name": "AdamW", "learning_rate": 0.0025, "weight_decay": 1e-4,
    "max_epochs": 3500, "min_epochs": 300, "warmup": 300, "patience": 45,
}
```

Use AdamW with the frozen learning rate and weight decay, maximum epochs, warmup/minimum epochs/patience, and training-loss-only stopping. Record parameter count, wall time, requested/selected device, zero-shot RMSE/MAE/NRMSE, and paired physical-vs-MLP differences.

- [ ] **Step 4: Verify GREEN and smoke**

Run: `python -m pytest tests/test_validation_benchmark.py -q`

Run the dedicated smoke command, inspect the 45-parameter assertion and fold pairing, then remove the temporary root.

- [ ] **Step 5: Run formal stage, append evidence, and commit**

Run: `python run.py --mode fullscan --validation-only benchmark --device cuda`

Then run the full suite, append stage-6 evidence, and commit.

## Task 8: Integrated resume, summaries, and final verification

**Files:**
- Modify: `src/sublevel_detect/validation_pipeline.py`
- Modify: `tests/test_smoke_pipeline.py`
- Modify: `docs/supplementary_experiments.md`

- [ ] **Step 1: Write failing integrated output and resume tests**

```python
def test_integrated_validation_writes_complete_schema_and_reuses_matching_stages(tmp_path: Path) -> None:
    first = run_smoke_validation(tmp_path)
    second = run_smoke_validation(tmp_path)
    assert first["ok"] and second["ok"]
    assert second["reused_stages"] == ["selector", "bootstrap", "holdout", "synthetic", "benchmark"]
    for name in ("experiment_manifest.json", "progress.json",
                 "validation_summary.json", "validation_summary.md"):
        assert (tmp_path / "validation" / name).exists()
```

Add mismatch-refusal, incomplete-stage non-zero return, scenario/seed counts, failed denominator, stage-status vocabulary, and two distinct temporary-root cleanup tests.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_smoke_pipeline.py tests/test_validation_cli.py -q`

Expected: integrated schema or reuse assertion fails.

- [ ] **Step 3: Implement final orchestration and evidence summary**

```python
STAGE_DEPENDENCIES = {
    "selector": (),
    "bootstrap": (),
    "holdout": (),
    "synthetic": (),
    "benchmark": ("holdout",),
}

def requested_stages(stage: str | None) -> list[str]:
    if stage is None:
        return list(VALIDATION_STAGES)
    required = list(STAGE_DEPENDENCIES[stage])
    return [*required, stage]
```

Write the manifest before any stage, update progress after every fit, reuse only identity-matching complete stages, preserve failed rows, and write `validation_summary.json/.md` using evidence–inference–limitations language.

- [ ] **Step 4: Run focused and full verification**

Run: `python -m pytest -q`

Expected: all tests pass.

Run: `python -m compileall -q run.py src tests`

Expected: exit 0.

Run twice with different roots:

```powershell
python run.py --mode smoke --exclude hpopt --validation --output C:\tmp\fh_validation_smoke_1 --device cpu
python run.py --mode smoke --exclude hpopt --validation --output C:\tmp\fh_validation_smoke_2 --device cpu
```

Verify both schemas and hashes, then remove both temporary directories.

- [ ] **Step 5: Run the integrated formal command**

Run: `python run.py --mode fullscan --validation --device cuda`

Expected: completed stages are identity-checked and reused; missing stages run; any failed preregistered unit yields a retained failure row, `incomplete` status, and non-zero CLI exit.

- [ ] **Step 6: Audit scope and append final ledger evidence**

Run:

```powershell
git diff 393a521 --name-only
git status --short
python -m pytest -q
python -m compileall -q run.py src tests
```

Confirm no paths under manuscript, supplement, figures, or `source_data_package/` changed; no `C:\tmp\fh_validation_*` directories remain; every manifest matrix/seed count matches the preregistration. Append stage-7 and final evidence–inference–limitations records.

- [ ] **Step 7: Commit final integration**

```powershell
git add src/sublevel_detect/validation_pipeline.py tests/test_smoke_pipeline.py docs/supplementary_experiments.md
git commit -m "test: verify integrated supplementary validation"
```

## Self-review checklist

- Every frozen CLI, stage, matrix, seed, endpoint, threshold, status, and output family maps to an explicit task.
- The plan never modifies production selector defaults, legacy `main/robustness/sensitivity` outputs, manuscript text, supplement, figures, or `source_data_package/`.
- Each production-code step is preceded by a named failing test and an explicit RED command.
- Formal training work is resumable only under a matching immutable identity; failed units stay in denominators.
- Smoke outputs use temporary roots and are deleted after assertions.
- Benchmark conclusions remain restricted to the specified 45-parameter MLP.

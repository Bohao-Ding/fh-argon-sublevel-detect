from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sublevel_detect import validation_bootstrap
from sublevel_detect.validation_common import Baseline, sha256_file
from sublevel_detect.validation_bootstrap import (
    BootstrapUnit,
    bootstrap_units,
    centered_residual_pools,
    circular_moving_block_sample,
    summarize_bootstrap,
    write_bootstrap_input,
)


def test_circular_blocks_preserve_contiguous_order_and_length() -> None:
    pool = np.arange(6, dtype=float)
    first = circular_moving_block_sample(pool, size=12, block_length=4, rng=np.random.default_rng(3))
    second = circular_moving_block_sample(pool, size=12, block_length=4, rng=np.random.default_rng(3))

    assert np.array_equal(first, second)
    assert len(first) == 12
    for block in first.reshape(-1, 4):
        assert np.all(np.diff(block) % len(pool) == 1)


def test_centered_residual_pools_are_curve_isolated_and_zero_mean() -> None:
    points = pd.DataFrame(
        {
            "curve_id": [1, 1, 1, 2, 2, 2],
            "residual": [10.0, 12.0, 14.0, -9.0, -6.0, -3.0],
        }
    )

    pools = centered_residual_pools(points)

    assert set(pools) == {1, 2}
    assert pools[1] == pytest.approx([-2.0, 0.0, 2.0])
    assert pools[2] == pytest.approx([-3.0, 0.0, 3.0])
    assert all(abs(float(values.mean())) < 1e-12 for values in pools.values())


def test_formal_bootstrap_matrix_matches_preregistration() -> None:
    units = bootstrap_units("fullscan")

    assert len(units) == 60
    assert sum(unit.block_length == 7 and unit.train_seed == 0 for unit in units) == 30
    assert sum(unit.block_length == 7 and unit.replicate < 5 and unit.train_seed in {1, 2} for unit in units) == 10
    assert sum(unit.block_length == 5 and unit.train_seed == 0 for unit in units) == 10
    assert sum(unit.block_length == 9 and unit.train_seed == 0 for unit in units) == 10
    assert bootstrap_units("smoke") == [BootstrapUnit(block_length=7, replicate=0, train_seed=0)]


def test_bootstrap_input_uses_prediction_plus_centered_block_noise_and_preserves_rows(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "curve_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "Vr": [0.0] * 4 + [4.0] * 4,
            "Va": [0.0, 0.5, 1.0, 1.5] * 2,
            "IuA": [100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0],
        }
    )
    points = source.rename(columns={"IuA": "observed"}).copy()
    points["predicted"] = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
    points["residual"] = [2.0, 4.0, 6.0, 8.0, -3.0, -1.0, 1.0, 3.0]
    output = tmp_path / "replicate.csv"

    metadata = write_bootstrap_input(
        source=source,
        prediction_points=points,
        unit=BootstrapUnit(block_length=3, replicate=2, train_seed=0),
        output_path=output,
        noise_scale=0.0,
    )
    generated = pd.read_csv(output)

    assert len(generated) == len(source)
    assert generated["curve_id"].tolist() == source["curve_id"].tolist()
    assert generated["Va"].tolist() == source["Va"].tolist()
    assert generated["IuA"].tolist() == pytest.approx(points["predicted"].tolist())
    assert metadata["row_count"] == len(source)
    assert metadata["curve_count"] == 2
    assert metadata["noise_scale"] == 0.0
    assert len(metadata["sha256"]) == 64


def test_bootstrap_summary_uses_only_primary_seed0_replicates_for_main_interval() -> None:
    rows = [
        {"block_length": 7, "replicate": replicate, "train_seed": 0, "status": "complete", "selected_k": 4}
        for replicate in range(8)
    ]
    rows.extend(
        [
            {"block_length": 7, "replicate": 8, "train_seed": 0, "status": "failed", "selected_k": None},
            {"block_length": 7, "replicate": 9, "train_seed": 0, "status": "complete", "selected_k": 8},
            {"block_length": 7, "replicate": 0, "train_seed": 1, "status": "complete", "selected_k": 4},
        ]
    )

    summary = summarize_bootstrap(rows, expected_primary_total=10)

    assert summary["primary"]["total"] == 10
    assert summary["primary"]["failed"] == 1
    assert summary["primary"]["k4_count"] == 8
    assert summary["primary"]["k4_fraction"] == pytest.approx(0.8)
    assert summary["primary"]["wilson_low"] < 0.8 < summary["primary"]["wilson_high"]


def test_smoke_runner_records_unit_config_and_outputs(monkeypatch, tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "curve_id": [1, 1, 1, 2, 2, 2],
            "Vr": [0.0, 0.0, 0.0, 4.0, 4.0, 4.0],
            "Va": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "IuA": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
        }
    )
    input_path = tmp_path / "input.csv"
    source.to_csv(input_path, index=False)
    points = source.rename(columns={"IuA": "observed"}).copy()
    points["predicted"] = [0.8, 2.1, 3.1, 1.9, 3.2, 3.8]
    points["residual"] = points["observed"] - points["predicted"]
    prediction_path = tmp_path / "prediction_points.csv"
    points.to_csv(prediction_path, index=False)
    config_path = tmp_path / "config_used.json"
    config_path.write_text("{}\n", encoding="utf-8")
    forward_path = tmp_path / "forward_evidence.json"
    forward_path.write_text("{}\n", encoding="utf-8")
    baseline = Baseline(
        kind="local",
        root=tmp_path,
        files={"config": config_path, "forward_evidence": forward_path, "prediction_points": prediction_path},
        hashes={
            "config": sha256_file(config_path),
            "forward_evidence": sha256_file(forward_path),
            "prediction_points": sha256_file(prediction_path),
        },
    )
    configs = []

    def fake_scan(cfg):
        configs.append(cfg)
        return {"decision": {"selected_k": 4}, "scan_dir": cfg.out_dir}

    monkeypatch.setattr(validation_bootstrap.model, "run_level_scan", fake_scan)
    output_dir = tmp_path / "validation" / "block_bootstrap"

    result = validation_bootstrap.run(
        mode="smoke",
        input_path=input_path,
        baseline=baseline,
        output_dir=output_dir,
        validation_root=output_dir.parent,
        device="cpu",
    )
    reused = validation_bootstrap.run(
        mode="smoke",
        input_path=input_path,
        baseline=baseline,
        output_dir=output_dir,
        validation_root=output_dir.parent,
        device="cpu",
    )

    assert result["ok"] is True
    assert reused["ok"] is True
    assert result["status"] == "smoke_passed"
    assert len(configs) == 1
    assert configs[0].hyperopt_enabled is False
    assert configs[0].scan_seeds == "0"
    assert configs[0].level_scan_min == 1
    assert configs[0].level_scan_max == 2
    assert configs[0].epochs == 2
    for name in (
        "replicate_manifest.csv",
        "selection_rows.csv",
        "block_length_summary.csv",
        "seed_sensitivity.csv",
        "block_bootstrap_summary.json",
        "stage_status.json",
    ):
        assert (output_dir / name).is_file()

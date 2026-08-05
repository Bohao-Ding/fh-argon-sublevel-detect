from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from sublevel_detect import validation_h4s
from sublevel_detect.validation_common import Baseline, sha256_file
from sublevel_detect.validation_h4s import (
    H4S_HYPOTHESES,
    NIST_AR_I_4S_ENERGIES_EV,
    H4sHypothesisModel,
    H4sUnit,
    build_paired_contrasts,
    h4s_units,
)


def _curves() -> list[dict]:
    return [
        {
            "curve_id": index + 1,
            "curve_idx": index,
            "Va": np.arange(161, dtype=np.float32) * 0.5,
            "Vr": float(vr),
            "Vr_is_vector": False,
            "Ip": np.linspace(0.0, 1.0 + index, 161, dtype=np.float32),
        }
        for index, vr in enumerate((0.0, 4.0, 6.0, 8.0, 10.0))
    ]


def test_h4s_unit_matrices_are_frozen() -> None:
    smoke = h4s_units("smoke")
    full = h4s_units("fullscan")

    assert smoke == [H4sUnit(0.0, name, 0) for name in H4S_HYPOTHESES]
    assert len(full) == 60
    assert {unit.heldout_vr for unit in full} == {0.0, 4.0, 6.0, 8.0, 10.0}
    assert {unit.hypothesis for unit in full} == set(H4S_HYPOTHESES)
    assert {unit.seed for unit in full} == {0, 1, 2}


def test_nist_levels_share_one_bounded_shift_and_normalized_weights() -> None:
    constrained = H4sHypothesisModel(
        hypothesis="h4s",
        n_curves=4,
        n_max=16,
        min_level_gap=0.04,
        V_exc_init=11.5,
        init_jitter_scale=0.05,
        init_seed=0,
    )
    with torch.no_grad():
        constrained.raw_common_energy_shift.fill_(100.0)
    params = constrained.level_params()
    energies = params["energies"].detach().cpu().numpy()
    weights = params["weights"].detach().cpu().numpy()
    shifts = energies - np.asarray(NIST_AR_I_4S_ENERGIES_EV)

    assert np.max(shifts) == pytest.approx(0.25, abs=1e-6)
    assert np.ptp(shifts) == pytest.approx(0.0, abs=1e-6)
    assert np.diff(energies) == pytest.approx(np.diff(NIST_AR_I_4S_ENERGIES_EV), abs=1e-6)
    assert np.all(weights >= 0.0)
    assert float(np.sum(weights)) == pytest.approx(1.0)

    with torch.no_grad():
        constrained.raw_common_energy_shift.fill_(-100.0)
    negative_shift = (
        constrained.level_params()["energies"].detach().cpu().numpy()
        - np.asarray(NIST_AR_I_4S_ENERGIES_EV)
    )
    assert np.min(negative_shift) == pytest.approx(-0.25, abs=1e-6)
    assert constrained.raw_width.numel() == 1


def test_background_switch_is_exact_and_freezes_only_background_parameters() -> None:
    without_background = H4sHypothesisModel(
        hypothesis="h1",
        n_curves=4,
        n_max=16,
        min_level_gap=0.04,
        V_exc_init=11.5,
        init_jitter_scale=0.05,
        init_seed=0,
    )
    with_background = H4sHypothesisModel(
        hypothesis="h1_background",
        n_curves=4,
        n_max=16,
        min_level_gap=0.04,
        V_exc_init=11.5,
        init_jitter_scale=0.05,
        init_seed=0,
    )

    assert float(without_background.phys_params()["high_energy_loss_strength"]) == 0.0
    for name in validation_h4s.BACKGROUND_PARAMETER_NAMES:
        assert getattr(without_background, name).requires_grad is False
        assert getattr(with_background, name).requires_grad is True


def test_paired_contrasts_cover_four_prespecified_effects() -> None:
    rows = [
        {
            "heldout_vr": 0.0,
            "seed": 0,
            "hypothesis": name,
            "rmse": 1.0 + index,
            "mae": 0.5 + index,
            "nrmse": 0.1 + 0.01 * index,
        }
        for index, name in enumerate(H4S_HYPOTHESES)
    ]

    contrasts = build_paired_contrasts(rows, expected_units=[asdict(unit) for unit in h4s_units("smoke")])

    assert len(contrasts) == 4
    assert {row["contrast"] for row in contrasts} == {
        "h4s_minus_h1",
        "h4s_background_minus_h1_background",
        "h1_background_minus_h1",
        "h4s_background_minus_h4s",
    }
    assert all(row["status"] == "complete" for row in contrasts)


def test_smoke_runner_writes_descriptive_outputs_and_propagates_failures(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    forward_path = tmp_path / "forward.json"
    forward_path.write_text("{}\n", encoding="utf-8")
    baseline = Baseline(
        kind="local",
        root=tmp_path,
        files={"config": config_path, "forward_evidence": forward_path},
        hashes={"config": sha256_file(config_path), "forward_evidence": sha256_file(forward_path)},
    )
    monkeypatch.setattr(validation_h4s, "load_all_curves", lambda **kwargs: _curves())

    def fake_execute(**kwargs):
        unit = kwargs["unit"]
        if unit.hypothesis == "h4s_background":
            raise RuntimeError("deliberate unit failure")
        return {
            "unit": asdict(unit),
            "metrics": {"rmse": 1.0, "mae": 0.8, "nrmse": 0.5, "n_points": 161},
            "predictions": [{"index": 0, "Va": 0.0, "observed": 1.0, "predicted": 0.0}],
            "elapsed_seconds": 0.01,
            "trainable_parameter_count": 10,
            "params": {"common_energy_shift_eV": 0.0},
            "reused": False,
        }

    monkeypatch.setattr(validation_h4s, "execute_unit", fake_execute)
    output_dir = tmp_path / "validation" / "h4s_comparison"

    result = validation_h4s.run(
        mode="smoke",
        input_path=tmp_path / "input.csv",
        baseline=baseline,
        output_dir=output_dir,
        validation_root=output_dir.parent,
        device="cpu",
    )

    assert result["ok"] is False
    assert result["completed"] == 3
    assert result["failed"] == 1
    assert result["claim_evaluable"] is False
    status = pd.read_csv(output_dir / "unit_status.csv")
    assert len(status) == 4
    assert set(status["status"]) == {"complete", "failed"}
    summary = json.loads((output_dir / "h4s_comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["claim_decision"] == "not_automated"
    assert len(summary["fold_median_direction_consistency"]) == 2
    contrasts = pd.read_csv(output_dir / "paired_contrasts.csv")
    assert (contrasts["status"] == "unavailable").sum() == 2
    for name in (
        "hypothesis_manifest.json",
        "unit_status.csv",
        "zero_shot_metrics.csv",
        "zero_shot_predictions.csv",
        "paired_contrasts.csv",
        "h4s_comparison_summary.json",
        "stage_status.json",
    ):
        assert (output_dir / name).is_file()

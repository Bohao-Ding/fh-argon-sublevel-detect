from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
import torch

from sublevel_detect import model, validation_synthetic
from sublevel_detect.validation_common import Baseline, sha256_file
from sublevel_detect.validation_synthetic import (
    ChannelTruth,
    apply_channel_truth,
    choose_epochs,
    match_channels,
    synthetic_units,
    summarize_recovery,
    truth_channels,
)


def _channel_table() -> pd.DataFrame:
    rows = []
    for channel in range(1, 9):
        rows.append(
            {
                "k": 8,
                "channel": channel,
                "energy_v": 11.0 + 0.4 * channel,
                "weight": channel,
            }
        )
    return pd.DataFrame(rows)


def test_truth_channels_cover_preregistered_k_values() -> None:
    table = _channel_table()

    k1 = truth_channels(1, delta=None, channel_table=table)
    k2 = truth_channels(2, delta=0.25, channel_table=table)
    k4 = truth_channels(4, delta=0.5, channel_table=table)
    k8 = truth_channels(8, delta=None, channel_table=table)

    assert k1.energies == pytest.approx((11.5,))
    assert k1.weights == pytest.approx((1.0,))
    assert k2.energies == pytest.approx((11.5, 11.75))
    assert k2.weights == pytest.approx((0.5, 0.5))
    assert k4.energies == pytest.approx((11.5, 12.0, 12.594, 13.965))
    assert sum(k4.weights) == pytest.approx(1.0)
    assert k8.energies == pytest.approx(tuple(table["energy_v"]))
    assert sum(k8.weights) == pytest.approx(1.0)


def test_full_synthetic_matrix_extends_confirmatory_cells_without_duplicate_units() -> None:
    units = synthetic_units("fullscan")
    keys = {
        (
            unit.truth_k,
            unit.delta,
            unit.noise_scale,
            unit.replicate,
            unit.train_seed,
            unit.prior_enabled,
        )
        for unit in units
    }

    assert len(units) == len(keys) == 297
    assert sum(unit.train_seed == 0 and unit.prior_enabled for unit in units) == 207
    assert sum(unit.train_seed in {1, 2} and unit.prior_enabled for unit in units) == 50
    assert sum(not unit.prior_enabled for unit in units) == 40


def test_epoch_gate_requires_every_k_below_one_percent_and_same_selected_k() -> None:
    good = [
        {"n_levels": k, "rmse_1600": 1.009, "rmse_3500": 1.0, "selected_k_1600": 4, "selected_k_3500": 4}
        for k in range(1, 9)
    ]
    bad_rmse = [dict(row) for row in good]
    bad_rmse[3]["rmse_1600"] = 1.011
    changed_k = [dict(row) for row in good]
    changed_k[0]["selected_k_1600"] = 5

    assert choose_epochs(good) == 1600
    assert choose_epochs(bad_rmse) == 3500
    assert choose_epochs(changed_k) == 3500


def test_hungarian_matching_reports_energy_and_weight_error() -> None:
    truth = ChannelTruth(2, (11.5, 12.0), (0.4, 0.6))
    estimate = ChannelTruth(2, (12.1, 11.4), (0.55, 0.45))

    matched = match_channels(truth, estimate)

    assert matched["matched_count"] == 2
    assert matched["energy_rmse_v"] == pytest.approx(0.1)
    assert matched["weight_mae"] == pytest.approx(0.05)


def test_recovery_summary_keeps_failed_units_in_exact_k_denominator() -> None:
    rows = [
        {"truth_k": 4, "selected_k": 4, "status": "complete"} for _ in range(24)
    ]
    rows.extend({"truth_k": 4, "selected_k": 8, "status": "complete"} for _ in range(3))
    rows.extend({"truth_k": 4, "selected_k": None, "status": "failed"} for _ in range(3))

    summary = summarize_recovery(rows, expected_total=30)

    assert summary["total"] == 30
    assert summary["failed"] == 3
    assert summary["exact_k_count"] == 24
    assert summary["exact_k_fraction"] == pytest.approx(0.8)
    assert summary["wilson_low"] < 0.8
    assert summary["reliable_recovery"] is False


def test_channel_truth_injection_changes_only_channel_parameters() -> None:
    network = model.PoissonRateFHCoreMultiLevel(n_curves=2, n_levels=4, min_level_gap=0.04)
    before = {
        name: parameter.detach().clone()
        for name, parameter in network.named_parameters()
        if name not in {"raw_E1", "raw_dE", "raw_level_logits"}
    }
    truth = ChannelTruth(4, (11.5, 11.75, 12.594, 13.965), (0.35, 0.40, 0.10, 0.15))

    apply_channel_truth(network, truth)

    levels = network.level_params()
    assert levels["energies"].detach().cpu().numpy() == pytest.approx(truth.energies, abs=1e-5)
    assert levels["weights"].detach().cpu().numpy() == pytest.approx(truth.weights, abs=1e-6)
    for name, parameter in network.named_parameters():
        if name in before:
            assert torch.equal(parameter.detach(), before[name])


def test_smoke_runner_records_recovery_outputs(monkeypatch, tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("curve_id,Vr,Va,IuA\n1,0,0,0\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    forward_path = tmp_path / "forward.json"
    forward_path.write_text("{}\n", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    prediction_path = tmp_path / "prediction.csv"
    prediction_path.write_text("curve_id,Vr,Va,observed,predicted,residual\n1,0,0,0,0,0\n", encoding="utf-8")
    channel_path = tmp_path / "channels.csv"
    _channel_table().to_csv(channel_path, index=False)
    baseline = Baseline(
        kind="local",
        root=tmp_path,
        files={
            "config": config_path,
            "forward_evidence": forward_path,
            "selected_checkpoint": checkpoint_path,
            "prediction_points": prediction_path,
            "channel_parameters": channel_path,
        },
        hashes={
            "config": sha256_file(config_path),
            "forward_evidence": sha256_file(forward_path),
            "selected_checkpoint": sha256_file(checkpoint_path),
            "prediction_points": sha256_file(prediction_path),
            "channel_parameters": sha256_file(channel_path),
        },
    )
    generated_path = tmp_path / "generated.csv"
    generated_path.write_text("curve_id,Vr,Va,IuA\n1,0,0,0\n", encoding="utf-8")
    monkeypatch.setattr(
        validation_synthetic,
        "prepare_synthetic_inputs",
        lambda **kwargs: {
            (4, 0.25, 1.0, 0): {
                "path": str(generated_path),
                "sha256": sha256_file(generated_path),
                "truth": ChannelTruth(4, (11.5, 11.75, 12.594, 13.965), (0.35, 0.4, 0.1, 0.15)),
            }
        },
    )
    monkeypatch.setattr(
        validation_synthetic.model,
        "run_level_scan",
        lambda cfg: {"decision": {"selected_k": 4}, "scan_dir": cfg.out_dir},
    )
    monkeypatch.setattr(
        validation_synthetic,
        "estimated_channels",
        lambda **kwargs: ChannelTruth(4, (11.5, 11.75, 12.594, 13.965), (0.35, 0.4, 0.1, 0.15)),
    )
    output_dir = tmp_path / "validation" / "synthetic_recovery"

    result = validation_synthetic.run(
        mode="smoke",
        input_path=input_path,
        baseline=baseline,
        output_dir=output_dir,
        validation_root=output_dir.parent,
        device="cpu",
    )

    assert result["ok"] is True
    assert result["status"] == "smoke_passed"
    assert result["unit_total"] == 1
    json.dumps(model.json_ready(result))
    for name in (
        "truth_manifest.csv",
        "selection_rows.csv",
        "channel_matching.csv",
        "recovery_summary.csv",
        "synthetic_summary.json",
        "stage_status.json",
    ):
        assert (output_dir / name).is_file()

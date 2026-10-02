"""Known-answer aggregation and checkpoint-free prediction replay."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sublevel_detect.spectral_revision import RevisedSpectrum, spectrum_summary
from sublevel_detect.spectral_revision_pipeline import SCHEMA, revision_sources
from sublevel_detect.validation_common import sha256_file
from sublevel_detect.validation_holdout import prediction_metrics
from sublevel_detect.spectral_pipeline import load_curves, training_curves

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_spectral_revision import recovery_table
from replay_spectral_revision import replay


def test_recovery_aggregates_starts_within_each_realization():
    rows = [dict(truth="single", replicate=rep, heldout_vr=0, seed=seed,
                 mode_eV=value, mean_eV=value, sd_eV=.1, truth_mode_eV=11.65,
                 truth_mean_eV=11.65, truth_sd_eV=0, peak_status="concentrated")
            for rep, values in enumerate(([10, 11.65, 16], [11, 11.75, 14]))
            for seed, value in enumerate(values)]
    result = recovery_table(pd.DataFrame(rows), ["truth", "replicate", "heldout_vr"])
    np.testing.assert_allclose(result.mode_absolute_error_eV, [0, .1], atol=1e-12)
    assert len(result) == 2
    assert result.concentrated_starts.tolist() == [3, 3]


def test_revision_replays_parameters_and_detects_changed_bytes(tmp_path):
    net = RevisedSpectrum(4, "H1")
    unit = "a" * 64
    spec = dict(schema=SCHEMA, family="H1", seed=0, excitation=11.5, step=.01,
                knot_step=.1, domain=[9., 16.], phase_response=False, n_curves=4,
                normalization_voltage=80., fixed_width=None, training_vr=[0., 4., 6., 8.])
    (tmp_path / "units").mkdir()
    receipt = dict(spec=spec, raw_parameter_values={n: p.detach().numpy().tolist() for n, p in net.named_parameters()},
                   distribution=spectrum_summary(net))
    (tmp_path / "units" / f"{unit[:20]}.json").write_text(json.dumps(receipt))
    identity = dict(schema=SCHEMA, sources=revision_sources(), input_sha256=sha256_file(ROOT / "data/argon/FHdata.xlsx"))
    (tmp_path / "run_identity.json").write_text(json.dumps(identity))
    archive = {float(c["Vr"]): c for c in load_curves(ROOT / "data/argon/FHdata.xlsx")}
    points = []
    for scope, vr, table in (("outer_0V", 0, "outer_scores"), ("final_training", 4, "training_scores"),
                              ("stress_10V", 10, "stress_scores")):
        va = torch.linspace(0, 80, 161, dtype=torch.float64)
        index, mode = (1, "curve") if scope == "final_training" else (0, "neutral")
        with torch.no_grad():
            prediction = net(va, torch.full_like(va, vr), index, mode).numpy()
        observed = archive[vr]["Ip"]
        metrics = prediction_metrics(observed, prediction, denominator_observed=observed)
        pd.DataFrame([dict(unit=unit, **({"Vr": vr} if table == "training_scores" else {"heldout_vr": vr}), **metrics)]).to_csv(tmp_path / f"{table}.csv", index=False)
        points += [dict(unit=unit, scope=scope, Vr=vr, Va=float(v), predicted_uA=float(p), observed_uA=float(o))
                   for v, p, o in zip(va, prediction, observed)]
    pd.DataFrame(points).to_csv(tmp_path / "real_prediction_points.csv", index=False)
    pd.DataFrame([dict(unit=unit, energy_eV=spectrum_summary(net)["mode_eV"], density_per_eV=np.nan)]).to_csv(tmp_path / "distribution_reference.csv", index=False)
    archived = training_curves(load_curves(ROOT / "data/argon/FHdata.xlsx"))[0]
    with torch.no_grad():
        phase_prediction = net(torch.tensor(archived["Va"], dtype=torch.float64), torch.zeros(161, dtype=torch.float64)).numpy()
    phase_metrics = prediction_metrics(archived["Ip"], phase_prediction, denominator_observed=archived["Ip"])
    pd.DataFrame([dict(unit=unit, heldout_vr=0, **phase_metrics)]).to_csv(tmp_path / "phase_scores.csv", index=False)
    continuous = RevisedSpectrum(4, "C")
    boot_unit = "b" * 64
    boot_receipt = dict(spec=dict(spec, family="C"),
        raw_parameter_values={n: p.detach().numpy().tolist() for n, p in continuous.named_parameters()},
        distribution=spectrum_summary(continuous))
    (tmp_path / "units" / f"{boot_unit[:20]}.json").write_text(json.dumps(boot_receipt))
    pd.DataFrame([dict(unit=boot_unit, replicate=rep, **spectrum_summary(continuous)) for rep in (0, 1)]).to_csv(tmp_path / "bootstrap.csv", index=False)
    with torch.no_grad():
        density = continuous.density().numpy()
    pd.DataFrame(dict(energy_eV=continuous.energy_grid.numpy(), lower_density=density,
                      median_density=density, upper_density=density)).to_csv(tmp_path / "bootstrap_density_band.csv", index=False)
    (tmp_path / "SHA256SUMS.txt").write_text("".join(
        f"{sha256_file(p)}  {p.relative_to(tmp_path).as_posix()}\n" for p in sorted(tmp_path.rglob("*")) if p.is_file()))
    result = replay(tmp_path)
    assert result["ok"] and result["replayed_points"] == 483
    assert result["replayed_phase_points"] == 161
    assert result["max_bootstrap_band_difference"] < 1e-12
    assert result["max_subset_difference_uA"] < 1e-12
    original_points = (tmp_path / "real_prediction_points.csv").read_bytes()
    modified = pd.read_csv(tmp_path / "real_prediction_points.csv")
    modified.loc[0, "observed_uA"] += .01
    modified.to_csv(tmp_path / "real_prediction_points.csv", index=False)
    original_manifest = (tmp_path / "SHA256SUMS.txt").read_bytes()
    (tmp_path / "SHA256SUMS.txt").write_text("".join(
        f"{sha256_file(p)}  {p.relative_to(tmp_path).as_posix()}\n" for p in sorted(tmp_path.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS.txt"))
    with pytest.raises(ValueError, match="archive_observation_mismatch"):
        replay(tmp_path)
    (tmp_path / "real_prediction_points.csv").write_bytes(original_points)
    (tmp_path / "SHA256SUMS.txt").write_bytes(original_manifest)
    with (tmp_path / "real_prediction_points.csv").open("a") as stream:
        stream.write("corrupted")
    with pytest.raises(ValueError, match="evidence_byte_mismatch"):
        replay(tmp_path)

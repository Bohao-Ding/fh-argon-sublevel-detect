import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from fh_transport.model_v2 import NIST_ARGON_4S_EV
from fh_transport.model_v2m import (
    LEVEL_LABELS,
    TabulatedPartialCrossSectionFranckHertz,
    load_cross_section_table,
)
from fh_transport.pipeline_v2m import run_hypothesis
from fh_transport.train import TrainConfig


def _write_table(directory: Path, *, violate_threshold: bool = False) -> Path:
    path = directory / "argon_4s.csv"
    energies = np.asarray([0.0, 11.5, 12.0, 15.0, 30.0, 60.0, 100.0])
    sigma = np.zeros((energies.size, 4), dtype=float)
    for level, threshold in enumerate(NIST_ARGON_4S_EV):
        excess = np.maximum(energies - threshold, 0.0)
        sigma[:, level] = (level + 1.0) * excess / (5.0 + excess) * 1.0e-20
    sigma[:, 0] *= np.linspace(1.0, 0.4, energies.size)
    sigma[:, 3] *= np.linspace(0.3, 1.2, energies.size)
    if violate_threshold:
        sigma[0, 0] = 1.0e-22
    lines = ["energy_eV,sigma_1_m2,sigma_2_m2,sigma_3_m2,sigma_4_m2"]
    for energy, row in zip(energies, sigma):
        lines.append(",".join([f"{energy:.6g}", *(f"{value:.12g}" for value in row)]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_name": "synthetic unit-test table",
                "source_url": "https://example.invalid/unit-test",
                "level_order": list(LEVEL_LABELS),
                "energy_unit": "eV",
                "cross_section_unit": "m2",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_v2m_h1_uses_the_sum_of_the_matched_partial_cross_sections() -> None:
    with TemporaryDirectory() as directory:
        table = load_cross_section_table(_write_table(Path(directory)))
        model = TabulatedPartialCrossSectionFranckHertz(
            hypothesis="h1",
            n_layers=6,
            max_collisions=3,
            quadrature_order=4,
            seed=0,
            cross_section_table=table,
        )
    energies, _ = model.excitation_parameters()
    params = model.physical_parameters()
    aggregate = model._cross_sections(
        torch.tensor([30.0]),
        energies,
        params["cross_section_rise_eV"],
        params["cross_section_decay_eV"],
        params["cross_section_power"],
    )
    expected = torch.sum(model.partial_cross_section_relative[4]).reshape(1, 1)
    assert torch.allclose(aggregate, expected)
    assert not model.raw_rate_logits.requires_grad
    assert not model.raw_cross_rise.requires_grad
    assert not model.raw_threshold_shape_v2f.requires_grad
    assert not model.raw_cross_decay_v2f.requires_grad


def test_v2m_h4s_fixes_atomic_energies_and_uses_dynamic_partial_cross_sections() -> None:
    with TemporaryDirectory() as directory:
        table = load_cross_section_table(_write_table(Path(directory)))
        model = TabulatedPartialCrossSectionFranckHertz(
            hypothesis="h4s",
            n_layers=6,
            max_collisions=3,
            quadrature_order=4,
            seed=0,
            cross_section_table=table,
        )
    energies, amplitudes = model.excitation_parameters()
    assert torch.allclose(energies, torch.tensor(NIST_ARGON_4S_EV))
    assert torch.equal(amplitudes, torch.ones(4))
    assert not model.raw_energy_shift.requires_grad
    assert not model.raw_rate_logits.requires_grad

    params = model.physical_parameters()
    low = model._cross_sections(
        torch.tensor([15.0]),
        energies,
        params["cross_section_rise_eV"],
        params["cross_section_decay_eV"],
        params["cross_section_power"],
    )[0]
    high = model._cross_sections(
        torch.tensor([60.0]),
        energies,
        params["cross_section_rise_eV"],
        params["cross_section_decay_eV"],
        params["cross_section_power"],
    )[0]
    assert not torch.allclose(low / low.sum(), high / high.sum())


def test_v2m_rejects_nonzero_cross_section_below_threshold() -> None:
    with TemporaryDirectory() as directory:
        path = _write_table(Path(directory), violate_threshold=True)
        try:
            load_cross_section_table(path)
        except ValueError as error:
            assert "below its NIST threshold" in str(error)
        else:
            raise AssertionError("Expected a threshold-validation failure")


def test_v2m_h4s_forward_is_finite_and_conserves_collision_mass() -> None:
    with TemporaryDirectory() as directory:
        table = load_cross_section_table(_write_table(Path(directory)))
        model = TabulatedPartialCrossSectionFranckHertz(
            hypothesis="h4s",
            n_layers=6,
            max_collisions=3,
            quadrature_order=4,
            seed=0,
            cross_section_table=table,
        )
    va = torch.linspace(0.0, 60.0, 13)
    mass = model.collision_state_probabilities(va)
    prediction = model(va, torch.full_like(va, 8.0))
    assert torch.allclose(mass.sum(dim=2), torch.ones_like(mass[:, :, 0]), atol=2.0e-6)
    assert float(prediction[0].detach()) == 0.0
    assert torch.all(torch.isfinite(prediction))


def test_v2m_h1_one_epoch_cpu_training_smoke_cleans_with_temporary_directory() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        table_path = _write_table(temporary_root)
        result = run_hypothesis(
            TrainConfig(
                data_path=str(project_root / "data" / "FHdata.xlsx"),
                output_dir=str(temporary_root / "unused"),
                device="cpu",
                max_epochs=1,
                min_epochs=1,
                patience=1,
                n_layers=4,
                max_collisions=2,
                quadrature_order=2,
                log_interval=1,
            ),
            "h1",
            temporary_root / "output",
            cross_section_table_path=table_path,
        )
        assert result["device"] == "cpu"
        assert result["epochs_completed"] == 1
        assert result["parameters"]["model_version"] == "v2m-tabulated-partial-cross-sections"

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model_v2 import NIST_ARGON_4S_EV
from .model_v2l import CalibratedCollisionFieldFranckHertz


LEVEL_LABELS = (
    "4s[3/2]2",
    "4s[3/2]1",
    "4s'[1/2]0",
    "4s'[1/2]1",
)
SIGMA_COLUMNS = tuple(f"sigma_{index}_m2" for index in range(1, 5))


@dataclass(frozen=True)
class CrossSectionTable:
    energy_eV: torch.Tensor
    sigma_m2: torch.Tensor
    source_name: str
    source_url: str
    csv_sha256: str
    csv_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_cross_section_table(path: str | Path) -> CrossSectionTable:
    """Load a provenance-bearing table of four Ar I 4s partial cross sections."""
    csv_path = Path(path).resolve()
    metadata_path = csv_path.with_suffix(".metadata.json")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cross-section CSV does not exist: {csv_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Formal v2m training requires the provenance sidecar: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", 0)) != 1:
        raise ValueError("Cross-section metadata schema_version must be 1")
    if tuple(metadata.get("level_order", ())) != LEVEL_LABELS:
        raise ValueError(f"level_order must be {list(LEVEL_LABELS)}")
    if metadata.get("energy_unit") != "eV" or metadata.get("cross_section_unit") != "m2":
        raise ValueError("Cross-section units must be energy=eV and cross_section=m2")
    source_name = str(metadata.get("source_name", "")).strip()
    source_url = str(metadata.get("source_url", "")).strip()
    if not source_name or not source_url.startswith(("https://", "http://")):
        raise ValueError("Metadata requires source_name and an http(s) source_url")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ("energy_eV", *SIGMA_COLUMNS)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"Cross-section CSV requires columns {list(required)}")
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError("Cross-section CSV requires at least three energy rows")

    energy = np.asarray([float(row["energy_eV"]) for row in rows], dtype=np.float64)
    sigma = np.asarray(
        [[float(row[column]) for column in SIGMA_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(sigma)):
        raise ValueError("Cross-section table contains non-finite values")
    if not np.all(np.diff(energy) > 0.0):
        raise ValueError("Cross-section energy grid must be strictly increasing")
    if energy[0] > 0.0 or energy[-1] < 100.0:
        raise ValueError("Cross-section grid must cover the current 0-80 V data through 100 eV")
    if np.any(sigma < 0.0) or not np.any(sigma > 0.0):
        raise ValueError("Partial cross sections must be non-negative and not all zero")
    for level_index, threshold in enumerate(NIST_ARGON_4S_EV):
        below_threshold = energy < float(threshold) - 1.0e-6
        if np.any(sigma[below_threshold, level_index] > 0.0):
            raise ValueError(
                f"sigma_{level_index + 1}_m2 must be zero below its NIST threshold"
            )

    return CrossSectionTable(
        energy_eV=torch.tensor(energy, dtype=torch.float32),
        sigma_m2=torch.tensor(sigma, dtype=torch.float32),
        source_name=source_name,
        source_url=source_url,
        csv_sha256=_sha256(csv_path),
        csv_path=str(csv_path),
    )


class TabulatedPartialCrossSectionFranckHertz(CalibratedCollisionFieldFranckHertz):
    """v2l with fixed atomic energies and tabulated H4s partial cross sections.

    H1 uses the sum of the four partial cross sections as one matched collision
    hazard.  H4s uses the same table split into four competing hazards and fixes
    the four atomic loss energies to NIST.  This removes both constant softmax
    channel fractions and the fitted atomic-level shift from H4s, while keeping
    the collision-rate input matched between H1 and H4s.  The shared optical
    depth absorbs only the table's common scale.
    """

    def __init__(
        self,
        *args: Any,
        cross_section_table: CrossSectionTable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if cross_section_table is None:
            raise ValueError(
                "v2m requires a provenance-bearing four-channel partial cross-section table"
            )
        if tuple(cross_section_table.sigma_m2.shape) != (
            cross_section_table.energy_eV.numel(),
            4,
        ):
            raise ValueError("The cross-section table must have shape [n_energy, 4]")
        scale = torch.max(torch.sum(cross_section_table.sigma_m2, dim=1))
        if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
            raise ValueError("Cross-section normalization scale must be positive")
        self.register_buffer(
            "partial_cross_section_energy_eV",
            cross_section_table.energy_eV.detach().clone(),
        )
        self.register_buffer(
            "partial_cross_section_relative",
            cross_section_table.sigma_m2.detach().clone() / scale,
        )
        self.register_buffer("partial_cross_section_scale_m2", scale.detach().clone())
        for parameter in (
            self.raw_cross_rise,
            self.raw_threshold_shape_v2f,
            self.raw_cross_decay_v2f,
        ):
            parameter.requires_grad_(False)
        self.raw_rate_logits.requires_grad_(False)
        if self.hypothesis == "h4s":
            self.raw_energy_shift.requires_grad_(False)
        self._cross_section_source = {
            "name": cross_section_table.source_name,
            "url": cross_section_table.source_url,
            "csv_sha256": cross_section_table.csv_sha256,
            "csv_path": cross_section_table.csv_path,
        }

    def excitation_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hypothesis == "h1":
            energies, _ = super().excitation_parameters()
            return energies, torch.ones_like(energies)
        energies = self.nist_4s_eV
        unit_amplitudes = torch.ones_like(energies)
        return energies, unit_amplitudes

    def _cross_sections(
        self,
        kinetic: torch.Tensor,
        energies: torch.Tensor,
        rise: torch.Tensor,
        decay: torch.Tensor,
        power: torch.Tensor,
    ) -> torch.Tensor:
        grid = self.partial_cross_section_energy_eV
        values = self.partial_cross_section_relative
        flat = kinetic.reshape(-1)
        bounded = torch.clamp(flat, min=grid[0], max=grid[-1])
        upper = torch.searchsorted(grid, bounded, right=True)
        upper = torch.clamp(upper, min=1, max=grid.numel() - 1)
        lower = upper - 1
        x0 = grid[lower]
        x1 = grid[upper]
        fraction = ((bounded - x0) / torch.clamp(x1 - x0, min=1.0e-12)).unsqueeze(-1)
        interpolated = values[lower] + fraction * (values[upper] - values[lower])
        interpolated = interpolated.reshape(*kinetic.shape, 4)
        if self.hypothesis == "h1":
            aggregate = torch.sum(interpolated, dim=-1, keepdim=True)
            threshold_open = kinetic[..., None] >= energies
            return torch.where(threshold_open, aggregate, torch.zeros_like(aggregate))
        threshold_open = kinetic[..., None] >= energies
        return torch.where(threshold_open, interpolated, torch.zeros_like(interpolated))

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "model_class": type(self).__name__,
                "model_version": "v2m-tabulated-partial-cross-sections",
                "atomic_energy_treatment": (
                    "H1 effective energy is fitted; H4s NIST loss energies are fixed exactly"
                ),
                "channel_hazard_treatment": (
                    "H1 uses the sum of four partial cross sections; H4s uses the same "
                    "table split into four hazards; no constant softmax channel weights"
                ),
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                ),
            }
        )
        report.pop("channel_rate_fractions", None)
        report.pop("cross_section_rise_eV", None)
        report.pop("cross_section_decay_eV", None)
        report.pop("cross_section_power", None)
        report.update(
            {
                "partial_cross_section_source": self._cross_section_source,
                "partial_cross_section_scale_m2": float(
                    self.partial_cross_section_scale_m2.detach().cpu()
                ),
                "partial_cross_section_energy_range_eV": [
                    float(self.partial_cross_section_energy_eV[0].detach().cpu()),
                    float(self.partial_cross_section_energy_eV[-1].detach().cpu()),
                ],
                "partial_cross_section_level_order": list(LEVEL_LABELS),
                "tabulated_cross_section_supersedes": [
                    "cross_section_rise_eV",
                    "cross_section_decay_eV",
                    "cross_section_power",
                ],
            }
        )
        return report

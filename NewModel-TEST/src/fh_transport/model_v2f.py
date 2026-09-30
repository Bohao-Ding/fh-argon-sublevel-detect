from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .model_v2 import _inverse_bounded
from .model_v2d import EnergyDependentScatteringFranckHertz


class ThresholdGaussianFranckHertz(EnergyDependentScatteringFranckHertz):
    """Energy-dependent scattering with a saturating threshold hazard.

    This kernel replaces the boundary-seeking rational cross-section surrogate
    by a Weibull-type threshold activation.  Signed axial energy enters the
    Gaussian collector gate directly, avoiding a separate hard reachability
    switch.  The injection-energy scale remains distinct from gate broadening.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for parameter in (
            self.raw_thermal_energy,
            self.raw_arrival_scale,
            self.raw_optical_depth,
            self.raw_cross_decay,
            self.raw_cross_power,
        ):
            parameter.requires_grad_(False)
        self.raw_injection_energy_v2f = nn.Parameter(
            torch.tensor(_inverse_bounded(0.08, 0.02, 0.30))
        )
        self.raw_optical_depth_v2f = nn.Parameter(
            torch.tensor(_inverse_bounded(30.0, 0.2, 120.0))
        )
        self.raw_threshold_shape_v2f = nn.Parameter(
            torch.tensor(_inverse_bounded(2.0, 0.5, 6.0))
        )
        self.raw_cross_decay_v2f = nn.Parameter(
            torch.tensor(_inverse_bounded(120.0, 10.0, 500.0))
        )

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        params = super().physical_parameters()
        params.update(
            {
                "thermal_energy_eV": self.bounded(
                    self.raw_injection_energy_v2f, 0.02, 0.30
                ),
                "optical_depth": self.bounded(self.raw_optical_depth_v2f, 0.2, 120.0),
                "cross_section_power": self.bounded(
                    self.raw_threshold_shape_v2f, 0.5, 6.0
                ),
                "cross_section_decay_eV": self.bounded(
                    self.raw_cross_decay_v2f, 10.0, 500.0
                ),
            }
        )
        return params

    def _cross_sections(
        self,
        kinetic: torch.Tensor,
        energies: torch.Tensor,
        rise: torch.Tensor,
        decay: torch.Tensor,
        power: torch.Tensor,
    ) -> torch.Tensor:
        excess = torch.relu(kinetic[..., None] - energies)
        scaled = excess / torch.clamp(rise, min=1.0e-8)
        activation = -torch.expm1(-torch.pow(scaled, power))
        return activation * torch.exp(-excess / decay)

    def collector_state_transmission(
        self,
        va: torch.Tensor,
        vr: torch.Tensor,
        last_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        va = va.reshape(-1).float()
        vr = vr.reshape(-1).float()
        if va.shape != vr.shape:
            raise ValueError("Va and Vr must have the same shape")
        if last_position is None:
            probability, last_mass = self.collision_state_moments(va)
            last_position = last_mass / torch.clamp(probability, min=1.0e-12)
        expected_shape = (va.numel(), self.thermal_nodes.numel(), self.collision_counts.shape[0])
        if tuple(last_position.shape) != expected_shape:
            raise ValueError(
                f"last_position must have shape {expected_shape}, got {tuple(last_position.shape)}"
            )

        params = self.physical_parameters()
        energies = params["energies_eV"]
        loss = self.collision_counts @ energies
        field_energy = params["acceleration_scale"] * va - params["arrival_offset_eV"]
        barrier = params["collector_offset_eV"] + params["retarding_scale"] * vr
        thermal = params["thermal_energy_eV"] * self.thermal_nodes
        post_collision_energy = torch.relu(
            field_energy[:, None, None] * last_position
            + thermal[None, :, None]
            - loss[None, None, :]
        )
        axial_recovery = field_energy[:, None, None] * (1.0 - last_position)
        angular_weights = self.angle_flux_weights / torch.clamp(
            torch.sum(self.angle_flux_weights), min=1.0e-8
        )

        isotropic_gate = torch.zeros(expected_shape, device=va.device, dtype=va.dtype)
        for mu, weight in zip(self.angle_mu.unbind(), angular_weights.unbind()):
            axial_energy = axial_recovery + post_collision_energy * torch.square(mu)
            z = (axial_energy - barrier[:, None, None]) / params["gate_sigma_eV"]
            isotropic_gate = isotropic_gate + weight * 0.5 * (
                1.0 + torch.erf(z / math.sqrt(2.0))
            )

        forward_energy = axial_recovery + post_collision_energy
        forward_z = (forward_energy - barrier[:, None, None]) / params["gate_sigma_eV"]
        forward_gate = 0.5 * (1.0 + torch.erf(forward_z / math.sqrt(2.0)))
        collided = (torch.sum(self.collision_counts, dim=1) > 0).to(va.dtype)
        anisotropy = post_collision_energy / (
            post_collision_energy + params["scattering_anisotropy_scale_eV"]
        )
        anisotropy = anisotropy * collided[None, None, :]
        return isotropic_gate * (1.0 - anisotropy) + forward_gate * anisotropy

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "model_class": type(self).__name__,
                "model_version": "v2f-threshold-gaussian",
                "collision_hazard": "Weibull threshold activation with exponential high-energy decay",
                "collector_reachability": "signed axial energy convolved by Gaussian gate",
                "legacy_frozen_parameters": [
                    "raw_thermal_energy",
                    "raw_arrival_scale",
                    "raw_optical_depth",
                    "raw_cross_decay",
                    "raw_cross_power",
                    "raw_angle_sigma",
                ],
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in self.parameters() if parameter.requires_grad
                ),
            }
        )
        return report

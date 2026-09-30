from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .model_v2 import _inverse_bounded
from .model_v2c import IsotropicLastCollisionFranckHertz


class EnergyDependentScatteringFranckHertz(IsotropicLastCollisionFranckHertz):
    """Last-collision closure with energy-dependent scattering anisotropy.

    Near-threshold post-collision electrons use isotropic forward flux.  As the
    post-collision kinetic energy grows, the angular gate continuously blends
    toward the forward direction.  The interpolation is a reduced DCS closure.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raw_scattering_scale = nn.Parameter(
            torch.tensor(_inverse_bounded(1.0, 0.05, 10.0))
        )

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        params = super().physical_parameters()
        params["scattering_anisotropy_scale_eV"] = self.bounded(
            self.raw_scattering_scale, 0.05, 10.0
        )
        return params

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
            reachability = -torch.expm1(
                -torch.relu(axial_energy)
                / torch.clamp(params["arrival_scale_eV"], min=1.0e-8)
            )
            z = (axial_energy - barrier[:, None, None]) / params["gate_sigma_eV"]
            isotropic_gate = isotropic_gate + weight * reachability * (
                0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            )

        forward_energy = axial_recovery + post_collision_energy
        forward_reachability = -torch.expm1(
            -torch.relu(forward_energy)
            / torch.clamp(params["arrival_scale_eV"], min=1.0e-8)
        )
        forward_z = (forward_energy - barrier[:, None, None]) / params["gate_sigma_eV"]
        forward_gate = forward_reachability * 0.5 * (
            1.0 + torch.erf(forward_z / math.sqrt(2.0))
        )
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
                "model_version": "v2d-energy-dependent-scattering",
                "post_collision_angular_distribution": (
                    "isotropic-to-forward blend controlled by post-collision energy"
                ),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in self.parameters() if parameter.requires_grad
                ),
            }
        )
        return report

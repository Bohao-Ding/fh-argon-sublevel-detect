from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .model_v2 import _inverse_bounded
from .model_v2h import ReachabilityGatedFranckHertz


class FiniteApertureFranckHertz(ReachabilityGatedFranckHertz):
    """v2h kernel with collision-aware finite angular acceptance."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raw_aperture_sigma_v2i = nn.Parameter(
            torch.tensor(_inverse_bounded(1.00, 0.15, math.pi / 2.0))
        )

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        params = super().physical_parameters()
        params["aperture_sigma_rad"] = self.bounded(
            self.raw_aperture_sigma_v2i, 0.15, math.pi / 2.0
        )
        return params

    def collector_energy_gate(
        self,
        axial_energy: torch.Tensor,
        barrier: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        z = (axial_energy - barrier[:, None, None]) / sigma
        return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

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
        loss = self.collision_counts @ params["energies_eV"]
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
        theta = torch.acos(torch.clamp(self.angle_mu, 1.0e-6, 1.0))
        aperture = torch.exp(
            -0.5 * torch.square(theta / params["aperture_sigma_rad"])
        )
        isotropic_gate = torch.zeros(expected_shape, device=va.device, dtype=va.dtype)
        for mu, weight, acceptance in zip(
            self.angle_mu.unbind(), angular_weights.unbind(), aperture.unbind()
        ):
            axial_energy = axial_recovery + post_collision_energy * torch.square(mu)
            reachability = -torch.expm1(
                -torch.relu(axial_energy)
                / torch.clamp(params["arrival_scale_eV"], min=1.0e-8)
            )
            isotropic_gate = isotropic_gate + weight * acceptance * reachability * (
                self.collector_energy_gate(
                    axial_energy, barrier, params["gate_sigma_eV"]
                )
            )

        forward_energy = axial_recovery + post_collision_energy
        forward_reachability = -torch.expm1(
            -torch.relu(forward_energy)
            / torch.clamp(params["arrival_scale_eV"], min=1.0e-8)
        )
        forward_gate = forward_reachability * self.collector_energy_gate(
            forward_energy, barrier, params["gate_sigma_eV"]
        )

        collided = (torch.sum(self.collision_counts, dim=1) > 0).to(va.dtype)
        collision_forward_fraction = post_collision_energy / (
            post_collision_energy + params["scattering_anisotropy_scale_eV"]
        )
        forward_fraction = (1.0 - collided)[None, None, :] + (
            collided[None, None, :] * collision_forward_fraction
        )
        return isotropic_gate * (1.0 - forward_fraction) + forward_gate * forward_fraction

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "model_class": type(self).__name__,
                "model_version": "v2i-finite-angular-aperture",
                "post_collision_angular_distribution": (
                    "uncollided states remain forward; collided states blend isotropic and forward"
                ),
                "angle_sigma_role": (
                    "frozen legacy parameter; superseded by aperture_sigma_rad"
                ),
                "angular_acceptance": (
                    "unnormalized Gaussian aperture exp[-theta^2/(2*sigma_aperture^2)]"
                ),
                "additional_trainable_parameters_vs_v2a": 2,
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in self.parameters() if parameter.requires_grad
                ),
            }
        )
        return report

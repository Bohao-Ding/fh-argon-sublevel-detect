from __future__ import annotations

import math
from typing import Any

import torch

from .model_v2b import LastCollisionFranckHertz


class IsotropicLastCollisionFranckHertz(LastCollisionFranckHertz):
    """Last-collision closure with fixed forward-hemisphere isotropic flux.

    The v2b angular width mixed collision scattering and collector acceptance.
    Here post-collision directions use p(mu)=2*mu on the forward hemisphere,
    leaving the energy barrier to perform collector selection.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raw_angle_sigma.requires_grad_(False)

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

        gate = torch.zeros(expected_shape, device=va.device, dtype=va.dtype)
        for mu, weight in zip(self.angle_mu.unbind(), angular_weights.unbind()):
            axial_energy = axial_recovery + post_collision_energy * torch.square(mu)
            reachability = -torch.expm1(
                -torch.relu(axial_energy)
                / torch.clamp(params["arrival_scale_eV"], min=1.0e-8)
            )
            z = (axial_energy - barrier[:, None, None]) / params["gate_sigma_eV"]
            energy_gate = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            gate = gate + weight * reachability * energy_gate
        return gate

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "model_class": type(self).__name__,
                "model_version": "v2c-fixed-isotropic-scattering",
                "post_collision_angular_distribution": "p(mu)=2mu, 0<=mu<=1",
                "angle_sigma_role": "frozen legacy parameter; not used in v2c collector",
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in self.parameters() if parameter.requires_grad
                ),
            }
        )
        return report

from __future__ import annotations

import math
from typing import Any

import torch

from .model_v2 import CompetingChannelFranckHertz


class LastCollisionFranckHertz(CompetingChannelFranckHertz):
    """Competing-history kernel with a last-collision position closure.

    Besides probability mass, the thin-layer master equation propagates the
    probability-weighted position of the most recent inelastic collision.  At
    the collector, post-collision kinetic energy is angularly redistributed,
    while field energy gained after the last collision remains axial.  This is
    a differentiable first-moment closure, not a trajectory-level solver.
    """

    def collision_state_moments(self, va: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return probability and probability-weighted last collision position."""
        va = va.reshape(-1).float()
        params = self.physical_parameters()
        energies = params["energies_eV"]
        loss = self.collision_counts @ energies
        field_energy = params["acceleration_scale"] * va - params["collision_region_offset_eV"]
        thermal = params["thermal_energy_eV"] * self.thermal_nodes
        shape = (va.numel(), thermal.numel(), self.collision_counts.shape[0])
        probability = torch.zeros(shape, device=va.device, dtype=va.dtype)
        last_position_mass = torch.zeros_like(probability)
        probability[:, :, 0] = 1.0
        active = (torch.sum(self.collision_counts, dim=1) < self.max_collisions).to(va.dtype)

        for layer in range(self.n_layers):
            position = (float(layer) + 0.5) / float(self.n_layers)
            kinetic = (
                field_energy[:, None, None] * position
                + thermal[None, :, None]
                - loss[None, None, :]
            )
            rates = self._cross_sections(
                kinetic,
                energies,
                params["cross_section_rise_eV"],
                params["cross_section_decay_eV"],
                params["cross_section_power"],
            )
            rates = (
                params["optical_depth"]
                * rates
                * params["channel_rate_fractions"][None, None, None, :]
                * active[None, None, :, None]
                / float(self.n_layers)
            )
            total_rate = torch.sum(rates, dim=-1)
            event_probability = -torch.expm1(-total_rate)
            channel_probability = event_probability[..., None] * rates / (
                total_rate[..., None] + 1.0e-12
            )
            updated_probability = probability * (1.0 - event_probability)
            updated_last_mass = last_position_mass * (1.0 - event_probability)
            for level in range(self.n_levels):
                destinations = self.next_state[level]
                valid = destinations >= 0
                contribution = probability[:, :, valid] * channel_probability[:, :, valid, level]
                updated_probability = torch.index_add(
                    updated_probability, 2, destinations[valid], contribution
                )
                updated_last_mass = torch.index_add(
                    updated_last_mass, 2, destinations[valid], contribution * position
                )
            probability = updated_probability
            last_position_mass = updated_last_mass
        return probability, last_position_mass

    def collision_state_probabilities(self, va: torch.Tensor) -> torch.Tensor:
        return self.collision_state_moments(va)[0]

    def collector_state_transmission(
        self,
        va: torch.Tensor,
        vr: torch.Tensor,
        last_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return collector transmission for each thermal node and history."""
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

        theta = torch.acos(torch.clamp(self.angle_mu, 1.0e-6, 1.0))
        angular_acceptance = torch.exp(-0.5 * torch.square(theta / params["angle_sigma_rad"]))
        angular_weights = self.angle_flux_weights * angular_acceptance
        angular_weights = angular_weights / torch.clamp(torch.sum(angular_weights), min=1.0e-8)

        post_collision_energy = torch.relu(
            field_energy[:, None, None] * last_position
            + thermal[None, :, None]
            - loss[None, None, :]
        )
        axial_recovery = field_energy[:, None, None] * (1.0 - last_position)
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

    def forward(self, va: torch.Tensor, vr: torch.Tensor) -> torch.Tensor:
        va = va.reshape(-1).float()
        vr = vr.reshape(-1).float()
        if va.shape != vr.shape:
            raise ValueError("Va and Vr must have the same shape")
        unique_va, inverse = torch.unique(va, sorted=True, return_inverse=True)
        collision_mass, last_mass = self.collision_state_moments(unique_va)
        collision_mass = collision_mass[inverse]
        last_mass = last_mass[inverse]
        last_position = last_mass / torch.clamp(collision_mass, min=1.0e-12)
        collector_gate = self.collector_state_transmission(va, vr, last_position)
        transmission = torch.sum(collision_mass * collector_gate, dim=2)
        transmission = torch.sum(transmission * self.thermal_weights[None, :], dim=1)

        params = self.physical_parameters()
        supply_voltage = torch.relu(params["acceleration_scale"] * va)
        reduced_voltage = supply_voltage / params["supply_scale_V"]
        child_langmuir = torch.pow(reduced_voltage, 1.5)
        supply = params["current_limit_uA"] * child_langmuir / (1.0 + child_langmuir)
        return torch.clamp(supply * transmission, min=0.0)

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "model_class": type(self).__name__,
                "model_version": "v2b-last-collision-moment",
                "collector_closure": "probability-weighted last inelastic collision position",
                "additional_trainable_parameters_vs_v2a": 0,
            }
        )
        return report

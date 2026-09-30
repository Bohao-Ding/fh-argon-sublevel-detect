from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from .model_v2 import _inverse_bounded
from .model_v2i import FiniteApertureFranckHertz


class CollisionFieldEnsembleFranckHertz(FiniteApertureFranckHertz):
    """v2i kernel with an elastic/nonlocal collision-field ensemble.

    The applied voltage still determines the supply and collector energy.  A
    Gaussian ensemble is used only while advancing the inelastic collision
    history, approximating the distribution of field progress and path length
    produced by elastic angular transport.  Its width decays as the applied
    accelerating field becomes dominant.
    """

    collision_ensemble_order = 5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raw_collision_field_spread_v2j = nn.Parameter(
            torch.tensor(_inverse_bounded(6.0, 0.0, 12.0))
        )
        self.raw_collision_spread_decay_v2j = nn.Parameter(
            torch.tensor(_inverse_bounded(30.0, 8.0, 80.0))
        )
        nodes, weights = np.polynomial.hermite.hermgauss(self.collision_ensemble_order)
        self.register_buffer(
            "collision_field_nodes",
            torch.tensor(math.sqrt(2.0) * nodes, dtype=torch.float32),
        )
        self.register_buffer(
            "collision_field_weights",
            torch.tensor(weights / math.sqrt(math.pi), dtype=torch.float32),
        )

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        params = super().physical_parameters()
        params.update(
            {
                "collision_field_spread_eV": self.bounded(
                    self.raw_collision_field_spread_v2j, 0.0, 12.0
                ),
                "collision_spread_decay_V": self.bounded(
                    self.raw_collision_spread_decay_v2j, 8.0, 80.0
                ),
            }
        )
        return params

    def forward(self, va: torch.Tensor, vr: torch.Tensor) -> torch.Tensor:
        va = va.reshape(-1).float()
        vr = vr.reshape(-1).float()
        if va.shape != vr.shape:
            raise ValueError("Va and Vr must have the same shape")

        params = self.physical_parameters()
        unique_va, inverse = torch.unique(va, sorted=True, return_inverse=True)
        spread = params["collision_field_spread_eV"] * torch.exp(
            -unique_va / params["collision_spread_decay_V"]
        )
        transmission = torch.zeros_like(va)
        for node, weight in zip(
            self.collision_field_nodes.unbind(),
            self.collision_field_weights.unbind(),
        ):
            collision_va = torch.relu(unique_va + node * spread)
            collision_mass, last_mass = self.collision_state_moments(collision_va)
            collision_mass = collision_mass[inverse]
            last_mass = last_mass[inverse]
            last_position = last_mass / torch.clamp(collision_mass, min=1.0e-12)
            collector_gate = self.collector_state_transmission(va, vr, last_position)
            state_transmission = torch.sum(collision_mass * collector_gate, dim=2)
            transmission = transmission + weight * torch.sum(
                state_transmission * self.thermal_weights[None, :], dim=1
            )

        reduced_voltage = (
            torch.relu(params["acceleration_scale"] * va) / params["supply_scale_V"]
        )
        space_charge = torch.pow(reduced_voltage, params["supply_exponent"])
        supply = (
            params["current_limit_uA"] * space_charge / (1.0 + space_charge)
        )
        return torch.clamp(supply * transmission, min=0.0)

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "model_class": type(self).__name__,
                "model_version": "v2j-collision-field-ensemble",
                "collision_field_closure": (
                    "five-point Gaussian ensemble applied only to collision-history advance"
                ),
                "collision_field_spread_law": "sigma(Va)=sigma0*exp(-Va/Vdecay)",
                "additional_trainable_parameters_vs_v2a": 4,
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in self.parameters() if parameter.requires_grad
                ),
            }
        )
        return report

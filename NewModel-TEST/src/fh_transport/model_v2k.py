from __future__ import annotations

import math
from typing import Any

import torch

from .model_v2j import CollisionFieldEnsembleFranckHertz


class TruncatedGaussianGateFranckHertz(CollisionFieldEnsembleFranckHertz):
    """v2j transport with an energy-conserving half-Gaussian collector gate.

    Transmission is exactly zero when axial energy is below the collector
    barrier.  Above the barrier, the accepted fraction is the cumulative
    half-Gaussian associated with the fitted energy-selection width.
    """

    def collector_energy_gate(
        self,
        axial_energy: torch.Tensor,
        barrier: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        excess = torch.relu(axial_energy - barrier[:, None, None])
        return torch.erf(excess / (math.sqrt(2.0) * sigma))

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "model_class": type(self).__name__,
                "model_version": "v2k-truncated-gaussian-gate",
                "collector_energy_gate": (
                    "erf(max(E_axial-E_barrier,0)/(sqrt(2)*sigma_E))"
                ),
                "subthreshold_collector_transmission": "exactly zero",
                "collision_field_closure": (
                    "five-point Gaussian ensemble applied only to collision-history advance"
                ),
                "additional_trainable_parameters_vs_v2a": 4,
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                ),
            }
        )
        return report

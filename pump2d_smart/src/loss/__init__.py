"""Loss Module for 2D Pump Surrogate Modeling."""

from src.loss.losses import CombinedLoss, RelL2Loss
from src.loss.physics_losses import (
    ECOTWINPhysicsLoss,
    divergence_free_loss,
    flux_continuity_loss,
    outlet_pressure_loss,
    wall_bc_loss,
)

__all__ = [
    "CombinedLoss",
    "ECOTWINPhysicsLoss",
    "RelL2Loss",
    "divergence_free_loss",
    "flux_continuity_loss",
    "outlet_pressure_loss",
    "wall_bc_loss",
]

"""Loss Functions and Composite Criteria for 2D Pump Surrogate Modeling.

Includes Relative L2 loss (RelL2Loss) and configurable CombinedLoss for
surface and volume CFD prediction fields.
"""

from typing import Any

import torch


class RelL2Loss:
    """Relative L2 loss for continuous fields and PDE surrogates."""

    def __init__(
        self,
        dim: int = -2,
        eps: float = 1e-5,
        reduction: str = "sum",
        reduce_all: bool = True,
    ) -> None:
        """Initializes RelL2Loss.

        Args:
            dim: Dimension along which to calculate norms.
            eps: Epsilon to avoid division by zero.
            reduction: 'sum' or 'mean' reduction for norms.
            reduce_all: If True, returns scalar mean over all dimensions.
        """
        self.dim = dim
        self.eps = eps
        self.reduction = reduction
        self.reduce_all = reduce_all

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes relative L2 loss between predictions and ground truth.

        Args:
            y_hat: Predicted field tensor.
            y: Ground truth field tensor.

        Returns:
            Relative L2 loss tensor.
        """
        if y_hat.shape != y.shape:
            raise ValueError(f"Shape mismatch: y_hat {y_hat.shape} vs y {y.shape}")

        reduce_fn = torch.mean if self.reduction == "mean" else torch.sum

        y_norm = reduce_fn((y**2), dim=self.dim)
        mask = y_norm < self.eps
        y_norm = torch.where(mask, torch.tensor(self.eps, device=y.device), y_norm)
        diff = reduce_fn((y_hat - y) ** 2, dim=self.dim)
        diff = diff / y_norm

        if self.reduce_all:
            return diff.sqrt().mean()
        return diff.sqrt()


class CombinedLoss:
    """Computes combined loss across surface and volume fields."""

    def __init__(self, loss_fn: Any, fields: dict[str, list[str]]) -> None:
        """Initializes CombinedLoss.

        Args:
            loss_fn: Base loss function (e.g. torch.nn.MSELoss, L1Loss, RelL2Loss).
            fields: Dictionary specifying active surface and volume field names.
        """
        self.loss_fn = loss_fn
        self.fields = fields

    def __call__(
        self,
        y_hat_surf: torch.Tensor | None,
        y_hat_vol: torch.Tensor,
        y_surf: torch.Tensor | None,
        y_vol: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the combined loss.

        Args:
            y_hat_surf: Predicted surface tensor.
            y_hat_vol: Predicted volume tensor.
            y_surf: Ground truth surface tensor.
            y_vol: Ground truth volume tensor.

        Returns:
            Scalar combined loss tensor.
        """
        if not self.fields.get("surface") or y_hat_surf is None or y_surf is None:
            return self.loss_fn(y_hat_vol, y_vol)

        loss_vol = self.loss_fn(y_hat_vol, y_vol)
        loss_surf = self.loss_fn(y_hat_surf, y_surf)
        return loss_vol + loss_surf

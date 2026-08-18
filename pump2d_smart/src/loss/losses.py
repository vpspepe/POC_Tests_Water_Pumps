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
    """Computes combined loss across surface and volume fields with component tracking and gradient norm calculation."""

    def __init__(self, loss_fn: Any, fields: dict[str, list[str]]) -> None:
        """Initializes CombinedLoss.

        Args:
            loss_fn: Base loss function (e.g. torch.nn.MSELoss, L1Loss, RelL2Loss).
            fields: Dictionary specifying active surface and volume field names.
        """
        self.loss_fn = loss_fn
        self.fields = fields

    def forward_with_components(
        self,
        y_hat_surf: torch.Tensor | None,
        y_hat_vol: torch.Tensor,
        y_surf: torch.Tensor | None,
        y_vol: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Computes total loss and breaks down individual per-channel field losses.

        Args:
            y_hat_surf: Predicted surface tensor.
            y_hat_vol: Predicted volume tensor.
            y_surf: Ground truth surface tensor.
            y_vol: Ground truth volume tensor.

        Returns:
            Tuple of [total_loss_tensor, components_dict].
        """
        components: dict[str, torch.Tensor] = {}

        # Individual field component losses
        vol_fields = self.fields.get("volume", ["pressure", "velocity_x", "velocity_y"])
        for i, name in enumerate(vol_fields):
            if i < y_hat_vol.shape[-1]:
                components[name] = self.loss_fn(y_hat_vol[..., i], y_vol[..., i])

        loss_vol = self.loss_fn(y_hat_vol, y_vol)

        if self.fields.get("surface") and y_hat_surf is not None and y_surf is not None:
            loss_surf = self.loss_fn(y_hat_surf, y_surf)
            components["surface_pressure"] = loss_surf
            total_loss = loss_vol + loss_surf
        else:
            total_loss = loss_vol

        components["total_data"] = total_loss
        return total_loss, components

    def compute_gradient_norms(
        self,
        components: dict[str, torch.Tensor],
        model_parameters: Any,
    ) -> dict[str, float]:
        """Computes Euclidean gradient norm ||d(L_k)/d(theta)||_2 for each loss component.

        Args:
            components: Dictionary mapping component names to scalar loss tensors.
            model_parameters: Iterable of model parameters.

        Returns:
            Dictionary mapping grad_norm_<component> to float magnitude.
        """
        params_list = [p for p in model_parameters if p.requires_grad]
        grad_norms: dict[str, float] = {}

        for name, comp_loss in components.items():
            if (
                comp_loss is None
                or not isinstance(comp_loss, torch.Tensor)
                or not comp_loss.requires_grad
            ):
                continue

            grads = torch.autograd.grad(
                comp_loss,
                params_list,
                retain_graph=True,
                allow_unused=True,
            )
            total_sq = sum(
                (g.detach() ** 2).sum().item() for g in grads if g is not None
            )
            grad_norms[f"grad_norm_{name}"] = float(
                torch.tensor(total_sq).sqrt().item()
            )

        return grad_norms

    def __call__(
        self,
        y_hat_surf: torch.Tensor | None,
        y_hat_vol: torch.Tensor,
        y_surf: torch.Tensor | None,
        y_vol: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the scalar combined loss."""
        total_loss, _ = self.forward_with_components(
            y_hat_surf, y_hat_vol, y_surf, y_vol
        )
        return total_loss

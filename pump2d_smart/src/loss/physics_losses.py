"""ECOTWIN Physics-Informed Loss Functions for 2D Pump Neural Surrogates.

Implements differentiable conservation laws and boundary conditions:
1. Divergence-Free Continuity Loss (L_mass): nabla . u = 0
2. Wall Boundary Condition Loss (L_BC_wall): No-slip on stator, rotating on rotor (Omega x r)
3. Global Mass Flux Continuity Loss (L_flux): Integral(u . n dl) matching target Q_in
4. Outlet Pressure Boundary Condition Loss (L_BC_out): Anchoring outlet static pressure

Note on Velocity Frame Convention:
Velocity u = (u_x, u_y) is expressed in the absolute reference frame across the entire domain.
In the rotor (MRF) zone, moving walls satisfy u_wall = Omega x r = (-Omega * y, Omega * x).
"""

import torch
from torch import nn


def divergence_free_loss(
    u_x: torch.Tensor,
    u_y: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Computes 2D divergence-free continuity loss: || d(u_x)/dx + d(u_y)/dy ||^2.

    Args:
        u_x: Predicted X-velocity component tensor.
        u_y: Predicted Y-velocity component tensor.
        coords: Continuous spatial coordinates tensor (..., 2) with requires_grad=True.

    Returns:
        torch.Tensor: Scalar Mean Squared Error divergence penalty.
    """
    if not coords.requires_grad:
        raise ValueError(
            "coords tensor must have requires_grad=True to compute divergence."
        )

    grad_ux = torch.autograd.grad(
        u_x.sum(),
        coords,
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_uy = torch.autograd.grad(
        u_y.sum(),
        coords,
        create_graph=True,
        retain_graph=True,
    )[0]

    du_x_dx = grad_ux[..., 0]
    du_y_dy = grad_uy[..., 1]

    div = du_x_dx + du_y_dy
    return torch.mean(div**2)


def wall_bc_loss(
    u_pred: torch.Tensor,
    wall_coords: torch.Tensor,
    mrf_mask: torch.Tensor,
    omega: float | torch.Tensor,
) -> torch.Tensor:
    """Computes wall boundary condition loss across stator and rotor walls.

    - Stator wall (mrf_mask = 0): u = 0 (no-slip in absolute frame)
    - Rotor wall  (mrf_mask = 1): u = Omega x r = (-Omega * y, Omega * x)

    Args:
        u_pred: Predicted velocity field tensor of shape (..., 2) [u_x, u_y].
        wall_coords: Coordinates on the wall boundary of shape (..., 2) [x, y].
        mrf_mask: Binary indicator tensor (1 = rotor, 0 = stator).
        omega: Impeller rotational speed in rad/s (positive = counterclockwise).

    Returns:
        torch.Tensor: Mean Squared Error wall velocity boundary loss.
    """
    x = wall_coords[..., 0]
    y = wall_coords[..., 1]

    u_rotor_x = -omega * y
    u_rotor_y = omega * x
    u_rotor = torch.stack([u_rotor_x, u_rotor_y], dim=-1)

    if mrf_mask.ndim == u_rotor.ndim - 1:
        mrf_mask = mrf_mask.unsqueeze(-1)

    u_target = mrf_mask * u_rotor
    return torch.mean((u_pred - u_target) ** 2)


def flux_continuity_loss(
    u_inlet: torch.Tensor,
    inlet_normals: torch.Tensor,
    inlet_weights: torch.Tensor,
    u_outlet: torch.Tensor,
    outlet_normals: torch.Tensor,
    outlet_weights: torch.Tensor,
    q_in: float | torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes global mass flux continuity loss across inlet and outlet boundaries.

    L_flux = (Flux_in - Q_in)^2 + (Flux_in + Flux_out)^2

    Where Flux = Integral(u . n dl) using boundary quadrature weights.

    Args:
        u_inlet: Predicted velocity at inlet points of shape (N_in, 2).
        inlet_normals: Unit outward normal vectors at inlet of shape (N_in, 2).
        inlet_weights: Quadrature arc-length weights along inlet curve of shape (N_in,).
        u_outlet: Predicted velocity at outlet points of shape (N_out, 2).
        outlet_normals: Unit outward normal vectors at outlet of shape (N_out, 2).
        outlet_weights: Quadrature arc-length weights along outlet curve of shape (N_out,).
        q_in: Prescribed volumetric flow rate.

    Returns:
        Tuple of [scalar_loss_tensor, details_dict].
    """
    flux_density_in = torch.sum(u_inlet * inlet_normals, dim=-1)
    flux_in = torch.sum(flux_density_in * inlet_weights)

    flux_density_out = torch.sum(u_outlet * outlet_normals, dim=-1)
    flux_out = torch.sum(flux_density_out * outlet_weights)

    term_inlet_match = (flux_in - q_in) ** 2
    term_flux_balance = (flux_in + flux_out) ** 2
    total_loss = term_inlet_match + term_flux_balance

    details = {
        "flux_in": flux_in,
        "flux_out": flux_out,
        "loss_inlet_target": term_inlet_match,
        "loss_flux_balance": term_flux_balance,
    }
    return total_loss, details


def outlet_pressure_loss(
    p_outlet_pred: torch.Tensor,
    p_out_target: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Computes outlet pressure boundary condition anchoring loss: || p_pred - p_out ||^2.

    Args:
        p_outlet_pred: Predicted static pressure at outlet boundary points.
        p_out_target: Target reference pressure at outlet (default: 0.0 Pa).

    Returns:
        torch.Tensor: Scalar Mean Squared Error pressure anchor loss.
    """
    if isinstance(p_out_target, (int, float)):
        p_out_target = torch.tensor(
            p_out_target, device=p_outlet_pred.device, dtype=p_outlet_pred.dtype
        )
    return torch.mean((p_outlet_pred - p_out_target) ** 2)


class ECOTWINPhysicsLoss(nn.Module):
    """Composite Physics-Informed Loss Manager for ECOTWIN 2D Pump Models."""

    def __init__(
        self,
        weight_mass: float = 0.0,
        weight_wall: float = 0.0,
        weight_flux: float = 0.0,
        weight_outlet_p: float = 0.0,
    ) -> None:
        """Initializes ECOTWINPhysicsLoss with loss term weights.

        Args:
            weight_mass: Weight for divergence-free mass conservation (lambda_mass).
            weight_wall: Weight for wall boundary condition (lambda_BC_wall).
            weight_flux: Weight for global flux continuity (lambda_flux).
            weight_outlet_p: Weight for outlet pressure anchoring (lambda_BC_out).
        """
        super().__init__()
        self.weight_mass = weight_mass
        self.weight_wall = weight_wall
        self.weight_flux = weight_flux
        self.weight_outlet_p = weight_outlet_p

    def forward(
        self,
        vol_u_x: torch.Tensor | None = None,
        vol_u_y: torch.Tensor | None = None,
        vol_coords: torch.Tensor | None = None,
        wall_u_pred: torch.Tensor | None = None,
        wall_coords: torch.Tensor | None = None,
        wall_mrf_mask: torch.Tensor | None = None,
        omega: float | torch.Tensor = 0.0,
        inlet_u_pred: torch.Tensor | None = None,
        inlet_normals: torch.Tensor | None = None,
        inlet_weights: torch.Tensor | None = None,
        outlet_u_pred: torch.Tensor | None = None,
        outlet_normals: torch.Tensor | None = None,
        outlet_weights: torch.Tensor | None = None,
        q_in: float | torch.Tensor = 0.0,
        outlet_p_pred: torch.Tensor | None = None,
        outlet_p_target: float | torch.Tensor = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes weighted composite physics loss and returns breakdown.

        Returns:
            Tuple of [total_physics_loss_tensor, metrics_dictionary].
        """
        total_loss = torch.tensor(
            0.0, device=vol_coords.device if vol_coords is not None else None
        )
        metrics: dict[str, float] = {}

        # 1. Divergence-Free Continuity Loss (L_mass)
        if (
            self.weight_mass > 0.0
            and vol_u_x is not None
            and vol_u_y is not None
            and vol_coords is not None
        ):
            l_mass = divergence_free_loss(vol_u_x, vol_u_y, vol_coords)
            total_loss = total_loss + self.weight_mass * l_mass
            metrics["loss_physics_mass"] = float(l_mass.item())

        # 2. Wall Boundary Condition Loss (L_BC_wall)
        if (
            self.weight_wall > 0.0
            and wall_u_pred is not None
            and wall_coords is not None
            and wall_mrf_mask is not None
        ):
            l_wall = wall_bc_loss(wall_u_pred, wall_coords, wall_mrf_mask, omega)
            total_loss = total_loss + self.weight_wall * l_wall
            metrics["loss_physics_wall_bc"] = float(l_wall.item())

        # 3. Global Mass Flux Continuity Loss (L_flux)
        if (
            self.weight_flux > 0.0
            and inlet_u_pred is not None
            and inlet_normals is not None
            and inlet_weights is not None
            and outlet_u_pred is not None
            and outlet_normals is not None
            and outlet_weights is not None
        ):
            l_flux, _ = flux_continuity_loss(
                inlet_u_pred,
                inlet_normals,
                inlet_weights,
                outlet_u_pred,
                outlet_normals,
                outlet_weights,
                q_in,
            )
            total_loss = total_loss + self.weight_flux * l_flux
            metrics["loss_physics_flux"] = float(l_flux.item())

        # 4. Outlet Pressure Boundary Condition Loss (L_BC_out)
        if self.weight_outlet_p > 0.0 and outlet_p_pred is not None:
            l_out_p = outlet_pressure_loss(outlet_p_pred, outlet_p_target)
            total_loss = total_loss + self.weight_outlet_p * l_out_p
            metrics["loss_physics_outlet_p"] = float(l_out_p.item())

        return total_loss, metrics

"""Feature Manager Module for SMART 2D Pump Inputs and Targets.

Encapsulates feature routing, selective Signed Distance Field (SDF) slicing,
and dynamic surface prediction toggles for clean object-oriented architecture.
"""

from typing import Any

import torch


class FeatureManager:
    """Manages active geometric features, SDF channels, and prediction modes."""

    def __init__(
        self,
        predict_surface: bool = False,
        surface_channels: int = 0,
        volume_channels: int = 3,
        use_sdf_blades: bool = True,
        use_sdf_mrf: bool = True,
    ) -> None:
        """Initializes the feature manager and configures active feature slices.

        Args:
            predict_surface: If True, queries blade surface wall pressure.
            surface_channels: Number of surface output channels (0 if disabled).
            volume_channels: Number of volume output channels (pressure, vx, vy).
            use_sdf_blades: If True, passes SDF to impeller blades.
            use_sdf_mrf: If True, passes SDF to MRF rotating interface circle.
        """
        self.predict_surface: bool = predict_surface
        self.surface_channels: int = surface_channels if predict_surface else 0
        self.volume_channels: int = volume_channels

        self.use_sdf_blades: bool = use_sdf_blades
        self.use_sdf_mrf: bool = use_sdf_mrf

        # Determine active SDF channel indices
        self.active_sdf_indices: list[int] = []
        if self.use_sdf_blades:
            self.active_sdf_indices.append(0)  # Channel 0: SDF to blades
        if self.use_sdf_mrf:
            self.active_sdf_indices.append(1)  # Channel 1: SDF to MRF

        self.extra_query_dim: int = len(self.active_sdf_indices)

    def extract_extra_features(self, volume_extra: torch.Tensor) -> torch.Tensor | None:
        """Extracts and slices the configured SDF feature channels.

        Args:
            volume_extra: Pre-computed SDF tensor of shape (B, N_vol, 2).

        Returns:
            Sliced tensor of shape (B, N_vol, extra_query_dim), or None if extra_query_dim == 0.
        """
        if self.extra_query_dim == 0:
            return None
        return volume_extra[:, :, self.active_sdf_indices]

    def prepare_surface_queries(
        self,
        batch: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepares surface coordinates and target tensors based on active mode.

        Args:
            batch: Collated DataLoader batch dictionary.
            device: PyTorch hardware execution device.

        Returns:
            Tuple of [surface_coords, surface_data] tensors matching active channels.
        """
        batch_size = batch["geometry"].shape[0]

        if not self.predict_surface or self.surface_channels == 0:
            surf_c = torch.zeros((batch_size, 0, 2), dtype=torch.float32, device=device)
            surf_d = torch.zeros((batch_size, 0, 0), dtype=torch.float32, device=device)
        else:
            surf_c = batch["surface_coords"].to(device)
            surf_d = batch["surface_data"].to(device)

        return surf_c, surf_d

"""Geometry Processing Module for 2D Pump Simulations (Pure PyTorch).

Provides classes and PyTorch GPU kernels to extract boundaries, open the discharge
outlet, mask valid fluid regions, and calculate Signed Distance Fields (SDFs).
"""

import os

import numpy as np
import pyvista as pv
import torch


def compute_sdf_pytorch(
    query_pts: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
    batch_size: int = 2000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Calculates Signed Distance Field (SDF) using pure PyTorch tensors on GPU.

    Args:
        query_pts: Coordinates tensor of shape (N, 2).
        seg_starts: Segment start coordinates tensor of shape (M, 2).
        seg_ends: Segment end coordinates tensor of shape (M, 2).
        batch_size: Evaluation chunk size for GPU VRAM stability.
        device: Hardware device ("cuda" or "cpu").

    Returns:
        SDF values tensor of shape (N,) (negative inside, positive outside).
    """
    n_pts = query_pts.shape[0]
    assert batch_size > 0 and batch_size <= n_pts, (
        f"Batch Size should be between 1 and {n_pts}"
    )
    query_pts = query_pts.to(device)
    seg_starts = seg_starts.to(device)
    seg_ends = seg_ends.to(device)

    sdf_results = []

    for i in range(0, n_pts, batch_size):
        P = query_pts[i : i + batch_size, None, :]  # (B, 1, 2)
        A = seg_starts[None, :, :]  # (1, M, 2)
        B = seg_ends[None, :, :]  # (1, M, 2)

        # Point-to-segment distance calculation
        AB = B - A
        AP = P - A

        ab_sq = torch.sum(AB**2, dim=2)
        ab_sq = torch.where(ab_sq == 0, 1e-12, ab_sq)

        t = torch.sum(AP * AB, dim=2) / ab_sq
        t = torch.clamp(t, 0.0, 1.0)[:, :, None]

        closest_pts = A + t * AB
        dists = torch.linalg.norm(P - closest_pts, dim=2)
        abs_distance = torch.min(dists, dim=1).values

        # Ray-casting inside/outside check (Even-Odd Rule)
        x, y = P[:, 0, 0:1], P[:, 0, 1:2]
        x1, y1 = seg_starts[None, :, 0], seg_starts[None, :, 1]
        x2, y2 = seg_ends[None, :, 0], seg_ends[None, :, 1]

        intersect_y = (y1 > y) != (y2 > y)
        dy = y2 - y1
        dy_safe = torch.where(dy == 0, 1e-12, dy)
        intersect_x = (x < (x2 - x1) * (y - y1) / dy_safe + x1) & (y1 != y2)

        is_inside = torch.sum(intersect_y & intersect_x, dim=1) % 2 == 1
        sdf_chunk = torch.where(is_inside, -abs_distance, abs_distance)
        sdf_results.append(sdf_chunk)

    return torch.cat(sdf_results, dim=0)


def is_inside_polygon_pytorch(
    query_pts: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_ends: torch.Tensor,
    batch_size: int = 2000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """Checks whether 2D query points reside inside a closed polygon using GPU ray-casting.

    Args:
        query_pts: Coordinates tensor of shape (N, 2).
        seg_starts: Segment start coordinates tensor of shape (M, 2).
        seg_ends: Segment end coordinates tensor of shape (M, 2).
        batch_size: Evaluation chunk size for GPU VRAM stability.
        device: Hardware device ("cuda" or "cpu").

    Returns:
        Boolean PyTorch Tensor of shape (N,) (True if inside, False if outside).
    """
    n_pts = query_pts.shape[0]
    query_pts = query_pts.to(device)
    seg_starts = seg_starts.to(device)
    seg_ends = seg_ends.to(device)

    x1, y1 = seg_starts[None, :, 0], seg_starts[None, :, 1]
    x2, y2 = seg_ends[None, :, 0], seg_ends[None, :, 1]
    dy = y2 - y1
    dy_safe = torch.where(dy == 0, 1e-12, dy)

    inside_results = []
    for i in range(0, n_pts, batch_size):
        P = query_pts[i : i + batch_size]  # (B, 2)
        x, y = P[:, 0:1], P[:, 1:2]  # (B, 1)

        intersect_y = (y1 > y) != (y2 > y)
        intersect_x = (x < (x2 - x1) * (y - y1) / dy_safe + x1) & (y1 != y2)

        is_inside = torch.sum(intersect_y & intersect_x, dim=1) % 2 == 1
        inside_results.append(is_inside)

    return torch.cat(inside_results, dim=0)


class Pump2DGeometryProcessor:
    """Processor to load pump VTUs, open casing, and compute fluid mask/SDFs in PyTorch."""

    def __init__(
        self,
        main_vtu_path: str,
        outlet_vtu_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initializes the processor and starts boundary parsing.

        Args:
            main_vtu_path: Absolute path to main grid VTU.
            outlet_vtu_path: Absolute path to outlet grid VTU.
            device: Computing device for tensor kernels ("cuda" or "cpu").
        """
        self.main_vtu_path: str = main_vtu_path
        self.outlet_vtu_path: str = outlet_vtu_path
        self.device: str = device

        if not os.path.exists(main_vtu_path):
            raise FileNotFoundError(f"Main VTU not found at: {main_vtu_path}")
        if not os.path.exists(outlet_vtu_path):
            raise FileNotFoundError(f"Outlet VTU not found at: {outlet_vtu_path}")

        self.grid: pv.UnstructuredGrid = pv.read(main_vtu_path)
        self.outlet_grid: pv.UnstructuredGrid = pv.read(outlet_vtu_path)

        boundary = self.grid.extract_feature_edges(boundary_edges=True)
        self.bodies = boundary.connectivity().split_bodies()

        self.starts_dict: dict[str, torch.Tensor] = {}
        self.ends_dict: dict[str, torch.Tensor] = {}
        self.casing_open_mesh: pv.PolyData = None

        self._process_boundaries()

    def _clean_circular_boundary(
        self, body: pv.UnstructuredGrid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cleans duplicate concentric edges in circular boundaries by polar sorting."""
        poly = (
            body.extract_surface(algorithm="dataset_surface")
            if not isinstance(body, pv.PolyData)
            else body
        )
        pts = np.unique(np.round(poly.points[:, :2], 6), axis=0)

        xc, yc = pts.mean(axis=0)[0], pts.mean(axis=0)[1]
        angles = np.arctan2(pts[:, 1] - yc, pts[:, 0] - xc)
        sorted_pts = pts[np.argsort(angles)]

        starts = torch.tensor(sorted_pts, dtype=torch.float32)
        ends = torch.tensor(np.roll(sorted_pts, -1, axis=0), dtype=torch.float32)
        return starts, ends

    def _process_boundaries(self) -> None:
        """Extracts and cleans casing, blades, MRF, and inlet boundaries."""
        print("Processing pump boundaries...")

        # 1. Closed Casing (Body 0)
        casing_poly = self.bodies[0].extract_surface(algorithm="dataset_surface")
        pts_casing = casing_poly.points[:, :2]
        lines_casing = casing_poly.lines.reshape(-1, 3)[:, 1:]
        self.starts_dict["casing_closed"] = torch.tensor(
            pts_casing[lines_casing[:, 0]], dtype=torch.float32
        )
        self.ends_dict["casing_closed"] = torch.tensor(
            pts_casing[lines_casing[:, 1]], dtype=torch.float32
        )

        # 2. Open Casing (remove outlet segments)
        outlet_pts = self.outlet_grid.points[:, :2]
        # Vectorized Euclidean distances between casing points (N, 1, 2) and outlet points (1, M, 2)
        dists = np.linalg.norm(pts_casing[:, None, :] - outlet_pts[None, :, :], axis=-1)
        remove_indices = np.where(dists.min(axis=1) < 1e-5)[0].tolist()

        casing_mesh = self.bodies[0].extract_surface(algorithm=None)
        open_casing_mesh = casing_mesh.remove_cells(remove_indices)
        open_casing_poly = open_casing_mesh.extract_surface(algorithm="dataset_surface")
        pts_casing_open = open_casing_poly.points[:, :2]
        lines_casing_open = open_casing_poly.lines.reshape(-1, 3)[:, 1:]

        self.starts_dict["casing_open"] = torch.tensor(
            pts_casing_open[lines_casing_open[:, 0]], dtype=torch.float32
        )
        self.ends_dict["casing_open"] = torch.tensor(
            pts_casing_open[lines_casing_open[:, 1]], dtype=torch.float32
        )

        pts_3d = np.column_stack([pts_casing_open, np.zeros(len(pts_casing_open))])
        self.casing_open_mesh = pv.PolyData(pts_3d)

        # 3. Blades (Bodies 1-7) - raw segment extraction without sorting
        blades_starts = []
        blades_ends = []
        for i in range(1, 8):
            blade_poly = self.bodies[i].extract_surface(algorithm="dataset_surface")
            pts = blade_poly.points[:, :2]
            lines = blade_poly.lines.reshape(-1, 3)[:, 1:]
            blades_starts.append(pts[lines[:, 0]])
            blades_ends.append(pts[lines[:, 1]])

        self.starts_dict["blades"] = torch.tensor(
            np.vstack(blades_starts), dtype=torch.float32
        )
        self.ends_dict["blades"] = torch.tensor(
            np.vstack(blades_ends), dtype=torch.float32
        )

        # 4. MRF Interface (Body 8) - polar sorting
        starts8, ends8 = self._clean_circular_boundary(self.bodies[8])
        self.starts_dict["mrf"] = starts8
        self.ends_dict["mrf"] = ends8

        # 5. Inlet Hub (Body 10) - polar sorting
        starts10, ends10 = self._clean_circular_boundary(self.bodies[10])
        self.starts_dict["inlet"] = starts10
        self.ends_dict["inlet"] = ends10
        print("Boundary extraction completed successfully.")

    def get_full_pump_geometry(self) -> np.ndarray:
        """Extracts and combines all physical boundaries (casing open, 7 blades, inlet).

        Returns:
            Numpy array of shape (N_geom, 2) representing the full pump geometry.
        """
        pts_casing_open = self.starts_dict["casing_open"].cpu().numpy()
        pts_blades = self.starts_dict["blades"].cpu().numpy()
        pts_inlet = self.starts_dict["inlet"].cpu().numpy()

        # Combine all physical boundaries: Casing (open) + Blades (1-7) + Inlet
        return np.vstack([pts_casing_open, pts_blades, pts_inlet])

    def get_full_surface_mesh(self) -> pv.PolyData:
        """Merges all pump surface meshes (casing open, 7 blades, inlet) into a single PolyData."""
        full_pts = self.get_full_pump_geometry()
        pts_3d = np.column_stack([full_pts, np.zeros(len(full_pts))])
        return pv.PolyData(pts_3d)

    def compute_sdfs(
        self, query_pts: torch.Tensor, batch_size: int = 2000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes Signed Distance Fields for blades and MRF interfaces in PyTorch.

        Args:
            query_pts: Coordinates tensor of shape (N, 2) to evaluate.
            batch_size: Evaluation chunk size for GPU VRAM stability.

        Returns:
            Tuple of [SDF_MRF, SDF_blades] PyTorch Tensors of shape (N,).
        """
        # Compute SDF to MRF circle
        mrf_starts = self.starts_dict["mrf"]
        mrf_ends = self.ends_dict["mrf"]
        print("Computing PyTorch SDF for MRF interface...")
        sdf_mrf = compute_sdf_pytorch(
            query_pts, mrf_starts, mrf_ends, batch_size, self.device
        )

        # Compute SDF to Blades
        blades_starts = self.starts_dict["blades"]
        blades_ends = self.ends_dict["blades"]
        print("Computing PyTorch SDF for Impeller Blades...")
        sdf_blades = compute_sdf_pytorch(
            query_pts, blades_starts, blades_ends, batch_size, self.device
        )

        return sdf_mrf.cpu(), sdf_blades.cpu()

    def is_valid_fluid_point(
        self, query_pts: torch.Tensor, batch_size: int = 2000
    ) -> torch.Tensor:
        """Determines if query points reside inside the valid fluid domain using PyTorch.

        Args:
            query_pts: Coordinates tensor of shape (N, 2) to evaluate.
            batch_size: Evaluation chunk size for GPU VRAM stability.

        Returns:
            Boolean PyTorch Tensor of shape (N,) where True represents a valid point.
        """
        # 1. Inside closed casing check
        is_inside_casing = is_inside_polygon_pytorch(
            query_pts,
            self.starts_dict["casing_closed"],
            self.ends_dict["casing_closed"],
            batch_size,
            self.device,
        )

        # 2. Outside inlet check
        is_inside_inlet = is_inside_polygon_pytorch(
            query_pts,
            self.starts_dict["inlet"],
            self.ends_dict["inlet"],
            batch_size,
            self.device,
        )

        # 3. Outside blades check
        is_inside_blades = is_inside_polygon_pytorch(
            query_pts,
            self.starts_dict["blades"],
            self.ends_dict["blades"],
            batch_size,
            self.device,
        )

        valid_mask = is_inside_casing & (~is_inside_inlet) & (~is_inside_blades)
        return valid_mask.cpu()

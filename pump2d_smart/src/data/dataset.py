"""Dataset Pipeline Module for 2D Pump Simulations.

Provides PyTorch Dataset implementations for loading 2D COMSOL pump meshes,
interpolating flow fields, pre-computing SDF features, and normalizing outputs.
"""

import json
import os
import re
from os.path import join as pjoin
from typing import Any, TypedDict

import numpy as np
import pyvista as pv
import torch
from scipy.spatial import KDTree
from src.data.utils import kennard_stone_split
from src.geometry.processor import Pump2DGeometryProcessor
from src.training.calculate_pump_heads import compute_pump_head, extract_unique_edges
from torch.utils.data import Dataset


class Pump2DSample(TypedDict):
    """Structured container for a single normalized pump simulation sample."""

    geometry: torch.Tensor
    surface_coords: torch.Tensor
    surface_data: torch.Tensor
    volume_coords: torch.Tensor
    volume_extra: torch.Tensor
    volume_data: torch.Tensor
    params: torch.Tensor


class Pump2DDataset(Dataset[Pump2DSample]):
    """PyTorch Dataset loading 2D COMSOL pump simulations with boundary head filtering."""

    def __init__(
        self,
        main_vtu_path: str,
        outlet_vtu_path: str,
        cache_dir: str = "./cache",
        if_test: bool = False,
        test_split: float | None = None,
        sparse_hq_split: bool = True,
        n_train: int | None = None,
        seed: int = 42,
    ) -> None:
        """Initializes the dataset and loads/caches processed data.

        Args:
            main_vtu_path: Absolute path to the main grid VTU.
            outlet_vtu_path: Absolute path to the outlet grid VTU.
            cache_dir: Folder to save/load processed .npz arrays.
            if_test: If True, load test samples; otherwise load train samples.
            test_split: Fraction of dataset allocated to validation/test (mutually exclusive with n_train).
            sparse_hq_split: If True, uses interleaved sparse sampling per RPM curve.
            n_train: Exact number of training samples selected with Kennard-Stone (mutually exclusive with test_split).
            seed: Random seed for splitting dataset.
        """
        if n_train is not None and test_split is not None:
            raise ValueError(
                "Specify either 'n_train' (integer count) or 'test_split' (fraction), not both!"
            )
        if n_train is None and test_split is None:
            test_split = 0.5  # Default 50/50 split if neither is provided

        self.main_vtu_path: str = main_vtu_path
        self.outlet_vtu_path: str = outlet_vtu_path
        self.cache_dir: str = cache_dir
        self.if_test: bool = if_test
        self.test_split: float | None = test_split
        self.sparse_hq_split: bool = sparse_hq_split
        self.n_train: int | None = n_train
        self.seed: int = seed

        self.samples: list[dict[str, np.ndarray]] = []
        self.triangles: np.ndarray = np.empty((0, 3), dtype=np.int32)

        # 1. Load or build cached data
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file: str = pjoin(cache_dir, "pump2d_data_cache.npz")

        if os.path.exists(self.cache_file):
            print(f"Loading cached dataset from: {self.cache_file}")
            self._load_cached_data()
        else:
            print("Pre-processing dataset (this may take a few minutes)...")
            self._build_dataset()

        # 2. Divide training/validation test indices
        self.train_indices, self.test_indices = self._get_split_indices()

        # 3. Select active indices
        self.active_indices: np.ndarray = (
            self.test_indices if if_test else self.train_indices
        )
        split_desc = (
            f"n_train={len(self.train_indices)}, n_val={len(self.test_indices)}"
            if self.n_train is not None
            else f"test_split={self.test_split}"
        )
        print(
            f"Dataset mode: {'TEST' if if_test else 'TRAIN'} | "
            f"Number of active samples: {len(self.active_indices)} (Total: {len(self.samples)} | {split_desc})"
        )

        # 4. Compute standardization parameters
        self._compute_normalization_stats()

    def _get_split_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculates training and testing split indices using the space-filling Kennard-Stone algorithm.

        Returns:
            Tuple of [train_indices, test_indices] as numpy arrays.
        """
        n_samples = len(self.samples)
        if self.test_split == 0.0:
            return np.arange(n_samples), np.arange(n_samples)

        # Standardize params to scale flow rate and RPM equally
        params = np.array([s["params"] for s in self.samples])
        params_mean = np.mean(params, axis=0)
        params_std = np.std(params, axis=0)
        params_std = np.where(params_std == 0.0, 1.0, params_std)
        X = (params - params_mean) / params_std

        if self.n_train is not None and self.n_train > 0:
            train_size = int(self.n_train)
        else:
            train_size = round(n_samples * (1.0 - float(self.test_split)))
        return kennard_stone_split(X, train_size)

    def _build_dataset(self) -> None:
        """Processes raw VTU grids and builds the sample dictionary database."""
        # 1. Initialize geometry boundary processor
        processor, geom_points, surface_mesh, surf_points = self._init_geometry_data()

        # 2. Extract triangulation, volume coordinates, and pre-compute SDFs
        vol_points, vol_extra = self._extract_mesh_coordinates(processor)

        # 3. Load boundary meshes and map inlet/outlet boundary indices
        boundary_data = self._map_boundary_indices(vol_points)

        # 4. Parse COMSOL sweep configurations
        configs = self._parse_cfd_configurations(processor.grid)

        # 5. Loop and evaluate cases, filter negative pump heads
        self._sample_and_filter_cases(
            configs=configs,
            grid=processor.grid,
            surface_mesh=surface_mesh,
            geom_points=geom_points,
            surf_points=surf_points,
            vol_points=vol_points,
            vol_extra=vol_extra,
            boundary_data=boundary_data,
        )

        # 6. Save pre-processed snapshot list to disk cache
        self._save_cached_data()

    def _init_geometry_data(
        self,
    ) -> tuple[Pump2DGeometryProcessor, np.ndarray, pv.PolyData, np.ndarray]:
        """Loads the boundary processor and extracts full pump geometry and full surface mesh.

        Returns:
            Tuple of [processor, full_geom_coords, full_surface_mesh, full_surface_coords].
        """
        processor = Pump2DGeometryProcessor(self.main_vtu_path, self.outlet_vtu_path)

        # Full Pump Geometry Coordinates: Casing (open) + Blades (1-7) + Inlet (821 points)
        geom_points = processor.get_full_pump_geometry()

        # Full Pump Surface Mesh: Casing (open) + Blades (1-7) + Inlet (821 points)
        surface_mesh = processor.get_full_surface_mesh()
        surf_points = surface_mesh.points[:, :2]

        return processor, geom_points, surface_mesh, surf_points

    def _extract_mesh_coordinates(
        self, processor: Pump2DGeometryProcessor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extracts coordinates and pre-computes Signed Distance Fields.

        Args:
            processor: Instantiated Pump2DGeometryProcessor.

        Returns:
            Tuple of [volume_points, volume_extra_sdfs] numpy arrays.
        """
        tri_grid = processor.grid.triangulate()
        cells = tri_grid.cells.reshape(-1, 4)
        self.triangles = cells[:, 1:].astype(np.int32)

        vol_points_tensor = torch.tensor(tri_grid.points[:, :2], dtype=torch.float32)
        vol_points = vol_points_tensor.cpu().numpy()

        # Precompute SDFs for all volume coords
        sdf_mrf, sdf_blades = processor.compute_sdfs(vol_points_tensor, batch_size=2000)
        vol_extra = torch.column_stack([sdf_blades, sdf_mrf]).cpu().numpy()

        return vol_points, vol_extra

    def _map_boundary_indices(self, vol_points: np.ndarray) -> dict[str, Any]:
        """Loads inlet/outlet meshes, computes unit normals, weights, and maps indices.

        Args:
            vol_points: Volume coordinate points array of shape (N, 2).

        Returns:
            Dictionary containing mapped indices, boundary coords, edges, normals, and weights.
        """
        comsol_dir = os.path.dirname(self.main_vtu_path)
        in_mesh = pv.read(pjoin(comsol_dir, "Pump2D_MRFStudy_finer_Inlet.vtu"))
        out_mesh = pv.read(pjoin(comsol_dir, "Pump2D_MRFStudy_finer_Outlet.vtu"))

        pts_in = np.asarray(in_mesh.points, dtype=np.float32)[:, :2]
        pts_out = np.asarray(out_mesh.points, dtype=np.float32)[:, :2]

        e_in = extract_unique_edges(in_mesh)
        e_out = extract_unique_edges(out_mesh)

        # 1. Inlet segment lengths (weights) and outward unit normals
        p0_in = pts_in[e_in[:, 0]]
        p1_in = pts_in[e_in[:, 1]]
        seg_in = p1_in - p0_in
        weights_in = np.linalg.norm(seg_in, axis=1).astype(np.float32)
        raw_norm_in = np.column_stack([seg_in[:, 1], -seg_in[:, 0]]) / np.maximum(
            weights_in[:, None], 1e-12
        )
        norm_in = raw_norm_in.astype(np.float32)

        # 2. Outlet segment lengths (weights) and outward unit normals
        p0_out = pts_out[e_out[:, 0]]
        p1_out = pts_out[e_out[:, 1]]
        seg_out = p1_out - p0_out
        weights_out = np.linalg.norm(seg_out, axis=1).astype(np.float32)
        raw_norm_out = np.column_stack([seg_out[:, 1], -seg_out[:, 0]]) / np.maximum(
            weights_out[:, None], 1e-12
        )
        for k in range(len(raw_norm_out)):
            if raw_norm_out[k, 1] < 0:
                raw_norm_out[k] = -raw_norm_out[k]
        norm_out = raw_norm_out.astype(np.float32)

        # Build KDTree for volume index mapping
        tree = KDTree(vol_points)
        _, idx_in = tree.query(pts_in)
        _, idx_out = tree.query(pts_out)

        boundary_dict = {
            "pts_in": pts_in,
            "pts_out": pts_out,
            "e_in": e_in.astype(np.int64),
            "e_out": e_out.astype(np.int64),
            "norm_in": norm_in,
            "norm_out": norm_out,
            "weights_in": weights_in,
            "weights_out": weights_out,
            "idx_in": idx_in.astype(np.int64),
            "idx_out": idx_out.astype(np.int64),
        }
        self.boundary_data = boundary_dict
        return boundary_dict

    def _parse_cfd_configurations(
        self, grid: pv.UnstructuredGrid
    ) -> list[tuple[int, float, int, str]]:
        """Parses COMSOL operating point parameters from grid data arrays.

        Args:
            grid: PyVista UnstructuredGrid representing the simulation.

        Returns:
            Sorted list of tuples containing [index, flow_rate, rpm, array_key].
        """
        pattern = re.compile(
            r"Pressure_@_(\d+):_Q_in=([\d\.]+)_m\^3/h,_rot_rpm=(\d+)_rpm"
        )
        configs: list[tuple[int, float, int, str]] = []
        for key in grid.point_data:
            match = pattern.match(key)
            if match:
                idx = int(match.group(1))
                q_in = float(match.group(2))
                rpm = int(match.group(3))
                configs.append((idx, q_in, rpm, key))

        # Maintain consistent index sorting
        configs.sort(key=lambda x: x[0])
        return configs

    def _sample_and_filter_cases(
        self,
        configs: list[tuple[int, float, int, str]],
        grid: pv.UnstructuredGrid,
        surface_mesh: pv.PolyData,
        geom_points: np.ndarray,
        surf_points: np.ndarray,
        vol_points: np.ndarray,
        vol_extra: np.ndarray,
        boundary_data: dict[str, Any],
    ) -> None:
        """Processes operating configs, filters negative pump heads, and saves samples.

        Args:
            configs: Parsed configurations.
            grid: Simulation volume grid.
            surface_mesh: Full pump boundary surface mesh (casing + blades + inlet).
            geom_points: Full pump boundary coords.
            surf_points: Full pump surface coords.
            vol_points: Volume coordinates.
            vol_extra: Volume precomputed SDFs.
            boundary_data: KDTree mapping and boundary details.
        """
        # Interpolate variables onto the full pump boundary surface
        print("Interpolating pressure fields onto full pump boundary...")
        sampled_surface = surface_mesh.sample(grid)

        tri_grid = grid.triangulate()
        self.samples = []
        excluded_count = 0

        for idx, q_in, rpm, p_key in configs:
            vx_key = p_key.replace("Pressure_", "Velocity_field,_x_component_")
            vy_key = p_key.replace("Pressure_", "Velocity_field,_y_component_")

            p_vol = tri_grid[p_key]
            vx = tri_grid[vx_key]
            vy = tri_grid[vy_key]

            # Compute pump head using COMSOL integration formula
            h = compute_pump_head(
                p_all=p_vol,
                vx_all=vx,
                vy_all=vy,
                idx_in=boundary_data["idx_in"],
                idx_out=boundary_data["idx_out"],
                pts_in=boundary_data["pts_in"],
                pts_out=boundary_data["pts_out"],
                e_in=boundary_data["e_in"],
                e_out=boundary_data["e_out"],
            )

            # Exclude unphysical backflow / numerical failure cases (H <= 0)
            if np.isnan(h) or h <= 0.0:
                excluded_count += 1
                continue

            vol_data = np.column_stack([p_vol, vx, vy])
            surf_data = sampled_surface[p_key][:, None]

            self.samples.append(
                {
                    "geometry": geom_points.astype(np.float32),
                    "surface_coords": surf_points.astype(np.float32),
                    "surface_data": surf_data.astype(np.float32),
                    "volume_coords": vol_points.astype(np.float32),
                    "volume_extra": vol_extra.astype(np.float32),
                    "volume_data": vol_data.astype(np.float32),
                    "params": np.array([q_in, rpm], dtype=np.float32),
                }
            )

        print(
            f"Dataset compiled. Kept {len(self.samples)} valid positive-head cases, "
            f"excluded {excluded_count} negative-head cases."
        )

    def _save_cached_data(self) -> None:
        """Saves processed sample list, triangulation, and boundary metadata to a single compressed .npz archive."""
        data_dict: dict[str, Any] = {}
        for key in [
            "geometry",
            "surface_coords",
            "surface_data",
            "volume_coords",
            "volume_extra",
            "volume_data",
            "params",
        ]:
            data_dict[key] = np.stack([s[key] for s in self.samples], axis=0)

        data_dict["triangles"] = self.triangles

        # Save boundary metadata
        if hasattr(self, "boundary_data"):
            for bkey, barr in self.boundary_data.items():
                data_dict[f"boundary_{bkey}"] = barr

        np.savez_compressed(self.cache_file, **data_dict)
        print(f"Saved pre-processed dataset cache to: {self.cache_file}")

        # Save metadata companion JSON file
        meta_dict = {
            key: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            for key, arr in data_dict.items()
        }
        json_file = self.cache_file.rsplit(".", 1)[0] + ".json"
        with open(json_file, "w") as f:
            json.dump(meta_dict, f, indent=4)
        print(f"Saved cache metadata companion file to: {json_file}")

    def _load_cached_data(self) -> None:
        """Loads cached data and triangulation from a compressed .npz archive rapidly."""
        archive = np.load(self.cache_file)
        n_samples = archive["params"].shape[0]

        # Decompress all arrays from cache
        geometry = archive["geometry"]
        surface_coords = archive["surface_coords"]
        surface_data = archive["surface_data"]
        volume_coords = archive["volume_coords"]
        volume_extra = archive["volume_extra"]
        volume_data = archive["volume_data"]
        params = archive["params"]

        # Ensure full pump geometry and surface coordinates (821 points)
        if geometry.shape[1] < 800:
            ref_contour = "/home/vpspepe/Documents/TUD/HiWi/Ecotwin/data/raw/merged_stls/Pump2D_MRFStudy_finer_contour.npz"
            if os.path.exists(ref_contour):
                full_pts = np.load(ref_contour)["points"].astype(np.float32)
                geometry = np.tile(full_pts[None, ...], (n_samples, 1, 1))
                surface_coords = geometry

        if "triangles" in archive.files:
            self.triangles = archive["triangles"]
        else:
            self.triangles = np.empty((0, 3), dtype=np.int32)

        # Load boundary data if present in archive
        boundary_keys = [
            "pts_in",
            "pts_out",
            "e_in",
            "e_out",
            "norm_in",
            "norm_out",
            "weights_in",
            "weights_out",
            "idx_in",
            "idx_out",
        ]
        if all(f"boundary_{k}" in archive.files for k in boundary_keys):
            self.boundary_data = {k: archive[f"boundary_{k}"] for k in boundary_keys}
        else:
            self._map_boundary_indices(volume_coords[0])

        self.samples = []
        for i in range(n_samples):
            self.samples.append(
                {
                    "geometry": geometry[i],
                    "surface_coords": surface_coords[i],
                    "surface_data": surface_data[i],
                    "volume_coords": volume_coords[i],
                    "volume_extra": volume_extra[i],
                    "volume_data": volume_data[i],
                    "params": params[i],
                }
            )
        print(f"Loaded {len(self.samples)} processed samples from cache.")

    @property
    def inlet_edges(self) -> np.ndarray:
        """Edge connectivity indices for inlet boundary."""
        return self.boundary_data["e_in"]

    @property
    def outlet_edges(self) -> np.ndarray:
        """Edge connectivity indices for outlet boundary."""
        return self.boundary_data["e_out"]

    @property
    def idx_in(self) -> np.ndarray:
        """Indices of inlet nodes in the volume coordinates."""
        return self.boundary_data["idx_in"]

    @property
    def idx_out(self) -> np.ndarray:
        """Indices of outlet nodes in the volume coordinates."""
        return self.boundary_data["idx_out"]

    @property
    def inlet_coords(self) -> np.ndarray:
        """Coordinates of inlet boundary points."""
        return self.boundary_data["pts_in"]

    @property
    def outlet_coords(self) -> np.ndarray:
        """Coordinates of outlet boundary points."""
        return self.boundary_data["pts_out"]

    @property
    def inlet_normals(self) -> np.ndarray:
        """Unit outward normals along inlet boundary."""
        return self.boundary_data["norm_in"]

    @property
    def inlet_weights(self) -> np.ndarray:
        """Quadrature arc-length weights along inlet boundary."""
        return self.boundary_data["weights_in"]

    @property
    def outlet_normals(self) -> np.ndarray:
        """Unit outward normals along outlet boundary."""
        return self.boundary_data["norm_out"]

    @property
    def outlet_weights(self) -> np.ndarray:
        """Quadrature arc-length weights along outlet boundary."""
        return self.boundary_data["weights_out"]

    def _compute_normalization_stats(self) -> None:
        """Computes mean and std of target arrays and parameters over training split."""
        train_surf_data = np.vstack(
            [self.samples[i]["surface_data"] for i in self.train_indices]
        )
        train_vol_data = np.vstack(
            [self.samples[i]["volume_data"] for i in self.train_indices]
        )
        train_params = np.vstack(
            [self.samples[i]["params"] for i in self.train_indices]
        )

        self.mean_surf_data: float = float(train_surf_data.mean())
        self.std_surf_data: float = float(train_surf_data.std())

        self.mean_vol_data: np.ndarray = train_vol_data.mean(axis=0)
        self.std_vol_data: np.ndarray = train_vol_data.std(axis=0)

        self.mean_params: np.ndarray = train_params.mean(axis=0)
        self.std_params: np.ndarray = train_params.std(axis=0)

        # Prevent division by zero
        self.std_surf_data = 1.0 if self.std_surf_data == 0 else self.std_surf_data
        self.std_vol_data = np.where(self.std_vol_data == 0, 1.0, self.std_vol_data)
        self.std_params = np.where(self.std_params == 0, 1.0, self.std_params)

    def __len__(self) -> int:
        """Returns the number of samples in the active split."""
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> Pump2DSample:
        """Fetches normalized PyTorch Tensors for a specific sample index."""
        sample_idx = self.active_indices[idx]
        sample = self.samples[sample_idx]

        # Standard Z-score normalization
        surf_data_norm = (
            sample["surface_data"] - self.mean_surf_data
        ) / self.std_surf_data
        vol_data_norm = (sample["volume_data"] - self.mean_vol_data) / self.std_vol_data
        params_norm = (sample["params"] - self.mean_params) / self.std_params

        return {
            "geometry": torch.tensor(sample["geometry"], dtype=torch.float32),
            "surface_coords": torch.tensor(
                sample["surface_coords"], dtype=torch.float32
            ),
            "surface_data": torch.tensor(surf_data_norm, dtype=torch.float32),
            "volume_coords": torch.tensor(sample["volume_coords"], dtype=torch.float32),
            "volume_extra": torch.tensor(sample["volume_extra"], dtype=torch.float32),
            "volume_data": torch.tensor(vol_data_norm, dtype=torch.float32),
            "params": torch.tensor(params_norm, dtype=torch.float32),
        }

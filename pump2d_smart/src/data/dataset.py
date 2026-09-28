"""Dataset Pipeline Module for 2D Pump Simulations.

Provides PyTorch Dataset implementations for loading 2D COMSOL pump meshes,
interpolating flow fields, pre-computing SDF features, and normalizing outputs.
Reuses boundary probes, geometric integration, and case discovery from pump_static_rom,
and supports high-performance HDF5 (.h5) and NumPy (.npz) caching.
"""

from __future__ import annotations

import json
import os
from os.path import join as pjoin
from typing import Any, TypedDict

import h5py
import numpy as np
import pyvista as pv
import torch
from pump_static_rom.boundary.comsol import (
    build_field_map_for_case,
    discover_cases,
)
from pump_static_rom.metrics.pump import (
    BoundaryProbe,
    build_boundary_probe_from_main,
    extract_edges_from_connectivity,
    _edge_normals_out,
    _point_normals_from_edges,
)
from scipy.spatial import KDTree
from src.data.utils import kennard_stone_split
from src.geometry.processor import Pump2DGeometryProcessor
from src.training.calculate_pump_heads import (
    compute_pump_head,
    extract_unique_edges,
)
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
        boundary_only: bool = False,
    ) -> None:
        """Initializes the dataset and loads/caches processed data.

        Args:
            main_vtu_path: Absolute path to the main grid VTU.
            outlet_vtu_path: Absolute path to the outlet grid VTU.
            cache_dir: Folder to save/load processed .h5 or .npz arrays.
            if_test: If True, load test samples; otherwise load train samples.
            test_split: Fraction of dataset allocated to validation/test (mutually exclusive with n_train).
            sparse_hq_split: If True, uses interleaved sparse sampling per RPM curve.
            n_train: Exact number of training samples selected with Kennard-Stone (mutually exclusive with test_split).
            seed: Random seed for splitting dataset.
            boundary_only: If True, restricts volume queries exclusively to Inlet, Outlet, and Blade boundaries.
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
        self.boundary_only: bool = boundary_only

        self.samples: list[dict[str, np.ndarray]] = []
        self.triangles: np.ndarray = np.empty((0, 3), dtype=np.int32)
        self.boundary_data: dict[str, np.ndarray] = {}
        self._inlet_probe: BoundaryProbe | None = None
        self._outlet_probe: BoundaryProbe | None = None
        self._blades_probe: BoundaryProbe | None = None

        # 1. Load or build cached data (prefers HDF5, falls back to NPZ)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file_h5: str = pjoin(cache_dir, "pump2d_data_cache.h5")
        self.cache_file_npz: str = pjoin(cache_dir, "pump2d_data_cache.npz")
        self.cache_file: str = (
            self.cache_file_h5
            if os.path.exists(self.cache_file_h5)
            else self.cache_file_npz
        )

        if os.path.exists(self.cache_file_h5):
            print(f"Loading cached dataset from HDF5: {self.cache_file_h5}")
            self._load_cached_h5(self.cache_file_h5)
        elif os.path.exists(self.cache_file_npz):
            print(f"Loading cached dataset from NPZ: {self.cache_file_npz}")
            self._load_cached_npz(self.cache_file_npz)
        else:
            print("Pre-processing dataset (this may take a few minutes)...")
            self._build_dataset()

        # Apply boundary-only filter if requested (Inlet + Outlet + Blades)
        if self.boundary_only:
            self._apply_boundary_only_slice()

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

        # 3. Load boundary meshes and map inlet/outlet boundary probes
        boundary_data = self._map_boundary_indices(vol_points, processor.grid)

        # 4. Parse COMSOL sweep configurations using pump_static_rom
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

        # 6. Save pre-processed snapshot list to disk cache (both HDF5 and NPZ)
        self._save_cached_data()

    def _init_geometry_data(
        self,
    ) -> tuple[Pump2DGeometryProcessor, np.ndarray, pv.PolyData, np.ndarray]:
        """Loads the boundary processor and extracts full pump geometry and full surface mesh.

        Returns:
            Tuple of [processor, full_geom_coords, full_surface_mesh, full_surface_coords].
        """
        processor = Pump2DGeometryProcessor(self.main_vtu_path, self.outlet_vtu_path)
        geom_points = processor.get_full_pump_geometry()
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

    def _map_boundary_indices(
        self, vol_points: np.ndarray, main_grid: pv.UnstructuredGrid | None = None
    ) -> dict[str, Any]:
        """Loads inlet/outlet meshes, computes unit normals, weights, and maps indices using BoundaryProbe.

        Args:
            vol_points: Volume coordinate points array of shape (N, 2).
            main_grid: Optional main mesh to build boundary probes.

        Returns:
            Dictionary containing mapped indices, boundary coords, edges, normals, and weights.
        """
        comsol_dir = os.path.dirname(self.main_vtu_path)
        in_mesh = pv.read(pjoin(comsol_dir, "Pump2D_MRFStudy_finer_Inlet.vtu"))
        out_mesh = pv.read(pjoin(comsol_dir, "Pump2D_MRFStudy_finer_Outlet.vtu"))
        blades_vtu = pjoin(comsol_dir, "Pump2D_MRFStudy_finer_Blades.vtu")
        pts_blades: np.ndarray | None = None
        e_blades: np.ndarray | None = None
        idx_blades: np.ndarray | None = None
        norm_blades: np.ndarray | None = None
        weights_blades: np.ndarray | None = None

        if main_grid is not None:
            self._inlet_probe = build_boundary_probe_from_main(
                main_grid, in_mesh, name="Inlet"
            )
            self._outlet_probe = build_boundary_probe_from_main(
                main_grid, out_mesh, name="Outlet"
            )

            pts_in = self._inlet_probe.points[:, :2].astype(np.float32)
            pts_out = self._outlet_probe.points[:, :2].astype(np.float32)
            e_in = self._inlet_probe.edges.astype(np.int64)
            e_out = self._outlet_probe.edges.astype(np.int64)
            idx_in = self._inlet_probe.idx_main.astype(np.int64)
            idx_out = self._outlet_probe.idx_main.astype(np.int64)
            norm_in = self._inlet_probe.normals_out[:, :2].astype(np.float32)
            norm_out = self._outlet_probe.normals_out[:, :2].astype(np.float32)

            p0_in = pts_in[e_in[:, 0]]
            p1_in = pts_in[e_in[:, 1]]
            weights_in = np.linalg.norm(p1_in - p0_in, axis=1).astype(np.float32)

            p0_out = pts_out[e_out[:, 0]]
            p1_out = pts_out[e_out[:, 1]]
            weights_out = np.linalg.norm(p1_out - p0_out, axis=1).astype(np.float32)

            if os.path.exists(blades_vtu):
                self._blades_probe = build_boundary_probe_from_main(
                    main_grid, pv.read(blades_vtu), name="Blades"
                )
                pts_blades = self._blades_probe.points[:, :2].astype(np.float32)
                e_blades = self._blades_probe.edges.astype(np.int64)
                idx_blades = self._blades_probe.idx_main.astype(np.int64)
                norm_blades = self._blades_probe.normals_out[:, :2].astype(np.float32)
                p0_bl = pts_blades[e_blades[:, 0]]
                p1_bl = pts_blades[e_blades[:, 1]]
                weights_blades = np.linalg.norm(p1_bl - p0_bl, axis=1).astype(np.float32)
        else:
            pts_in = np.asarray(in_mesh.points, dtype=np.float32)[:, :2]
            pts_out = np.asarray(out_mesh.points, dtype=np.float32)[:, :2]
            e_in = extract_unique_edges(in_mesh)
            e_out = extract_unique_edges(out_mesh)

            p0_in = pts_in[e_in[:, 0]]
            p1_in = pts_in[e_in[:, 1]]
            seg_in = p1_in - p0_in
            weights_in = np.linalg.norm(seg_in, axis=1).astype(np.float32)
            raw_norm_in = np.column_stack([seg_in[:, 1], -seg_in[:, 0]]) / np.maximum(
                weights_in[:, None], 1e-12
            )
            norm_in = raw_norm_in.astype(np.float32)

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

            tree = KDTree(vol_points)
            _, idx_in = tree.query(pts_in)
            _, idx_out = tree.query(pts_out)

            if os.path.exists(blades_vtu):
                blades_mesh = pv.read(blades_vtu)
                pts_blades = np.asarray(blades_mesh.points, dtype=np.float32)[:, :2]
                e_blades = np.asarray(
                    extract_edges_from_connectivity(blades_mesh), dtype=np.int64
                )
                edge_n = _edge_normals_out(
                    blades_mesh.points, e_blades, np.array([0.0, 0.0, 0.0])
                )
                norm_blades = _point_normals_from_edges(
                    blades_mesh.points, e_blades, edge_n
                ).astype(np.float32)[:, :2]
                p0_bl = pts_blades[e_blades[:, 0]]
                p1_bl = pts_blades[e_blades[:, 1]]
                weights_blades = np.linalg.norm(p1_bl - p0_bl, axis=1).astype(np.float32)
                _, idx_blades = tree.query(pts_blades)

        boundary_dict: dict[str, Any] = {
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
        if pts_blades is not None and e_blades is not None and idx_blades is not None:
            boundary_dict.update(
                {
                    "pts_blades": pts_blades,
                    "e_blades": e_blades.astype(np.int64),
                    "norm_blades": norm_blades,
                    "weights_blades": weights_blades,
                    "idx_blades": idx_blades.astype(np.int64),
                }
            )

        self.boundary_data = boundary_dict
        self._reconstruct_probes_from_boundary_data()
        return boundary_dict

    def _reconstruct_probes_from_boundary_data(self) -> None:
        """Instantiates BoundaryProbe objects for Inlet, Outlet, and Blades from boundary_data."""
        pts_in = self.boundary_data["pts_in"]
        pts_out = self.boundary_data["pts_out"]
        pts_in_3d = (
            np.column_stack([pts_in, np.zeros(len(pts_in))])
            if pts_in.shape[1] == 2
            else pts_in
        )
        pts_out_3d = (
            np.column_stack([pts_out, np.zeros(len(pts_out))])
            if pts_out.shape[1] == 2
            else pts_out
        )

        norm_in = self.boundary_data["norm_in"]
        norm_out = self.boundary_data["norm_out"]
        norm_in_3d = (
            np.column_stack([norm_in, np.zeros(len(norm_in))])
            if norm_in.shape[1] == 2
            else norm_in
        )
        norm_out_3d = (
            np.column_stack([norm_out, np.zeros(len(norm_out))])
            if norm_out.shape[1] == 2
            else norm_out
        )

        self._inlet_probe = BoundaryProbe(
            name="Inlet",
            points=pts_in_3d.astype(np.float64),
            edges=self.boundary_data["e_in"].astype(np.int64),
            idx_main=self.boundary_data["idx_in"].astype(np.int64),
            normals_out=norm_in_3d.astype(np.float64),
        )
        self._outlet_probe = BoundaryProbe(
            name="Outlet",
            points=pts_out_3d.astype(np.float64),
            edges=self.boundary_data["e_out"].astype(np.int64),
            idx_main=self.boundary_data["idx_out"].astype(np.int64),
            normals_out=norm_out_3d.astype(np.float64),
        )

        if "pts_blades" in self.boundary_data:
            pts_bl = self.boundary_data["pts_blades"]
            pts_bl_3d = (
                np.column_stack([pts_bl, np.zeros(len(pts_bl))])
                if pts_bl.shape[1] == 2
                else pts_bl
            )
            norm_bl = self.boundary_data.get("norm_blades")
            if norm_bl is not None:
                norm_bl_3d = (
                    np.column_stack([norm_bl, np.zeros(len(norm_bl))])
                    if norm_bl.shape[1] == 2
                    else norm_bl
                )
            else:
                norm_bl_3d = np.zeros_like(pts_bl_3d)

            self._blades_probe = BoundaryProbe(
                name="Blades",
                points=pts_bl_3d.astype(np.float64),
                edges=self.boundary_data["e_blades"].astype(np.int64),
                idx_main=self.boundary_data["idx_blades"].astype(np.int64),
                normals_out=norm_bl_3d.astype(np.float64),
            )

    def _parse_cfd_configurations(
        self, grid: pv.UnstructuredGrid
    ) -> list[tuple[int, float, int, str]]:
        """Parses COMSOL operating point parameters using pump_static_rom.boundary.comsol.

        Args:
            grid: PyVista UnstructuredGrid representing the simulation.

        Returns:
            Sorted list of tuples containing [index, flow_rate, rpm, array_key].
        """
        cases = discover_cases(grid)
        configs: list[tuple[int, float, int, str]] = []
        for idx, (q_in, rpm) in enumerate(cases):
            fmap = build_field_map_for_case(grid, (q_in, rpm))
            if "pressure" in fmap:
                configs.append((idx, float(q_in), int(rpm), fmap["pressure"]))

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
        print("Interpolating pressure fields onto full pump boundary...")
        sampled_surface = surface_mesh.sample(grid)

        tri_grid = grid.triangulate()
        self.samples = []
        excluded_count = 0

        for idx, q_in, rpm, p_key in configs:
            fmap = build_field_map_for_case(grid, (q_in, rpm))
            vx_key = fmap.get(
                "vx", p_key.replace("Pressure_", "Velocity_field,_x_component_")
            )
            vy_key = fmap.get(
                "vy", p_key.replace("Pressure_", "Velocity_field,_y_component_")
            )

            p_vol = tri_grid[p_key]
            vx = tri_grid[vx_key]
            vy = tri_grid[vy_key]

            # Compute pump head using pump_static_rom integration formula
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
        """Saves processed sample list, triangulation, and boundary metadata to HDF5 and NPZ archives."""
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

        # 1. Save to HDF5 with per-sample chunking and gzip compression
        with h5py.File(self.cache_file_h5, "w") as f:
            for k, arr in data_dict.items():
                chunks = (1, *arr.shape[1:]) if arr.ndim > 1 else True
                f.create_dataset(
                    k, data=arr, compression="gzip", compression_opts=4, chunks=chunks
                )
        print(f"Saved pre-processed dataset HDF5 cache to: {self.cache_file_h5}")

        # 2. Save to compressed NPZ for full backwards compatibility
        np.savez_compressed(self.cache_file_npz, **data_dict)
        print(f"Saved pre-processed dataset NPZ cache to: {self.cache_file_npz}")

        # Save metadata companion JSON file
        meta_dict = {
            key: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            for key, arr in data_dict.items()
        }
        json_file = self.cache_file_npz.rsplit(".", 1)[0] + ".json"
        with open(json_file, "w") as f:
            json.dump(meta_dict, f, indent=4)
        print(f"Saved cache metadata companion file to: {json_file}")

    def _load_cached_h5(self, h5_path: str) -> None:
        """Loads cached data and triangulation rapidly from an HDF5 archive."""
        with h5py.File(h5_path, "r") as archive:
            n_samples = archive["params"].shape[0]

            geometry = archive["geometry"][:]
            surface_coords = archive["surface_coords"][:]
            surface_data = archive["surface_data"][:]
            volume_coords = archive["volume_coords"][:]
            volume_extra = archive["volume_extra"][:]
            volume_data = archive["volume_data"][:]
            params = archive["params"][:]

            if "triangles" in archive:
                self.triangles = archive["triangles"][:]
            else:
                self.triangles = np.empty((0, 3), dtype=np.int32)

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
            if all(f"boundary_{k}" in archive for k in boundary_keys):
                self.boundary_data = {
                    k: archive[f"boundary_{k}"][:] for k in boundary_keys
                }
                blade_keys = [
                    "pts_blades",
                    "e_blades",
                    "norm_blades",
                    "weights_blades",
                    "idx_blades",
                ]
                if all(f"boundary_{k}" in archive for k in blade_keys):
                    for k in blade_keys:
                        self.boundary_data[k] = archive[f"boundary_{k}"][:]
                else:
                    self._ensure_blades_boundary(volume_coords[0])
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
        self._reconstruct_probes_from_boundary_data()
        print(f"Loaded {len(self.samples)} processed samples from HDF5 cache.")

    def _load_cached_npz(self, npz_path: str) -> None:
        """Loads cached data and triangulation from a compressed .npz archive."""
        archive = np.load(npz_path)
        n_samples = archive["params"].shape[0]

        geometry = archive["geometry"]
        surface_coords = archive["surface_coords"]
        surface_data = archive["surface_data"]
        volume_coords = archive["volume_coords"]
        volume_extra = archive["volume_extra"]
        volume_data = archive["volume_data"]
        params = archive["params"]

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
            blade_keys = [
                "pts_blades",
                "e_blades",
                "norm_blades",
                "weights_blades",
                "idx_blades",
            ]
            if all(f"boundary_{k}" in archive.files for k in blade_keys):
                for k in blade_keys:
                    self.boundary_data[k] = archive[f"boundary_{k}"]
            else:
                self._ensure_blades_boundary(volume_coords[0])
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
        self._reconstruct_probes_from_boundary_data()
        print(f"Loaded {len(self.samples)} processed samples from NPZ cache.")

    @property
    def inlet_probe(self) -> BoundaryProbe:
        """BoundaryProbe object for inlet boundary."""
        if self._inlet_probe is None:
            self._reconstruct_probes_from_boundary_data()
        return self._inlet_probe

    @property
    def outlet_probe(self) -> BoundaryProbe:
        """BoundaryProbe object for outlet boundary."""
        if self._outlet_probe is None:
            self._reconstruct_probes_from_boundary_data()
        return self._outlet_probe

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

    @property
    def blades_probe(self) -> BoundaryProbe | None:
        """BoundaryProbe object for blades boundary."""
        if self._blades_probe is None and "pts_blades" in self.boundary_data:
            self._reconstruct_probes_from_boundary_data()
        return self._blades_probe

    @property
    def idx_blades(self) -> np.ndarray:
        """Indices of blade nodes in the volume coordinates."""
        return self.boundary_data["idx_blades"]

    @property
    def blades_coords(self) -> np.ndarray:
        """Coordinates of blade boundary points."""
        return self.boundary_data["pts_blades"]

    @property
    def blades_edges(self) -> np.ndarray:
        """Edge connectivity indices for blade boundary."""
        return self.boundary_data["e_blades"]

    @property
    def blades_normals(self) -> np.ndarray:
        """Unit outward normals along blade boundary."""
        return self.boundary_data["norm_blades"]

    @property
    def blades_weights(self) -> np.ndarray:
        """Quadrature arc-length weights along blade boundary."""
        return self.boundary_data["weights_blades"]

    def _ensure_blades_boundary(self, vol_points: np.ndarray) -> None:
        """Populates blades boundary information and probes from VTU."""
        comsol_dir = os.path.dirname(self.main_vtu_path) if self.main_vtu_path else ""
        blades_vtu = pjoin(comsol_dir, "Pump2D_MRFStudy_finer_Blades.vtu")
        if not os.path.exists(blades_vtu):
            alt_path = "/home/vpspepe/Documents/TUD/HiWi/Ecotwin/data/raw/comsol/Pump2D_MRFStudy_finer_Blades.vtu"
            if os.path.exists(alt_path):
                blades_vtu = alt_path

        if os.path.exists(blades_vtu):
            blades_mesh = pv.read(blades_vtu)
            pts = np.asarray(blades_mesh.points, dtype=np.float32)[:, :2]
            edges = np.asarray(
                extract_edges_from_connectivity(blades_mesh), dtype=np.int64
            )
            edge_n = _edge_normals_out(
                blades_mesh.points, edges, np.array([0.0, 0.0, 0.0])
            )
            pt_n = _point_normals_from_edges(
                blades_mesh.points, edges, edge_n
            ).astype(np.float32)[:, :2]

            p0 = pts[edges[:, 0]]
            p1 = pts[edges[:, 1]]
            weights = np.linalg.norm(p1 - p0, axis=1).astype(np.float32)

            tree = KDTree(vol_points)
            _, idx_blades = tree.query(pts)
            idx_blades = idx_blades.astype(np.int64)

            self.boundary_data["pts_blades"] = pts
            self.boundary_data["e_blades"] = edges
            self.boundary_data["norm_blades"] = pt_n
            self.boundary_data["weights_blades"] = weights
            self.boundary_data["idx_blades"] = idx_blades

    def _apply_boundary_only_slice(self) -> None:
        """Restricts query volume coordinates and targets strictly to Inlet, Outlet, and Blades."""
        if "idx_blades" not in self.boundary_data:
            self._ensure_blades_boundary(self.samples[0]["volume_coords"])

        idx_in = self.boundary_data["idx_in"]
        idx_out = self.boundary_data["idx_out"]
        idx_blades = self.boundary_data.get("idx_blades")
        if idx_blades is None:
            raise RuntimeError(
                "Blades boundary indices could not be determined for boundary-only mode."
            )

        bnd_indices = np.concatenate([idx_in, idx_out, idx_blades])
        n_in = len(idx_in)
        n_out = len(idx_out)
        n_blades = len(idx_blades)

        for s in self.samples:
            s["volume_coords"] = s["volume_coords"][bnd_indices]
            s["volume_data"] = s["volume_data"][bnd_indices]
            s["volume_extra"] = np.empty((len(bnd_indices), 0), dtype=np.float32)

        # Re-index boundary mapping into the sliced query points
        self.boundary_data["idx_in"] = np.arange(0, n_in, dtype=np.int64)
        self.boundary_data["idx_out"] = np.arange(n_in, n_in + n_out, dtype=np.int64)
        self.boundary_data["idx_blades"] = np.arange(
            n_in + n_out, n_in + n_out + n_blades, dtype=np.int64
        )

        # Clear volume triangles as fluid mesh is no longer a 2D surface triangulation
        self.triangles = np.empty((0, 3), dtype=np.int32)

        self._reconstruct_probes_from_boundary_data()
        print(
            f"Boundary-Only mode active: Sliced queries to {len(bnd_indices)} surface points "
            f"(Inlet: {n_in}, Outlet: {n_out}, Blades: {n_blades}) | No extra SDF queries."
        )

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

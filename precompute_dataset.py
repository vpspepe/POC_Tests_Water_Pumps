#!/usr/bin/env python
"""Precompute and cache preprocessed points, features, normals, and curvature for pump meshes in parallel."""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

# Add shape directory to sys.path
shape_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "shape"))
if os.path.isdir(shape_dir) and shape_dir not in sys.path:
    sys.path.insert(0, shape_dir)

from shape_foundation.configs.default import load_config
from shape_foundation.data.preprocessing import MeshPreprocessor
from shape_foundation.data.sampling import SurfaceSampler


class ProcessStatus(Enum):
    """Execution status for processing a single mesh."""

    SUCCESS = auto()
    ALREADY_DONE = auto()
    ERROR = auto()


def check_all_single_mesh_views_exist(mesh_cache_dir: Path, num_views: int) -> bool:
    """Check if all expected view files already exist in the cache directory.

    Args:
        mesh_cache_dir: The directory containing cached view files for a mesh.
        num_views: The number of stochastic views expected.

    Returns:
        True if all view files exist, False otherwise.
    """
    for v in range(num_views):
        view_path = mesh_cache_dir / f"view_{v:02d}.npz"
        if not view_path.exists():
            return False
    return True


def load_and_clean_mesh(stl_path: Path) -> trimesh.Trimesh:
    """Load an STL file and consolidate it into a single clean trimesh object.

    Args:
        stl_path: The filesystem path to the STL file.

    Returns:
        A consolidated trimesh.Trimesh object.
    """
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def sample_and_save_views(
    preprocessed: dict[str, Any],
    sampler: SurfaceSampler,
    preprocessor: MeshPreprocessor,
    mesh_cache_dir: Path,
    num_views: int,
) -> None:
    """Generate stochastic views from preprocessed mesh data and cache them.

    Args:
        preprocessed: The preprocessed mesh vertices, faces, normals, and curvature.
        sampler: The surface sampler component.
        preprocessor: The mesh preprocessor component.
        mesh_cache_dir: Destination cache directory.
        num_views: Number of stochastic views to generate.
    """
    for v in range(num_views):
        view_path = mesh_cache_dir / f"view_{v:02d}.npz"
        sampled = sampler.sample(
            vertices=preprocessed["vertices"],
            faces=preprocessed["faces"],
            normals=preprocessed["normals"],
            curvature=preprocessed["curvature"],
        )
        features = preprocessor.build_features(
            points=sampled["points"],
            normals=sampled["normals"],
            curvature=sampled["curvature"],
        )
        np.savez_compressed(
            view_path,
            points=sampled["points"],
            features=features,
            normals=sampled["normals"]
            if sampled["normals"] is not None
            else np.zeros((32768, 3)),
            curvature=sampled["curvature"]
            if sampled["curvature"] is not None
            else np.zeros((32768, 1)),
        )


def process_single_mesh(
    stl_path: Path, config_path: Path, cache_dir: Path, num_views: int
) -> tuple[str, ProcessStatus, str]:
    """Preprocess a single mesh, generate stochastic views, and cache the result.

    Args:
        stl_path: Path to the mesh STL file.
        config_path: Path to the SHAPE YAML config file.
        cache_dir: Directory where cached features are saved.
        num_views: Number of views to generate.

    Returns:
        A tuple of (mesh_id, ProcessStatus, error_message).
    """
    try:
        cfg = load_config(str(config_path))
        cfg.input.num_surface_points = 32768

        preprocessor = MeshPreprocessor(cfg.input)
        sampler = SurfaceSampler(cfg.input)

        mesh_id = stl_path.parent.name
        mesh_cache_dir = cache_dir / mesh_id
        mesh_cache_dir.mkdir(parents=True, exist_ok=True)

        if check_all_single_mesh_views_exist(mesh_cache_dir, num_views):
            return mesh_id, ProcessStatus.ALREADY_DONE, ""

        mesh = load_and_clean_mesh(stl_path)
        preprocessed = preprocessor(mesh.vertices, mesh.faces)
        sample_and_save_views(
            preprocessed, sampler, preprocessor, mesh_cache_dir, num_views
        )

        return mesh_id, ProcessStatus.SUCCESS, ""
    except Exception as e:
        return stl_path.name, ProcessStatus.ERROR, str(e)


def collect_stl_files(data_root: Path) -> list[Path]:
    """Find all STL files recursively under the given directory.

    Args:
        data_root: Root directory to search.

    Returns:
        A sorted list of Paths to the discovered STL files.
    """
    stl_files = sorted(list(data_root.rglob("merged_surfaces.stl")))
    if not stl_files:
        stl_files = sorted(list(data_root.rglob("*.stl")))
    return stl_files


def run_parallel_precomputation(
    stl_files: list[Path], config_path: Path, cache_dir: Path, num_views: int
) -> None:
    """Orchestrate the parallel processing of the collected meshes.

    Args:
        stl_files: List of paths to the STL files.
        config_path: Path to the config file.
        cache_dir: Target cache directory.
        num_views: Number of stochastic views per mesh.
    """
    max_workers = min(os.cpu_count() or 4, 8)
    print(
        f"Precomputing {num_views} stochastic views each using {max_workers} processes..."
    )

    completed, errors, already_done = 0, 0, 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_mesh, path, config_path, cache_dir, num_views
            ): path
            for path in stl_files
        }
        for future in as_completed(futures):
            mesh_id, status, err_msg = future.result()
            if status == ProcessStatus.SUCCESS:
                completed += 1
            elif status == ProcessStatus.ALREADY_DONE:
                already_done += 1
            else:
                print(f"Error for {mesh_id}: {err_msg}")
                errors += 1

            total = completed + already_done + errors
            if total % 10 == 0 or total == len(stl_files):
                print(
                    f"Progress: {total}/{len(stl_files)} processed ({completed} new, {already_done} cached, {errors} errors)"
                )


def main() -> None:
    """Main entry point for precomputing and caching mesh features."""
    config_path = Path("shape/configs/small.yaml").resolve()
    data_root = Path("pump_metadata").resolve()
    cache_dir = Path("pump_metadata_cache").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    stl_files = collect_stl_files(data_root)
    if not stl_files:
        print("No STL files found!")
        return

    print(f"Found {len(stl_files)} STL meshes.")
    run_parallel_precomputation(stl_files, config_path, cache_dir, num_views=10)
    print("Precomputation finished successfully!")


if __name__ == "__main__":
    main()

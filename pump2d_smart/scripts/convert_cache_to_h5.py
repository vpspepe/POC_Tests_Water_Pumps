"""CLI Utility to convert NumPy .npz cache files to chunked, compressed HDF5 (.h5).

Usage:
    python scripts/convert_cache_to_h5.py [--npz cache/pump2d_data_cache.npz] [--h5 cache/pump2d_data_cache.h5]
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np


def convert_npz_to_h5(npz_path: str, h5_path: str, compression_level: int = 4) -> None:
    """Converts a .npz dataset archive into a chunked HDF5 archive."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Source NPZ file not found: {npz_path}")

    print(f"Reading NPZ: {npz_path} ({os.path.getsize(npz_path) / 1e6:.2f} MB)...")
    t0 = time.time()
    npz_data = np.load(npz_path)

    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        for k in npz_data.files:
            arr = npz_data[k]
            # Chunk per sample along first dimension for instant slice access
            chunks = (1, *arr.shape[1:]) if arr.ndim > 1 else True
            f.create_dataset(
                k,
                data=arr,
                compression="gzip",
                compression_opts=compression_level,
                chunks=chunks,
            )

    t1 = time.time()
    h5_size_mb = os.path.getsize(h5_path) / 1e6
    print(
        f"Successfully wrote HDF5: {h5_path} ({h5_size_mb:.2f} MB) in {t1 - t0:.2f}s!"
    )

    # Verify exact bit-for-bit match
    print("Verifying array equivalence...")
    with h5py.File(h5_path, "r") as f:
        for k in npz_data.files:
            assert np.array_equal(npz_data[k], f[k][:]), f"Mismatch in array: {k}"
    print("Verification passed! All arrays match exactly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NPZ cache to HDF5.")
    parser.add_argument(
        "--npz",
        type=str,
        default="cache/pump2d_data_cache.npz",
        help="Path to source .npz file",
    )
    parser.add_argument(
        "--h5",
        type=str,
        default="cache/pump2d_data_cache.h5",
        help="Path to target .h5 file",
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=4,
        help="GZIP compression level (0-9, default: 4)",
    )
    args = parser.parse_args()

    convert_npz_to_h5(args.npz, args.h5, args.compression)


if __name__ == "__main__":
    main()

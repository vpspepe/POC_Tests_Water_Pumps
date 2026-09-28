"""Physics and Numerical Integration Helpers for Pump Simulation Analysis.

Reuses core geometric line integration, undirected edge extraction, and pump metric
calculations from pump_static_rom while maintaining full backwards compatibility.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pump_static_rom.boundary.geometry import (
    extract_edges_from_connectivity,
    line_integral,
    unique_undirected_edges,
)
from pump_static_rom.metrics.pump import (
    BoundaryProbe,
    PumpMetrics,
    compute_pump_metrics_for_case,
)

__all__ = [
    "BoundaryProbe",
    "PumpMetrics",
    "compute_full_pump_metrics",
    "compute_pump_head",
    "extract_unique_edges",
    "line_integral",
]


def extract_unique_edges(mesh: pv.UnstructuredGrid) -> np.ndarray:
    """Safely extracts unique undirected line edges from UnstructuredGrid cell arrays.

    Delegates directly to pump_static_rom.boundary.geometry routines.

    Args:
        mesh: PyVista UnstructuredGrid boundary mesh.

    Returns:
        Array of unique point index pairs of shape (M, 2).
    """
    raw_edges = extract_edges_from_connectivity(mesh)
    return unique_undirected_edges(raw_edges)


def compute_pump_head(
    p_all: np.ndarray,
    vx_all: np.ndarray,
    vy_all: np.ndarray,
    idx_in: np.ndarray,
    idx_out: np.ndarray,
    pts_in: np.ndarray,
    pts_out: np.ndarray,
    e_in: np.ndarray,
    e_out: np.ndarray,
    rho: float = 998.0,
    g: float = 9.81,
) -> float:
    """Calculates the average total pressure difference and head between inlet and outlet.

    Delegates to pump_static_rom.metrics.pump.compute_pump_metrics_for_case.

    Args:
        p_all: Volume static pressure field.
        vx_all: Volume velocity field (X component).
        vy_all: Volume velocity field (Y component).
        idx_in: Mapped volume indices at inlet boundary.
        idx_out: Mapped volume indices at outlet boundary.
        pts_in: Coordinates of inlet boundary of shape (N_in, 2).
        pts_out: Coordinates of outlet boundary of shape (N_out, 2).
        e_in: Connectivity edges of inlet boundary of shape (M_in, 2).
        e_out: Connectivity edges of outlet boundary of shape (M_out, 2).
        rho: Density of fluid.
        g: Gravitational acceleration constant.

    Returns:
        Calculated physical head H (m).
    """
    metrics = compute_full_pump_metrics(
        p_all=p_all,
        vx_all=vx_all,
        vy_all=vy_all,
        idx_in=idx_in,
        idx_out=idx_out,
        pts_in=pts_in,
        pts_out=pts_out,
        e_in=e_in,
        e_out=e_out,
        rho=rho,
        g=g,
    )
    return float(metrics.h)


def compute_full_pump_metrics(
    p_all: np.ndarray,
    vx_all: np.ndarray,
    vy_all: np.ndarray,
    idx_in: np.ndarray,
    idx_out: np.ndarray,
    pts_in: np.ndarray,
    pts_out: np.ndarray,
    e_in: np.ndarray,
    e_out: np.ndarray,
    vz_all: np.ndarray | None = None,
    qin: float = 0.0,
    rpm: int = 0,
    rho: float = 998.0,
    g: float = 9.81,
    blades_probe: BoundaryProbe | None = None,
    t_tracx: np.ndarray | None = None,
    t_tracy: np.ndarray | None = None,
    alphat: np.ndarray | None = None,
) -> PumpMetrics:
    """Computes comprehensive engineering pump metrics using pump_static_rom.

    Returns:
        PumpMetrics dataclass containing head H, static and total delta_p,
        mass flow rates, torque, power, and efficiency.
    """
    pts_in_arr = np.asarray(pts_in, dtype=float)
    pts_out_arr = np.asarray(pts_out, dtype=float)

    # Ensure 3D coordinate shape for probes
    if pts_in_arr.shape[1] == 2:
        pts_in_3d = np.column_stack([pts_in_arr, np.zeros(len(pts_in_arr))])
    else:
        pts_in_3d = pts_in_arr

    if pts_out_arr.shape[1] == 2:
        pts_out_3d = np.column_stack([pts_out_arr, np.zeros(len(pts_out_arr))])
    else:
        pts_out_3d = pts_out_arr

    inlet_probe = BoundaryProbe(
        name="Inlet",
        points=pts_in_3d,
        edges=np.asarray(e_in, dtype=int),
        idx_main=np.asarray(idx_in, dtype=int),
        normals_out=np.zeros_like(pts_in_3d),
    )

    outlet_probe = BoundaryProbe(
        name="Outlet",
        points=pts_out_3d,
        edges=np.asarray(e_out, dtype=int),
        idx_main=np.asarray(idx_out, dtype=int),
        normals_out=np.zeros_like(pts_out_3d),
    )

    if blades_probe is None:
        # Dummy blade probe to satisfy required probe check in pump_static_rom
        blades_probe = BoundaryProbe(
            name="Blades",
            points=pts_in_3d[:2],
            edges=np.array([[0, 1]], dtype=int),
            idx_main=np.asarray(idx_in[:2], dtype=int),
            normals_out=np.zeros((2, 3)),
        )

    probes = {
        "Inlet": inlet_probe,
        "Outlet": outlet_probe,
        "Blades": blades_probe,
    }

    if vz_all is None:
        vz_arr = np.zeros_like(vx_all)
    else:
        vz_arr = vz_all

    return compute_pump_metrics_for_case(
        qin=float(qin),
        rpm=int(rpm),
        pressure=p_all,
        vx=vx_all,
        vy=vy_all,
        vz=vz_arr,
        probes=probes,
        rho_ref=float(rho),
        g_const=float(g),
        t_tracx=t_tracx,
        t_tracy=t_tracy,
        alphat=alphat,
    )

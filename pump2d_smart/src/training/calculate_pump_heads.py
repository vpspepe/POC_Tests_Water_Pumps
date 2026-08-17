"""Physics and Numerical Integration Helpers for Pump Simulation Analysis.

Provides line integration, undirected edge extraction, and physical pump head calculations.
"""

import numpy as np
import pyvista as pv


def extract_unique_edges(mesh: pv.UnstructuredGrid) -> np.ndarray:
    """Safely extracts unique undirected line edges from UnstructuredGrid cell arrays.

    Args:
        mesh: PyVista UnstructuredGrid boundary mesh.

    Returns:
        Array of unique point index pairs of shape (M, 2).
    """
    VTK_LINE = 3
    VTK_POLY_LINE = 4
    edges = []
    celltypes = np.asarray(mesh.celltypes, dtype=np.uint8)
    cells = np.asarray(mesh.cells, dtype=np.int64)

    i = 0
    while i < len(cells):
        n = int(cells[i])
        ids = cells[i + 1 : i + 1 + n]
        ctype = (
            celltypes[i // (n + 1)]
            if (i // (n + 1)) < len(celltypes)
            else celltypes[-1]
        )
        if ctype in (VTK_LINE, VTK_POLY_LINE) and n >= 2:
            for j in range(n - 1):
                a, b = int(ids[j]), int(ids[j + 1])
                if a != b:
                    edges.append((a, b))
        i += n + 1

    seen = set()
    uniq = []
    for a, b in edges:
        key = (a, b) if a < b else (b, a)
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    return np.array(uniq, dtype=np.int64)


def line_integral(values: np.ndarray, points: np.ndarray, edges: np.ndarray) -> float:
    """Calculates the trapezoidal line integral of values over points connected by edges.

    Args:
        values: Field values at boundary points.
        points: Coordinates of boundary points of shape (N, 2).
        edges: Connectivity indices of shape (M, 2).

    Returns:
        Trapezoidal line integral value.
    """
    if edges.shape[0] == 0:
        return 0.0
    p0 = points[edges[:, 0], :2]
    p1 = points[edges[:, 1], :2]
    ds = np.linalg.norm(p1 - p0, axis=1)
    f0 = values[edges[:, 0]]
    f1 = values[edges[:, 1]]
    return float(np.sum(0.5 * (f0 + f1) * ds))


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
    # 1. Extract boundary field values
    p_in = p_all[idx_in]
    vx_in = vx_all[idx_in]
    vy_in = vy_all[idx_in]
    v_in_sq = vx_in**2 + vy_in**2

    p_out = p_all[idx_out]
    vx_out = vx_all[idx_out]
    vy_out = vy_all[idx_out]
    v_out_sq = vx_out**2 + vy_out**2

    # 2. Total pressure fields Pt
    pt_in = p_in + 0.5 * rho * v_in_sq
    pt_out = p_out + 0.5 * rho * v_out_sq

    # 3. Integrate weighted total pressure over boundary lines
    int_pt_in = line_integral(pt_in, pts_in, e_in)
    int_pt_out = line_integral(pt_out, pts_out, e_out)

    # 4. Integrate line lengths
    ones_in = np.ones(len(pts_in))
    ones_out = np.ones(len(pts_out))
    len_in = line_integral(ones_in, pts_in, e_in)
    len_out = line_integral(ones_out, pts_out, e_out)

    # 5. Average total pressures
    pt_in_avg = int_pt_in / len_in if len_in > 0 else 0.0
    pt_out_avg = int_pt_out / len_out if len_out > 0 else 0.0

    # 6. Calculate head
    head = (pt_out_avg - pt_in_avg) / (rho * g)
    return float(head)

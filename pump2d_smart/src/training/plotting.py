"""Visualization and Plotting Module for 2D Pump Surrogate Modeling.

Pure visualization library responsible for generating:
1. Dataset H-Q split distribution plots (available vs training operating points).
2. Loss progression curves over training epochs.
3. Flow field side-by-side contour comparison plots (COMSOL Truth | SMART Prediction | Error).
4. Physical pump head (H-Q) performance curves and head error calculations.
"""

import os
from os.path import join as pjoin
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from matplotlib import tri
from matplotlib.lines import Line2D
from scipy.spatial import KDTree
from src.data.dataset import Pump2DDataset
from src.training.calculate_pump_heads import compute_pump_head, extract_unique_edges
from src.training.feature_manager import FeatureManager


def plot_side_by_side(
    triangulation: tri.Triangulation,
    truth: np.ndarray,
    pred: np.ndarray,
    title: str,
    save_path: str,
    subtitle: str = "",
    cmap: str = "viridis",
) -> None:
    """Generates side-by-side comparison plots: Ground Truth | SMART Prediction | Absolute Error."""
    error = np.abs(truth - pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Ground Truth
    ax0 = axes[0]
    cnt0 = ax0.tricontourf(triangulation, truth, levels=100, cmap=cmap)
    fig.colorbar(cnt0, ax=ax0)
    ax0.set_title(f"COMSOL Ground Truth ({title})\n{subtitle}")
    ax0.set_aspect("equal")
    ax0.axis("off")

    # 2. Prediction
    ax1 = axes[1]
    cnt1 = ax1.tricontourf(triangulation, pred, levels=100, cmap=cmap)
    fig.colorbar(cnt1, ax=ax1)
    ax1.set_title(f"SMART Prediction ({title})\n{subtitle}")
    ax1.set_aspect("equal")
    ax1.axis("off")

    # 3. Absolute Error
    ax2 = axes[2]
    cnt2 = ax2.tricontourf(triangulation, error, levels=100, cmap="inferno")
    fig.colorbar(cnt2, ax=ax2)
    ax2.set_title(f"Absolute Error ({title})")
    ax2.set_aspect("equal")
    ax2.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to: {save_path}")


def plot_hq_data_split(
    dataset: Pump2DDataset,
    save_path: str,
    exp_name: str = "",
) -> str:
    """Plots and saves the dataset H-Q split distribution graph.

    Shows available/validation points as dots (●) and training points as 'X' (✖).

    Args:
        dataset: Pump2DDataset instance.
        save_path: Filepath to save PNG plot.
        exp_name: Optional experiment name for title.

    Returns:
        str: Filepath where plot was saved.
    """
    try:
        pts_in = dataset.inlet_coords
        pts_out = dataset.outlet_coords
        e_in = dataset.inlet_edges
        e_out = dataset.outlet_edges
        idx_in = dataset.idx_in
        idx_out = dataset.idx_out

        train_set = set(dataset.train_indices)
        rpm_groups: dict[float, list[tuple[float, float, bool]]] = {}

        for idx, sample in enumerate(dataset.samples):
            q_in = float(sample["params"][0])
            rpm = float(sample["params"][1])
            p_vol = sample["volume_data"][:, 0]
            vx = (
                sample["volume_data"][:, 1]
                if sample["volume_data"].shape[1] > 1
                else np.zeros_like(p_vol)
            )
            vy = (
                sample["volume_data"][:, 2]
                if sample["volume_data"].shape[1] > 2
                else np.zeros_like(p_vol)
            )

            h = compute_pump_head(
                p_vol, vx, vy, idx_in, idx_out, pts_in, pts_out, e_in, e_out
            )

            if rpm not in rpm_groups:
                rpm_groups[rpm] = []
            rpm_groups[rpm].append((q_in, h, idx in train_set))
    except Exception as e:
        print(
            f"Notice: Could not compute CFD H-Q curves ({e}), falling back to parameter space plot."
        )
        train_set = set(dataset.train_indices)
        rpm_groups = {}
        for idx, sample in enumerate(dataset.samples):
            q_in = float(sample["params"][0])
            rpm = float(sample["params"][1])
            if rpm not in rpm_groups:
                rpm_groups[rpm] = []
            rpm_groups[rpm].append((q_in, q_in * rpm, idx in train_set))

    _fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=170)
    cmap = plt.get_cmap("tab10")

    for k, (rpm, points) in enumerate(sorted(rpm_groups.items())):
        points.sort(key=lambda x: x[0])  # Sort by Q_in
        q_vals = np.array([p[0] for p in points])
        h_vals = np.array([p[1] for p in points])
        is_train = np.array([p[2] for p in points])

        color = cmap(k % 10)
        ax.plot(q_vals, h_vals, "-", color=color, alpha=0.35, lw=1.2)

        # Validation Points (dots)
        val_mask = ~is_train
        if np.any(val_mask):
            ax.plot(
                q_vals[val_mask],
                h_vals[val_mask],
                "o",
                color=color,
                ms=5.5,
                alpha=0.85,
            )

        # Training Points ('X' overlay)
        if np.any(is_train):
            ax.plot(
                q_vals[is_train],
                h_vals[is_train],
                "x",
                color="black",
                ms=9,
                mew=2.2,
            )
            ax.plot(q_vals[is_train], h_vals[is_train], "x", color=color, ms=7, mew=1.5)

    title_str = (
        f"Experiment Dataset H-Q Split ({exp_name})\n"
        if exp_name
        else "Experiment Dataset H-Q Split\n"
    )
    ax.set_title(
        f"{title_str}"
        f"● Available / Validation Points ({len(dataset.test_indices)}) | "
        f"✖ Training Points ({len(dataset.train_indices)})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Flow Rate $Q_{in}$ [m$^3$/h]", fontsize=11)
    ax.set_ylabel("Pump Head $H$ [m]", fontsize=11)
    ax.grid(True, ls=":", alpha=0.6)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            label="Available / Validation Point (●)",
            markerfacecolor="gray",
            markersize=7,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            label="Training Point (✖)",
            markeredgewidth=2,
            markersize=8,
            linestyle="None",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved dataset H-Q split plot to: {save_path}")
    return save_path


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: str,
    exp_name: str = "",
) -> str:
    """Plots training vs validation loss curves over epochs.

    Args:
        train_losses: List of training loss values per epoch.
        val_losses: List of validation loss values per epoch.
        save_path: Destination filepath for plot.
        exp_name: Optional experiment name for title.

    Returns:
        str: Destination filepath where plot was saved.
    """
    epochs = list(range(1, len(train_losses) + 1))
    _fig, ax = plt.subplots(figsize=(9, 5), dpi=170)

    ax.plot(epochs, train_losses, "b-", lw=1.8, label="Train Loss (MSE)")
    ax.plot(epochs, val_losses, "r--", lw=1.8, label="Val Loss (MSE)")

    ax.set_yscale("log")
    title_str = (
        f"Optimization Loss Progression ({exp_name})"
        if exp_name
        else "Optimization Loss Progression"
    )
    ax.set_title(title_str, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (MSE, Log Scale)", fontsize=11)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved loss progression curve to: {save_path}")
    return save_path


def plot_field_contours(
    dataset: Pump2DDataset,
    model: torch.nn.Module,
    feature_manager: FeatureManager,
    device: torch.device,
    save_dir: str,
    sample_idx: int = 0,
) -> list[str]:
    """Plots side-by-side Ground Truth | SMART Prediction | Error contours for all fields.

    Args:
        dataset: Test dataset instance.
        model: Trained SMART model.
        feature_manager: FeatureManager instance.
        device: Hardware device.
        save_dir: Destination folder where plots are saved.
        sample_idx: Test sample index to visualize.

    Returns:
        List of generated plot paths.
    """
    sample = dataset[sample_idx]
    raw_params = dataset.samples[dataset.active_indices[sample_idx]]["params"]
    q_in, rot_rpm = float(raw_params[0]), float(raw_params[1])

    geom = sample["geometry"].unsqueeze(0).to(device)
    vol_c = sample["volume_coords"].unsqueeze(0).to(device)
    vol_e = sample["volume_extra"].unsqueeze(0).to(device)
    vol_d_norm = sample["volume_data"]
    params = sample["params"].unsqueeze(0).to(device)

    extra_vol = feature_manager.extract_extra_features(vol_e)
    surf_c, _ = feature_manager.prepare_surface_queries({"geometry": geom}, device)

    model.eval()
    with torch.no_grad():
        _, pred_vol_norm = model.inference(
            geom,
            surf_c,
            vol_c,
            params,
            extra_surf_features=None,
            extra_vol_features=extra_vol,
        )

    mean_vol = dataset.mean_vol_data
    std_vol = dataset.std_vol_data

    truth_vol = vol_d_norm.numpy() * std_vol + mean_vol
    pred_vol = pred_vol_norm[0].cpu().numpy() * std_vol + mean_vol

    truth_p, pred_p = truth_vol[:, 0], pred_vol[:, 0]
    truth_vx, pred_vx = truth_vol[:, 1], pred_vol[:, 1]
    truth_vy = truth_vol[:, 2] if truth_vol.shape[1] > 2 else np.zeros_like(truth_vx)
    pred_vy = pred_vol[:, 2] if pred_vol.shape[1] > 2 else np.zeros_like(pred_vx)

    truth_mag = np.sqrt(truth_vx**2 + truth_vy**2)
    pred_mag = np.sqrt(pred_vx**2 + pred_vy**2)

    pts = dataset.samples[0]["volume_coords"]
    triangulation = tri.Triangulation(pts[:, 0], pts[:, 1], dataset.triangles)

    plots = []
    field_configs = [
        ("Pressure [Pa]", truth_p, pred_p, "pressure_comparison.png", "coolwarm"),
        ("Velocity X [m/s]", truth_vx, pred_vx, "velocity_x_comparison.png", "viridis"),
        ("Velocity Y [m/s]", truth_vy, pred_vy, "velocity_y_comparison.png", "viridis"),
        (
            "Velocity Magnitude [m/s]",
            truth_mag,
            pred_mag,
            "velocity_mag_comparison.png",
            "plasma",
        ),
    ]

    subtitle = f"$Q_{{in}}={q_in:.2f}$ m$^3$/h, {rot_rpm:.0f} RPM"
    for title, truth_arr, pred_arr, filename, cmap in field_configs:
        out_path = pjoin(save_dir, filename)
        plot_side_by_side(
            triangulation=triangulation,
            truth=truth_arr,
            pred=pred_arr,
            title=title,
            save_path=out_path,
            subtitle=subtitle,
            cmap=cmap,
        )
        plots.append(out_path)

    print(f"Saved {len(plots)} field contour plots to: {save_dir}")
    return plots


def plot_hq_head_predictions(
    dataset: Pump2DDataset,
    model: torch.nn.Module,
    feature_manager: FeatureManager,
    device: torch.device,
    save_path: str,
    exp_name: str = "",
) -> tuple[str, dict[str, float]]:
    """Calculates CFD vs SMART pump heads for all operating points and plots performance curves.

    Args:
        dataset: Test or full Pump2DDataset instance.
        model: Trained SMART model.
        feature_manager: FeatureManager instance.
        device: Hardware device.
        save_path: Destination filepath for plot.
        exp_name: Optional experiment name for title.

    Returns:
        Tuple of [save_path, head_metrics_dictionary].
    """
    pts_in = dataset.inlet_coords
    pts_out = dataset.outlet_coords
    e_in = dataset.inlet_edges
    e_out = dataset.outlet_edges
    idx_in = dataset.idx_in
    idx_out = dataset.idx_out

    mean_vol = dataset.mean_vol_data
    std_vol = dataset.std_vol_data

    mean_params = dataset.mean_params
    std_params = dataset.std_params

    model.eval()
    train_set = set(dataset.train_indices)
    rpm_groups: dict[float, list[dict[str, Any]]] = {}
    head_errors = []
    head_rel_errors = []

    for idx, sample in enumerate(dataset.samples):
        q_in = float(sample["params"][0])
        rpm = float(sample["params"][1])

        # 1. Ground Truth physical head from raw unnormalized volume data
        p_vol_true = sample["volume_data"][:, 0]
        vx_true = sample["volume_data"][:, 1]
        vy_true = sample["volume_data"][:, 2]
        h_true = compute_pump_head(
            p_vol_true, vx_true, vy_true, idx_in, idx_out, pts_in, pts_out, e_in, e_out
        )

        # 2. SMART inference with normalized parameter inputs
        norm_params = (sample["params"] - mean_params) / std_params
        params_tensor = (
            torch.tensor(norm_params, dtype=torch.float32).unsqueeze(0).to(device)
        )
        geom = (
            torch.tensor(sample["geometry"], dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        vol_c = (
            torch.tensor(sample["volume_coords"], dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )
        vol_e = (
            torch.tensor(sample["volume_extra"], dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

        extra_vol = feature_manager.extract_extra_features(vol_e)
        surf_c, _ = feature_manager.prepare_surface_queries({"geometry": geom}, device)

        with torch.no_grad():
            _, pred_vol_norm = model.inference(
                geom,
                surf_c,
                vol_c,
                params_tensor,
                extra_surf_features=None,
                extra_vol_features=extra_vol,
            )

        # 3. De-normalize SMART volume outputs back to physical units
        pred_vol_phys = pred_vol_norm[0].cpu().numpy() * std_vol + mean_vol
        p_vol_pred = pred_vol_phys[:, 0]
        vx_pred = pred_vol_phys[:, 1]
        vy_pred = pred_vol_phys[:, 2]
        h_pred = compute_pump_head(
            p_vol_pred, vx_pred, vy_pred, idx_in, idx_out, pts_in, pts_out, e_in, e_out
        )

        is_train = idx in train_set
        abs_err = abs(h_true - h_pred)
        rel_err = abs_err / h_true if h_true > 1e-4 else 0.0

        if not is_train:
            head_errors.append(abs_err)
            head_rel_errors.append(rel_err)

        if rpm not in rpm_groups:
            rpm_groups[rpm] = []
        rpm_groups[rpm].append(
            {
                "q_in": q_in,
                "h_true": h_true,
                "h_pred": h_pred,
                "is_train": is_train,
            }
        )

    mean_head_mae = float(np.mean(head_errors)) if head_errors else 0.0
    mean_head_rel = float(np.mean(head_rel_errors) * 100.0) if head_rel_errors else 0.0

    # Plot H-Q Performance Curves
    _fig, ax = plt.subplots(figsize=(10, 6), dpi=170)
    cmap = plt.get_cmap("tab10")

    for k, (rpm, points) in enumerate(sorted(rpm_groups.items())):
        points.sort(key=lambda x: x["q_in"])
        q_vals = np.array([p["q_in"] for p in points])
        h_true_vals = np.array([p["h_true"] for p in points])
        h_pred_vals = np.array([p["h_pred"] for p in points])

        color = cmap(k % 10)
        # True CFD curve (solid line + circular markers)
        ax.plot(
            q_vals,
            h_true_vals,
            "-o",
            color=color,
            lw=1.8,
            ms=4,
            label=f"{rpm:.0f} RPM (CFD)",
        )
        # SMART Predicted curve (dashed line + square markers)
        ax.plot(
            q_vals,
            h_pred_vals,
            "--s",
            color=color,
            lw=1.5,
            ms=4.5,
            alpha=0.85,
            label=f"{rpm:.0f} RPM (SMART)",
        )

    title_str = (
        f"Pump Performance H-Q Curves ({exp_name})\n"
        if exp_name
        else "Pump Performance H-Q Curves\n"
    )
    ax.set_title(
        f"{title_str}"
        f"Validation Head MAE: {mean_head_mae:.4f} m | Rel Error: {mean_head_rel:.2f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Flow Rate $Q_{in}$ [m$^3$/h]", fontsize=11)
    ax.set_ylabel("Head $H$ [m]", fontsize=11)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved H-Q pump head prediction curves to: {save_path}")
    return save_path, {
        "val_Head_MAE_m": mean_head_mae,
        "val_Head_Rel_Error_pct": mean_head_rel,
    }

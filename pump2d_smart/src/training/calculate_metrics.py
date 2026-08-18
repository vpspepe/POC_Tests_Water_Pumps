"""Script to calculate quantitative validation metrics for the trained 2D SMART model.

Computes physical MSE, MAE, Relative L1, Relative L2, and R2 scores across all validation samples.
"""

import os
from os.path import join as pjoin
from typing import Any

import numpy as np
import torch
from models.smart.smart import SMART
from src.data.dataset import Pump2DDataset
from src.training.config import Pump2DConfig, load_hydra_config
from src.training.feature_manager import FeatureManager
from src.training.metrics import compute_field_metrics


def calculate_validation_metrics(
    config: Pump2DConfig | Any | None = None,
    checkpoint_path: str = "",
) -> dict[str, dict[str, float]]:
    """Computes error metrics over all validation samples.

    Args:
        config: Pump2DConfig instance or Hydra DictConfig. If None, loads defaults.
        checkpoint_path: Explicit path to model checkpoint.

    Returns:
        Dictionary of computed metrics by field.
    """
    if config is None:
        try:
            config = load_hydra_config()
        except Exception:
            config = Pump2DConfig()

    cfg_training = getattr(config, "training", config)
    cfg_model = getattr(config, "model", config)
    cfg_features = getattr(config, "features", config)
    cfg_data = getattr(config, "data", config)

    device_str = getattr(cfg_training, "device", "cuda")
    device = torch.device(
        device_str if torch.cuda.is_available() and device_str != "cpu" else "cpu"
    )

    feature_manager = FeatureManager(
        predict_surface=bool(getattr(cfg_features, "predict_surface", False)),
        surface_channels=int(getattr(cfg_features, "surface_channels", 0)),
        volume_channels=int(getattr(cfg_features, "volume_channels", 3)),
        use_sdf_blades=bool(getattr(cfg_features, "use_sdf_blades", True)),
        use_sdf_mrf=bool(getattr(cfg_features, "use_sdf_mrf", True)),
    )

    test_split = getattr(cfg_data, "test_split", 0.5)
    test_split_val = float(test_split) if test_split is not None else None
    n_train = getattr(cfg_data, "n_train", None)
    n_train_val = int(n_train) if n_train is not None else None

    test_dataset = Pump2DDataset(
        main_vtu_path=getattr(cfg_data, "main_vtu_path", ""),
        outlet_vtu_path=getattr(cfg_data, "outlet_vtu_path", ""),
        cache_dir=getattr(cfg_data, "cache_dir", "./cache"),
        if_test=True,
        test_split=test_split_val,
        n_train=n_train_val,
        sparse_hq_split=bool(getattr(cfg_data, "sparse_hq_split", True)),
    )

    save_dir = getattr(cfg_training, "save_model_dir", "./checkpoints")
    ckpt_file = checkpoint_path or pjoin(save_dir, "best_smart_pump2d.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    model = SMART(
        spatial_dim=int(getattr(cfg_model, "spatial_dim", 2)),
        surface_channels=feature_manager.surface_channels,
        volume_channels=feature_manager.volume_channels,
        parameter_channels=int(getattr(cfg_model, "parameter_channels", 2)),
        latent_dim=int(getattr(cfg_model, "latent_dim", 64)),
        latent_geometry_points=int(getattr(cfg_model, "latent_geometry_points", 64)),
        subsampled_geometry_points=int(
            getattr(cfg_model, "subsampled_geometry_points", 128)
        ),
        num_encoder_decoder_blocks=int(
            getattr(cfg_model, "num_encoder_decoder_blocks", 2)
        ),
        num_heads=int(getattr(cfg_model, "num_heads", 4)),
        pos_scale_factor=float(getattr(cfg_model, "pos_scale_factor", 1000.0)),
        dropout=float(getattr(cfg_model, "dropout", 0.0)),
        extra_query_dim=feature_manager.extra_query_dim,
        subregion_size=int(getattr(cfg_model, "subregion_size", 1000)),
    )

    checkpoint = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    fields = ["Volume_Pressure", "Velocity_X", "Velocity_Y", "Velocity_Magnitude"]
    metrics: dict[str, dict[str, list[float]]] = {
        f: {"MSE": [], "MAE": [], "RelL1": [], "RelL2": [], "R2": []} for f in fields
    }

    mean_vol = test_dataset.mean_vol_data
    std_vol = test_dataset.std_vol_data

    print(f"Calculating metrics across {len(test_dataset)} validation samples...")

    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]

        geom = sample["geometry"].unsqueeze(0).to(device)
        vol_c = sample["volume_coords"].unsqueeze(0).to(device)
        vol_e = sample["volume_extra"].unsqueeze(0).to(device)
        vol_d_norm = sample["volume_data"]
        params = sample["params"].unsqueeze(0).to(device)

        extra_vol = feature_manager.extract_extra_features(vol_e)
        surf_c, _ = feature_manager.prepare_surface_queries({"geometry": geom}, device)

        with torch.no_grad():
            _, pred_vol_norm = model.inference(
                geom,
                surf_c,
                vol_c,
                params,
                extra_surf_features=None,
                extra_vol_features=extra_vol,
            )

        truth_vol_phys = vol_d_norm.numpy() * std_vol + mean_vol
        pred_vol_phys = pred_vol_norm[0].cpu().numpy() * std_vol + mean_vol

        truth_p = truth_vol_phys[:, 0]
        pred_p = pred_vol_phys[:, 0]
        truth_vx = truth_vol_phys[:, 1]
        pred_vx = pred_vol_phys[:, 1]
        truth_vy = (
            truth_vol_phys[:, 2]
            if truth_vol_phys.shape[1] > 2
            else np.zeros_like(truth_vx)
        )
        pred_vy = (
            pred_vol_phys[:, 2]
            if pred_vol_phys.shape[1] > 2
            else np.zeros_like(pred_vx)
        )

        truth_mag = np.sqrt(truth_vx**2 + truth_vy**2)
        pred_mag = np.sqrt(pred_vx**2 + pred_vy**2)

        pairs = [
            ("Volume_Pressure", truth_p, pred_p),
            ("Velocity_X", truth_vx, pred_vx),
            ("Velocity_Y", truth_vy, pred_vy),
            ("Velocity_Magnitude", truth_mag, pred_mag),
        ]

        for fname, y_t, y_p in pairs:
            res = compute_field_metrics(y_t, y_p)
            for mname, mval in res.items():
                metrics[fname][mname].append(mval)

    # Print summary table
    print("\n| Predicted Field | MAE | MSE | Rel L1 | Rel L2 | R² Score |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    summary: dict[str, dict[str, float]] = {}
    for f in fields:
        summary[f] = {m: float(np.mean(vals)) for m, vals in metrics[f].items()}
        print(
            f"| **{f}** | {summary[f]['MAE']:.4e} | {summary[f]['MSE']:.4e} | "
            f"{summary[f]['RelL1']:.4f} | {summary[f]['RelL2']:.4f} | {summary[f]['R2']:.4f} |"
        )

    return summary

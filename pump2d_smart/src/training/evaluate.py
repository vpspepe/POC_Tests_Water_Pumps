"""Inference and visualization script for SMART 2D Pump predictions.

Loads the trained model checkpoint, runs prediction on a validation sample,
de-normalizes prediction targets, and plots comparison contours using `src.training.plotting`.
"""

import os
from os.path import join as pjoin
from typing import Any

import torch
from smart.smart.models.smart.smart import SMART
from src.data.dataset import Pump2DDataset
from src.training.config import Pump2DConfig, load_hydra_config
from src.training.feature_manager import FeatureManager
from src.training.plotting import plot_field_contours, plot_side_by_side

__all__ = ["evaluate_model", "plot_side_by_side"]


def evaluate_model(
    config: Pump2DConfig | Any | None = None,
    checkpoint_path: str = "",
    save_dir: str = "",
    sample_idx: int = 0,
) -> list[str]:
    """Loads weights and plots predictions on a validation operating point."""
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

    # Load dataset to extract stats and coordinates
    print("Loading dataset stats...")
    test_dataset = Pump2DDataset(
        main_vtu_path=getattr(cfg_data, "main_vtu_path", ""),
        outlet_vtu_path=getattr(cfg_data, "outlet_vtu_path", ""),
        cache_dir=getattr(cfg_data, "cache_dir", "./cache"),
        if_test=True,
        test_split=float(getattr(cfg_data, "test_split", 0.5)),
        sparse_hq_split=bool(getattr(cfg_data, "sparse_hq_split", True)),
    )

    # Load model configuration
    chk_dir = getattr(cfg_training, "save_model_dir", "./checkpoints")
    ckpt_file = checkpoint_path or pjoin(chk_dir, "best_smart_pump2d.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    print(f"Instantiating model from checkpoint: {ckpt_file}")
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

    out_dir = save_dir or pjoin(
        getattr(cfg_training, "experiments_dir", "./experiments"),
        getattr(config, "exp_name", "exp_eval"),
        "plots",
    )

    return plot_field_contours(
        dataset=test_dataset,
        model=model,
        feature_manager=feature_manager,
        device=device,
        save_dir=out_dir,
        sample_idx=sample_idx,
    )

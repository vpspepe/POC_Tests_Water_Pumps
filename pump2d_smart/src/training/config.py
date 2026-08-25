"""Configuration Module for 2D Pump Surrogate Modeling.

Defines structured dataclasses corresponding to nested Hydra config groups
(model, training, optimizer, lr_scheduler, early_stopping, features, loss, data, tracking).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for SMART neural operator architecture."""

    spatial_dim: int = 2
    parameter_channels: int = 2
    latent_dim: int = 64
    latent_geometry_points: int = 64
    subsampled_geometry_points: int = 128
    num_encoder_decoder_blocks: int = 2
    num_heads: int = 4
    pos_scale_factor: float = 1000.0
    dropout: float = 0.0
    subregion_size: int = 1000


@dataclass
class TrainingConfig:
    """Configuration for training loop and hardware parameters."""

    epochs: int = 50
    batch_size: int = 2
    seed: int = 42
    device: str = "cuda"
    num_threads: int = 2
    num_workers: int = 0
    save_model_dir: str = "./checkpoints"
    experiments_dir: str = "./experiments"


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    type: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class LRSchedulerConfig:
    """Configuration for learning rate scheduler."""

    type: str = "reduce_on_plateau"
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-6
    threshold: float = 1e-4
    T_max: int = 50
    eta_min: float = 1e-6
    gamma: float = 0.95
    verbose: bool = True


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping monitor."""

    enabled: bool = True
    patience: int = 15
    min_delta: float = 1e-4
    mode: str = "min"
    metric: str = "val_loss"
    verbose: bool = True


@dataclass
class FeaturesConfig:
    """Configuration for selective features, SDFs, and prediction modes."""

    predict_surface: bool = False
    surface_channels: int = 0
    volume_channels: int = 3
    use_sdf_blades: bool = True
    use_sdf_mrf: bool = True


@dataclass
class PhysicsTermsConfig:
    """Weights for physics-informed penalty terms in loss."""

    mass_weight: float = 0.0
    flux_weight: float = 0.0
    outlet_p_weight: float = 0.0


@dataclass
class LossConfig:
    """Configuration for composite and physics loss terms."""

    type: str = "volume_rel_l2"
    surface_fields: list[str] = field(default_factory=list)
    volume_fields: list[str] = field(
        default_factory=lambda: ["pressure", "velocity_x", "velocity_y"]
    )
    physics_terms: PhysicsTermsConfig = field(default_factory=PhysicsTermsConfig)


@dataclass
class DataConfig:
    """Configuration for raw VTU paths and dataset splits."""

    main_vtu_path: str = "/home/vpspepe/Documents/TUD/HiWi/Ecotwin/data/raw/comsol/Pump2D_MRFStudy_finer_Main.vtu"
    outlet_vtu_path: str = "/home/vpspepe/Documents/TUD/HiWi/Ecotwin/data/raw/comsol/Pump2D_MRFStudy_finer_Outlet.vtu"
    cache_dir: str = "./cache"
    test_split: float | None = 0.5
    sparse_hq_split: bool = True
    n_train: int | None = None
    seed: int = 42


@dataclass
class TrackingConfig:
    """Configuration for MLflow experiment tracking."""

    enabled: bool = True
    experiment_name: str = "Pump2D_SMART_Surrogate"
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str = "./mlflow_artifacts"
    log_model_artifacts: bool = True
    log_config_yaml: bool = True



@dataclass
class Pump2DConfig:
    """Master structured configuration container."""

    exp_name: str = "exp_003_hydra_volume"
    resume_from_checkpoint: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


def load_hydra_config(
    config_path: str = "../conf",
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> Any:
    """Loads and composes Hydra DictConfig safely.

    Args:
        config_path: Relative or absolute path to conf directory.
        config_name: Main configuration YAML file name.
        overrides: List of CLI-style override strings.

    Returns:
        Composed OmegaConf DictConfig.
    """
    import os

    from hydra import compose, initialize_config_dir

    abs_conf_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "conf")
    )
    with initialize_config_dir(config_dir=abs_conf_dir, version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides or [])

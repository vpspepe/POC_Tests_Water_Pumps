"""Trainer Module for SMART 2D Pump Surrogate with Hydra & MLflow Tracking.

Manages data loaders, optimizer, learning rate schedulers (ReduceLROnPlateau,
CosineAnnealingLR, ExponentialLR), Early Stopping, selective SDF features,
and comprehensive MLflow metric logging.
"""

import os
from os.path import join as pjoin
from typing import Any

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from smart.smart.models.smart.smart import SMART
from src.data.dataset import Pump2DDataset
from src.loss.losses import CombinedLoss, RelL2Loss
from src.loss.physics_losses import (
    ECOTWINPhysicsLoss,
)
from src.training.config import Pump2DConfig
from src.training.early_stopping import EarlyStopping
from src.training.experiment_manager import ExperimentManager
from src.training.feature_manager import FeatureManager
from src.training.lr_scheduler import build_lr_scheduler
from src.training.metrics import compute_field_metrics
from src.training.plotting import (
    plot_field_contours,
    plot_hq_data_split,
    plot_hq_head_predictions,
    plot_loss_curves,
)
from torch.utils.data import DataLoader

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


class Pump2DTrainer:
    """Object-oriented training manager for 2D pump neural operators."""

    def __init__(self, config: Pump2DConfig | DictConfig | Any) -> None:
        """Initializes trainer, hardware limits, model, optimizer, and tracking.

        Args:
            config: Pump2DConfig dataclass or Hydra DictConfig.
        """
        self.config = config

        # Unpack nested configs safely whether dataclass or DictConfig
        self.exp_name = getattr(config, "exp_name", "exp_default")
        self.cfg_training = getattr(config, "training", config)
        self.cfg_model = getattr(config, "model", config)
        self.cfg_optimizer = getattr(config, "optimizer", config)
        self.cfg_scheduler = getattr(config, "lr_scheduler", config)
        self.cfg_early_stopping = getattr(config, "early_stopping", config)
        self.cfg_features = getattr(config, "features", config)
        self.cfg_loss = getattr(config, "loss", config)
        self.cfg_data = getattr(config, "data", config)
        self.cfg_tracking = getattr(config, "tracking", config)

        # 1. Hardware, seeds, and feature manager
        self._init_hardware_and_seeds()
        self._init_feature_manager()

        # 2. Datasets and loaders
        self._init_data_loaders()

        # 3. Model instantiation
        self._init_model()

        # 4. Optimization & scheduling
        self._init_optimizer_and_scheduler()

        # 5. Early stopping monitor
        self._init_early_stopping()

        # 6. Loss function
        self._init_loss_function()

        # 7. Experiment directories & tracking
        self._init_experiment_tracking()

        # 8. Resume from checkpoint if specified
        self._resume_checkpoint_if_specified()

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _init_hardware_and_seeds(self) -> None:
        """Configures CPU multithreading, target device, and random seeds."""
        num_threads = int(getattr(self.cfg_training, "num_threads", 2))
        torch.set_num_threads(num_threads)
        req_device = getattr(self.cfg_training, "device", "cuda")
        self.device = torch.device(
            req_device if torch.cuda.is_available() and req_device != "cpu" else "cpu"
        )
        print(f"Trainer device: {self.device} | PyTorch CPU threads: {num_threads}")
        self._set_seed(int(getattr(self.cfg_training, "seed", 42)))

    def _set_seed(self, seed: int) -> None:
        """Sets random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_feature_manager(self) -> None:
        """Initializes selective SDF and boundary feature routing."""
        self.feature_manager = FeatureManager(
            predict_surface=bool(getattr(self.cfg_features, "predict_surface", False)),
            surface_channels=int(getattr(self.cfg_features, "surface_channels", 0)),
            volume_channels=int(getattr(self.cfg_features, "volume_channels", 3)),
            use_sdf_blades=bool(getattr(self.cfg_features, "use_sdf_blades", True)),
            use_sdf_mrf=bool(getattr(self.cfg_features, "use_sdf_mrf", True)),
        )
        self.extra_query_dim = self.feature_manager.extra_query_dim

    def _init_data_loaders(self) -> None:
        """Instantiates training and validation datasets and data loaders."""
        cache_dir = getattr(self.cfg_data, "cache_dir", "./cache")
        main_vtu = getattr(self.cfg_data, "main_vtu_path", "")
        outlet_vtu = getattr(self.cfg_data, "outlet_vtu_path", "")
        n_train = getattr(self.cfg_data, "n_train", None)
        test_split = getattr(self.cfg_data, "test_split", None)
        if n_train is not None:
            test_split = None
        elif test_split is None:
            test_split = 0.5

        sparse_hq_split = bool(getattr(self.cfg_data, "sparse_hq_split", True))
        seed = int(getattr(self.cfg_training, "seed", 42))

        self.train_dataset = Pump2DDataset(
            main_vtu_path=main_vtu,
            outlet_vtu_path=outlet_vtu,
            cache_dir=cache_dir,
            if_test=False,
            test_split=test_split,
            sparse_hq_split=sparse_hq_split,
            n_train=n_train,
            seed=seed,
        )
        self.test_dataset = Pump2DDataset(
            main_vtu_path=main_vtu,
            outlet_vtu_path=outlet_vtu,
            cache_dir=cache_dir,
            if_test=True,
            test_split=test_split,
            sparse_hq_split=sparse_hq_split,
            n_train=n_train,
            seed=seed,
        )

        batch_size = int(getattr(self.cfg_training, "batch_size", 2))
        num_workers = int(getattr(self.cfg_training, "num_workers", 0))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def _init_model(self) -> None:
        """Instantiates the SMART neural operator model and counts parameters."""
        self.model = SMART(
            spatial_dim=int(getattr(self.cfg_model, "spatial_dim", 2)),
            surface_channels=self.feature_manager.surface_channels,
            volume_channels=self.feature_manager.volume_channels,
            parameter_channels=int(getattr(self.cfg_model, "parameter_channels", 2)),
            latent_dim=int(getattr(self.cfg_model, "latent_dim", 64)),
            latent_geometry_points=int(
                getattr(self.cfg_model, "latent_geometry_points", 64)
            ),
            subsampled_geometry_points=int(
                getattr(self.cfg_model, "subsampled_geometry_points", 128)
            ),
            num_encoder_decoder_blocks=int(
                getattr(self.cfg_model, "num_encoder_decoder_blocks", 2)
            ),
            num_heads=int(getattr(self.cfg_model, "num_heads", 4)),
            pos_scale_factor=float(getattr(self.cfg_model, "pos_scale_factor", 1000.0)),
            dropout=float(getattr(self.cfg_model, "dropout", 0.0)),
            extra_query_dim=self.extra_query_dim,
            subregion_size=int(getattr(self.cfg_model, "subregion_size", 1000)),
        ).to(self.device)

        self.model.initialize_weights()

        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            f"SMART Model instantiated | Total Parameters: {self.total_params:,} "
            f"| Trainable: {self.trainable_params:,} | Extra Query Dim: {self.extra_query_dim}"
        )

    def _init_optimizer_and_scheduler(self) -> None:
        """Configures the optimizer (Adam/AdamW) and learning rate scheduler."""
        lr = float(getattr(self.cfg_optimizer, "learning_rate", 1e-3))
        weight_decay = float(getattr(self.cfg_optimizer, "weight_decay", 1e-4))
        opt_type = getattr(self.cfg_optimizer, "type", "adamw").lower()

        if opt_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

        self.scheduler = build_lr_scheduler(self.optimizer, self.cfg_scheduler)

    def _init_early_stopping(self) -> None:
        """Instantiates the EarlyStopping monitor."""
        self.early_stopping = EarlyStopping(
            enabled=bool(getattr(self.cfg_early_stopping, "enabled", True)),
            patience=int(getattr(self.cfg_early_stopping, "patience", 15)),
            min_delta=float(getattr(self.cfg_early_stopping, "min_delta", 1e-4)),
            mode=str(getattr(self.cfg_early_stopping, "mode", "min")),
            verbose=bool(getattr(self.cfg_early_stopping, "verbose", True)),
        )

    def _init_loss_function(self) -> None:
        """Builds the composite loss function and initializes ECOTWINPhysicsLoss."""
        loss_type = str(getattr(self.cfg_loss, "type", "mse")).lower()
        if loss_type == "l1":
            base_loss = torch.nn.L1Loss(reduction="mean")
        elif loss_type == "rel_l2":
            base_loss = RelL2Loss(dim=-2, reduction="sum", reduce_all=True)
        else:
            base_loss = torch.nn.MSELoss(reduction="mean")

        fields = {
            "surface": ["pressure"]
            if self.feature_manager.surface_channels > 0
            else [],
            "volume": ["pressure", "velocity_x", "velocity_y"][
                : self.feature_manager.volume_channels
            ],
        }
        self.loss_fn = CombinedLoss(base_loss, fields)

        # 2. Physics-Informed Loss Manager
        cfg_phys = getattr(self.cfg_loss, "physics_terms", None)
        w_mass = float(
            getattr(
                cfg_phys, "mass_weight", getattr(cfg_phys, "continuity_weight", 0.0)
            )
        )
        w_flux = float(
            getattr(cfg_phys, "flux_weight", getattr(cfg_phys, "momentum_weight", 0.0))
        )
        w_outlet_p = float(
            getattr(
                cfg_phys,
                "outlet_p_weight",
                getattr(cfg_phys, "boundary_penalty_weight", 0.0),
            )
        )
        w_wall = float(getattr(cfg_phys, "wall_bc_weight", 0.0))

        self.physics_loss = ECOTWINPhysicsLoss(
            weight_mass=w_mass,
            weight_wall=w_wall,
            weight_flux=w_flux,
            weight_outlet_p=w_outlet_p,
        )
        self.use_physics = (
            w_mass > 0.0 or w_flux > 0.0 or w_outlet_p > 0.0 or w_wall > 0.0
        )

        # 3. Pre-allocated boundary tensors on target device
        self.idx_in_tensor = torch.as_tensor(
            self.train_dataset.idx_in, device=self.device
        )
        self.idx_out_tensor = torch.as_tensor(
            self.train_dataset.idx_out, device=self.device
        )
        self.norm_in_tensor = torch.as_tensor(
            self.train_dataset.inlet_normals, dtype=torch.float32, device=self.device
        )
        self.norm_out_tensor = torch.as_tensor(
            self.train_dataset.outlet_normals, dtype=torch.float32, device=self.device
        )
        self.weights_in_tensor = torch.as_tensor(
            self.train_dataset.inlet_weights, dtype=torch.float32, device=self.device
        )
        self.weights_out_tensor = torch.as_tensor(
            self.train_dataset.outlet_weights, dtype=torch.float32, device=self.device
        )
        self.e_in_tensor = torch.as_tensor(
            self.train_dataset.inlet_edges, dtype=torch.int64, device=self.device
        )
        self.e_out_tensor = torch.as_tensor(
            self.train_dataset.outlet_edges, dtype=torch.int64, device=self.device
        )

        # Target normalized outlet pressure (100 kPa)
        self.target_p_out_norm = (
            100000.0 - self.train_dataset.mean_vol_data[0]
        ) / self.train_dataset.std_vol_data[0]

    def _compute_physics_loss(
        self, pred_vol: torch.Tensor, vol_coords: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Evaluates ECOTWINPhysicsLoss for the batch.

        Args:
            pred_vol: Predicted volume field tensor [p, vx, vy].
            vol_coords: Collocation spatial coordinates (with requires_grad=True).

        Returns:
            Tuple of [weighted_scalar_physics_loss, metrics_dict].
        """
        # Node velocities at boundaries
        u_in_nodes = pred_vol[0, self.idx_in_tensor, 1:3]
        u_out_nodes = pred_vol[0, self.idx_out_tensor, 1:3]

        # Mid-segment velocities for line integrals
        u_in_edges = 0.5 * (
            u_in_nodes[self.e_in_tensor[:, 0]] + u_in_nodes[self.e_in_tensor[:, 1]]
        )
        u_out_edges = 0.5 * (
            u_out_nodes[self.e_out_tensor[:, 0]] + u_out_nodes[self.e_out_tensor[:, 1]]
        )

        p_out_pred = pred_vol[..., self.idx_out_tensor, 0]

        return self.physics_loss(
            vol_u_x=pred_vol[..., 1],
            vol_u_y=pred_vol[..., 2],
            vol_coords=vol_coords,
            inlet_u_pred=u_in_edges,
            inlet_normals=self.norm_in_tensor,
            inlet_weights=self.weights_in_tensor,
            outlet_u_pred=u_out_edges,
            outlet_normals=self.norm_out_tensor,
            outlet_weights=self.weights_out_tensor,
            q_in=0.0,
            outlet_p_pred=p_out_pred,
            outlet_p_target=self.target_p_out_norm,
        )

    def _init_experiment_tracking(self) -> None:
        """Sets up isolated experiment directory structure and checkpoint paths."""
        self.exp_manager = ExperimentManager(self.config)
        self.save_model_dir = self.exp_manager.checkpoints_dir
        self.plots_dir = self.exp_manager.plots_dir

        self.best_checkpoint_path = pjoin(self.save_model_dir, "best_smart_pump2d.pt")

    def _resume_checkpoint_if_specified(self) -> None:
        """Resumes weights, epoch, score, and optimizer state from checkpoint if requested."""
        self.start_epoch = 1
        resume_path = getattr(self.config, "resume_from_checkpoint", "")
        if not (resume_path and os.path.exists(resume_path)):
            return

        print(f"Resuming weights from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        if "best_score" in ckpt:
            self.early_stopping.best_score = float(ckpt["best_score"])
            print(f"Restored best score: {self.early_stopping.best_score:.6f}")
        elif "val_rel_l1" in ckpt:
            self.early_stopping.best_score = float(ckpt["val_rel_l1"])
            print(
                f"Restored best score (val_rel_l1): {self.early_stopping.best_score:.6f}"
            )
        elif "val_loss" in ckpt:
            self.early_stopping.best_score = float(ckpt["val_loss"])
            print(
                f"Restored best score (val_loss): {self.early_stopping.best_score:.6f}"
            )

        if "epoch" in ckpt:
            self.start_epoch = int(ckpt["epoch"]) + 1
            print(f"Resuming from epoch: {self.start_epoch}")

        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("Successfully restored optimizer momentum state.")
            except Exception as e:
                print(f"Notice: Could not restore optimizer momentum ({e})")

    # =========================================================================
    # Training & Validation Loops
    # =========================================================================

    def _forward_batch(
        self, batch: dict[str, Any], requires_grad_coords: bool = False
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Unified helper to extract batch tensors, prepare queries, and run SMART forward pass.

        Args:
            batch: Dictionary batch from DataLoader.
            requires_grad_coords: If True, enables autograd on volume coordinates for PDE loss.

        Returns:
            Tuple of [pred_surf, pred_vol, surf_d, vol_d, vol_c].
        """
        geo = batch["geometry"].to(self.device)
        vol_c = batch["volume_coords"].to(self.device)
        if requires_grad_coords:
            vol_c.requires_grad_(True)

        vol_e = batch["volume_extra"].to(self.device)
        vol_d = batch["volume_data"].to(self.device)
        params = batch["params"].to(self.device)

        extra_features = self.feature_manager.extract_extra_features(vol_e)
        surf_c, surf_d = self.feature_manager.prepare_surface_queries(
            batch, self.device
        )

        if requires_grad_coords and torch.cuda.is_available():
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                pred_surf, pred_vol = self.model(
                    geo, surf_c, vol_c, params, extra_query_features=extra_features
                )
        else:
            pred_surf, pred_vol = self.model(
                geo, surf_c, vol_c, params, extra_query_features=extra_features
            )

        return pred_surf, pred_vol, surf_d, vol_d, vol_c

    def train_epoch(self) -> tuple[float, dict[str, float]]:
        """Trains the model for one epoch and evaluates loss component gradient norms.

        Returns:
            Tuple of [mean_epoch_loss, gradient_norms_dict].
        """
        self.model.train()
        epoch_loss = 0.0
        req_grad = self.use_physics and self.physics_loss.weight_mass > 0.0
        grad_norms: dict[str, float] = {}

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            pred_surf, pred_vol, surf_d, vol_d, vol_c = self._forward_batch(
                batch, requires_grad_coords=req_grad
            )

            loss, loss_components = self.loss_fn.forward_with_components(
                pred_surf, pred_vol, surf_d, vol_d
            )

            if self.use_physics:
                phys_loss, _ = self._compute_physics_loss(pred_vol, vol_c)
                loss = loss + phys_loss

            # Compute component gradient norms on last batch of epoch via loss_fn
            if batch_idx == len(self.train_loader) - 1:
                grad_norms = self.loss_fn.compute_gradient_norms(
                    loss_components, self.model.parameters()
                )

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader), grad_norms

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluates model loss and regression metrics on the validation split.

        Returns:
            Flat dictionary mapping metric names to scalar values.
        """
        self.model.eval()
        val_loss = 0.0

        for batch in self.test_loader:
            pred_surf, pred_vol, surf_d, vol_d, _ = self._forward_batch(batch)
            loss = self.loss_fn(pred_surf, pred_vol, surf_d, vol_d)
            val_loss += loss.item()

        mean_val_loss = val_loss / len(self.test_loader)
        field_metrics = self._calculate_validation_metrics()

        # Calculate mean Relative L1 error across volume fields (p, vx, vy)
        rel_l1_vals = [
            field_metrics[f]["RelL1"]
            for f in ["Volume_Pressure", "Velocity_X", "Velocity_Y"]
            if f in field_metrics and "RelL1" in field_metrics[f]
        ]
        mean_val_rel_l1 = (
            float(np.mean(rel_l1_vals)) if rel_l1_vals else float(mean_val_loss)
        )

        all_metrics: dict[str, float] = {
            "val_loss": float(mean_val_loss),
            "val_rel_l1": float(mean_val_rel_l1),
        }
        for fname, mdict in field_metrics.items():
            for mname, mval in mdict.items():
                all_metrics[f"{fname}_{mname}"] = float(mval)

        return all_metrics

    def _calculate_validation_metrics(self) -> dict[str, dict[str, float]]:
        """Computes comprehensive regression metrics (R2, RelL1, RelL2, MAE, MSE)."""
        fields = ["Volume_Pressure", "Velocity_X", "Velocity_Y", "Velocity_Magnitude"]
        accum_metrics: dict[str, dict[str, list[float]]] = {
            f: {"MSE": [], "MAE": [], "RelL1": [], "RelL2": [], "R2": []}
            for f in fields
        }

        mean_vol = self.test_dataset.mean_vol_data
        std_vol = self.test_dataset.std_vol_data

        for idx in range(len(self.test_dataset)):
            sample = self.test_dataset[idx]
            geo = sample["geometry"].unsqueeze(0).to(self.device)
            vol_c = sample["volume_coords"].unsqueeze(0).to(self.device)
            vol_e = sample["volume_extra"].unsqueeze(0).to(self.device)
            vol_d_norm = sample["volume_data"]
            params = sample["params"].unsqueeze(0).to(self.device)

            extra_vol = self.feature_manager.extract_extra_features(vol_e)
            surf_c, _ = self.feature_manager.prepare_surface_queries(
                {"geometry": geo}, self.device
            )

            with torch.no_grad():
                _, pred_vol_norm = self.model.inference(
                    geo,
                    surf_c,
                    vol_c,
                    params,
                    extra_surf_features=None,
                    extra_vol_features=extra_vol,
                )

            # De-normalize volume predictions back to physical units
            truth_vol_phys = vol_d_norm.numpy() * std_vol + mean_vol
            pred_vol_phys = pred_vol_norm[0].cpu().numpy() * std_vol + mean_vol

            truth_p, pred_p = truth_vol_phys[:, 0], pred_vol_phys[:, 0]
            truth_vx, pred_vx = truth_vol_phys[:, 1], pred_vol_phys[:, 1]
            truth_vy = truth_vol_phys[:, 2]
            pred_vy = pred_vol_phys[:, 2]

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
                    accum_metrics[fname][mname].append(mval)

        return {
            f: {m: float(np.mean(vals)) for m, vals in mdict.items()}
            for f, mdict in accum_metrics.items()
        }

    # =========================================================================
    # Fit Execution & Stepping Helpers
    # =========================================================================

    def _persist_and_export_config(self) -> tuple[dict[str, Any], str]:
        """Exports standalone YAML and JSON configurations to experiment folder.

        Returns:
            Tuple of [config_dictionary, hydra_yaml_path].
        """
        if isinstance(self.config, DictConfig):
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            yaml_content = OmegaConf.to_yaml(self.config)
        else:
            try:
                conf_obj = OmegaConf.structured(self.config)
                config_dict = OmegaConf.to_container(conf_obj, resolve=True)
                yaml_content = OmegaConf.to_yaml(conf_obj)
            except Exception:
                config_dict = (
                    self.config.__dict__ if hasattr(self.config, "__dict__") else {}
                )
                yaml_content = str(config_dict)

        hydra_yaml_path = pjoin(self.exp_manager.exp_dir, "hydra_config.yaml")
        try:
            with open(hydra_yaml_path, "w") as f:
                f.write(yaml_content)
        except Exception as e:
            print(f"Notice: Could not write hydra_config.yaml ({e})")

        self.exp_manager.save_config(config_dict)
        return config_dict, hydra_yaml_path

    def _flatten_config_for_mlflow(
        self, d: dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> dict[str, Any]:
        """Recursively flattens nested dictionary for MLflow parameter table."""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, dict):
                items.extend(
                    self._flatten_config_for_mlflow(v, new_key, sep=sep).items()
                )
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _step_lr_scheduler(
        self, epoch_metrics: dict[str, float], fallback_val: float
    ) -> None:
        """Steps learning rate scheduler using the metric key configured in Hydra."""
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_metric_key = getattr(self.cfg_scheduler, "metric", "val_rel_l1")
            self.scheduler.step(epoch_metrics.get(lr_metric_key, fallback_val))
        else:
            self.scheduler.step()

    def _step_early_stopping(
        self,
        epoch: int,
        epoch_metrics: dict[str, float],
        fallback_val: float,
    ) -> bool:
        """Steps EarlyStopping monitor using the metric key configured in Hydra."""
        es_metric_key = getattr(self.cfg_early_stopping, "metric", "val_rel_l1")
        es_metric_val = epoch_metrics.get(es_metric_key, fallback_val)

        return self.early_stopping.step(
            current_value=es_metric_val,
            model=self.model,
            save_path=self.best_checkpoint_path,
            extra_checkpoint_data={
                "epoch": epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "monitored_metric": es_metric_key,
                "metric_value": es_metric_val,
            },
        )

    def _generate_post_training_evaluation(
        self,
        train_losses: list[float],
        val_losses: list[float],
        tracking_enabled: bool,
    ) -> None:
        """Generates loss curves, contour comparisons, H-Q performance curves, and logs MLflow artifacts."""
        print("\nGenerating final evaluation figures in experiment folder...")
        try:
            # 1. Loss progression curve
            plot_loss_curves(
                train_losses,
                val_losses,
                save_path=pjoin(self.plots_dir, "loss_curve.png"),
                exp_name=self.exp_name,
            )

            # 2. Restore best weights for evaluation
            if os.path.exists(self.best_checkpoint_path):
                print(
                    f"Loading best model checkpoint for final evaluation: {self.best_checkpoint_path}"
                )
                best_ckpt = torch.load(
                    self.best_checkpoint_path, map_location=self.device
                )
                self.model.load_state_dict(best_ckpt["model_state_dict"])

            # Compute final quantitative field error metrics exclusively on the BEST model
            best_val_raw = self.validate()
            best_final_metrics = {f"best_val_{k}": v for k, v in best_val_raw.items()}

            # 3. Flow field prediction contours (Pressure, Vx, Vy, VelMag)
            plot_field_contours(
                self.test_dataset,
                self.model,
                self.feature_manager,
                self.device,
                save_dir=self.plots_dir,
                sample_idx=0,
            )

            # 4. H-Q Pump head performance curves
            _, head_metrics = plot_hq_head_predictions(
                self.train_dataset,
                self.model,
                self.feature_manager,
                self.device,
                save_path=pjoin(self.plots_dir, "eval_pump_head_curves.png"),
                exp_name=self.exp_name,
            )
            print(
                f"Best Model Validation Head MAE: {head_metrics['val_Head_MAE_m']:.4f} m | "
                f"Rel Error: {head_metrics['val_Head_Rel_Error_pct']:.2f}%"
            )

            # 5. Log all figures, best final metrics, and model checkpoint to MLflow
            if tracking_enabled:
                mlflow.log_metrics(best_final_metrics)
                mlflow.log_metrics(head_metrics)
                self.exp_manager.log_artifacts_to_mlflow(self.best_checkpoint_path)

        except Exception as e:
            print(f"Notice: Error generating post-training figures ({e})")

    def fit(self) -> None:
        """Executes full training epochs, tracks MLflow metrics, and early stops."""
        epochs = int(getattr(self.cfg_training, "epochs", 50))
        tracking_enabled = bool(getattr(self.cfg_tracking, "enabled", True))
        exp_name = getattr(self.cfg_tracking, "experiment_name", "Pump2D_Surrogate")

        if tracking_enabled:
            raw_tracking_uri = getattr(
                self.cfg_tracking, "tracking_uri", "sqlite:///mlflow.db"
            )
            raw_artifact_loc = getattr(
                self.cfg_tracking, "artifact_location", "./mlflow_artifacts"
            )

            pump2d_root = os.path.abspath(
                pjoin(os.path.dirname(__file__), "..", "..")
            )

            # 1. Resolve SQLite Tracking URI (Metrics and experiment metadata)
            if raw_tracking_uri.startswith("sqlite:///"):
                db_subpath = raw_tracking_uri.replace("sqlite:///", "")
                if not os.path.isabs(db_subpath):
                    abs_db_path = os.path.abspath(pjoin(pump2d_root, db_subpath))
                else:
                    abs_db_path = db_subpath
                os.makedirs(os.path.dirname(abs_db_path), exist_ok=True)
                tracking_uri = f"sqlite:///{abs_db_path}"
            elif raw_tracking_uri.startswith("file:"):
                f_path = raw_tracking_uri.replace("file://", "").replace(
                    "file:", ""
                )
                abs_f_path = (
                    os.path.abspath(pjoin(pump2d_root, f_path))
                    if not os.path.isabs(f_path)
                    else f_path
                )
                tracking_uri = f"file://{abs_f_path}"
            else:
                tracking_uri = raw_tracking_uri

            # 2. Resolve Dedicated Artifact Location (Model weights, config files, figures)
            if raw_artifact_loc.startswith("file:"):
                art_path = raw_artifact_loc.replace("file://", "").replace(
                    "file:", ""
                )
            else:
                art_path = raw_artifact_loc

            if not os.path.isabs(art_path):
                abs_artifact_dir = os.path.abspath(pjoin(pump2d_root, art_path))
            else:
                abs_artifact_dir = art_path
            os.makedirs(abs_artifact_dir, exist_ok=True)
            artifact_location_uri = f"file://{abs_artifact_dir}"

            mlflow.set_tracking_uri(tracking_uri)

            # Ensure experiment is created with dedicated artifact location
            try:
                client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
                experiment = client.get_experiment_by_name(exp_name)
                if experiment is None:
                    client.create_experiment(
                        name=exp_name,
                        artifact_location=artifact_location_uri,
                    )
                else:
                    print(
                        f"MLflow tracking on experiment '{exp_name}' (Artifacts at: {experiment.artifact_location})"
                    )
            except Exception as e:
                print(f"Notice: MLflow experiment setup: {e}")

            mlflow.set_experiment(exp_name)

        run_context = (
            mlflow.start_run(run_name=self.exp_name) if tracking_enabled else None
        )

        # 1. Parse and Save Full Hydra Configuration
        config_dict, hydra_yaml_path = self._persist_and_export_config()

        # 2. Initial Dataset H-Q Split Visualization
        try:
            plot_hq_data_split(
                self.train_dataset,
                save_path=pjoin(self.plots_dir, "hq_data_split.png"),
                exp_name=self.exp_name,
            )
        except Exception as e:
            print(f"Notice: Initial data split plot skipped ({e})")

        train_losses: list[float] = []
        val_losses: list[float] = []

        try:
            if tracking_enabled:
                flat_params = self._flatten_config_for_mlflow(config_dict)
                flat_params["total_params"] = self.total_params
                flat_params["trainable_params"] = self.trainable_params
                flat_params["extra_query_dim"] = self.extra_query_dim
                mlflow.log_params({k: str(v)[:490] for k, v in flat_params.items()})

                if os.path.exists(hydra_yaml_path):
                    mlflow.log_artifact(hydra_yaml_path, artifact_path="config")

            total_target_epoch = (
                self.start_epoch + epochs - 1 if self.start_epoch > 1 else epochs
            )
            print(
                f"--- Starting Training Run: {self.exp_name} (Epochs {self.start_epoch} to {total_target_epoch}) ---"
            )

            try:
                for epoch in range(self.start_epoch, total_target_epoch + 1):
                    train_loss, grad_norms = self.train_epoch()
                    val_metrics = self.validate()

                    val_loss = val_metrics["val_loss"]
                    val_rel_l1 = val_metrics.get("val_rel_l1", val_loss)
                    val_r2 = val_metrics.get("Velocity_Magnitude_R2", 0.0)

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

                    current_lr = self.optimizer.param_groups[0]["lr"]

                    epoch_metrics: dict[str, float] = {
                        "train_loss": train_loss,
                        "learning_rate": current_lr,
                        **grad_norms,
                        **val_metrics,
                    }

                    # 1. Step Learning Rate Scheduler
                    self._step_lr_scheduler(epoch_metrics, fallback_val=val_loss)

                    print(
                        f"Epoch {epoch:03d}/{total_target_epoch:03d} | Train Loss (MSE): {train_loss:.5e} "
                        f"| Val Loss (MSE): {val_loss:.5e} | Val Rel L1: {val_rel_l1:.4%} | LR: {current_lr:.2e} "
                        f"| Val R2 (VelMag): {val_r2:.4f}"
                    )

                    # 2. Log ALL metrics to MLflow
                    if tracking_enabled:
                        mlflow.log_metrics(epoch_metrics, step=epoch)

                    # 3. Early Stopping Step (Checkpoints model ONLY when metric improves)
                    should_stop = self._step_early_stopping(
                        epoch=epoch, epoch_metrics=epoch_metrics, fallback_val=val_loss
                    )

                    if should_stop:
                        print(f"Early stopping triggered at epoch {epoch}.")
                        break

            except KeyboardInterrupt:
                print(
                    "\n[Ctrl+C] Training interrupted by user. Proceeding directly to post-training evaluation..."
                )

            print(
                f"Training completed. Best Val Score: {self.early_stopping.best_score:.6f}. "
                f"Checkpoints saved to '{self.save_model_dir}'."
            )

            # --- Post-Training Artifacts & Figures ---
            self._generate_post_training_evaluation(
                train_losses=train_losses,
                val_losses=val_losses,
                tracking_enabled=tracking_enabled,
            )

        finally:
            if run_context is not None:
                mlflow.end_run()

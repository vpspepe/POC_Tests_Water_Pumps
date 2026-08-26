"""Experiment Manager Module for Managing Checkpoint Locations and DVC Pointers.

Responsible exclusively for:
1. Managing checkpoint directory under `pump2d_smart/experiments/<exp_name>/checkpoints/`.
2. Providing run-specific model filenames (`model_{run_id}.pt`) and DVC pointer names (`model_{run_id}.pt.dvc`).
"""

import os
from os.path import join as pjoin
from typing import Any


class ExperimentManager:
    """Object-Oriented Manager for experiment checkpoint directories and DVC pointers."""

    def __init__(self, config: Any) -> None:
        """Initializes the experiment manager and creates the checkpoints directory.

        Args:
            config: Pump2DConfig dataclass or Hydra DictConfig containing exp_name and paths.
        """
        self.config: Any = config
        self.exp_name: str = getattr(config, "exp_name", "exp_default")

        cfg_training = getattr(config, "training", config)
        experiments_base = getattr(cfg_training, "experiments_dir", "./experiments")

        self.exp_dir: str = pjoin(experiments_base, self.exp_name)
        self.checkpoints_dir: str = pjoin(self.exp_dir, "checkpoints")

        # Ensure checkpoints directory exists locally
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def get_model_checkpoint_info(self, run_id: str) -> tuple[str, str, str]:
        """Returns the local checkpoint filepath, model filename, and DVC pointer filename.

        Args:
            run_id: MLflow run identifier or fallback experiment name.

        Returns:
            Tuple of [full_checkpoint_path, model_filename, dvc_pointer_filename].
        """
        model_filename = f"model_{run_id}.pt"
        dvc_pointer = f"{model_filename}.dvc"
        checkpoint_path = pjoin(self.checkpoints_dir, model_filename)
        return checkpoint_path, model_filename, dvc_pointer

    @staticmethod
    def get_model_info_from_run(
        run_id: str, experiments_base: str = "./experiments"
    ) -> tuple[str, str, str]:
        """Retrieves checkpoint path, model filename, and DVC pointer directly from MLflow run tags.

        Args:
            run_id: MLflow run ID.
            experiments_base: Base directory where experiments are stored.

        Returns:
            Tuple of [full_checkpoint_path, model_filename, dvc_pointer_filename].
        """
        import mlflow

        run = mlflow.get_run(run_id)
        tags = run.data.tags

        model_filename = tags.get("model_filename", f"model_{run_id}.pt")
        dvc_pointer = tags.get("dvc_pointer", f"{model_filename}.dvc")
        exp_name = tags.get(
            "mlflow.runName", getattr(run.info, "run_name", "exp_default")
        )

        checkpoint_path = pjoin(
            experiments_base, exp_name, "checkpoints", model_filename
        )
        return checkpoint_path, model_filename, dvc_pointer



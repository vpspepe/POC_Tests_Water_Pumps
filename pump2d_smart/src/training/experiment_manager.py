"""Experiment Manager Module for Organized Surrogate Modeling Experiments.

Responsible exclusively for:
1. Managing isolated experiment directory structures under `pump2d_smart/experiments/<exp_name>/`.
2. Persisting experiment configuration metadata (JSON).
3. Logging experiment artifacts (plots, checkpoints, configurations) to MLflow.
"""

import json
import os
from os.path import join as pjoin
from typing import Any

import mlflow


class ExperimentManager:
    """Object-Oriented Manager for isolated experiment directories, configuration, and MLflow artifacts."""

    def __init__(self, config: Any) -> None:
        """Initializes the experiment manager and creates isolated directories.

        Args:
            config: Pump2DConfig dataclass or Hydra DictConfig containing exp_name and paths.
        """
        self.config: Any = config
        self.exp_name: str = getattr(config, "exp_name", "exp_default")

        cfg_training = getattr(config, "training", config)
        experiments_base = getattr(cfg_training, "experiments_dir", "./experiments")

        self.exp_dir: str = pjoin(experiments_base, self.exp_name)
        self.checkpoints_dir: str = pjoin(self.exp_dir, "checkpoints")
        self.plots_dir: str = pjoin(self.exp_dir, "plots")

        # Ensure directories exist
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def save_config(self, config_dict: dict[str, Any]) -> str:
        """Saves experiment configuration metadata to JSON.

        Args:
            config_dict: Dictionary of configuration parameters.

        Returns:
            str: Path to saved JSON file.
        """
        config_file = pjoin(self.checkpoints_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=4, default=str)
        print(f"Saved experiment configuration to: {config_file}")
        return config_file

    def log_artifacts_to_mlflow(self, best_checkpoint_path: str = "") -> None:
        """Logs all generated figures, JSON configs, and best model weights to MLflow.

        Args:
            best_checkpoint_path: Path to best checkpoint weights file.
        """
        try:
            if os.path.exists(self.plots_dir):
                mlflow.log_artifacts(self.plots_dir, artifact_path="plots")
                print(f"Logged all plots from '{self.plots_dir}' to MLflow.")

            config_json = pjoin(self.checkpoints_dir, "config.json")
            if os.path.exists(config_json):
                mlflow.log_artifact(config_json, artifact_path="config")

            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                mlflow.log_artifact(best_checkpoint_path, artifact_path="checkpoints")
                print(f"Logged best checkpoint '{best_checkpoint_path}' to MLflow.")
        except Exception as e:
            print(f"Notice: MLflow artifact logging encountered: {e}")

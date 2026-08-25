#!/usr/bin/env python3
"""Automated Batch Runner for Physics Loss Ablation Experiments on 2D Centrifugal Pump.

Executes 4 sequential experiments:
1. MSE Only Loss
2. MSE + Mass Loss (weight = 0.01)
3. MSE + Flux Continuity Loss (weight = 0.01)
4. MSE + Outlet Pressure BC Loss (weight = 0.01)

All experiments run:
- SMART neural operator (3 encoder/decoder blocks, concat+fusion MLP, per-block FiLM modulation)
- 15 Training Points (Kennard-Stone Space Filling) | 229 Validation Points
- 400 Epochs with ReduceLROnPlateau (factor=0.5, patience=5)
- EarlyStopping (patience=15 epochs)
- Evaluation & logged metrics exclusively evaluated on the BEST checkpoint model
- Metrics & artifacts logged to centralized POC_Tests/mlruns directory
"""

import os
import sys
import time
from typing import Any

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from src.training.config import load_hydra_config
from src.training.train import Pump2DTrainer

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "exp_005_15pts_mse_only",
        "description": "Experiment 1: Pure MSE Loss (No Physics)",
        "loss_type": "volume_mse",
        "mass_weight": 0.0,
        "flux_weight": 0.0,
        "outlet_p_weight": 0.0,
    },
    {
        "name": "exp_006_15pts_mse_mass",
        "description": "Experiment 2: MSE + Mass Conservation Loss (Divergence-Free, weight=0.01)",
        "loss_type": "physics_informed",
        "mass_weight": 0.01,
        "flux_weight": 0.0,
        "outlet_p_weight": 0.0,
    },
    {
        "name": "exp_007_15pts_mse_flux",
        "description": "Experiment 3: MSE + Mass Flux Line-Integral Continuity Loss (weight=0.01)",
        "loss_type": "physics_informed",
        "mass_weight": 0.0,
        "flux_weight": 0.01,
        "outlet_p_weight": 0.0,
    },
    {
        "name": "exp_008_15pts_mse_pressure_bc",
        "description": "Experiment 4: MSE + Outlet Pressure Dirichlet BC Loss (100 kPa, weight=0.01)",
        "loss_type": "physics_informed",
        "mass_weight": 0.0,
        "flux_weight": 0.0,
        "outlet_p_weight": 0.01,
    },
]


def run_experiment(exp: dict[str, Any]) -> None:
    """Configures and runs a single experiment."""
    print("=" * 80)
    print(f"LAUNCHING: {exp['description']}")
    print(f"Experiment Name: {exp['name']}")
    print(f"Loss Configuration: type={exp['loss_type']}")
    print(
        f"Weights: Mass={exp['mass_weight']} | Flux={exp['flux_weight']} | Outlet P={exp['outlet_p_weight']}"
    )
    print("=" * 80)

    overrides = [
        f"exp_name={exp['name']}",
        "model.num_encoder_decoder_blocks=3",
        "features=volume_only",
        f"loss={exp['loss_type']}",
        f"loss.physics_terms.mass_weight={exp['mass_weight']}",
        f"loss.physics_terms.flux_weight={exp['flux_weight']}",
        f"loss.physics_terms.outlet_p_weight={exp['outlet_p_weight']}",
        "data.n_train=15",
        "data.test_split=null",
        "training.epochs=400",
        "training.device=cuda",
        "early_stopping=default",
        "early_stopping.patience=15",
        "lr_scheduler=reduce_on_plateau",
    ]

    cfg = load_hydra_config(overrides=overrides)
    trainer = Pump2DTrainer(cfg)
    trainer.fit()

    print(f" COMPLETED: {exp['name']}\n")


def main() -> None:
    """Sequentially launches all defined ablation experiments."""
    total_start = time.time()
    print("*" * 80)
    print(
        f"STARTING BATCH RUN OF {len(EXPERIMENTS)} ABLATION EXPERIMENTS (400 MAX EPOCHS EACH)"
    )
    print("*" * 80)

    for i, exp in enumerate(EXPERIMENTS, start=1):
        print(f"\n[Run {i}/{len(EXPERIMENTS)}]")
        start_time = time.time()
        try:
            run_experiment(exp)
            elapsed = time.time() - start_time
            print(f"Run {i} finished in {elapsed / 60:.1f} minutes.")
        except Exception as e:
            print(f" ERROR in run {i} ({exp['name']}): {e}", file=sys.stderr)

    total_elapsed = time.time() - total_start
    print("*" * 80)
    print(
        f"ALL {len(EXPERIMENTS)} EXPERIMENTS COMPLETED IN {total_elapsed / 3600:.2f} HOURS!"
    )
    print("All results, models, and MLflow runs are saved.")
    print("*" * 80)


if __name__ == "__main__":
    main()

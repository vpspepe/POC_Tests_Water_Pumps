# 2D Pump Surrogate Modeling with SMART

This repository contains the production-grade implementation of the 2D centrifugal pump surrogate model built on top of the **SMART** (*Scalable Mesh-free Aerodynamic Simulations from Raw Geometries*) neural operator architecture.

---

## 1. Directory Structure

```text
pump2d_smart/
├── conf/                          # Modular Hydra configuration hierarchy
│   ├── config.yaml                # Master entry point defaults
│   ├── model/                     # Architecture complexity (smart_small, smart_base)
│   ├── training/                  # Batch size, epochs, device, paths
│   ├── optimizer/                 # Adam, AdamW
│   ├── lr_scheduler/              # ReduceLROnPlateau, CosineAnnealing, Exponential, None
│   ├── early_stopping/            # Early stopping patience, min delta, target metric
│   ├── features/                  # Surface prediction & selective SDF feature toggles
│   ├── loss/                      # Pure MSE, RelL2, physics-informed losses
│   ├── data/                      # Dataset paths, test split ratio, caching
│   └── tracking/                  # MLflow experiment tracking settings
├── cache/                         # Pre-processed .npz datasets & metadata JSONs
├── conf/                          # Modular Hydra configuration hierarchy
├── docs/                          # In-depth architectural notes & technical reports
├── experiments/                   # Isolated experiment directories (<exp_name>/plots & checkpoints)
├── mlflow_artifacts/              # Dedicated MLflow artifact store (model weights, plots, configs)
├── mlflow.db                      # Local SQLite tracking database for MLflow
├── src/                           # Source code (data, geometry, loss, training)
├── README.md                      # This user & CLI guide
├── README_architecture.md         # Comprehensive system & architectural specification
└── RESULTS.md                     # Single master ledger tracking all experiment metrics
```

---

## 2. Quick Start & CLI Usage

### A. Run Training with Hydra
To launch training with default settings (or customize parameters via Hydra overrides):

```bash
# Default training run
PYTHONPATH=./src:../smart/smart uv run python -c "
from src.training.config import load_hydra_config
from src.training.train import Pump2DTrainer

cfg = load_hydra_config(overrides=[
    'exp_name=exp_003_hydra_volume',
    'loss=volume_mse',
    'training.epochs=300',
    'training.device=cuda',
    'features=volume_only',
    'early_stopping=default',
    'lr_scheduler=reduce_on_plateau'
])
trainer = Pump2DTrainer(cfg)
trainer.fit()
"
```

### B. Launch MLflow Dashboard
To browse training loss curves, compare parameters across runs, and inspect generated figures:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts
```
Open **`http://localhost:5000`** in your browser.


### C. Evaluate Trained Checkpoint
To run inference on unseen validation cases, compute error tables, and generate contour plots:

```bash
PYTHONPATH=./src:../smart/smart uv run python -c "
from src.training.calculate_metrics import calculate_validation_metrics
from src.training.evaluate import evaluate_model
from src.training.config import load_hydra_config

cfg = load_hydra_config(overrides=['loss=volume_mse', 'features=volume_only'])
calculate_validation_metrics(cfg, checkpoint_path='./experiments/exp_003_hydra_volume/checkpoints/best_smart_pump2d.pt')
evaluate_model(cfg, checkpoint_path='./experiments/exp_003_hydra_volume/checkpoints/best_smart_pump2d.pt', save_dir='./experiments/exp_003_hydra_volume/plots')

"
```

---

## 3. Core Features & Modularity

1. **Pure MSE Loss Optimization:** Backpropagation uses strictly standard Mean Squared Error on normalized fluid flow targets ($p, v_x, v_y$).
2. **Generic Metric Tracking:** `EarlyStopping` and `ReduceLROnPlateau` dynamically look up any metric from the computed validation dictionary (e.g. `metric: "val_rel_l1"` or `metric: "val_loss"`).
3. **Decoupled Architecture:** Follows the Single Responsibility Principle (SRP):
   * `plotting.py`: Pure rendering functions for H-Q curves, loss curves, and field contours.
   * `experiment_manager.py`: Pure filesystem path management and MLflow artifact uploading.
   * `feature_manager.py`: Feature dimension slicing and surface query routing.
4. **Physical Line Integration:** Automatic computation of physical pump head ($H$) using trapezoidal boundary line integrals on inlet and outlet boundaries.

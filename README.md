# RL-based Vibroacoustic Structure Optimization

## Overview

This repository contains Python tools for automating vibroacoustic FEM simulations using COMSOL Multiphysics and training/running surrogate models for fast CASING field prediction.

The current implementation focuses on an oil pan (`oelwanne`) model. The code is used to automate COMSOL simulations, vary tuned mass damper (TMD) parameters, extract simulation results, compute vibroacoustic quantities such as Structural Intensity (STI) and Equivalent Radiated Power (ERP), visualize results, and generate datasets for later analysis or machine learning applications.

---

## 🚀 Getting Started & Data Access (DVC)

To save Git storage, all large binary files—such as precomputed coordinate mappings (`.npy`) and trained surrogate weights (`.pt`/`.pkl`)—are tracked using **DVC (Data Version Control)** and stored in a shared Hessenbox WebDAV folder.

### 1. Clone and Install Dependencies

Clone the repository and install the dependencies using `uv`:

```bash
uv sync
```

*(This sets up the virtual environment and automatically installs all dependencies, including `dvc` and `scienceplots`).*

### 2. Configure Your Hessenbox User & Password

Since credentials and personal WebDAV paths are stored in `.dvc/config.local` (which is git-ignored), you must configure your local connection details:

1. Log in to **next.hessenbox.de**.
2. Go to **Settings** → **Security** → **Devices & sessions**.
3. Generate a new app password (e.g., named `DVC`).
4. Create a local, git-ignored file named **`.env`** in the repository root and add your details:

   ```ini
   HESSENBOX_USER=your_username_or_email
   HESSENBOX_PASSWORD=your_app_password
   ```

5. Run the configuration script:

   ```bash
   uv run configure_dvc.py
   ```

   *(This loads your credentials from `.env` and automatically configures DVC).*
   *(Note: Ensure your Hessenbox account has access to the shared folder `/oelwanne/rlvso_models_mappings/`)*

### 3. Pull all Data & Models

Run the following command to download all model weights and arrays into the correct folders automatically:

```bash
dvc pull
```

---

## Workflow

```text
COMSOL Model
      ↓
Parameter Definition
      ↓
Study 2 Simulation
      ↓
COMSOL Export
      ↓
Data Extraction
      ↓
STI / ERP Evaluation
      ↓
Visualization
      ↓
Logging & Dataset Generation
```

The COMSOL model acts as the source of all simulation data, while Python handles automation, post-processing, visualization, and storage.

---

## Repository Structure

```text
src/
│
├── mappings/               # Static precomputed CAD mesh mappings (normals, coordinates, areas)
│
├── reinforcement_learning/  # RL optimization environments & SAC training
│   └── optimizers/         # Gym environments & model checkpoints
│
├── surrogate_model/        # Core surrogate predictors & sub-packages
│   ├── nn/                 # PyTorch Neural Networks (ERP, STI, Vel)
│   ├── pod_gp/             # Gaussian Process & POD models
│   └── utils/              # Plotting, COMSOL, extracting, and math helpers
│
├── run_oelwanne.py         # Main COMSOL evaluation script
└── run_random_multi_process.py # Dataset generation script
```

---

## Main Scripts

### `run_oelwanne.py`

The primary example script and recommended starting point for understanding the codebase.

The script demonstrates the complete workflow:

1. Start the COMSOL model.
2. Read eigenfrequencies from Study 1.
3. Select a target frequency.
4. Load geometry coordinates.
5. Find valid mesh points for the tuned mass damper and excitation force.
6. Update COMSOL parameters.
7. Run Study 2.
8. Export simulation results.
9. Extract velocity, ERP, energy flux, and STI data.
10. Visualize the results.

This script serves as a reference implementation for working with the existing code.

### `run_random_multi_process.py`

Dataset generation script for large-scale parameter studies.

The script starts multiple COMSOL workers in parallel. Each worker:

1. Starts its own COMSOL client.
2. Selects a random frequency.
3. Selects a random damper mass.
4. Selects a random valid damper position.
5. Runs Study 2.
6. Exports result data.
7. Evaluates ERP.
8. Stores simulation results.
9. Creates normalized differential datasets relative to a reference simulation.

This script is primarily used to generate datasets for later analysis and machine learning experiments.

---

## Utility Modules

### `comsol.py`

Provides a lightweight wrapper around the `mph` interface for COMSOL automation.

Main functionality:

* start COMSOL models
* inspect model parameters
* extract eigenfrequency data
* trigger predefined COMSOL exports

Some COMSOL functionality is accessed through the underlying Java API, for example running evaluation groups.

The module assumes that the required studies, exports, and evaluation groups are already defined inside the COMSOL model.

---

### `extracting.py`

Acts as the interface between FEM exports and the Python workflow.

Responsibilities include:

* loading COMSOL CSV exports
* handling complex-valued simulation data
* converting raw exports into structured dictionaries
* extracting coordinate data
* extracting velocity data
* extracting ERP data
* extracting stress tensor data
* extracting energy flux data

The current workflow primarily uses the solid-model extraction functions. Some shell-model and ANSYS-related functionality remains as legacy or auxiliary code.

Example representation:

```python
{
    "X": ...,
    "Y": ...,
    "Z": ...,
    "velocity_X": ...,
    "velocity_Y": ...,
    "velocity_Z": ...,
    "Stress_tensor_xx": ...,
    "Stress_tensor_xy": ...,
    ...
}
```

This abstraction layer keeps the rest of the code independent from the exact COMSOL export format.

---

### `functions.py`

Contains numerical helper functions and physical post-processing routines.

The most important functionality is the computation of Structural Intensity (STI) from exported stress and velocity data.

Additional functionality includes:

* stress tensor construction
* vector norms
* coordinate transformations
* hemisphere interpolation
* random hemisphere sampling
* Fibonacci hemisphere sampling

The current workflow is based on solid-model calculations. Some shell-model related code remains for historical reasons but is not part of the primary workflow.

---

### `find_points.py`

Provides geometric search and filtering utilities for FEM meshes and point clouds.

Examples include:

* nearest-point searches
* radius-based searches
* line-based searches
* cylinder-based searches
* bounding-box filtering
* geometry outline detection

These functions are primarily used to ensure that selected damper positions correspond to valid points on the FEM geometry.

---

### `plotting.py`

Provides reusable visualization routines for vibroacoustic simulation results.

Supported visualizations include:

* 3D scalar fields
* 3D vector fields
* hemisphere projections
* geometry markers
* bounding box visualization

Typical plotted quantities include:

* velocity magnitude
* COMSOL energy flux
* computed STI
* active STI
* reactive STI

---

### `save_and_log.py`

Handles simulation storage and dataset generation.

Standard simulations are stored as:

```text
simu_<id>_<timestamp>.npz
simu_<id>_<timestamp>_meta.json
```

The `.npz` file contains numerical field data.

The `.json` file contains simulation parameters and metadata.

Additional functionality includes:

* loading simulations
* creating simulation indices
* generating Pandas DataFrames
* filtering datasets
* extracting stored quantities

The module also supports normalized differential storage relative to a reference simulation.

Conceptually:

```text
Reference Simulation
         +
Differential Simulation
         ↓
Reconstructed Field
```

This representation reduces redundancy when generating larger datasets.

---

### `definitions.py`

Contains project-wide constants and helper functions.

Examples include:

* selected frequencies
* available damper masses
* normalization functions

The module is intentionally lightweight and may evolve as the project structure grows.

---

## Experimental Components

### `rl_optimizer.py`

Experimental reinforcement learning draft.

Currently contains early experiments using:

* Gymnasium
* Stable-Baselines3

The module is not yet connected to the COMSOL simulation workflow.

### `surrogate_model/`

Contains preliminary surrogate-model experiments.

These files are work-in-progress and are not currently part of the main workflow.

---

## COMSOL Requirements

The COMSOL model is external to the Python package.

The Python code assumes that the required studies, parameters, exports, and evaluation groups already exist in the `.mph` model.

Expected components include:

### Study 1

Eigenfrequency analysis used to obtain natural frequencies.

### Study 2

Frequency-domain simulation used for vibroacoustic evaluation.

### Exports

The workflow expects predefined exports such as:

```text
Data 1
Data 2
Table 1
Table 2
```

and evaluation groups such as:

```text
eg1
```

The exact configuration is defined inside the COMSOL model itself.

---

## Recommended Usage

COMSOL startup can be time-consuming.

For development and testing it is recommended to:

1. Start a Python terminal.
2. Load the required modules interactively.
3. Start the COMSOL model once.
4. Work within the active session.

This avoids repeated model initialization and significantly speeds up development.

Recommended starting point:

```text
run_oelwanne.py
```

For larger randomized dataset generation:

```text
run_random_multi_process.py
```

---

## Dependencies

### FEM

* COMSOL Multiphysics
* mph

### Numerical Computing

* numpy
* scipy
* pandas

### Visualization

* matplotlib
* scienceplots

### Experimental

* gymnasium
* stable-baselines3

---

## Current State

Actively used:

* COMSOL automation
* oil pan simulation workflow
* solid-model STI calculation
* ERP evaluation
* visualization
* simulation logging
* multiprocessing dataset generation

Experimental:

* reinforcement learning workflows
* surrogate models

The codebase currently serves primarily as a framework for automated vibroacoustic simulations, post-processing, visualization, and dataset generation. Machine learning and reinforcement learning based optimization remain active areas of development.

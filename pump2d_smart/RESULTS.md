# Master Experiment Results Log

This document serves as the single centralized ledger for tracking all training runs, validation metrics, pump head predictions, and evaluation plots for the `pump2d_smart` surrogate model.

---

## 1. Experiments Summary Overview

| Experiment ID | Split Method | Train / Test Ratio | Best Val Loss | Pressure Rel L2 | Velocity Mag Rel L2 | Head MAE [m] | Head Rel Err [%] | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`exp_002_volume_sparse20`](#experiment-002-volume-only-sparse-interleaved-2080) | Sparse Interleaved (Stride=5) | 20% / 80% (53 / 191) | `0.0078` | **2.67%** | **13.76%** | **0.1712 m** | **14.18%** | Completed |
| [`exp_003_hydra_volume_mse`](#experiment-003-hydra-volume-only-mse-training) | Sparse H-Q Interleaved (50/50) | 50% / 50% (122 / 122) | `0.0024` | **1.00%** | **3.77%** | **0.0524 m** | **2.19%** | **Completed** |

---

## 2. Experiment 002: Volume-Only Sparse Interleaved (20% Train / 80% Test)

* **Date:** 2026-08-13
* **Directory:** [`experiments/exp_002_volume_sparse20/`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/experiments/exp_002_volume_sparse20)
* **Goal:** Test surrogate generalization on a minimal training set (20%) using a stride-based interleaved split along each RPM curve.

### A. Hyperparameters
| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Epochs** | 50 | **Batch Size** | 2 |
| **Learning Rate** | 1e-3 (Cosine Annealing) | **Latent Dim** | 64 |
| **Enc/Dec Blocks** | 2 | **Attention Heads** | 4 |
| **Pos Scale Factor** | 0.05 | **Dropout** | 0.0 |
| **Target Fields** | `[pressure, velocity_x, velocity_y]` | **Parameter Norm** | Z-score Standardized |

### B. Quantitative Field Errors (Evaluated on 191 Unseen Test Cases)
| Field | MAE | MSE | Relative $L_2$ Error | Physical Unit |
| :--- | :---: | :---: | :---: | :---: |
| **Volume Pressure ($p$)** | $1.8993 \times 10^3$ | $7.9553 \times 10^6$ | **`0.0267` (2.67%)** | Pa |
| **Velocity X ($v_x$)** | $4.4235 \times 10^{-1}$ | $3.7223 \times 10^{-1}$ | **`0.1900` (19.00%)** | m/s |
| **Velocity Y ($v_y$)** | $4.6710 \times 10^{-1}$ | $4.1062 \times 10^{-1}$ | **`0.1865` (18.65%)** | m/s |
| **Velocity Magnitude ($|v|$)** | $4.7269 \times 10^{-1}$ | $4.1575 \times 10^{-1}$ | **`0.1376` (13.76%)** | m/s |

### C. Integrated Physical Pump Head ($H$) Metrics
* **Average Head MAE:** `0.1712 m` (17.1 cm across the entire operating range)
* **Average Head Relative Error:** `14.18%`

#### Sample Test Case Comparison
| Case | $Q_{in}$ [m$^3$/h] | RPM | $H_{\text{COMSOL}}$ [m] | $H_{\text{SMART}}$ [m] | Absolute Error [m] | Relative Error [%] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 0.50 | 1000 | 1.788 | 1.779 | 0.010 | 0.53% |
| 02 | 1.00 | 1000 | 1.782 | 1.815 | 0.033 | 1.84% |
| 03 | 1.50 | 1000 | 1.767 | 1.842 | 0.074 | 4.21% |
| 04 | 2.00 | 1000 | 1.753 | 1.799 | 0.046 | 2.61% |
| 13 | 0.50 | 1100 | 2.167 | 2.107 | 0.060 | 2.76% |
| 14 | 1.00 | 1100 | 2.165 | 2.130 | 0.034 | 1.59% |
| 16 | 2.00 | 1100 | 2.116 | 2.139 | 0.023 | 1.10% |

### D. Generated Plots & Visualizations
* **Data Split Graph:** [`results/hq_data_split.png`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/results/hq_data_split.png)
* **H-Q Performance Curves:** [`results/eval_pump_head_curves.png`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/results/eval_pump_head_curves.png)
* **Pressure Contour Comparison:** [`results/eval_pressure.png`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/results/eval_pressure.png)
* **Velocity Contours:** [`results/eval_velocity_magnitude.png`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/results/eval_velocity_magnitude.png)

---

## 3. Experiment 003: Hydra Volume-Only MSE Training

* **Date:** 2026-08-14
* **Configuration:** Modular Hydra (`conf/config.yaml`, `loss: volume_mse`, `lr_scheduler: reduce_on_plateau`)
* **Tracking:** MLflow SQLite backend (`sqlite:///mlflow.db`)
* **Goal:** Full 3-channel volume prediction ($p, v_x, v_y$) trained with pure MSE loss, `ReduceLROnPlateau`, and `EarlyStopping`.

### A. Hyperparameters
| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Epochs** | 153 (Early Stopped at Convergence) | **Batch Size** | 2 |
| **Loss Function** | **MSE** (`torch.nn.MSELoss`) | **LR Scheduler** | `ReduceLROnPlateau` |
| **Enc/Dec Blocks** | 2 | **Attention Heads** | 4 |
| **Latent Dim** | 64 | **Total Parameters** | 337,539 |
| **Active SDFs** | Blades (Ch 0) + MRF (Ch 1) | **Target Fields** | `[pressure, velocity_x, velocity_y]` |

### B. Quantitative Field Errors (Evaluated on 122 Unseen Test Cases)
| Field | MAE | MSE | Relative $L_1$ Error | Relative $L_2$ Error | $R^2$ Score | Physical Unit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Volume Pressure ($p$)** | $6.4459 \times 10^2$ | $8.2899 \times 10^5$ | **`0.77%`** | **`1.00%`** | **`0.9954`** | Pa |
| **Velocity X ($v_x$)** | $1.2075 \times 10^{-1}$ | $2.8554 \times 10^{-2}$ | **`4.66%`** | **`5.08%`** | **`0.9973`** | m/s |
| **Velocity Y ($v_y$)** | $1.2758 \times 10^{-1}$ | $3.1621 \times 10^{-2}$ | **`4.58%`** | **`5.04%`** | **`0.9974`** | m/s |
| **Velocity Magnitude ($|v|$)** | $1.3132 \times 10^{-1}$ | $3.3602 \times 10^{-2}$ | **`3.08%`** | **`3.77%`** | **`0.9918`** | m/s |

### C. Generated Plots & Visualizations
* **Dataset H-Q Split Graph:** [`results/hq_data_split.png`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/results/hq_data_split.png) *(Validation points ● vs. Training points ✖)*
* **H-Q Performance Curves:** [`results/eval_pump_head_curves.png`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/results/eval_pump_head_curves.png) *(Validation Head MAE: 0.0524 m | Relative Error: 2.19%)*
* **Pressure Contour Comparison:** [`results/pressure_comparison.png`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/results/pressure_comparison.png)
* **Velocity X Contours:** [`results/velocity_x_comparison.png`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/results/velocity_x_comparison.png)
* **Velocity Y Contours:** [`results/velocity_y_comparison.png`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/results/velocity_y_comparison.png)
* **Velocity Magnitude Contours:** [`results/velocity_mag_comparison.png`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/results/velocity_mag_comparison.png)

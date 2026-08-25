# 2D Pump SMART Surrogate Training Results (Experiment: exp_002_volume_sparse20)

This report documents the final metrics, model parameters, and physical predictions of **Experiment 002**, trained using a **20% Training / 80% Validation Sparse Interleaved Split**.

---

## 1. Model Configuration

| Parameter | Config Value | Explanation |
| :--- | :--- | :--- |
| **Experiment Name** | `exp_002_volume_sparse20` | Isolated experiment directory |
| **Split Ratio** | **20% Train / 80% Test** | 53 Training samples, 191 Validation Test samples |
| **Sampling Strategy** | **Sparse Interleaved (Stride = 5)** | Picks every 5th point per RPM curve for uniform coverage |
| **Latent Dimensions** | `64` | Embedding feature vector length per point |
| **Encoder/Decoder Blocks** | `2` | Stacked self-attention and cross-attention blocks |
| **Epochs** | `50` | Maximum training iterations |
| **Batch Size** | `2` | Operates on 2 unique operating points $(Q_{in}, rot\_rpm)$ per step |
| **Target Fields** | **Volume-Only** | `[pressure, velocity_x, velocity_y]` |
| **Parameter Scaling** | **Z-score Standardized** | Standardizes $[Q_{in}, rot\_rpm]$ inputs |

---

## 2. Quantitative Validation Metrics (191 Unseen Test Cases)

The following metrics are averaged across all **191 validation test samples** (80% of dataset unseen during training):

| Predicted Field | MAE | MSE | Relative L2 Error | Physical Unit |
| :--- | :---: | :---: | :---: | :---: |
| **Volume Pressure ($p_{\text{fluid}}$)** | $1.8993 \times 10^3$ | $7.9553 \times 10^6$$ | **`0.0267`** (2.67%) | Pa |
| **Velocity X ($v_x$)** | $4.4235 \times 10^{-1}$ | $3.7223 \times 10^{-1}$ | **`0.1900`** (19.00%) | m/s |
| **Velocity Y ($v_y$)** | $4.6710 \times 10^{-1}$ | $4.1062 \times 10^{-1}$ | **`0.1865`** (18.65%) | m/s |
| **Velocity Magnitude ($|v|$)** | $4.7269 \times 10^{-1}$ | $4.1575 \times 10^{-1}$ | **`0.1376`** (13.76%) | m/s |

---

## 3. Pump Head Validation Metrics ($H$)

The pump head ($H$ [m]) is computed by integrating the total pressure $p_t = p + \frac{1}{2}\rho (v_x^2 + v_y^2)$ along the inlet and outlet boundary lines using the trapezoidal line integral rule.

Comparing these integrated heads over the **191 validation test samples**:
* **Average Head MAE:** **`0.1712 m`** (Less than 17 cm error across 191 unseen operating points!)
* **Average Head Relative Error:** **`14.18%`**

### Dataset H-Q Split Distribution (Validation Points ● vs Training Points ✖)
![Dataset H-Q Split Distribution](./plots/hq_data_split.png)

### Pump Performance H-Q Curves (Real vs. SMART Predictions)
![Pump Performance Curves](./plots/eval_pump_head_curves.png)

### Sample Head Comparison Table (Sparse 20/80 Test Cases):
| Operating Case | $Q_{in}$ [m$^3$/h] | RPM | $H_{\text{Real}}$ [m] | $H_{\text{SMART}}$ [m] | Abs Error [m] | Rel Error [%] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Case 01 | 0.50 | 1000 | 1.788 | 1.779 | 0.010 | 0.53% |
| Case 02 | 1.00 | 1000 | 1.782 | 1.815 | 0.033 | 1.84% |
| Case 03 | 1.50 | 1000 | 1.767 | 1.842 | 0.074 | 4.21% |
| Case 04 | 2.00 | 1000 | 1.753 | 1.799 | 0.046 | 2.61% |
| Case 13 | 0.50 | 1100 | 2.167 | 2.107 | 0.060 | 2.76% |
| Case 14 | 1.00 | 1100 | 2.165 | 2.130 | 0.034 | 1.59% |
| Case 16 | 2.00 | 1100 | 2.116 | 2.139 | 0.023 | 1.10% |

---

## 4. Checkpoints and Artifacts

All experiment files are isolated in:
* Model Checkpoints: [`experiments/exp_002_volume_sparse20/checkpoints/best_smart_pump2d.pt`](file:///home/vpspepe/Documents/TUD/HiWi/EcoTwin/POC_Tests/pump2d_smart/experiments/exp_002_volume_sparse20/checkpoints/best_smart_pump2d.pt)
* Configuration JSON: [`experiments/exp_002_volume_sparse20/checkpoints/config.json`](file:///home/vpspepe/Documents/TUD/HiWi/EcoTwin/POC_Tests/pump2d_smart/experiments/exp_002_volume_sparse20/checkpoints/config.json)
* Plots Folder: [`experiments/exp_002_volume_sparse20/plots/`](file:///home/vpspepe/Documents/TUD/HiWi/EcoTwin/POC_Tests/pump2d_smart/experiments/exp_002_volume_sparse20/plots/)

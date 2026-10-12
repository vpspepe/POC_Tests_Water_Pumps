# ECOTWIN Cluster Execution & Handoff Guide

> **Target:** Cluster Environment (SLURM / HPC / GPU Nodes)  
> **Workspace:** `POC_Tests/pump2d_smart`  
> **Branch:** `smart2d_refactor_physics`  
> **Author:** Victor Pepe / ECOTWIN AI Pair Programming  
> **Last Updated:** October 2026  

---

## 1. Context & Starting Point

This repository implements the **SMART 2D Neural Operator** surrogate for industrial water pump digital twins (MRF CFD sweeps).

### 1.1 Architectural Milestones Achieved
1. **Upstream SMART Cleanliness:** The upstream Google Research repository (`smart/`) is **100% untouched**. All novel physics logic, boundary routing, and hard constraints are implemented in the wrapper `EcotwinSMART` ([`src/models/ecotwin_smart.py`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/src/models/ecotwin_smart.py)).
2. **Boundary-Only Formulation:** The query domain was sliced from ~12k dense interior mesh nodes down to strictly the **669 physical boundary points** (12 inlet nodes, 13 outlet nodes, 644 impeller blade nodes).
3. **Exact Outlet Pressure Hard Constraint:**
   $$p(\mathbf{x}) = p_{\text{target}} + \tanh(|\alpha| \cdot |d_{\text{out}}(\mathbf{x})|) \cdot \mathcal{N}_p(\mathbf{x})$$
   Enforces exact atmospheric pressure ($p_{\text{target}} = 100\text{ kPa}$) at the outlet boundary analytically with **$< 1.0\text{ Pa}$ error**.
4. **Pre-computed HDF5 Cache:** All 244 positive-head CFD sweeps are stored in [`cache/pump2d_data_cache.h5`](file:///home/vpspepe/Documents/TUD/HiWi/Ecotwin/POC_Tests/pump2d_smart/cache/pump2d_data_cache.h5) with pre-computed Signed Distance Fields (`sdf_blades`, `sdf_mrf`, `sdf_outlet`), boundary edge connectivity, outward unit normals, and arc-length quadrature weights $\Delta l_e$.

---

## 2. Verified Gold-Standard Checkpoints

The following checkpoints represent our verified best models and are our baseline:

| Run ID | Name / Setting | $N_{\text{train}}$ | Formulation | Learning Rate | Head MAE | Head Rel Error | Val Rel $L_1$ | Status |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `f00584bd8fb140d598a105934d1de5b0` | `local_hard_pressure_flux_15pts` | 15 | Hard BC + Soft Flux | **`0.001`** | **0.1292 m** | **6.30%** | **5.92%** | **Best Model (15 Pts)** |
| `2f03d9be07104b08b1c0b6467c880be0` | `local_hard_pressure_flux_only` | 10 | Hard BC + Soft Flux | **`0.001`** | **0.1572 m** | **6.18%** | **7.04%** | **Best Model (10 Pts)** |
| `5b0512a6ea164ec997212b7eea76f3c9` | `exp_lb2_boundary_continue_physics` | 10 | Soft Physics Continue | **`0.001`** | 0.2004 m | ~9.2% | 9.27% | Prior Baseline |
| `e8804517986b47ee9a73f6dc3f9b4d56` | `exp_lb2_boundary_continue_physics` | 10 | Soft Physics Continue | **`0.001`** | 0.2051 m | ~9.2% | 9.23% | Prior Baseline |

> [!IMPORTANT]
> **CRITICAL LESSON LEARNED: DO NOT USE `lr = 0.005`!**  
> We experimentally verified that setting the initial learning rate to `0.005` (5× higher) destabilizes the Transformer attention maps in SMART, causing divergence and poor local minima ($> 25\%$ error).  
> **Always use `lr = 0.001` (`1e-3`) with NAdamW!**

---

## 3. The 9-Experiment Grid Matrix ($lr = 0.001$)

The target task on the cluster is to train the full 9-experiment ablation grid with the verified stable learning rate **`0.001`**:

| # | Exp Name | $N_{\text{train}}$ (KS) | Formulation | Hard Outlet Pressure BC | Soft Incompressible Flux Loss |
| :---: | :--- | :---: | :--- | :---: | :---: |
| **1** | `grid_5pts_mse_only` | 5 | Data MSE Only | `false` | `0.0` |
| **2** | `grid_5pts_hard_bc_only` | 5 | Hard BC Only | `true` | `0.0` |
| **3** | `grid_5pts_hard_bc_flux` | 5 | Hard BC + Flux | `true` | `0.01` (warmup 20, rampup 15) |
| **4** | `grid_10pts_mse_only` | 10 | Data MSE Only | `false` | `0.0` |
| **5** | `grid_10pts_hard_bc_only` | 10 | Hard BC Only | `true` | `0.0` |
| **6** | `grid_10pts_hard_bc_flux` | 10 | Hard BC + Flux | `true` | `0.01` (warmup 20, rampup 15) |
| **7** | `grid_15pts_mse_only` | 15 | Data MSE Only | `false` | `0.0` |
| **8** | `grid_15pts_hard_bc_only` | 15 | Hard BC Only | `true` | `0.0` |
| **9** | `grid_15pts_hard_bc_flux` | 15 | Hard BC + Flux | `true` | `0.01` (warmup 20, rampup 15) |

---

## 4. How to Run on the Cluster

### 4.1 Prerequisites
```bash
# Ensure you are in the workspace
cd POC_Tests/pump2d_smart

# Install/sync dependencies via uv (creates fast virtualenv)
uv sync

# Ensure cache/pump2d_data_cache.h5 is present (copy from local or verify)
ls -lh cache/pump2d_data_cache.h5
```

### 4.2 Automated 9-Experiment Execution
To run all 9 experiments sequentially with automatic evaluation, metric aggregation, and plot generation:

```bash
# Option A: Direct / Interactive GPU Run
uv run python scripts/run_9_experiments_grid.py --lr 0.001

# Option B: SLURM Batch Job
sbatch scripts/submit_grid_cluster.sbatch
```

### 4.3 Running an Individual Experiment
To run any specific configuration manually:

```bash
uv run python -m src.training.train \
  --config-name=local_train \
  exp_name=grid_15pts_hard_bc_flux \
  data.n_train=15 \
  optimizer.learning_rate=0.001 \
  model.use_pressure_constraint=true \
  loss.physics_terms.flux_weight=0.01 \
  loss.physics_terms.mass_weight=0.0 \
  loss.physics_terms.outlet_p_weight=0.0 \
  loss.physics_terms.wall_weight=0.0 \
  loss.physics_terms.warmup_epochs=20 \
  loss.physics_terms.rampup_epochs=15 \
  training.epochs=200 \
  early_stopping.patience=40 \
  lr_scheduler.patience=15 \
  tracking.target=remote \
  tracking.experiment_name=Pump2D_SMART_Surrogate
```

### 4.4 Evaluating Any Trained Checkpoint
```bash
uv run python -m src.training.evaluate \
  --config-name=local_train \
  --checkpoint=experiments/grid_15pts_hard_bc_flux/checkpoints/<model_file>.pt \
  --output-dir=experiments/grid_results/grid_15pts_hard_bc_flux \
  exp_name=grid_15pts_hard_bc_flux \
  data.n_train=15 \
  model.use_pressure_constraint=true \
  loss.physics_terms.flux_weight=0.01
```

---

## 5. Output Artifacts & Results

When `scripts/run_9_experiments_grid.py` finishes, it writes:
1. **Consolidated Metrics:** `experiments/grid_results/grid_summary_metrics.json`
2. **Individual Output Folders:** `experiments/grid_results/<exp_name>/` containing:
   - `eval_pump_head_curves_<exp_name>.png`
   - `pressure_comparison_<exp_name>.png`
   - `velocity_mag_comparison_<exp_name>.png`
   - `eval_metrics.json`
3. **DAGsHub MLflow Tracking:** All runs log live to:  
   🔗 `https://dagshub.com/victor101pepe/POC_Tests_Water_Pumps.mlflow`

---

## 6. Instructions for Antigravity on the Cluster

When Antigravity opens in the cluster:
1. **Verify git status & branch:** Ensure you are on `smart2d_refactor_physics`.
2. **Read this file:** Review `CLUSTER_HANDOFF.md` before executing anything.
3. **Execute the grid:** Run `uv run python scripts/run_9_experiments_grid.py --lr 0.001`.
4. **Collect Results:** Once completed, read `experiments/grid_results/grid_summary_metrics.json` and generate the final comparison table.
5. **Update Presentation:** Embed the generated PNGs into `presentation_max_kannapin.html` and `research_update_max_kannapin.md`.

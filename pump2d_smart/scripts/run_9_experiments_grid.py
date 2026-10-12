#!/usr/bin/env python3
"""Automated Grid Runner for SMART 2D Pump Experiments.

Runs a systematic 9-experiment grid:
- Datasets: 5, 10, 15 Kennard-Stone (KS) training points
- Formulations:
  1. Pure Data MSE (unconstrained)
  2. Data MSE + Dirichlet Hard Boundary Condition on Outlet Pressure
  3. Data MSE + Dirichlet Hard BC + Soft Flux Loss (with curriculum warmup)
- Initial learning rate: 0.001 (1e-3, verified stable for SMART attention)
- Epochs: 200 (with Early Stopping patience=40 and LR Scheduler patience=15)
- Full DAGsHub MLflow tracking
- Automated evaluation, metric logging, and artifact generation
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from os.path import join as pjoin
from typing import Any

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = pjoin(WORKSPACE_DIR, "experiments/grid_results")
ARTIFACT_DIR = os.environ.get(
    "ANTIGRAVITY_ARTIFACT_DIR",
    "/home/vpspepe/.gemini/antigravity-cli/brain/ecc11987-9388-4b26-830a-6d0daea18a14"
    if os.path.exists("/home/vpspepe/.gemini/antigravity-cli/brain/ecc11987-9388-4b26-830a-6d0daea18a14")
    else pjoin(WORKSPACE_DIR, "artifacts")
)

EXPERIMENTS = [
    # --- 5 KS Points ---
    {
        "exp_name": "grid_5pts_mse_only",
        "n_train": 5,
        "use_pressure_constraint": False,
        "flux_weight": 0.0,
        "label": "5 Pts | MSE Only",
    },
    {
        "exp_name": "grid_5pts_hard_bc_only",
        "n_train": 5,
        "use_pressure_constraint": True,
        "flux_weight": 0.0,
        "label": "5 Pts | Hard BC",
    },
    {
        "exp_name": "grid_5pts_hard_bc_flux",
        "n_train": 5,
        "use_pressure_constraint": True,
        "flux_weight": 0.01,
        "label": "5 Pts | Hard BC + Flux",
    },
    # --- 10 KS Points ---
    {
        "exp_name": "grid_10pts_mse_only",
        "n_train": 10,
        "use_pressure_constraint": False,
        "flux_weight": 0.0,
        "label": "10 Pts | MSE Only",
    },
    {
        "exp_name": "grid_10pts_hard_bc_only",
        "n_train": 10,
        "use_pressure_constraint": True,
        "flux_weight": 0.0,
        "label": "10 Pts | Hard BC",
    },
    {
        "exp_name": "grid_10pts_hard_bc_flux",
        "n_train": 10,
        "use_pressure_constraint": True,
        "flux_weight": 0.01,
        "label": "10 Pts | Hard BC + Flux",
    },
    # --- 15 KS Points ---
    {
        "exp_name": "grid_15pts_mse_only",
        "n_train": 15,
        "use_pressure_constraint": False,
        "flux_weight": 0.0,
        "label": "15 Pts | MSE Only",
    },
    {
        "exp_name": "grid_15pts_hard_bc_only",
        "n_train": 15,
        "use_pressure_constraint": True,
        "flux_weight": 0.0,
        "label": "15 Pts | Hard BC",
    },
    {
        "exp_name": "grid_15pts_hard_bc_flux",
        "n_train": 15,
        "use_pressure_constraint": True,
        "flux_weight": 0.01,
        "label": "15 Pts | Hard BC + Flux",
    },
]


def run_single_experiment(spec: dict[str, Any], lr: float = 0.001) -> dict[str, Any]:
    """Runs training and evaluation for a single experiment via isolated subprocesses."""
    exp_name = spec["exp_name"]
    n_train = spec["n_train"]
    use_hard_bc = spec["use_pressure_constraint"]
    flux_weight = spec["flux_weight"]
    label = spec["label"]

    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT: {exp_name} ({label})")
    print(f"Params: n_train={n_train}, Hard BC={use_hard_bc}, Flux Weight={flux_weight}, LR={lr}")
    print("=" * 80 + "\n", flush=True)

    # 1. Launch Training Subprocess
    train_cmd = [
        "uv", "run", "python", "-m", "src.training.train",
        "--config-name=local_train",
        f"exp_name={exp_name}",
        f"data.n_train={n_train}",
        f"optimizer.learning_rate={lr}",
        f"model.use_pressure_constraint={'true' if use_hard_bc else 'false'}",
        f"loss.physics_terms.flux_weight={flux_weight}",
        "loss.physics_terms.mass_weight=0.0",
        "loss.physics_terms.outlet_p_weight=0.0",
        "loss.physics_terms.wall_weight=0.0",
        "loss.physics_terms.warmup_epochs=20",
        "loss.physics_terms.rampup_epochs=15",
        "training.epochs=200",
        "early_stopping.patience=40",
        "lr_scheduler.patience=15",
        "tracking.target=remote",
        "tracking.experiment_name=Pump2D_SMART_Surrogate",
    ]

    t0 = time.time()
    res = subprocess.run(train_cmd, cwd=WORKSPACE_DIR)
    train_duration = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"Training failed for {exp_name} with return code {res.returncode}")

    print(f"\n[DONE] Training {exp_name} completed in {train_duration:.1f} s.", flush=True)

    # 2. Locate Best Checkpoint
    ckpt_dir = pjoin(WORKSPACE_DIR, "experiments", exp_name, "checkpoints")
    model_checkpoints = glob.glob(pjoin(ckpt_dir, "model_*.pt"))
    if not model_checkpoints:
        fallback = pjoin(ckpt_dir, "best_smart_pump2d.pt")
        if os.path.exists(fallback):
            best_checkpoint = fallback
        else:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    else:
        best_checkpoint = max(model_checkpoints, key=os.path.getmtime)

    print(f"Best model checkpoint located: {best_checkpoint}", flush=True)

    # 3. Launch Evaluation Subprocess
    exp_out_dir = pjoin(RESULTS_DIR, exp_name)
    os.makedirs(exp_out_dir, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    eval_cmd = [
        "uv", "run", "python", "-m", "src.training.evaluate",
        "--config-name=local_train",
        f"--checkpoint={best_checkpoint}",
        f"--output-dir={exp_out_dir}",
        f"exp_name={exp_name}",
        f"data.n_train={n_train}",
        f"model.use_pressure_constraint={'true' if use_hard_bc else 'false'}",
        f"loss.physics_terms.flux_weight={flux_weight}",
    ]

    res_eval = subprocess.run(eval_cmd, cwd=WORKSPACE_DIR)
    if res_eval.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {exp_name} with return code {res_eval.returncode}")

    # 4. Copy and Rename Generated Plots
    # Eval pump head curves
    head_src = pjoin(exp_out_dir, "eval_pump_head_curves.png")
    head_dst = pjoin(exp_out_dir, f"eval_pump_head_curves_{exp_name}.png")
    if os.path.exists(head_src):
        shutil.copy2(head_src, head_dst)
        shutil.copy2(head_dst, pjoin(ARTIFACT_DIR, f"eval_pump_head_curves_{exp_name}.png"))

    # Contours
    for fbase in ["pressure_comparison", "velocity_x_comparison", "velocity_y_comparison", "velocity_mag_comparison"]:
        src_f = pjoin(exp_out_dir, f"{fbase}.png")
        dst_f = pjoin(exp_out_dir, f"{fbase}_{exp_name}.png")
        if os.path.exists(src_f):
            shutil.copy2(src_f, dst_f)
            shutil.copy2(dst_f, pjoin(ARTIFACT_DIR, f"{fbase}_{exp_name}.png"))

    # 5. Parse Metrics JSON
    metrics_json_path = pjoin(exp_out_dir, "eval_metrics.json")
    if not os.path.exists(metrics_json_path):
        raise FileNotFoundError(f"Metrics file {metrics_json_path} not found")

    with open(metrics_json_path, "r") as f:
        metrics_data = json.load(f)

    field_m = metrics_data.get("field_metrics", {})
    eng_m = metrics_data.get("engineering_metrics", {})
    head_m = metrics_data.get("head_metrics", {})

    record = {
        "exp_name": exp_name,
        "label": label,
        "n_train": n_train,
        "use_pressure_constraint": use_hard_bc,
        "flux_weight": flux_weight,
        "learning_rate": lr,
        "training_time_s": round(train_duration, 2),
        "best_checkpoint": os.path.basename(best_checkpoint),
        # Head metrics
        "val_Head_MAE_m": head_m.get("val_Head_MAE_m", 0.0),
        "val_Head_Rel_Error_pct": head_m.get("val_Head_Rel_Error_pct", 0.0),
        "Head_Truth_Mean_m": eng_m.get("Head_H [m]", {}).get("truth_mean", 0.0),
        "Head_Pred_Mean_m": eng_m.get("Head_H [m]", {}).get("pred_mean", 0.0),
        # Volume Pressure metrics
        "Pressure_MAE_Pa": field_m.get("Volume_Pressure", {}).get("MAE", 0.0),
        "Pressure_RelL1_pct": field_m.get("Volume_Pressure", {}).get("RelL1", 0.0) * 100.0,
        "Pressure_R2": field_m.get("Volume_Pressure", {}).get("R2", 0.0),
        # Velocity Magnitude metrics
        "Velocity_Mag_MAE_ms": field_m.get("Velocity_Magnitude", {}).get("MAE", 0.0),
        "Velocity_Mag_RelL1_pct": field_m.get("Velocity_Magnitude", {}).get("RelL1", 0.0) * 100.0,
        "Velocity_Mag_R2": field_m.get("Velocity_Magnitude", {}).get("R2", 0.0),
        # Velocity Components
        "Velocity_X_R2": field_m.get("Velocity_X", {}).get("R2", 0.0),
        "Velocity_Y_R2": field_m.get("Velocity_Y", {}).get("R2", 0.0),
        # Integrated Pressure Differences
        "DeltaP_tot_MAE_Pa": eng_m.get("DeltaP_tot [Pa]", {}).get("mae", 0.0),
        "DeltaP_tot_Rel_pct": eng_m.get("DeltaP_tot [Pa]", {}).get("rel_pct", 0.0),
        "DeltaP_static_MAE_Pa": eng_m.get("DeltaP_static [Pa]", {}).get("mae", 0.0),
        "DeltaP_static_Rel_pct": eng_m.get("DeltaP_static [Pa]", {}).get("rel_pct", 0.0),
    }

    print("\nRecorded Summary Metrics:")
    for k, v in record.items():
        print(f"  {k}: {v}")

    return record


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run 9-experiment grid for SMART 2D Pump.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (default 0.001)")
    parser.add_argument("--filter", type=str, default="", help="Filter experiment names by substring")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    summary_file = pjoin(RESULTS_DIR, "grid_summary_metrics.json")
    
    all_results = []
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = []

    completed_names = {r["exp_name"] for r in all_results}

    for spec in EXPERIMENTS:
        if args.filter and args.filter not in spec["exp_name"]:
            continue
        if spec["exp_name"] in completed_names:
            print(f"\nSkipping already completed experiment: {spec['exp_name']}", flush=True)
            continue

        try:
            record = run_single_experiment(spec, lr=args.lr)
            all_results.append(record)
            with open(summary_file, "w") as f:
                json.dump(all_results, f, indent=2)
            with open(pjoin(ARTIFACT_DIR, "grid_summary_metrics.json"), "w") as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"\nERROR during experiment {spec['exp_name']}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {summary_file}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

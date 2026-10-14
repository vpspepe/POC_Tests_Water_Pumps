"""Inference, quantitative metrics evaluation, and visualization pipeline for SMART 2D Pump.

Supports:
1. Quantitative field metrics table (MAE, MSE, Rel L1, Rel L2, R2) across validation cases.
2. Side-by-side ground truth vs SMART prediction vs absolute error contour plots.
3. Multi-RPM pump performance head curves (H-Q).
4. Full CLI interface for evaluating any trained checkpoint.
"""

import argparse
import os
from os.path import join as pjoin
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from smart.smart.models.smart.smart import SMART
from src.models.ecotwin_smart import EcotwinSMART

from src.data.dataset import Pump2DDataset
from src.training.calculate_pump_heads import compute_full_pump_metrics
from src.training.config import Pump2DConfig, load_hydra_config
from src.training.feature_manager import FeatureManager
from src.training.metrics import compute_field_metrics
from src.training.plotting import (
    plot_field_contours,
    plot_hq_head_predictions,
    plot_side_by_side,
)

__all__ = ["evaluate_model", "plot_side_by_side"]


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    default_cfg_model: Any | None = None,
    extra_query_dim: int = 2,
    surface_channels: int = 0,
    volume_channels: int = 3,
    p_target: float = 0.0,
) -> tuple[nn.Module, dict[str, Any]]:
    """Loads and configures SMART model from a checkpoint, handling legacy architectures."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    m_cfg = checkpoint.get("model_config", {})

    # Fallback to defaults if not in checkpoint
    spatial_dim = m_cfg.get("spatial_dim", getattr(default_cfg_model, "spatial_dim", 2))
    latent_dim = m_cfg.get("latent_dim", getattr(default_cfg_model, "latent_dim", 64))
    param_channels = m_cfg.get(
        "parameter_channels", getattr(default_cfg_model, "parameter_channels", 2)
    )
    num_heads = m_cfg.get("num_heads", getattr(default_cfg_model, "num_heads", 4))
    pos_scale_factor = m_cfg.get(
        "pos_scale_factor", getattr(default_cfg_model, "pos_scale_factor", 1000.0)
    )
    dropout = m_cfg.get("dropout", getattr(default_cfg_model, "dropout", 0.0))

    # Detect number of blocks from state dict
    blocks = set()
    for k in state_dict.keys():
        if k.startswith("encoder_blocks."):
            blocks.add(k.split(".")[1])
    num_blocks = len(blocks) if blocks else m_cfg.get("num_encoder_decoder_blocks", 2)

    base_model = SMART(
        spatial_dim=int(spatial_dim),
        surface_channels=surface_channels,
        volume_channels=volume_channels,
        parameter_channels=int(param_channels),
        latent_dim=int(latent_dim),
        num_encoder_decoder_blocks=int(num_blocks),
        num_heads=int(num_heads),
        pos_scale_factor=float(pos_scale_factor),
        dropout=float(dropout),
        extra_query_dim=extra_query_dim,
    )

    # Legacy feature projection detection
    if any("feature_projection" in k for k in state_dict.keys()) and not any(
        "fusion_mlp" in k for k in state_dict.keys()
    ):
        base_model.use_legacy_feature_proj = True

    # If the state_dict comes from EcotwinSMART, the keys will start with "smart."
    is_ecotwin = any(k.startswith("smart.") for k in state_dict.keys())
    
    if is_ecotwin:
        # Wrap it with default targets, we will just load the state_dict over it
        model = EcotwinSMART(
            smart_model=base_model,
            use_pressure_constraint=m_cfg.get("use_pressure_constraint", True),
            pressure_target=p_target,
            sdf_outlet_index=extra_query_dim - 1 if extra_query_dim > 2 else -1,
        )
    else:
        # For backwards compatibility with old checkpoints
        model = base_model

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, checkpoint


def evaluate_model(
    config: Pump2DConfig | Any | None = None,
    checkpoint_path: str = "",
    save_dir: str = "",
    sample_idx: int = 0,
    compute_metrics: bool = True,
    generate_contours: bool = True,
    generate_head_curves: bool = True,
) -> dict[str, Any]:
    """Complete evaluation pipeline: quantitative metrics, contour plots, and head curves."""
    if config is None:
        try:
            config = load_hydra_config()
        except Exception:
            config = Pump2DConfig()

    cfg_training = getattr(config, "training", config)
    cfg_model = getattr(config, "model", config)
    cfg_features = getattr(config, "features", config)
    cfg_data = getattr(config, "data", config)

    device_str = getattr(cfg_training, "device", "cuda")
    device = torch.device(
        device_str if torch.cuda.is_available() and device_str != "cpu" else "cpu"
    )

    feature_manager = FeatureManager(
        predict_surface=bool(getattr(cfg_features, "predict_surface", False)),
        surface_channels=int(getattr(cfg_features, "surface_channels", 0)),
        volume_channels=int(getattr(cfg_features, "volume_channels", 3)),
        use_sdf_blades=bool(getattr(cfg_features, "use_sdf_blades", True)),
        use_sdf_mrf=bool(getattr(cfg_features, "use_sdf_mrf", True)),
        use_sdf_outlet=bool(getattr(cfg_features, "use_sdf_outlet", True)),
    )

    chk_dir = getattr(cfg_training, "save_model_dir", "./checkpoints")
    ckpt_file = checkpoint_path or pjoin(chk_dir, "best_smart_pump2d.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    n_train = getattr(cfg_data, "n_train", None)
    test_split = getattr(cfg_data, "test_split", None)
    if n_train is not None:
        test_split_val = None
        n_train_val = int(n_train)
    else:
        test_split_val = float(test_split) if test_split is not None else 0.5
        n_train_val = None

    print("Loading test dataset...")
    test_dataset = Pump2DDataset(
        main_vtu_path=getattr(cfg_data, "main_vtu_path", ""),
        outlet_vtu_path=getattr(cfg_data, "outlet_vtu_path", ""),
        cache_dir=getattr(cfg_data, "cache_dir", "./cache"),
        if_test=True,
        test_split=test_split_val,
        n_train=n_train_val,
        sparse_hq_split=bool(getattr(cfg_data, "sparse_hq_split", True)),
        boundary_only=bool(getattr(cfg_data, "boundary_only", False)),
    )
    
    # Configurable target outlet pressure [Pa] (normalized to dataset stats)
    target_p_pa = float(getattr(cfg_model, "outlet_pressure_target", 100000.0)) if cfg_model else 100000.0
    try:
        p_target = (target_p_pa - test_dataset.mean_vol_data[0]) / test_dataset.std_vol_data[0]
    except (AttributeError, IndexError):
        p_target = 0.0

    print(f"\n--- Loading Checkpoint: {ckpt_file} ---")
    model, _ = load_model_from_checkpoint(
        checkpoint_path=ckpt_file,
        device=device,
        default_cfg_model=cfg_model,
        extra_query_dim=feature_manager.extra_query_dim,
        surface_channels=feature_manager.surface_channels,
        volume_channels=feature_manager.volume_channels,
        p_target=p_target,
    )

    out_dir = save_dir or pjoin(
        os.path.dirname(os.path.abspath(ckpt_file)), "..", "plots"
    )
    os.makedirs(out_dir, exist_ok=True)

    results: dict[str, Any] = {}

    # 1. Quantitative Validation Metrics
    if compute_metrics:
        print(f"\nEvaluating metrics on {len(test_dataset)} validation samples...")
        fields = ["Volume_Pressure", "Velocity_X", "Velocity_Y", "Velocity_Magnitude"]
        field_metrics: dict[str, dict[str, list[float]]] = {
            f: {"MSE": [], "MAE": [], "RelL1": [], "RelL2": [], "R2": []}
            for f in fields
        }

        mean_vol = test_dataset.mean_vol_data
        std_vol = test_dataset.std_vol_data

        pts_in = test_dataset.inlet_coords
        pts_out = test_dataset.outlet_coords
        e_in = test_dataset.inlet_edges
        e_out = test_dataset.outlet_edges
        idx_in = test_dataset.idx_in
        idx_out = test_dataset.idx_out

        eng_metrics: dict[str, dict[str, list[float]]] = {
            "Head_H [m]": {"truth": [], "pred": [], "err": [], "rel": []},
            "DeltaP_tot [Pa]": {"truth": [], "pred": [], "err": [], "rel": []},
            "DeltaP_static [Pa]": {"truth": [], "pred": [], "err": [], "rel": []},
        }

        with torch.no_grad():
            for idx in range(len(test_dataset)):
                sample = test_dataset[idx]
                geom = sample["geometry"].unsqueeze(0).to(device)
                vol_c = sample["volume_coords"].unsqueeze(0).to(device)
                vol_e = sample["volume_extra"].unsqueeze(0).to(device)
                vol_d_norm = sample["volume_data"]
                params = sample["params"].unsqueeze(0).to(device)

                extra_vol = feature_manager.extract_extra_features(vol_e)
                d_out = vol_e[..., 2:3] if vol_e.shape[-1] >= 3 else None
                surf_c, _ = feature_manager.prepare_surface_queries(
                    {"geometry": geom}, device
                )

                _, pred_vol_norm = model.inference(
                    geom,
                    surf_c,
                    vol_c,
                    params,
                    extra_surf_features=None,
                    extra_vol_features=extra_vol,
                    d_out=d_out,
                )

                truth_vol = vol_d_norm.numpy() * std_vol + mean_vol
                pred_vol = pred_vol_norm[0].cpu().numpy() * std_vol + mean_vol

                truth_p, pred_p = truth_vol[:, 0], pred_vol[:, 0]
                truth_vx, pred_vx = truth_vol[:, 1], pred_vol[:, 1]
                truth_vy = (
                    truth_vol[:, 2]
                    if truth_vol.shape[1] > 2
                    else np.zeros_like(truth_vx)
                )
                pred_vy = (
                    pred_vol[:, 2] if pred_vol.shape[1] > 2 else np.zeros_like(pred_vx)
                )

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
                        field_metrics[fname][mname].append(mval)

                # Integrated physical engineering metrics via PumpStaticROM
                raw_p = test_dataset.samples[test_dataset.active_indices[idx]]["params"]
                q_in_val, rpm_val = float(raw_p[0]), int(raw_p[1])

                m_t = compute_full_pump_metrics(
                    truth_p,
                    truth_vx,
                    truth_vy,
                    idx_in,
                    idx_out,
                    pts_in,
                    pts_out,
                    e_in,
                    e_out,
                    qin=q_in_val,
                    rpm=rpm_val,
                    blades_probe=test_dataset.blades_probe,
                )
                m_p = compute_full_pump_metrics(
                    pred_p,
                    pred_vx,
                    pred_vy,
                    idx_in,
                    idx_out,
                    pts_in,
                    pts_out,
                    e_in,
                    e_out,
                    qin=q_in_val,
                    rpm=rpm_val,
                    blades_probe=test_dataset.blades_probe,
                )

                for k, v_t, v_p in [
                    ("Head_H [m]", m_t.h, m_p.h),
                    ("DeltaP_tot [Pa]", m_t.delta_p_tot, m_p.delta_p_tot),
                    ("DeltaP_static [Pa]", m_t.delta_p, m_p.delta_p),
                ]:
                    err = abs(v_t - v_p)
                    rel = err / abs(v_t) if abs(v_t) > 1e-4 else 0.0
                    eng_metrics[k]["truth"].append(v_t)
                    eng_metrics[k]["pred"].append(v_p)
                    eng_metrics[k]["err"].append(err)
                    eng_metrics[k]["rel"].append(rel)

        summary_metrics: dict[str, dict[str, float]] = {}
        print("\n| Predicted Field | MAE | MSE | Rel L1 [%] | Rel L2 [%] | R² Score |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for f in fields:
            summary_metrics[f] = {
                m: float(np.mean(vals)) for m, vals in field_metrics[f].items()
            }
            rel_l1_pct = summary_metrics[f]["RelL1"] * 100.0
            rel_l2_pct = summary_metrics[f]["RelL2"] * 100.0
            print(
                f"| **{f}** | {summary_metrics[f]['MAE']:.3e} | {summary_metrics[f]['MSE']:.3e} | "
                f"**{rel_l1_pct:.2f}%** | **{rel_l2_pct:.2f}%** | **{summary_metrics[f]['R2']:.4f}** |"
            )
        results["field_metrics"] = summary_metrics

        print(
            "\n| Physical Metric (PumpStaticROM) | Truth (Mean) | SMART Pred (Mean) | MAE | Rel Error [%] |"
        )
        print("| :--- | :---: | :---: | :---: | :---: |")
        summary_eng: dict[str, dict[str, float]] = {}
        for k, v in eng_metrics.items():
            t_mean = float(np.mean(v["truth"]))
            p_mean = float(np.mean(v["pred"]))
            err_mean = float(np.mean(v["err"]))
            rel_mean_pct = float(np.mean(v["rel"])) * 100.0
            summary_eng[k] = {
                "truth_mean": t_mean,
                "pred_mean": p_mean,
                "mae": err_mean,
                "rel_pct": rel_mean_pct,
            }
            print(
                f"| **{k}** | {t_mean:.3e} | {p_mean:.3e} | {err_mean:.3e} | **{rel_mean_pct:.2f}%** |"
            )
        results["engineering_metrics"] = summary_eng

    # 2. Side-by-Side Field Contours
    if generate_contours:
        print(
            f"\nGenerating field contour plots for validation sample #{sample_idx}..."
        )
        contour_plots = plot_field_contours(
            dataset=test_dataset,
            model=model,
            feature_manager=feature_manager,
            device=device,
            save_dir=out_dir,
            sample_idx=sample_idx,
        )
        results["contour_plots"] = contour_plots

    # 3. Pump Head Performance Curves (H-Q)
    if generate_head_curves:
        print("\nGenerating H-Q pump head performance curves...")
        hq_plot_path = pjoin(out_dir, "eval_pump_head_curves.png")
        _, head_metrics = plot_hq_head_predictions(
            dataset=test_dataset,
            model=model,
            feature_manager=feature_manager,
            device=device,
            save_path=hq_plot_path,
            exp_name=getattr(config, "exp_name", "Evaluation"),
        )
        results["head_metrics"] = head_metrics
        results["head_plot"] = hq_plot_path
        print(
            f"Validation Head MAE: {head_metrics['val_Head_MAE_m']:.4f} m | "
            f"Head Rel Error: {head_metrics['val_Head_Rel_Error_pct']:.2f}%"
        )

    # Save metrics JSON to out_dir
    try:
        import json

        def _json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return str(obj)

        metrics_file = pjoin(out_dir, "eval_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2, default=_json_serializable)
        print(f"Saved evaluation metrics to: {metrics_file}")
    except Exception as e:
        print(f"Notice: Failed to save eval_metrics.json ({e})")

    print(f"\nAll evaluation artifacts saved to: {out_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SMART 2D Pump neural surrogate checkpoint."
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Hydra config file name (e.g. local_15pts or local_train).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to trained checkpoint (.pt file).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory where output plots and metrics will be saved.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Validation sample index to visualize contour comparisons.",
    )
    parser.add_argument(
        "--no-contours",
        action="store_true",
        help="Skip generating field contour plots.",
    )
    parser.add_argument(
        "--no-head-curves",
        action="store_true",
        help="Skip generating H-Q pump head curves.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any Hydra key=value overrides.",
    )
    args = parser.parse_args()

    try:
        config = load_hydra_config(
            config_name=args.config_name,
            overrides=args.overrides if args.overrides else None,
        )
    except Exception as e:
        print(f"Warning: Failed to load hydra config '{args.config_name}': {e}. Using defaults.")
        config = None

    evaluate_model(
        config=config,
        checkpoint_path=args.checkpoint,
        save_dir=args.output_dir,
        sample_idx=args.sample_idx,
        generate_contours=not args.no_contours,
        generate_head_curves=not args.no_head_curves,
    )


if __name__ == "__main__":
    main()

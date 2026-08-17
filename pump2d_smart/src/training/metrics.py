"""Evaluation Metrics Module for Flow Field Surrogate Predictions.

Provides vectorized NumPy implementations for evaluating standard regression
and engineering accuracy metrics (R2 score, RelL2, RelL1, MSE, MAE).
"""

import numpy as np


def compute_field_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Computes comprehensive regression metrics for a physical field.

    Args:
        y_true: Ground truth numpy array of shape (N,).
        y_pred: Predicted numpy array of shape (N,).

    Returns:
        Dictionary containing MSE, MAE, RelL1, RelL2, and R2 metrics.
    """
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    sq_diff = diff**2

    mse = float(np.mean(sq_diff))
    mae = float(np.mean(abs_diff))

    # Relative L2 Error: ||y - y_hat||_2 / ||y||_2
    norm_l2_true = float(np.linalg.norm(y_true))
    norm_l2_diff = float(np.linalg.norm(diff))
    rel_l2 = norm_l2_diff / norm_l2_true if norm_l2_true > 1e-12 else 0.0

    # Relative L1 Error: ||y - y_hat||_1 / ||y||_1
    norm_l1_true = float(np.sum(np.abs(y_true)))
    norm_l1_diff = float(np.sum(abs_diff))
    rel_l1 = norm_l1_diff / norm_l1_true if norm_l1_true > 1e-12 else 0.0

    # R^2 (Coefficient of Determination): 1 - SS_res / SS_tot
    ss_res = float(np.sum(sq_diff))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "MSE": mse,
        "MAE": mae,
        "RelL1": rel_l1,
        "RelL2": rel_l2,
        "R2": r2,
    }

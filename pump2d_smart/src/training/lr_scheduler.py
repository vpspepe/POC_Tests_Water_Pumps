"""Learning Rate Scheduler Factory Module.

Provides configurable learning rate decay implementations (ReduceLROnPlateau,
CosineAnnealingLR, ExponentialLR, Constant) using clean factory patterns.
"""

from typing import Any

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LRScheduler,
    ReduceLROnPlateau,
)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
) -> LRScheduler | ReduceLROnPlateau | None:
    """Builds and returns the configured learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer instance.
        cfg: Configuration dictionary or dataclass with scheduler settings.

    Returns:
        Configured PyTorch scheduler, or None if type is 'none'.
    """
    sched_type = getattr(cfg, "type", "none").lower()

    if sched_type == "reduce_on_plateau":
        mode = getattr(cfg, "mode", "min")
        factor = float(getattr(cfg, "factor", 0.5))
        patience = int(getattr(cfg, "patience", 5))
        min_lr = float(getattr(cfg, "min_lr", 1e-6))
        threshold = float(getattr(cfg, "threshold", 1e-4))
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            threshold=threshold,
        )

    elif sched_type == "cosine_annealing":
        t_max = int(getattr(cfg, "T_max", 50))
        eta_min = float(getattr(cfg, "eta_min", 1e-6))
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    elif sched_type == "exponential":
        gamma = float(getattr(cfg, "gamma", 0.95))
        return ExponentialLR(optimizer, gamma=gamma)

    elif sched_type in ("none", "constant"):
        return None

    else:
        raise ValueError(
            f"Unsupported lr_scheduler type: '{sched_type}'. "
            "Supported: ['reduce_on_plateau', 'cosine_annealing', 'exponential', 'none']"
        )

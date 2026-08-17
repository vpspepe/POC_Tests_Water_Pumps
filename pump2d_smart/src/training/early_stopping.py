"""Early Stopping Module for Safe and Efficient Model Training.

Tracks validation performance metrics and automatically terminates optimization
when improvements plateau within the specified patience epochs.
"""

import os
from typing import Any

import torch


class EarlyStopping:
    """Monitors a target metric and halts training when no improvement occurs."""

    def __init__(
        self,
        enabled: bool = True,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        """Initializes the EarlyStopping monitor.

        Args:
            enabled: If False, early stopping is disabled and step() always returns False.
            patience: Number of epochs to wait without improvement before stopping.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            mode: 'min' for decreasing metrics (e.g. loss), 'max' for increasing (e.g. R2).
            verbose: If True, prints status messages upon improvement or plateau.
        """
        self.enabled: bool = enabled
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.mode: str = mode.lower()
        self.verbose: bool = verbose

        self.best_score: float = float("inf") if self.mode == "min" else float("-inf")
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False

    def step(
        self,
        current_value: float,
        model: torch.nn.Module | None = None,
        save_path: str = "",
        extra_checkpoint_data: dict[str, Any] | None = None,
    ) -> bool:
        """Evaluates the current metric and decides whether to trigger early stopping.

        Args:
            current_value: Value of monitored metric in current epoch.
            model: PyTorch model whose state dictionary should be saved on improvement.
            save_path: Filepath to save the best checkpoint weights.
            extra_checkpoint_data: Additional dictionary metadata to include in the checkpoint.

        Returns:
            True if training should be halted, False otherwise.
        """
        if not self.enabled:
            return False

        is_improvement = False
        if self.mode == "min":
            if current_value < self.best_score - self.min_delta:
                is_improvement = True
        else:
            if current_value > self.best_score + self.min_delta:
                is_improvement = True

        if is_improvement:
            if self.verbose:
                print(
                    f"EarlyStopping: Metric improved from {self.best_score:.6f} to {current_value:.6f}."
                )
            self.best_score = current_value
            self.counter = 0

            # Save best checkpoint
            if model is not None and save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "best_score": self.best_score,
                }
                if extra_checkpoint_data:
                    ckpt.update(extra_checkpoint_data)
                torch.save(ckpt, save_path)
                if self.verbose:
                    print(f"EarlyStopping: Saved best model weights to '{save_path}'.")
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs "
                    f"(Best: {self.best_score:.6f})."
                )
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(
                        f"EarlyStopping: Patience of {self.patience} epochs reached. Terminating training."
                    )

        return self.should_stop

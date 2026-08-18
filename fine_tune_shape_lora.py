#!/usr/bin/env python
"""Fine‑tune the SHAPE foundation model with LoRA adapters using Hydra.

All configurable parameters are supplied via a Hydra YAML file (`fine_tune_config.yaml`).
"""

import os
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Add shape directory to sys.path to ensure local imports work
shape_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "shape"))
if os.path.isdir(shape_dir) and shape_dir not in sys.path:
    sys.path.insert(0, shape_dir)

# Monkeypatch shape_foundation._radius_search_native to perform search in chunks.
# This reduces peak VRAM usage by 10x (from 2.26 GB to 226 MB) to prevent CUDA OOM on 6GB GPUs.
import shape_foundation.models.tokenizer_magno as tm
from shape_foundation.configs.default import ShapeConfig, load_config
from shape_foundation.models.gaot_backbone import GAOTBackbone


def chunked_radius_search_native(
    query: torch.Tensor,
    support: torch.Tensor,
    radius: float,
    max_neighbors: int,
    batch_q=None,
    batch_s=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform radius search in chunks to save peak VRAM memory.

    Args:
        query: Query points.
        support: Support points.
        radius: Radius bounds for neighbor search.
        max_neighbors: Maximum neighbor limit.

    Returns:
        Indices of query points and support points.
    """
    chunk_size = 1728
    all_q_idx = []
    all_s_idx = []

    Q = query.shape[0]
    k = min(max_neighbors, support.shape[0])

    for start in range(0, Q, chunk_size):
        end = min(start + chunk_size, Q)
        query_chunk = query[start:end]

        dists_chunk = torch.cdist(query_chunk, support)
        dists_k, indices_k = torch.topk(dists_chunk, k=k, dim=-1, largest=False)
        mask_k = dists_k < radius

        q_idx_local = (
            torch.arange(start, end, device=query.device)
            .unsqueeze(1)
            .expand_as(indices_k)
        )

        all_q_idx.append(q_idx_local[mask_k])
        all_s_idx.append(indices_k[mask_k])

    return torch.cat(all_q_idx), torch.cat(all_s_idx)


tm._radius_search_native = chunked_radius_search_native


def load_shape_model(
    config_path: Path, checkpoint_dir: Path, neighbor_backend: str = "native"
) -> tuple[GAOTBackbone, ShapeConfig]:
    """Initialize the configuration and load backbone weights.

    Args:
        config_path: Path to small.yaml config.
        checkpoint_dir: Path containing checkpoint_final.pt.
        neighbor_backend: Search backend to use.

    Returns:
        A loaded GAOTBackbone model and ShapeConfig.
    """
    cfg = load_config(str(config_path))
    cfg.tokenizer.neighbor.backend = neighbor_backend

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAOTBackbone(cfg).to(device)

    ckpt_path = checkpoint_dir / "checkpoint_final.pt"
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, cfg


class PumpDataset(Dataset):
    """Dataset class that loads cached stochastic views of pump meshes."""

    def __init__(
        self,
        root_dir: Path,
        num_points: int = 8192,
        num_views: int = 10,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Initialize the PumpDataset.

        Args:
            root_dir: Path to directory containing cached views.
            num_points: Number of points to subsample.
            num_views: Number of cached views per model.
            split: Either 'train' or 'val'.
            val_ratio: Fractional size of validation set.
            seed: RNG seed for deterministic splits.
        """
        self.root = Path(root_dir).resolve()
        dirs = sorted(
            [
                d
                for d in self.root.iterdir()
                if d.is_dir() and (d / "view_00.npz").exists()
            ]
        )
        if not dirs:
            raise RuntimeError(f"No cached directories found under {root_dir}")

        state = np.random.RandomState(seed)
        shuffled_dirs = list(dirs)
        state.shuffle(shuffled_dirs)

        val_size = int(len(dirs) * val_ratio)
        if split == "train":
            self.dirs = shuffled_dirs[val_size:]
        elif split == "val":
            self.dirs = shuffled_dirs[:val_size]
        else:
            self.dirs = shuffled_dirs

        self.num_views = num_views
        self.num_points = num_points

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        folder = self.dirs[idx]
        v_a = np.random.randint(0, self.num_views)
        v_b = np.random.randint(0, self.num_views)
        while v_a == v_b:
            v_b = np.random.randint(0, self.num_views)

        data_a = np.load(folder / f"view_{v_a:02d}.npz")
        data_b = np.load(folder / f"view_{v_b:02d}.npz")

        N_a, N_b = data_a["points"].shape[0], data_b["points"].shape[0]
        idx_a = (
            np.random.choice(N_a, self.num_points, replace=False)
            if self.num_points < N_a
            else np.arange(N_a)
        )
        idx_b = (
            np.random.choice(N_b, self.num_points, replace=False)
            if self.num_points < N_b
            else np.arange(N_b)
        )

        return {
            "points_a": torch.from_numpy(data_a["points"][idx_a]).float(),
            "features_a": torch.from_numpy(data_a["features"][idx_a]).float(),
            "normals_a": torch.from_numpy(data_a["normals"][idx_a]).float(),
            "curvature_a": torch.from_numpy(data_a["curvature"][idx_a]).float(),
            "points_b": torch.from_numpy(data_b["points"][idx_b]).float(),
            "features_b": torch.from_numpy(data_b["features"][idx_b]).float(),
            "normals_b": torch.from_numpy(data_b["normals"][idx_b]).float(),
            "curvature_b": torch.from_numpy(data_b["curvature"][idx_b]).float(),
        }


def attach_lora(model: nn.Module, r: int, alpha: int) -> nn.Module:
    """Attach LoRA adapters to the model.

    Args:
        model: Backbone model.
        r: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        The PeftModel wrapped backbone.
    """
    from peft import LoraConfig, get_peft_model

    peft_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        inference_mode=False,
    )
    return get_peft_model(model, peft_cfg)


class MultiResContrastiveLoss(nn.Module):
    """Symmetric InfoNCE Contrastive Loss class."""

    def __init__(self, temperature: float = 0.07) -> None:
        """Initialize the loss module.

        Args:
            temperature: Softmax scaling temperature.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of InfoNCE.

        Args:
            embeddings_a: Embeddings of View A.
            embeddings_b: Embeddings of View B.

        Returns:
            Contrastive loss value.
        """
        a = nn.functional.normalize(embeddings_a, dim=-1)
        b = nn.functional.normalize(embeddings_b, dim=-1)
        logits_ab = a @ b.T / self.temperature
        logits_ba = logits_ab.T
        labels = torch.arange(a.shape[0], device=a.device)
        return (
            nn.functional.cross_entropy(logits_ab, labels)
            + nn.functional.cross_entropy(logits_ba, labels)
        ) / 2


def freeze_latent_grid(model: nn.Module) -> None:
    """Freeze token coordinate embeddings and position encoding parameters.

    Args:
        model: Model containing weights to freeze.
    """
    for name, param in model.named_parameters():
        if "latent" in name.lower() or "token_pos" in name.lower():
            param.requires_grad = False


def prepare_dataloaders(
    data_root: Path, num_points: int, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    """Initialize train and validation data loaders.

    Args:
        data_root: Path to cached dataset.
        num_points: Subsampled resolution points.
        batch_size: Loader batch size.
        num_workers: Multi-process worker count.

    Returns:
        Train and validation loaders.
    """
    train_dataset = PumpDataset(data_root, num_points=num_points, split="train")
    val_dataset = PumpDataset(data_root, num_points=num_points, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_embeddings_and_targets(
    model: nn.Module, batch: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor]:
    """Run forward passes to extract token features and target embeddings.

    Args:
        model: PeftModel backbone.
        batch: Data batch containing features.

    Returns:
        Output A, Output B, and target token states.
    """
    out_a = model.forward_tokens(
        points=batch["points_a"],
        features=batch["features_a"],
        normals=batch.get("normals_a"),
        curvature=batch.get("curvature_a"),
    )
    out_b = model.forward_tokens(
        points=batch["points_b"],
        features=batch["features_b"],
        normals=batch.get("normals_b"),
        curvature=batch.get("curvature_b"),
    )
    with torch.no_grad(), model.disable_adapter():
        tgt_a = model.forward_tokens(
            points=batch["points_a"],
            features=batch["features_a"],
            normals=batch.get("normals_a"),
            curvature=batch.get("curvature_a"),
        )["token_embeddings"]
        tgt_b = model.forward_tokens(
            points=batch["points_b"],
            features=batch["features_b"],
            normals=batch.get("normals_b"),
            curvature=batch.get("curvature_b"),
        )["token_embeddings"]
    return out_a, out_b, tgt_a, tgt_b


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    contrast_criterion: nn.Module,
    accumulation_steps: int,
    device: torch.device,
    recon_weight: float = 0.8,
    contrast_weight: float = 0.2,
) -> tuple[float, float, float]:
    """Run a single training epoch.

    Args:
        model: PeftModel adapter-wrapped.
        loader: Training dataset loader.
        optimizer: Optimization object.
        contrast_criterion: Loss evaluation class.
        accumulation_steps: Steps to pile up samples.
        device: CUDA/CPU device.
        recon_weight: Weight for reconstruction MSE loss.
        contrast_weight: Weight for contrastive InfoNCE loss.

    Returns:
        A tuple of (total_loss, recon_loss, contrastive_loss) averages.
    """
    model.train()
    epoch_recon, epoch_contrast, epoch_total = 0.0, 0.0, 0.0
    accum_embed_a, accum_embed_b, accum_recon_losses = [], [], []
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        out_a, out_b, tgt_a, tgt_b = get_embeddings_and_targets(model, batch)

        recon_loss = 0.5 * (
            nn.functional.mse_loss(out_a["token_embeddings"], tgt_a)
            + nn.functional.mse_loss(out_b["token_embeddings"], tgt_b)
        )
        accum_recon_losses.append(recon_loss)
        accum_embed_a.append(out_a["pooled_embedding"])
        accum_embed_b.append(out_b["pooled_embedding"])

        if len(accum_embed_a) == accumulation_steps or batch_idx == len(loader) - 1:
            emb_a = torch.cat(accum_embed_a, dim=0)
            emb_b = torch.cat(accum_embed_b, dim=0)
            contrast_loss = (
                contrast_criterion(emb_a, emb_b)
                if emb_a.shape[0] > 1
                else torch.tensor(0.0, device=device)
            )

            mean_recon = torch.stack(accum_recon_losses).mean()
            loss = recon_weight * mean_recon + contrast_weight * contrast_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            count = len(accum_recon_losses)
            epoch_recon += mean_recon.item() * count
            epoch_contrast += contrast_loss.item() * count
            epoch_total += loss.item() * count
            accum_embed_a, accum_embed_b, accum_recon_losses = [], [], []

    return (
        epoch_total / len(loader),
        epoch_recon / len(loader),
        epoch_contrast / len(loader),
    )


def run_val_epoch(
    model: nn.Module,
    loader: DataLoader,
    contrast_criterion: nn.Module,
    accumulation_steps: int,
    device: torch.device,
    recon_weight: float = 0.8,
    contrast_weight: float = 0.2,
) -> tuple[float, float, float]:
    """Run validation phase loops.

    Args:
        model: Backbone model.
        loader: Validation dataset loader.
        contrast_criterion: Contrastive loss calculator.
        accumulation_steps: Batch accumulation step configuration.
        device: Active device.
        recon_weight: Weight for reconstruction MSE loss.
        contrast_weight: Weight for contrastive InfoNCE loss.

    Returns:
        Validation (total_loss, recon_loss, contrastive_loss) averages.
    """
    model.eval()
    val_recon, val_contrast, val_total = 0.0, 0.0, 0.0
    val_accum_a, val_accum_b, val_recon_list = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out_a, out_b, tgt_a, tgt_b = get_embeddings_and_targets(model, batch)

            recon_loss = 0.5 * (
                nn.functional.mse_loss(out_a["token_embeddings"], tgt_a)
                + nn.functional.mse_loss(out_b["token_embeddings"], tgt_b)
            )
            val_recon_list.append(recon_loss)
            val_accum_a.append(out_a["pooled_embedding"])
            val_accum_b.append(out_b["pooled_embedding"])

            if len(val_accum_a) == accumulation_steps or batch_idx == len(loader) - 1:
                emb_a = torch.cat(val_accum_a, dim=0)
                emb_b = torch.cat(val_accum_b, dim=0)
                contrast_loss = (
                    contrast_criterion(emb_a, emb_b)
                    if emb_a.shape[0] > 1
                    else torch.tensor(0.0, device=device)
                )

                mean_recon = torch.stack(val_recon_list).mean()
                loss = recon_weight * mean_recon + contrast_weight * contrast_loss

                count = len(val_recon_list)
                val_recon += mean_recon.item() * count
                val_contrast += contrast_loss.item() * count
                val_total += loss.item() * count
                val_accum_a, val_accum_b, val_recon_list = [], [], []

    return val_total / len(loader), val_recon / len(loader), val_contrast / len(loader)


@hydra.main(version_base=None, config_path=".", config_name="fine_tune_config")
def main(cfg: DictConfig) -> None:
    """Main entry point for LoRA fine-tuning of the SHAPE foundation model."""
    print("Hydra configuration:")
    print(OmegaConf.to_yaml(cfg))

    config_path = Path(cfg.config_path).resolve()
    checkpoint_dir = Path(cfg.checkpoint_dir).resolve()
    data_root = Path(cfg.data_root).resolve()
    output_dir = Path(cfg.output_dir).resolve()

    model, _ = load_shape_model(config_path, checkpoint_dir, cfg.neighbor_backend)
    freeze_latent_grid(model)
    model = attach_lora(model, r=cfg.lora_rank, alpha=cfg.lora_alpha)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Trainable parameters with LoRA adapters:")
    model.print_trainable_parameters()

    train_loader, val_loader = prepare_dataloaders(
        data_root, cfg.num_points, cfg.batch_size, cfg.num_workers
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    contrast_criterion = MultiResContrastiveLoss(temperature=cfg.temperature)

    recon_w = cfg.get("recon_weight", 0.8)
    contrast_w = cfg.get("contrast_weight", 0.2)

    for epoch in range(1, cfg.epochs + 1):
        t_loss, t_rec, t_con = run_train_epoch(
            model,
            train_loader,
            optimizer,
            contrast_criterion,
            cfg.accumulation_steps,
            device,
            recon_weight=recon_w,
            contrast_weight=contrast_w,
        )
        v_loss, v_rec, v_con = run_val_epoch(
            model,
            val_loader,
            contrast_criterion,
            cfg.accumulation_steps,
            device,
            recon_weight=recon_w,
            contrast_weight=contrast_w,
        )

        print(
            f"Epoch {epoch:03d}/{cfg.epochs:03d} - "
            f"Train Loss: {t_loss:.5f} (Recon: {t_rec:.5f}, Contrast: {t_con:.5f}) | "
            f"Val Loss: {v_loss:.5f} (Recon: {v_rec:.5f}, Contrast: {v_con:.5f})"
        )

        if epoch % cfg.save_every == 0:
            out_dir = output_dir / f"epoch_{epoch:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir))
            print(f"Saved LoRA adapter checkpoint to {out_dir}")

    final_dir = output_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    print(f"Fine‑tuning complete! Final adapter saved to {final_dir}")


if __name__ == "__main__":
    main()

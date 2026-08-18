---
language:
- en
library_name: pytorch
license: apache-2.0
pipeline_tag: other
tags:
- 3d
- geometry
- point-cloud
- mesh
- cad
- foundation-model
- self-supervised
- masked-modeling
- contrastive-learning
---

# Shape Foundation Model — Small v3

Shape is a self-supervised foundation model that converts surface meshes into dense per-token embeddings for industrial CAD analysis. It combines a structured 3D latent grid, a multi-scale geometry-aware tokenizer (MAGNO), and a transformer processor to enable accurate geometric representations and explainable predictions through a learned reconstruction prior.

The model was introduced in the paper: [Shape: A Self-Supervised 3D Geometry Foundation Model for Industrial CAD Analysis](https://huggingface.co/papers/2604.22826).

[Code](https://github.com/simd-ai/shape) | [Project Page & Demo](https://shape.simd.space)

## Model Details

| | |
|---|---|
| **Architecture** | GAOTBackbone (MAGNO Encoder → Transformer Processor → Task Heads) |
| **Parameters** | 10,913,297 |
| **Training objective** | Self-supervised masked token reconstruction + multi-resolution contrastive learning |
| **Training data** | 61,052 industrial CAD meshes from Fusion360, MFCAD, and Thingi10K |
| **Precision** | bf16 mixed precision |
| **Status** | Self-supervised backbone only — supervised task heads are present but disabled |

## Evaluation Results

Metrics on the held-out validation split (N = 2,983 meshes, deterministic hash-based split):

**Reconstruction (pretraining objective):**

| Metric | Value |
|---|---:|
| SmoothL1 loss (β=1.0) at masked positions | 0.024 |
| Coefficient of determination (R²) | **0.729** |

**Contrastive embedding quality (Wang & Isola 2020):**

| Metric | Value |
|---|---:|
| Top-1 positive-pair retrieval accuracy | **98.1%** |
| Alignment (positive pairs) | 0.132 |
| Uniformity (random pairs) | −3.84 |

## Usage

### Install dependencies

```bash
pip install torch trimesh einops numpy scipy huggingface-hub
```

Note: You also need the `shape_foundation` package from the [official repository](https://github.com/simd-ai/shape).

### Load and run inference

```python
import torch
import trimesh
from shape_foundation.configs.default import ShapeConfig
from shape_foundation.models.gaot_backbone import GAOTBackbone
from shape_foundation.data.preprocessing import MeshPreprocessor
from shape_foundation.data.sampling import SurfaceSampler

# Load checkpoint
ckpt = torch.load(
    "shape-foundation-small-v3/checkpoint_final.pt",
    map_location="cpu",
    weights_only=False,
)
cfg: ShapeConfig = ckpt["config"]

# Build model + load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GAOTBackbone(cfg).to(device).eval()
model.load_state_dict(ckpt["model_state_dict"], strict=False)

# Preprocess a mesh
mesh = trimesh.load("your_mesh.stl", force="mesh")
prep = MeshPreprocessor(cfg.input)(
    torch.tensor(mesh.vertices, dtype=torch.float32),
    torch.tensor(mesh.faces, dtype=torch.int64),
    torch.tensor(mesh.vertex_normals, dtype=torch.float32),
)
sampled = SurfaceSampler(cfg.input).sample(
    prep["vertices"],
    prep["faces"],
    prep["normals"],
    prep.get("curvature"),
)

# Run forward pass
with torch.no_grad():
    out = model.forward_tokens(
        sampled["points"].unsqueeze(0).to(device),
        sampled["features"].unsqueeze(0).to(device),
        sampled["normals"].unsqueeze(0).to(device)
        if sampled.get("normals") is not None
        else None,
        sampled["curvature"].unsqueeze(0).to(device)
        if sampled.get("curvature") is not None
        else None,
    )

pooled = out["pooled_embedding"]  # (1, 128) — global mesh embedding
tokens = out["token_embeddings"]  # (1, 13824, 128) — per-token features
```

## Intended Use

- **Dense geometric feature extraction** for downstream CAD / engineering tasks.
- **Shape retrieval** via learned embedding similarity.
- **Per-region anomaly detection** via masked reconstruction error heatmaps.

## Limitations

- **Supervised task heads are disabled.** This checkpoint only supports backbone embeddings and masked reconstruction.
- **Domain specificity.** The model is trained on 100% industrial CAD data and will transfer poorly to organic shapes or noisy 3D scans.

## Citation

```bibtex
@software{notelink_shape_foundation_2026,
  author = {{Notelink LLC}},
  title  = {Shape Foundation Model},
  year   = {2026},
  url    = {https://huggingface.co/bayang/shape-foundation-small-v3}
}
```
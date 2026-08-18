# SHAPE Encoder Fine-Tuning & Evaluation Report

This report documents the workflow, implementation details, and evaluation results of adapting the **SHAPE 3D Foundation Model** to the pump geometry dataset using **LoRA (Low-Rank Adaptation)**.

---

## 1. Overview of the Adaptation Workflow

The goal of this task is to extract compact, 256-dimensional semantic shape representations (embeddings) from raw pump STL meshes. These embeddings serve as geometric descriptors to train downstream surrogate models predicting pump performance (such as static ROMs).

To optimize the model's representations for our specific geometries, we implemented the following pipeline:

```
[STL Meshes]
     │
     ▼
[precompute_dataset.py]  ──> CPU-parallel preprocessing (Curvature, Normals, Scaling)
     │                       Creates 10 stochastic views per pump model
     ▼
[fine_tune_shape_lora.py] ──> GPU LoRA Fine-Tuning (Joint InfoNCE + Reconstruction Loss)
     │                       VRAM memory chunking optimizations
     ▼
[Shape_Tuned_Test.ipynb] ──> Evaluation (3D Plots, 2D/3D PCA & UMAP Projections)
```

---

## 2. Technical Implementation Details

### A. Precomputation (`precompute_dataset.py`)
To prevent severe CPU bottlenecks during training, we pre-process the meshes in parallel across CPU cores:
* **Canonicalization**: Centers the mesh at the origin and scales it uniformly to fit inside a unit bounding sphere.
* **Feature Extraction**: Computes vertex normals and cotangent Laplacian curvature.
* **Stochastic Multi-View Generation**: Samples $10$ different views of $32,768$ points per pump, saving them as compressed `.npz` files in `pump_metadata_cache/`.

### B. VRAM Memory Optimization
To enable training on GPUs with limited VRAM (e.g., 6 GB), we monkey-patched the tokenizer's native neighbor search module with a **chunked radius search** (`chunked_radius_search_native`). This splits large coordinate matrices into chunks of $1,728$ points during the distance calculation:
* **VRAM reduction**: From **2.26 GB** down to **226 MB** per batch item (a **10x** memory saving).
* This allowed us to run gradient accumulation steps without CUDA Out-of-Memory (OOM) failures.

### C. Parameter-Efficient Fine-Tuning (LoRA)
We used Hugging Face's `peft` library to wrap the backbone transformer block's attention projection layers:
* **Target Modules**: `q_proj`, `k_proj`, `v_proj`, and `out_proj`.
* **Trainable Parameters**: Freezing the coordinates and latent grid parameters reduced the trainable parameters to **35,840** (only **0.32%** of the model's total 10.9M weights).

### D. Joint Loss Objective
The training loop utilizes a dual-objective loss function:
1. **Reconstruction Regularizer (80% Weight)**: Computes the Mean Squared Error (MSE) between the fine-tuned token embeddings and the original frozen backbone embeddings (retrieved by disabling the LoRA adapter). This prevents *catastrophic forgetting* and anchors the model's spatial understanding.
2. **Symmetric InfoNCE Contrastive Loss (20% Weight)**: Normalizes and projects the global pooled embeddings of two different stochastic views of the same pump, maximizing their similarity while pushing different pumps apart.

---

## 3. Training & Validation Results

The model was fine-tuned for 10 epochs using the AdamW optimizer with a learning rate of $5 \times 10^{-4}$ and an accumulation size of 4.

The metrics showing stable convergence were saved to `lora_checkpoints/metrics.csv`:

| Epoch | Train Loss | Train Recon (MSE) | Train Contrastive | Val Loss | Val Recon (MSE) | Val Contrastive |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **001** | 0.06677 | 0.00734 | 0.30446 | 0.02967 | 0.01093 | 0.10466 |
| **002** | 0.03156 | 0.01207 | 0.10949 | 0.02384 | 0.00960 | 0.08081 |
| **003** | 0.01724 | 0.01029 | 0.04503 | 0.02012 | 0.01056 | 0.05838 |
| **004** | 0.02716 | 0.01183 | 0.08846 | 0.02238 | 0.01408 | 0.05560 |
| **005** | 0.01940 | 0.01222 | 0.04813 | 0.02110 | 0.00964 | 0.06691 |
| **006** | 0.02384 | 0.01158 | 0.07289 | 0.01925 | 0.01053 | 0.05411 |
| **007** | 0.02650 | 0.01163 | 0.08594 | 0.01957 | 0.00974 | 0.05890 |
| **008** | 0.02056 | 0.01126 | 0.05778 | 0.01708 | 0.01101 | 0.04135 |
| **009** | 0.01718 | 0.01225 | 0.03689 | 0.01778 | 0.00991 | 0.04929 |
| **010** | 0.02337 | 0.01238 | 0.06731 | 0.01833 | 0.01199 | 0.04370 |

---

## 4. Evaluation and Visualization

Here we compare the baseline behavior (before fine-tuning) with the customized behavior (after LoRA fine-tuning).

### A. Curvature Feature Mapping
Geometric curvature values calculated from 3D meshes are typically highly skewed (smooth surfaces have curvature near 0, while sharp impeller edges have massive values). Applying a logarithmic scale ($\log(1 + \text{curvature})$) compresses this skewness, allowing us to see the detailed transitions over blades and housings.

![Curvature Histogram Comparison](Shape_Tuned_Test_cell23_out1.png)

---

### B. Cosine Similarity Heatmaps: Before vs. After Fine-Tuning

The primary goal of contrastive fine-tuning is to differentiate pump geometries in embedding space. By plotting the cosine similarity matrices for the same set of diverse pumps:

#### **Before Fine-Tuning (Untuned)**
The pre-trained foundation model is overly generic. It outputs very similar embeddings for all pumps, resulting in an average cross-similarity near **0.999** (the heatmap is solid red/white with almost no contrast).

![Cosine Similarity before Fine-Tuning](Shape_Test_cell22_out0.png)

#### **After Fine-Tuning (Tuned with LoRA)**
The fine-tuned model has learned to distinguish the pump shapes. The cross-similarity between different pumps drops to an average of **0.70**, exposing the true geometric variations (the heatmap now has distinct contrast patterns, while the diagonal remains at **1.0**).

![Cosine Similarity after Fine-Tuning](Shape_Tuned_Test_cell16_out0.png)

---

### C. Pairwise Correlation Histograms: Before vs. After Fine-Tuning

We plot the distribution of all pairwise similarities between different pumps to see the spread:

#### **Before Fine-Tuning (Untuned)**
Almost all similarity values are tightly clustered in a narrow band above **0.998**.

![Pairwise Correlation Histogram before Fine-Tuning](Shape_Test_cell23_out0.png)

#### **After Fine-Tuning (Tuned with LoRA)**
The similarity distribution spreads out between **0.40** and **0.90**, centered around **0.70**. This indicates that the embedding space has successfully expanded to capture detailed, fine-grained variations.

![Pairwise Correlation Histogram after Fine-Tuning](Shape_Tuned_Test_cell17_out0.png)

---

![2D UMAP Embedding Space](Shape_Tuned_Test_cell24_out0.png)

---

### E. Masked Geometry Infill Evaluation
We evaluated the ability of the fine-tuned LoRA model to reconstruct the raw geometric statistics (coordinates, normals, and curvatures) of our pump shapes under different levels of masking/occlusion. This evaluates the robustness of the model's 3D structural prior.

#### **1. 0% Mask (Autoencoder Baseline)**
* **Objective**: The model processes the full shape without any masking to verify compression fidelity.
* **Results**: Reconstruction MSE is **~0.03 - 0.04** with an $R^2$ score of **~0.94 - 0.95**.
* **Significance**: The 128-dimensional latent tokens successfully capture **95% of the total geometric variance** of the raw pump shapes.

#### **2. 25% Mask Infill (Standard Pretraining Ratio)**
* **Objective**: We erase 25% of the shape's keypoints and ask the model to fill in the missing regions.
* **Results**: Reconstruction MSE remains at **~0.03 - 0.04** with an $R^2$ score of **~0.94 - 0.95**.
* **Significance**: The model has a highly effective spatial attention prior. Even when a quarter of the pump is missing, it looks at the surrounding context to accurately predict the geometric flow of the missing regions.

#### **3. 50% Mask Infill (High Occlusion)**
* **Objective**: Half of the shape is masked out.
* **Results**: Reconstruction MSE rises to **~0.35 - 0.58**, and the $R^2$ score drops to **~0.15 - 0.50**.
* **Significance**: The prediction task becomes significantly harder. The model can approximate the general shape but begins to lose fine-grained details of the impellers/housings.

#### **4. 80% Mask Infill (Extreme Occlusion)**
* **Objective**: 80% of the shape is masked out, leaving only scattered fragments.
* **Results**: Reconstruction MSE is **>1.5**, and the $R^2$ score becomes negative (**~-1.2**).
* **Significance**: At this extreme level of occlusion, the remaining context is insufficient. A negative $R^2$ score indicates that the model's prediction is worse than simply guessing the global average of the pump features.


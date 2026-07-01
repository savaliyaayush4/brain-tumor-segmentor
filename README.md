# 3D Multi-Task Brain Tumor Segmentation and Classification

## Project Overview
This project implements an end-to-end 3D multi-task deep learning pipeline for brain tumor analysis. It processes heterogeneous, multi-source MRI datasets and trains a custom 3D Attention U-Net to simultaneously perform:
1. **Binary Tumor Segmentation**
2. **Signed Distance Field (SDF) Regression** (for boundary-aware supervision)
3. **Tumor Type Classification** (Glioma, Meningioma, Metastatic)

The implementation is built on PyTorch and the MONAI framework, prioritizing memory efficiency, topological accuracy, and robust handling of extreme class imbalance.

---

## Methodological Foundations
The architectural and preprocessing choices in this project are grounded in established practices from leading medical imaging research:
*   **Preprocessing Paradigm:** Follows the self-configuring best practices established by **nnU-Net** (Isensee et al., *Nature Methods*, 2021) to ensure spatial consistency and intensity comparability across multi-site MRI data.
*   **Skip Attention Mechanisms:** Integrates Multi-Head Self-Attention (MHSA) into deep encoder skips, aligning with **Attention U-Net** (Oktay et al., 2018). This captures long-range spatial dependencies that purely convolutional receptive fields miss in large 3D volumes.
*   **Boundary-Aware Supervision:** Standard segmentation losses struggle with boundary precision due to class imbalance. Introducing an auxiliary **Signed Distance Field (SDF)** regression head provides continuous, boundary-aware supervision to enforce strict topological adherence.

---

## Data Engineering & Preprocessing Pipeline
To handle the heterogeneity of the **BraTS-GLI**, **BraTS-MEN-RT**, and **BCBM-RadioGenomics** datasets, a rigorous preprocessing pipeline was engineered to standardize the data prior to training. 

The pipeline executes the following transformations:
1.  **Spatial Harmonization:** All NIfTI volumes are reoriented to the **RAS+ canonical coordinate system**. Volumes are then resampled to an isotropic resolution of $1 \times 1 \times 1 \text{ mm}^3$ using trilinear interpolation for images and nearest-neighbor interpolation for labels.
2.  **Foreground Cropping (Skull-Stripping):** To eliminate background noise and reduce computational overhead, a tight bounding box is computed around non-zero brain voxels. An 8-voxel margin is applied to ensure no peripheral tumor tissue is clipped.
3.  **Intensity Normalization:** To mitigate multi-scanner intensity variations, image intensities are clipped to the **0.5th and 99.5th percentiles**, followed by z-score normalization computed strictly within the brain mask.
4.  **Stratified Export:** The dataset is subjected to an 80/20 stratified train/validation split. To maintain class balance, samples are capped per category and exported as optimized `.npy` tensors for high-throughput dataloading.

*Note: Truncated Signed Distance Fields (TSDF) are computed dynamically on-the-fly during dataloading to provide continuous boundary supervision without bloating the stored dataset.*

---

## Model Architecture: `BTUnetAttention`
The core model is a custom 3D U-Net augmented with attention mechanisms and multi-task heads. 

### Strategic Attention Placement
Applying self-attention to high-resolution 3D feature maps causes an $O(N^2)$ memory explosion. To resolve this, MHSA is **strategically restricted to the deep encoder skips** (spatial dimensions $16^3$ and $8^3$). Shallow skips ($32^3$ and $64^3$) rely on standard convolutional residual blocks to maintain computational feasibility on standard GPU hardware while still capturing global context at the bottleneck.

### Multi-Task Heads & Deep Supervision
*   **Segmentation & SDF Heads:** Output binary tumor probability maps and the truncated SDF scalar field, respectively.
*   **Classification Head:** Utilizes global average pooled bottleneck features (320-dim) passed through an MLP to classify the tumor type.
*   **Deep Supervision:** Auxiliary segmentation losses are applied to intermediate decoder stages to stabilize gradient flow and improve feature learning in the lower layers.

---

## Training & Optimization Strategy

### Loss Function Formulation
To address the extreme foreground/background imbalance and boundary ambiguity, the segmentation loss is a weighted combination:
*   **Tversky Loss** ($\alpha=0.3, \beta=0.7$): Explicitly penalizes False Negatives more than False Positives, crucial for ensuring the tumor core is not missed.
*   **Focal Loss** ($\gamma=2.0$): Forces the network to focus on hard-to-segment boundary voxels.
*   **SDF Loss:** **Huber Loss** ($\delta=0.1$) applied to the SDF regression to ensure smooth but precise boundary gradients.
*   **Classification Loss:** Standard Cross-Entropy for tumor typing.

### Engineering & Memory Optimization
Training 3D medical images requires strict memory management:
*   **Automatic Mixed Precision (AMP):** Utilizes `torch.amp` to reduce memory footprint and accelerate training without sacrificing convergence.
*   **Gradient Clipping:** Prevents exploding gradients common in deep 3D residual networks.
*   **Cosine Annealing:** Learning rate scheduler with a low `eta_min` to ensure fine-grained weight updates in later epochs.
*   **Early Stopping & Checkpointing:** Monitors validation Dice score and Classification Accuracy independently to save the best models for both tasks.

---

## Inference Protocol
During evaluation, full-resolution 3D volumes exceed standard GPU memory limits. To resolve this, the project utilizes **Sliding Window Inference** (via MONAI):
*   **ROI Size:** $128 \times 128 \times 128$ voxels (matching the training crop size).
*   **Gaussian Blending:** Uses a Gaussian importance map with 25% overlap to smoothly blend predictions at the patch boundaries, eliminating visible stitching artifacts.

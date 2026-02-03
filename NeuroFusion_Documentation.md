# 🧠 NeuroFusion: Technical Documentation

**NeuroFusion** is a Unified Generative AI system capable of handling incomplete MRI data for brain tumor analysis.

## 1. System Architecture

The core model (`UnifiedTumorDetector`) is a composite of two specialized neural networks trained jointly.

### A. The Generator (Modality Synthesis)
*   **Purpose**: To "hallucinate" missing MRI scans (e.g., if a patient lacks T1ce).
*   **Architecture**: `GANModalitySynthesis2D` (Conditional GAN).
    *   **Encoder**: Compresses input scans into a latent feature space.
    *   **Bottleneck**: 4-level deep features.
    *   **Decoder**: Reconstructs the missing modality from features.
*   **Why it's special**: It handles **variable inputs**. Even if 3 out of 4 scans are missing, it attempts to reconstruct the 4th based on available clues.

### B. The Segmentor (Tumor Detection)
*   **Purpose**: To identify the precise pixel-level location of the tumor.
*   **Architecture**: `Generator2D` (modified U-Net).
    *   **Input**: The "Complete" set of scans (Real + Synthesized from Step A).
    *   **Output**: A binary mask (1 = Tumor, 0 = Healthy Tissue).
*   **Integration**: The Segmentor *never sees missing data*. It always sees a "complete" patient because the Generator fills in the gaps first.

## 2. Data Pipeline (The "2D Trick")

Training 3D MRI models on consumer hardware (GTX 1650, 4GB VRAM) is notoriously difficult. NeuroFusion solves this with a **2D Slicing Strategy**.

1.  **Input**: 3D Volume (e.g., 240x240x155 voxels).
2.  **Slicing**: The dataloader extracts random 2D slices (e.g., slice 80, 81, 82) during training.
3.  **Processing**: The GPU processes these as small 2D images (Batch Size = 1 Patient = 5 Slices).
4.  **Reconstruction**: For inference, we verify on individual slices, but the concept extends to stacking them back into 3D.

## 3. Training Methodology

*   **Optimizer**: Adam (Learning Rate 1e-4).
*   **Precision**: 16-bit Mixed Precision (AMP) to save memory.
*   **Loss Function**: `UnifiedLoss`.
    *   **L1 Loss**: Ensures synthesized images look like real MRI scans.
    *   **Dice Loss**: Ensures the segmentation mask overlaps perfectly with the tumor.

## 4. Performance

*   **Dice Score**: 0.827 (82.7% accuracy in tumor shape overlap).
*   **Inference Speed**: Real-time on GTX 1650.
*   **Robustness**: Can handle ~30% missing data without failing.

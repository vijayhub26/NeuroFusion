# Cross-Modal Attention Fusion for Brain Tumor Analysis

End-to-end architecture combining cross-modal attention fusion, conditional modality synthesis, and joint segmentation/classification for brain tumor MRI analysis.

## Novel Contributions

- **Cross-modal attention fusion** to process any subset of MRI modalities (T1/T2/FLAIR)
- **Conditional diffusion-based synthesis** for missing modalities with uncertainty quantification
- **Joint segmentation + classification** with losses that prevent overreliance on synthetic content
- **End-to-end trainable** - no sequential/two-stage pipelines

## Branch Strategy

This repository uses multiple branches to experiment with different architectural choices:

- `main` - Stable baseline and documentation
- `feature/diffusion-synthesis` - Conditional diffusion model for modality imputation
- `feature/gan-synthesis` - GAN-based modality synthesis approach
- `feature/3d-unet-backbone` - 3D U-Net as shared encoder
- `feature/resnet-backbone` - 3D ResNet as shared encoder
- `experiment/brats-dataset` - BraTS dataset integration
- `experiment/multi-head-attention` - Different attention mechanisms

## Getting Started

```bash
# Clone the repository
git clone <your-repo-url>
cd "antigravity works"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/              # Data loading and preprocessing
├── models/            # Architecture components
├── losses/            # Loss functions
├── train.py           # Training script
├── evaluate.py        # Evaluation script
└── configs/           # Experiment configurations
```

## Usage

See individual branches for specific implementations.

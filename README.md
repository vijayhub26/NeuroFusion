# Cross-Modal Attention Fusion for Brain Tumor Analysis

**End-to-end PyTorch implementation** combining cross-modal attention fusion, conditional diffusion-based modality synthesis, and joint segmentation/classification for brain tumor MRI analysis.

## 🔬 Novel Contributions

1. **Cross-modal attention fusion** - Processes any subset of MRI modalities (T1/T1ce/T2/FLAIR)
2. **Conditional diffusion synthesis** - Generates missing modalities using DDPM/DDIM with uncertainty quantification
3. **Joint segmentation + classification** - Tumor masks + grade (LGG/HGG) in one model
4. **Synthetic content penalty** - Prevents overreliance on synthesized modalities

Unlike existing approaches that use sequential pipelines, this architecture performs **end-to-end joint training**.

## 🏗️ Architecture

```
Input (Any subset of modalities) → Modality Encoders
→ Missing Modality Synthesis (Diffusion + Uncertainty)
→ Cross-Modal Attention Fusion → Shared Encoder
├→ Segmentation → Tumor masks
└→ Classification → Grade (LGG/HGG)
```

## 📦 Installation

```bash
cd "antigravity works"
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 📊 Dataset (BraTS Format)

```
data/BraTS/
├── train/Patient001/
│   ├── Patient001_t1.nii.gz
│   ├── Patient001_t1ce.nii.gz
│   ├── Patient001_t2.nii.gz
│   ├── Patient001_flair.nii.gz
│   ├── Patient001_seg.nii.gz
│   └── grade.txt  # "LGG" or "HGG"
├── val/
└── test/
```

## 🚀 Usage

**Training:**
```bash
python train.py
```

**Evaluation:**
```bash
python evaluate.py
```

## 📁 Structure

```
├── config.py           # Hyperparameters
├── train.py            # Training (PyTorch Lightning)
├── evaluate.py         # Missing modality evaluation
├── data/
│   ├── dataset.py      # BraTS dataloader
│   └── __init__.py
├── models/
│   ├── encoders.py     # Modality + shared encoders
│   ├── attention_fusion.py  # Cross-modal fusion
│   ├── synthesis.py    # Conditional diffusion
│   ├── segmentation.py # Decoder
│   ├── classification.py    # Grade classifier
│   ├── unified_model.py     # End-to-end model
│   └── __init__.py
├── losses/
│   ├── combined_loss.py     # Multi-task loss
│   └── __init__.py
└── utils/
    ├── metrics.py      # Dice, HD95, accuracy
    └── __init__.py
```

## 🎯 Key Features

- **Modality Encoders**: Lightweight 3D CNNs per modality
- **Diffusion Synthesis**: DDPM with DDIM sampling (50 steps)
- **Uncertainty**: Pixel-wise variance from multiple samples
- **Fusion**: Attention with uncertainty-aware gating
- **Losses**: Dice + CE (seg) + Focal (cls) + MSE (synthesis) + uncertainty + attention penalties
- **Metrics**: Dice, HD95, accuracy, precision, recall, F1

## 🔧 Configuration

Edit `config.py` to modify:
- Loss weights (λ_seg, λ_cls, λ_synthesis, etc.)
- Diffusion parameters (steps, beta schedule)
- Model dimensions (channels, heads, fusion_dim)
- Training hyperparameters (LR, epochs, batch size)

## 🧪 Branching Strategy (Future)

When ready to experiment, create branches for:
- `feature/gan-synthesis` - Replace diffusion with GAN
- `feature/transformer-fusion` - Replace attention with transformers
- `experiment/loss-ablation` - Test different loss weights
- `baseline/two-stage` - Compare against sequential pipeline

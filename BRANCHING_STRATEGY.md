# Git Branching Strategy for Experimental ML Research

## Branch Types

### 1. Main Branch
- `main` - Stable code, documentation, baseline implementations
- Always working and tested
- Merge only from feature branches after verification

### 2. Feature Branches
For implementing specific architectural components:

- `feature/diffusion-synthesis` - Conditional DDPM/DDIM for modality imputation
- `feature/gan-synthesis` - Conditional GAN for modality translation
- `feature/3d-unet-backbone` - 3D U-Net as shared encoder backbone
- `feature/resnet-backbone` - 3D ResNet as shared encoder
- `feature/transformer-fusion` - Transformer-based cross-modal fusion
- `feature/attention-fusion` - Standard multi-head attention fusion

### 3. Experiment Branches
For dataset-specific implementations and hyperparameter experiments:

- `experiment/brats-dataset` - BraTS dataset integration
- `experiment/tcga-dataset` - TCGA dataset integration
- `experiment/loss-ablation` - Testing different loss weight combinations
- `experiment/uncertainty-quantification` - Different uncertainty estimation methods

### 4. Comparison Branches
For baseline comparisons:

- `baseline/two-stage-pipeline` - Traditional synthesis → segmentation pipeline
- `baseline/zero-filling` - Simple zero-filling for missing modalities
- `baseline/single-modality` - Best single-modality model

## Workflow

### Creating a New Experimental Branch

```bash
# Start from main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/diffusion-synthesis

# Work on your implementation
# ... make changes ...

# Commit regularly
git add .
git commit -m "Implement DDPM conditioned on available modalities"

# Push to remote
git push -u origin feature/diffusion-synthesis
```

### Comparing Different Approaches

```bash
# Work on diffusion approach
git checkout feature/diffusion-synthesis
# ... implement and test ...

# Switch to GAN approach
git checkout feature/gan-synthesis
# ... implement and test ...

# Compare results
git diff feature/diffusion-synthesis feature/gan-synthesis models/synthesis.py
```

### Merging Successful Experiments

```bash
# After testing shows diffusion works better
git checkout main
git merge feature/diffusion-synthesis
git push origin main

# Archive or delete unsuccessful branch
git branch -d feature/gan-synthesis
# Or keep for reference
```

## Recommended Experimental Matrix

| Branch | Synthesis Method | Backbone | Fusion | Dataset |
|--------|-----------------|----------|--------|---------|
| `exp/diff-unet-attn-brats` | Diffusion | 3D U-Net | Attention | BraTS |
| `exp/gan-unet-attn-brats` | GAN | 3D U-Net | Attention | BraTS |
| `exp/diff-resnet-attn-brats` | Diffusion | ResNet | Attention | BraTS |
| `exp/diff-unet-transformer-brats` | Diffusion | 3D U-Net | Transformer | BraTS |

## Tracking Experiments

Use this alongside experiment tracking tools:

```bash
# Tag successful experiments
git tag -a v1.0-diffusion-baseline -m "Baseline with diffusion synthesis, Dice=0.85"
git push origin v1.0-diffusion-baseline

# Reference in wandb/mlflow
# experiment_name = f"exp_{git.branch}_{git.commit[:7]}"
```

## Best Practices

1. **One major change per branch** - Don't combine synthesis method + backbone changes
2. **Frequent commits** - Commit after each working sub-component
3. **Descriptive commit messages** - Include metrics when available
4. **Don't mix data and code** - Keep large datasets out of Git (use .gitignore)
5. **Document in branch README** - Each experimental branch should document its specific setup

## Quick Commands

```bash
# List all branches
git branch -a

# Switch between experiments
git checkout feature/diffusion-synthesis

# Create new experiment from another branch
git checkout -b experiment/new-idea feature/diffusion-synthesis

# See what changed between branches
git diff feature/gan-synthesis feature/diffusion-synthesis

# Stash changes before switching
git stash
git checkout other-branch
git stash pop
```

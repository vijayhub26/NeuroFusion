"""Debug segmentation values."""

import torch
from data import get_brats_dataloaders_2d
from config import Config

config = Config()

print(f"Config num_classes: {config.data.num_classes}")
print("\nLoading dataset...")
train_loader, val_loader = get_brats_dataloaders_2d(
    data_root=config.data.data_root,
    batch_size=1,
    num_workers=0,
    modalities=config.data.modalities,
    image_size=config.data.image_size,
    missing_prob=0.0,
    num_slices_per_scan=config.data.num_slices_per_scan
)

print("\nChecking segmentation values in first batch:")
for i, batch in enumerate(train_loader):
    if i >= 1:
        break
    
    seg = batch["seg"]  # (1, num_slices, H, W)
    
    print(f"\nBatch {i}:")
    print(f"  Seg shape: {seg.shape}")
    print(f"  Seg dtype: {seg.dtype}")
    print(f"  Unique values: {torch.unique(seg)}")
    print(f"  Min value: {seg.min().item()}")
    print(f"  Max value: {seg.max().item()}")
    
    # Check if any value is >= num_classes
    if seg.max() >= config.data.num_classes:
        print(f"  ⚠️ WARNING: Max value {seg.max().item()} >= num_classes {config.data.num_classes}!")
    else:
        print(f"  ✅ All seg values are valid")

print("\nDone!")

"""Debug script to check grade values in dataset."""

import torch
from data import get_brats_dataloaders_2d
from config import Config

config = Config()

print("Loading dataset...")
train_loader, val_loader = get_brats_dataloaders_2d(
    data_root=config.data.data_root,
    batch_size=1,
    num_workers=0,  # No workers for debugging
    modalities=config.data.modalities,
    image_size=config.data.image_size,
    missing_prob=0.0,  # No missing for debugging
    num_slices_per_scan=config.data.num_slices_per_scan
)

print("\nChecking first 5 batches:")
for i, batch in enumerate(train_loader):
    if i >= 5:
        break
    
    grade = batch["grade"]
    patient_id = batch["patient_id"]
    
    print(f"\nBatch {i}:")
    print(f"  Patient: {patient_id}")
    print(f"  Grade tensor: {grade}")
    print(f"  Grade value: {grade.item()}")
    print(f"  Grade dtype: {grade.dtype}")
    print(f"  Grade shape: {grade.shape}")

print("\nDone!")

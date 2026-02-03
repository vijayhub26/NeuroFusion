"""
Visualization script for checking synthesis quality.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from config import Config
from models.simple_detector import SimpleTumorDetector
from data import get_brats_dataloaders_2d

def visualize_synthesis(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load config and model
    config = Config()
    model = SimpleTumorDetector(config)
    
    # Load state dict
    # Set weights_only=False to allow loading the Config object if pickled
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # State dict keys might have 'model.' prefix if saved from PL system
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Load one batch
    print("Loading data...")
    train_loader, val_loader = get_brats_dataloaders_2d(
        data_root=config.data.data_root,
        batch_size=1,
        num_workers=0,
        num_slices_per_scan=5,
        missing_prob=0.0 # Get full data first
    )
    
    batch = next(iter(val_loader))
    modalities = batch["modalities"] # (1, num_slices, 4, H, W)
    
    # Select middle slice
    slice_idx = 2
    input_slice = modalities[0, slice_idx].unsqueeze(0) # (1, 4, H, W)
    
    # Simulate missing T1ce (index 1)
    # Modalities: T1, T1ce, T2, FLAIR
    missing_idx = 1
    modality_mask = torch.ones(4)
    modality_mask[missing_idx] = 0
    
    masked_input = input_slice.clone()
    masked_input[:, missing_idx] = 0
    
    print("Running synthesis...")
    # Expand mask to (1, 4)
    modality_mask = modality_mask.unsqueeze(0)
    
    with torch.no_grad():
        # model expects (N, 4, H, W) and (N, 4) mask
        complete_modalities, _, _ = model.synthesize_missing_modalities(
            masked_input, modality_mask
        )
        
    # Plotting
    print("Saving visualization...")
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]
    
    # Row 1: Ground Truth
    for i in range(4):
        img = input_slice[0, i].numpy()
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"GT {modality_names[i]}")
        axes[0, i].axis("off")
        
    # Row 2: Synthesized / Input
    for i in range(4):
        if i == missing_idx:
            # Show synthesized
            img = complete_modalities[0, i].numpy()
            axes[1, i].imshow(img, cmap="gray")
            axes[1, i].set_title(f"SYNTHESIZED {modality_names[i]}")
            
            # Add box/border to highlight
            for spine in axes[1, i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
        else:
            # Show masked input (should be same as GT)
            img = masked_input[0, i].numpy()
            axes[1, i].imshow(img, cmap="gray")
            axes[1, i].set_title(f"Input {modality_names[i]}")
            axes[1, i].axis("off")
            
    plt.tight_layout()
    plt.savefig("synthesis_result.png")
    print("Saved to synthesis_result.png")

if __name__ == "__main__":
    # Find latest checkpoint
    log_dir = "logs/simple_tumor_detection/version_2/checkpoints"
    checkpoints = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".ckpt"):
                checkpoints.append(os.path.join(root, file))
    
    if checkpoints:
        # Sort by creation time to get latest? Or just take one.
        # Let's take the one with highest accuracy if possible, but latest is fine.
        latest = checkpoints[-1] 
        print(f"Found {len(checkpoints)} checkpoints. Using: {latest}")
        visualize_synthesis(latest)
    else:
        print("No checkpoints found!")

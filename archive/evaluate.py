"""
Evaluation script for testing the model on different missing modality scenarios.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from config import Config, default_config
from models import UnifiedBrainTumorModel
from data import BraTSDataset
from utils.metrics import dice_coefficient, hausdorff_distance_95, classification_metrics


def evaluate_missing_modality_scenarios(
    model: UnifiedBrainTumorModel,
    test_dataset: BraTSDataset,
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on different missing modality scenarios.
    
    Scenarios:
    1. All modalities present
    2. Missing 1 modality (4 scenarios)
    3. Missing 2 modalities (6 scenarios)
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device to run on
    
    Returns:
        Dict of results per scenario
    """
    model.eval()
    model.to(device)
    
    num_modalities = 4  # T1, T1ce, T2, FLAIR
    scenarios = {}
    
    # Scenario 1: All modalities
    scenarios["all_modalities"] = torch.ones(num_modalities)
    
    # Scenario 2: Missing one modality each
    for i in range(num_modalities):
        mask = torch.ones(num_modalities)
        mask[i] = 0
        scenarios[f"missing_mod_{i}"] = mask
    
    # Scenario 3: Missing two modalities
    for i in range(num_modalities):
        for j in range(i + 1, num_modalities):
            mask = torch.ones(num_modalities)
            mask[i] = 0
            mask[j] = 0
            scenarios[f"missing_mods_{i}_{j}"] = mask
    
    results = {}
    
    for scenario_name, modality_mask in scenarios.items():
        print(f"\nEvaluating scenario: {scenario_name}")
        
        seg_dice_scores = []
        cls_accuracies = []
        
        with torch.no_grad():
            for sample in tqdm(test_dataset):
                modalities = sample["modalities"].unsqueeze(0).to(device)
                seg_target = sample["seg"].unsqueeze(0).to(device)
                grade_target = sample["grade"].unsqueeze(0).to(device)
                
                # Apply scenario mask
                masked_modalities = modalities.clone()
                for mod_idx in range(num_modalities):
                    if modality_mask[mod_idx] == 0:
                        masked_modalities[:, mod_idx] = 0
                
                mask_batch = modality_mask.unsqueeze(0).to(device)
                
                # Forward pass
                outputs = model(masked_modalities, mask_batch, training=False)
                
                # Segmentation metrics
                seg_pred = outputs["seg_logits"].argmax(dim=1)
                dice = dice_coefficient(seg_pred, seg_target, num_classes=4)
                seg_dice_scores.append(dice["dice_mean"])
                
                # Classification metrics
                cls_pred = outputs["grade_logits"]
                cls_acc = classification_metrics(cls_pred, grade_target)
                cls_accuracies.append(cls_acc["accuracy"])
        
        # Aggregate results
        results[scenario_name] = {
            "dice_mean": np.mean(seg_dice_scores),
            "dice_std": np.std(seg_dice_scores),
            "cls_accuracy": np.mean(cls_accuracies)
        }
        
        print(f"  Dice: {results[scenario_name]['dice_mean']:.4f} ± {results[scenario_name]['dice_std']:.4f}")
        print(f"  Accuracy: {results[scenario_name]['cls_accuracy']:.4f}")
    
    return results


def main():
    """Main evaluation function."""
    config = default_config
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = BraTSDataset(
        data_root=os.path.join(config.data.data_root, "test"),
        modalities=config.data.modalities,
        image_size=config.data.image_size,
        is_training=False,
        missing_prob=0.0
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load trained model
    print("Loading model...")
    model = UnifiedBrainTumorModel(config)
    
    # Load checkpoint (update path to your best checkpoint)
    checkpoint_path = "checkpoints/best_model.ckpt"  # Update this
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using random initialization")
    
    # Evaluate
    print("\nRunning evaluation on missing modality scenarios...")
    results = evaluate_missing_modality_scenarios(
        model, test_dataset, device=config.device
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for scenario, metrics in results.items():
        print(f"\n{scenario}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save results
    import json
    output_path = os.path.join(config.output_dir, "evaluation_results.json")
    os.makedirs(config.output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

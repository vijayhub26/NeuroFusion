"""
Utility functions for computing evaluation metrics.
"""

import torch
import numpy as np
from typing import Dict, List
from scipy.spatial.distance import directed_hausdorff


def dice_coefficient(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True
) -> Dict[str, float]:
    """
    Compute Dice coefficient per class.
    
    Args:
        predictions: Predicted segmentation (B, D, H, W) with class indices
        targets: Ground truth (B, D, H, W)
        num_classes: Number of classes
        ignore_background: Whether to exclude background from mean
    
    Returns:
        Dict with per-class Dice scores and mean
    """
    dice_scores = {}
    
    for class_idx in range(num_classes):
        pred_mask = (predictions == class_idx).float()
        target_mask = (targets == class_idx).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / union
        
        dice_scores[f"dice_class_{class_idx}"] = dice.item()
    
    # Compute mean (optionally excluding background)
    start_idx = 1 if ignore_background else 0
    mean_dice = np.mean([dice_scores[f"dice_class_{i}"] for i in range(start_idx, num_classes)])
    dice_scores["dice_mean"] = mean_dice
    
    return dice_scores


def hausdorff_distance_95(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """
    Compute 95th percentile Hausdorff Distance per class.
    
    Args:
        predictions: Predicted segmentation (D, H, W) with class indices 
        targets: Ground truth (D, H, W)
        num_classes: Number of classes
    
    Returns:
        Dict with per-class HD95 scores
    """
    hd_scores = {}
    
    for class_idx in range(1, num_classes):  # Skip background
        pred_points = np.argwhere(predictions == class_idx)
        target_points = np.argwhere(targets == class_idx)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            hd_scores[f"hd95_class_{class_idx}"] = np.nan
            continue
        
        # Compute directed Hausdorff distances
        hd_1 = directed_hausdorff(pred_points, target_points)[0]
        hd_2 = directed_hausdorff(target_points, pred_points)[0]
        
        # 95th percentile (symmetric)
        hd95 = max(hd_1, hd_2)
        hd_scores[f"hd95_class_{class_idx}"] = hd95
    
    return hd_scores


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute classification accuracy and per-class metrics.
    
    Args:
        predictions: Predicted class logits (B, num_classes)
        targets: Ground truth class indices (B,)
    
    Returns:
        Dict with accuracy, precision, recall, F1
    """
    pred_classes = predictions.argmax(dim=1)
    
    # Accuracy
    correct = (pred_classes == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    
    # Convert to numpy for sklearn
    pred_np = pred_classes.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # Per-class metrics (for binary: LGG=0, HGG=1)
    metrics = {"accuracy": accuracy}
    
    # True positive, false positive, etc. for class 1 (HGG)
    tp = ((pred_np == 1) & (target_np == 1)).sum()
    fp = ((pred_np == 1) & (target_np == 0)).sum()
    fn = ((pred_np == 0) & (target_np == 1)).sum()
    tn = ((pred_np == 0) & (target_np == 0)).sum()
    
    # Precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics.update({
        "precision": precision,
        "recall": recall,
        "f1": f1
    })
    
    return metrics

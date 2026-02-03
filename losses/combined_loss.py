"""
Combined loss function for joint training.
Includes segmentation, classification, synthesis, and regularization losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DiceLoss(nn.Module):
    """
    Dice loss for multi-class segmentation.
    """
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Logits (B, num_classes, D, H, W)
            targets: Ground truth (B, D, H, W) with class indices
        
        Returns:
            Dice loss
        """
        num_classes = predictions.shape[1]
        
        # Convert logits to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Clip targets to valid range [0, num_classes-1]
        targets = torch.clamp(targets.long(), 0, num_classes - 1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes)
        
        # Permute based on dimensionality (2D or 3D)
        if targets_one_hot.dim() == 4:  # 2D: (B, H, W, C)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        else:  # 3D: (B, D, H, W, C)
            targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Compute Dice coefficient per class
        dice_scores = []
        for class_idx in range(num_classes):
            pred_class = probs[:, class_idx]
            target_class = targets_one_hot[:, class_idx]
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average across classes
        dice_score = torch.stack(dice_scores).mean()
        
        # Return loss (1 - dice)
        return 1.0 - dice_score


class FocalLoss(nn.Module):
    """
    Focal loss for classification with class imbalance.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Logits (B, num_classes)
            targets: Ground truth class indices (B,)
        
        Returns:
            Focal loss
        """
        # Clip targets to valid range
        num_classes = predictions.shape[-1]
        targets = torch.clamp(targets.long(), 0, num_classes - 1)

        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined multi-task loss for unified model.
    
    Components:
    1. Segmentation: Dice + Cross-Entropy
    2. Classification: Focal or Cross-Entropy
    3. Synthesis: MSE reconstruction (when applicable)
    4. Uncertainty: Penalty on high uncertainty in predictions
    5. Attention regularization: Penalty on synthetic modality reliance
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Loss weights
        self.lambda_seg = config.loss.lambda_seg
        self.lambda_cls = config.loss.lambda_cls
        self.lambda_synthesis = config.loss.lambda_synthesis
        self.lambda_uncertainty = config.loss.lambda_uncertainty
        self.lambda_attention_reg = config.loss.lambda_attention_reg
        
        # Segmentation losses
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = config.loss.dice_weight
        self.ce_weight = config.loss.ce_weight
        
        # Classification loss
        if config.loss.use_focal_loss:
            self.cls_loss = FocalLoss(
                alpha=config.loss.focal_alpha,
                gamma=config.loss.focal_gamma
            )
        else:
            self.cls_loss = nn.CrossEntropyLoss()
    
    def segmentation_loss(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute segmentation loss (Dice + CE)."""
        dice = self.dice_loss(seg_logits, seg_targets)
        ce = self.ce_loss(seg_logits, seg_targets)
        return self.dice_weight * dice + self.ce_weight * ce
    
    def classification_loss(
        self,
        grade_logits: torch.Tensor,
        grade_targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute classification loss."""
        # Clip targets to valid range
        num_classes = grade_logits.shape[-1]
        grade_targets = torch.clamp(grade_targets.long(), 0, num_classes - 1)
        return self.cls_loss(grade_logits, grade_targets)
    
    def synthesis_regularization(
        self,
        synthesized: Optional[torch.Tensor],
        ground_truth: torch.Tensor,
        modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruction loss for synthesized modalities.
        Only computed when modalities were synthesized during training.
        
        Args:
            synthesized: Synthesized modalities (B, num_mods, D, H, W)
            ground_truth: Ground truth modalities (B, num_mods, D, H, W)
            modality_mask: Which modalities were originally available (B, num_mods)
        
        Returns:
            MSE loss on synthesized (missing) modalities
        """
        if synthesized is None:
            return torch.tensor(0.0, device=ground_truth.device)
        
        # Only compute loss for synthesized (originally missing) modalities
        missing_mask = (modality_mask == 0).float()  # (B, num_mods)
        
        if missing_mask.sum() == 0:
            return torch.tensor(0.0, device=ground_truth.device)
        
        # MSE on synthesized modalities only
        diff = (synthesized - ground_truth) ** 2  # (B, num_mods, D, H, W)
        
        # Weight by missing mask
        weighted_diff = diff * missing_mask[:, :, None, None, None]
        
        loss = weighted_diff.sum() / (missing_mask.sum() + 1e-8)
        
        return loss
    
    def uncertainty_penalty(
        self,
        uncertainty_maps: Dict[int, torch.Tensor],
        seg_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize high uncertainty in final predictions.
        Encourages the model to reduce reliance on uncertain synthesized regions.
        
        Args:
            uncertainty_maps: Dict of uncertainty maps for synthesized modalities
            seg_logits: Segmentation predictions
        
        Returns:
            Uncertainty penalty
        """
        if len(uncertainty_maps) == 0:
            return torch.tensor(0.0, device=seg_logits.device)
        
        # Average uncertainty across synthesized modalities
        total_uncertainty = torch.stack(list(uncertainty_maps.values())).mean()
        
        # Penalize high uncertainty
        return total_uncertainty.mean()
    
    def attention_regularization(
        self,
        modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Regularize attention to prefer real modalities over synthesized.
        This prevents the model from over-relying on synthetic data.
        
        Args:
            modality_mask: Binary mask (B, num_mods) - 0 for synthesized
        
        Returns:
            Regularization loss
        """
        # Penalize proportion of synthesized modalities used
        # This is a simple version - can be enhanced with actual attention weights
        num_synthesized = (modality_mask == 0).sum().float()
        total_modalities = modality_mask.numel()
        
        reg = num_synthesized / (total_modalities + 1e-8)
        
        return reg
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        seg_targets: torch.Tensor,
        grade_targets: torch.Tensor,
        modalities: torch.Tensor,
        modality_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.
        
        Args:
            outputs: Model outputs dict with seg_logits, grade_logits, etc.
            seg_targets: Segmentation ground truth (B, D, H, W)
            grade_targets: Grade ground truth (B,)
            modalities: Ground truth modalities (B, num_mods, D, H, W)
            modality_mask: Modality availability mask (B, num_mods)
        
        Returns:
            Total loss, dict of individual loss components
        """
        # Segmentation loss
        loss_seg = self.segmentation_loss(outputs["seg_logits"], seg_targets)
        
        # Classification loss
        loss_cls = self.classification_loss(outputs["grade_logits"], grade_targets)
        
        # Synthesis reconstruction loss (if applicable)
        loss_synthesis = self.synthesis_regularization(
            outputs.get("synthesized_modalities"),
            modalities,
            modality_mask
        )
        
        # Uncertainty penalty
        loss_uncertainty = self.uncertainty_penalty(
            outputs.get("uncertainty_maps", {}),
            outputs["seg_logits"]
        )
        
        # Attention regularization
        loss_attention = self.attention_regularization(modality_mask)
        
        # Total weighted loss
        total_loss = (
            self.lambda_seg * loss_seg +
            self.lambda_cls * loss_cls +
            self.lambda_synthesis * loss_synthesis +
            self.lambda_uncertainty * loss_uncertainty +
            self.lambda_attention_reg * loss_attention
        )
        
        # Individual losses for logging
        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/segmentation": loss_seg.item(),
            "loss/classification": loss_cls.item(),
            "loss/synthesis": loss_synthesis.item(),
            "loss/uncertainty": loss_uncertainty.item(),
            "loss/attention_reg": loss_attention.item()
        }
        
        return total_loss, loss_dict

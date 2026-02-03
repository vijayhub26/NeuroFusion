"""
Unified loss function for Segmentation + Synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedLoss(nn.Module):
    def __init__(self, lambda_seg=1.0, lambda_recon=1.0):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_recon = lambda_recon
        
        # Segmentation Loss: Cross Entropy + Dice
        self.ce_loss = nn.BCEWithLogitsLoss()
        
    def dice_loss(self, pred, target):
        """
        Binary Dice Loss.
        pred: (B, 1, H, W) logits
        target: (B, H, W) long or float, 0=background, >0=tumor
        """
        smooth = 1e-5
        
        # Apply sigmoid to logits
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1).float()
        
        # Bjarize target (Tumor vs Backgound) - assumption for now
        target = (target > 0).float()
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice

    def forward(self, 
                seg_logits, seg_targets,
                synthesized, gt_modalities, modality_mask):
        """
        Compute total loss.
        """
        # 1. Segmentation Loss
        # Target: (B, H, W), Logits: (B, 1, H, W)
        seg_targets_bin = (seg_targets > 0).float().unsqueeze(1) # (B, 1, H, W)
        
        loss_ce = self.ce_loss(seg_logits, seg_targets_bin)
        loss_dice = self.dice_loss(seg_logits, seg_targets_bin)
        
        total_seg_loss = loss_ce + loss_dice
        
        # 2. Reconstruction Loss
        # Only compute on missing modalities if possible? 
        # Or all modalities?
        # Let's compute on ALL modalities (Ground Truth vs Complete Synthesized)
        # This enforces the GAN to output correct non-missing parts too (identity mapping)
        # and correct missing parts.
        
        loss_recon = F.l1_loss(synthesized, gt_modalities)
        
        # Total
        total_loss = (self.lambda_seg * total_seg_loss) + (self.lambda_recon * loss_recon)
        
        return {
            "loss": total_loss,
            "seg_loss": total_seg_loss,
            "recon_loss": loss_recon,
            "dice_loss": loss_dice,
            "ce_loss": loss_ce
        }

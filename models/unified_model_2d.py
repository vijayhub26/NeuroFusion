"""
Simplified 2D unified model for brain tumor analysis.
Memory-efficient version using 2D slices instead of 3D volumes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from models.synthesis_2d import GANModalitySynthesis2D


class Simple2DEncoder(nn.Module):
    """Lightweight 2D encoder for modality features."""
    
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UnifiedBrainTumorModel2D(nn.Module):
    """Simplified 2D model for slice-based training."""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.num_modalities = config.data.num_modalities
        
        # GAN synthesis
        self.synthesis_network = GANModalitySynthesis2D(config)
        
        # Simple concatenation fusion (no attention for memory efficiency)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.num_modalities, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Segmentation decoder
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, config.data.num_classes, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, config.data.num_grades)
        )
    
    def synthesize_missing_modalities(
        self,
        modalities: torch.Tensor,
        modality_mask: torch.Tensor,
        num_samples: int = 1
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Optional[torch.Tensor]]:
        """Synthesize missing modalities using GAN."""
        return self.synthesis_network(modalities, modality_mask, num_samples)
    
    def forward(
        self,
        modalities: torch.Tensor,
        modality_mask: torch.Tensor,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            modalities: (B, num_mods, H, W) or (B*num_slices, num_mods, H, W)
            modality_mask: (num_mods,) shared across batch
            training: Training mode
        
        Returns:
            seg_logits, grade_logits, etc.
        """
        # Handle batched slices - modality_mask is shared
        B = modalities.shape[0]
        
        # Synthesize missing (only during inference for now)
        has_missing = (modality_mask == 0).any()
        if has_missing and not training:
            # Expand mask for batch
            batch_mask = modality_mask.unsqueeze(0).expand(B, -1)
            complete_modalities, uncertainty_maps, _ = self.synthesize_missing_modalities(
                modalities, batch_mask, num_samples=1
            )
        else:
            complete_modalities = modalities
            uncertainty_maps = {}
        
        # Simple fusion: concatenate all modalities
        fused = self.fusion_conv(complete_modalities)
        
        # Shared encoder
        encoded = self.encoder(fused)
        
        # Segmentation
        seg_logits = self.seg_decoder(encoded)
        
        # Classification (aggregate across slices if needed)
        grade_logits = self.classifier(encoded)
        
        return {
            "seg_logits": seg_logits,
            "grade_logits": grade_logits,
            "synthesized_modalities": complete_modalities if has_missing else None,
            "uncertainty_maps": uncertainty_maps,
            "encoded_features": encoded
        }
    
    def get_generator(self):
        """Return generator for adversarial training."""
        return self.synthesis_network.generator
    
    def get_discriminator(self):
        """Return discriminator for adversarial training."""
        return self.synthesis_network.discriminator

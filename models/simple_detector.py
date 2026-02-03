"""
Simplified model for Binary Tumor Detection with Missing Modality Synthesis.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from models.synthesis_2d import GANModalitySynthesis2D


class SimpleTumorDetector(nn.Module):
    """
    Simplified system:
    1. Synthesizes missing modalities using GAN.
    2. Detects tumor presence (Binary Classification) using all modalities.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_modalities = config.data.num_modalities
        
        # 1. Modality Generator (Reusing existing GAN)
        self.synthesis_network = GANModalitySynthesis2D(config)
        
        # 2. Binary Tumor Classifier
        # Input: 4 modalities (real or synthesized)
        # Output: Probability of tumor presence
        self.classifier = nn.Sequential(
            # Block 1
            nn.Conv2d(self.num_modalities, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Classifier Head
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Binary output (logits)
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
            modalities: (B, num_slices, 4, H, W) OR (B, 4, H, W)
                If 5D input, it will be flattened to 4D to process slices independently.
            modality_mask: (4,) mask indicating available modalities.
        
        Returns:
            Dict containing:
                - tumor_logits: (N, 1)
                - synthesized_modalities: (N, 4, H, W)
        """
        # Handle 5D input (batched slices): (1, num_slices, 4, H, W) -> (num_slices, 4, H, W)
        if modalities.dim() == 5:
            B, S, C, H, W = modalities.shape
            modalities = modalities.view(-1, C, H, W)
            
            # Expand mask if needed (though mask is usually per batch, here shared)
            # The synthesis network expects (N, 4) mask if input is (N, 4, H, W)
            # But the dataset returns (4,), so we repeat it.
            modality_mask = modality_mask.unsqueeze(0).expand(modalities.shape[0], -1)
        
        # 1. Synthesize Missing Modalities
        # We always synthesize if mask indicates missing, regardless of training
        # to ensure the classifier sees complete data.
        has_missing = (modality_mask == 0).any()
        
        if has_missing:
            complete_modalities, _, synthesized_only = self.synthesize_missing_modalities(
                modalities, modality_mask
            )
        else:
            complete_modalities = modalities
            synthesized_only = None
            
        # 2. Detect Tumor
        tumor_logits = self.classifier(complete_modalities)
        
        return {
            "tumor_logits": tumor_logits,
            "synthesized_modalities": complete_modalities,
            "synthesized_only": synthesized_only
        }
    
    def get_generator(self):
        return self.synthesis_network.generator
        
    def get_discriminator(self):
        return self.synthesis_network.discriminator

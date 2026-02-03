"""
Unified model for Synthesis + Segmentation.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from models.synthesis_2d import GANModalitySynthesis2D, Generator2D


class UnifiedTumorDetector(nn.Module):
    """
    Unified system:
    1. Synthesizes missing modalities using GAN.
    2. Segment Tumor (Binary) using all modalities (Real + Synthesized).
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_modalities = config.data.num_modalities
        
        # 1. Modality Generator (Reusing existing GAN)
        self.synthesis_network = GANModalitySynthesis2D(config)
        
        # 2. Segmentation Network (U-Net)
        # Input: 4 modalities (real or synthesized)
        # Output: 1 channel (logits) -> Tumor vs Background
        self.segmentor = Generator2D(
            in_channels=self.num_modalities,
            out_channels=1,
            base_channels=32,
            num_levels=4,
            use_dropout=False,
            output_activation=None # Logits for Dice/BCE loss
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
        args:
            modalities: (B, num_slices, 4, H, W) OR (B, 4, H, W)
            modality_mask: (4,)
        """
        # Handle 5D input
        if modalities.dim() == 5:
            B, S, C, H, W = modalities.shape
            modalities = modalities.view(-1, C, H, W)
            # Expand mask
            modality_mask = modality_mask.unsqueeze(0).expand(modalities.shape[0], -1)
        
        # 1. Synthesize Missing Modalities
        has_missing = (modality_mask == 0).any()
        
        if has_missing:
            complete_modalities, _, synthesized_only = self.synthesize_missing_modalities(
                modalities, modality_mask
            )
        else:
            complete_modalities = modalities
            synthesized_only = None
            
        # 2. Segment Tumor
        seg_logits = self.segmentor(complete_modalities)
        
        return {
            "seg_logits": seg_logits, # (N, 1, H, W)
            "synthesized_modalities": complete_modalities,
            "synthesized_only": synthesized_only
        }
    
    def get_generator(self):
        return self.synthesis_network.generator
        
    def get_discriminator(self):
        return self.synthesis_network.discriminator

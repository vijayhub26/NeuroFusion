"""
Unified model integrating all components:
- Modality encoders
- Cross-modal attention fusion
- GAN-based modality synthesis
- Shared encoder
- Segmentation decoder
- Classification head
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from models.encoders import ModalityEncoder, SharedEncoder
from models.attention_fusion import SimpleCrossModalFusion
from models.synthesis import GANModalitySynthesis
from models.segmentation import SegmentationDecoder
from models.classification import ClassificationHead


class UnifiedBrainTumorModel(nn.Module):
    """
    End-to-end model for brain tumor analysis with modality imputation.
    
    Novel contributions:
    1. Cross-modal attention fusion for any subset of modalities
    2. Conditional diffusion synthesis for missing modalities with uncertainty
    3. Joint segmentation + classification with synthetic content penalty
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object with model, data, and loss parameters
        """
        super().__init__()
        
        self.config = config
        self.num_modalities = config.data.num_modalities
        self.modality_names = config.data.modalities
        
        # 1. Modality-specific encoders
        self.modality_encoders = nn.ModuleList([
            ModalityEncoder(
                in_channels=1,
                out_channels=config.model.encoder_channels
            )
            for _ in range(self.num_modalities)
        ])
        
        # 2. GAN-based synthesis
        self.synthesis_network = GANModalitySynthesis(config)
        
        # 3. Cross-modal attention fusion
        self.fusion_module = SimpleCrossModalFusion(
            modality_dim=config.model.encoder_channels[-1],
            fusion_dim=config.model.fusion_dim,
            num_modalities=self.num_modalities
        )
        
        # 4. Shared encoder backbone
        self.shared_encoder = SharedEncoder(
            in_channels=config.model.fusion_dim,
            channels=config.model.backbone_channels
        )
        
        # 5. Segmentation decoder
        self.segmentation_decoder = SegmentationDecoder(
            encoder_channels=config.model.backbone_channels,
            decoder_channels=config.model.decoder_channels,
            num_classes=config.data.num_classes
        )
        
        # 6. Classification head
        self.classification_head = ClassificationHead(
            in_channels=config.model.backbone_channels[-1],
            hidden_dim=config.model.classifier_hidden_dim,
            num_classes=config.data.num_grades,
            dropout=config.model.classifier_dropout
        )
    
    def synthesize_missing_modalities(
        self,
        modalities: torch.Tensor,
        modality_mask: torch.Tensor,
        num_samples: int = 1
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Optional[torch.Tensor]]:
        """
        Synthesize missing modalities using GAN.
        
        Args:
            modalities: All modalities (B, num_mods, D, H, W) - missing ones are zeros
            modality_mask: Binary mask (B, num_mods) - 1 if present, 0 if missing
            num_samples: Number of samples for uncertainty (uses dropout)
        
        Returns:
            Complete modalities (with synthesized filled in)
            Uncertainty maps for synthesized modalities
            Synthesized modalities only (for reconstruction loss)
        """
        return self.synthesis_network(modalities, modality_mask, num_samples)
    
    def forward(
        self,
        modalities: torch.Tensor,
        modality_mask: torch.Tensor,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the unified model.
        
        Args:
            modalities: Input modalities (B, num_mods, D, H, W)
            modality_mask: Binary mask (B, num_mods)
            training: Whether in training mode
        
        Returns:
            Dictionary with:
                - seg_logits: Segmentation predictions (B, num_classes, D, H, W)
                - grade_logits: Classification predictions (B, num_grades)
                - synthesized_modalities: Complete modalities (if synthesis occurred)
                - uncertainty_maps: Uncertainty for synthesized modalities
                - attention_weights: Attention weights (for regularization)
        """
        B, num_mods, D, H, W = modalities.shape
        
        # Step 1: Synthesize missing modalities (if any)
        has_missing = (modality_mask == 0).any()
        
        if has_missing and not training:
            # During inference, synthesize missing modalities
            complete_modalities, uncertainty_maps, _ = self.synthesize_missing_modalities(
                modalities, modality_mask, num_samples=self.config.model.num_inference_samples
            )
        else:
            # During training, use ground truth (simulate missing later in loss)
            complete_modalities = modalities
            uncertainty_maps = {}
            synthesized_for_loss = None
        
        # Step 2: Encode each modality
        modality_features = {}
        for mod_idx in range(num_mods):
            # Extract single modality
            mod_data = complete_modalities[:, mod_idx:mod_idx+1]  # (B, 1, D, H, W)
            
            # Encode
            feat = self.modality_encoders[mod_idx](mod_data)
            modality_features[mod_idx] = feat
        
        # Step 3: Cross-modal attention fusion
        fused_features = self.fusion_module(
            modality_features=modality_features,
            modality_mask=modality_mask,
            uncertainty_maps=uncertainty_maps if has_missing else None
        )
        
        # Step 4: Shared encoder
        skip_connections, encoded = self.shared_encoder(fused_features)
        
        # Step 5: Segmentation
        seg_logits = self.segmentation_decoder(encoded, skip_connections)
        
        # Step 6: Classification
        grade_logits = self.classification_head(encoded)
        
        return {
            "seg_logits": seg_logits,
            "grade_logits": grade_logits,
            "synthesized_modalities": complete_modalities if has_missing else None,
            "uncertainty_maps": uncertainty_maps,
            "encoded_features": encoded
        }
    

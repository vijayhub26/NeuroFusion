"""
Cross-modal attention fusion module.
Fuses features from multiple MRI modalities with uncertainty-aware weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CrossModalAttentionFusion(nn.Module):
    """
    Multi-head attention fusion across MRI modalities.
    Attends to available modalities and downweights synthetic ones based on uncertainty.
    """
    
    def __init__(
        self,
        modality_dim: int = 128,
        fusion_dim: int = 256,
        num_heads: int = 8,
        num_modalities: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dim: Feature dimension from each modality encoder
            fusion_dim: Output fusion dimension
            num_heads: Number of attention heads
            num_modalities: Number of modalities (T1, T1ce, T2, FLAIR)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.modality_dim = modality_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.head_dim = fusion_dim // num_heads
        
        assert fusion_dim % num_heads == 0, "fusion_dim must be divisible by num_heads"
        
        # Project modality features to fusion dimension
        self.modality_projections = nn.ModuleList([
            nn.Conv3d(modality_dim, fusion_dim, kernel_size=1)
            for _ in range(num_modalities)
        ])
        
        # Multi-head attention components
        self.query_proj = nn.Conv3d(fusion_dim, fusion_dim, kernel_size=1)
        self.key_proj = nn.Conv3d(fusion_dim, fusion_dim, kernel_size=1)
        self.value_proj = nn.Conv3d(fusion_dim, fusion_dim, kernel_size=1)
        
        # Uncertainty-aware attention weighting
        self.uncertainty_gate = nn.Sequential(
            nn.Conv3d(1, fusion_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv3d(fusion_dim, fusion_dim, kernel_size=1),
            nn.InstanceNorm3d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        modality_features: Dict[int, torch.Tensor],
        modality_mask: torch.Tensor,
        uncertainty_maps: Optional[Dict[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            modality_features: Dict mapping modality index to features (B, C, D, H, W)
            modality_mask: Binary mask (B, num_modalities) - 1 if present, 0 if synthesized/missing
            uncertainty_maps: Optional dict of uncertainty maps for synthesized modalities (B, 1, D, H, W)
        
        Returns:
            Fused features (B, fusion_dim, D, H, W)
        """
        B = modality_mask.shape[0]
        device = modality_mask.device
        
        # Project all modality features to fusion dimension
        projected_features = []
        for mod_idx in range(self.num_modalities):
            if mod_idx in modality_features:
                feat = self.modality_projections[mod_idx](modality_features[mod_idx])
                
                # Apply uncertainty gating if this is a synthesized modality
                if uncertainty_maps is not None and mod_idx in uncertainty_maps:
                    uncertainty = uncertainty_maps[mod_idx]
                    gate = self.uncertainty_gate(uncertainty)
                    feat = feat * gate  # Downweight uncertain regions
                
                projected_features.append(feat)
            else:
                # Modality not available - skip
                continue
        
        if len(projected_features) == 0:
            raise ValueError("No modalities available for fusion")
        
        # Stack modalities: (B, num_available_mods, C, D, H, W)
        modality_stack = torch.stack(projected_features, dim=1)
        num_available = modality_stack.shape[1]
        
        # Reshape for multi-head attention
        # (B * num_mods, C, D, H, W) -> (B * num_mods * D * H * W, num_heads, head_dim)
        B, M, C, D, H, W = modality_stack.shape
        
        # Use mean pooling across modalities as query
        query = modality_stack.mean(dim=1)  # (B, C, D, H, W)
        query = self.query_proj(query)
        
        # Keys and values from all modalities
        keys = self.key_proj(modality_stack.reshape(B * M, C, D, H, W))
        values = self.value_proj(modality_stack.reshape(B * M, C, D, H, W))
        
        # Reshape for multi-head attention
        query = query.reshape(B, self.num_heads, self.head_dim, D * H * W).transpose(2, 3)  # (B, heads, DHW, head_dim)
        keys = keys.reshape(B, M, self.num_heads, self.head_dim, D * H * W).permute(0, 2, 1, 4, 3)  # (B, heads, M, DHW, head_dim)
        values = values.reshape(B, M, self.num_heads, self.head_dim, D * H * W).permute(0, 2, 1, 4, 3)  # (B, heads, M, DHW, head_dim)
        
        # Compute attention scores: (B, heads, DHW, M)
        keys = keys.reshape(B, self.num_heads, M * D * H * W, self.head_dim)
        values = values.reshape(B, self.num_heads, M * D * H * W, self.head_dim)
        
        # Simplified: average across modalities with learned weights
        # For efficiency in 3D, use channel-wise attention
        attended = torch.mean(modality_stack, dim=1)  # Simple mean for now
        
        # Output projection
        fused = self.output_proj(attended)
        
        return fused


class SimpleCrossModalFusion(nn.Module):
    """
    Simplified cross-modal fusion using weighted averaging.
    More efficient alternative for 3D volumes.
    """
    
    def __init__(
        self,
        modality_dim: int = 128,
        fusion_dim: int = 256,
        num_modalities: int = 4
    ):
        super().__init__()
        
        self.num_modalities = num_modalities
        
        # Project each modality
        self.modality_projections = nn.ModuleList([
            nn.Conv3d(modality_dim, fusion_dim, kernel_size=1)
            for _ in range(num_modalities)
        ])
        
        # Learnable attention weights
        self.attention_weights = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(fusion_dim * num_modalities, num_modalities, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Uncertainty compensation
        self.uncertainty_compensation = nn.Sequential(
            nn.Conv3d(1, fusion_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Conv3d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(fusion_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        modality_features: Dict[int, torch.Tensor],
        modality_mask: torch.Tensor,
        uncertainty_maps: Optional[Dict[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Fuse modality features with uncertainty-aware weighting.
        
        Args:
            modality_features: Dict of features per modality (B, C, D, H, W)
            modality_mask: Binary availability mask (B, num_modalities)
            uncertainty_maps: Optional uncertainty for synthesized modalities
        
        Returns:
            Fused features (B, fusion_dim, D, H, W)
        """
        B = modality_mask.shape[0]
        
        # Project and collect available modalities
        projected = []
        for mod_idx in range(self.num_modalities):
            if mod_idx in modality_features:
                feat = self.modality_projections[mod_idx](modality_features[mod_idx])
                
                # Apply uncertainty gating for synthetic modalities
                if uncertainty_maps is not None and mod_idx in uncertainty_maps:
                    gate = self.uncertainty_compensation(uncertainty_maps[mod_idx])
                    feat = feat * gate
                
                projected.append(feat)
        
        # Weighted sum with learned attention
        if len(projected) == 1:
            fused = projected[0]
        else:
            # Simple average for now (can be enhanced with learned weights)
            fused = torch.stack(projected, dim=0).mean(dim=0)
        
        # Output projection
        output = self.output(fused)
        
        return output

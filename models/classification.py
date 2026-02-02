"""
Classification head for tumor grade classification.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head for tumor grade (LGG vs HGG).
    Uses global pooling from encoder features.
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        hidden_dim: int = 512,
        num_classes: int = 2,  # LGG (0) vs HGG (1)
        dropout: float = 0.3
    ):
        """
        Args:
            in_channels: Channel dimension from encoder
            hidden_dim: Hidden layer dimension
            num_classes: Number of tumor grades (2 for LGG/HGG)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded features (B, C, D, H, W)
        
        Returns:
            Classification logits (B, num_classes)
        """
        # Global average pooling
        x = self.global_pool(x)  # (B, C, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        
        # Classify
        logits = self.classifier(x)
        
        return logits

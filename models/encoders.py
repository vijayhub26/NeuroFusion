"""
Modality-specific encoders for extracting features from individual MRI sequences.
"""

import torch
import torch.nn as nn
from typing import List


class ModalityEncoder(nn.Module):
    """
    Lightweight 3D CNN encoder for a single MRI modality.
    Extracts modality-specific features before fusion.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        padding: int = 1
    ):
        """
        Args:
            in_channels: Number of input channels (1 for single modality)
            out_channels: List of channel dimensions for each conv block
            kernel_size: Convolution kernel size
            padding: Padding for convolutions
        """
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList()
        
        current_channels = in_channels
        for out_ch in out_channels:
            block = nn.Sequential(
                nn.Conv3d(current_channels, out_ch, kernel_size, padding=padding),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size, padding=padding),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.encoder_blocks.append(block)
            current_channels = out_ch
        
        # Output projection to fusion dimension
        self.out_channels = out_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 1, D, H, W)
        
        Returns:
            Encoded features (B, C, D', H', W')
        """
        for block in self.encoder_blocks:
            x = block(x)
        return x


class SharedEncoder(nn.Module):
    """
    Shared 3D U-Net style encoder backbone.
    Processes fused multi-modal features.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        channels: List[int] = [64, 128, 256, 512],
        kernel_size: int = 3,
        padding: int = 1
    ):
        """
        Args:
            in_channels: Number of input channels from fusion module
            channels: Channel dimensions for each encoder level
            kernel_size: Convolution kernel size
            padding: Padding
        """
        super().__init__()
        
        self.encoder_levels = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        current_channels = in_channels
        for ch in channels:
            # Encoder block
            block = nn.Sequential(
                nn.Conv3d(current_channels, ch, kernel_size, padding=padding),
                nn.InstanceNorm3d(ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(ch, ch, kernel_size, padding=padding),
                nn.InstanceNorm3d(ch),
                nn.ReLU(inplace=True)
            )
            self.encoder_levels.append(block)
            
            # Downsampling (except for last level)
            if ch != channels[-1]:
                self.downsample.append(nn.MaxPool3d(2))
            
            current_channels = ch
        
        self.channels = channels
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Fused input (B, C, D, H, W)
        
        Returns:
            Tuple of features at each level (for skip connections)
            and final encoded features
        """
        skip_connections = []
        
        for i, block in enumerate(self.encoder_levels):
            x = block(x)
            skip_connections.append(x)
            
            # Downsample if not at last level
            if i < len(self.downsample):
                x = self.downsample[i](x)
        
        return skip_connections, x

"""
Segmentation decoder for brain tumor segmentation.
""" 

import torch
import torch.nn as nn
from typing import List


class SegmentationDecoder(nn.Module):
    """
    3D U-Net style decoder for multi-class brain tumor segmentation.
    """
    
    def __init__(
        self,
        encoder_channels: List[int] = [64, 128, 256, 512],
        decoder_channels: List[int] = [512, 256, 128, 64],
        num_classes: int = 4,  # background, necrosis, edema, enhancing
        kernel_size: int = 3,
        padding: int = 1
    ):
        """
        Args:
            encoder_channels: Channel dims from encoder (for skip connections)
            decoder_channels: Channel dims for decoder blocks
            num_classes: Number of segmentation classes
            kernel_size: Convolution kernel size
            padding: Padding
        """
        super().__init__()
        
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        # Build decoder levels
        for i in range(len(decoder_channels)):
            # Input channels: current level + skip connection from encoder
            if i == 0:
                in_ch = encoder_channels[-1]  # From bottleneck
            else:
                in_ch = decoder_channels[i - 1] + encoder_channels[-(i + 1)]
            
            out_ch = decoder_channels[i]
            
            # Decoder block
            block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size, padding=padding),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.decoder_blocks.append(block)
            
            # Upsampling
            if i < len(decoder_channels) - 1:
                self.upsample_layers.append(
                    nn.ConvTranspose3d(out_ch, out_ch, kernel_size=2, stride=2)
                )
        
        # Final output layer
        self.output = nn.Conv3d(decoder_channels[-1], num_classes, kernel_size=1)
    
    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: Encoded features from bottleneck (B, C, D, H, W)
            skip_connections: List of skip connections from encoder
                (from shallow to deep)
        
        Returns:
            Segmentation logits (B, num_classes, D, H, W)
        """
        # Reverse skip connections (we need deep to shallow)
        skip_connections = skip_connections[::-1]
        
        # First decoder block (no skip connection)
        x = self.decoder_blocks[0](x)
        
        # Remaining decoder blocks with skip connections
        for i in range(1, len(self.decoder_blocks)):
            # Upsample
            x = self.upsample_layers[i - 1](x)
            
            # Concatenate with skip connection
            skip = skip_connections[i]
            
            # Handle size mismatch (if any)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode='trilinear', align_corners=True
                )
            
            x = torch.cat([x, skip], dim=1)
            
            # Decoder block
            x = self.decoder_blocks[i](x)
        
        # Output
        output = self.output(x)
        
        return output

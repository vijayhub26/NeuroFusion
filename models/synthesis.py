"""
GAN-based conditional modality synthesis.
Replaces diffusion model for faster, lower-memory training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class Generator3D(nn.Module):
    """
    3D Conditional GAN Generator for missing modality synthesis.
    
    Takes available modalities as condition and generates missing modality.
    """
    
    def __init__(
        self,
        in_channels: int = 4,  # Number of input modalities (condition)
        out_channels: int = 1,  # Generated modality
        base_channels: int = 32,
        num_levels: int = 4,
        use_dropout: bool = True
    ):
        super().__init__()
        
        self.use_dropout = use_dropout
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        channels = base_channels
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else channels // 2
            out_ch = channels
            self.encoder.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            channels *= 2
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(channels // 2, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.5) if use_dropout else nn.Identity(),
        )
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        for i in range(num_levels):
            in_ch = channels
            out_ch = channels // 2
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout3d(0.3) if use_dropout else nn.Identity(),
            ))
            channels //= 2
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv3d(base_channels, out_channels, kernel_size=1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Generate missing modality conditioned on available modalities.
        
        Args:
            condition: Available modalities (B, C, D, H, W)
        
        Returns:
            Generated modality (B, 1, D, H, W)
        """
        # Encoder
        x = condition
        skip_connections = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = F.max_pool3d(x, kernel_size=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x)
            # Add skip connection (except last level)
            if i < len(self.decoder) - 1:
                skip = skip_connections[-(i + 1)]
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
                x = x + skip
        
        # Output
        return self.output(x)


class Discriminator3D(nn.Module):
    """
    3D Conditional GAN Discriminator.
    
    Determines if a modality is real or generated, conditioned on other modalities.
    """
    
    def __init__(
        self,
        in_channels: int = 5,  # Target modality + condition modalities
        base_channels: int = 32,
        num_levels: int = 4
    ):
        super().__init__()
        
        layers = []
        channels = base_channels
        
        # First conv (no norm)
        layers.append(nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        # Downsampling blocks
        for i in range(num_levels - 1):
            in_ch = channels
            out_ch = min(channels * 2, 512)
            layers.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            channels = out_ch
        
        # Final conv
        layers.append(nn.Conv3d(channels, 1, kernel_size=4, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Classify if target modality is real or fake.
        
        Args:
            target: Modality to classify (B, 1, D, H, W)
            condition: Conditioning modalities (B, C, D, H, W)
        
        Returns:
            Logits (B, 1, D', H', W') - PatchGAN output
        """
        x = torch.cat([target, condition], dim=1)
        return self.model(x)


class GANModalitySynthesis(nn.Module):
    """
    Complete GAN-based modality synthesis module.
    
    Replaces diffusion model for faster, lower-memory synthesis.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.num_modalities = config.data.num_modalities
        
        # Generator and Discriminator
        self.generator = Generator3D(
            in_channels=self.num_modalities,  # All modalities as condition
            out_channels=1,
            base_channels=16,  # Reduced for memory
            num_levels=3,
            use_dropout=True
        )
        
        self.discriminator = Discriminator3D(
            in_channels=self.num_modalities + 1,  # Target + condition
            base_channels=16,
            num_levels=3
        )
    
    def synthesize_modality(
        self,
        modalities: torch.Tensor,
        modality_mask: torch.Tensor,
        missing_idx: int,
        num_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Synthesize a missing modality.
        
        Args:
            modalities: All modalities (B, num_mods, D, H, W)
            modality_mask: Which are available (B, num_mods)
            missing_idx: Index of modality to synthesize
            num_samples: Number of samples for uncertainty (uses dropout)
        
        Returns:
            synthesized: Generated modality (B, 1, D, H, W)
            uncertainty: Pixel-wise uncertainty (B, 1, D, H, W)
        """
        # Prepare condition (all modalities including missing one set to 0)
        condition = modalities.clone()
        condition[:, missing_idx] = 0
        
        if num_samples == 1:
            # Single forward pass
            synthesized = self.generator(condition)
            uncertainty = torch.zeros_like(synthesized)
        else:
            # Multiple forward passes with dropout for uncertainty
            self.generator.train()  # Enable dropout
            samples = []
            for _ in range(num_samples):
                with torch.no_grad():
                    sample = self.generator(condition)
                    samples.append(sample)
            
            # Mean and variance
            samples = torch.stack(samples, dim=0)  # (num_samples, B, 1, D, H, W)
            synthesized = samples.mean(dim=0)
            uncertainty = samples.var(dim=0)
            
            self.generator.eval()
        
        return synthesized, uncertainty
    
    def forward(
        self,
        modalities: torch.Tensor,
        modality_mask: torch.Tensor,
        num_samples: int = 1
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Optional[torch.Tensor]]:
        """
        Synthesize all missing modalities.
        
        Args:
            modalities: Input modalities (B, num_mods, D, H, W)
            modality_mask: Binary mask (B, num_mods) - 1=available, 0=missing
            num_samples: Number of samples for uncertainty estimation
        
        Returns:
            complete_modalities: All modalities with missing ones synthesized
            uncertainty_maps: Dict mapping modality index to uncertainty
            synthesized_modalities: Only the synthesized ones (for loss)
        """
        batch_size = modalities.shape[0]
        
        # Find which modalities are missing
        missing_indices = (modality_mask == 0).nonzero(as_tuple=True)
        
        if len(missing_indices[0]) == 0:
            # No missing modalities
            return modalities, {}, None
        
        # Synthesize each missing modality
        complete_modalities = modalities.clone()
        uncertainty_maps = {}
        synthesized_list = []
        
        for batch_idx, mod_idx in zip(missing_indices[0], missing_indices[1]):
            synthesized, uncertainty = self.synthesize_modality(
                modalities[batch_idx:batch_idx+1],
                modality_mask[batch_idx:batch_idx+1],
                mod_idx.item(),
                num_samples=num_samples
            )
            
            complete_modalities[batch_idx, mod_idx] = synthesized.squeeze(1)
            uncertainty_maps[mod_idx.item()] = uncertainty
            synthesized_list.append(synthesized)
        
        # Stack synthesized modalities for loss computation
        if synthesized_list:
            synthesized_modalities = torch.cat(synthesized_list, dim=0)
        else:
            synthesized_modalities = None
        
        return complete_modalities, uncertainty_maps, synthesized_modalities

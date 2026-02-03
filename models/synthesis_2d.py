"""
2D GAN-based conditional modality synthesis.
Optimized for memory efficiency with 2D slices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class Generator2D(nn.Module):
    """2D Conditional GAN Generator for missing modality synthesis."""
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 4,
        use_dropout: bool = True
    ):
        super().__init__()
        
        self.use_dropout = use_dropout
        
        # Encoder
        self.encoder = nn.ModuleList()
        channels = base_channels
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else channels // 2
            out_ch = channels
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            channels *= 2
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5) if use_dropout else nn.Identity(),
        )
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_levels):
            in_ch = channels
            out_ch = channels // 2
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.3) if use_dropout else nn.Identity(),
            ))
            channels //= 2
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Generate missing modality from condition."""
        x = condition
        skip_connections = []
        
        # Encoder
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x)
            if i < len(self.decoder) - 1:
                skip = skip_connections[-(i + 1)]
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = x + skip
        
        return self.output(x)


class Discriminator2D(nn.Module):
    """2D Conditional GAN Discriminator."""
    
    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 32,
        num_levels: int = 4
    ):
        super().__init__()
        
        layers = []
        channels = base_channels
        
        # First conv
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        # Downsampling blocks
        for i in range(num_levels - 1):
            in_ch = channels
            out_ch = min(channels * 2, 512)
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            channels = out_ch
        
        # Final conv
        layers.append(nn.Conv2d(channels, 1, kernel_size=4, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Classify if target modality is real or fake."""
        x = torch.cat([target, condition], dim=1)
        return self.model(x)


class GANModalitySynthesis2D(nn.Module):
    """Complete 2D GAN-based modality synthesis module."""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.num_modalities = config.data.num_modalities
        
        self.generator = Generator2D(
            in_channels=self.num_modalities,
            out_channels=1,
            base_channels=32,
            num_levels=3,
            use_dropout=True
        )
        
        self.discriminator = Discriminator2D(
            in_channels=self.num_modalities + 1,
            base_channels=32,
            num_levels=3
        )
    
    def synthesize_modality(
        self,
        modalities: torch.Tensor,
        modality_mask: torch.Tensor,
        missing_idx: int,
        num_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize a missing modality."""
        condition = modalities.clone()
        condition[:, missing_idx] = 0
        
        if num_samples == 1:
            synthesized = self.generator(condition)
            uncertainty = torch.zeros_like(synthesized)
        else:
            self.generator.train()
            samples = []
            for _ in range(num_samples):
                with torch.no_grad():
                    sample = self.generator(condition)
                    samples.append(sample)
            
            samples = torch.stack(samples, dim=0)
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
        """Synthesize all missing modalities."""
        batch_size = modalities.shape[0]
        
        missing_indices = (modality_mask == 0).nonzero(as_tuple=True)
        
        if len(missing_indices[0]) == 0:
            return modalities, {}, None
        
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
        
        if synthesized_list:
            synthesized_modalities = torch.cat(synthesized_list, dim=0)
        else:
            synthesized_modalities = None
        
        return complete_modalities, uncertainty_maps, synthesized_modalities

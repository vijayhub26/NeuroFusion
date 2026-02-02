"""
Conditional diffusion model for MRI modality synthesis.
Generates missing modalities conditioned on available ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embeddings for diffusion."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) tensor of timestep indices
        Returns:
            (B, dim) embeddings
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class UNet3D(nn.Module):
    """
    3D U-Net denoising network for diffusion model.
    Conditioned on timestep and available modalities.
    """
    
    def __init__(
        self,
        in_channels: int = 1,  # Single modality to synthesize
        condition_channels: int = 3,  # Available modalities as condition
        model_channels: int = 64,
        time_embed_dim: int = 256
    ):
        super().__init__()
        
        self.time_embed_dim = time_embed_dim
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )
        
        # Encoder
        self.enc1 = self._make_block(in_channels + condition_channels, model_channels, time_embed_dim)
        self.enc2 = self._make_block(model_channels, model_channels * 2, time_embed_dim)
        self.enc3 = self._make_block(model_channels * 2, model_channels * 4, time_embed_dim)
        
        # Bottleneck
        self.bottleneck = self._make_block(model_channels * 4, model_channels * 8, time_embed_dim)
        
        # Decoder
        self.dec3 = self._make_block(model_channels * 8 + model_channels * 4, model_channels * 4, time_embed_dim )
        self.dec2 = self._make_block(model_channels * 4 + model_channels * 2, model_channels * 2, time_embed_dim)
        self.dec1 = self._make_block(model_channels * 2 + model_channels, model_channels, time_embed_dim)
        
        # Output
        self.out = nn.Conv3d(model_channels, in_channels, kernel_size=1)
        
        # Downsampling and upsampling
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    
    def _make_block(self, in_ch: int, out_ch: int, time_dim: int) -> nn.Module:
        """Create a residual block with time conditioning."""
        return ResidualBlock3D(in_ch, out_ch, time_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy image (B, 1, D, H, W)
            timesteps: Timestep indices (B,)
            condition: Available modalities (B, C_cond, D, H, W)
        
        Returns:
            Predicted noise (B, 1, D, H, W)
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)  # (B, time_embed_dim)
        
        # Concatenate input with condition
        x = torch.cat([x, condition], dim=1)  # (B, 1+C_cond, D, H, W)
        
        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3), t_emb)
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upsample(b), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1), t_emb)
        
        # Output
        out = self.out(d1)
        
        return out


class ResidualBlock3D(nn.Module):
    """Residual block with time conditioning for 3D U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        self.act = nn.SiLU()
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, C, D, H, W)
            t_emb: Time embedding (B, time_dim)
        """
        residual = self.skip(x)
        
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Add time conditioning
        t_proj = self.time_proj(t_emb)[:, :, None, None, None]
        h = h + t_proj
        
        h = self.act(h)
        
        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + residual


class ConditionalDiffusionSynthesis(nn.Module):
    """
    Conditional DDPM for MRI modality synthesis.
    Generates missing modalities conditioned on available ones.
    """
    
    def __init__(
        self,
        num_modalities: int = 4,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        model_channels: int = 64
    ):
        super().__init__()
        
        self.num_modalities = num_modalities
        self.num_steps = num_steps
        
        # Diffusion schedule (linear)
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, num_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        
        # Denoising network (one per modality to synthesize, or shared)
        # For simplicity, using a shared network
        self.denoising_net = UNet3D(
            in_channels=1,
            condition_channels=num_modalities - 1,  # Conditioned on other modalities
            model_channels=model_channels
        )
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to x_start at timestep t.
        
        Args:
            x_start: Clean image (B, 1, D, H, W)
            t: Timesteps (B,)
            noise: Optional noise (if None, sample from N(0, I))
        
        Returns:
            Noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def forward(
        self,
        target_modality: torch.Tensor,
        condition_modalities: torch.Tensor,
        modality_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass: predict noise in target modality.
        
        Args:
            target_modality: Ground truth modality to synthesize (B, 1, D, H, W)
            condition_modalities: Available modalities (B, C, D, H, W)
            modality_mask: Which modalities are available (B, num_modalities)
        
        Returns:
            Predicted noise, target noise (for loss computation)
        """
        B = target_modality.shape[0]
        device = target_modality.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (B,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(target_modality)
        
        # Add noise to target
        x_noisy = self.q_sample(target_modality, t, noise)
        
        # Predict noise
        predicted_noise = self.denoising_net(x_noisy, t, condition_modalities)
        
        return predicted_noise, noise
    
    @torch.no_grad()
    def sample(
        self,
        condition_modalities: torch.Tensor,
        num_samples: int = 1,
        ddim_steps: Optional[int] = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (synthesize) a modality using DDIM sampling.
        
        Args:
            condition_modalities: Available modalities (B, C, D, H, W)
            num_samples: Number of samples for uncertainty quantification
            ddim_steps: Number of DDIM steps (None = full DDPM)
        
        Returns:
            Synthesized modality (B, 1, D, H, W)
            Uncertainty map (pixel-wise variance across samples)
        """
        B, _, D, H, W = condition_modalities.shape
        device = condition_modalities.device
        
        samples = []
        
        for _ in range(num_samples):
            # Start from random noise
            x = torch.randn(B, 1, D, H, W, device=device)
            
            # DDIM sampling (simplified - use fewer steps)
            if ddim_steps is not None:
                timesteps = torch.linspace(self.num_steps - 1, 0, ddim_steps, device=device).long()
            else:
                timesteps = torch.arange(self.num_steps - 1, -1, -1, device=device)
            
            for t in timesteps:
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = self.denoising_net(x, t_batch, condition_modalities)
                
                # DDIM update (simplified)
                alpha_t = self.alphas_cumprod[t]
                alpha_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
                
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                pred_x0 = torch.clamp(pred_x0, -1, 1)
                
                if t > 0:
                    x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * predicted_noise
                else:
                    x = pred_x0
            
            samples.append(x)
        
        # Compute mean and uncertainty
        samples_tensor = torch.stack(samples, dim=0)  # (num_samples, B, 1, D, H, W)
        synthesized = samples_tensor.mean(dim=0)
        uncertainty = samples_tensor.var(dim=0)
        
        return synthesized, uncertainty

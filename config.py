"""
Configuration file for the Cross-Modal Fusion Brain Tumor Analysis project.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""
    # BraTS dataset paths
    data_root: str = "data/BraTS"
    train_dir: str = "data/BraTS/train"
    val_dir: str = "data/BraTS/val"
    test_dir: str = "data/BraTS/test"
    
    # Modalities
    modalities: List[str] = None
    num_modalities: int = 4  # T1, T1ce, T2, FLAIR
    
    # Data dimensions
    image_size: Tuple[int, int, int] = (128, 128, 128)  # (D, H, W)
    num_classes: int = 4  # background, necrosis, edema, enhancing tumor
    
    # Classification
    num_grades: int = 2  # LGG (Low Grade), HGG (High Grade)
    
    # Data loading
    batch_size: int = 2
    num_workers: int = 4
    
    # Missing modality simulation
    missing_prob: float = 0.3  # Probability of randomly masking a modality during training
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ["t1", "t1ce", "t2", "flair"]


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Modality encoders
    encoder_channels: List[int] = None
    
    # Cross-modal attention fusion
    fusion_dim: int = 256
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Shared encoder backbone (3D U-Net)
    backbone_channels: List[int] = None
    
    # Diffusion synthesis
    diffusion_steps: int = 1000
    diffusion_beta_start: float = 0.0001
    diffusion_beta_end: float = 0.02
    diffusion_schedule: str = "linear"  # or "cosine"
    num_inference_samples: int = 5  # For uncertainty quantification
    
    # Segmentation decoder
    decoder_channels: List[int] = None
    
    # Classification head
    classifier_hidden_dim: int = 512
    classifier_dropout: float = 0.3
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [32, 64, 128]
        if self.backbone_channels is None:
            self.backbone_channels = [64, 128, 256, 512]
        if self.decoder_channels is None:
            self.decoder_channels = [512, 256, 128, 64]


@dataclass
class LossConfig:
    """Loss function weights and configuration."""
    # Loss weights
    lambda_seg: float = 1.0  # Segmentation loss weight
    lambda_cls: float = 0.5  # Classification loss weight
    lambda_synthesis: float = 0.1  # Modality synthesis reconstruction loss
    lambda_uncertainty: float = 0.05  # Uncertainty regularization
    lambda_attention_reg: float = 0.01  # Penalty on synthetic modality attention
    
    # Segmentation loss
    dice_weight: float = 0.5
    ce_weight: float = 0.5
    
    # Classification loss
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine", "plateau", or "step"
    warmup_epochs: int = 5
    
    # Training
    max_epochs: int = 200
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Validation
    val_check_interval: float = 1.0  # Check every epoch
    
    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "val/dice_mean"
    monitor_mode: str = "max"
    
    # Early stopping
    early_stopping_patience: int = 20
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_every_n_steps: int = 10
    experiment_name: str = "cross_modal_fusion"
    
    # Reproducibility
    seed: int = 42


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    data: DataConfig = None
    model: ModelConfig = None
    loss: LossConfig = None
    training: TrainingConfig = None
    
    # Device
    device: str = "cuda"
    num_gpus: int = 1
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.loss is None:
            self.loss = LossConfig()
        if self.training is None:
            self.training = TrainingConfig()


# Default configuration instance
default_config = Config()

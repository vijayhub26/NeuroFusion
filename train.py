"""
Training script for the unified brain tumor analysis model.
Uses PyTorch Lightning for structured training.
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from config import Config, default_config
from models import UnifiedBrainTumorModel2D  # Use 2D model
from data import get_brats_dataloaders_2d  # Use 2D slices
from losses import CombinedLoss
from utils.metrics import dice_coefficient, classification_metrics


class BrainTumorLightningModule(pl.LightningModule):
    """PyTorch Lightning module for training."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Model (2D version)
        self.model = UnifiedBrainTumorModel2D(config)
        
        # Loss
        self.criterion = CombinedLoss(config)
    
    def forward(self, modalities, modality_mask):
        return self.model(modalities, modality_mask, training=self.training)
    
    def training_step(self, batch, batch_idx):
        """Training step - handle batched slices."""
        # Dataset returns: (batch_size=1, num_slices, 4, H, W)
        # Need to reshape to: (num_slices, 4, H, W) for 2D model
        
        modalities = batch["modalities"]  # (1, num_slices, 4, H, W)
        batch_size = modalities.shape[0]
        num_slices = modalities.shape[1]
        
        # Reshape: (1, num_slices, 4, H, W) -> (num_slices, 4, H, W)
        modalities = modalities.view(-1, *modalities.shape[2:])  # (num_slices, 4, H, W)
        
        modality_mask = batch["modality_mask"]  # (4,)
        seg_targets = batch["seg"].view(-1, *batch["seg"].shape[2:])  # (num_slices, H, W)
        grade_targets = batch["grade"]  # scalar
        
        # Forward pass
        outputs = self(modalities, modality_mask)
        
        # Expand grade for all slices
        grade_targets_expanded = grade_targets.repeat(num_slices * batch_size)
        
        # Debug: Print shapes and values
        if batch_idx == 0:
            print(f"\n[DEBUG] Batch {batch_idx}:")
            print(f"  grade_logits shape: {outputs['grade_logits'].shape}")
            print(f"  grade_targets_expanded shape: {grade_targets_expanded.shape}")
            print(f"  grade_targets values: {grade_targets_expanded[:5]}")
            print(f"  grade_logits num_classes: {outputs['grade_logits'].shape[-1]}")
        
        # Compute loss
        total_loss, loss_dict = self.criterion(
            outputs, seg_targets, grade_targets_expanded, modalities, modality_mask
        )
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Compute metrics
        seg_preds = outputs["seg_logits"].argmax(dim=1)
        dice_scores = dice_coefficient(
            seg_preds, seg_targets, self.config.data.num_classes
        )
        for key, value in dice_scores.items():
            self.log(f"train/{key}", value, on_step=False, on_epoch=True)
        
        cls_metrics = classification_metrics(outputs["grade_logits"], grade_targets)
        for key, value in cls_metrics.items():
            self.log(f"train/cls_{key}", value, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - handle batched slices."""
        modalities = batch["modalities"]  # (1, num_slices, 4, H, W)
        batch_size = modalities.shape[0]
        num_slices = modalities.shape[1]
        
        # Reshape
        modalities = modalities.view(-1, *modalities.shape[2:])  # (num_slices, 4, H, W)
        modality_mask = batch["modality_mask"]  # (4,)
        seg_targets = batch["seg"].view(-1, *batch["seg"].shape[2:])  # (num_slices, H, W)
        grade_targets = batch["grade"]  # scalar
        
        # Forward pass
        outputs = self(modalities, modality_mask)
        
        # Expand grade
        grade_targets_expanded = grade_targets.repeat(num_slices * batch_size)
        
        # Compute loss
        total_loss, loss_dict = self.criterion(
            outputs, seg_targets, grade_targets_expanded, modalities, modality_mask
        )
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute metrics
        seg_preds = outputs["seg_logits"].argmax(dim=1)
        dice_scores = dice_coefficient(
            seg_preds, seg_targets, self.config.data.num_classes
        )
        for key, value in dice_scores.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True)
        
        cls_metrics = classification_metrics(outputs["grade_logits"], grade_targets)
        for key, value in cls_metrics.items():
            self.log(f"val/cls_{key}", value, on_step=False, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        if self.config.training.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=1e-6
            )
        elif self.config.training.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config.training.monitor_mode,
                patience=5,
                factor=0.5
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.config.training.monitor_metric
            }
        }


def train(config: Config = None):
    """Main training function."""
    if config is None:
        config = default_config
    
    # Set seed for reproducibility
    pl.seed_everything(config.training.seed)
    
    # Create data loaders
    print("Loading BraTS dataset...")
    train_loader, val_loader = get_brats_dataloaders_2d(
        data_root=config.data.data_root,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        modalities=config.data.modalities,
        image_size=config.data.image_size,
        missing_prob=config.data.missing_prob,
        num_slices_per_scan=config.data.num_slices_per_scan
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Initializing model...")
    model = BrainTumorLightningModule(config)
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="{epoch}-{val/dice_mean:.3f}",
        monitor=config.training.monitor_metric,
        mode=config.training.monitor_mode,
        save_top_k=config.training.save_top_k,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor=config.training.monitor_metric,
        patience=config.training.early_stopping_patience,
        mode=config.training.monitor_mode,
        verbose=True
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name=config.training.experiment_name
    )
    
    # Optional: W&B logger (uncomment if using)
    # logger = WandbLogger(
    #     project="brain-tumor-fusion",
    #     name=config.training.experiment_name,
    #     save_dir=config.log_dir
    # )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu" if config.device == "cuda" else "cpu",
        devices=config.num_gpus if config.device == "cuda" else 1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        precision="16-mixed" if config.training.use_amp else 32,
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Training complete!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    # You can modify config here or load from YAML
    config = default_config
    
    # Example: Override some settings
    # config.training.max_epochs = 100
    # config.data.batch_size = 4
    
    train(config)

"""
Full training script for Segmentation + Synthesis (Unified Model).
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import Config
from data import get_brats_dataloaders_2d
from models.unified_detector import UnifiedTumorDetector
from losses.unified_loss import UnifiedLoss

class UnifiedSystem(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        self.model = UnifiedTumorDetector(config)
        self.loss_fn = UnifiedLoss(lambda_seg=1.0, lambda_recon=10.0) # Recon needs higher weight usually
        
    def forward(self, modalities, modality_mask):
        return self.model(modalities, modality_mask, training=self.training)
    
    def training_step(self, batch, batch_idx):
        # Unpack batch
        # modalities: (1, num_slices, 4, H, W) -> flattened
        modalities = batch["modalities"]
        gt_modalities = batch["gt_modalities"] # Clean GT
        
        num_slices = modalities.shape[1]
        
        # Flatten
        cols = modalities.shape[2]
        H, W = modalities.shape[3], modalities.shape[4]
        
        modalities = modalities.view(-1, cols, H, W)
        gt_modalities = gt_modalities.view(-1, cols, H, W)
        
        seg_targets = batch["seg"].view(-1, H, W)
        modality_mask = batch["modality_mask"]
        
        # Forward
        outputs = self(modalities, modality_mask)
        seg_logits = outputs["seg_logits"]
        synthesized = outputs["synthesized_modalities"]
        
        # Loss
        loss_dict = self.loss_fn(
            seg_logits, seg_targets,
            synthesized, gt_modalities, modality_mask
        )
        
        # Log
        self.log("train/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/seg_loss", loss_dict["seg_loss"], on_step=False, on_epoch=True)
        self.log("train/recon_loss", loss_dict["recon_loss"], on_step=False, on_epoch=True)
        self.log("train/dice_loss", loss_dict["dice_loss"], on_step=False, on_epoch=True)
        
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        modalities = batch["modalities"] # For val, these are clean (missing_prob=0) usually
        gt_modalities = batch["gt_modalities"]
        
        # But we want to simulate missingness in VALIDATION to verify robustness?
        # With current dataloader, missing_prob=0 for Val.
        # So we should manually mask some modalities here to test synthesis performance?
        # Or just trust the dataloader config (maybe user set val missing_prob > 0)?
        # Let's assume input is what we evaluate on.
        
        cols = modalities.shape[2]
        H, W = modalities.shape[3], modalities.shape[4]
        modalities = modalities.view(-1, cols, H, W)
        gt_modalities = gt_modalities.view(-1, cols, H, W)
        seg_targets = batch["seg"].view(-1, H, W)
        modality_mask = batch["modality_mask"]
        
        outputs = self(modalities, modality_mask)
        seg_logits = outputs["seg_logits"]
        synthesized = outputs["synthesized_modalities"]
        
        loss_dict = self.loss_fn(
            seg_logits, seg_targets,
            synthesized, gt_modalities, modality_mask
        )
        
        self.log("val/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice_loss", loss_dict["dice_loss"], on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate explicit Dice Score (1 - dice_loss)
        dice_score = 1.0 - loss_dict["dice_loss"]
        self.log("val/dice_score", dice_score, on_step=False, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def main():
    config = Config()
    
    # Setup data
    print("Loading Data (Full pipeline)...")
    train_loader, val_loader = get_brats_dataloaders_2d(
        data_root=config.data.data_root,
        batch_size=1, 
        num_workers=2,
        num_slices_per_scan=5,
        missing_prob=0.3 # Masking enabled for training
    )
    
    # Setup model
    print("Initializing Unified Model...")
    system = UnifiedSystem(config)
    
    # Logging
    logger = TensorBoardLogger("logs", name="unified_segmentation")
    
    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val/dice_score",
        mode="max",
        filename="unified-epoch={epoch:02d}-dice={val/dice_score:.2f}",
        save_top_k=3
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    
    print("Starting Training (Segmentation + Synthesis)...")
    trainer.fit(system, train_loader, val_loader)


if __name__ == "__main__":
    main()

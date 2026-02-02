"""
Dataset class for BraTS (Brain Tumor Segmentation) dataset.
Handles loading of multi-modal MRI data with support for missing modalities.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, RandSpatialCropd, RandFlipd, RandRotate90d,
    NormalizeIntensityd, ScaleIntensityRanged, EnsureTyped
)


class BraTSDataset(Dataset):
    """
    BraTS dataset for multi-modal brain tumor MRI.
    
    Supports:
    - Multi-modal inputs: T1, T1ce, T2, FLAIR
    - Segmentation masks (4 classes)
    - Tumor grade labels (LGG vs HGG)
    - Random modality masking to simulate missing data
    """
    
    def __init__(
        self,
        data_root: str,
        modalities: List[str] = ["t1", "t1ce", "t2", "flair"],
        image_size: Tuple[int, int, int] = (128, 128, 128),
        is_training: bool = True,
        missing_prob: float = 0.3,
        transform: Optional[Compose] = None
    ):
        """
        Args:
            data_root: Root directory containing BraTS data
            modalities: List of modality names
            image_size: Target image size (D, H, W)
            is_training: Whether this is training set (enables augmentation)
            missing_prob: Probability of masking each modality during training
            transform: Optional custom transforms
        """
        self.data_root = data_root
        self.modalities = modalities
        self.image_size = image_size
        self.is_training = is_training
        self.missing_prob = missing_prob
        
        # Load dataset index
        self.samples = self._load_samples()
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
    
    def _load_samples(self) -> List[Dict[str, str]]:
        """
        Load dataset samples from directory structure.
        Expected structure:
            data_root/
                PatientID_001/
                    PatientID_001_t1.nii.gz
                    PatientID_001_t1ce.nii.gz
                    PatientID_001_t2.nii.gz
                    PatientID_001_flair.nii.gz
                    PatientID_001_seg.nii.gz
                    grade.txt  # Contains "LGG" or "HGG"
        """
        samples = []
        
        for patient_dir in sorted(os.listdir(self.data_root)):
            patient_path = os.path.join(self.data_root, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            
            sample = {
                "patient_id": patient_dir,
                "modalities": {},
                "seg": os.path.join(patient_path, f"{patient_dir}_seg.nii.gz")
            }
            
            # Load modality paths
            for mod in self.modalities:
                mod_path = os.path.join(patient_path, f"{patient_dir}_{mod}.nii.gz")
                if os.path.exists(mod_path):
                    sample["modalities"][mod] = mod_path
            
            # Load grade label
            grade_file = os.path.join(patient_path, "grade.txt")
            if os.path.exists(grade_file):
                with open(grade_file, 'r') as f:
                    grade = f.read().strip()
                    sample["grade"] = 1 if grade == "HGG" else 0  # HGG=1, LGG=0
            else:
                # If no grade file, try to infer from directory name
                sample["grade"] = 1 if "HGG" in patient_dir else 0
            
            samples.append(sample)
        
        return samples
    
    def _get_default_transforms(self) -> Compose:
        """Get default preprocessing and augmentation transforms."""
        transforms_list = [
            # Normalization
            NormalizeIntensityd(
                keys=self.modalities,
                nonzero=True,
                channel_wise=True
            ),
        ]
        
        if self.is_training:
            # Training augmentations
            transforms_list.extend([
                RandSpatialCropd(
                    keys=self.modalities + ["seg"],
                    roi_size=self.image_size,
                    random_size=False
                ),
                RandFlipd(
                    keys=self.modalities + ["seg"],
                    prob=0.5,
                    spatial_axis=0
                ),
                RandFlipd(
                    keys=self.modalities + ["seg"],
                    prob=0.5,
                    spatial_axis=1
                ),
                RandFlipd(
                    keys=self.modalities + ["seg"],
                    prob=0.5,
                    spatial_axis=2
                ),
                RandRotate90d(
                    keys=self.modalities + ["seg"],
                    prob=0.5,
                    spatial_axes=(0, 1)
                ),
            ])
        
        return Compose(transforms_list)
    
    def _apply_missing_mask(
        self, 
        modality_data: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Randomly mask modalities during training to simulate missing data.
        
        Returns:
            modality_data: Dict with potentially masked (zeroed) modalities
            mask: Binary tensor indicating which modalities are present (1) or missing (0)
        """
        mask = torch.ones(len(self.modalities))
        
        if self.is_training and random.random() < self.missing_prob:
            # Randomly select which modalities to mask (keep at least one)
            num_to_mask = random.randint(0, len(self.modalities) - 1)
            if num_to_mask > 0:
                modalities_to_mask = random.sample(
                    range(len(self.modalities)), 
                    num_to_mask
                )
                
                for idx in modalities_to_mask:
                    mod_name = self.modalities[idx]
                    modality_data[mod_name] = torch.zeros_like(modality_data[mod_name])
                    mask[idx] = 0
        
        return modality_data, mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns dict with keys:
            - modalities: Dict[str, Tensor] - shape (1, D, H, W) per modality
            - modality_mask: Tensor - shape (num_modalities,)
            - seg: Tensor - shape (1, D, H, W)
            - grade: Tensor - scalar (0 or 1)
            - patient_id: str
        """
        sample = self.samples[idx]
        
        # Load modality images
        modality_data = {}
        for mod_name, mod_path in sample["modalities"].items():
            img = nib.load(mod_path)
            data = img.get_fdata().astype(np.float32)
            modality_data[mod_name] = torch.from_numpy(data).unsqueeze(0)  # Add channel dim
        
        # Load segmentation
        seg_img = nib.load(sample["seg"])
        seg_data = seg_img.get_fdata().astype(np.int64)
        seg = torch.from_numpy(seg_data).unsqueeze(0)
        
        # Apply transforms if specified
        if self.transform is not None:
            data_dict = {**modality_data, "seg": seg}
            data_dict = self.transform(data_dict)
            modality_data = {k: v for k, v in data_dict.items() if k != "seg"}
            seg = data_dict["seg"]
        
        # Apply missing modality masking
        modality_data, modality_mask = self._apply_missing_mask(modality_data)
        
        # Stack modalities in fixed order
        modality_tensors = [modality_data[mod] for mod in self.modalities]
        modalities = torch.cat(modality_tensors, dim=0)  # (num_modalities, D, H, W)
        
        return {
            "modalities": modalities,
            "modality_mask": modality_mask,
            "seg": seg.squeeze(0).long(),  # Remove channel dim for seg
            "grade": torch.tensor(sample["grade"], dtype=torch.long),
            "patient_id": sample["patient_id"]
        }


def get_brats_dataloaders(
    data_root: str,
    batch_size: int = 2,
    num_workers: int = 4,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for BraTS.
    
    Args:
        data_root: Root directory (should contain 'train' and 'val' subdirs)
        batch_size: Batch size
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for BraTSDataset
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = BraTSDataset(
        data_root=os.path.join(data_root, "train"),
        is_training=True,
        **kwargs
    )
    
    val_dataset = BraTSDataset(
        data_root=os.path.join(data_root, "val"),
        is_training=False,
        missing_prob=0.0,  # No masking during validation
        **kwargs
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

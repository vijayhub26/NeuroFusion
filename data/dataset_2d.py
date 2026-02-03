"""
2D slice-based BraTS dataset loader.
Extracts 2D slices from 3D volumes for memory-efficient training.
"""

import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import random


class BraTSDataset2D(Dataset):
    """
    BraTS dataset that loads 2D slices from 3D MRI volumes.
    """
    
    def __init__(
        self,
        data_root: str,
        modalities: List[str] = None,
        image_size: Tuple[int, int] = (128, 128),
        is_training: bool = True,
        missing_prob: float = 0.3,
        num_slices_per_scan: int = 5,
        axis: int = 2  # 0=sagittal, 1=coronal, 2=axial
    ):
        """
        Args:
            data_root: Path to BraTS data directory
            modalities: List of modality names
            image_size: Target 2D size (H, W)
            is_training: Training or validation mode
            missing_prob: Probability of masking modalities during training
            num_slices_per_scan: Number of slices to sample per 3D scan
            axis: Which axis to slice (2=axial is most common)
        """
        self.data_root = data_root
        self.modalities = modalities or ["t1", "t1ce", "t2", "flair"]
        self.image_size = image_size
        self.is_training = is_training
        self.missing_prob = missing_prob
        self.num_slices_per_scan = num_slices_per_scan
        self.axis = axis
        
        # Load samples
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} 3D scans from {data_root}")
    
    def _load_samples(self) -> List[Dict]:
        """Load all patient scans."""
        samples = []
        
        for patient_dir in sorted(os.listdir(self.data_root)):
            patient_path = os.path.join(self.data_root, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            
            sample = {
                "patient_id": patient_dir,
                "modalities": {},
                "seg": None,
                "grade": None
            }
            
            # Load modality paths
            for mod in self.modalities:
                for ext in [".nii.gz", ".nii"]:
                    mod_path = os.path.join(patient_path, f"{patient_dir}_{mod}{ext}")
                    if os.path.exists(mod_path):
                        sample["modalities"][mod] = mod_path
                        break
            
            # Load segmentation
            for ext in [".nii.gz", ".nii"]:
                seg_path = os.path.join(patient_path, f"{patient_dir}_seg{ext}")
                if os.path.exists(seg_path):
                    sample["seg"] = seg_path
                    break
            
            # Load grade
            grade_file = os.path.join(patient_path, "grade.txt")
            if os.path.exists(grade_file):
                with open(grade_file, 'r') as f:
                    grade = f.read().strip()
                    sample["grade"] = 1 if grade == "HGG" else 0
            else:
                sample["grade"] = 1 if "HGG" in patient_dir else 0
            
            samples.append(sample)
        
        return samples
    
    def _load_and_preprocess_volume(self, path: str) -> np.ndarray:
        """Load 3D volume and normalize."""
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        
        # Normalize to [0, 1]
        if data.max() > 0:
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        return data
    
    def _extract_2d_slice(self, volume: np.ndarray, slice_idx: int, is_seg: bool = False) -> np.ndarray:
        """Extract 2D slice from 3D volume."""
        if self.axis == 0:
            slice_2d = volume[slice_idx, :, :]
        elif self.axis == 1:
            slice_2d = volume[:, slice_idx, :]
        else:  # axis == 2 (axial)
            slice_2d = volume[:, :, slice_idx]
        
        # Resize if needed
        if slice_2d.shape != self.image_size:
            import torch.nn.functional as F
            slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)
            
            if is_seg:
                # Use nearest neighbor for segmentation masks
                slice_tensor = slice_tensor.float()  # Convert to float for interpolation
                resized = F.interpolate(
                    slice_tensor,
                    size=self.image_size,
                    mode='nearest'
                )
                slice_2d = resized.squeeze().numpy().astype(np.int64)
            else:
                # Use bilinear for modalities
                resized = F.interpolate(
                    slice_tensor,
                    size=self.image_size,
                    mode='bilinear',
                    align_corners=False
                )
                slice_2d = resized.squeeze().numpy()
        
        return slice_2d
    
    def __len__(self) -> int:
        """Return number of 3D scans (not slices)."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a batch of 2D slices from one 3D scan.
        
        Returns a dict with batched slices from the same patient.
        """
        sample = self.samples[idx]
        
        # Load all volumes for this patient
        volumes = {}
        for mod in self.modalities:
            if mod in sample["modalities"]:
                volumes[mod] = self._load_and_preprocess_volume(sample["modalities"][mod])
        
        # Load segmentation
        if sample["seg"]:
            seg_volume = nib.load(sample["seg"]).get_fdata().astype(np.int64)
        else:
            # Create dummy seg if missing
            first_mod = list(volumes.values())[0]
            seg_volume = np.zeros(first_mod.shape, dtype=np.int64)
        
        # Get volume depth
        depth = list(volumes.values())[0].shape[self.axis]
        
        # Sample slices (avoid empty slices at edges)
        margin = int(depth * 0.1)  # Skip first/last 10%
        valid_range = range(margin, depth - margin)
        
        if self.is_training:
            # Random slices during training
            slice_indices = random.sample(list(valid_range), min(self.num_slices_per_scan, len(valid_range)))
        else:
            # Evenly spaced slices during validation
            step = max(1, len(valid_range) // self.num_slices_per_scan)
            slice_indices = list(valid_range)[::step][:self.num_slices_per_scan]
        
        # Extract slices
        modality_slices = []
        seg_slices = []
        
        for slice_idx in slice_indices:
            # Stack modalities for this slice
            mod_slice_stack = []
            for mod in self.modalities:
                if mod in volumes:
                    mod_2d = self._extract_2d_slice(volumes[mod], slice_idx)
                    mod_slice_stack.append(mod_2d)
                else:
                    # Missing modality - zeros
                    mod_slice_stack.append(np.zeros(self.image_size, dtype=np.float32))
            
            modality_slices.append(np.stack(mod_slice_stack, axis=0))  # (4, H, W)
            
            # Extract seg slice
            seg_2d = self._extract_2d_slice(seg_volume, slice_idx, is_seg=True)
            seg_slices.append(seg_2d)
        
        # Stack all slices: (num_slices, 4, H, W)
        modalities_tensor = torch.from_numpy(np.stack(modality_slices, axis=0)).float()
        seg_tensor = torch.from_numpy(np.stack(seg_slices, axis=0)).long()
        
        # Simulate missing modalities
        modality_mask = torch.ones(len(self.modalities))
        if self.is_training and self.missing_prob > 0:
            for i in range(len(self.modalities)):
                if random.random() < self.missing_prob:
                    modality_mask[i] = 0
                    modalities_tensor[:, i] = 0  # Mask out
        
        # Grade is same for all slices
        grade = torch.tensor(sample["grade"], dtype=torch.long)
        
        return {
            "modalities": modalities_tensor,  # (num_slices, 4, H, W)
            "modality_mask": modality_mask,  # (4,)
            "seg": seg_tensor,  # (num_slices, H, W)
            "grade": grade,  # scalar
            "patient_id": sample["patient_id"],
            "num_slices": len(slice_indices)
        }


def get_brats_dataloaders_2d(
    data_root: str = None,
    train_dir: str = None,
    val_dir: str = None,
    batch_size: int = 1,  # Each "batch" is multiple slices from one patient
    num_workers: int = 2,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for 2D slices.
    """
    if train_dir is None:
        train_dir = os.path.join(data_root, "train")
    if val_dir is None:
        val_dir = os.path.join(data_root, "val")
    
    train_dataset = BraTSDataset2D(
        data_root=train_dir,
        is_training=True,
        **kwargs
    )
    
    val_kwargs = {k: v for k, v in kwargs.items() if k != 'missing_prob'}
    val_dataset = BraTSDataset2D(
        data_root=val_dir,
        is_training=False,
        missing_prob=0.0,
        **val_kwargs
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader

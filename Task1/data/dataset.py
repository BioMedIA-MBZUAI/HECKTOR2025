"""HECKTOR dataset implementation."""

import os
import glob
from typing import Dict, List, Optional, Callable
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from monai.transforms import Compose


class HecktorDataset(Dataset):
    """HECKTOR dataset for 3D segmentation."""
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform: Optional[Callable] = None,
        split: str = "train",
        case_ids: Optional[List[str]] = None
    ):
        """
        Initialize HECKTOR dataset.
        
        Args:
            images_dir: Directory containing CT and PET images
            labels_dir: Directory containing segmentation masks
            transform: Data transforms to apply
            split: Dataset split ("train" or "val")
            case_ids: Optional list of specific case IDs to include in the dataset.
                      If None, all available case IDs will be used.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.split = split
        
        # Get all case IDs or use provided ones
        if case_ids is not None:
            self.case_ids = case_ids
        else:
            self.case_ids = self._get_case_ids()
        
    def _get_case_ids(self) -> List[str]:
        """Get all case IDs from the dataset."""
        # Look for CT files to get case IDs
        ct_pattern = os.path.join(self.images_dir, "*_ct.npz")
        ct_files = glob.glob(ct_pattern)
        
        case_ids = []
        for ct_file in ct_files:
            # Extract case ID (remove _CT.nii.gz suffix)
            case_id = os.path.basename(ct_file).replace("_ct.npz", "")
            case_ids.append(case_id)
        
        return sorted(case_ids)
    
    def __len__(self) -> int:
        return len(self.case_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        case_id = self.case_ids[idx]
        
        ct_path = os.path.join(self.images_dir, f"{case_id}_ct.npz")
        pet_path = os.path.join(self.images_dir, f"{case_id}_pt.npz")
        label_path = os.path.join(self.labels_dir, f"{case_id}_label.npz")

        data = {
            "ct": ct_path,
            "pet": pet_path,
            "label": label_path,
            "case_id": case_id
        }

        if self.transform:
            data = self.transform(data)

        return data

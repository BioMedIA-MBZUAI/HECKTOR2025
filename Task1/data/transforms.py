"""Data transforms for HECKTOR dataset, adapted for preprocessed .npz files."""

import numpy as np
import torch
from monai.transforms import (
    Compose,
    MapTransform,
    EnsureChannelFirstd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    EnsureTyped,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    ConcatItemsd,
    SelectItemsd,
)

class LoadNpzDictd(MapTransform):
    """
    Custom transform to load data from .npz files.
    Each .npz file is expected to contain an 'image' array and a 'meta' dictionary.
    This transform correctly loads the data and reconstructs it into a
    MONAI-compatible format with a tensor and its associated metadata dictionary.
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            filepath = d[key]
            # Use allow_pickle=True for loading the metadata dictionary.
            # This is required for security reasons in recent NumPy versions.
            loaded = np.load(filepath, allow_pickle=True)
            
            image_array = loaded['image']
            meta_dict = loaded['meta'].item()  # .item() extracts the dict from the array

            # Reconstruct the data in a MONAI-friendly format.
            # Create a PyTorch tensor from the numpy array.
            d[key] = torch.from_numpy(image_array)
            
            # MONAI transforms expect metadata in a separate dict with the format {key}_meta_dict.
            d[f"{key}_meta_dict"] = meta_dict
        return d


def get_train_transforms(config):
    """
    Get the transforms that are fast and should be run on-the-fly for training.
    This pipeline starts by loading the preprocessed .npz files.
    
    Args:
        config: A configuration object with parameters like `spatial_size`,
                `use_augmentation`, `aug_probability`, and `num_samples`.
    """
    keys = ["ct", "pet", "label"]
    
    # Start with our custom loader.
    transforms = [
        LoadNpzDictd(keys=keys),
        # Note: EnsureChannelFirstd is not needed here because the channel 
        # dimension was saved during the preprocessing step.
    ]

    # --- Training: Random Cropping and Augmentation ---
    
    # 1. Create random patches from the full-resolution preprocessed image.
    # This is very fast as it operates on data already in memory.
    transforms.append(
        RandCropByLabelClassesd(
            keys=keys,
            label_key="label",
            spatial_size=config.spatial_size,  # e.g., (192, 192, 192)
            ratios=[0.1, 0.45, 0.45], # Example ratios, tune as needed
            num_classes=3,
            num_samples=3,
            allow_missing_keys=True,
            warn=False,
        )
    )

    if config.use_augmentation:
        # 2. Apply random augmentations to the small patches.
        aug_transforms = [
            RandFlipd(keys=keys, spatial_axis=[0, 1, 2], prob=config.aug_probability),
            RandScaleIntensityd(keys=["ct"], factors=0.1, prob=config.aug_probability),
            RandShiftIntensityd(keys=["ct"], offsets=0.1, prob=config.aug_probability),
            RandGaussianNoised(keys=["ct"], std=0.01, prob=config.aug_probability),
            RandGaussianSmoothd(
                keys=["ct"],
                sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15),
                prob=config.aug_probability
            ),
        ]
        transforms.extend(aug_transforms)

    # 3. Final steps: combine CT and PET into a single multi-channel image and ensure type.
    final_steps = [
        ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        SelectItemsd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"])
    ]
    transforms.extend(final_steps)
        
    return Compose(transforms)

def get_validation_transforms():
    """
    Get the transforms for validation or testing.
    This pipeline loads the full preprocessed .npz files without applying
    any random cropping or augmentations.
    """
    keys = ["ct", "pet", "label"]

    return Compose([
        # 1. Load the full preprocessed data from .npz files.
        LoadNpzDictd(keys=keys),

        # 2. Combine the CT and PET modalities into a single multi-channel tensor.
        ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        SelectItemsd(keys=["image", "label"]),

        # 3. Ensure the final output keys ('image' and 'label') are tensors.
        EnsureTyped(keys=["image", "label"]),
    ])


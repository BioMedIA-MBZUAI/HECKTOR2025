from monai.data import DataLoader, CacheDataset  # Make sure to import CacheDataset
from .transforms import get_train_transforms, get_validation_transforms
from typing import Tuple, List, Dict
import json
import os

def load_splits(splits_file: str) -> List[Dict[str, List[str]]]:
    """
    Load train/validation splits from a nnUNet-style JSON file.
    (This function remains unchanged)
    """
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    return splits


def get_dataloaders(config, fold: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders using CacheDataset for training.
    """
    splits = load_splits(config.splits_file)
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} is out of range.")
        
    train_ids = splits[fold]['train']
    val_ids = splits[fold]['val']
    print(f"Using fold {fold}: {len(train_ids)} training cases, {len(val_ids)} validation cases")

    # --- Create file lists (this part is unchanged) ---
    train_files = [
        {"ct": os.path.join(config.data_root, config.train_images_dir, f"{case_id}_ct.npz"),
         "pet": os.path.join(config.data_root, config.train_images_dir, f"{case_id}_pet.npz"),
         "label": os.path.join(config.data_root, config.train_labels_dir, f"{case_id}_label.npz")}
        for case_id in train_ids
    ]
    val_files = [
        {"ct": os.path.join(config.data_root, config.train_images_dir, f"{case_id}_ct.npz"),
         "pet": os.path.join(config.data_root, config.train_images_dir, f"{case_id}_pet.npz"),
         "label": os.path.join(config.data_root, config.train_labels_dir, f"{case_id}_label.npz")}
        for case_id in val_ids
    ]

    # --- Training Data Pipeline with Caching ---
    train_transforms = get_train_transforms(config)
    
    # MODIFICATION 1: Use CacheDataset instead of Dataset
    # This will store transformed data in RAM to speed up epochs after the first one.
    print(f"Creating CacheDataset for training with cache_rate={config.cache_rate}...")
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=config.cache_rate,      # Uses the value from your config file
        num_workers=config.num_workers   # Uses workers to build the cache in parallel
    )

    # MODIFICATION 2: Add persistent_workers=True for efficiency
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True  # Recommended for saving overhead between epochs
    )

    # --- Validation Data Pipeline (unchanged) ---
    val_transforms = get_validation_transforms()
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=config.cache_rate) # Cache all of validation
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, persistent_workers=True)
    
    return train_loader, val_loader

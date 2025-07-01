"""SwinUNETR specific configuration."""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class SwinUNETRConfig(BaseConfig):
    """Configuration for SwinUNETR model."""
    
    # Model specific experiment name
    experiment_name: str = "swinunetr"
    
    # SwinUNETR architecture parameters
    img_size: tuple = (128, 128, 128)  # Must match spatial_size
    in_channels: int = 2  # CT + PET (same as input_channels in BaseConfig)
    out_channels: int = 3  # background + primary tumor + metastatic tumor (same as num_classes in BaseConfig)
    feature_size: int = 48
    depths: tuple = (2, 2, 2, 2)
    num_heads: tuple = (3, 6, 12, 24)
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    normalize: bool = True
    use_checkpoint: bool = False
    spatial_dims: int = 3

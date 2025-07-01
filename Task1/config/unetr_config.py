"""UNETR specific configuration."""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class UNETRConfig(BaseConfig):
    """Configuration for UNETR model."""
    
    # Model specific experiment name
    experiment_name: str = "unetr"
    
    # UNETR architecture parameters
    img_size: tuple = (128, 128, 128)  # Must match spatial_size
    feature_size: int = 16
    hidden_size: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12
    dropout_rate: float = 0.0
    norm_name: str = "instance"
    res_block: bool = True
    conv_block: bool = True
    spatial_dims: int = 3

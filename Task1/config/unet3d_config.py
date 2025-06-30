"""UNet3D specific configuration."""

from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class UNet3DConfig(BaseConfig):
    """Configuration for UNet3D model."""
    
    # Model specific experiment name
    experiment_name: str = "unet3d"
    
    # UNet3D architecture parameters
    spatial_dims: int = 3
    channels: tuple = (16,   32, 64, 128, 256)
    strides: tuple = (2, 2, 2, 2)
    kernel_size: int = 3
    up_kernel_size: int = 3
    dropout: float = 0.0
    num_res_units: int = 2
    

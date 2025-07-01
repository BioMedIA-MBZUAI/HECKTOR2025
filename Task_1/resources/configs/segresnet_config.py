"""SegResNet specific configuration."""

from dataclasses import dataclass
from .base_config import BaseConfig
from pathlib import Path

# Define a temporary path for storing intermediate files
TMP_PATH= Path("/tmp")

@dataclass
class SegResNetConfig(BaseConfig):
    """Configuration for SegResNet model."""
    
    # Model specific experiment name
    experiment_name: str = TMP_PATH / "segresnet"
    
    # SegResNet architecture parameters
    blocks_down: tuple = (1, 2, 2, 4)
    blocks_up: tuple = (1, 1, 1)
    init_filters: int = 16
    in_channels: int = 2  # CT + PET (same as input_channels in BaseConfig)
    out_channels: int = 3  # background + primary tumor + metastatic tumor (same as num_classes in BaseConfig)
    dropout_prob: float = 0.2

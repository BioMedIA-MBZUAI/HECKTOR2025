"""SegResNet model implementation using MONAI."""

import torch
import torch.nn as nn
from monai.networks.nets import SegResNet

from .base_model import BaseModel


class SegResNetModel(BaseModel):
    """SegResNet model for segmentation using MONAI."""
    
    def __init__(self, config):
        """
        Initialize SegResNet model.
        
        Args:
            config: SegResNetConfig object
        """
        super().__init__(config)
        
        # Initialize MONAI SegResNet
        self.segresnet = SegResNet(
            blocks_down=config.blocks_down,
            blocks_up=config.blocks_up,
            init_filters=config.init_filters,
            in_channels=config.input_channels,  # Using input_channels from BaseConfig
            out_channels=config.num_classes,    # Using num_classes from BaseConfig
            dropout_prob=config.dropout_prob
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SegResNet.
        
        Args:
            x: Input tensor of shape (B, C, H, W, D)
            
        Returns:
            Output tensor of shape (B, num_classes, H, W, D)
        """
        return self.segresnet(x)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def get_model_info(self) -> str:
        """Get model architecture information."""
        params = self.get_parameters()
        
        info = f"""
SegResNet Model Information:
---------------------------
Architecture: SegResNet (MONAI)
Input Channels: {self.config.input_channels}
Output Channels: {self.config.num_classes}
Init Filters: {self.config.init_filters}
Blocks Down: {self.config.blocks_down}
Blocks Up: {self.config.blocks_up}
Dropout Probability: {self.config.dropout_prob}

Parameters:
-----------
Total Parameters: {params['total_parameters']:,}
Trainable Parameters: {params['trainable_parameters']:,}
Model Size: {params['model_size_mb']:.2f} MB
        """
        
        return info.strip()

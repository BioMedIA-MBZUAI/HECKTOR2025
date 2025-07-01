"""UNet3D model implementation using MONAI."""

import torch
import torch.nn as nn
from monai.networks.nets import UNet

from .base_model import BaseModel


class UNet3DModel(BaseModel):
    """3D U-Net model for segmentation using MONAI."""
    
    def __init__(self, config):
        """
        Initialize UNet3D model.
        
        Args:
            config: UNet3DConfig object
        """
        super().__init__(config)
        
        # Initialize MONAI UNet
        self.unet = UNet(
            spatial_dims=config.spatial_dims,
            in_channels=config.input_channels,
            out_channels=config.num_classes,
            channels=config.channels,
            strides=config.strides,
            kernel_size=config.kernel_size,
            up_kernel_size=config.up_kernel_size,
            dropout=config.dropout,
            num_res_units=config.num_res_units          
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UNet3D.
        
        Args:
            x: Input tensor of shape (B, C, H, W, D)
            
        Returns:
            Output tensor of shape (B, num_classes, H, W, D)
        """
        return self.unet(x)
    
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
UNet3D Model Information:
------------------------
Architecture: 3D U-Net (MONAI)
Input Channels: {self.config.input_channels}
Output Channels: {self.config.num_classes}
Feature Channels: {self.config.channels}
Strides: {self.config.strides}
Kernel Size: {self.config.kernel_size}
Dropout: {self.config.dropout}

Parameters:
-----------
Total Parameters: {params['total_parameters']:,}
Trainable Parameters: {params['trainable_parameters']:,}
Model Size: {params['model_size_mb']:.2f} MB
        """
        
        return info.strip()

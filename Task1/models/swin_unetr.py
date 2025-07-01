"""SwinUNETR model implementation using MONAI."""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from .base_model import BaseModel


class SwinUNETRModel(BaseModel):
    """SwinUNETR model for segmentation using MONAI."""
    
    def __init__(self, config):
        """
        Initialize SwinUNETR model.
        
        Args:
            config: SwinUNETRConfig object
        """
        super().__init__(config)
        
        # Initialize MONAI SwinUNETR
        self.swinunetr = SwinUNETR(
            img_size=config.img_size,
            in_channels=config.input_channels,
            out_channels=config.num_classes,
            feature_size=config.feature_size,
            depths=config.depths,
            num_heads=config.num_heads,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            dropout_path_rate=config.dropout_path_rate,
            normalize=config.normalize,
            use_checkpoint=config.use_checkpoint,
            spatial_dims=config.spatial_dims
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwinUNETR.
        
        Args:
            x: Input tensor of shape (B, C, H, W, D)
            
        Returns:
            Output tensor of shape (B, num_classes, H, W, D)
        """
        return self.swinunetr(x)
    
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
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_model_info(self) -> str:
        """Get model architecture information."""
        params = self.get_parameters()
        
        info = f"""
SwinUNETR Model Information:
---------------------------
Architecture: SwinUNETR (MONAI)
Input Channels: {self.config.input_channels}
Output Channels: {self.config.num_classes}
Image Size: {self.config.img_size}
Feature Size: {self.config.feature_size}
Depths: {self.config.depths}
Number of Heads: {self.config.num_heads}
Drop Rate: {self.config.drop_rate}
Attention Drop Rate: {self.config.attn_drop_rate}
Dropout Path Rate: {self.config.dropout_path_rate}

Parameters:
-----------
Total Parameters: {params['total_parameters']:,}
Trainable Parameters: {params['trainable_parameters']:,}
Model Size: {params['model_size_mb']:.2f} MB
        """
        
        return info.strip()

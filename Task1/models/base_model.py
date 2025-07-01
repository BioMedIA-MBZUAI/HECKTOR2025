"""Base model class for all segmentation models."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """Abstract base class for all segmentation models."""
    
    def __init__(self, config):
        """
        Initialize base model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Dict = None, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "model_config": self.config.__dict__,
            **kwargs
        }
        
        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
            
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, device: str = "cpu"):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

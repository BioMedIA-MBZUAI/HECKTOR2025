"""Loss functions for segmentation."""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss


# Simple loss function definitions
dice_ce_loss = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True, reduction="mean")
dice_focal_loss = DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=True, reduction="mean")


def get_loss_function(loss_type="dice_ce"):
    """
    Get loss function by type.
    
    Args:
        loss_type: Type of loss function ("dice_ce" or "dice_focal")
        
    Returns:
        Loss function instance
    """
    if loss_type == "dice_ce":
        return dice_ce_loss
    elif loss_type == "dice_focal":
        return dice_focal_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose 'dice_ce' or 'dice_focal'")



# For backward compatibility
DiceCELoss = dice_ce_loss
FocalLoss = dice_focal_loss

from resources.configs import SegResNetConfig
from resources.models import SegResNetModel
import torch
import SimpleITK as sitk
import os
import numpy as np
from monai.inferers import sliding_window_inference


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint with proper architecture reconstruction."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return None, None
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model config
        if 'model_config' not in checkpoint:
            print("Error: No model_config found in checkpoint")
            return None, None
        
        model_config = checkpoint['model_config']
        print(f"Found model config for experiment: {model_config.get('experiment_name', 'unknown')}")
        
        # Create config object from saved dictionary
        experiment_name = model_config.get('experiment_name', 'unet3d')
        
        # Recreate the config object
        config = SegResNetConfig()


        
        # Update config with saved values
        for key, value in model_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        print(f"Recreated config for {experiment_name}")
        print(f"  Input channels: {config.input_channels}")
        print(f"  Num classes: {config.num_classes}")
        print(f"  Spatial size: {config.spatial_size}")
        

        model = SegResNetModel(config)
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set eval mode
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully!")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, config
    
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
def arrays_to_tensor(ct_vol, pet_vol):
    """
    Cast CT & PET volumes (of several possible types) into
    a single 5-D torch tensor [1, 2, Z, Y, X].

    Accepts:
      • SimpleITK.Image   (assumed Z-Y-X grid)
      • np.ndarray        (Z, Y, X)  OR (1, Z, Y, X)
      • torch.Tensor / MetaTensor (1, Z, Y, X)
    """

    def to_tensor(vol):
        # --- SimpleITK -> torch -----------------------------------------
        if isinstance(vol, sitk.Image):
            vol = torch.from_numpy(sitk.GetArrayFromImage(vol))  # Z,Y,X
            vol = vol.unsqueeze(0)                               # 1,Z,Y,X
            return vol.float()

        # --- NumPy -> torch ---------------------------------------------
        if isinstance(vol, np.ndarray):
            vol = torch.from_numpy(vol)
            if vol.ndim == 3:           # Z,Y,X
                vol = vol.unsqueeze(0)  # 1,Z,Y,X
            return vol.float()

        # --- torch / MetaTensor -----------------------------------------
        if isinstance(vol, torch.Tensor):
            if vol.ndim == 3:           # Z,Y,X   (rare case)
                vol = vol.unsqueeze(0)  # 1,Z,Y,X
            return vol.float()

        raise TypeError(f"Unsupported type {type(vol)}")

    ct_t  = to_tensor(ct_vol)
    pet_t = to_tensor(pet_vol)

    # Shape check: both must now be (1, Z, Y, X)
    if ct_t.shape[1:] != pet_t.shape[1:]:
        raise ValueError(f"Shape mismatch: CT {ct_t.shape} vs PET {pet_t.shape}")

    combined = torch.cat([ct_t, pet_t], dim=0)   # (2, Z, Y, X)
    return combined.unsqueeze(0)                 # (1, 2, Z, Y, X)




def run_inference(model, input_tensor, config, device='cuda', use_sliding_window=True):
    """Run inference with the loaded model using sliding window."""
    print(f"Input tensor shape: {input_tensor.shape}")
    
    try:
        # Move to device
        input_tensor = input_tensor.to(device)
        
        # Run inference
        if use_sliding_window:
            print("Running sliding window inference...")
            print(f"  ROI size: {config.spatial_size}")
            print(f"  SW batch size: 4")
            print(f"  Overlap: 0.5")
            
            with torch.no_grad():
                output = sliding_window_inference(
                    inputs=input_tensor,
                    roi_size=config.spatial_size,
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                    sigma_scale=0.125,
                    padding_mode="constant",
                    cval=0.0,
                    sw_device=device,
                )
        else:
            print("Running direct inference...")
            with torch.no_grad():
                output = model(input_tensor)
        
        print(f"Output shape: {output.shape}")
        print(f"Output min/max: {output.min().item():.4f} / {output.max().item():.4f}")
        
        # Apply softmax and get prediction
        if output.shape[1] > 1:  # Multi-class
            probs = torch.softmax(output, dim=1)
            prediction = probs.argmax(dim=1)
            print(f"Using softmax + argmax for {output.shape[1]} classes")
        else:  # Binary with sigmoid
            probs = torch.sigmoid(output)
            prediction = (probs > 0.5).long()
            print("Using sigmoid for binary classification")
        
        print(f"Prediction shape: {prediction.shape}")
        unique_vals = torch.unique(prediction).tolist()
        print(f"Unique values in prediction: {unique_vals}")
        
        # Calculate class distribution
        total_voxels = prediction.numel()
        for val in unique_vals:
            count = (prediction == val).sum().item()
            percentage = (count / total_voxels) * 100
            print(f"  Class {val}: {count:,} voxels ({percentage:.2f}%)")
        
        return prediction.cpu().numpy()
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None

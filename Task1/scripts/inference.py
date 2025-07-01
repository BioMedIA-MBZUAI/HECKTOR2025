#!/usr/bin/env python3
"""Simple inference script to test model loading and basic inference."""

import os
import sys
import torch
import numpy as np
import argparse
from monai.inferers import sliding_window_inference

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

# Import model classes and configs
try:
    from models import UNet3DModel, SegResNetModel, UNETRModel, SwinUNETRModel
    from config import UNet3DConfig, SegResNetConfig, UNETRConfig, SwinUNETRConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the Task1 directory")
    sys.exit(1)


def load_npz_data(ct_path, pet_path):
    """Load CT and PET data from NPZ files."""
    print(f"Loading CT data from: {ct_path}")
    ct_npz = np.load(ct_path)
    ct_data = ct_npz['image']  # Use 'image' key instead of 'data'
    
    print(f"Loading PET data from: {pet_path}")
    pet_npz = np.load(pet_path)
    pet_data = pet_npz['image']  # Use 'image' key instead of 'data'
    
    print(f"CT shape: {ct_data.shape}, PET shape: {pet_data.shape}")
    
    # Remove channel dimension if present (1, H, W, D) -> (H, W, D)
    if ct_data.shape[0] == 1:
        ct_data = ct_data.squeeze(0)
    if pet_data.shape[0] == 1:
        pet_data = pet_data.squeeze(0)
    
    print(f"After squeeze - CT shape: {ct_data.shape}, PET shape: {pet_data.shape}")
    
    # Stack as multi-channel input (channels first: [2, D, H, W])
    image = np.stack([ct_data, pet_data], axis=0)
    
    # Convert to tensor and add batch dimension [1, 2, D, H, W]
    tensor = torch.from_numpy(image).float().unsqueeze(0)
    
    return tensor


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
        if experiment_name == 'unet3d':
            config = UNet3DConfig()
        elif experiment_name == 'segresnet':
            config = SegResNetConfig()
        elif experiment_name == 'unetr':
            config = UNETRConfig()
        elif experiment_name == 'swinunetr':
            config = SwinUNETRConfig()
        else:
            print(f"Error: Unknown experiment type: {experiment_name}")
            return None, None
        
        # Update config with saved values
        for key, value in model_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        print(f"Recreated config for {experiment_name}")
        print(f"  Input channels: {config.input_channels}")
        print(f"  Num classes: {config.num_classes}")
        print(f"  Spatial size: {config.spatial_size}")
        
        # Create model with the config
        if experiment_name == 'unet3d':
            model = UNet3DModel(config)
        elif experiment_name == 'segresnet':
            model = SegResNetModel(config)
        elif experiment_name == 'unetr':
            model = UNETRModel(config)
        elif experiment_name == 'swinunetr':
            model = SwinUNETRModel(config)
        
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


def main():
    """Main function to test inference."""
    parser = argparse.ArgumentParser(description='Test model inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--ct_path', type=str, required=True,
                       help='Path to CT NPZ file')
    parser.add_argument('--pet_path', type=str, required=True,
                       help='Path to PET NPZ file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Optional path to save prediction as NPZ file')
    parser.add_argument('--no_sliding_window', action='store_true',
                       help='Disable sliding window inference (use direct inference)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Load model from checkpoint
    model, config = load_model_from_checkpoint(args.model_path, args.device)
    if model is None or config is None:
        print("Failed to load model or config")
        return
    
    # Load test data
    try:
        input_tensor = load_npz_data(args.ct_path, args.pet_path)
        print(f"✓ Data loaded successfully: {input_tensor.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Run inference
    use_sliding_window = not hasattr(args, 'no_sliding_window') or not args.no_sliding_window
    prediction = run_inference(model, input_tensor, config, args.device, use_sliding_window)
    
    if prediction is not None:
        print(f"\n✓ Inference completed successfully!")
        print(f"Final prediction shape: {prediction.shape}")
        
        # Save prediction if requested
        if args.output_path:
            try:
                np.savez_compressed(args.output_path, prediction=prediction)
                print(f"✓ Prediction saved to: {args.output_path}")
            except Exception as e:
                print(f"Error saving prediction: {e}")
    else:
        print(f"\n✗ Inference failed")


if __name__ == "__main__":
    main()

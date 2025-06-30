"""Visualization utilities."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import pandas as pd


def plot_training_curves(metrics_file: str, save_path: Optional[str] = None):
    """
    Plot training curves from metrics CSV.
    
    Args:
        metrics_file: Path to metrics CSV file
        save_path: Path to save the plot
    """
    # Load metrics
    df = pd.read_csv(metrics_file)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', color='blue')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation', color='red')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot Dice score
    axes[0, 1].plot(df['epoch'], df['train_dice'], label='Train', color='blue')
    axes[0, 1].plot(df['epoch'], df['val_dice'], label='Validation', color='red')
    axes[0, 1].set_title('Dice Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot IoU
    axes[1, 0].plot(df['epoch'], df['train_iou'], label='Train', color='blue')
    axes[1, 0].plot(df['epoch'], df['val_iou'], label='Validation', color='red')
    axes[1, 0].set_title('IoU Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot learning rate
    axes[1, 1].plot(df['epoch'], df['lr'], color='green')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def visualize_segmentation(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    slice_idx: int = None,
    save_path: Optional[str] = None
):
    """
    Visualize segmentation results.
    
    Args:
        image: Input image (H, W, D) or (C, H, W, D)
        prediction: Predicted mask (H, W, D)
        ground_truth: Ground truth mask (H, W, D)
        slice_idx: Slice index to visualize (middle slice if None)
        save_path: Path to save the plot
    """
    # Handle multi-channel images (take first channel, e.g., CT)
    if image.ndim == 4:
        image = image[0]
    
    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = image.shape[2] // 2
    
    # Extract slices
    img_slice = image[:, :, slice_idx]
    pred_slice = prediction[:, :, slice_idx]
    gt_slice = ground_truth[:, :, slice_idx]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(img_slice, cmap='gray')
    axes[0, 0].set_title('Input Image (CT)')
    axes[0, 0].axis('off')
    
    # Ground truth
    axes[0, 1].imshow(img_slice, cmap='gray')
    axes[0, 1].imshow(gt_slice, cmap='Reds', alpha=0.5)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Prediction
    axes[1, 0].imshow(img_slice, cmap='gray')
    axes[1, 0].imshow(pred_slice, cmap='Blues', alpha=0.5)
    axes[1, 0].set_title('Prediction')
    axes[1, 0].axis('off')
    
    # Overlay comparison
    axes[1, 1].imshow(img_slice, cmap='gray')
    axes[1, 1].imshow(gt_slice, cmap='Reds', alpha=0.3, label='Ground Truth')
    axes[1, 1].imshow(pred_slice, cmap='Blues', alpha=0.3, label='Prediction')
    axes[1, 1].set_title('Overlay Comparison')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_metrics_comparison(experiment_dirs: List[str], metric_name: str = 'val_dice'):
    """
    Compare metrics across multiple experiments.
    
    Args:
        experiment_dirs: List of experiment directories
        metric_name: Metric to compare
    """
    plt.figure(figsize=(12, 6))
    
    for exp_dir in experiment_dirs:
        metrics_file = os.path.join(exp_dir, 'logs', 'metrics.csv')
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            exp_name = os.path.basename(exp_dir)
            plt.plot(df['epoch'], df[metric_name], label=exp_name)
    
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.legend()
    plt.grid(True)
    plt.show()


def create_confusion_matrix(predictions: np.ndarray, targets: np.ndarray):
    """
    Create confusion matrix visualization.
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
    """
    from sklearn.metrics import confusion_matrix
    
    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Compute confusion matrix
    cm = confusion_matrix(target_flat, pred_flat)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Background', 'Tumor'],
                yticklabels=['Background', 'Tumor'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# HECKTOR 2025 Challenge - Task 1 Segmentation Baselines

A simple and modular framework for training baseline segmentation models for Task 1 of the HECKTOR 2025 Challenge - automatic detection and segmentation of Head and Neck (H&N) primary tumors and lymph nodes.

## Overview

This project implements baseline segmentation models for the HECKTOR 2025 Challenge Task 1, which focuses on the automatic detection and segmentation of Head and Neck (H&N) primary tumors and lymph nodes in FDG-PET/CT images. The framework is designed to be simple, easy to understand, and easy to extend for the HECKTOR 2025 Challenge.

## Features

- **Simple Architecture**: Clean, modular code structure
- **MONAI Integration**: Uses MONAI library for medical image processing
- **Dual Modality**: Supports CT + PET input
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Flexible Configuration**: Easy-to-modify configuration system
- **Ready-to-Use Scripts**: Training, evaluation, and inference scripts

## Installation

1. Clone this repository and navigate to the Task1 directory:
```bash
git clone https://github.com/BioMedIA-MBZUAI/HECKTOR2025.git
cd HECKTOR2025/Task1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Setup

Ensure your HECKTOR 2025 Task 1 data is organized as follows:
```
/path/to/hecktor2025_task1_dataset/
├── imagesTr_cropped/
│   ├── CASE001_0000.nii.gz  # CT images
│   ├── CASE001_0001.nii.gz  # PET images
│   └── ...
└── labelsTr_cropped/
    ├── CASE001.nii.gz       # Tumor and lymph node segmentation masks
    └── ...
```

## Quick Start

### 1. Training

Train a segmentation model for HECKTOR 2025 Task 1 (available models: unet3d, segresnet, unetr, swinunetr):
```bash
python scripts/train.py --config unet3d
```

Training options:
- `--config`: Model configuration (unet3d, segresnet, unetr, swinunetr)
- `--resume`: Path to checkpoint to resume from
- `--device`: Device to use (default: cuda)

### 2. Inference

Run inference on HECKTOR 2025 Task 1 data for a single case:
```bash
python scripts/inference.py \
    --model_path experiments/unet3d/checkpoints/best_model.pth \
    --ct_path /path/to/ct.nii.gz \
    --pet_path /path/to/pet.nii.gz \
    --output_path /path/to/output
```

### 3. Evaluation

For evaluation with ground truth labels, use the `InferenceEvaluator` class programmatically:

```python
from evaluation.inference_evaluator import InferenceEvaluator
from config import UNet3DConfig

# Load config and evaluator
config = UNet3DConfig()
evaluator = InferenceEvaluator(
    model="experiments/unet3d/checkpoints/best_model.pth",
    config=config
)

# Evaluate dataset
results = evaluator.evaluate_dataset(
    data_dir="/path/to/test/data",
    output_dir="evaluation_results"
)
```

The evaluation includes:
- **Dice Score**: Primary segmentation metric
- **Hausdorff Distance**: Surface distance metric
- **IoU**: Intersection over Union
- **Visualization**: Worst case analysis and result plots

## Configuration

Model configurations are defined in the `config/` directory. Available models and configurations:

- **UNet3D**: 3D U-Net with skip connections
- **SegResNet**: Segmentation ResNet architecture
- **UNETR**: Transformer-based U-Net
- **SwinUNETR**: Swin Transformer U-Net

The main parameters include:

- **Data paths**: Location of training data
- **Model architecture**: Network parameters
- **Training settings**: Learning rate, batch size, epochs
- **Augmentation**: Data augmentation parameters

Example configuration (UNet3D):
```python
@dataclass
class UNet3DConfig(BaseConfig):
    # Model
    experiment_name: str = "unet3d"
    channels: tuple = (32, 64, 128, 256, 512)
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 2
    num_epochs: int = 200
    
    # Loss
    dice_weight: float = 1.0
    ce_weight: float = 1.0
```

## Model Architecture

### Available Models for HECKTOR 2025 Task 1

**UNet3D (MONAI)**
- **Input**: 3D dual-modality images (CT + PET) for HECKTOR 2025 Task 1
- **Output**: Multi-class segmentation (background + primary tumor + lymph nodes)
- **Architecture**: 3D U-Net with skip connections optimized for head and neck anatomy
- **Features**: 5 encoder/decoder levels with residual connections

**SegResNet**
- **Architecture**: Segmentation ResNet with residual blocks
- **Features**: Deep residual learning for medical image segmentation

**UNETR**
- **Architecture**: Transformer-based U-Net for 3D medical image segmentation
- **Features**: Vision Transformer encoder with CNN decoder

**SwinUNETR**
- **Architecture**: Swin Transformer-based U-Net
- **Features**: Hierarchical vision transformer with U-Net decoder

## Evaluation Metrics

The framework computes the following metrics for HECKTOR 2025 Task 1 evaluation:

- **Dice Score**: Primary segmentation metric for tumor and lymph node regions
- **IoU**: Intersection over Union for overlap assessment
- **Sensitivity/Specificity**: Clinical metrics for detection performance

## Output Structure

After training, the following structure is created:

```
experiments/unet3d/
├── checkpoints/
│   ├── best_model.pth
│   ├── latest_checkpoint.pth
│   └── checkpoint_epoch_*.pth
└── logs/
    ├── metrics.csv
    ├── training_*.log
    └── tensorboard_logs/
```

## Monitoring Training

### TensorBoard
View training progress in real-time:
```bash
tensorboard --logdir experiments/unet3d/logs
```

### Logs
Training logs are saved in `experiments/{model}/logs/training_*.log`

## Data Preprocessing

The framework automatically handles:
- **Intensity normalization**: CT values scaled to [0, 1]
- **Spatial resampling**: Images resized to target size
- **Data augmentation**: Random flips, rotations, scaling
- **Multi-modal stacking**: CT and PET combined as channels

## Extending the Framework

### Adding New Models

1. Create model class in `models/`:
```python
class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model
    
    def forward(self, x):
        # Implement forward pass
        return output
```

2. Create configuration in `config/`:
```python
@dataclass
class MyModelConfig(BaseConfig):
    experiment_name: str = "my_model"
    # Add model-specific parameters
```

3. Update imports and use in training scripts

### Adding New Loss Functions

Add loss functions to `training/losses.py`:
```python
class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        # Implement loss calculation
        return loss
```

### Adding New Metrics

Add metrics to `training/metrics.py`:
```python
class MyMetric:
    def __call__(self, predictions, targets):
        # Implement metric calculation
        return metric_value
```

## Results

Expected performance on HECKTOR 2025 Task 1 dataset:
- **Dice Score**: ~0.70-0.80 (depending on data quality and tumor complexity)
- **Training time**: ~2-4 hours on modern GPU
- **Memory usage**: ~8-12GB GPU memory

## License

This project is for research purposes related to the HECKTOR 2025 Challenge. Please cite the HECKTOR dataset and challenge if you use this code.

## References

- HECKTOR 2025 Challenge: https://hecktor25.grand-challenge.org/
- MONAI Framework: https://monai.io/
- PyTorch: https://pytorch.org/

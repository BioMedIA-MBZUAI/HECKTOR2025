"""Logging utilities."""

import os
import logging
from datetime import datetime


def setup_logging(log_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Setup logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Setup logger
    logger = logging.getLogger("hecktor_segmentation")
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


class TrainingLogger:
    """Training progress logger."""
    
    def __init__(self, log_dir: str):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        self.logger = setup_logging(log_dir)
        self.metrics_file = os.path.join(log_dir, "metrics.csv")
        
        # Initialize metrics file
        with open(self.metrics_file, "w") as f:
            f.write("epoch,train_loss,train_dice,train_iou,val_loss,val_dice,val_iou,lr\n")
    
    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, lr: float):
        """
        Log epoch metrics.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            lr: Current learning rate
        """
        # Log to console/file
        self.logger.info(f"Epoch {epoch}")
        self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                        f"Dice: {train_metrics['dice']:.4f}, "
                        f"IoU: {train_metrics['iou']:.4f}")
        self.logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                        f"Dice: {val_metrics['dice']:.4f}, "
                        f"IoU: {val_metrics['iou']:.4f}")
        self.logger.info(f"Learning Rate: {lr:.6f}")
        
        # Log to CSV
        with open(self.metrics_file, "a") as f:
            f.write(f"{epoch},{train_metrics['loss']:.6f},{train_metrics['dice']:.6f},"
                   f"{train_metrics['iou']:.6f},{val_metrics['loss']:.6f},"
                   f"{val_metrics['dice']:.6f},{val_metrics['iou']:.6f},{lr:.8f}\n")
    
    def log_info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message."""
        self.logger.error(message)

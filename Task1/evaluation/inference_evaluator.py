"""Unified inference and evaluation engine for medical image segmentation."""

import os
import glob
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from ..data.transforms import get_transforms
from ..utils.io import load_nifti, save_nifti
from ..utils.visualization import visualize_segmentation
from ..training.metrics import DiceScore, HausdorffDistance, IoU


class InferenceEvaluator:
    """Unified class for model inference and evaluation."""
    
    def __init__(self, model, config, device: str = "cuda"):
        """
        Initialize inference and evaluation engine.
        
        Args:
            model: Trained model or path to model weights
            config: Configuration object
            device: Device to use for inference/evaluation
        """
        # Load model if path is provided
        if isinstance(model, str):
            self.model = torch.load(model, map_location=device)
        else:
            self.model = model
            
        self.config = config
        self.device = device
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
        
        # Setup transforms
        self.transforms = get_transforms(config, mode="val")
        
        # Initialize metrics for evaluation
        self.dice_metric = DiceScore()
        self.hd_metric = HausdorffDistance()
        self.iou_metric = IoU()
    
    def preprocess(self, ct_path: str, pet_path: str) -> Tuple[torch.Tensor, dict]:
        """
        Preprocess input images.
        
        Args:
            ct_path: Path to CT image
            pet_path: Path to PET image
            
        Returns:
            Preprocessed tensor and metadata
        """
        # Load images
        ct_data, ct_header = load_nifti(ct_path)
        pet_data, pet_header = load_nifti(pet_path)
        
        # Stack as multi-channel input
        image = np.stack([ct_data, pet_data], axis=0)
        
        # Create data dictionary
        data = {
            "image": image.astype(np.float32),
            "label": np.zeros_like(ct_data).astype(np.float32)  # Dummy label for inference
        }
        
        # Apply transforms
        transformed = self.transforms(data)
        
        # Get tensor and add batch dimension
        tensor = transformed["image"].unsqueeze(0)
        
        # Store metadata
        metadata = {
            "original_shape": ct_data.shape,
            "ct_header": ct_header,
            "pet_header": pet_header
        }
        
        return tensor, metadata
    
    def postprocess(self, prediction: torch.Tensor, metadata: dict) -> np.ndarray:
        """
        Postprocess model output.
        
        Args:
            prediction: Model prediction tensor
            metadata: Preprocessing metadata
            
        Returns:
            Postprocessed segmentation mask
        """
        # Apply softmax and get argmax
        pred_probs = torch.softmax(prediction, dim=1)
        pred_mask = pred_probs.argmax(dim=1).squeeze(0).cpu().numpy()
        
        return pred_mask.astype(np.uint8)
    
    def predict_case(
        self,
        ct_path: str,
        pet_path: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict segmentation for a single case.
        
        Args:
            ct_path: Path to CT image
            pet_path: Path to PET image
            output_path: Path to save prediction
            
        Returns:
            Segmentation mask
        """
        # Preprocess
        input_tensor, metadata = self.preprocess(ct_path, pet_path)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        prediction = self.postprocess(output, metadata)
        
        # Save if output path provided
        if output_path:
            save_nifti(
                prediction,
                output_path,
                header=metadata.get("ct_header")
            )
        
        return prediction
    
    def predict_directory(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*_0000.nii.gz"
    ) -> List[str]:
        """
        Predict segmentations for all cases in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save predictions
            pattern: Pattern to match CT files
            
        Returns:
            List of processed case IDs
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_cases = []
        
        # Find all CT files
        ct_files = glob.glob(os.path.join(input_dir, pattern))
        
        for ct_file in tqdm(ct_files, desc="Processing cases"):
            # Get case ID and construct PET file path
            case_id = os.path.basename(ct_file).replace("_0000.nii.gz", "")
            pet_file = os.path.join(input_dir, f"{case_id}_0001.nii.gz")
            
            if not os.path.exists(pet_file):
                print(f"Warning: PET file not found for case {case_id}")
                continue
            
            # Predict
            output_path = os.path.join(output_dir, f"{case_id}_prediction.nii.gz")
            
            try:
                self.predict_case(ct_file, pet_file, output_path)
                processed_cases.append(case_id)
                print(f"Processed case: {case_id}")
            except Exception as e:
                print(f"Error processing case {case_id}: {e}")
        
        return processed_cases
    
    def evaluate_case(
        self,
        ct_path: str,
        pet_path: str,
        label_path: str
    ) -> Dict[str, float]:
        """
        Evaluate a single case with ground truth.
        
        Args:
            ct_path: Path to CT image
            pet_path: Path to PET image
            label_path: Path to ground truth label
            
        Returns:
            Dictionary of metrics
        """
        # Load ground truth
        label_data, _ = load_nifti(label_path)
        
        # Get prediction
        prediction = self.predict_case(ct_path, pet_path)
        
        # Convert to tensors for metric calculation
        pred_tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label_data).unsqueeze(0).float()
        
        # Calculate metrics
        dice_score = self.dice_metric(pred_tensor, label_tensor.unsqueeze(1))
        hd_score = self.hd_metric(pred_tensor, label_tensor.unsqueeze(1))
        iou_score = self.iou_metric(pred_tensor, label_tensor)
        
        return {
            "dice": dice_score.item() if torch.is_tensor(dice_score) else dice_score,
            "hausdorff": hd_score.item() if torch.is_tensor(hd_score) else hd_score,
            "iou": iou_score.item() if torch.is_tensor(iou_score) else iou_score
        }
    
    def evaluate_dataset(
        self,
        data_dir: str,
        labels_dir: str,
        output_dir: Optional[str] = None,
        save_predictions: bool = False
    ) -> Dict[str, any]:
        """
        Evaluate entire dataset with ground truth labels.
        
        Args:
            data_dir: Directory containing CT/PET images
            labels_dir: Directory containing ground truth labels
            output_dir: Directory to save predictions and results
            save_predictions: Whether to save prediction files
            
        Returns:
            Evaluation results
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        all_metrics = []
        case_metrics = {}
        
        # Find all CT files
        ct_files = glob.glob(os.path.join(data_dir, "*_0000.nii.gz"))
        
        for ct_file in tqdm(ct_files, desc="Evaluating dataset"):
            # Get case paths
            case_id = os.path.basename(ct_file).replace("_0000.nii.gz", "")
            pet_file = os.path.join(data_dir, f"{case_id}_0001.nii.gz")
            label_file = os.path.join(labels_dir, f"{case_id}.nii.gz")
            
            # Check if all files exist
            if not all(os.path.exists(f) for f in [pet_file, label_file]):
                print(f"Warning: Missing files for case {case_id}")
                continue
            
            try:
                # Evaluate case
                metrics = self.evaluate_case(ct_file, pet_file, label_file)
                case_metrics[case_id] = metrics
                all_metrics.append(metrics)
                
                # Save prediction if requested
                if save_predictions and output_dir:
                    pred_path = os.path.join(output_dir, f"{case_id}_prediction.nii.gz")
                    self.predict_case(ct_file, pet_file, pred_path)
                
            except Exception as e:
                print(f"Error evaluating case {case_id}: {e}")
        
        # Calculate aggregate metrics
        if all_metrics:
            aggregate_metrics = {
                "mean_dice": np.mean([m["dice"] for m in all_metrics]),
                "std_dice": np.std([m["dice"] for m in all_metrics]),
                "mean_hausdorff": np.mean([m["hausdorff"] for m in all_metrics]),
                "std_hausdorff": np.std([m["hausdorff"] for m in all_metrics]),
                "mean_iou": np.mean([m["iou"] for m in all_metrics]),
                "std_iou": np.std([m["iou"] for m in all_metrics]),
            }
        else:
            aggregate_metrics = {}
        
        results = {
            "aggregate_metrics": aggregate_metrics,
            "case_metrics": case_metrics,
            "num_cases": len(all_metrics)
        }
        
        # Save results if output directory provided
        if output_dir:
            self.save_results(results, output_dir)
        
        return results
    
    def save_results(self, results: Dict, output_dir: str):
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save results
        """
        # Generate and save report
        report = self.generate_report(results)
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        # Save metrics to CSV
        if results["case_metrics"]:
            csv_path = os.path.join(output_dir, "case_metrics.csv")
            self.save_metrics_csv(results, csv_path)
    
    def generate_report(self, results: Dict) -> str:
        """Generate evaluation report."""
        if not results["aggregate_metrics"]:
            return "No valid cases found for evaluation."
        
        agg = results["aggregate_metrics"]
        
        report = f"""
HECKTOR Segmentation Evaluation Report
====================================

Dataset Summary:
- Total cases evaluated: {results["num_cases"]}

Performance Metrics:
-------------------
Dice Score:
  - Mean: {agg["mean_dice"]:.4f} ± {agg["std_dice"]:.4f}
  
Hausdorff Distance:
  - Mean: {agg["mean_hausdorff"]:.4f} ± {agg["std_hausdorff"]:.4f}
  
Intersection over Union (IoU):
  - Mean: {agg["mean_iou"]:.4f} ± {agg["std_iou"]:.4f}

Case-wise Results:
-----------------
"""
        
        # Add case-wise results
        for case_id, metrics in results["case_metrics"].items():
            report += f"{case_id}: Dice={metrics['dice']:.4f}, HD={metrics['hausdorff']:.4f}, IoU={metrics['iou']:.4f}\n"
        
        return report
    
    def save_metrics_csv(self, results: Dict, save_path: str):
        """Save case-wise metrics to CSV."""
        data = []
        for case_id, metrics in results["case_metrics"].items():
            data.append({
                "case_id": case_id,
                **metrics
            })
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
    
    def run_inference_only(
        self,
        data_dir: str,
        output_dir: str,
        model_weights: Optional[str] = None
    ):
        """
        Run inference only on a dataset.
        
        Args:
            data_dir: Directory containing CT/PET images
            output_dir: Directory to save predictions
            model_weights: Optional path to model weights (if not loaded in init)
        """
        if model_weights:
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device))
        
        processed_cases = self.predict_directory(data_dir, output_dir)
        print(f"Inference completed. Processed {len(processed_cases)} cases.")
        return processed_cases
    
    def run_full_evaluation(
        self,
        data_dir: str,
        labels_dir: str,
        output_dir: str,
        model_weights: Optional[str] = None
    ):
        """
        Run complete evaluation with metrics and reports.
        
        Args:
            data_dir: Directory containing CT/PET images
            labels_dir: Directory containing ground truth labels
            output_dir: Directory to save results
            model_weights: Optional path to model weights
        """
        if model_weights:
            self.model.load_state_dict(torch.load(model_weights, map_location=self.device))
        
        results = self.evaluate_dataset(
            data_dir, 
            labels_dir, 
            output_dir, 
            save_predictions=True
        )
        
        print("Evaluation completed!")
        print(f"Processed {results['num_cases']} cases")
        if results['aggregate_metrics']:
            print(f"Mean Dice Score: {results['aggregate_metrics']['mean_dice']:.4f}")
        
        return results

"""Evaluation metrics for segmentation."""

import SimpleITK as sitk
import numpy as np
import torch
from typing import List, Dict, Any, Union


def tensor_to_sitk_image(tensor, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Convert a PyTorch tensor to a SimpleITK Image.
    
    Args:
        tensor: PyTorch tensor of shape [C, D, H, W] or [D, H, W]
        spacing: Image spacing (default: 1mm isotropic)
        origin: Image origin (default: 0,0,0)
        
    Returns:
        SimpleITK Image
    """
    # Ensure tensor is on CPU and convert to numpy
    if tensor.dim() == 4:  # [C, D, H, W]
        tensor = tensor[0]  # Take first channel
    
    # Convert to numpy array
    array = tensor.detach().cpu().numpy().astype(np.uint8)
    
    # Create SimpleITK image
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    
    return image


class DiceAggScore:
    """
    A stateful metric class to compute the Aggregated Dice Similarity Coefficient (DSC).

    This class accumulates statistics from multiple prediction/target mask pairs
    and computes the final aggregated DSC, which is more robust than averaging
    individual scores, especially when dealing with highly variable region sizes.

    Attributes:
        _intermediate_results (List[Dict[str, float]]): A list to store the TP
            and volume sums for each sample processed.
        class_labels (List[int]): The list of class labels to evaluate (e.g., [1, 2]).

    Example:
        >>> metric = DiceAggScore(class_labels=[1, 2])
        >>> for pred_mask, target_mask in dataset:
        ...     metric.update(pred_mask, target_mask)
        >>> final_scores = metric.compute()
        >>> print(final_scores)
    """

    def __init__(self, class_labels: List[int] = [1, 2]):
        """
        Initializes the DiceAggScore metric.

        Args:
            class_labels (List[int]): A list of integer labels for the classes
                                      to be evaluated. Background (0) is ignored.
        """
        self.class_labels = class_labels
        self._intermediate_results: List[Dict[str, float]] = []
        # Validate that we have SimpleITK
        if not hasattr(sitk, 'LabelOverlapMeasuresImageFilter'):
            raise ImportError("SimpleITK is required for this class. Please install it.")
    
    def _compute_volumes(self, image: sitk.Image) -> Dict[str, float]:
        """Calculates the volume of each specified label in the image."""
        volumes = {}
        spacing = image.GetSpacing()
        voxvol = spacing[0] * spacing[1] * spacing[2]
        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(image, image)
        
        for label in self.class_labels:
            key = f"vol{label}"
            try:
                nvoxels = stats.GetCount(label)
                volumes[key] = nvoxels * voxvol
            except RuntimeError: # Label does not exist in the image
                volumes[key] = 0.0
        return volumes

    def update(self, pred_mask: sitk.Image, target_mask: sitk.Image):
        """
        Updates the internal state with a new pair of prediction and target masks.

        This method calculates the intermediate metrics (True Positives and volume sums)
        for the given pair and appends them to the results list.

        Args:
            pred_mask (sitk.Image): The prediction segmentation mask as a SimpleITK Image.
            target_mask (sitk.Image): The ground truth segmentation mask as a SimpleITK Image.
        """
        # Ensure masks are of integer type
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkUInt8)
        pred_mask = caster.Execute(pred_mask)
        target_mask = caster.Execute(target_mask)

        # Check for consistent physical properties (resample if necessary)
        if pred_mask.GetSize() != target_mask.GetSize() or \
           np.any(np.array(pred_mask.GetSpacing()) != np.array(target_mask.GetSpacing())):
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(target_mask)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            pred_mask = resampler.Execute(pred_mask)

        overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures.Execute(target_mask, pred_mask)

        vol_gt = self._compute_volumes(target_mask)
        vol_pred = self._compute_volumes(pred_mask)
        
        patient_metrics = {}
        for label in self.class_labels:
            # Get Dice for the current label
            try:
                dsc = overlap_measures.GetDiceCoefficient(label)
            except RuntimeError: # Label might not be in both images
                dsc = 0.0

            vol_sum = vol_gt.get(f"vol{label}", 0) + vol_pred.get(f"vol{label}", 0)
            
            # Calculate True Positives: TP = DSC * (vol_gt + vol_pred) / 2
            tp = dsc * vol_sum / 2
            
            patient_metrics[f"TP{label}"] = tp
            patient_metrics[f"vol_sum{label}"] = vol_sum
        
        self._intermediate_results.append(patient_metrics)
        
    def update_from_tensors(self, pred_tensor, target_tensor, spacing=(1.0, 1.0, 1.0)):
        """
        Updates the metric using PyTorch tensors.
        
        Args:
            pred_tensor: Prediction tensor [B, C, D, H, W]
            target_tensor: Target tensor [B, D, H, W]
            spacing: Image spacing
        """
        batch_size = pred_tensor.shape[0]
        
        for b in range(batch_size):
            # Get single sample
            pred = pred_tensor[b]
            target = target_tensor[b]
            
            # Convert predictions to binary mask (class 1)
            if pred.dim() > 3:  # [C, D, H, W]
                pred = torch.softmax(pred, dim=0)
                pred = torch.argmax(pred, dim=0)
            
            # Convert to SimpleITK images
            pred_sitk = tensor_to_sitk_image(pred, spacing)
            target_sitk = tensor_to_sitk_image(target, spacing)
            
            # Update metrics
            self.update(pred_sitk, target_sitk)

    def compute(self) -> Dict[str, float]:
        """
        Computes the final aggregated Dice score from all updated samples.

        Returns:
            Dict[str, float]: A dictionary containing the aggregated DSC for each class
                              and the mean aggregated DSC across all classes.
        """
        if not self._intermediate_results:
            return {'mean': 0.0}

        aggregate_scores = {}
        all_dsc_scores = []

        for label in self.class_labels:
            tp_key = f"TP{label}"
            vol_sum_key = f"vol_sum{label}"
            
            total_tp = np.sum([res.get(tp_key, 0) for res in self._intermediate_results])
            total_vol_sum = np.sum([res.get(vol_sum_key, 0) for res in self._intermediate_results])

            if total_vol_sum == 0:
                # Handle case where class is absent in all ground truth and predictions
                dsc_agg = 1.0 if total_tp == 0 else 0.0
            else:
                dsc_agg = 2 * total_tp / total_vol_sum
            
            aggregate_scores[f'Class_{label}'] = dsc_agg
            all_dsc_scores.append(dsc_agg)

        aggregate_scores['mean'] = np.mean(all_dsc_scores) if all_dsc_scores else 0.0
        return aggregate_scores

    def reset(self):
        """Reset the metric state by clearing intermediate results."""
        self._intermediate_results = []

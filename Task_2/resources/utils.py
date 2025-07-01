"""
HECKTOR Survival Model Utilities for Container Deployment
"""

import os
import numpy as np
import pandas as pd
import torch
import pickle
import warnings
import SimpleITK as sitk
from pathlib import Path
from monai.transforms import (
    Compose, EnsureChannelFirst, ScaleIntensity, ToTensor, Resize
)
import torch.nn as nn
from monai.networks.nets import resnet18
from monai.data import MetaTensor
from monai.utils.misc import ImageMetaKey
from skimage.measure import label
import math


class FusedFeatureExtractor(nn.Module):
    """
    Feature extractor specifically designed for BaggedIcareSurvival.
    Combines 3D medical imaging and clinical data into rich survival features.
    """
    def __init__(self, clinical_feature_dim, feature_output_dim=128):
        super().__init__()
        
        # Store dimensions for saving
        self.clinical_feature_dim = clinical_feature_dim
        self.feature_output_dim = feature_output_dim
        
        # 3D ResNet-18 for combined CT+PET input
        self.imaging_backbone = resnet18(
            spatial_dims=3,
            n_input_channels=2,
            num_classes=1,
        )
        self.imaging_backbone.fc = nn.Identity()

        # Clinical data processor with deeper architecture
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Feature fusion with multiple pathways
        self.feature_fusion = nn.Sequential(
            nn.Linear(512 + 32, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, feature_output_dim)
        )
        
        # Risk prediction head for training guidance
        self.risk_head = nn.Sequential(
            nn.Linear(feature_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, medical_images, clinical_features, return_risk=False):
        # Extract imaging features
        imaging_features = self.imaging_backbone(medical_images)
        
        # Process clinical features
        clinical_features_processed = self.clinical_processor(clinical_features)
        
        # Combine and fuse
        combined_features = torch.cat([imaging_features, clinical_features_processed], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        if return_risk:
            risk_scores = self.risk_head(fused_features).squeeze(-1)
            return fused_features, risk_scores
        
        return fused_features


class HecktorInferenceModel:
    """
    HECKTOR survival prediction model for container inference.
    Processes single patients and returns RFS risk predictions.
    """
    
    def __init__(self, resource_path, resampling=(1.0, 1.0, 1.0), crop_box_size=[200, 200, 310]):
        """
        Initialize the inference model.
        
        Args:
            resource_path: Path to resources directory containing model files
            resampling: Tuple of (x,y,z) resampling resolution in mm
            crop_box_size: List of [x,y,z] crop box size in mm
        """
        self.resource_path = Path(resource_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resampling = resampling
        self.crop_box_size = np.array(crop_box_size)
        
        # Model components
        self.ensemble_data = None
        self.fold_models = []
        self.clinical_preprocessors = None
        
        # Image preprocessing transforms (applied after resampling and cropping)
        self.image_transforms = self._create_image_transforms()
        
        self._load_model_components()
    
    def _create_image_transforms(self):
        """Create image preprocessing transforms."""
        return Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensity(),
            Resize(spatial_size=(96, 96, 96)),
            ToTensor()
        ])
    
    def _get_bounding_boxes(self, ct_sitk, pet_sitk):
        """
        Get the bounding boxes of the CT and PET images.
        This works since all images have the same direction.
        """
        ct_origin = np.array(ct_sitk.GetOrigin())
        pet_origin = np.array(pet_sitk.GetOrigin())

        ct_position_max = ct_origin + np.array(ct_sitk.GetSize()) * np.array(ct_sitk.GetSpacing())
        pet_position_max = pet_origin + np.array(pet_sitk.GetSize()) * np.array(pet_sitk.GetSpacing())
        
        return np.concatenate([
            np.maximum(ct_origin, pet_origin),
            np.minimum(ct_position_max, pet_position_max),
        ], axis=0)
    
    def _resample_images(self, ct_array, pet_array):
        """
        Resample CT and PET images to specified resolution using SimpleITK.
        
        Args:
            ct_array: CT image as numpy array
            pet_array: PET image as numpy array
            
        Returns:
            Tuple of (resampled_ct_array, resampled_pet_array)
        """
        # Convert numpy arrays to SimpleITK images
        ct_sitk = sitk.GetImageFromArray(ct_array)
        pet_sitk = sitk.GetImageFromArray(pet_array)
        
        # Set default spacing and origin if not present
        ct_sitk.SetSpacing([1.0, 1.0, 1.0])
        pet_sitk.SetSpacing([1.0, 1.0, 1.0])
        ct_sitk.SetOrigin([0.0, 0.0, 0.0])
        pet_sitk.SetOrigin([0.0, 0.0, 0.0])
        
        # Get bounding box for both modalities
        bb = self._get_bounding_boxes(ct_sitk, pet_sitk)
        size = np.round((bb[3:] - bb[:3]) / np.array(self.resampling)).astype(int)
        
        # Set up resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
        resampler.SetOutputSpacing(self.resampling)
        resampler.SetOutputOrigin(bb[:3])
        resampler.SetSize([int(k) for k in size])
        
        # Resample CT with B-spline interpolation
        resampler.SetInterpolator(sitk.sitkBSpline)
        ct_resampled = resampler.Execute(ct_sitk)
        
        # Resample PET with B-spline interpolation
        pet_resampled = resampler.Execute(pet_sitk)
        
        # Convert back to numpy arrays
        ct_resampled_array = sitk.GetArrayFromImage(ct_resampled)
        pet_resampled_array = sitk.GetArrayFromImage(pet_resampled)
        
        return ct_resampled_array, pet_resampled_array
    
    def _get_roi_center(self, pet_tensor, z_top_fraction=0.75, z_score_threshold=1.0):
        """
        Calculates the center of the largest high-intensity region in the top part of the PET scan.
        """
        # 1. Isolate top of the scan based on the z-axis
        image_shape_voxels = np.array(pet_tensor.shape)
        crop_z_start = int(z_top_fraction * image_shape_voxels[2])
        top_of_scan = pet_tensor[..., crop_z_start:]

        # 2. Threshold to find high-intensity regions (potential brain/tumor)
        # Using a small epsilon to avoid division by zero in blank images
        mask = ((top_of_scan - top_of_scan.mean()) / (top_of_scan.std() + 1e-8)) > z_score_threshold
        
        if not mask.any():
            # If no pixels are above the threshold, fall back to the geometric center of the top part
            warnings.warn("No high-intensity region found. Using geometric center of the upper scan region.")
            center_in_top = (np.array(top_of_scan.shape) / 2).astype(int)
        else:
            # Find the largest connected component to remove noise
            labeled_mask, num_features = label(mask, return_num=True, connectivity=3)
            if num_features > 0:
                component_sizes = np.bincount(labeled_mask.ravel())[1:]  # ignore background
                largest_component_label = np.argmax(component_sizes) + 1
                largest_component_mask = labeled_mask == largest_component_label
                comp_idx = np.argwhere(largest_component_mask)
            else:  # Should not happen if mask.any() is true, but as a safeguard
                comp_idx = np.argwhere(mask)

            # 3. Calculate the centroid of the largest component
            center_in_top = np.mean(comp_idx, axis=0)

        # 4. Adjust center to be in the original full-image coordinate system
        center_full_image = center_in_top + np.array([0, 0, crop_z_start])
        return center_full_image.astype(int)
    
    def _crop_neck_region(self, ct_array, pet_array):
        """
        Crop the head and neck region from CT and PET images.
        
        Args:
            ct_array: CT image as numpy array
            pet_array: PET image as numpy array
            
        Returns:
            Tuple of (cropped_ct_array, cropped_pet_array)
        """
        # Convert to torch tensor for processing
        pet_tensor = torch.from_numpy(pet_array).float()
        
        # Get box size in voxels (assuming 1mm spacing after resampling)
        box_size_voxels = self.crop_box_size.astype(int)
        
        # 1. Find the robust center of the ROI using PET
        center_voxels = self._get_roi_center(pet_tensor)
        
        # 2. Calculate crop box and handle boundaries safely
        image_shape_voxels = np.array(pet_array.shape)
        box_start = center_voxels - box_size_voxels // 2
        box_end = box_start + box_size_voxels
        
        # Clamp coordinates to ensure they are within the image boundaries
        box_start = np.maximum(box_start, 0)
        box_end = np.minimum(box_end, image_shape_voxels)
        
        # Recalculate start to handle cases where box goes over the 0-boundary
        box_start = np.maximum(box_end - box_size_voxels, 0)
        
        box_start = box_start.astype(int)
        box_end = box_end.astype(int)
        
        # Apply the crop to both CT and PET
        ct_cropped = ct_array[box_start[0]:box_end[0], box_start[1]:box_end[1], box_start[2]:box_end[2]]
        pet_cropped = pet_array[box_start[0]:box_end[0], box_start[1]:box_end[1], box_start[2]:box_end[2]]
        
        return ct_cropped, pet_cropped
    
    def _load_model_components(self):
        """Load ensemble model and clinical preprocessors."""
        # Load ensemble model
        ensemble_path = self.resource_path / "ensemble_model.pt"
        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble model not found: {ensemble_path}")
        
        self.ensemble_data = torch.load(ensemble_path, map_location=self.device, weights_only=False)
        
        # Initialize fold models
        self.fold_models = []
        for fold_data in self.ensemble_data['fold_models']:
            fold_id = fold_data['fold_id']
            weight = fold_data['weight']
                        
            # Create feature extractor
            feature_extractor = FusedFeatureExtractor(
                clinical_feature_dim=self.ensemble_data['clinical_feature_dim'],
                feature_output_dim=self.ensemble_data['feature_output_dim']
            ).to(self.device)
            
            # Load weights
            feature_extractor.load_state_dict(fold_data['feature_extractor_state_dict'])
            feature_extractor.eval()
            
            # Store fold model
            fold_model = {
                'fold_id': fold_id,
                'feature_extractor': feature_extractor,
                'icare_model': fold_data['icare_model'],
                'weight': weight
            }
            
            self.fold_models.append(fold_model)
        
        # Load clinical preprocessors
        preprocessors_path = self.resource_path / "clinical_preprocessors.pkl"
        if preprocessors_path.exists():
            with open(preprocessors_path, 'rb') as f:
                self.clinical_preprocessors = pickle.load(f)
        else:
            raise FileNotFoundError(f"Clinical preprocessors not found at {preprocessors_path}")        
    
    def _preprocess_images(self, ct_image, pet_image):
        """
        Preprocess CT and PET images with resampling, cropping, and MONAI transforms.
        
        Args:
            ct_image: CT image as numpy array
            pet_image: PET image as numpy array
            
        Returns:
            Combined CT+PET tensor ready for model inference
        """
        print("Starting image preprocessing...")
        
        # Step 1: Resample images to consistent resolution
        print(f"Resampling images to {self.resampling} mm resolution...")
        ct_resampled, pet_resampled = self._resample_images(ct_image, pet_image)
        print(f"Resampled shapes - CT: {ct_resampled.shape}, PET: {pet_resampled.shape}")
        
        # Step 2: Crop head and neck region
        print(f"Cropping neck region with box size {self.crop_box_size} mm...")
        ct_cropped, pet_cropped = self._crop_neck_region(ct_resampled, pet_resampled)
        print(f"Cropped shapes - CT: {ct_cropped.shape}, PET: {pet_cropped.shape}")
        
        # Step 3: Apply MONAI transforms
        print("Applying MONAI transforms...")
        ct_transformed = self.image_transforms(ct_cropped)
        pet_transformed = self.image_transforms(pet_cropped)
        
        # Step 4: Combine CT and PET channels
        combined_image = torch.cat([ct_transformed, pet_transformed], dim=0)
        
        # Add batch dimension
        combined_image = combined_image.unsqueeze(0)  # [1, 2, H, W, D]
        
        print(f"Final preprocessed shape: {combined_image.shape}")
        return combined_image.to(self.device)
    
    def _preprocess_clinical_data(self, clinical_data):
        """
        Preprocess clinical data from EHR JSON.
        
        Args:
            clinical_data: Dictionary containing clinical information
            
        Returns:
            Preprocessed clinical features as tensor
        """
        try:
            # Extract values, handling NaN/None
            def handle_nan(value):
                if value is None or (isinstance(value, (int, float)) and math.isnan(value)):
                    return float('nan')
                return value
            
            age = handle_nan(clinical_data.get('Age', None))
            gender = handle_nan(clinical_data.get('Gender', None))
            tobacco = handle_nan(clinical_data.get('Tobacco Consumption', None))
            alcohol = handle_nan(clinical_data.get('Alcohol Consumption', None))
            performance = handle_nan(clinical_data.get('Performance Status', None))
            treatment = handle_nan(clinical_data.get('Treatment', None))
            m_stage = clinical_data.get('M-stage', None)  
            
            # Create DataFrame exactly like training expects
            patient_df = pd.DataFrame({
                'PatientID': ['TEMP_PATIENT'],
                'Age': [age],
                'Gender': [gender],
                'Tobacco Consumption': [tobacco],
                'Alcohol Consumption': [alcohol],
                'Performance Status': [performance],
                'M-stage': [m_stage],
                'Treatment': [treatment]
            })
            
        except Exception as e:
            print(f"Error extracting clinical data: {e}")
            raise e
        
        # Use the existing preprocessing method
        clinical_result = self.preprocess_test_clinical_data(patient_df)
        processed_features = clinical_result['features']['TEMP_PATIENT']
        
        return torch.tensor(processed_features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def preprocess_test_clinical_data(self, dataframe):
        """
        Preprocess clinical data using the same parameters as training.
        """
        # All clinical features (same as training)
        ALL_CLINICAL_FEATURES = [
            "Age", "Gender", "Tobacco Consumption", "Alcohol Consumption", 
            "Performance Status", "M-stage", "Treatment"
        ]
        
        CATEGORICAL_FEATURES = [
            "Gender", "Tobacco Consumption", "Alcohol Consumption", 
            "Performance Status", "M-stage", "Treatment"
        ]
        
        feature_subset = dataframe[ALL_CLINICAL_FEATURES].copy()
        
        # Handle Age using training parameters
        age_median = self.clinical_preprocessors['age_median']
        age_scaler = self.clinical_preprocessors['age_scaler']
        
        feature_subset["Age"] = feature_subset["Age"].fillna(age_median)
        age_scaled = age_scaler.transform(feature_subset[["Age"]])
        
        # Handle categorical features
        categorical_data = feature_subset[CATEGORICAL_FEATURES].copy()
        for col in CATEGORICAL_FEATURES:
            categorical_data[col] = categorical_data[col].fillna('Unknown')
            categorical_data[col] = categorical_data[col].astype(str)
        
        # Apply one-hot encoding (same structure as training)
        categorical_encoded = pd.get_dummies(
            categorical_data, 
            columns=CATEGORICAL_FEATURES,
            prefix=CATEGORICAL_FEATURES,
            dummy_na=False,
            drop_first=False
        )
        
        # Ensure same feature structure as training
        training_categorical_columns = [col for col in self.clinical_preprocessors['categorical_columns']]
        
        # Add missing columns with zeros (ensure they are numeric)
        for col in training_categorical_columns:
            if col not in categorical_encoded.columns:
                categorical_encoded[col] = 0
        
        # Remove extra columns and reorder to match training
        categorical_encoded = categorical_encoded[training_categorical_columns]
        
        # CRITICAL: Ensure all categorical features are numeric
        categorical_encoded = categorical_encoded.astype(np.float32)
        
        # Process all patients
        processed_features = {}
        
        for idx, row in dataframe.iterrows():
            patient_id = row["PatientID"]
            patient_row_idx = dataframe.index.get_loc(idx)
            
            age_features = age_scaled[patient_row_idx].flatten().astype(np.float32)
            categorical_features = categorical_encoded.iloc[patient_row_idx].values.astype(np.float32)
            
            complete_features = np.concatenate([age_features, categorical_features]).astype(np.float32)
            
            # Verify no object types
            if complete_features.dtype == np.object_:
                # Convert any remaining object types to float
                complete_features = complete_features.astype(np.float32)
            
            processed_features[patient_id] = complete_features
        
        return {
            'features': processed_features,
            'preprocessors': self.clinical_preprocessors
        }
    
    def predict_single_patient(self, ct_image, pet_image, clinical_data):
        """
        Predict RFS risk for a single patient.
        
        Args:
            ct_image: CT image as numpy array
            pet_image: PET image as numpy array
            clinical_data: Clinical data as dictionary
            
        Returns:
            RFS risk prediction as float
        """
        # Preprocess images (includes resampling, cropping, and MONAI transforms)
        image_tensor = self._preprocess_images(ct_image, pet_image)
        
        # Preprocess clinical data
        clinical_tensor = self._preprocess_clinical_data(clinical_data)
        
        # Get predictions from all folds
        fold_predictions = []
        fold_weights = []
        
        for fold_model in self.fold_models:
            fold_id = fold_model['fold_id']
            weight = fold_model['weight']
            
            # Extract features
            with torch.no_grad():
                features = fold_model['feature_extractor'](image_tensor, clinical_tensor)
                features_np = features.cpu().numpy()
            
            # Get prediction from icare model
            prediction = fold_model['icare_model'].predict(features_np)[0]
            
            fold_predictions.append(prediction)
            fold_weights.append(weight)
        
        # Combine predictions
        fold_predictions = np.array(fold_predictions)
        combination_method = self.ensemble_data['combination_method']
        
        if combination_method == "median":
            final_prediction = np.median(fold_predictions)
        elif combination_method == "average":
            final_prediction = np.mean(fold_predictions)
        elif combination_method == "weighted_average":
            fold_weights = np.array(fold_weights)
            normalized_weights = fold_weights / np.sum(fold_weights)
            final_prediction = np.average(fold_predictions, weights=normalized_weights)
        elif combination_method == "best_fold":
            best_fold_idx = np.argmax(fold_weights)
            final_prediction = fold_predictions[best_fold_idx]
        else:
            final_prediction = np.median(fold_predictions)
                
        return float(final_prediction)
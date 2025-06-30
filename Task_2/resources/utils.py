"""
HECKTOR Survival Model Utilities for Container Deployment
"""

import os
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from monai.transforms import (
    Compose, EnsureChannelFirst, ScaleIntensity, ToTensor, Resize
)
import torch.nn as nn
from monai.networks.nets import resnet18
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
    
    def __init__(self, resource_path):
        """
        Initialize the inference model.
        
        Args:
            resource_path: Path to resources directory containing model files
        """
        self.resource_path = Path(resource_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.ensemble_data = None
        self.fold_models = []
        self.clinical_preprocessors = None
        
        # Image preprocessing
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
        Preprocess CT and PET images.
        
        Args:
            ct_image: CT image as numpy array
            pet_image: PET image as numpy array
            
        Returns:
            Combined CT+PET tensor
        """
        # Apply transforms to each image
        ct_transformed = self.image_transforms(ct_image)
        pet_transformed = self.image_transforms(pet_image)
        
        # Combine CT and PET channels
        combined_image = torch.cat([ct_transformed, pet_transformed], dim=0)
        
        # Add batch dimension
        combined_image = combined_image.unsqueeze(0)  # [1, 2, H, W, D]
        
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
        # Preprocess images
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
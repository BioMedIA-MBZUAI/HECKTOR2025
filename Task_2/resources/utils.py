"""
Utility functions and classes for HECKTOR survival model inference.
Contains model architectures, preprocessing, and inference logic.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
from monai.networks.nets import resnet18

# =============================================================================
# Model Architecture
# =============================================================================

class FusedFeatureExtractor(nn.Module):
    """Feature extractor for combining 3D medical imaging and clinical data."""
    
    def __init__(self, clinical_feature_dim, feature_output_dim=128):
        super().__init__()
        
        self.clinical_feature_dim = clinical_feature_dim
        self.feature_output_dim = feature_output_dim
        
        # 3D ResNet-18 for combined CT+PET input
        self.imaging_backbone = resnet18(
            spatial_dims=3,
            n_input_channels=2,
            num_classes=1,
        )
        self.imaging_backbone.fc = nn.Identity()

        # Clinical data processor
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

        # Feature fusion
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
        
        # Risk prediction head
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

# =============================================================================
# Clinical Data Preprocessor
# =============================================================================

class ClinicalDataPreprocessor:
    """Handles clinical data preprocessing using saved preprocessors from training."""
    
    def __init__(self, preprocessors):
        self.imputer = preprocessors['imputer']
        self.age_scaler = preprocessors['age_scaler']
        self.category_encoders = preprocessors['category_encoders']
        self.ordinal_encoder = preprocessors['ordinal_encoder']
        
        self.ALL_CLINICAL_FEATURES = [
            "Age", "Gender", "Tobacco Consumption", "Alcohol Consumption", 
            "Performance Status", "M-stage"
        ]
        
        self.LABEL_ENCODED_FEATURES = [
            "Gender", "Tobacco Consumption", "Alcohol Consumption", "Performance Status"
        ]

    def preprocess_patient_json(self, clinical_json):
        """
        Preprocess clinical data from JSON format.
        
        Args:
            clinical_json: Dict with clinical data from EHR JSON
            
        Returns:
            numpy array of processed features
        """
        # Convert JSON to expected format
        patient_dict = self._convert_json_to_features(clinical_json)
        
        # Create DataFrame for processing
        patient_df = pd.DataFrame([patient_dict])
        
        # Ensure all required features are present
        for feature in self.ALL_CLINICAL_FEATURES:
            if feature not in patient_df.columns:
                patient_df[feature] = np.nan
        
        feature_subset = patient_df[self.ALL_CLINICAL_FEATURES].copy()
        
        # Temporary encoding for imputation
        imputation_data = feature_subset.copy()
        
        for column in self.LABEL_ENCODED_FEATURES:
            known_classes = list(self.category_encoders[column].classes_)
            value = str(imputation_data[column].iloc[0]).replace('nan', 'Unknown')
            if value not in known_classes:
                value = known_classes[0]
            imputation_data[column] = known_classes.index(value)
        
        # Handle M-stage for imputation
        m_stage_value = str(imputation_data['M-stage'].iloc[0]).replace('nan', 'Unknown')
        if m_stage_value in ['M0', '0', '0.0']:
            m_stage_encoded = 0
        elif m_stage_value in ['M1', '1', '1.0']:
            m_stage_encoded = 1
        else:
            m_stage_encoded = 2
        imputation_data['M-stage'] = m_stage_encoded
        
        # Apply KNN imputation
        imputed_features = self.imputer.transform(imputation_data.values)
        imputed_df = pd.DataFrame(imputed_features, columns=self.ALL_CLINICAL_FEATURES)
        
        # Process final features
        age_standardized = self.age_scaler.transform(imputed_df[["Age"]].values)
        
        # Categorical features encoding
        encoded_categoricals = []
        for column in self.LABEL_ENCODED_FEATURES:
            column_value = imputed_df[column].iloc[0]
            column_value_str = str(int(round(column_value)))
            
            try:
                encoded_value = self.category_encoders[column].transform([column_value_str])
            except ValueError:
                encoded_value = np.array([0])
            
            encoded_categoricals.append(encoded_value.reshape(-1, 1))
        
        # M-stage ordinal encoding
        m_stage_value = int(round(imputed_df['M-stage'].iloc[0]))
        if m_stage_value == 0:
            m_stage_category = 'M0'
        elif m_stage_value == 1:
            m_stage_category = 'M1'
        else:
            m_stage_category = 'Mx'
        
        m_stage_encoded = self.ordinal_encoder.transform([[m_stage_category]])
        
        # Combine all features
        categorical_array = np.hstack(encoded_categoricals)
        complete_features = np.hstack([age_standardized, categorical_array, m_stage_encoded])
        
        return complete_features.flatten()
    
    def _convert_json_to_features(self, clinical_json):
        """Convert EHR JSON to expected feature format."""
        # Map JSON keys to expected feature names
        # This mapping might need adjustment based on actual JSON structure
        feature_dict = {}
        
        # Age
        feature_dict["Age"] = clinical_json.get("age", clinical_json.get("Age", np.nan))
        
        # Gender
        gender = clinical_json.get("gender", clinical_json.get("Gender", "Unknown"))
        feature_dict["Gender"] = str(gender).upper()
        
        # Tobacco Consumption
        tobacco = clinical_json.get("tobacco", clinical_json.get("Tobacco Consumption", "Unknown"))
        if str(tobacco).upper() in ['YES', 'Y', '1', 'TRUE', 'SMOKER']:
            feature_dict["Tobacco Consumption"] = "YES"
        elif str(tobacco).upper() in ['NO', 'N', '0', 'FALSE', 'NON-SMOKER']:
            feature_dict["Tobacco Consumption"] = "NO"
        else:
            feature_dict["Tobacco Consumption"] = "Unknown"
        
        # Alcohol Consumption
        alcohol = clinical_json.get("alcohol", clinical_json.get("Alcohol Consumption", "Unknown"))
        if str(alcohol).upper() in ['YES', 'Y', '1', 'TRUE']:
            feature_dict["Alcohol Consumption"] = "YES"
        elif str(alcohol).upper() in ['NO', 'N', '0', 'FALSE']:
            feature_dict["Alcohol Consumption"] = "NO"
        else:
            feature_dict["Alcohol Consumption"] = "Unknown"
        
        # Performance Status
        performance = clinical_json.get("performance_status", 
                                      clinical_json.get("Performance Status", 
                                      clinical_json.get("karnofsky", np.nan)))
        feature_dict["Performance Status"] = str(performance) if performance is not None else "Unknown"
        
        # M-stage
        m_stage = clinical_json.get("m_stage", 
                                   clinical_json.get("M-stage",
                                   clinical_json.get("M_stage", "Unknown")))
        feature_dict["M-stage"] = str(m_stage)
        
        return feature_dict

# =============================================================================
# Image Preprocessing
# =============================================================================

class ImagePreprocessor:
    """Handles image preprocessing for inference."""
    
    def __init__(self, target_size=(96, 96, 96)):
        self.target_size = target_size
    
    def preprocess_images(self, ct_array, pet_array):
        """
        Preprocess CT and PET images from numpy arrays.
        
        Args:
            ct_array: CT image as numpy array
            pet_array: PET image as numpy array
            
        Returns:
            Combined tensor ready for model input
        """
        # Ensure arrays are float32
        ct_array = ct_array.astype(np.float32)
        pet_array = pet_array.astype(np.float32)
        
        # Process each image separately
        ct_tensor = self._process_single_image(ct_array)
        pet_tensor = self._process_single_image(pet_array)
        
        # Combine CT and PET (both have shape [1, D, H, W])
        combined_tensor = torch.cat([ct_tensor, pet_tensor], dim=0)  # Shape: [2, D, H, W]
        
        # Add batch dimension
        combined_tensor = combined_tensor.unsqueeze(0)  # Shape: [1, 2, D, H, W]
        
        return combined_tensor
    
    def _process_single_image(self, image_array):
        """Process a single image array to tensor."""
        # Ensure we have the right shape - add channel dimension if needed
        if len(image_array.shape) == 3:
            # Shape is (D, H, W), add channel dimension -> (1, D, H, W)
            image_array = image_array[np.newaxis, ...]
        elif len(image_array.shape) == 4 and image_array.shape[0] == 1:
            # Already has channel dimension (1, D, H, W)
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image_array.shape}. Expected 3D or 4D array.")
        
        # Convert to tensor
        tensor = torch.from_numpy(image_array).float()
        
        # Apply intensity scaling (normalize to 0-1 range)
        tensor = self._scale_intensity(tensor)
        
        # Resize to target size
        tensor = self._resize_tensor(tensor)
        
        return tensor
    
    def _scale_intensity(self, tensor):
        """Scale intensity values."""
        # Get min and max values, ignoring any potential NaN values
        tensor_flat = tensor.flatten()
        tensor_flat = tensor_flat[~torch.isnan(tensor_flat)]
        
        if len(tensor_flat) == 0:
            return tensor
        
        min_val = tensor_flat.min()
        max_val = tensor_flat.max()
        
        # Avoid division by zero
        if max_val - min_val > 1e-8:
            tensor = (tensor - min_val) / (max_val - min_val)
        
        return tensor
    
    def _resize_tensor(self, tensor):
        """Resize tensor to target size."""
        # tensor shape: (1, D, H, W)
        current_size = tensor.shape[1:]  # (D, H, W)
        
        if current_size != self.target_size:
            # Add batch dimension for interpolation: (1, 1, D, H, W)
            tensor = tensor.unsqueeze(0)
            
            # Resize using trilinear interpolation
            tensor = F.interpolate(
                tensor, 
                size=self.target_size, 
                mode='trilinear', 
                align_corners=False
            )
            
            # Remove batch dimension: (1, D, H, W)
            tensor = tensor.squeeze(0)
        
        return tensor

# =============================================================================
# Main Inference Model
# =============================================================================

class HecktorInferenceModel:
    """Main inference model for HECKTOR survival prediction."""
    
    def __init__(self, resource_path):
        self.resource_path = resource_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.feature_extractor = None
        self.icare_model = None
        self.clinical_preprocessor = None
        self.image_preprocessor = ImagePreprocessor()
        
        # Load all components
        self._load_model()
        self._load_preprocessors()
    
    def _load_model(self):
        """Load the trained model components."""
        print(f"Loading model from {self.resource_path}...")
        
        # Load config
        config_path = self.resource_path / "model_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        clinical_dim = config['clinical_feature_dim']
        feature_dim = config['feature_dim']
        
        # Initialize and load feature extractor
        self.feature_extractor = FusedFeatureExtractor(
            clinical_feature_dim=clinical_dim,
            feature_output_dim=feature_dim
        ).to(self.device)
        
        # Load feature extractor weights
        feature_extractor_path = self.resource_path / "feature_extractor.pt"
        state_dict = torch.load(feature_extractor_path, map_location=self.device)
        self.feature_extractor.load_state_dict(state_dict)
        self.feature_extractor.eval()
        
        # Load BaggedIcareSurvival model
        icare_path = self.resource_path / "icare_model.pkl"
        with open(icare_path, 'rb') as f:
            self.icare_model = pickle.load(f)
        
        print(f"Model loaded successfully (Best C-index: {config.get('best_c_index', 'N/A')})")
    
    def _load_preprocessors(self):
        """Load clinical data preprocessors."""
        preprocessors_path = self.resource_path / "clinical_preprocessors.pkl"
        print(f"Loading preprocessors from {preprocessors_path}...")
        
        with open(preprocessors_path, 'rb') as f:
            preprocessors = pickle.load(f)
        
        self.clinical_preprocessor = ClinicalDataPreprocessor(preprocessors)
        print("Preprocessors loaded successfully")
    
    def predict_single_patient(self, ct_image, pet_image, clinical_data):
        """
        Generate prediction for a single patient.
        
        Args:
            ct_image: CT image as numpy array
            pet_image: PET image as numpy array
            clinical_data: Clinical data as JSON dict
            
        Returns:
            Float risk score
        """
        try:
            # Print debug info
            print(f"CT image shape: {ct_image.shape}, dtype: {ct_image.dtype}")
            print(f"PET image shape: {pet_image.shape}, dtype: {pet_image.dtype}")
            print(f"Clinical data keys: {list(clinical_data.keys())}")
            
            # Preprocess images
            print("Preprocessing images...")
            combined_image = self.image_preprocessor.preprocess_images(ct_image, pet_image)
            print(f"Combined image tensor shape: {combined_image.shape}")
            
            # Preprocess clinical data
            print("Preprocessing clinical data...")
            clinical_features = self.clinical_preprocessor.preprocess_patient_json(clinical_data)
            print(f"Clinical features shape: {clinical_features.shape}")
            
            # Extract features
            print("Extracting features...")
            self.feature_extractor.eval()
            with torch.no_grad():
                combined_image = combined_image.to(self.device)
                clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                features = self.feature_extractor(combined_image, clinical_tensor)
                features_np = features.cpu().numpy()
                print(f"Extracted features shape: {features_np.shape}")
            
            # Generate prediction using BaggedIcareSurvival
            print("Generating prediction...")
            prediction = self.icare_model.predict(features_np)
            print(f"Raw prediction: {prediction}")
            
            # Return single prediction value
            return float(prediction[0])
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            raise e

# =============================================================================
# Utility Functions
# =============================================================================

def check_resource_files(resource_path):
    """Check if all required resource files are present."""
    required_files = [
        "model_config.json",
        "feature_extractor.pt", 
        "icare_model.pkl",
        "clinical_preprocessors.pkl"
    ]
    
    missing_files = []
    for file_name in required_files:
        if not (resource_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required resource files: {missing_files}")
    
    print("All required resource files found.")

def prepare_resources_from_training(training_fold_prefix, training_preprocessors_path, output_resource_path):
    """
    Helper function to prepare resource files from training outputs.
    
    Args:
        training_fold_prefix: Path prefix from training (e.g., "cv_logs/fold0_system")
        training_preprocessors_path: Path to clinical preprocessors from training
        output_resource_path: Where to copy files for container deployment
    """
    import shutil
    from pathlib import Path
    
    output_path = Path(output_resource_path)
    output_path.mkdir(exist_ok=True)
    
    # Copy model files
    shutil.copy(f"{training_fold_prefix}_feature_extractor.pt", output_path / "feature_extractor.pt")
    shutil.copy(f"{training_fold_prefix}_icare_model.pkl", output_path / "icare_model.pkl")
    shutil.copy(f"{training_fold_prefix}_config.json", output_path / "model_config.json")
    shutil.copy(training_preprocessors_path, output_path / "clinical_preprocessors.pkl")
    
    print(f"Resource files prepared in {output_resource_path}")
#!/usr/bin/env python3
"""
HECKTOR Survival Model Inference Script

Standalone script for generating survival predictions on new HECKTOR data.
Requires trained model files and clinical preprocessors from training.

Usage:
    python inference.py --csv input.csv --images /path/to/images --model cv_logs/fold0_system --output predictions.csv

Input CSV should contain columns:
    PatientID, Age, Gender, Tobacco Consumption, Alcohol Consumption, Performance Status, M-stage

Image files should be named: {PatientID}__CT.nii.gz and {PatientID}__PT.nii.gz
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Resized
)
from monai.networks.nets import resnet18
import torch.nn as nn
from tqdm import tqdm

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
        imaging_features = self.imaging_backbone(medical_images)
        clinical_features_processed = self.clinical_processor(clinical_features)
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

    def preprocess_patient(self, patient_row):
        """Preprocess clinical data for a single patient."""
        if hasattr(patient_row, 'to_dict'):
            patient_dict = patient_row.to_dict()
        else:
            patient_dict = patient_row.copy()
        
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
                value = 'Unknown'
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

# =============================================================================
# Dataset for Inference
# =============================================================================

class InferenceDataset(Dataset):
    """Dataset for inference on new data."""
    def __init__(self, patient_data):
        self.patient_ids = list(patient_data.keys())
        self.patient_data = patient_data

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data = self.patient_data[patient_id]
        
        image_tensor = data['image']
        clinical_tensor = torch.tensor(data['clinical'], dtype=torch.float32)
        
        return patient_id, image_tensor, clinical_tensor

# =============================================================================
# Main Inference Class
# =============================================================================

class HecktorInference:
    """Main inference pipeline for HECKTOR survival model."""
    
    def __init__(self, model_prefix, preprocessors_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_prefix = model_prefix
        self.preprocessors_path = preprocessors_path
        
        # Model components
        self.feature_extractor = None
        self.icare_model = None
        self.clinical_preprocessor = None
        
        # Image transforms
        self.image_transforms = Compose([
            LoadImaged(keys=["ct", "pet"]),
            EnsureChannelFirstd(keys=["ct", "pet"]),
            ScaleIntensityd(keys=["ct", "pet"]),
            Resized(keys=["ct", "pet"], spatial_size=(96, 96, 96)),
            ToTensord(keys=["ct", "pet"]),
        ])
        
        self._load_model()
        self._load_preprocessors()
    
    def _load_model(self):
        """Load the trained model components."""
        print(f"Loading model from {self.model_prefix}...")
        
        # Load config
        config_path = f"{self.model_prefix}_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        clinical_dim = config['clinical_feature_dim']
        feature_dim = config['feature_dim']
        
        # Initialize and load feature extractor
        self.feature_extractor = FusedFeatureExtractor(
            clinical_feature_dim=clinical_dim,
            feature_output_dim=feature_dim
        ).to(self.device)
        
        feature_extractor_path = f"{self.model_prefix}_feature_extractor.pt"
        state_dict = torch.load(feature_extractor_path, map_location=self.device)
        self.feature_extractor.load_state_dict(state_dict)
        self.feature_extractor.eval()
        
        # Load BaggedIcareSurvival model
        icare_path = f"{self.model_prefix}_icare_model.pkl"
        with open(icare_path, 'rb') as f:
            self.icare_model = pickle.load(f)
        
        print(f"Model loaded successfully (Best C-index: {config.get('best_c_index', 'N/A')})")
    
    def _load_preprocessors(self):
        """Load clinical data preprocessors."""
        print(f"Loading preprocessors from {self.preprocessors_path}...")
        with open(self.preprocessors_path, 'rb') as f:
            preprocessors = pickle.load(f)
        
        self.clinical_preprocessor = ClinicalDataPreprocessor(preprocessors)
        print("Preprocessors loaded successfully")
    
    def _find_image_path(self, patient_id, modality, image_directory):
        """Find image file for a patient."""
        filename = f"{patient_id}__{modality}.nii.gz"
        full_path = os.path.join(image_directory, filename)
        if os.path.exists(full_path):
            return full_path
        raise FileNotFoundError(f"{modality} image for patient {patient_id} not found: {full_path}")
    
    def _preprocess_patient(self, patient_id, clinical_data, image_directory):
        """Preprocess data for a single patient."""
        # Process images
        ct_path = self._find_image_path(patient_id, "CT", image_directory)
        pet_path = self._find_image_path(patient_id, "PT", image_directory)
        
        transformed_data = self.image_transforms({"ct": ct_path, "pet": pet_path})
        combined_image = torch.cat([transformed_data["ct"], transformed_data["pet"]], dim=0)
        
        # Process clinical data
        processed_clinical = self.clinical_preprocessor.preprocess_patient(clinical_data)
        
        return {
            'image': combined_image,
            'clinical': processed_clinical
        }
    
    def predict(self, csv_path, image_directory, batch_size=4):
        """
        Generate predictions for patients in CSV file.
        
        Args:
            csv_path: Path to CSV with patient clinical data
            image_directory: Directory containing patient images
            batch_size: Batch size for processing
            
        Returns:
            pandas DataFrame with predictions
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Preprocess all patients
        patient_data = {}
        failed_patients = []
        
        print("Preprocessing patients...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            patient_id = row["PatientID"]
            
            try:
                processed_data = self._preprocess_patient(patient_id, row, image_directory)
                patient_data[patient_id] = processed_data
            except Exception as e:
                print(f"Failed to preprocess {patient_id}: {e}")
                failed_patients.append(patient_id)
        
        if failed_patients:
            print(f"Skipped {len(failed_patients)} patients due to preprocessing errors")
        
        if len(patient_data) == 0:
            raise ValueError("No patients could be processed successfully")
        
        # Create dataset and dataloader
        dataset = InferenceDataset(patient_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Extract features
        print("Extracting features...")
        all_features = []
        patient_order = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                patient_ids, images, clinical = batch
                images = images.to(self.device)
                clinical = clinical.to(self.device)
                
                features = self.feature_extractor(images, clinical)
                all_features.append(features.cpu().numpy())
                patient_order.extend(patient_ids)
        
        # Generate predictions
        print("Generating predictions...")
        feature_matrix = np.vstack(all_features)
        predictions = self.icare_model.predict(feature_matrix)
        
        # Create results dataframe
        results_data = []
        for patient_id, prediction in zip(patient_order, predictions):
            results_data.append({
                'PatientID': patient_id,
                'RiskScore': float(prediction)
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Merge with original data
        results_df = results_df.merge(df, on='PatientID', how='left')
        
        print(f"Generated predictions for {len(results_df)} patients")
        
        return results_df

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='HECKTOR Survival Model Inference')
    parser.add_argument('--csv', required=True, help='Path to CSV file with patient data')
    parser.add_argument('--images', required=True, help='Directory containing patient images')
    parser.add_argument('--model', required=True, help='Model prefix (e.g., cv_logs/fold0_system)')
    parser.add_argument('--preprocessors', required=True, help='Path to clinical preprocessors pickle file')
    parser.add_argument('--output', required=True, help='Output CSV path for predictions')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)
    
    if not os.path.isdir(args.images):
        print(f"Error: Image directory not found: {args.images}")
        sys.exit(1)
    
    required_files = [
        f"{args.model}_config.json",
        f"{args.model}_feature_extractor.pt",
        f"{args.model}_icare_model.pkl"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Model file not found: {file_path}")
            sys.exit(1)
    
    if not os.path.exists(args.preprocessors):
        print(f"Error: Preprocessors file not found: {args.preprocessors}")
        sys.exit(1)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    try:
        # Initialize inference
        inference = HecktorInference(
            model_prefix=args.model,
            preprocessors_path=args.preprocessors,
            device=device
        )
        
        # Generate predictions
        results_df = inference.predict(
            csv_path=args.csv,
            image_directory=args.images,
            batch_size=args.batch_size
        )
        
        # Save results
        results_df.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Processed: {len(results_df)} patients")
        print(f"- Risk scores range: {results_df['RiskScore'].min():.4f} to {results_df['RiskScore'].max():.4f}")
        print(f"- Mean risk score: {results_df['RiskScore'].mean():.4f}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
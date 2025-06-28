#!/usr/bin/env python3
"""
Inference script for HECKTOR survival prediction using ensemble model.
Usage: python inference_script.py --csv test_data.csv --images_dir ./test_images --ensemble ensemble_model.pt --clinical_preprocessors  hecktor_cache_clinical_preprocessors.pkl 
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

# Import necessary components from training script
from task2_prognosis import (
    FusedFeatureExtractor, 
    HecktorSurvivalDataset,
    create_image_transforms,
    find_image_path,
    set_random_seed,
    RANDOM_SEED)

class InferenceModel:
    """
    Ensemble inference class that loads multiple fold models and combines predictions.
    """
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ensemble_data = None
        self.fold_models = []
        self.clinical_preprocessors = None
    
    def load_ensemble_from_single_file(self, ensemble_path):
        """Load ensemble model from single file created by create_ensemble_file.py"""
        
        self.ensemble_data = torch.load(ensemble_path, map_location=self.device, weights_only=False)
        
        # Initialize feature extractors for each fold
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
            
            # Store fold model components
            fold_model = {
                'fold_id': fold_id,
                'feature_extractor': feature_extractor,
                'icare_model': fold_data['icare_model'],
                'weight': weight
            }
            
            self.fold_models.append(fold_model)
    
    def load_clinical_preprocessors(self, preprocessors_path):
        """Load clinical preprocessing parameters used during training."""
        if os.path.exists(preprocessors_path):
            with open(preprocessors_path, 'rb') as f:
                self.clinical_preprocessors = pickle.load(f)
        else:
            self.clinical_preprocessors = None
    
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
    
    def extract_features_from_fold(self, fold_model, data_loader):
        """Extract features using a specific fold model."""
        fold_model['feature_extractor'].eval()
        all_features = []
        
        with torch.no_grad():
            for images, clinical, *_ in data_loader:
                images, clinical = images.to(self.device), clinical.to(self.device)
                features = fold_model['feature_extractor'](images, clinical)
                all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)
    
    def predict_ensemble(self, data_loader):
        """
        Make predictions using all fold models.
        """     
        all_fold_predictions = []
        fold_weights = []
        
        # Get predictions from each fold
        for fold_model in self.fold_models:
            fold_id = fold_model['fold_id']
            weight = fold_model['weight']
            
            # Extract features using this fold's feature extractor
            features = self.extract_features_from_fold(fold_model, data_loader)
            
            # Get predictions using this fold's icare model
            predictions = fold_model['icare_model'].predict(features)
            
            all_fold_predictions.append(predictions)
            fold_weights.append(weight)
        
        # Convert to numpy arrays
        all_fold_predictions = np.array(all_fold_predictions)  # [n_folds, n_samples]
        fold_weights = np.array(fold_weights)
        
        # Combine predictions based on ensemble method
        combination_method = self.ensemble_data['combination_method']
        
        if combination_method == "median":
            final_predictions = np.median(all_fold_predictions, axis=0)
        elif combination_method == "average":
            final_predictions = np.mean(all_fold_predictions, axis=0)
        elif combination_method == "weighted_average":
            # Normalize weights
            normalized_weights = fold_weights / np.sum(fold_weights)
            final_predictions = np.average(all_fold_predictions, axis=0, weights=normalized_weights)
        elif combination_method == "best_fold":
            # Use the fold with highest weight
            best_fold_idx = np.argmax(fold_weights)
            final_predictions = all_fold_predictions[best_fold_idx]
        else:
            final_predictions = np.median(all_fold_predictions, axis=0)
                
        return final_predictions, all_fold_predictions


def load_and_preprocess_test_data(csv_path, images_dir, clinical_preprocessors_path=None):
    """
    Load and preprocess test data following the exact same pipeline as training.
    """
    
    # Load CSV
    test_df = pd.read_csv(csv_path)
    
    # Check if survival outcome data is available
    has_survival_data = 'RFS' in test_df.columns and 'Relapse' in test_df.columns

    # Set up image transforms (same as training)
    image_transforms = create_image_transforms()
    
    # Load images
    patient_images = {}
    failed_loads = []
    
    for idx, row in test_df.iterrows():
        patient_id = row["PatientID"]
        
        try:
            ct_path = find_image_path(patient_id, "CT", [images_dir])
            pet_path = find_image_path(patient_id, "PT", [images_dir])
            
            transformed_data = image_transforms({"ct": ct_path, "pet": pet_path})
            combined_image = torch.cat([transformed_data["ct"], transformed_data["pet"]], dim=0)
            patient_images[patient_id] = combined_image
            
        except Exception as e:
            failed_loads.append(patient_id)
    
    # Remove failed loads
    if failed_loads:
        test_df = test_df[~test_df["PatientID"].isin(failed_loads)]
        
    # Initialize ensemble for preprocessing
    ensemble = InferenceModel()
    if clinical_preprocessors_path:
        ensemble.load_clinical_preprocessors(clinical_preprocessors_path)
    
    # Process clinical features
    clinical_features = ensemble.preprocess_test_clinical_data(test_df)
    
    # Handle survival data
    survival_outcomes = {}
    for idx, row in test_df.iterrows():
        patient_id = row["PatientID"]
        if patient_id in patient_images:
            if has_survival_data:
                # Use actual survival data
                survival_outcomes[patient_id] = {
                    'time': float(row["RFS"]),
                    'event': int(row["Relapse"])
                }
    
    # Package data
    test_dataset_cache = {
        'images': patient_images,
        'clinical_features': clinical_features,
        'survival_data': survival_outcomes,
        'dataframe': test_df[test_df["PatientID"].isin(patient_images.keys())],
        'has_survival_data': has_survival_data
    }
    
    return test_dataset_cache

def run_inference(csv_path, images_dir, ensemble_path,
                 clinical_preprocessors_path=None, batch_size=4):
    """
    Complete inference pipeline.
    """
    set_random_seed(RANDOM_SEED)
    
    # Load and preprocess test data
    test_data = load_and_preprocess_test_data(
        csv_path=csv_path,
        images_dir=images_dir,
        clinical_preprocessors_path=clinical_preprocessors_path
    )
    
    # Get patient IDs in order
    test_patient_ids = list(test_data['images'].keys())
    
    # Create dataset and dataloader
    test_dataset = HecktorSurvivalDataset(test_data, test_patient_ids)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    # Load ensemble model
    ensemble = InferenceModel()
    ensemble.load_ensemble_from_single_file(ensemble_path)
    
    # Make predictions
    print("\nRunning inference...")
    final_predictions, all_fold_predictions = ensemble.predict_ensemble(test_loader)
    
    # Calculate C-index if survival data is available
    c_index = None
    if test_data['has_survival_data']:
        
        # Extract true survival outcomes
        true_times = []
        true_events = []
        
        for patient_id in test_patient_ids:
            survival_data = test_data['survival_data'][patient_id]
            true_times.append(survival_data['time'])
            true_events.append(survival_data['event'])
        
        true_times = np.array(true_times)
        true_events = np.array(true_events)
        
        # Calculate C-index (negative predictions because we want higher risk = lower survival)
        c_index = concordance_index(true_times, -final_predictions, true_events)
        
        print(f"C-index: {c_index:.4f}")
        
    return c_index

def main():
    parser = argparse.ArgumentParser(description="HECKTOR survival prediction inference")
    
    parser.add_argument("--csv", required=True,
                       help="Path to test CSV file with patient data")
    parser.add_argument("--images_dir", required=True,
                       help="Directory containing test images (CT and PET)")
    parser.add_argument("--ensemble", required=True,
                       help="Path to ensemble model file (.pt)")
    parser.add_argument("--clinical_preprocessors", required=True, 
                       help="Path to clinical preprocessors file (optional)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    
    if not os.path.exists(args.images_dir):
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    
    if not os.path.exists(args.ensemble):
        raise FileNotFoundError(f"Ensemble model not found: {args.ensemble}")
    
    if not os.path.exists(args.clinical_preprocessors):
        raise FileNotFoundError(f"Clinical Preprocessors not found: {args.clinical_preprocessors}")
    
    c_index = run_inference(
            csv_path=args.csv,
            images_dir=args.images_dir,
            ensemble_path=args.ensemble,
            clinical_preprocessors_path=args.clinical_preprocessors,
            batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
import torch
import csv 
import pickle
import random
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Resized
)
from monai.networks.nets import resnet18

from icare.survival import BaggedIcareSurvival

# =============================================================================
# Configuration and Constants
# =============================================================================

RANDOM_SEED = 42
IMAGE_SIZE = (96, 96, 96)
KNN_NEIGHBORS = 5
CONCORDANCE_EPSILON = 1e-8

# Data paths - update these for your environment
TRAINING_IMAGES_PATH = "./Task_2"
EHR_DATA_PATH = "./HECKTOR_2025_Training_Task_2.csv"
FOLDS_PATH = "./cv_splits_stratified.json"


# =============================================================================
# Utility Functions
# =============================================================================

def set_random_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducible results across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_device():
    """Configure and return the appropriate device for computation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

# =============================================================================
# Loss Functions for Feature Extractor Training
# =============================================================================

class SurvivalContrastiveLoss(nn.Module):
    """
    Contrastive loss for survival analysis that encourages similar survival times
    to have similar representations and different survival times to be separated.
    """
    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, features, survival_times, event_indicators):
        """
        Args:
            features: Extracted features [batch_size, feature_dim]
            survival_times: Survival times [batch_size]
            event_indicators: Event indicators [batch_size]
        """
        batch_size = features.size(0)
        
        # Normalize features
        features = nn.functional.normalize(features, p=2, dim=1)
        
        # Compute pairwise distances
        distance_matrix = torch.cdist(features, features, p=2)
        
        # Create similarity targets based on survival times
        time_diff_matrix = torch.abs(survival_times.unsqueeze(0) - survival_times.unsqueeze(1))
        
        # Similar pairs: small time differences and both have events
        event_matrix = event_indicators.unsqueeze(0) * event_indicators.unsqueeze(1)
        similar_mask = (time_diff_matrix < torch.median(time_diff_matrix)) & (event_matrix > 0)
        
        # Dissimilar pairs: large time differences
        dissimilar_mask = time_diff_matrix > torch.quantile(time_diff_matrix, 0.75)
        
        # Remove diagonal
        eye_mask = torch.eye(batch_size, device=features.device).bool()
        similar_mask = similar_mask & ~eye_mask
        dissimilar_mask = dissimilar_mask & ~eye_mask
        
        loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        
        if similar_mask.sum() > 0:
            # Similar pairs should be close
            similar_distances = distance_matrix[similar_mask]
            similar_loss = similar_distances.mean()
            loss = loss + similar_loss
        
        if dissimilar_mask.sum() > 0:
            # Dissimilar pairs should be far apart
            dissimilar_distances = distance_matrix[dissimilar_mask]
            dissimilar_loss = torch.clamp(self.margin - dissimilar_distances, min=0).mean()
            loss = loss + dissimilar_loss
        
        return loss

class DeepHitLoss(nn.Module):
    """
    DeepHit-style loss combining likelihood and ranking components.
    """
    def __init__(self, ranking_weight=0.2, ranking_scale=1.0):
        super().__init__()
        self.ranking_weight = ranking_weight
        self.ranking_scale = ranking_scale

    def forward(self, risk_scores, survival_times, event_indicators):
        if event_indicators.sum() == 0:
            return torch.tensor(0.01, device=risk_scores.device, requires_grad=True)
        
        device = risk_scores.device
        batch_size = len(survival_times)
        
        # Likelihood component
        sorted_indices = torch.argsort(survival_times, descending=True)
        sorted_risk_scores = risk_scores[sorted_indices]
        sorted_events = event_indicators[sorted_indices]
        
        likelihood_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for i in range(batch_size):
            if sorted_events[i] == 1:
                risk_set_scores = sorted_risk_scores[i:]
                numerator = sorted_risk_scores[i]
                denominator = torch.logsumexp(risk_set_scores, dim=0)
                likelihood_loss = likelihood_loss + numerator - denominator
        
        likelihood_loss = -likelihood_loss / (event_indicators.sum() + CONCORDANCE_EPSILON)
        
        # Ranking component
        ranking_loss = torch.tensor(0.0, device=device, requires_grad=True)
        pair_count = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if event_indicators[i] == 1 and survival_times[i] < survival_times[j]:
                    risk_difference = risk_scores[j] - risk_scores[i]
                    ranking_loss = ranking_loss + torch.exp(self.ranking_scale * risk_difference)
                    pair_count += 1
                elif event_indicators[j] == 1 and survival_times[j] < survival_times[i]:
                    risk_difference = risk_scores[i] - risk_scores[j]
                    ranking_loss = ranking_loss + torch.exp(self.ranking_scale * risk_difference)
                    pair_count += 1
        
        if pair_count > 0:
            ranking_loss = ranking_loss / pair_count
        
        return likelihood_loss + self.ranking_weight * ranking_loss

# =============================================================================
# Dataset Class
# =============================================================================

class HecktorSurvivalDataset(Dataset):
    """Dataset class for pre-loaded HECKTOR survival data."""
    def __init__(self, cached_data, patient_ids):
        self.patient_ids = [pid for pid in patient_ids if pid in cached_data['images']]
        self.images = cached_data['images']
        
        self.clinical_features = cached_data['clinical_features']['features']
            
        self.survival_data = cached_data['survival_data']
        
        print(f"Dataset initialized with {len(self.patient_ids)} patients")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        image_tensor = self.images[patient_id]
        clinical_tensor = torch.tensor(self.clinical_features[patient_id], dtype=torch.float32)
        survival_time = torch.tensor(self.survival_data[patient_id]['time'], dtype=torch.float32)
        event_indicator = torch.tensor(self.survival_data[patient_id]['event'], dtype=torch.float32)
        
        return image_tensor, clinical_tensor, survival_time, event_indicator

# =============================================================================
# BaggedIcareSurvival Model
# =============================================================================

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

class HecktorSurvivalModel:
    """
    Complete system that jointly trains feature extractor and BaggedIcareSurvival.
    Uses iterative optimization to improve both components together.
    """
    
    def __init__(self, clinical_feature_dim, device, feature_dim=128):
        self.device = device
        self.feature_dim = feature_dim
        self.clinical_feature_dim = clinical_feature_dim
        
        # Initialize feature extractor
        self.feature_extractor = FusedFeatureExtractor(
            clinical_feature_dim, feature_dim
        ).to(device)
        
        # BaggedIcareSurvival model (will be initialized during training)
        self.icare_model = None
        
        # Training components
        self.optimizer = optim.Adam(self.feature_extractor.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Loss functions
        self.survival_loss = DeepHitLoss(ranking_weight=0.3)
        self.contrastive_loss = SurvivalContrastiveLoss(margin=2.0, temperature=0.1)
        
        # Training state
        self.best_c_index = 0.0
        self.best_feature_state = None
        self.best_icare_model = None
        
    def extract_features_and_targets(self, data_loader):
        """Extract features using current feature extractor."""
        self.feature_extractor.eval()
        all_features = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for images, clinical, times, events in data_loader:
                images, clinical = images.to(self.device), clinical.to(self.device)
                features = self.feature_extractor(images, clinical)
                
                all_features.append(features.cpu().numpy())
                all_times.extend(times.numpy())
                all_events.extend(events.numpy())
        
        feature_matrix = np.vstack(all_features)
        time_array = np.array(all_times)
        event_array = np.array(all_events)
        
        # Create structured array for BaggedIcareSurvival
        # Format: [(event, time), (event, time), ...]
        # Note: BaggedIcareSurvival expects event as first field, time as second
        survival_dtype = [('event', bool), ('time', float)]
        survival_outcomes = np.array(
            list(zip(event_array.astype(bool), time_array.astype(float))), 
            dtype=survival_dtype
        )
        
        return feature_matrix, survival_outcomes
    

    def train_feature_extractor_epoch(self, train_loader):
        """Train feature extractor for one epoch using survival losses."""
        self.feature_extractor.train()
        total_loss = 0
        batch_count = 0

        epoch_pbar = tqdm(train_loader, desc="Training feature extractor", leave=False)
        
        for images, clinical, times, events in epoch_pbar:
            images, clinical = images.to(self.device), clinical.to(self.device)
            times, events = times.to(self.device), events.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Get features and risk scores
            features, risk_scores = self.feature_extractor(
                images, clinical, return_risk=True
            )
            
            # Compute combined loss
            survival_loss = self.survival_loss(risk_scores, times, events)
            contrastive_loss = self.contrastive_loss(features, times, events)
            
            total_loss_batch = survival_loss + 0.1 * contrastive_loss
            
            if not torch.isnan(total_loss_batch):
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.feature_extractor.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += total_loss_batch.item()
                batch_count += 1

            epoch_pbar.set_postfix(loss=total_loss_batch.item())
        
        return total_loss / max(batch_count, 1)
    
    def train_icare_model(self, train_loader):
        """Train or retrain BaggedIcareSurvival model with current features."""        
        # Extract features using current feature extractor
        X_train, y_train = self.extract_features_and_targets(train_loader)
        
        
        try:
            # Initialize BaggedIcareSurvival with conservative parameters
            # Check what parameters are actually supported
            self.icare_model = BaggedIcareSurvival(
                aggregation_method='median',
                n_jobs=-1,
                random_state=RANDOM_SEED
            )
            
            self.icare_model.fit(X_train, y_train)
            
        except Exception as e:
            print(f"BaggedIcareSurvival training failed: {e}")
            raise e
    
    def evaluate_system(self, data_loader):
        """Evaluate the complete system."""
        try:
            # Extract features and get predictions
            X_test, y_test = self.extract_features_and_targets(data_loader)
            predictions = self.icare_model.predict(X_test)
            
            # Extract time and event from structured array for concordance calculation
            times = y_test['time']
            events = y_test['event'].astype(int)
            
            # Calculate C-index (negative predictions because icare gives risk scores)
            c_index = concordance_index(times, -predictions, events)
            return c_index
            
        except Exception as e:
            print(f"System evaluation failed: {e}")
            raise e
    
    def fit(self, train_loader, val_loader, num_iterations=20, feature_epochs_per_iteration=5):
        """
        Train the complete system using iterative optimization.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            num_iterations: Number of alternating training iterations
            feature_epochs_per_iteration: Feature extractor epochs per iteration
        """
        print(f"Training: {num_iterations} iterations × {feature_epochs_per_iteration} epochs")
        
        # Initial BaggedIcareSurvival training
        self.train_icare_model(train_loader)
        initial_c_index = self.evaluate_system(val_loader)
        print(f"Initial validation C-index: {initial_c_index:.4f}")
        
        self.best_c_index = initial_c_index
        self.best_feature_state = self.feature_extractor.state_dict().copy()
        self.best_icare_model = pickle.loads(pickle.dumps(self.icare_model))
        
        # Iterative joint training
        for iteration in tqdm(range(num_iterations), desc="Training iterations"):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            
            # Train feature extractor for several epochs
            print("Training feature extractor...")
            epoch_losses = []
            
            epoch_pbar = tqdm(range(feature_epochs_per_iteration), desc="Feature extractor epochs", leave=False)
            #add loss in tqdm
            for epoch in epoch_pbar:
                avg_loss = self.train_feature_extractor_epoch(train_loader)
                epoch_losses.append(avg_loss)   
            
            # Retrain BaggedIcareSurvival with updated features
            self.train_icare_model(train_loader)
            
            # Evaluate complete system
            current_c_index = self.evaluate_system(val_loader)
            print(f"Validation C-index: {current_c_index:.4f}")
            
            # Update learning rate based on performance
            self.scheduler.step(current_c_index)
            
            # Save best model
            if current_c_index > self.best_c_index:
                print(f"New best C-index: {current_c_index:.4f} (improvement: +{current_c_index - self.best_c_index:.4f})")
                self.best_c_index = current_c_index
                self.best_feature_state = self.feature_extractor.state_dict().copy()
                self.best_icare_model = pickle.loads(pickle.dumps(self.icare_model))
            else:
                print(f"No improvement (best: {self.best_c_index:.4f})")

            #update tqdm with loss, c-index and best c-index
            epoch_pbar.set_postfix(loss=avg_loss, c_index=current_c_index, best_c_index=self.best_c_index)
        
        # Load best models
        print(f"\nTraining complete! Best validation C-index: {self.best_c_index:.4f}")
        self.feature_extractor.load_state_dict(self.best_feature_state)
        self.icare_model = self.best_icare_model
    
    def predict(self, data_loader):
        """Generate predictions using the trained system."""
        if self.icare_model is None:
            raise ValueError("System must be trained before making predictions")
        
        X_test, y_test = self.extract_features_and_targets(data_loader)
        predictions = self.icare_model.predict(X_test)
        
        # Extract time and event from structured array
        times = y_test['time']
        events = y_test['event'].astype(int)
        
        return predictions, times, events  # predictions, times, events
    
    def save_model(self, filepath_prefix):
        """Save the complete trained system with config for inference."""
        # Save feature extractor
        torch.save(self.feature_extractor.state_dict(), f"{filepath_prefix}_feature_extractor.pt")
        
        # Save BaggedIcareSurvival model
        with open(f"{filepath_prefix}_icare_model.pkl", 'wb') as f:
            pickle.dump(self.icare_model, f)
        
        # Save model configuration for inference
        config = {
            'clinical_feature_dim': self.clinical_feature_dim,
            'feature_dim': self.feature_dim,
            'model_type': 'HecktorSurvivalModel',
            'image_size': IMAGE_SIZE,
            'best_c_index': float(self.best_c_index)
        }
        with open(f"{filepath_prefix}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved with prefix: {filepath_prefix}")
    
    def load_model(self, filepath_prefix):
        """Load a previously trained system."""
        # Load feature extractor
        self.feature_extractor.load_state_dict(
            torch.load(f"{filepath_prefix}_feature_extractor.pt", map_location=self.device)
        )
        
        # Load BaggedIcareSurvival model
        with open(f"{filepath_prefix}_icare_model.pkl", 'rb') as f:
            self.icare_model = pickle.load(f)
        
        # Load config if available
        config_path = f"{filepath_prefix}_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded model with C-index: {config.get('best_c_index', 'unknown')}")
        
        print(f"System loaded from prefix: {filepath_prefix}")

# =============================================================================
# Data Loading Functions (Updated)
# =============================================================================

def create_image_transforms():
    """Create MONAI transforms for CT and PET image preprocessing."""
    transforms = Compose([
        LoadImaged(keys=["ct","pet"]),
        EnsureChannelFirstd(keys=["ct", "pet"]),
        ScaleIntensityd(keys=["ct","pet"]),
        Resized(keys=["ct", "pet"], spatial_size=IMAGE_SIZE), 
        ToTensord(keys=["ct","pet"]),
    ])
    return transforms

def find_image_path(patient_id, modality, directories):
    """Find the file path for a specific patient and imaging modality."""
    for directory in directories:
        filename = f"{patient_id}__{modality}.nii.gz"
        full_path = os.path.join(directory, filename)
        if os.path.exists(full_path):
            return full_path
    raise FileNotFoundError(f"{modality} image for patient {patient_id} not found")


def preprocess_clinical_data(dataframe):
    """Preprocess clinical features with one-hot encoding and proper NaN handling."""
    
    # All clinical features used in preprocessing
    ALL_CLINICAL_FEATURES = [
        "Age", "Gender", "Tobacco Consumption", "Alcohol Consumption", 
        "Performance Status", "M-stage", "Treatment"
    ]
    
    # Categorical features for one-hot encoding (all except Age)
    CATEGORICAL_FEATURES = [
        "Gender", "Tobacco Consumption", "Alcohol Consumption", 
        "Performance Status", "M-stage", "Treatment"
    ]
    
    feature_subset = dataframe[ALL_CLINICAL_FEATURES].copy()
    
    # Handle Age (continuous variable)
    # Fill NaN values with median
    age_median = feature_subset["Age"].median()
    feature_subset["Age"] = feature_subset["Age"].fillna(age_median)
    
    # Standardize Age
    age_scaler = StandardScaler()
    age_scaled = age_scaler.fit_transform(feature_subset[["Age"]])
    
    # Handle categorical features with one-hot encoding
    # Fill NaN values with 'Unknown' category for each categorical feature
    categorical_data = feature_subset[CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        categorical_data[col] = categorical_data[col].fillna('Unknown')
        # Convert to string to ensure consistent data type
        categorical_data[col] = categorical_data[col].astype(str)
    
    # Apply one-hot encoding
    # Use drop_first=False to keep all categories (including Unknown)
    categorical_encoded = pd.get_dummies(
        categorical_data, 
        columns=CATEGORICAL_FEATURES,
        prefix=CATEGORICAL_FEATURES,
        dummy_na=False,  # We already handled NaN by filling with 'Unknown'
        drop_first=False  # Keep all categories for completeness
    )
    
    # Process all patients
    processed_features = {}
    
    for idx, row in dataframe.iterrows():
        patient_id = row["PatientID"]
        patient_row_idx = dataframe.index.get_loc(idx)
        
        # Get standardized age for this patient
        age_features = age_scaled[patient_row_idx].flatten()
        
        # Get one-hot encoded categorical features for this patient
        categorical_features = categorical_encoded.iloc[patient_row_idx].values
        
        # Combine all features
        complete_features = np.concatenate([age_features, categorical_features])
        processed_features[patient_id] = complete_features
    
    # Store preprocessors for inference
    preprocessors = {
        'age_scaler': age_scaler,
        'age_median': age_median,
        'categorical_columns': list(categorical_encoded.columns),
        'feature_names': ['Age'] + list(categorical_encoded.columns),
        'n_features': len(complete_features)
    }
    
    print(f"Preprocessed features shape: {len(complete_features)} features per patient")
    print(f"Age feature: 1 (standardized)")
    print(f"Categorical features: {len(categorical_encoded.columns)} (one-hot encoded)")
    print(f"Feature breakdown:")
    for i, feature_name in enumerate(preprocessors['feature_names']):
        print(f"  {i}: {feature_name}")
    
    return {
        'features': processed_features,
        'preprocessors': preprocessors
    }

def load_and_cache_dataset(csv_path, image_directories, cache_path="cached_hecktor_data.pkl"):
    """Load and cache all data for faster access. Now also saves preprocessors for inference."""
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Loading and preprocessing dataset...")
    clinical_df = pd.read_csv(csv_path)
    
    # Set up image transforms
    image_transforms = create_image_transforms()
    
    # Load all medical images
    patient_images = {}
    failed_loads = []
    
    for idx, row in tqdm(clinical_df.iterrows(), total=len(clinical_df), desc="Loading images"):
        patient_id = row["PatientID"]
        
        try:
            ct_path = find_image_path(patient_id, "CT", image_directories)
            pet_path = find_image_path(patient_id, "PT", image_directories)
            
            transformed_data = image_transforms({"ct": ct_path, "pet": pet_path})
            combined_image = torch.cat([transformed_data["ct"], transformed_data["pet"]], dim=0)
            patient_images[patient_id] = combined_image
            
        except Exception as e:
            print(f"Failed to load images for {patient_id}: {e}")
            failed_loads.append(patient_id)
    
    # Remove failed loads from clinical data
    if failed_loads:
        print(f"Excluding {len(failed_loads)} patients due to missing images")
        clinical_df = clinical_df[~clinical_df["PatientID"].isin(failed_loads)]
    
    # Process clinical features (now uses one-hot encoding)
    clinical_features = preprocess_clinical_data(clinical_df)
    
    # Save clinical preprocessors separately for inference
    preprocessors_path = cache_path.replace('.pkl', '_clinical_preprocessors.pkl')
    with open(preprocessors_path, 'wb') as f:
        pickle.dump(clinical_features['preprocessors'], f)
    print(f"Clinical preprocessors saved to {preprocessors_path}")
    
    # Prepare survival data
    survival_outcomes = {}
    for idx, row in clinical_df.iterrows():
        patient_id = row["PatientID"]
        if patient_id in patient_images:
            survival_outcomes[patient_id] = {
                'time': float(row["RFS"]),
                'event': int(row["Relapse"])
            }
    
    # Package data
    cached_dataset = {
        'images': patient_images,
        'clinical_features': clinical_features,
        'survival_data': survival_outcomes,
        'dataframe': clinical_df[clinical_df["PatientID"].isin(patient_images.keys())]
    }
    
    # Save cache
    print(f"Saving preprocessed data to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_dataset, f)
    
    print(f"Successfully processed {len(patient_images)} patients")
    print(f"Each patient has {clinical_features['preprocessors']['n_features']} clinical features")
    return cached_dataset

# =============================================================================
# Cross-Validation 
# =============================================================================

def load_cv_folds(json_path):
    """Load cross-validation folds from JSON."""
    with open(json_path, 'r') as f:
        fold_config = json.load(f)
    print(f"Loaded {fold_config['n_folds']} CV folds from {json_path}")
    return fold_config

def run_cross_validation(cached_data, folds_json_path=None, batch_size=4, 
                                  test_proportion=0.2, training_iterations=15, 
                                  feature_epochs_per_iteration=3):
    """
    Run cross-validation using the BaggedIcareSurvival model.
    """
    # Setup logging
    os.makedirs("cv_logs", exist_ok=True)
    results_path = "cv_logs/cv_results.csv"
    
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Validation_CIndex", "Test_CIndex"])

    # Prepare data
    clinical_df = cached_data['dataframe']
    all_patient_ids = clinical_df["PatientID"].values
    all_events = clinical_df["Relapse"].values
    device = setup_device()
    
    # Handle fold configuration
    if folds_json_path and os.path.exists(folds_json_path):
        fold_config = load_cv_folds(folds_json_path)
        cv_folds = fold_config['folds']
        num_folds = fold_config['n_folds']
        
        # Set up test set if needed
        if test_proportion > 0:
            fold_patient_ids = set()
            for fold in cv_folds:
                fold_patient_ids.update(fold['train'])
                fold_patient_ids.update(fold['val'])
            
            remaining_patients = [pid for pid in all_patient_ids if pid not in fold_patient_ids]
            
            if len(remaining_patients) > 0:
                test_patient_ids = np.array(remaining_patients)
                print(f"Using {len(test_patient_ids)} remaining patients as test set")
            else:
                train_val_ids, test_patient_ids, _, _ = train_test_split(
                    all_patient_ids, all_events, 
                    test_size=test_proportion, stratify=all_events, random_state=RANDOM_SEED
                )
                print(f"Held out {len(test_patient_ids)} patients as test set")
        else:
            test_patient_ids = []
    else:
        print("Generating stratified CV folds...")
        train_val_ids, test_patient_ids, train_val_events, _ = train_test_split(
            all_patient_ids, all_events, 
            test_size=test_proportion, stratify=all_events, random_state=RANDOM_SEED
        )
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_folds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_ids, train_val_events)):
            cv_folds.append({
                'fold': fold_idx + 1,
                'train': train_val_ids[train_idx].tolist(),
                'val': train_val_ids[val_idx].tolist()
            })
        
        num_folds = 5

    # Create test loader if needed
    if len(test_patient_ids) > 0:
        test_dataset = HecktorSurvivalDataset(cached_data, test_patient_ids)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    # Store results
    validation_c_indices = []
    test_c_indices = []
    
    # Process each fold
    for fold_info in cv_folds:
        fold_num = fold_info['fold'] - 1
        train_pids = [pid for pid in fold_info['train'] if pid in cached_data['images']]
        val_pids = [pid for pid in fold_info['val'] if pid in cached_data['images']]
        
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num + 1}/{num_folds}")
        print(f"{'='*60}")
        print(f"Training patients: {len(train_pids)}")
        print(f"Validation patients: {len(val_pids)}")
        
        # Create datasets and loaders
        train_dataset = HecktorSurvivalDataset(cached_data, train_pids)
        val_dataset = HecktorSurvivalDataset(cached_data, val_pids)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Get clinical feature dimension
        clinical_dim = train_dataset[0][1].shape[0]
        
        # Initialize model
        model = HecktorSurvivalModel(
            clinical_feature_dim=clinical_dim,
            device=device,
            feature_dim=256
        )
        
        # Train the complete system
        print(f"Training...")
        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_iterations=training_iterations,
            feature_epochs_per_iteration=feature_epochs_per_iteration
        )
        
        # Final evaluation
        val_c_index = model.evaluate_system(val_loader)
        
        if test_loader:
            test_c_index = model.evaluate_system(test_loader)
            test_predictions, test_times, test_events = model.predict(test_loader)
            eval_patient_ids = test_patient_ids
        else:
            test_c_index = val_c_index
            test_predictions, test_times, test_events = model.predict(val_loader)
            eval_patient_ids = val_pids
        
        # Store results
        validation_c_indices.append(val_c_index)
        test_c_indices.append(test_c_index)
        
        # Save predictions
        results_df = pd.DataFrame({
            'patient_id': eval_patient_ids,
            'risk_score': test_predictions,
            'survival_time': test_times,
            'event_indicator': test_events
        })
        results_df.to_csv(f"cv_logs/fold{fold_num}_predictions.csv", index=False)
        
        # Save trained system (now includes config)
        model.save_model(f"cv_logs/fold{fold_num}_system")
        
        print(f"Fold {fold_num + 1} Results:")
        print(f"  Validation C-index: {val_c_index:.4f}")
        print(f"  Test C-index: {test_c_index:.4f}")
        
        # Log to CSV
        with open(results_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([fold_num, f"{val_c_index:.4f}", f"{test_c_index:.4f}"])

    # Final summary
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Validation C-indices: {[f'{x:.4f}' for x in validation_c_indices]}")
    print(f"Mean validation C-index: {np.mean(validation_c_indices):.4f} ± {np.std(validation_c_indices):.4f}")
    print(f"Test C-indices: {[f'{x:.4f}' for x in test_c_indices]}")
    print(f"Mean test C-index: {np.mean(test_c_indices):.4f} ± {np.std(test_c_indices):.4f}")
    
    return validation_c_indices, test_c_indices

# =============================================================================
# Main Function (Updated)
# =============================================================================

def main():
    """Main function for task 2 survival prognosis with inference support."""
    
    set_random_seed(RANDOM_SEED)
    
    # Load and cache dataset (now also saves preprocessors)
    print("Loading dataset...")
    dataset_cache = load_and_cache_dataset(
        csv_path=EHR_DATA_PATH,
        image_directories=[TRAINING_IMAGES_PATH],
        cache_path="hecktor_cache.pkl"
    )
    
    print(f"\nDataset Summary:")
    print(f"- Total patients: {len(dataset_cache['images'])}")
        
    # Run cross-validation    
    print(f"\nStarting cross-validation...")
    
    validation_results, test_results = run_cross_validation(
        cached_data=dataset_cache,
        folds_json_path=FOLDS_PATH if os.path.exists(FOLDS_PATH) else None,
        batch_size=6,
        test_proportion=0.2,
        training_iterations=25,  # Number of alternating training cycles
        feature_epochs_per_iteration=10  # Feature extractor epochs per cycle
    )
    
    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Validation C-index: {np.mean(validation_results):.4f} ± {np.std(validation_results):.4f}")
    print(f"Test C-index: {np.mean(test_results):.4f} ± {np.std(test_results):.4f}")
    
    # Find best fold and demonstrate inference setup
    best_fold_idx = np.argmax(validation_results)
    print(f"\nBest fold: {best_fold_idx} (C-index: {validation_results[best_fold_idx]:.4f})")

if __name__ == "__main__":
    main()
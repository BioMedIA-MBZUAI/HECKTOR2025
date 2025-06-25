import pandas as pd
import numpy as np
import json
import os
import time
from pathlib import Path
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import warnings
from tqdm import tqdm
from scipy.ndimage import zoom
from monai.networks.nets import ResNet
from monai.networks.layers import Norm
import pickle

# Filter out warnings
warnings.filterwarnings('ignore')

# Set all random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
import random
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Check for GPU/CPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

class CoxPHLoss(nn.Module):
    """Cox Proportional Hazards Loss for survival analysis"""
    def __init__(self):
        super(CoxPHLoss, self).__init__()
        
    def forward(self, risk_scores, durations, events):
        """
        Args:
            risk_scores: predicted risk scores (higher = higher risk)
            durations: survival times
            events: event indicators (1 if event occurred, 0 if censored)
        """
        # Ensure tensors have proper dimensions
        if risk_scores.dim() == 0:
            risk_scores = risk_scores.unsqueeze(0)
        if durations.dim() == 0:
            durations = durations.unsqueeze(0)
        if events.dim() == 0:
            events = events.unsqueeze(0)
        
        # Flatten to 1D if needed
        risk_scores = risk_scores.flatten()
        durations = durations.flatten()
        events = events.flatten()
        
        # Handle single sample case
        if len(risk_scores) == 1:
            # For single sample, return simple loss
            return torch.abs(risk_scores).mean()
        
        # Sort by survival time (descending)
        sorted_indices = torch.argsort(durations, descending=True)
        risk_scores = risk_scores[sorted_indices]
        durations = durations[sorted_indices]
        events = events[sorted_indices]
        
        # Cox partial likelihood
        hazard_ratio = torch.exp(risk_scores)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-7)
        
        # Only consider patients who had events
        uncensored_likelihood = risk_scores - log_risk
        censored_likelihood = uncensored_likelihood * events
        
        # Negative log likelihood (we want to minimize this)
        neg_likelihood = -torch.sum(censored_likelihood) / (torch.sum(events) + 1e-7)
        
        return neg_likelihood

class SpatialAttention3D(nn.Module):
    """Enhanced 3D Spatial Attention Module with multi-scale features"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv1(attention)
        attention = self.conv2(attention)  
        return x * self.sigmoid(attention)

class ChannelAttention3D(nn.Module):
    """Enhanced 3D Channel Attention Module with better feature modeling"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.gate = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attention = self.sigmoid(avg_out + max_out)
        
        gated = torch.sigmoid(self.gate(x))
        return x * attention * gated

class HECKTORAdvancedResNet(nn.Module):
    """Advanced MONAI ResNet50 with enhanced attention for HECKTOR"""
    
    def __init__(self, input_channels=1, feature_dim=1024, spatial_dims=3):
        super(HECKTORAdvancedResNet, self).__init__()
        self.backbone = ResNet(
            block='bottleneck',  
            layers=[3, 4, 6, 3], 
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=spatial_dims,
            n_input_channels=input_channels,
            num_classes=feature_dim,
            norm=Norm.BATCH,
            bias_downsample=False
        )
        
        self.backbone.fc = nn.Identity()
        
        self.channel_attention1 = ChannelAttention3D(512, reduction=16)  
        self.channel_attention2 = ChannelAttention3D(2048, reduction=16) 
        self.spatial_attention1 = SpatialAttention3D(kernel_size=7)
        self.spatial_attention2 = SpatialAttention3D(kernel_size=5)
        
        self.feature_fusion = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),  
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048 * 4 * 4 * 4, feature_dim * 2),  
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim * 2),  
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.feature_enhancer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim),  
            nn.Dropout(0.1)
        )
        
        self._init_attention_weights()
        
    def _init_attention_weights(self):
        """Initialize attention module weights"""
        for m in [self.channel_attention1, self.channel_attention2, 
                  self.spatial_attention1, self.spatial_attention2]:
            for module in m.modules():
                if isinstance(module, nn.Conv3d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(module, nn.BatchNorm3d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = F.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        if x.size(1) >= 256:
            x = self.channel_attention1(x)
            x = self.spatial_attention1(x)
        
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        if x.size(1) >= 512:
            x = self.channel_attention2(x)
            x = self.spatial_attention2(x)
        
        output = self.feature_fusion(x)
        output = self.feature_enhancer(output)
            
        return output

class HECKTORFeatureExtractor(nn.Module):
    """Feature extractor that outputs combined features for CoxPH model"""
    
    def __init__(self, feature_dim=128, ehr_input_dim=100):
        super(HECKTORFeatureExtractor, self).__init__()
        
        self.ct_resnet = HECKTORAdvancedResNet(
            input_channels=1, 
            feature_dim=feature_dim,
            spatial_dims=3
        )
        
        self.pt_resnet = HECKTORAdvancedResNet(
            input_channels=1, 
            feature_dim=feature_dim,
            spatial_dims=3
        )
        
        # EHR feature processor
        self.ehr_processor = nn.Sequential(
            nn.Linear(ehr_input_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Feature fusion (without final risk score layer)
        total_features = feature_dim * 2 + feature_dim // 2  # CT + PT + EHR
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.output_dim = feature_dim // 2  # Final feature dimension
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize fusion layer weights"""
        for m in [self.ehr_processor, self.fusion_layer]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, ct_images, pt_images, ehr_features):
        """
        Returns combined features instead of risk scores
        """
        # Extract image features - GRADIENTS FLOW THROUGH HERE
        ct_features = self.ct_resnet(ct_images)
        pt_features = self.pt_resnet(pt_images)
        
        # Process EHR features
        ehr_processed = self.ehr_processor(ehr_features)
        
        # Concatenate all features
        combined_features = torch.cat([ct_features, pt_features, ehr_processed], dim=1)
        
        # Final feature fusion
        final_features = self.fusion_layer(combined_features)
        
        return final_features

class HECKTORImageCache:
    """Image cache that loads all images once and reuses them"""
    
    def __init__(self, patient_ids, image_dir, target_size=(96, 96, 96)):
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.cache = {}
        
        self._load_all_images(patient_ids)
    
    def _load_all_images(self, patient_ids):
        """Load and preprocess all images once"""
        
        for patient_id in tqdm(patient_ids, desc="Loading all images"):
            ct_tensor, pt_tensor = self._load_patient_images(patient_id)
            self.cache[patient_id] = {
                'ct': ct_tensor,
                'pt': pt_tensor
            }
    
    def _load_patient_images(self, patient_id):
        """Load and preprocess CT and PT images for a patient"""
        
        # Load CT image
        ct_path = self.image_dir / f"{patient_id}__CT.nii.gz"
        if ct_path.exists():
            try:
                ct_img = nib.load(str(ct_path))
                ct_data = ct_img.get_fdata()
                ct_tensor = self._preprocess_image(ct_data)
            except Exception as e:
                print(f"Warning: Error loading CT for {patient_id}: {e}")
                ct_tensor = torch.zeros(1, *self.target_size, dtype=torch.float32)
        else:
            ct_tensor = torch.zeros(1, *self.target_size, dtype=torch.float32)
        
        # Load PT image
        pt_path = self.image_dir / f"{patient_id}__PT.nii.gz"
        if pt_path.exists():
            try:
                pt_img = nib.load(str(pt_path))
                pt_data = pt_img.get_fdata()
                pt_tensor = self._preprocess_image(pt_data)
            except Exception as e:
                print(f"Warning: Error loading PT for {patient_id}: {e}")
                pt_tensor = torch.zeros(1, *self.target_size, dtype=torch.float32)
        else:
            pt_tensor = torch.zeros(1, *self.target_size, dtype=torch.float32)
        
        return ct_tensor, pt_tensor
    
    def _preprocess_image(self, image_data):
        """Preprocess medical images"""
        # Handle NaN values
        if np.isnan(image_data).any():
            image_data = np.nan_to_num(image_data, nan=0.0)
        
        # Resize to target size
        current_shape = image_data.shape
        zoom_factors = [t/c for t, c in zip(self.target_size, current_shape)]
        resized_image = zoom(image_data, zoom_factors, order=3)
        
        # Normalization
        if np.any(resized_image > 0):
            # Percentile-based robust normalization
            p1, p99 = np.percentile(resized_image[resized_image > 0], [1, 99])
            if p99 > p1:
                resized_image = np.clip((resized_image - p1) / (p99 - p1), 0, 1)
            
            # Z-score normalization
            non_zero_mask = resized_image > 0
            if np.any(non_zero_mask):
                mean_val = np.mean(resized_image[non_zero_mask])
                std_val = np.std(resized_image[non_zero_mask])
                if std_val > 0:
                    resized_image[non_zero_mask] = (resized_image[non_zero_mask] - mean_val) / std_val
                    resized_image = (resized_image - resized_image.min()) / (resized_image.max() - resized_image.min() + 1e-8)
        
        # Convert to tensor
        tensor_image = torch.FloatTensor(resized_image).unsqueeze(0)  # Add channel dimension
        
        return tensor_image
    
    def get_images(self, patient_id):
        """Get cached images for a patient"""
        if patient_id in self.cache:
            return self.cache[patient_id]['ct'], self.cache[patient_id]['pt']
        else:
            # Return zero tensors if not found
            return (torch.zeros(1, *self.target_size, dtype=torch.float32),
                    torch.zeros(1, *self.target_size, dtype=torch.float32))
    
    def clear_cache(self):
        """Clear the image cache to free memory"""
        self.cache.clear()

class HECKTORDataset(Dataset):
    """Dataset that uses pre-loaded image cache"""
    
    def __init__(self, patient_ids, image_cache, ehr_features, durations, events):
        self.patient_ids = patient_ids
        self.image_cache = image_cache
        self.ehr_features = ehr_features
        self.durations = durations
        self.events = events
        
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        ct_tensor, pt_tensor = self.image_cache.get_images(patient_id)
        
        # Get EHR features
        ehr_tensor = torch.FloatTensor(self.ehr_features[idx])
        
        return {
            'patient_id': patient_id,
            'ct_image': ct_tensor,
            'pt_image': pt_tensor,
            'ehr_features': ehr_tensor,
            'duration': torch.FloatTensor([self.durations[idx]]),
            'event': torch.FloatTensor([self.events[idx]])
        }

def train_feature_extractor_with_coxph(model, train_loader, val_loader, epochs=10, lr=1e-4, fold_num=1, save_dir="./checkpoints"):
    """
    Train feature extractor with CoxPH model 
    
    This function:
    1. Extracts features using the neural network
    2. Trains CoxPH on those features
    3. Uses CoxPH performance to update neural network weights via backpropagation
    4. GRADIENTS FLOW ALL THE WAY BACK TO RESNET LAYERS
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Optimizer for the neural network (includes ResNet parameters)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Cox loss for neural network updates
    cox_loss_fn = CoxPHLoss()
    
    best_val_c_index = 0.0
    best_epoch = 0
    
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # ===============================================
        # TRAINING PHASE
        # ===============================================
        model.train()
        
        # Collect features and targets for CoxPH training
        all_train_features = []
        all_train_durations = []
        all_train_events = []
        train_losses = []
        
        print(f"Epoch {epoch+1}/{epochs}: Extracting training features...")
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            ct_images = batch['ct_image'].to(device)
            pt_images = batch['pt_image'].to(device)
            ehr_features = batch['ehr_features'].to(device)
            durations = batch['duration'].to(device)
            events = batch['event'].to(device)
            
            # FORWARD PASS - GRADIENTS ARE TRACKED HERE
            features = model(ct_images, pt_images, ehr_features)
            
            # Store for CoxPH training
            all_train_features.append(features.detach().cpu().numpy())
            all_train_durations.extend(durations.cpu().numpy().flatten())
            all_train_events.extend(events.cpu().numpy().flatten())
            
            # UPDATE NEURAL NETWORK WEIGHTS EVERY FEW BATCHES
            if batch_idx % 3 == 0:  # Update every 3 batches
                optimizer.zero_grad()
                
                # Create a simple risk score for Cox loss
                # This ensures gradients flow back through the entire network
                risk_scores = torch.sum(features, dim=1)  # Simple linear combination
                
                # Calculate Cox loss
                loss = cox_loss_fn(risk_scores, durations.flatten(), events.flatten())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
        
        # Combine all training features
        X_train_combined = np.vstack(all_train_features)
        y_train_durations = np.array(all_train_durations)
        y_train_events = np.array(all_train_events).astype(bool)
        
        # Create structured array for CoxPH
        y_train_structured = Surv.from_arrays(event=y_train_events, time=y_train_durations)
        
        # Train CoxPH model on extracted features
        cox_model = CoxPHSurvivalAnalysis(alpha=0.001)
        
        try:
            cox_model.fit(X_train_combined, y_train_structured)
            cox_training_successful = True
        except Exception as e:
            print(f"Warning: CoxPH fitting failed - {e}")
            cox_training_successful = False
            cox_model = None
        
        # ===============================================
        # VALIDATION PHASE
        # ===============================================
        model.eval()

        all_val_features = []
        all_val_durations = []
        all_val_events = []

        print(f"Epoch {epoch+1}/{epochs}: Extracting validation features...")

        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                ct_images = batch['ct_image'].to(device)
                pt_images = batch['pt_image'].to(device)
                ehr_features = batch['ehr_features'].to(device)
                durations = batch['duration'].to(device)  # Keep on GPU
                events = batch['event'].to(device)        # Keep on GPU
                
                # Extract features (no gradients needed)
                features = model(ct_images, pt_images, ehr_features)
                
                # Calculate risk scores for validation loss (same as training)
                risk_scores = torch.sum(features, dim=1)  # Simple linear combination
                
                # Calculate validation loss
                val_loss = cox_loss_fn(risk_scores, durations.flatten(), events.flatten())
                val_losses.append(val_loss.item())
                
                # Store for CoxPH evaluation (convert to CPU/numpy here)
                all_val_features.append(features.cpu().numpy())
                all_val_durations.extend(durations.cpu().numpy().flatten())
                all_val_events.extend(events.cpu().numpy().flatten())
 

        # Combine validation features
        X_val_combined = np.vstack(all_val_features)
        y_val_durations = np.array(all_val_durations)
        y_val_events = np.array(all_val_events).astype(bool)
        
        # Evaluate with CoxPH
        val_c_index = 0.0
        if cox_training_successful and cox_model is not None:
            try:
                val_risk_scores = cox_model.predict(X_val_combined)

                val_c_index = concordance_index_censored(y_val_events, y_val_durations, val_risk_scores)[0]
            except Exception as e:
                print(f"Warning: CoxPH prediction failed - {e}")
                val_c_index = 0.0
        
        # Calculate average training loss
        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val C-index: {val_c_index:.4f}, Time: {epoch_time:.1f}s")
        print(f"Best Val C-index: {best_val_c_index:.4f}")

        # Save best model
        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            best_epoch = epoch + 1
            
            # Save both feature extractor and CoxPH model
            checkpoint = {
                'epoch': epoch + 1,
                'feature_extractor_state_dict': model.state_dict(),
                'cox_model': cox_model,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_c_index': val_c_index,
                'fold': fold_num,
                'feature_dim': model.output_dim
            }
            
            best_checkpoint_path = os.path.join(save_dir, f"best_coxph_model_fold_{fold_num}.pth")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Best model saved (C-index: {val_c_index:.4f})")
        
        # Learning rate scheduling based on C-index
        scheduler.step(val_c_index)
        
        # Early stopping check
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-7:
            print(f"Learning rate too small ({current_lr:.2e}), stopping early...")
            break
    
    # Load best checkpoint
    best_checkpoint_path = os.path.join(save_dir, f"best_coxph_model_fold_{fold_num}.pth")
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best checkpoint from epoch {best_epoch}...")
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint['feature_extractor_state_dict'])
        cox_model = best_checkpoint['cox_model']
        print(f"Best model loaded (Fold {fold_num}, Epoch {best_epoch}, Val C-index: {best_val_c_index:.4f})")
    
    return model, cox_model, best_checkpoint_path

def evaluate_coxph_model(feature_extractor, cox_model, test_loader, device):
    """Evaluate the combined feature extractor + CoxPH model"""
    
    feature_extractor.eval()
    
    all_features = []
    all_durations = []
    all_events = []
    all_patient_ids = []
    
    # Extract features for all test samples
    print("Extracting test features...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test evaluation"):
            ct_images = batch['ct_image'].to(device)
            pt_images = batch['pt_image'].to(device)
            ehr_features = batch['ehr_features'].to(device)
            durations = batch['duration'].cpu().numpy()
            events = batch['event'].cpu().numpy()
            patient_ids = batch['patient_id']
            
            features = feature_extractor(ct_images, pt_images, ehr_features)
            all_features.append(features.cpu().numpy())
            all_durations.extend(durations.flatten())
            all_events.extend(events.flatten())
            all_patient_ids.extend(patient_ids)
    
    # Combine all features
    X_test_combined = np.vstack(all_features)
    y_test_durations = np.array(all_durations)
    y_test_events = np.array(all_events).astype(bool)
    
    # Get CoxPH predictions
    try:        
        risk_scores = cox_model.predict(X_test_combined)

        # Calculate C-index
        c_index = concordance_index_censored(y_test_events, y_test_durations, risk_scores)[0]
        
        print(f"Test evaluation completed: C-index = {c_index:.4f}")
        
        return {
            'c_index': c_index,
            'risk_scores': risk_scores,
            'durations': y_test_durations,
            'events': y_test_events,
            'patient_ids': all_patient_ids,
            'features': X_test_combined
        }
        
    except Exception as e:
        print(f"Error in CoxPH evaluation: {e}")
        return {
            'c_index': np.nan,
            'risk_scores': np.zeros(len(y_test_durations)),
            'durations': y_test_durations,
            'events': y_test_events,
            'patient_ids': all_patient_ids,
            'features': X_test_combined
        }

def calculate_concordance_index(y_test, risk_scores):
    """Calculate concordance index (C-index) using scikit-survival"""
    try:
        # Extract event indicators and times from structured array
        events = y_test['event']
        times = y_test['time']
        
        # Calculate concordance index
        c_index = concordance_index_censored(events, times, risk_scores)[0]
        return c_index
    except Exception as e:
        print(f"Warning: Could not calculate C-index - {e}")
        return np.nan

def main():
    """Main function using CoxPH approach with ResNet weight updates"""
    
    # Configuration
    IMAGE_DIR = "./Task_2"
    FEATURE_DIM = 64  # ResNet feature dimension
    EPOCHS = 50
    BATCH_SIZE = 2  # Small batch size for medical images
    LEARNING_RATE = 1e-4
    CHECKPOINT_DIR = "./checkpoints"

    if FEATURE_DIM > 256:
        LEARNING_RATE = LEARNING_RATE * 0.1  # Reduce LR
        print(f"Reduced learning rate to {LEARNING_RATE} for high dimensions")
    
    # Load data
    df = pd.read_csv('./HECKTOR_2025_Training_Task_2.csv')
    
    # Load splits
    with open('./cv_folds_5fold.json', 'r') as f:
        splits = json.load(f)
    
    print(f"\nData loaded:")
    print(f"  Total patients: {len(df)}")
    print(f"  Cross-validation folds: {len(splits['folds'])}")
    
    # Load and cache all images
    all_patient_ids = df['PatientID'].tolist()
    image_cache_path = os.path.join(CHECKPOINT_DIR, 'image_cache.pkl')
    
    if os.path.exists(image_cache_path):
        print(f"Loading cached images from {image_cache_path}...")
        with open(image_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        image_cache = HECKTORImageCache.__new__(HECKTORImageCache)
        image_cache.image_dir = Path(IMAGE_DIR)
        image_cache.target_size = (96, 96, 96)
        image_cache.cache = cache_data
    else:
        print("Creating image cache...")
        image_cache = HECKTORImageCache(
            patient_ids=all_patient_ids,
            image_dir=IMAGE_DIR,
            target_size=(96, 96, 96)
        )
        
        # Save image cache
        print(f"Saving image cache to {image_cache_path}...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        with open(image_cache_path, 'wb') as f:
            pickle.dump(image_cache.cache, f)
        print(f"Image cache saved")
    
    # Enhanced categorical features preprocessing
    categorical_features_to_encode = [
        'Gender', 'Tobacco Consumption', 'Alcohol Consumption', 
        'Performance Status', 'Treatment', 'M-stage'
    ]
    
    # Identify feature columns
    ehr_features = df.columns[2:-2]  # Original EHR features
    
    # Advanced EHR preprocessing
    categorical_ehr_features = [col for col in ehr_features if col in categorical_features_to_encode]
    numerical_ehr_features = [col for col in ehr_features if col not in categorical_features_to_encode]
    
    # Enhanced categorical encoding
    encoded_dfs = []
    
    for cat_feature in categorical_ehr_features:
        if cat_feature in df.columns:
            # Missing value handling
            mode_value = df[cat_feature].mode().iloc[0] if len(df[cat_feature].mode()) > 0 else 'Unknown'
            df[cat_feature] = df[cat_feature].fillna(mode_value)
            
            # Create one-hot encoded features
            one_hot = pd.get_dummies(df[cat_feature], prefix=cat_feature, dummy_na=False)
            encoded_dfs.append(one_hot)
    
    # Combine encoded features
    if encoded_dfs:
        one_hot_features_df = pd.concat(encoded_dfs, axis=1)
        one_hot_feature_cols = list(one_hot_features_df.columns)
    else:
        one_hot_features_df = pd.DataFrame(index=df.index)
        one_hot_feature_cols = []
    
    # Combine all EHR features
    all_ehr_features = numerical_ehr_features + one_hot_feature_cols
    
    # Create final feature matrix for EHR
    final_ehr_df = pd.concat([
        df[numerical_ehr_features],
        one_hot_features_df
    ], axis=1)
    
    # Add PatientID and target columns back
    final_df = pd.concat([
        df[['PatientID', 'RFS', 'Relapse']],
        final_ehr_df
    ], axis=1)
    
    # Convert EHR features to float64
    for col in all_ehr_features:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype('float64')
        
    # Initialize metrics storage
    coxph_c_indices = []
    saved_checkpoints = []
    
    # Process each fold
    print(f"\nStarting {len(splits['folds'])}-fold cross-validation...")
    cv_start_time = time.time()
    
    for fold_idx, fold in enumerate(splits['folds']):
        fold_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Processing fold {fold['fold']} ({fold_idx + 1}/{len(splits['folds'])})")
        print(f"{'='*60}")
        
        # Split data
        train_df = final_df[final_df['PatientID'].isin(fold['train'])].copy()
        test_df = final_df[final_df['PatientID'].isin(fold['test'])].copy()
        
        # Further split training data into train/val for model training
        train_size = int(0.8 * len(train_df))
        train_fold_df = train_df.iloc[:train_size].copy()
        val_fold_df = train_df.iloc[train_size:].copy()
        
        print(f"Data split:")
        print(f"  Training: {len(train_fold_df)} patients")
        print(f"  Validation: {len(val_fold_df)} patients")
        print(f"  Test: {len(test_df)} patients")
        
        # KNN imputation
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        
        X_train_ehr = knn_imputer.fit_transform(train_fold_df[all_ehr_features].values)
        X_val_ehr = knn_imputer.transform(val_fold_df[all_ehr_features].values)
        X_test_ehr = knn_imputer.transform(test_df[all_ehr_features].values)
        
        # Robust scaling
        robust_scaler = RobustScaler()
        X_train_ehr = robust_scaler.fit_transform(X_train_ehr)
        X_val_ehr = robust_scaler.transform(X_val_ehr)
        X_test_ehr = robust_scaler.transform(X_test_ehr)
        
        # Feature selection for EHR (keep reasonable number for neural network)
        events_train = train_fold_df['Relapse'].values.astype(bool)
        durations_train = train_fold_df['RFS'].values.astype(float)
        y_selection = durations_train * (2 * events_train.astype(int) - 1)
        
        # selector = SelectKBest(score_func=f_regression, k=min(100, X_train_ehr.shape[1]))
        # X_train_ehr = selector.fit_transform(X_train_ehr, y_selection)
        # X_val_ehr = selector.transform(X_val_ehr)
        # X_test_ehr = selector.transform(X_test_ehr)
                
        # Create datasets using cached images
        train_data = HECKTORDataset(
            patient_ids=train_fold_df['PatientID'].tolist(),
            image_cache=image_cache,
            ehr_features=X_train_ehr,
            durations=train_fold_df['RFS'].values,
            events=train_fold_df['Relapse'].values
        )
        
        val_data = HECKTORDataset(
            patient_ids=val_fold_df['PatientID'].tolist(),
            image_cache=image_cache,
            ehr_features=X_val_ehr,
            durations=val_fold_df['RFS'].values,
            events=val_fold_df['Relapse'].values
        )
        
        test_data = HECKTORDataset(
            patient_ids=test_df['PatientID'].tolist(),
            image_cache=image_cache,
            ehr_features=X_test_ehr,
            durations=test_df['RFS'].values,
            events=test_df['Relapse'].values
        )
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        
        # Initialize feature extractor
        feature_extractor = HECKTORFeatureExtractor(
            feature_dim=FEATURE_DIM,
            ehr_input_dim=X_train_ehr.shape[1]
        ).to(device)
        
        # Train the combined model
        print(f"\n Starting training...")
        
        trained_feature_extractor, trained_cox_model, best_checkpoint_path = train_feature_extractor_with_coxph(
            model=feature_extractor,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            fold_num=fold['fold'],
            save_dir=CHECKPOINT_DIR
        )
        
        # Evaluate on test set
        print(f"\nEvaluating on test set...")
        test_results = evaluate_coxph_model(trained_feature_extractor, trained_cox_model, test_loader, device)
        
        c_index = test_results['c_index']
        coxph_c_indices.append(c_index)
        
        # Store checkpoint information
        saved_checkpoints.append({
            'fold': fold['fold'],
            'checkpoint': best_checkpoint_path,
            'c_index': c_index,
            'feature_dim': feature_extractor.output_dim
        })
        
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        
        print(f"\nFold {fold['fold']} Results:")
        print(f"   CoxPH + Feature Extractor C-index: {c_index:.4f}")
        print(f"   Best checkpoint: {os.path.basename(best_checkpoint_path)}")
        print(f"   Fold duration: {fold_duration/60:.1f} minutes")
        
        # Estimate remaining time
        if fold_idx < len(splits['folds']) - 1:
            elapsed_total = time.time() - cv_start_time
            avg_time_per_fold = elapsed_total / (fold_idx + 1)
            remaining_folds = len(splits['folds']) - (fold_idx + 1)
            estimated_remaining = (avg_time_per_fold * remaining_folds) / 60
            print(f"   Estimated remaining time: {estimated_remaining:.1f} minutes")
        
        # Clean up GPU memory
        del feature_extractor, trained_feature_extractor, trained_cox_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Clean up image cache if needed
    print(f"\nClearing image cache to free memory...")
    image_cache.clear_cache()
    
    # Print final results
    total_time = time.time() - cv_start_time
    
    print("\n" + "="*80)
    print("FINAL COXPH + FEATURE EXTRACTOR RESULTS")
    print("="*80)
    
    print(f"\nTraining Summary:")
    print(f"  Total Training Time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"  Average time per fold: {total_time/len(splits['folds'])/60:.1f} minutes")

    
    print(f"\nModel Performance:")
    print(f"  Average C-index: {np.mean(coxph_c_indices):.4f} ± {np.std(coxph_c_indices):.4f}")
    print(f"  C-index per fold: {[f'{c:.4f}' for c in coxph_c_indices]}")
    print(f"  Best fold C-index: {np.max(coxph_c_indices):.4f}")
    print(f"  Worst fold C-index: {np.min(coxph_c_indices):.4f}")
    print(f"  C-index consistency (std): {np.std(coxph_c_indices):.4f}")
    
    # Checkpoint Summary
    print(f"\nSAVED CHECKPOINTS SUMMARY")
    print("="*80)
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Total models saved: {len(saved_checkpoints)}")
    
    print(f"\nBest performing models by fold:")
    for checkpoint_info in saved_checkpoints:
        fold_num = checkpoint_info['fold']
        c_idx = checkpoint_info['c_index']
        checkpoint_path = os.path.basename(checkpoint_info['checkpoint'])
        feature_dim = checkpoint_info['feature_dim']
        
        print(f"  Fold {fold_num}: {checkpoint_path}")
        print(f"           C-index: {c_idx:.4f}, Features: {feature_dim}")
    
    return {
        'c_indices': coxph_c_indices,
        'checkpoints': saved_checkpoints,
        'avg_c_index': np.mean(coxph_c_indices),
        'std_c_index': np.std(coxph_c_indices)
    }

if __name__ == "__main__":
    results = main()
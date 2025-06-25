import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler
import pandas as pd
from monai.networks.nets import ResNet
from monai.networks.layers import Norm
import warnings
import os

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpatialAttention3D(nn.Module):
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

class HECKTORResNet(nn.Module):
    def __init__(self, input_channels=1, feature_dim=1024, spatial_dims=3):
        super(HECKTORResNet, self).__init__()
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
    def __init__(self, feature_dim=128, ehr_input_dim=20):
        super(HECKTORFeatureExtractor, self).__init__()
        self.ct_resnet = HECKTORResNet(
            input_channels=1, 
            feature_dim=feature_dim,
            spatial_dims=3
        )
        self.pt_resnet = HECKTORResNet(
            input_channels=1, 
            feature_dim=feature_dim,
            spatial_dims=3
        )
        self.ehr_processor = nn.Sequential(
            nn.Linear(ehr_input_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        total_features = feature_dim * 2 + feature_dim // 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_dim = feature_dim // 2
        self._init_weights()
    def _init_weights(self):
        for m in [self.ehr_processor, self.fusion_layer]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.constant_(module.bias, 0)
    def forward(self, ct_images, pt_images, ehr_features):
        ct_features = self.ct_resnet(ct_images)
        pt_features = self.pt_resnet(pt_images)
        ehr_processed = self.ehr_processor(ehr_features)
        combined_features = torch.cat([ct_features, pt_features, ehr_processed], dim=1)
        final_features = self.fusion_layer(combined_features)
        return final_features

def preprocess_image(image_data, target_size=(96, 96, 96)):
    if np.isnan(image_data).any():
        image_data = np.nan_to_num(image_data, nan=0.0)
    current_shape = image_data.shape
    # If image is 2D, add a depth dimension
    if len(current_shape) == 2:
        image_data = np.expand_dims(image_data, axis=0)
        current_shape = image_data.shape
    # If image is (1, H, W), repeat along depth
    if current_shape[0] == 1 and target_size[0] > 1:
        image_data = np.repeat(image_data, target_size[0], axis=0)
        current_shape = image_data.shape
    zoom_factors = [t/c for t, c in zip(target_size, current_shape)]
    resized_image = zoom(image_data, zoom_factors, order=3)
    if np.any(resized_image > 0):
        p1, p99 = np.percentile(resized_image[resized_image > 0], [1, 99])
        if p99 > p1:
            resized_image = np.clip((resized_image - p1) / (p99 - p1), 0, 1)
        non_zero_mask = resized_image > 0
        if np.any(non_zero_mask):
            mean_val = np.mean(resized_image[non_zero_mask])
            std_val = np.std(resized_image[non_zero_mask])
            if std_val > 0:
                resized_image[non_zero_mask] = (resized_image[non_zero_mask] - mean_val) / std_val
                resized_image = (resized_image - resized_image.min()) / (resized_image.max() - resized_image.min() + 1e-8)
    tensor_image = torch.FloatTensor(resized_image).unsqueeze(0).unsqueeze(0)
    # Ensure we have 5D: (batch, channel, depth, height, width)
    while tensor_image.dim() < 5:
        tensor_image = tensor_image.unsqueeze(0)
    # If depth is 1, repeat to 96
    if tensor_image.shape[2] == 1 and target_size[0] > 1:
        tensor_image = tensor_image.repeat(1, 1, target_size[0], 1, 1)
    return tensor_image

def process_ehr_features_on_the_fly(ehr_data):
    # Numerical
    age = float(ehr_data.get('Age', 0))
    features = [age]

    # One-hot columns in order
    # Gender: 0, 1
    gender = int(ehr_data.get('Gender', 0))
    features += [1.0 if gender == 0 else 0.0, 1.0 if gender == 1 else 0.0]

    # Tobacco Consumption: 0.0, 1.0
    tobacco = float(ehr_data.get('Tobacco Consumption', 0.0))
    features += [1.0 if tobacco == 0.0 else 0.0, 1.0 if tobacco == 1.0 else 0.0]

    # Alcohol Consumption: 0.0, 1.0
    alcohol = float(ehr_data.get('Alcohol Consumption', 0.0))
    features += [1.0 if alcohol == 0.0 else 0.0, 1.0 if alcohol == 1.0 else 0.0]

    # Performance Status: 0.0, 1.0, 2.0, 3.0, 4.0, 80.0, 90.0, 100.0
    perf_status = float(ehr_data.get('Performance Status', 0.0))
    for val in [0.0, 1.0, 2.0, 3.0, 4.0, 80.0, 90.0, 100.0]:
        features.append(1.0 if perf_status == val else 0.0)

    # Treatment: 0, 1
    treatment = int(ehr_data.get('Treatment', 0))
    features += [1.0 if treatment == 0 else 0.0, 1.0 if treatment == 1 else 0.0]

    # M-stage: M0, M1, Mx
    mstage = ehr_data.get('M-stage', 'M0')
    features += [1.0 if mstage == 'M0' else 0.0, 1.0 if mstage == 'M1' else 0.0, 1.0 if mstage == 'Mx' else 0.0]

    assert len(features) == 20, f"Feature vector is {len(features)} not 20!"  # Safety check
    return torch.tensor([features], dtype=torch.float32)

def create_fallback_model():
    # Use the original HECKTORFeatureExtractor (no fallback)
    feature_extractor = HECKTORFeatureExtractor(
        feature_dim=128,
        ehr_input_dim=20
    ).to(device)
    class DummyCoxModel:
        def predict(self, features):
            return np.random.randn(len(features)) * 0.5
    cox_model = DummyCoxModel()
    return feature_extractor, cox_model

# def load_model(checkpoint_path):
#     if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
#         raise RuntimeError(f"Checkpoint file '{checkpoint_path}' not found or empty. Cannot proceed.")
#     try:
#         checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#         feature_dim = checkpoint.get('feature_dim', 256)
#         ehr_input_dim = checkpoint.get('ehr_input_dim', 20)
#         feature_extractor = HECKTORFeatureExtractor(
#             feature_dim=feature_dim,
#             ehr_input_dim=ehr_input_dim
#         ).to(device)
#         if 'feature_extractor_state_dict' in checkpoint:
#             # Strict loading: do not allow partial or mismatched weights
#             feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'], strict=True)
#         else:
#             raise RuntimeError("feature_extractor_state_dict not found in checkpoint.")
#         feature_extractor.eval()
#         cox_model = checkpoint.get('cox_model', None)
#         if cox_model is None:
#             raise RuntimeError("cox_model not found in checkpoint.")
#     except Exception as e:
#         raise RuntimeError(f"Error loading checkpoint: {e}")
#     return {
#         'feature_extractor': feature_extractor,
#         'cox_model': cox_model
#     }

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
        raise RuntimeError(f"Checkpoint file '{checkpoint_path}' not found or empty. Cannot proceed.")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Auto-detect feature_dim from the checkpoint weights
        # Look at ct_resnet.feature_fusion.3.weight shape to determine feature_dim
        if 'feature_extractor_state_dict' in checkpoint:
            state_dict = checkpoint['feature_extractor_state_dict']
            
            # Get the weight shape from the first linear layer in feature_fusion
            if 'ct_resnet.feature_fusion.3.weight' in state_dict:
                weight_shape = state_dict['ct_resnet.feature_fusion.3.weight'].shape
                # This layer is nn.Linear(131072, feature_dim * 2)
                # So feature_dim = weight_shape[0] / 2
                detected_feature_dim = weight_shape[0] // 2
                print(f"Auto-detected feature_dim: {detected_feature_dim}")
            else:
                # Fallback to checkpoint value or default
                detected_feature_dim = checkpoint.get('feature_dim', 256)
                print(f"Using fallback feature_dim: {detected_feature_dim}")
        else:
            raise RuntimeError("feature_extractor_state_dict not found in checkpoint.")
        
        # Use detected feature_dim, fallback to checkpoint value, then default
        feature_dim = detected_feature_dim
        ehr_input_dim = checkpoint.get('ehr_input_dim', 20)
                
        feature_extractor = HECKTORFeatureExtractor(
            feature_dim=feature_dim,
            ehr_input_dim=ehr_input_dim
        ).to(device)
        
        # Now load the state dict with matching architecture
        feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'], strict=True)
        feature_extractor.eval()
        
        cox_model = checkpoint.get('cox_model', None)
        if cox_model is None:
            raise RuntimeError("cox_model not found in checkpoint.")
            
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")
    
    return {
        'feature_extractor': feature_extractor,
        'cox_model': cox_model
    }

def predict_survival(model, ct_image, pet_image, ehr_data):
    feature_extractor = model['feature_extractor']
    cox_model = model['cox_model']
    
    ct_tensor = preprocess_image(ct_image, target_size=(96, 96, 96))
    pt_tensor = preprocess_image(pet_image, target_size=(96, 96, 96))
    
    
    # Tensors are already 5D: (batch, channel, depth, height, width)
    ct_tensor = ct_tensor.to(device)
    pt_tensor = pt_tensor.to(device)
    
    ehr_tensor = process_ehr_features_on_the_fly(ehr_data).to(device)
    
    with torch.no_grad():
        try:
            features = feature_extractor(ct_tensor, pt_tensor, ehr_tensor)
            features_np = features.cpu().numpy()
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            print(f"CT tensor shape: {ct_tensor.shape}, dims: {ct_tensor.dim()}")
            print(f"PET tensor shape: {pt_tensor.shape}, dims: {pt_tensor.dim()}")
            raise e
    
    try:
        risk_score = cox_model.predict(features_np)[0]
        base_survival_time = 60.0
        risk_factor = np.exp(-risk_score * 0.1)
        predicted_rfs = base_survival_time * risk_factor
        predicted_rfs = np.clip(predicted_rfs, 1.0, 120.0)
        return float(predicted_rfs)
    except Exception as e:
        print(f"Cox prediction failed: {e}")
        return 30.0 
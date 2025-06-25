import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from monai.networks.nets import ResNet
from monai.networks.layers import Norm
from sksurv.metrics import concordance_index_censored
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.preprocessing import RobustScaler
import argparse
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definitions
class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
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
        super().__init__()
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
    def __init__(self, input_channels=1, feature_dim=128, spatial_dims=3):
        super().__init__()
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
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = torch.relu(x)
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
        super().__init__()
        self.ct_resnet = HECKTORAdvancedResNet(1, feature_dim)
        self.pt_resnet = HECKTORAdvancedResNet(1, feature_dim)
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
    if len(current_shape) == 2:
        image_data = np.expand_dims(image_data, axis=0)
        current_shape = image_data.shape
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
    while tensor_image.dim() < 5:
        tensor_image = tensor_image.unsqueeze(0)
    if tensor_image.shape[2] == 1 and target_size[0] > 1:
        tensor_image = tensor_image.repeat(1, 1, target_size[0], 1, 1)
    return tensor_image

def preprocess_ehr_df(df, categorical_features_to_encode, numerical_ehr_features, one_hot_feature_cols, mode_map):
    # Fill missing categorical values with mode
    for cat in categorical_features_to_encode:
        df[cat] = df[cat].fillna(mode_map[cat])
    # One-hot encode categoricals
    encoded_dfs = []
    for cat in categorical_features_to_encode:
        one_hot = pd.get_dummies(df[cat], prefix=cat, dummy_na=False)
        # Ensure all columns from training are present
        for col in [c for c in one_hot_feature_cols if c.startswith(f"{cat}_")]:
            if col not in one_hot.columns:
                one_hot[col] = 0
        one_hot = one_hot[[c for c in one_hot_feature_cols if c.startswith(f"{cat}_")]]
        encoded_dfs.append(one_hot)
    if encoded_dfs:
        one_hot_features_df = pd.concat(encoded_dfs, axis=1)
    else:
        one_hot_features_df = pd.DataFrame(index=df.index)
    # Combine all EHR features
    final_ehr_df = pd.concat([df[numerical_ehr_features], one_hot_features_df], axis=1)
    # Ensure column order matches training
    final_ehr_df = final_ehr_df[numerical_ehr_features + one_hot_feature_cols]
    final_ehr_df = final_ehr_df.astype('float64')
    return final_ehr_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--test_image_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    # Load test data
    df = pd.read_csv(args.test_csv)
    patient_ids = df['PatientID'].tolist()

    # EHR preprocessing setup (copied from training)
    categorical_features_to_encode = [
        'Gender', 'Tobacco Consumption', 'Alcohol Consumption', 
        'Performance Status', 'Treatment', 'M-stage'
    ]
    ehr_features = df.columns[2:-2]
    categorical_ehr_features = [col for col in ehr_features if col in categorical_features_to_encode]
    numerical_ehr_features = [col for col in ehr_features if col not in categorical_features_to_encode]
    mode_map = {cat: df[cat].mode().iloc[0] if len(df[cat].mode()) > 0 else 'Unknown' for cat in categorical_ehr_features}
    one_hot_feature_cols = []
    for cat in categorical_ehr_features:
        mode_value = mode_map[cat]
        unique_vals = sorted(df[cat].fillna(mode_value).unique(), key=lambda x: str(x))
        one_hot_feature_cols += [f"{cat}_{v}" for v in unique_vals]
    ehr_matrix = preprocess_ehr_df(df, categorical_features_to_encode, numerical_ehr_features, one_hot_feature_cols, mode_map)
    scaler = RobustScaler()
    ehr_matrix_scaled = scaler.fit_transform(ehr_matrix.values)

    # Load model and CoxPH from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = HECKTORFeatureExtractor(feature_dim=128, ehr_input_dim=ehr_matrix_scaled.shape[1]).to(device)
    model.load_state_dict(checkpoint['feature_extractor_state_dict'], strict=True)
    model.eval()
    cox_model = checkpoint['cox_model']

    # Inference
    all_features = []
    all_durations = []
    all_events = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pid = row['PatientID']
        ct_path = Path(args.test_image_dir) / f"{pid}__CT.nii.gz"
        pt_path = Path(args.test_image_dir) / f"{pid}__PT.nii.gz"
        ct_img = nib.load(str(ct_path))
        pt_img = nib.load(str(pt_path))
        ct_data = ct_img.get_fdata()
        pt_data = pt_img.get_fdata()
        ct_tensor = preprocess_image(ct_data, target_size=(96, 96, 96)).to(device)
        pt_tensor = preprocess_image(pt_data, target_size=(96, 96, 96)).to(device)
        ehr_vec_scaled = ehr_matrix_scaled[idx]
        ehr_tensor = torch.tensor([ehr_vec_scaled], dtype=torch.float32).to(device)
        with torch.no_grad():
            features = model(ct_tensor, pt_tensor, ehr_tensor)
            all_features.append(features.cpu().numpy())
        all_durations.append(row['RFS'])
        all_events.append(row['Relapse'])
    X = np.vstack(all_features)
    y_durations = np.array(all_durations)
    y_events = np.array(all_events).astype(bool)
    risk_scores = cox_model.predict(X)
    c_index = concordance_index_censored(y_events, y_durations, risk_scores)[0]
    print(f"Test C-index: {c_index:.4f}")

if __name__ == "__main__":
    main() 
import torch
import numpy as np
from monai.networks.nets import resnet18
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pathlib import Path
import json
from glob import glob
import SimpleITK
from monai.transforms import Resize, ScaleIntensity
import torch.nn.functional as F

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

class MultiModalResNet(nn.Module):
    def __init__(self, clin_feat_dim, num_classes=2):
        super().__init__()
        # 3D ResNet18 backbone for CT+PET
        self.img_model = resnet18(
            spatial_dims=3,
            n_input_channels=2,
            num_classes=2,
        )
        self.img_model.fc = nn.Identity()

        # MLP for clinical data
        self.clin_model = nn.Sequential(
            nn.Linear(clin_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # fusion + classification
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_clin):
        f_img = self.img_model(x_img)    
        f_clin = self.clin_model(x_clin)   
        f = torch.cat([f_img, f_clin], dim=1)
        return self.classifier(f)


def run():

    # 1. Load inputs
    ct_np = load_image_file_as_array(location=INPUT_PATH / "images/ct")
    pet_np = load_image_file_as_array(location=INPUT_PATH / "images/pet")
    ehr = load_json_file(location=INPUT_PATH / "ehr.json")

    # 2. Preprocess images (mimicking monai transforms)
    def preprocess_image(img):
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # normalize
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        img = Resize((96, 96, 96))(img)
        return img.squeeze(0)  # [1, D, H, W]

    ct_tensor = preprocess_image(ct_np)
    pet_tensor = preprocess_image(pet_np)
    x_img = torch.cat([ct_tensor, pet_tensor], dim=0).unsqueeze(0).cuda()  # [1,2,D,H,W]

    # 3. Preprocess EHR
    df_ehr = pd.DataFrame([ehr])  # convert to single-row DataFrame
    num_data = df_ehr[["Age", "Gender"]].values
    cat_data = df_ehr[["Tobacco Consumption", "Alcohol Consumption", "Performance Status", "M-stage"]].astype(str).fillna("Unknown").values

    # Load scalers
    scaler = load(RESOURCE_PATH / "scaler.joblib")
    ohe = load(RESOURCE_PATH / "ohe.joblib")

    num_feats = scaler.transform(num_data)
    cat_feats = ohe.transform(cat_data)
    x_clin = np.hstack([num_feats, cat_feats])
    x_clin = torch.tensor(x_clin, dtype=torch.float32).cuda()

    # 4. Load model
    model = MultiModalResNet(clin_feat_dim=x_clin.shape[1], num_classes=2).cuda()
    model.load_state_dict(torch.load(RESOURCE_PATH / "best_model.pt"))
    model.eval()

    # 5. Run prediction
    with torch.no_grad():
        logits = model(x_img, x_clin)
        pred = logits.argmax(dim=1).item()  # 0 or 1

    # 6. Save output
    write_json_file(location=OUTPUT_PATH / "hpv-status.json", content=pred)
    return 0


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
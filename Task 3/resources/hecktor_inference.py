import torch
import numpy as np
from monai.transforms import Resize
import pandas as pd


def preprocess_image(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # normalize
    img = torch.tensor(img)  # [D, H, W]
    img = Resize((96, 96, 96))(img.unsqueeze(0))  # [1, D, H, W]
    return img  # [1, D, H, W]


def prepare_input_tensor(ct_image, pet_image):

    ct_tensor = preprocess_image(ct_image)
    pet_tensor = preprocess_image(pet_image)
    
    x_img = torch.cat([ct_tensor, pet_tensor], dim=0).unsqueeze(0).cuda()  # [1,2,D,H,W]

    return x_img

def preprocess_ehr(ehr, scaler, ohe):
    df_ehr = pd.DataFrame([ehr])  # convert to single-row DataFrame
    num_data = df_ehr[["Age", "Gender"]].values
    cat_data = df_ehr[["Tobacco Consumption", "Alcohol Consumption", "Performance Status", "M-stage"]].astype(str).fillna("Unknown").values
    
    
    num_feats = scaler.transform(num_data)
    cat_feats = ohe.transform(cat_data)
    x_clin = np.hstack([num_feats, cat_feats])
    x_clin = torch.tensor(x_clin, dtype=torch.float32).cuda()

    return x_clin


def run_inference(model, x_img, x_clin):
    model.eval()

    with torch.no_grad():
        logits = model(x_img, x_clin)
        pred = logits.argmax(dim=1).item()  # 0 or 1

    return bool(pred)

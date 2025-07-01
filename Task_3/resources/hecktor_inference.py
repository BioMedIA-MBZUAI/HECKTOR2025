import torch
import numpy as np
from monai.transforms import Resize
import pandas as pd
import SimpleITK as sitk
import warnings
from skimage.measure import label

def get_bounding_boxes(ct_sitk, pet_sitk):
    """
    Get the bounding boxes of the CT and PET images.
    This works since all images have the same direction.
    """
    ct_origin = np.array(ct_sitk.GetOrigin())
    pet_origin = np.array(pet_sitk.GetOrigin())

    ct_position_max = ct_origin + np.array(ct_sitk.GetSize()) * np.array(ct_sitk.GetSpacing())
    pet_position_max = pet_origin + np.array(pet_sitk.GetSize()) * np.array(pet_sitk.GetSpacing())
    
    return np.concatenate([
        np.maximum(ct_origin, pet_origin),
        np.minimum(ct_position_max, pet_position_max),
    ], axis=0)

def resample_images(ct_array, pet_array):
    """
    Resample CT and PET images to specified resolution using SimpleITK.
    
    Args:
        ct_array: CT image as numpy array
        pet_array: PET image as numpy array
        
    Returns:
        Tuple of (resampled_ct_array, resampled_pet_array)
    """
    # Convert numpy arrays to SimpleITK images
    ct_sitk = sitk.GetImageFromArray(ct_array)
    pet_sitk = sitk.GetImageFromArray(pet_array)
    
    # Set default spacing and origin if not present
    ct_sitk.SetSpacing([1.0, 1.0, 1.0])
    pet_sitk.SetSpacing([1.0, 1.0, 1.0])
    ct_sitk.SetOrigin([0.0, 0.0, 0.0])
    pet_sitk.SetOrigin([0.0, 0.0, 0.0])
    
    # Get bounding box for both modalities
    bb = get_bounding_boxes(ct_sitk, pet_sitk)
    resampling= (1.0, 1.0, 1.0)
    size = np.round((bb[3:] - bb[:3]) / np.array(resampling)).astype(int)
    
    # Set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)
    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in size])
    
    # Resample CT with B-spline interpolation
    resampler.SetInterpolator(sitk.sitkBSpline)
    ct_resampled = resampler.Execute(ct_sitk)
    
    # Resample PET with B-spline interpolation
    pet_resampled = resampler.Execute(pet_sitk)
    
    # Convert back to numpy arrays
    ct_resampled_array = sitk.GetArrayFromImage(ct_resampled)
    pet_resampled_array = sitk.GetArrayFromImage(pet_resampled)
    
    return ct_resampled_array, pet_resampled_array
    
def get_roi_center(pet_tensor, z_top_fraction=0.75, z_score_threshold=1.0):
    """
    Calculates the center of the largest high-intensity region in the top part of the PET scan.
    """
    # 1. Isolate top of the scan based on the z-axis
    image_shape_voxels = np.array(pet_tensor.shape)
    crop_z_start = int(z_top_fraction * image_shape_voxels[2])
    top_of_scan = pet_tensor[..., crop_z_start:]

    # 2. Threshold to find high-intensity regions (potential brain/tumor)
    # Using a small epsilon to avoid division by zero in blank images
    mask = ((top_of_scan - top_of_scan.mean()) / (top_of_scan.std() + 1e-8)) > z_score_threshold
    
    if not mask.any():
        # If no pixels are above the threshold, fall back to the geometric center of the top part
        warnings.warn("No high-intensity region found. Using geometric center of the upper scan region.")
        center_in_top = (np.array(top_of_scan.shape) / 2).astype(int)
    else:
        # Find the largest connected component to remove noise
        labeled_mask, num_features = label(mask, return_num=True, connectivity=3)
        if num_features > 0:
            component_sizes = np.bincount(labeled_mask.ravel())[1:]  # ignore background
            largest_component_label = np.argmax(component_sizes) + 1
            largest_component_mask = labeled_mask == largest_component_label
            comp_idx = np.argwhere(largest_component_mask)
        else:  # Should not happen if mask.any() is true, but as a safeguard
            comp_idx = np.argwhere(mask)

        # 3. Calculate the centroid of the largest component
        center_in_top = np.mean(comp_idx, axis=0)

    # 4. Adjust center to be in the original full-image coordinate system
    center_full_image = center_in_top + np.array([0, 0, crop_z_start])
    return center_full_image.astype(int)

def crop_neck_region(ct_array, pet_array):
    """
    Crop the head and neck region from CT and PET images.
    
    Args:
        ct_array: CT image as numpy array
        pet_array: PET image as numpy array
        
    Returns:
        Tuple of (cropped_ct_array, cropped_pet_array)
    """
    # Convert to torch tensor for processing
    pet_tensor = torch.from_numpy(pet_array).float()
    
    # Get box size in voxels (assuming 1mm spacing after resampling)
    crop_box_size=[200, 200, 310]
    crop_box_size = np.array(crop_box_size)
    box_size_voxels = crop_box_size.astype(int)
    
    # 1. Find the robust center of the ROI using PET
    center_voxels = get_roi_center(pet_tensor)
    
    # 2. Calculate crop box and handle boundaries safely
    image_shape_voxels = np.array(pet_array.shape)
    box_start = center_voxels - box_size_voxels // 2
    box_end = box_start + box_size_voxels
    
    # Clamp coordinates to ensure they are within the image boundaries
    box_start = np.maximum(box_start, 0)
    box_end = np.minimum(box_end, image_shape_voxels)
    
    # Recalculate start to handle cases where box goes over the 0-boundary
    box_start = np.maximum(box_end - box_size_voxels, 0)
    
    box_start = box_start.astype(int)
    box_end = box_end.astype(int)
    
    # Apply the crop to both CT and PET
    ct_cropped = ct_array[box_start[0]:box_end[0], box_start[1]:box_end[1], box_start[2]:box_end[2]]
    pet_cropped = pet_array[box_start[0]:box_end[0], box_start[1]:box_end[1], box_start[2]:box_end[2]]
    
    return ct_cropped, pet_cropped
 
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
    # handle missing values and convert to DataFrame
    df_ehr = pd.DataFrame([{
        "Age": ehr.get("Age", 0),
        "Gender": ehr.get("Gender", "Unknown"),
        "Tobacco Consumption": ehr.get("Tobacco Consumption", "Unknown"),
        "Alcohol Consumption": ehr.get("Alcohol Consumption", "Unknown"),
        "Performance Status": ehr.get("Performance Status", "Unknown"),
        "M-stage": ehr.get("M-stage", "Unknown")
    }])

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

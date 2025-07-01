import SimpleITK as sitk
import numpy as np
import torch
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
 
def resample_images(ct, pt):
    """
    Resample CT and PET images to specified resolution using SimpleITK.
    
    Args:
        ct_array: CT image as numpy array
        pet_array: PET image as numpy array
        
    Returns:
        Tuple of (resampled_ct_array, resampled_pet_array)
    """
    resampling = [1, 1, 1]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)
    bb = get_bounding_boxes(ct, pt)
    size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in size])  
    resampler.SetInterpolator(sitk.sitkBSpline)
    ct = resampler.Execute(ct)
    pt = resampler.Execute(pt)
    
    return ct,pt, bb
    
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
 
def crop_neck_region(ct_sitk, pet_sitk):
    """
    Crop CT and PET images using SimpleITK to preserve spacing/origin/direction.
    
    Args:
        ct_sitk: Resampled CT as SimpleITK.Image
        pet_sitk: Resampled PET as SimpleITK.Image
        
    Returns:
        Cropped CT, PET, and crop box (box_start, box_end) in voxel space
    """
    pet_np = sitk.GetArrayFromImage(pet_sitk)
    pet_tensor = torch.from_numpy(pet_np).float()

    crop_box_size = np.array([200, 200, 310])
    center_voxels = get_roi_center(pet_tensor)  # (z, y, x)

    box_start = center_voxels - crop_box_size // 2
    box_end = box_start + crop_box_size

    shape = np.array(pet_np.shape)
    box_start = np.maximum(box_start, 0)
    box_end = np.minimum(box_end, shape)
    box_start = np.maximum(box_end - crop_box_size, 0)

    box_start = box_start.astype(int)
    box_end = box_end.astype(int)

    # Convert to SITK index/size (x, y, z)
    roi_start = [int(box_start[2]), int(box_start[1]), int(box_start[0])]
    roi_size = [int(box_end[2] - box_start[2]), int(box_end[1] - box_start[1]), int(box_end[0] - box_start[0])]

    ct_cropped = sitk.RegionOfInterest(ct_sitk, size=roi_size, index=roi_start)
    pet_cropped = sitk.RegionOfInterest(pet_sitk, size=roi_size, index=roi_start)

    return ct_cropped, pet_cropped, box_start, box_end

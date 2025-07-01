import SimpleITK as sitk
import numpy as np

def prediction_to_original_space(
    pred_crop_np: np.ndarray,          # shape: [D, H, W]
    box_start: np.ndarray,             # in (z, y, x) voxel order of resampled space
    box_end:   np.ndarray,
    resampled_ct_sitk: sitk.Image,
    original_ct_sitk:  sitk.Image,
) -> sitk.Image:
    """
    Convert predicted mask back to original CT space.
    """
    # (1) Create empty volume in resampled space
    resampled_shape = sitk.GetArrayFromImage(resampled_ct_sitk).shape
    full_mask_arr = np.zeros(resampled_shape, dtype=np.uint8)

    # (2) Paste predicted crop into correct voxel location
    z0, y0, x0 = box_start.astype(int)
    z1, y1, x1 = box_end.astype(int)

    assert pred_crop_np.shape == (z1 - z0, y1 - y0, x1 - x0), "Shape mismatch with crop box"
    full_mask_arr[z0:z1, y0:y1, x0:x1] = pred_crop_np

    # (3) Convert to SimpleITK image and copy spatial metadata
    full_mask_img = sitk.GetImageFromArray(full_mask_arr)
    full_mask_img.SetSpacing(resampled_ct_sitk.GetSpacing())
    full_mask_img.SetOrigin(resampled_ct_sitk.GetOrigin())
    full_mask_img.SetDirection(resampled_ct_sitk.GetDirection())

    # (4) Resample to original CT space using nearest neighbor
    resampled_back = sitk.Resample(
        full_mask_img,
        original_ct_sitk,               # this ensures matching size, origin, spacing, direction
        sitk.Transform(),               # identity transform
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )

    return resampled_back

import numpy as np
import SimpleITK as sitk


def prediction_to_original_space(
    pred_np: np.ndarray,            # (Z,Y,X)   – network output, CH squeezed
    meta: dict,                     # ct_proc.meta  coming from apply_monai_transforms
    box_start_xyz, box_end_xyz,     # lists returned by crop_neck_region_sitk
    resampled_ct_sitk: sitk.Image,  # 1-mm CT fed to MONAI
    original_ct_sitk: sitk.Image,   # the scanner-resolution CT
):
    """
    1) remove symmetric Z-padding (computed from shape diff, no meta["padding"] needed)
    2) undo CropForegroundd (roi_start / roi_end in `meta`)
    3) undo neck ROI crop (box_start_xyz / box_end_xyz)
    4) resample mask from 1-mm space back to original CT space (nearest-neighbor)

    returns: SimpleITK.Image aligned with `original_ct_sitk`
    """
    # ------------------------------------------------------- 1. undo SpatialPadd
    Zt = box_end_xyz[2] - box_start_xyz[2]               # target depth (e.g. 298)
    z_extra = pred_np.shape[0] - Zt                      # 12 when 310→298
    if z_extra > 0:
        z_trim = z_extra // 2
        pred_np = pred_np[z_trim:z_trim + Zt, ...]       # (Zt,Y,X)

    # ------------------------------------------------------- 2. undo CropForegroundd
    if "roi_start" in meta:                              # might be missing
        z0,y0,x0 = meta["roi_start"]
        z1,y1,x1 = meta["roi_end"]
        canvas_cf = np.zeros((Zt,       # already trimmed depth
                              y1-y0,
                              x1-x0), dtype=pred_np.dtype)
        canvas_cf[z0:z1, y0:y1, x0:x1] = pred_np
    else:
        canvas_cf = pred_np                               # nothing to undo

    # ------------------------------------------------------- 3. undo neck ROI crop
    x0,y0,z0 = box_start_xyz
    x1,y1,z1 = box_end_xyz
    Zr,Yr,Xr = resampled_ct_sitk.GetSize()[2], resampled_ct_sitk.GetSize()[1], resampled_ct_sitk.GetSize()[0]
    canvas_roi = np.zeros((Zr, Yr, Xr), dtype=pred_np.dtype)
    canvas_roi[z0:z1, y0:y1, x0:x1] = canvas_cf

    # ------------------------------------------------------- 4. resample to original CT
    mask_1mm = sitk.GetImageFromArray(canvas_roi.astype(np.uint8))
    mask_1mm.CopyInformation(resampled_ct_sitk)

    mask_orig = sitk.Resample(
        mask_1mm,                # moving
        original_ct_sitk,        # reference
        sitk.Transform(),        # identity
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    return mask_orig

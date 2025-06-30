"""I/O utilities for medical images."""

import os
import numpy as np
import nibabel as nib
from typing import Tuple, Optional


def load_nifti(file_path: str) -> Tuple[np.ndarray, nib.Nifti1Header]:
    """
    Load NIfTI image.
    
    Args:
        file_path: Path to NIfTI file
        
    Returns:
        Image data and header
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    nii = nib.load(file_path)
    data = nii.get_fdata()
    header = nii.header
    
    return data, header


def save_nifti(
    data: np.ndarray,
    file_path: str,
    affine: Optional[np.ndarray] = None,
    header: Optional[nib.Nifti1Header] = None
):
    """
    Save NIfTI image.
    
    Args:
        data: Image data
        file_path: Output file path
        affine: Affine transformation matrix
        header: NIfTI header
    """
    # Create default affine if not provided
    if affine is None:
        affine = np.eye(4)
    
    # Create NIfTI image
    nii = nib.Nifti1Image(data, affine, header)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save image
    nib.save(nii, file_path)


def get_image_info(file_path: str) -> dict:
    """
    Get information about a NIfTI image.
    
    Args:
        file_path: Path to NIfTI file
        
    Returns:
        Dictionary with image information
    """
    data, header = load_nifti(file_path)
    
    info = {
        "file_path": file_path,
        "shape": data.shape,
        "dtype": data.dtype,
        "min_value": np.min(data),
        "max_value": np.max(data),
        "mean_value": np.mean(data),
        "std_value": np.std(data),
        "voxel_size": header.get_zooms(),
        "file_size_mb": os.path.getsize(file_path) / (1024 * 1024)
    }
    
    return info


def check_data_integrity(images_dir: str, labels_dir: str) -> dict:
    """
    Check data integrity between images and labels.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        
    Returns:
        Dictionary with integrity check results
    """
    import glob
    
    # Get all CT files
    ct_pattern = os.path.join(images_dir, "*__CT.nii.gz")
    ct_files = glob.glob(ct_pattern)
    
    # Expected image size (200x200x310)
    EXPECTED_SHAPE = (200, 200, 310)
    
    results = {
        "total_cases": 0,
        "missing_pet": [],
        "missing_labels": [],
        "shape_mismatches": [],
        "unexpected_size": [],  # New field for images not 200x200x310
        "valid_cases": []
    }
    
    for ct_file in ct_files:
        case_id = os.path.basename(ct_file).replace("__CT.nii.gz", "")
        results["total_cases"] += 1
        
        # Check for corresponding PET file
        pet_file = os.path.join(images_dir, f"{case_id}__PT.nii.gz")
        if not os.path.exists(pet_file):
            results["missing_pet"].append(case_id)
            continue
        
        # Check for corresponding label file
        label_file = os.path.join(labels_dir, f"{case_id}.nii.gz")
        if not os.path.exists(label_file):
            results["missing_labels"].append(case_id)
            continue
        
        # Check shapes
        try:
            ct_data, _ = load_nifti(ct_file)
            pet_data, _ = load_nifti(pet_file)
            label_data, _ = load_nifti(label_file)
            
            # Check if shapes match between modalities
            if ct_data.shape != pet_data.shape or ct_data.shape != label_data.shape:
                results["shape_mismatches"].append({
                    "case_id": case_id,
                    "ct_shape": ct_data.shape,
                    "pet_shape": pet_data.shape,
                    "label_shape": label_data.shape
                })
                continue
            
            # Check if shape matches expected dimensions (200x200x310)
            if ct_data.shape != EXPECTED_SHAPE:
                results["unexpected_size"].append({
                    "case_id": case_id,
                    "actual_shape": ct_data.shape,
                    "expected_shape": EXPECTED_SHAPE
                })
                continue
                
            # If all checks pass, add to valid cases
            results["valid_cases"].append(case_id)
                
        except Exception as e:
            print(f"Error checking case {case_id}: {e}")
    
    return results


def create_data_summary(data_dir: str, images_folder: str, labels_folder: str) -> dict:
    """
    Create a summary of the dataset.
    
    Args:
        data_dir: Root data directory
        
    Returns:
        Dictionary with dataset summary
    """
    images_dir = os.path.join(data_dir, images_folder)
    labels_dir = os.path.join(data_dir, labels_folder)
    
    # Check integrity
    integrity = check_data_integrity(images_dir, labels_dir)
    
    # Get sample image info
    sample_info = None
    if integrity["valid_cases"]:
        sample_case = integrity["valid_cases"][0]
        sample_ct = os.path.join(images_dir, f"{sample_case}__CT.nii.gz")
        sample_info = get_image_info(sample_ct)
    
    summary = {
        "dataset_path": data_dir,
        "integrity_check": integrity,
        "sample_image_info": sample_info,
        "recommendations": []
    }
    
    # Add recommendations
    if integrity["missing_pet"]:
        summary["recommendations"].append(f"Missing PET files for {len(integrity['missing_pet'])} cases")
    
    if integrity["missing_labels"]:
        summary["recommendations"].append(f"Missing label files for {len(integrity['missing_labels'])} cases")
    
    if integrity["shape_mismatches"]:
        summary["recommendations"].append(f"Shape mismatches in {len(integrity['shape_mismatches'])} cases")
    
    if not summary["recommendations"]:
        summary["recommendations"].append("Dataset appears to be complete and consistent")
    
    return summary

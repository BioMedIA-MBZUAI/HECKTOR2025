"""
HECKTOR Survival Model Inference for Container Deployment
This script processes a single patient and outputs recurrence-free survival prediction.
"""

from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np
import torch
from resources.utils import HecktorInferenceModel
import os

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

def run():
    """Main inference function."""
    # Read the input
    input_pet_image = load_image_file_as_array(
        location=INPUT_PATH / "images/pet",
    )
    input_ct_image = load_image_file_as_array(
        location=INPUT_PATH / "images/ct",
    )
    input_electronic_health_record = load_json_file(
        location=INPUT_PATH / "ehr.json",
    )
    
    # We don't use these for our model, but they're part of the interface
    try:
        ct_planning = load_image_file_as_array(
            location=INPUT_PATH / "images/ct-planning",
        )

    except FileNotFoundError:
        print(f"No image files found in {INPUT_PATH}/images/ct-planning")
        ct_planning = None
    
    try:
        rt_dose_map = load_image_file_as_array(
            location=INPUT_PATH / "images/rt-dose",
        )
    except FileNotFoundError:
        print(f"No image files found in {INPUT_PATH}/images/rt-dose")
        rt_dose_map = None
    
    # Show torch info
    show_torch_cuda_info()
    
    try:
        # Initialize the inference model
        print("Loading HECKTOR survival model...")
        model = HecktorInferenceModel(resource_path=RESOURCE_PATH)
        
        # Process the inputs and generate prediction
        print("Processing patient data...")
        output_recurrence_free_survival = model.predict_single_patient(
            ct_image=input_ct_image,
            pet_image=input_pet_image,
            clinical_data=input_electronic_health_record
        )
        
        print(f"Predicted RFS risk score: {output_recurrence_free_survival:.6f}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise e
    
    # Save the output
    write_json_file(
        location=OUTPUT_PATH / "rfs.json", 
        content=float(output_recurrence_free_survival)
    )
    
    return 0

def load_json_file(*, location):
    """Reads a json file."""
    with open(location, "r") as f:
        return json.loads(f.read())

def write_json_file(*, location, content):
    """Writes a json file."""
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

def load_image_file_as_array(*, location):
    """Use SimpleITK to read a file."""
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
        + glob(str(location / "*.nii"))
        + glob(str(location / "*.nii.gz"))
    )
    
    if not input_files:
        raise FileNotFoundError(f"No image files found in {location}")
    
    result = SimpleITK.ReadImage(input_files[0])
    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)

def show_torch_cuda_info():
    """Display PyTorch CUDA information."""
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

if __name__ == "__main__":
    raise SystemExit(run())
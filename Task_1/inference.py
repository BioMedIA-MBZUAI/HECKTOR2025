"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy
import os
import numpy as np
from monai.transforms import SaveImageD

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

from resources.preprocess import resample_images, crop_neck_region_sitk, apply_monai_transforms
from resources.utils import load_model_from_checkpoint, arrays_to_tensor, run_inference
from resources.postprocess import prediction_to_original_space

def run():
    # Read the input
    ct_path  = load_image_file_as_array(
        location=INPUT_PATH / "images/ct",
    )
    input_electronic_health_record = load_json_file(
        location=INPUT_PATH / "ehr.json",
    )
    pt_path = load_image_file_as_array(
        location=INPUT_PATH / "images/pet",
    )

    # resample the images to 1mm isotropic resolution
    ct, pt, bb= resample_images(
        ct_path=ct_path,
        pet_path=pt_path,
    )
    # crop the images to the bounding box
    ct_cropped, pet_cropped, box_start, box_end = crop_neck_region_sitk(
        ct_sitk=ct,
        pet_sitk=pt,
    )

    # apply transformation to the cropped images
    ct_transformed, pet_transformed, meta = apply_monai_transforms(ct_cropped, pet_cropped)
    input_tensor = arrays_to_tensor(ct_transformed, pet_transformed)
    input_tensor = input_tensor.permute(0, 1, 3, 4, 2).contiguous()


    # # Load the model and run inference
    model_path = RESOURCE_PATH / "checkpoints" / "best_model.pth"
    model, config = load_model_from_checkpoint(model_path, device="cuda")

    prediction = run_inference(model, input_tensor, config, device="cuda", use_sliding_window=True)
    
    pred_np = prediction.transpose(0, 3, 1, 2)[0]     # → (310, 200, 200)

        
    mask_orig = prediction_to_original_space(
        pred_np,         
        meta,                        # from apply_monai_transforms
        box_start, box_end,
        ct,                          # 1-mm resampled CT
        SimpleITK.ReadImage(ct_path),     # original-resolution CT
    )
 

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/tumor-lymph-node-segmentation",
        array=mask_orig,
    )
    return 0


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    return input_files[0]


def write_array_as_image_file(*, location, array, filename="output.mha"):
    location.mkdir(parents=True, exist_ok=True)

    if isinstance(array, SimpleITK.Image):
        img = array                       # already SimpleITK
    else:                                 # assume NumPy
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]              # drop batch dim if present
        img = SimpleITK.GetImageFromArray(array)

    SimpleITK.WriteImage(img, str(location / filename), useCompression=True)



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

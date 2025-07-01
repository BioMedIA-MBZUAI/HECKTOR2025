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

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

from resources.preprocess import resample_images, crop_neck_region
from resources.utils import load_model_from_checkpoint, arrays_to_tensor, run_inference
from resources.postprocess import prediction_to_original_space

def run():
    # Read the input
    input_ct_image, input_ct_sitk  = load_image_file_as_array(
        location=INPUT_PATH / "images/ct",
    )
    input_electronic_health_record = load_json_file(
        location=INPUT_PATH / "ehr.json",
    )
    input_pet_image, input_pet_sitk = load_image_file_as_array(
        location=INPUT_PATH / "images/pet",
    )

    # Preprocess the inputs and save the bounding boxes for postprocessing
    ct_resampled, pet_resampled, bb = resample_images(
        ct=input_ct_sitk,
        pt=input_pet_sitk,
    )
    ct_cropped, pet_cropped, box_start, box_end = crop_neck_region(
        ct_sitk=ct_resampled,
        pet_sitk=pet_resampled,
    )

    input_tensor = arrays_to_tensor(
        ct_array=ct_cropped,
        pet_array=pet_cropped
    )
    
    # Load the model and run inference
    model_path = RESOURCE_PATH / "checkpoints" / "best_model.pth"
    model, config = load_model_from_checkpoint(model_path, device="cuda")

    prediction = run_inference(model, input_tensor, config, device="cuda", use_sliding_window=True)
    pred_np   = np.squeeze(prediction, axis=0)   # [D,H,W] uint8

    # Post-process the prediction
    # 1-mm bbox geometry image

    output_tumor_and_lymph_node_segmentation = prediction_to_original_space(
        pred_crop_np=pred_np,
        box_start=box_start,
        box_end=box_end,
        resampled_ct_sitk=ct_resampled,
        original_ct_sitk=input_ct_sitk,     
    )


    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/tumor-lymph-node-segmentation",
        array=output_tumor_and_lymph_node_segmentation,
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
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result), result


def write_array_as_image_file(*, location, array, filename="output.mha"):
    location.mkdir(parents=True, exist_ok=True)

    if isinstance(array, SimpleITK.Image):
        img = array                       # already SimpleITK
    else:                                 # assume NumPy
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]              # drop batch dim if present
        img = SimpleITK.GetImageFromArray(array.astype(np.uint8))

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

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

# Paths inside the container, do not change
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

# Import the necessary functions from the resources
from resources.model import MultiModalResNet
from resources.hecktor_inference import prepare_input_tensor, preprocess_ehr, run_inference, resample_images, crop_neck_region
import torch
from joblib import load


def run():
    # Read the input
    input_ct_image = load_image_file_as_array(
        location=INPUT_PATH / "images/ct",
    )
    input_electronic_health_record = load_json_file(
        location=INPUT_PATH / "ehr.json",
    )
    input_pet_image = load_image_file_as_array(
        location=INPUT_PATH / "images/pet",
    )

    _show_torch_cuda_info()

    # Preprocess the inputs
    ct_resampled_array, pet_resampled_array = resample_images(
        ct_array=input_ct_image,
        pet_array=input_pet_image,
    )

    ct_cropped, pet_cropped = crop_neck_region(
        ct_array=ct_resampled_array,
        pet_array=pet_resampled_array,
    )
    
    x_img = prepare_input_tensor(ct_cropped, pet_cropped)

    scaler = load(RESOURCE_PATH / "checkpoints/scaler.joblib")
    ohe = load(RESOURCE_PATH / "checkpoints/ohe.joblib")
    x_clin = preprocess_ehr(input_electronic_health_record, scaler, ohe)

    # Load HECKTOR model and make prediction
    model = MultiModalResNet(clin_feat_dim=x_clin.shape[1], num_classes=2).cuda()
    model.load_state_dict(torch.load(RESOURCE_PATH / "checkpoints/best_model.pt"))

    output_hpv_status = run_inference(model=model, x_img=x_img, x_clin=x_clin)

    # Save your output
    write_json_file(location=OUTPUT_PATH / "hpv-status.json", content=output_hpv_status)

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

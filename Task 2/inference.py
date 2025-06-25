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

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
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
    input_dosimetry_ct = load_image_file_as_array(
        location=INPUT_PATH / "images/dosimetry-ct",
    )
    input_radiotherapy_planning_dose_map = load_image_file_as_array(
        location=INPUT_PATH / "images/radiotherapy-planning-dose-map",
    )

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Load HECKTOR model and make prediction
    try:
        from resources.hecktor_inference import load_model, predict_survival
        model = load_model(RESOURCE_PATH / "best_model.pth")
        output_recurrence_free_survival = predict_survival(
            model=model,
            ct_image=input_ct_image,
            pet_image=input_pet_image,
            ehr_data=input_electronic_health_record
        )

        # Save your output
        write_json_file(
            location=OUTPUT_PATH / "rfs.json", content=output_recurrence_free_survival
        )
        
    except Exception as e:
        print(f"Model prediction failed: {e}")
        raise e

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
import argparse
from pathlib import Path

import numpy as np
import requests
import timm
import torch
from comet_ml import API

from astropy.io import fits

from arccnet.models import utilities as ut


def download_model(api, workspace, model_name, model_version, model_path):
    model_comet = api.get_model(workspace, model_name)
    model_assets = model_comet.get_assets(model_version)
    for asset in model_assets:
        if asset["fileName"] == "model-data/comet-torch-model.pth":
            model_url = asset["s3Link"]
            break
    if model_path.exists():
        print(f"Model file already exists at {model_path}. \nSkipping download.")
    else:
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            print(f"Model downloaded successfully and saved to {model_path}")
        else:
            print(f"Failed to download model. Status code: {response.status_code}")


def preprocess_fits_data(fits_file_path, hardtanh=True, target_height=224, target_width=224):
    with fits.open(fits_file_path, memmap=True) as img_fits:
        image_data = np.array(img_fits[1].data, dtype=np.float32)
    image_data = np.nan_to_num(image_data, nan=0.0)
    if hardtanh:
        image_data = ut.hardtanh_transform_npy(image_data, divisor=800, min_val=-1.0, max_val=1.0)
    image_data = ut.pad_resize_normalize(image_data, target_height=target_height, target_width=target_width)
    return torch.from_numpy(image_data).unsqueeze(0)


def run_inference(model, fits_file_path, device):
    model.eval()
    with torch.no_grad():
        data = preprocess_fits_data(fits_file_path)
        data = data.unsqueeze(0).to(device)  # dimensions: (batch_size, channels, height, width)
        output = model(data)
    return output.cpu().numpy()


def main(args):
    api = API()
    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir.parent / "trained_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.model_name}-{args.model_version}.pth"

    download_model(api, args.workspace, args.model_name, args.model_version, model_path)

    # Find number of classes from project name
    substring_after_v2 = args.project_name.split("arcaff-v2-")[1]
    values = substring_after_v2.split("-")
    num_classes = len(values)

    # Create the model
    model = timm.create_model(args.model_name, num_classes=num_classes, in_chans=1)
    ut.replace_activations(model, torch.nn.ReLU, torch.nn.LeakyReLU, negative_slope=0.01)

    # Load the model state
    device = "cpu"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Run inference
    print(f"FITS file: {args.fits_file_path}")
    result = run_inference(model, args.fits_file_path, device)
    predicted_class = np.argmax(result)
    probabilities = torch.softmax(torch.tensor(result), dim=1).numpy()

    print("Normalized Predictions:", probabilities)
    print("Predicted class:", ut.index_to_label[predicted_class])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on FITS data using a pre-trained model.")
    parser.add_argument(
        "--fits_file_path",
        type=str,
        default="/ARCAFF/data/arccnet-cutout-dataset-v20240715/fits/19970125_235945_QS-5_MDI.fits",
        help="Path to the FITS file.",
    )
    parser.add_argument("--project_name", type=str, default="arcaff-v2-qs-ia-a-b-bg", help="Name of the project.")
    parser.add_argument("--workspace", type=str, default="arcaff", help="Workspace name in Comet.ml.")
    parser.add_argument("--model_name", type=str, default="resnet10t", help="Model name.")
    parser.add_argument("--model_version", type=str, default="1.0.0", help="Model version.")

    args = parser.parse_args()
    main(args)

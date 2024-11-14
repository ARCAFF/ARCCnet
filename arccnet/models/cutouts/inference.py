import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import timm
import torch
from comet_ml import API

from astropy.io import fits

from arccnet.models import train_utils as ut_t
from arccnet.models.cutouts import config
from arccnet.utils.logging import get_logger
from arccnet.visualisation import utils as ut_v

# Initialize the logger
logger = get_logger(__name__)


def download_model(api, workspace, model_name, model_version, model_path):
    try:
        model_comet = api.get_model(workspace, model_name)
        model_assets = model_comet.get_assets(model_version)
        model_url = None
        for asset in model_assets:
            if asset["fileName"] == "model-data/comet-torch-model.pth":
                model_url = asset["s3Link"]
                break
        if model_url is None:
            logger.error("Model URL not found in assets.")
            return

        if model_path.exists():
            logger.info(f"Model file already exists at {model_path}. \nSkipping download.")
        else:
            response = requests.get(model_url)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Model downloaded successfully and saved to {model_path}")
            else:
                logger.error(f"Failed to download model. Status code: {response.status_code}")
    except Exception:
        logger.exception("An error occurred while downloading the model.")


def preprocess_fits_data(fits_file_path, hardtanh=True, target_height=224, target_width=224):
    try:
        with fits.open(fits_file_path, memmap=True) as img_fits:
            image_data = np.array(img_fits[1].data, dtype=np.float32)
        image_data = np.nan_to_num(image_data, nan=0.0)
        if hardtanh:
            image_data = ut_v.hardtanh_transform_npy(image_data, divisor=800, min_val=-1.0, max_val=1.0)
        image_data = ut_v.pad_resize_normalize(image_data, target_height=target_height, target_width=target_width)
        return torch.from_numpy(image_data).unsqueeze(0)
    except Exception:
        logger.exception(f"Failed to preprocess FITS data from {fits_file_path}.")
        raise


def run_inference(model, fits_file_path, device):
    try:
        model.eval()
        with torch.no_grad():
            data = preprocess_fits_data(fits_file_path)
            data = data.to(device)  # Removed extra unsqueeze since preprocess already has unsqueeze
            output = model(data)
        return output.cpu().numpy()
    except Exception:
        logger.exception("An error occurred during model inference.")
        raise


def predict(args):
    try:
        api = API()
        script_dir = Path(__file__).parent.resolve()
        output_dir = script_dir.parent / "trained_models"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{args.model_name}-{args.model_version}.pth"

        download_model(api, args.workspace, args.model_name, args.model_version, model_path)

        # Find number of classes from project name
        try:
            substring_after_v2 = args.project_name.split("arcaff-v2-")[1]
            values = substring_after_v2.split("-")
            num_classes = len(values)
        except IndexError:
            logger.error("Project name format is incorrect. Expected 'arcaff-v2-<classes>'.")
            return

        # Create the model
        try:
            model = timm.create_model(args.model_name, num_classes=num_classes, in_chans=1)
            ut_t.replace_activations(model, torch.nn.ReLU, torch.nn.LeakyReLU, negative_slope=0.01)
        except Exception:
            logger.exception("Failed to create or modify the model architecture.")
            return

        # Load the model state
        device = "cpu"
        try:
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            logger.info(f"Model state loaded from {model_path}")
        except Exception:
            logger.exception(f"Failed to load model state from {model_path}.")
            return

        # Run inference
        logger.info(f"FITS file: {args.fits_file_path}")
        result = run_inference(model, args.fits_file_path, device)
        predicted_class = np.argmax(result)
        probabilities = torch.softmax(torch.tensor(result), dim=1).numpy()
        df_prob = pd.DataFrame(probabilities, columns=[ut_t.index_to_label[idx] for idx in range(num_classes)])
        logger.info("\nPredictions:\n" + df_prob.to_string(index=False))
        logger.info(f"\nPredicted class: {ut_t.index_to_label[predicted_class]}")
    except Exception:
        logger.exception("An unexpected error occurred during prediction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on FITS data using a pre-trained model.")
    parser.add_argument(
        "--fits_file_path",
        type=str,
        default=os.path.join(
            config.data_folder, config.dataset_folder, "fits", "20160203_235809_I-12493_HMI_SIDE1.fits"
        ),
        help="Path to the FITS file.",
    )
    parser.add_argument("--project_name", type=str, default="arcaff-v2-qs-ia-a-b-bg", help="Name of the project.")
    parser.add_argument("--workspace", type=str, default="arcaff", help="Workspace name in Comet.ml.")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model name.")
    parser.add_argument("--model_version", type=str, default="1.0.0", help="Model version.")

    args = parser.parse_args()
    predict(args)

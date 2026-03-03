"""
Dataset utilities for Hale classification using existing ARCCNet utilities.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2

import arccnet.models.cutouts.hale.config as config
from arccnet.models import preprocessing_common as pp_common

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_transforms(is_training: bool = False):
    """Create torchvision transform pipeline."""
    return (
        v2.Compose(
            [
                v2.RandomVerticalFlip(p=config.VERTICAL_FLIP_PROB),
                v2.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
                v2.RandomPerspective(distortion_scale=config.PERSPECTIVE_DISTORTION_SCALE, p=config.PERSPECTIVE_PROB),
                v2.RandomAffine(
                    degrees=config.ROTATION_DEGREES,
                    translate=config.AFFINE_TRANSLATE,
                    scale=config.AFFINE_SCALE,
                    shear=config.AFFINE_SHEAR,
                ),
            ]
        )
        if is_training and config.USE_AUGMENTATION
        else None
    )


def get_fold_data(df: pd.DataFrame, fold_num: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get train/val/test splits for a specific fold."""
    return tuple(df[df[f"Fold {fold_num}"] == split].copy() for split in ("train", "val", "test"))


def load_image(row, data_type="magnetogram"):
    """Load magnetogram or continuum FITS file as np.ndarray."""
    magnetogram_path = pp_common.resolve_preferred_cutout_fits_path(
        row,
        data_folder=config.DATA_FOLDER,
        dataset_folder=config.DATASET_FOLDER,
    )
    if magnetogram_path is None:
        raise RuntimeError(f"No cutout FITS path available for row index {getattr(row, 'name', 'unknown')}")

    if data_type == "magnetogram":
        fits_file_path = magnetogram_path
    elif data_type == "continuum":
        mag_name = Path(magnetogram_path).name
        continuum_names = [
            mag_name.replace("_mag_", "_cont_"),
            mag_name.replace(".magnetogram.", ".continuum."),
        ]
        continuum_names = list(dict.fromkeys(continuum_names))

        fits_file_path = None
        for continuum_name in continuum_names:
            candidate = Path(magnetogram_path).with_name(continuum_name)
            if candidate.exists():
                fits_file_path = candidate
                break
            resolved = pp_common.resolve_cutout_fits_path(
                continuum_name,
                data_folder=config.DATA_FOLDER,
                dataset_folder=config.DATASET_FOLDER,
            )
            if resolved is not None:
                fits_file_path = resolved
                break

        if fits_file_path is None:
            raise RuntimeError(f"Failed to resolve continuum FITS for magnetogram {magnetogram_path}")
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    try:
        return pp_common.load_fits_hdu_data(fits_file_path, hdu_index=1, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to load FITS file {fits_file_path}: {exc}") from exc


def normalize_continuum(image: np.ndarray) -> np.ndarray:
    """
    Normalize continuum image using inverted min-max scaling.
    Output is in [0, 1] with background near 0 and brightest points near 1.
    NaNs are set to 0.
    """
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image, dtype=np.float32)
    min_val = np.nanmin(image[finite])
    max_val = np.nanmax(image[finite])
    denom = max_val - min_val
    if denom == 0 or not np.isfinite(denom):
        norm = np.zeros_like(image, dtype=np.float32)
    else:
        norm = 1.0 - (image - min_val) / denom
    norm = np.clip(norm, 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0, copy=False)
    return norm.astype(np.float32)


class HaleDataset(Dataset):
    """Dataset for Hale classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_type: str = "magnetogram",
        divisor: float = None,
        target_height: int = None,
        target_width: int = None,
        is_training: bool = False,
    ):
        self.df = df.copy()
        self.data_type = data_type
        self.divisor = divisor or config.IMAGE_DIVISOR
        self.target_height = target_height or config.IMAGE_TARGET_HEIGHT
        self.target_width = target_width or config.IMAGE_TARGET_WIDTH
        self.is_training = is_training
        self.transform = get_transforms(is_training=is_training)
        if "model_labels" not in self.df.columns:
            raise ValueError("DataFrame must contain 'model_labels' column with contiguous indices starting from 0")
        unique_labels = sorted(self.df["model_labels"].unique())
        if unique_labels != list(range(len(unique_labels))):
            raise ValueError(
                f"model_labels must be contiguous starting from 0. Got: {unique_labels}, Expected: {list(range(len(unique_labels)))}"
            )
        if not hasattr(HaleDataset, "_logged_labels"):
            logger.info(f"Dataset using model_labels: {[int(label) for label in unique_labels]}")
            if "grouped_labels" in self.df.columns:
                grouped_mapping = self.df.groupby("model_labels")["grouped_labels"].first().sort_index().to_dict()
                logger.info("Original grouped_labels mapping: %s", grouped_mapping)
            HaleDataset._logged_labels = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.data_type == "magnetogram":
            image = torch.from_numpy(np.nan_to_num(load_image(row, "magnetogram"), nan=0.0)).unsqueeze(0)
            image = torch.clamp(image / self.divisor, config.HARDTANH_MIN_VAL, config.HARDTANH_MAX_VAL)
            image = transforms.functional.resize(image, [self.target_height, self.target_width], antialias=True)
        elif self.data_type == "continuum":
            image = load_image(row, "continuum")
            image = normalize_continuum(image)
            image = torch.from_numpy(image).unsqueeze(0)
            image = transforms.functional.resize(image, [self.target_height, self.target_width], antialias=True)
        elif self.data_type == "both":
            # Load both magnetogram and continuum, stack as channels
            mag = torch.from_numpy(np.nan_to_num(load_image(row, "magnetogram"), nan=0.0))
            mag = torch.clamp(mag / self.divisor, config.HARDTANH_MIN_VAL, config.HARDTANH_MAX_VAL)
            cont = load_image(row, "continuum")
            cont = normalize_continuum(cont)
            stacked = torch.stack([mag, torch.from_numpy(cont)], dim=0)
            image = transforms.functional.resize(stacked, [self.target_height, self.target_width], antialias=True)
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(row["model_labels"], dtype=torch.long)

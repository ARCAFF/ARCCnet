"""Utility helpers for flare datasets."""

import re
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from arccnet.models import preprocessing_common as pp_common

logger = logging.getLogger(__name__)

FLARE_CLASSES = ["A", "B", "C", "M", "X"]
MAG_CLASS_MAPPING = {
    "Alpha": "α",
    "Beta": "β",
    "Beta-Delta": "β-δ",
    "Beta-Gamma": "β-γ",
    "Beta-Gamma-Delta": "β-γ-δ",
    "Gamma": "γ",
    "Gamma-Delta": "γ-δ",
}
MAG_CLASS_ORDER = ["α", "β", "β-δ", "β-γ", "β-γ-δ", "γ", "γ-δ"]

_CUTOUT_KEY_PATTERN = re.compile(r"(\d{8})_\d{6}_[A-Z]+-(\d+).*_([A-Z]+)(?:_SIDE\d+)?\.fits")


def _extract_cutout_key(filename: str) -> tuple[str, str, str] | None:
    match = _CUTOUT_KEY_PATTERN.match(filename)
    if not match:
        return None
    return match.groups()


def _build_cutout_key_map(
    data_folder: str, dataset_folder: str, image_type: str = "magnetograms"
) -> dict[tuple[str, str, str], Path]:
    """Build a mapping from cutout keys to FITS file paths."""
    key_to_path: dict[tuple[str, str, str], Path] = {}
    scanned_files = 0
    prefer_magnetogram = image_type == "magnetograms"

    for fits_dir in pp_common.candidate_cutout_fits_dirs(data_folder, dataset_folder):
        if not fits_dir.exists():
            continue
        for path in fits_dir.glob("*.fits"):
            scanned_files += 1
            key = _extract_cutout_key(path.name)
            if key is None:
                continue
            existing = key_to_path.get(key)
            is_mag = "_mag_" in path.name.lower()
            if existing is None or (prefer_magnetogram and is_mag) or (not prefer_magnetogram and not is_mag):
                key_to_path[key] = path

    logger.info("%d FITS files present. %d key mappings considered (%s)", scanned_files, len(key_to_path), image_type)
    return key_to_path


def check_fits_file_existence(df, data_folder, dataset_folder, image_type: str = "magnetograms"):
    """Add resolved FITS paths and a boolean existence flag to the dataframe."""
    df = df.copy()
    df["file_exists"] = False
    df["resolved_path"] = None
    missing_path_indices = []

    hmi_col = "path_image_cutout_hmi"
    mdi_col = "path_image_cutout_mdi"
    key_to_file = _build_cutout_key_map(data_folder, dataset_folder, image_type)

    for index, row in df.iterrows():
        hmi_path = row.get(hmi_col)
        mdi_path = row.get(mdi_col)

        if pp_common.is_missing_path_value(hmi_path) and pp_common.is_missing_path_value(mdi_path):
            missing_path_indices.append(index)
            continue

        resolved = None
        resolved_col = None
        for col, value in ((hmi_col, hmi_path), (mdi_col, mdi_path)):
            if pp_common.is_missing_path_value(value):
                continue
            value_text = str(value).strip()
            candidate = pp_common.resolve_cutout_fits_path(
                value,
                data_folder=data_folder,
                dataset_folder=dataset_folder,
            )
            if candidate is None:
                key = _extract_cutout_key(Path(value_text).name)
                if key is not None:
                    candidate = key_to_file.get(key)
            if candidate is not None:
                resolved = candidate
                resolved_col = col
                break

        if resolved is not None:
            df.loc[index, "file_exists"] = True
            df.loc[index, "resolved_path"] = str(resolved)
            if resolved_col is not None:
                df.loc[index, resolved_col] = resolved.name

    files_found = df["file_exists"].sum()
    logger.info(f"Found existing files for {files_found}/{len(df)} rows")

    return df, missing_path_indices


def split_dataframe(df, stratify_col, test_size=0.1, val_size=0.2, random_state=42):
    """Split dataframe into train/val/test with AR-aware stratification."""
    if test_size + val_size >= 1:
        raise ValueError("Combined test and validation sizes must be less than 1")
    if stratify_col not in df.columns:
        raise ValueError(f"Stratification column '{stratify_col}' not found in dataframe")

    ar_groups = df["number"].unique()
    ar_labels = df.groupby("number")[stratify_col].max().loc[ar_groups].values

    train_val_ars, test_ars = train_test_split(
        ar_groups, test_size=test_size, stratify=ar_labels, random_state=random_state
    )

    train_val_labels = df.groupby("number")[stratify_col].max().loc[train_val_ars].values
    adjusted_val_size = val_size / (1 - test_size)  # Relative to train_val size

    train_ars, val_ars = train_test_split(
        train_val_ars, test_size=adjusted_val_size, stratify=train_val_labels, random_state=random_state
    )

    train_df = df[df["number"].isin(train_ars)]
    val_df = df[df["number"].isin(val_ars)]
    test_df = df[df["number"].isin(test_ars)]

    overlap_tv = len(set(train_ars) & set(val_ars))
    overlap_tt = len(set(train_ars) & set(test_ars))
    overlap_vt = len(set(val_ars) & set(test_ars))
    logger.info(
        "ARs - Train: %d, Val: %d, Test: %d | Overlaps - Train/Val: %d, Train/Test: %d, Val/Test: %d",
        train_df["number"].nunique(),
        val_df["number"].nunique(),
        test_df["number"].nunique(),
        overlap_tv,
        overlap_tt,
        overlap_vt,
    )

    return train_df, val_df, test_df

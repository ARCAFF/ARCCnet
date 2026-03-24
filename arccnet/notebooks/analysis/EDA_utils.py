"""
Utility functions for Exploratory Data Analysis (EDA) of ARCCnet cutout data.
"""

from pathlib import Path

import numpy as np

from astropy.io import fits

# Quality flag definitions
QUALITY_FLAGS = {
    "SOHO/MDI": {
        0x00000001: "Missing Data",
        0x00000002: "Saturated Pixel",
        0x00000004: "Truncated (Top)",
        0x00000008: "Truncated (Bottom)",
        0x00000200: "Shutterless Mode",
        0x00010000: "Cosmic Ray",
        0x00020000: "Calibration Mode",
        0x00040000: "Image Bad",
    },
    "SDO/HMI": {
        0x00000020: "Missing >50% Data",
        0x00000080: "Limb Darkening Correction Bad",
        0x00000400: "Shutterless Mode",
        0x00001000: "Partial/Missing Frame",
        0x00010000: "Cosmic Ray",
    },
}


def decode_flags(flag_hex, flag_dict):
    """
    Decode hexadecimal quality flag to human-readable status.
    """
    try:
        flag_str = str(flag_hex).strip().lstrip("0x")
        if not flag_str or flag_str in ["nan", "None", "<NA>"]:
            return "Good Quality"
        flag_int = int(flag_str, 16)
        if flag_int == 0:
            return "Good Quality"
        meanings = [meaning for bit_val, meaning in flag_dict.items() if flag_int & bit_val]
        return " | ".join(meanings) or "Unknown Flag"
    except (ValueError, TypeError):
        return "Invalid Format"


def analyze_quality_flags(df, instrument_name):
    """
    Analyze and summarize quality flags for "SOHO/MDI" or "SDO/HMI".

    Returns
    -------
    pandas.DataFrame or None
        Flag statistics with columns ['Flag_Hex', 'Count', 'Percentage', 'Description'].
        Returns None if quality column missing or DataFrame empty.
    """
    quality_column = "QUALITY_mdi" if instrument_name == "SOHO/MDI" else "QUALITY_hmi"

    if quality_column not in df.columns or len(df) == 0:
        return None

    series = (
        df[quality_column]
        .astype(str)
        .replace(["nan", "None", "<NA>", ""], "00000000")
        .str.strip()
        .str.replace("0x", "", regex=False)
    )

    counts = series.value_counts().reset_index()
    counts.columns = ["Flag", "Count"]
    total = counts["Count"].sum()

    return (
        counts.assign(
            Percentage=(counts["Count"] / total * 100).round(2).apply(lambda p: f"{p:.2f}%"),
            Flag_Hex=counts["Flag"].apply(lambda f: f"0x{f.upper()}"),
            Description=counts["Flag"].apply(lambda f: decode_flags(f, QUALITY_FLAGS[instrument_name])),
        )[["Flag_Hex", "Count", "Percentage", "Description"]]
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )


def analyze_nan_pattern(data, longitude):
    """
    Analyze NaN patterns considering longitude position.

    Parameters
    ----------
    data : numpy.ndarray
        2D array to analyze for NaN patterns.
    longitude : float
        Longitude position in degrees for limb detection.

    Returns
    -------
    dict
        Dictionary with NaN fraction statistics and position info.
    """
    nan_mask = np.isnan(data)
    total_nans = np.sum(nan_mask)

    # Check for instrumental artifacts
    rows_all_nan = np.all(nan_mask, axis=1)
    cols_all_nan = np.all(nan_mask, axis=0)
    horizontal_nan_rows = np.sum(rows_all_nan)
    vertical_nan_cols = np.sum(cols_all_nan)

    # Calculate bar NaNs
    horizontal_bar_nans = horizontal_nan_rows * data.shape[1]
    vertical_bar_nans = vertical_nan_cols * data.shape[0]

    # Estimate limb vs instrumental NaNs
    abs_longitude = abs(longitude)
    is_near_limb = abs_longitude > 60

    if is_near_limb:
        edge_nans = total_nans - horizontal_bar_nans - vertical_bar_nans
        limb_nans = max(0, edge_nans)
        instrumental_nans = horizontal_bar_nans + vertical_bar_nans
    else:
        limb_nans = 0
        instrumental_nans = total_nans

    return {
        "total_nans": total_nans,
        "horizontal_nan_rows": horizontal_nan_rows,
        "vertical_nan_cols": vertical_nan_cols,
        "limb_nans": limb_nans,
        "instrumental_nans": instrumental_nans,
        "nan_fraction": total_nans / data.size,
        "longitude": longitude,
        "is_near_limb": is_near_limb,
    }


def compute_stats(data, longitude):
    """
    Compute statistics with longitude-informed NaN handling.

    Parameters
    ----------
    data : numpy.ndarray
        2D array to compute statistics for.
    longitude : float
        Longitude position in degrees.

    Returns
    -------
    dict
        Dictionary with statistical measures and NaN analysis.
    """
    nan_analysis = analyze_nan_pattern(data, longitude)
    valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        stats = {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "shape": data.shape,
            "valid_pixels": 0,
            "total_pixels": data.size,
        }
    else:
        with np.errstate(invalid="ignore"):
            stats = {
                "mean": np.nanmean(data),
                "median": np.nanmedian(data),
                "std": np.nanstd(data),
                "min": np.nanmin(data),
                "max": np.nanmax(data),
                "shape": data.shape,
                "valid_pixels": len(valid_data),
                "total_pixels": data.size,
            }

    stats.update(nan_analysis)
    return stats


def _resolve_fits_pair_paths(path_value, data_folder, dataset_folder):
    """
    Resolve magnetogram/continuum FITS paths across supported dataset layouts.

    Supports:
    1) New flat layout under <data_folder>/<dataset_folder>/<filename>.fits
    2) Legacy nested layout under .../data/cutout_classification/fits/<filename>.fits
    3) Relative paths stored in dataframe columns.
    """
    dataset_root = Path(data_folder) / dataset_folder
    raw_path = Path(str(path_value).strip())

    candidate_mag_paths = []
    if raw_path.is_absolute():
        candidate_mag_paths.append(raw_path)
    if str(path_value).strip():
        candidate_mag_paths.append(dataset_root / str(path_value).strip())
    candidate_mag_paths.append(dataset_root / raw_path.name)
    candidate_mag_paths.append(dataset_root / "data/cutout_classification/fits" / raw_path.name)

    # Preserve order while removing duplicates.
    seen = set()
    unique_candidates = []
    for candidate in candidate_mag_paths:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    checked = []
    for mag_path in unique_candidates:
        checked.append(str(mag_path))
        cont_path = mag_path.with_name(mag_path.name.replace("_mag_", "_cont_"))
        if mag_path.exists() and cont_path.exists():
            return mag_path, cont_path

    checked_locations = "\n - ".join(checked)
    raise FileNotFoundError(
        "Could not locate matching magnetogram/continuum FITS files. Checked:\n"
        f" - {checked_locations}"
    )


def load_and_analyze_fits_pair(idx, df_clean, data_folder, dataset_folder):
    """
    Load magnetogram and continuum FITS files and compute statistics.

    Returns
    -------
    dict
        Dictionary with loaded data, statistics, and metadata.
    """
    # Check if index is valid
    if idx >= len(df_clean):
        raise ValueError(f"Index {idx} is out of range. df_clean has {len(df_clean)} rows (0-{len(df_clean) - 1})")

    row = df_clean.iloc[idx]
    path = row["path_image_cutout_hmi"] if row["path_image_cutout_mdi"] == "" else row["path_image_cutout_mdi"]
    fits_magn_path, fits_cont_path = _resolve_fits_pair_paths(path, data_folder, dataset_folder)
    fits_magn_filename = fits_magn_path.name

    # Load data
    with fits.open(fits_magn_path) as hdul:
        mag_data = hdul[1].data
    with fits.open(fits_cont_path) as hdul:
        cont_data = hdul[1].data

    # Check if data is not empty
    if mag_data is None or mag_data.size == 0:
        raise ValueError(f"Magnetogram data is empty: {fits_magn_filename}")
    if cont_data is None or cont_data.size == 0:
        raise ValueError(f"Continuum data is empty: {fits_cont_path.name}")

    # Get longitude information
    longitude = row["longitude_hmi"] if row["path_image_cutout_mdi"] == "" else row["longitude_mdi"]

    return {
        "row": row,
        "mag_data": mag_data,
        "cont_data": cont_data,
        "mag_stats": compute_stats(mag_data, longitude),
        "cont_stats": compute_stats(cont_data, longitude),
        "mag_filename": fits_magn_filename,
        "cont_filename": fits_cont_path.name,
    }


def process_row(idx, df_clean, data_folder, dataset_folder):
    """
    Process a single row to extract statistics from FITS files.

    Returns
    -------
    dict or None
        Statistics dictionary or None if processing fails.
    """
    try:
        data = load_and_analyze_fits_pair(idx, df_clean, data_folder, dataset_folder)
        label = df_clean.iloc[idx]["label"]
        mcintosh_class = df_clean.iloc[idx]["mcintosh_class"]
        mag_stats = data["mag_stats"]
        cont_stats = data["cont_stats"]
        return {
            "index": idx,
            "label": label,
            "mcintosh_class": mcintosh_class,
            "mag_stats": mag_stats,
            "cont_stats": cont_stats,
        }
    except Exception as e:
        print(f"Failed at idx={idx}: {e}")
        return None

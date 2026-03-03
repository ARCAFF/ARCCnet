"""
Shared preprocessing helpers for cutout-based model pipelines.

This module centralizes quality/path/location filtering and FITS path loading so
cutouts and flares use consistent preprocessing behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MISSING_PATH_TOKENS = {"", "none", "nan", "<na>"}

# Canonical quality values after normalization by ``normalize_quality_flag``.
HMI_GOOD_QUALITY_FLAGS = {"", "0x00000000", "0x00000400"}
MDI_GOOD_QUALITY_FLAGS = {"", "0x00000000", "0x00000200"}
LEGACY_ARCAFF_PREFIX = "/mnt/ARCAFF/v0.3.0/"


def is_missing_path_value(value: object) -> bool:
    """Return True for empty/None/NaN-like path values."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_PATH_TOKENS
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def nonempty_path_mask(series: pd.Series) -> pd.Series:
    """Vectorized mask for non-empty path values."""
    text = series.astype("string").str.strip().str.lower()
    return series.notna() & (~text.isin(MISSING_PATH_TOKENS))


def _normalize_path_columns(path_cols: str | Iterable[str] | None) -> list[str]:
    if path_cols is None:
        return []
    if isinstance(path_cols, str):
        return [path_cols]
    return [col for col in path_cols if isinstance(col, str) and col]


def availability_mask(
    df: pd.DataFrame,
    path_cols: str | Iterable[str] | None,
) -> pd.Series:
    """
    Return per-row availability mask for one or more path columns.

    Any non-empty value across the provided columns marks the row as available.
    Missing columns are ignored.
    """
    columns = _normalize_path_columns(path_cols)
    if not columns:
        return pd.Series(False, index=df.index, dtype=bool)

    mask = pd.Series(False, index=df.index, dtype=bool)
    found = False
    for column in columns:
        if column in df.columns:
            mask |= nonempty_path_mask(df[column])
            found = True
    if not found:
        return pd.Series(False, index=df.index, dtype=bool)
    return mask


def normalize_quality_flag(value: object) -> str:
    """
    Normalize quality flags to ``0x########`` form.

    Empty/None-like values map to ``""``.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip().lower()
    if text in MISSING_PATH_TOKENS:
        return ""

    if text.startswith("0x"):
        hex_part = text[2:]
    else:
        hex_part = text

    if not hex_part:
        return ""

    if all(char in "0123456789abcdef" for char in hex_part):
        if len(hex_part) <= 8:
            hex_part = hex_part.zfill(8)
        return f"0x{hex_part}"

    # Fall back to lowercase original text for unexpected formats.
    return text


def is_good_quality_flag(value: object, instrument: str) -> bool:
    """Return whether a flag value is acceptable for the requested instrument."""
    normalized = normalize_quality_flag(value)
    if instrument.lower() == "hmi":
        return normalized in HMI_GOOD_QUALITY_FLAGS
    if instrument.lower() == "mdi":
        return normalized in MDI_GOOD_QUALITY_FLAGS
    raise ValueError(f"Unsupported instrument: {instrument}")


def apply_quality_filter(
    df: pd.DataFrame,
    hmi_path_col: str | Iterable[str] = "path_image_cutout_hmi",
    mdi_path_col: str | Iterable[str] = "path_image_cutout_mdi",
    hmi_quality_col: str = "QUALITY_hmi",
    mdi_quality_col: str = "QUALITY_mdi",
) -> pd.DataFrame:
    """
    Filter rows by instrument-aware quality flags.

    Rows are checked against the quality flag of the instrument(s) with available
    path(s). Missing instruments do not force a row to be removed.
    """
    hmi_available = availability_mask(df, hmi_path_col)
    mdi_available = availability_mask(df, mdi_path_col)

    if hmi_quality_col in df.columns:
        hmi_quality_good = df[hmi_quality_col].map(lambda value: is_good_quality_flag(value, "hmi"))
    else:
        logger.warning("Missing column '%s'. Skipping HMI quality filtering.", hmi_quality_col)
        hmi_quality_good = pd.Series(True, index=df.index)

    if mdi_quality_col in df.columns:
        mdi_quality_good = df[mdi_quality_col].map(lambda value: is_good_quality_flag(value, "mdi"))
    else:
        logger.warning("Missing column '%s'. Skipping MDI quality filtering.", mdi_quality_col)
        mdi_quality_good = pd.Series(True, index=df.index)

    keep_mask = ((~hmi_available) | hmi_quality_good) & ((~mdi_available) | mdi_quality_good)
    return df.loc[keep_mask].copy()


def apply_path_filter(
    df: pd.DataFrame,
    hmi_path_col: str | Iterable[str] = "path_image_cutout_hmi",
    mdi_path_col: str | Iterable[str] = "path_image_cutout_mdi",
) -> pd.DataFrame:
    """Keep rows with at least one non-empty cutout path."""
    hmi_cols = _normalize_path_columns(hmi_path_col)
    mdi_cols = _normalize_path_columns(mdi_path_col)
    if not any(col in df.columns for col in hmi_cols + mdi_cols):
        logger.warning("No path columns found. Path filtering skipped.")
        return df.copy()

    hmi_available = availability_mask(df, hmi_path_col)
    mdi_available = availability_mask(df, mdi_path_col)
    keep_mask = hmi_available | mdi_available
    return df.loc[keep_mask].copy()


def select_coordinate_series(
    df: pd.DataFrame,
    hmi_value_col: str,
    mdi_value_col: str,
    hmi_path_col: str | Iterable[str] = "path_image_cutout_hmi",
    mdi_path_col: str | Iterable[str] = "path_image_cutout_mdi",
) -> pd.Series:
    """
    Build per-row coordinate values with HMI-preferred, MDI fallback selection.
    """
    hmi_available = availability_mask(df, hmi_path_col)
    mdi_available = availability_mask(df, mdi_path_col)

    hmi_values = (
        pd.to_numeric(df[hmi_value_col], errors="coerce")
        if hmi_value_col in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    mdi_values = (
        pd.to_numeric(df[mdi_value_col], errors="coerce")
        if mdi_value_col in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    combined = pd.Series(np.nan, index=df.index, dtype=np.float64)
    combined.loc[hmi_available] = hmi_values.loc[hmi_available]

    mdi_fallback = combined.isna() & mdi_available
    combined.loc[mdi_fallback] = mdi_values.loc[mdi_fallback]

    return combined


def select_longitude_series(
    df: pd.DataFrame,
    hmi_longitude_col: str = "longitude_hmi",
    mdi_longitude_col: str = "longitude_mdi",
    hmi_path_col: str | Iterable[str] = "path_image_cutout_hmi",
    mdi_path_col: str | Iterable[str] = "path_image_cutout_mdi",
) -> pd.Series:
    """
    Build per-row longitude with HMI-preferred, MDI fallback selection.
    """
    return select_coordinate_series(
        df,
        hmi_value_col=hmi_longitude_col,
        mdi_value_col=mdi_longitude_col,
        hmi_path_col=hmi_path_col,
        mdi_path_col=mdi_path_col,
    )


def select_latitude_series(
    df: pd.DataFrame,
    hmi_latitude_col: str = "latitude_hmi",
    mdi_latitude_col: str = "latitude_mdi",
    hmi_path_col: str | Iterable[str] = "path_image_cutout_hmi",
    mdi_path_col: str | Iterable[str] = "path_image_cutout_mdi",
) -> pd.Series:
    """Build per-row latitude with HMI-preferred, MDI fallback selection."""
    return select_coordinate_series(
        df,
        hmi_value_col=hmi_latitude_col,
        mdi_value_col=mdi_latitude_col,
        hmi_path_col=hmi_path_col,
        mdi_path_col=mdi_path_col,
    )


def apply_longitude_filter(
    df: pd.DataFrame,
    max_longitude: float = 65.0,
    drop_missing_longitude: bool = True,
    hmi_longitude_col: str = "longitude_hmi",
    mdi_longitude_col: str = "longitude_mdi",
    hmi_path_col: str | Iterable[str] = "path_image_cutout_hmi",
    mdi_path_col: str | Iterable[str] = "path_image_cutout_mdi",
) -> pd.DataFrame:
    """
    Keep front-hemisphere rows using selected HMI/MDI longitudes.
    """
    if max_longitude is None:
        return df.copy()

    longitude = select_longitude_series(
        df,
        hmi_longitude_col=hmi_longitude_col,
        mdi_longitude_col=mdi_longitude_col,
        hmi_path_col=hmi_path_col,
        mdi_path_col=mdi_path_col,
    )
    within_limit = longitude.abs() <= max_longitude
    valid_longitude = longitude.notna()

    if drop_missing_longitude:
        keep_mask = valid_longitude & within_limit
    else:
        keep_mask = (~valid_longitude) | within_limit

    return df.loc[keep_mask].copy()


def dataset_root_path(data_folder: str | Path, dataset_folder: str | Path) -> Path:
    """Return absolute dataset root path."""
    dataset_path = Path(dataset_folder)
    if dataset_path.is_absolute():
        return dataset_path
    return Path(data_folder) / dataset_path


def remap_legacy_project_path(path_text: str, local_root: str | Path) -> Path:
    """
    Map common legacy path prefixes into the active local project root.

    This helper is dataset-agnostic and intended for full-disk/cross-task reuse.
    """
    root = Path(local_root)

    if path_text.startswith(LEGACY_ARCAFF_PREFIX):
        rel = path_text.split(LEGACY_ARCAFF_PREFIX, maxsplit=1)[1]
        return root / rel

    if path_text.startswith("arccnet_data/"):
        rel = path_text.split("arccnet_data/", maxsplit=1)[1]
        return root / rel

    return root / path_text


def resolve_project_path(path_value: object, local_root: str | Path) -> Path | None:
    """
    Resolve a generic project path value to an existing local path.

    Strategy:
    1) Existing absolute path.
    2) Legacy remap under ``local_root``.
    3) Relative path under ``local_root``.
    """
    if is_missing_path_value(path_value):
        return None

    path_text = str(path_value).strip()
    explicit = Path(path_text)
    if explicit.is_absolute() and explicit.exists():
        return explicit

    remapped = remap_legacy_project_path(path_text, local_root=local_root)
    if remapped.exists():
        return remapped

    candidate = Path(local_root) / path_text
    if candidate.exists():
        return candidate

    return None


def candidate_cutout_fits_dirs(data_folder: str | Path, dataset_folder: str | Path) -> list[Path]:
    """Return likely cutout FITS directories for a dataset layout."""
    root = dataset_root_path(data_folder, dataset_folder)
    return [
        root / "data" / "cutout_classification" / "fits",
        root / "data" / "region_cutouts" / "fits",
        root / "fits",
    ]


def remap_legacy_path(path_text: str, data_folder: str | Path, dataset_folder: str | Path) -> Path:
    """Map historical mount-style paths into the active local dataset layout."""
    data_root = Path(data_folder)
    ds_root = dataset_root_path(data_folder, dataset_folder)

    if path_text.startswith("/mnt/ARCAFF/v0.3.0/04_final/"):
        rel = path_text.split("/mnt/ARCAFF/v0.3.0/04_final/", maxsplit=1)[1]
        return ds_root / rel

    if path_text.startswith("/mnt/ARCAFF/v0.3.0/"):
        rel = path_text.split("/mnt/ARCAFF/v0.3.0/", maxsplit=1)[1]
        return data_root / rel

    if path_text.startswith("arccnet_data/04_final/"):
        rel = path_text.split("arccnet_data/04_final/", maxsplit=1)[1]
        return ds_root / rel

    return ds_root / path_text


def resolve_cutout_fits_path(
    path_value: object,
    data_folder: str | Path,
    dataset_folder: str | Path,
) -> Path | None:
    """
    Resolve a cutout path value to an existing local FITS path.
    """
    if is_missing_path_value(path_value):
        return None

    path_text = str(path_value).strip()

    explicit = Path(path_text)
    if explicit.is_absolute() and explicit.exists():
        return explicit

    remapped = remap_legacy_path(path_text, data_folder=data_folder, dataset_folder=dataset_folder)
    if remapped.exists():
        return remapped

    # If provided path is relative to the dataset root, try it directly.
    ds_root = dataset_root_path(data_folder, dataset_folder)
    rel_candidate = ds_root / path_text
    if rel_candidate.exists():
        return rel_candidate

    # Finally, search by basename under known FITS directories.
    basename = Path(path_text).name
    for fits_dir in candidate_cutout_fits_dirs(data_folder, dataset_folder):
        candidate = fits_dir / basename
        if candidate.exists():
            return candidate

    return None


def choose_preferred_path_value(
    row: pd.Series,
    hmi_path_col: str = "path_image_cutout_hmi",
    mdi_path_col: str = "path_image_cutout_mdi",
    prefer_hmi: bool = True,
) -> str | None:
    """Pick HMI path first (default), then MDI."""
    hmi_value = row.get(hmi_path_col)
    mdi_value = row.get(mdi_path_col)

    if prefer_hmi:
        if not is_missing_path_value(hmi_value):
            return str(hmi_value).strip()
        if not is_missing_path_value(mdi_value):
            return str(mdi_value).strip()
    else:
        if not is_missing_path_value(mdi_value):
            return str(mdi_value).strip()
        if not is_missing_path_value(hmi_value):
            return str(hmi_value).strip()

    return None


def iter_preferred_path_values(
    row: pd.Series,
    hmi_path_col: str = "path_image_cutout_hmi",
    mdi_path_col: str = "path_image_cutout_mdi",
    prefer_hmi: bool = True,
) -> list[str]:
    """Return ordered non-missing path candidates with HMI/MDI preference."""
    ordered_values = []
    primary_col, secondary_col = (hmi_path_col, mdi_path_col) if prefer_hmi else (mdi_path_col, hmi_path_col)

    for column in (primary_col, secondary_col):
        value = row.get(column)
        if is_missing_path_value(value):
            continue
        normalized = str(value).strip()
        if normalized and normalized not in ordered_values:
            ordered_values.append(normalized)

    return ordered_values


def resolve_preferred_cutout_fits_path(
    row: pd.Series,
    data_folder: str | Path,
    dataset_folder: str | Path,
    hmi_path_col: str = "path_image_cutout_hmi",
    mdi_path_col: str = "path_image_cutout_mdi",
    prefer_hmi: bool = True,
) -> Path | None:
    """
    Resolve a row's preferred cutout path to a local FITS file.

    Tries preferred instrument first, then falls back to the alternate path when
    the preferred value is present but cannot be resolved locally.
    """
    candidates = iter_preferred_path_values(
        row,
        hmi_path_col=hmi_path_col,
        mdi_path_col=mdi_path_col,
        prefer_hmi=prefer_hmi,
    )
    for candidate in candidates:
        resolved = resolve_cutout_fits_path(candidate, data_folder=data_folder, dataset_folder=dataset_folder)
        if resolved is not None:
            return resolved
    return None


def load_fits_hdu_data(
    fits_path: str | Path,
    hdu_index: int = 1,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load FITS image data from HDU ``hdu_index`` with fallback to first data HDU.
    """
    from astropy.io import fits

    with fits.open(fits_path, memmap=True) as hdul:
        data = None

        if 0 <= hdu_index < len(hdul):
            data = hdul[hdu_index].data

        if data is None:
            for hdu in hdul:
                if hdu.data is not None:
                    data = hdu.data
                    break

        if data is None:
            raise ValueError(f"No image data found in FITS file: {fits_path}")

        return np.array(data, dtype=dtype)


def compute_cutout_nan_fraction(
    row: pd.Series,
    data_folder: str | Path,
    dataset_folder: str | Path,
    hdu_index: int = 1,
) -> float:
    """
    Compute NaN fraction for the row's preferred cutout FITS image.

    Missing/unreadable images return ``1.0`` so callers can conservatively
    filter them out when enforcing quality thresholds.
    """
    try:
        fits_path = resolve_preferred_cutout_fits_path(
            row,
            data_folder=data_folder,
            dataset_folder=dataset_folder,
        )
        if fits_path is None:
            return 1.0

        cutout = load_fits_hdu_data(fits_path, hdu_index=hdu_index, dtype=np.float32)
        if cutout.size == 0:
            return 1.0

        return float(np.isnan(cutout).sum() / cutout.size)
    except Exception:
        return 1.0


def filter_cutouts_by_nan_threshold(
    df: pd.DataFrame,
    nan_threshold: float,
    data_folder: str | Path,
    dataset_folder: str | Path,
    hdu_index: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Filter cutout rows by NaN fraction threshold.

    Returns the filtered dataframe and the per-row NaN fractions aligned with
    ``df.iterrows()`` order.
    """
    if df.empty:
        return df.copy(), np.array([], dtype=np.float64)

    from p_tqdm import p_map

    def _check_nan(idx_row: tuple[object, pd.Series]) -> tuple[object, float, bool]:
        idx, row = idx_row
        nan_fraction = compute_cutout_nan_fraction(
            row,
            data_folder=data_folder,
            dataset_folder=dataset_folder,
            hdu_index=hdu_index,
        )
        keep = nan_fraction <= nan_threshold
        return idx, nan_fraction, keep

    rows = list(df.iterrows())
    try:
        results = p_map(_check_nan, rows)
    except (PermissionError, OSError):
        logger.warning("Parallel NaN filtering unavailable in this environment; falling back to serial execution.")
        results = [_check_nan(item) for item in rows]

    keep_indices = [idx for idx, _, keep in results if keep]
    nan_fractions = np.array([value for _, value, _ in results], dtype=np.float64)
    return df.loc[keep_indices].copy(), nan_fractions

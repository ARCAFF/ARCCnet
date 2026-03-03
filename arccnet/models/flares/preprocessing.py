"""Preprocessing utilities for flare datasets."""

import logging

import pandas as pd

from arccnet.models import preprocessing_common as pp_common

logger = logging.getLogger(__name__)

DEFAULT_NAN_THRESHOLD = 0.05
DEFAULT_LONGITUDE_LIMIT = 65.0


def filter_by_nan_threshold(
    df: pd.DataFrame,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
    data_folder: str = "/ARCAFF/data",
    dataset_folder: str = "arcnet-v20251017/04_final",
) -> pd.DataFrame:
    """Filter out magnetogram cutouts with excessive NaN values."""
    logger.info(f"Filtering cutouts with NaN threshold: {nan_threshold * 100:.1f}%")

    logger.info("Computing NaN statistics (this may take a while)...")
    df_filtered, nan_stats_array = pp_common.filter_cutouts_by_nan_threshold(
        df,
        nan_threshold=nan_threshold,
        data_folder=data_folder,
        dataset_folder=dataset_folder,
        hdu_index=1,
    )

    if nan_stats_array.size == 0:
        logger.warning("No rows were evaluated during NaN filtering.")
        return df.iloc[0:0].copy()

    removed_count = len(df) - len(df_filtered)
    retained_pct = (len(df_filtered) / len(df) * 100.0) if len(df) else 0.0

    logger.info(
        f"NaN filtering: {len(df):,} → {len(df_filtered):,} ({retained_pct:.1f}% retained, {removed_count:,} removed) | Mean NaN: {nan_stats_array.mean() * 100:.2f}%, Max: {nan_stats_array.max() * 100:.2f}%"
    )

    return df_filtered


def apply_quality_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality flag filtering using the cutout pipeline rules."""
    logger.info("Applying quality flag filtering...")

    df_clean = pp_common.apply_quality_filter(df)

    removed = len(df) - len(df_clean)
    logger.info(f"Quality filtering: {len(df):,} → {len(df_clean):,} ({removed:,} removed)")

    return df_clean.reset_index(drop=True)


def apply_path_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where both HMI and MDI image paths are missing."""
    logger.info("Applying path filtering...")

    df_filtered = pp_common.apply_path_filter(df).reset_index(drop=True)
    removed = len(df) - len(df_filtered)

    logger.info(f"Path filtering: {len(df):,} → {len(df_filtered):,} ({removed:,} removed)")

    hmi_available = (
        pp_common.nonempty_path_mask(df_filtered["path_image_cutout_hmi"])
        if "path_image_cutout_hmi" in df_filtered.columns
        else pd.Series(False, index=df_filtered.index)
    )
    mdi_available = (
        pp_common.nonempty_path_mask(df_filtered["path_image_cutout_mdi"])
        if "path_image_cutout_mdi" in df_filtered.columns
        else pd.Series(False, index=df_filtered.index)
    )

    stats = {
        "total": len(df_filtered),
        "hmi_only": int((hmi_available & ~mdi_available).sum()),
        "mdi_only": int((mdi_available & ~hmi_available).sum()),
        "both": int((hmi_available & mdi_available).sum()),
    }

    logger.debug(f"Path distribution: HMI-only={stats['hmi_only']}, MDI-only={stats['mdi_only']}, Both={stats['both']}")

    return df_filtered


def apply_longitude_filtering(
    df: pd.DataFrame,
    max_longitude: float = DEFAULT_LONGITUDE_LIMIT,
) -> pd.DataFrame:
    """Filter to keep only front-hemisphere observations based on longitude."""
    logger.info(f"Applying longitude filtering (max |lon| = {max_longitude}°)...")

    initial_count = len(df)
    df_filtered = pp_common.apply_longitude_filter(df, max_longitude=max_longitude).reset_index(drop=True)
    removed = initial_count - len(df_filtered)
    logger.info(f"Longitude filtering: {initial_count:,} → {len(df_filtered):,} ({removed:,} removed)")

    return df_filtered


def preprocess_flare_data(
    df: pd.DataFrame,
    apply_quality_filter: bool = True,
    apply_path_filter: bool = True,
    apply_longitude_filter: bool = True,
    apply_nan_filter: bool = False,
    max_longitude: float = DEFAULT_LONGITUDE_LIMIT,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
    data_folder: str = "/ARCAFF/data",
    dataset_folder: str = "arcnet-v20251017/04_final",
) -> pd.DataFrame:
    """Run the full flare preprocessing pipeline."""
    logger.info(f"Preprocessing pipeline: initial size = {len(df):,} records")

    df_processed = df.copy()

    if apply_quality_filter:
        df_processed = apply_quality_filtering(df_processed)

    if apply_path_filter:
        df_processed = apply_path_filtering(df_processed)

    if apply_longitude_filter:
        df_processed = apply_longitude_filtering(df_processed, max_longitude)

    if apply_nan_filter:
        df_processed = filter_by_nan_threshold(
            df_processed,
            nan_threshold=nan_threshold,
            data_folder=data_folder,
            dataset_folder=dataset_folder,
        )

    total_removed = len(df) - len(df_processed)
    retention_rate = (len(df_processed) / len(df) * 100) if len(df) > 0 else 0

    logger.info(
        f"Preprocessing complete: {len(df_processed):,} records | Removed: {total_removed:,} ({100 - retention_rate:.1f}%) | Retention: {retention_rate:.1f}%"
    )

    return df_processed

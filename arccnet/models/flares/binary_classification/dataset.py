"""Load, validate, and split flare datasets for binary training."""

import os
import logging

import pandas as pd

from arccnet.models.flares import utils as ut_f
from arccnet.models.flares.binary_classification import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_REQUIRED_PATH_COLUMNS = ("path_image_cutout_hmi", "path_image_cutout_mdi")
_GROUP_COLUMN = "number"


def _validate_binary_target_column(df: pd.DataFrame, target_column: str) -> None:
    """Validate target column exists and is binary-encodable."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    values = pd.to_numeric(pd.Series(df[target_column]), errors="raise").fillna(0)
    unique_values = set(values.unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"Target column '{target_column}' must be binary (0/1). Found values: {sorted(unique_values)}")


def _format_count_percentage_table(distribution: pd.DataFrame) -> pd.DataFrame:
    """Format per-split class counts as 'count (pct%)' strings."""
    formatted = distribution.copy()
    for column in formatted.columns:
        counts = formatted[column].astype(int)
        total = int(counts.sum())
        if total > 0:
            percentages = (counts / total * 100.0).round(1)
        else:
            percentages = pd.Series(0.0, index=counts.index)
        formatted[column] = counts.astype(str) + " (" + percentages.astype(str) + "%)"
    return formatted


def _log_split_distributions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> None:
    """Log target and magnetic-class distributions for split dataframes."""
    flare_dist = pd.concat(
        [
            train_df[target_column].value_counts().rename("Train"),
            val_df[target_column].value_counts().rename("Validation"),
            test_df[target_column].value_counts().rename("Test"),
        ],
        axis=1,
    ).fillna(0)
    flare_dist = _format_count_percentage_table(flare_dist)
    logger.info("\nFlare Classification Distribution (%s):\n%s", target_column, flare_dist.to_string())

    if "magnetic_class" not in train_df.columns:
        return

    mag_dist = (
        pd.concat(
            [
                train_df["magnetic_class"].map(ut_f.MAG_CLASS_MAPPING).value_counts().rename("Train"),
                val_df["magnetic_class"].map(ut_f.MAG_CLASS_MAPPING).value_counts().rename("Validation"),
                test_df["magnetic_class"].map(ut_f.MAG_CLASS_MAPPING).value_counts().rename("Test"),
            ],
            axis=1,
        )
        .reindex(ut_f.MAG_CLASS_ORDER)
        .fillna(0)
    )
    mag_dist = _format_count_percentage_table(mag_dist)
    logger.info("\nMagnetic Class Distribution:\n%s", mag_dist.to_string())


def _split_dataframe(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    val_size: float,
    random_state: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data with the project-standard AR-aware split utility."""
    if _GROUP_COLUMN not in df.columns:
        raise ValueError(f"Required group column '{_GROUP_COLUMN}' not found in dataframe.")

    _validate_binary_target_column(df, target_column)

    train_df, val_df, test_df = ut_f.split_dataframe(
        df=df,
        stratify_col=target_column,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def load_prepared_dataframe(
    data_folder: str,
    df_flares_name: str,
    dataset_folder: str,
) -> pd.DataFrame:
    """Load flare catalog, validate FITS availability, and add target columns."""
    data_path = os.path.join(data_folder, df_flares_name)
    logger.info(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    df_flares = pd.read_parquet(data_path)
    logger.info("Loaded dataframe with shape: %s", df_flares.shape)

    missing_path_columns = [col for col in _REQUIRED_PATH_COLUMNS if col not in df_flares.columns]
    if missing_path_columns:
        raise ValueError(f"Input dataframe is missing required columns: {missing_path_columns}")

    df_flares_exists, none_idxs = ut_f.check_fits_file_existence(
        df_flares.copy(), data_folder, dataset_folder, image_type=config.IMAGE_TYPE
    )
    files_found = int(df_flares_exists["file_exists"].sum())
    files_missing = len(df_flares_exists) - files_found
    logger.info("FITS check complete: %d no path, %d found, %d missing", len(none_idxs), files_found, files_missing)
    df_flares = df_flares_exists[df_flares_exists["file_exists"]].copy()

    if df_flares.empty:
        raise ValueError("No rows remain after FITS path existence filtering.")

    logger.info("Creating binary 'flares_above_*' columns.")
    for i in range(len(ut_f.FLARE_CLASSES)):
        threshold_class = ut_f.FLARE_CLASSES[i]
        columns_to_check = [f for f in ut_f.FLARE_CLASSES[i:] if f in df_flares.columns]
        if not columns_to_check:
            logger.warning(
                "No columns found for flare classes %s in dataframe. Skipping 'flares_above_%s'.",
                ut_f.FLARE_CLASSES[i:],
                threshold_class,
            )
            continue

        new_col_name = f"flares_above_{threshold_class}"
        df_flares[new_col_name] = (df_flares[columns_to_check].fillna(0) > 0).any(axis=1).astype(int)
        logger.debug("Created binary target column: %s", new_col_name)

    return df_flares


def load_and_split_data(
    data_folder: str,
    df_flares_name: str,
    dataset_folder: str,
    target_column: str,
    test_size: float,
    val_size: float,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data, derive targets, split, and log distributions."""
    df_flares = load_prepared_dataframe(
        data_folder=data_folder,
        df_flares_name=df_flares_name,
        dataset_folder=dataset_folder,
    )

    logger.info("Splitting data: test_size=%s, val_size=%s, random_state=%s", test_size, val_size, random_state)
    train_df, val_df, test_df = _split_dataframe(
        df=df_flares,
        target_column=target_column,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    logger.info("Split complete. Shapes: Train=%s, Val=%s, Test=%s", train_df.shape, val_df.shape, test_df.shape)
    _log_split_distributions(train_df, val_df, test_df, target_column)

    return train_df, val_df, test_df


def split_preprocessed_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    val_size: float,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a preprocessed dataframe into train/val/test sets."""
    logger.info("Splitting preprocessed data: test_size=%s, val_size=%s", test_size, val_size)
    train_df, val_df, test_df = _split_dataframe(
        df=df,
        target_column=target_column,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    logger.info("Split complete. Shapes: Train=%s, Val=%s, Test=%s", train_df.shape, val_df.shape, test_df.shape)
    _log_split_distributions(train_df, val_df, test_df, target_column)

    return train_df, val_df, test_df

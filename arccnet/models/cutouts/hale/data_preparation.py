"""Data preparation for Hale classification: load, clean, label mapping,
filtering, and cross-validation with AR number separation."""

import os
import logging
import argparse
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import arccnet.models.cutouts.hale.config as config
from arccnet.models import dataset_utils as ut_d
from arccnet.models import preprocessing_common as pp_common

# NaN filtering parameters
DEFAULT_NAN_THRESHOLD = 0.05  # Maximum allowed fraction of NaN values (5%)
LOGGING_LEVEL = logging.DEBUG
ENABLE_NAN_FILTERING_LOG = True
LOG_DETAILED_STATS = True
PLOT_FILTERED = False


def filter_by_nan_threshold(
    df: pd.DataFrame,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
) -> pd.DataFrame:
    """
    Filter out magnetogram cutouts with excessive NaN values.

    Args:
        df: DataFrame with cutout references
        nan_threshold: Maximum allowed fraction of NaN values (0-1)

    Returns:
        pd.DataFrame: Filtered dataframe with low-NaN cutouts only
    """
    logging.info(f"Filtering cutouts with NaN threshold: {nan_threshold * 100:.1f}%")

    df_filtered, nan_stats = pp_common.filter_cutouts_by_nan_threshold(
        df,
        nan_threshold=nan_threshold,
        data_folder=config.DATA_FOLDER,
        dataset_folder=config.DATASET_FOLDER,
        hdu_index=1,
    )
    if nan_stats.size == 0:
        return df.iloc[0:0].copy()

    removed_indices = df.index.difference(df_filtered.index).tolist()
    if ENABLE_NAN_FILTERING_LOG and LOG_DETAILED_STATS:
        original_count = len(df)
        filtered_count = len(df_filtered)
        retained_pct = (filtered_count / original_count * 100.0) if original_count else 0.0
        logging.info("NaN filtering results:")
        logging.info(f"  Original: {original_count:,} cutouts")
        logging.info(f"  Filtered: {filtered_count:,} cutouts ({retained_pct:.1f}% retained)")
        logging.info(f"  Removed:  {original_count - filtered_count:,} cutouts")
        logging.info(f"  Mean NaN fraction: {nan_stats.mean() * 100:.2f}%")
        logging.info(f"  Max NaN fraction:  {nan_stats.max() * 100:.2f}%")
    if ENABLE_NAN_FILTERING_LOG and removed_indices:
        logging.debug(f"Indices of removed images due to NaN: {removed_indices}")
    if PLOT_FILTERED:
        os.makedirs("temp", exist_ok=True)
        for idx in removed_indices:
            row = df.loc[idx]
            fits_path = pp_common.resolve_preferred_cutout_fits_path(
                row,
                data_folder=config.DATA_FOLDER,
                dataset_folder=config.DATASET_FOLDER,
            )
            if fits_path is None:
                continue
            try:
                image = pp_common.load_fits_hdu_data(fits_path, hdu_index=1, dtype=np.float32)
                plt.imshow(image, cmap="gray")
                plt.colorbar()
                plt.title(f"idx: {idx}, {row['dates']}")
                plt.savefig(f"temp/removed_nan_{idx}.png", dpi=300)
                plt.close()
            except Exception:
                logging.exception("Failed to plot filtered NaN sample idx=%s", idx)

    return df_filtered


def load_and_clean_dataset(nan_threshold: float = DEFAULT_NAN_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw dataset and apply cleaning + NaN filtering.

    Args:
        nan_threshold: Maximum allowed NaN fraction

    Returns:
        Raw and cleaned datasets
    """
    logging.info("Loading and cleaning dataset...")

    df, _, _ = ut_d.make_dataframe(config.DATA_FOLDER, config.DATASET_FOLDER, config.DF_FILE_NAME)
    logging.info(f"Original DataFrame shape: {df.shape}")

    df_clean = ut_d.cleanup_df(df)
    retained_pct = (len(df_clean) / len(df) * 100.0) if len(df) else 0.0
    logging.info(f"After cleanup: {df_clean.shape} ({retained_pct:.1f}% retained)")

    # NaN filtering
    df_clean = filter_by_nan_threshold(df_clean, nan_threshold=nan_threshold)

    return df, df_clean


def apply_label_mapping_and_filter(
    df_clean: pd.DataFrame, label_mapping: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map Hale class labels and apply longitude filtering/undersampling.

    Args:
        df_clean: Cleaned dataframe
        label_mapping: Dict mapping original to grouped labels

    Returns:
        DataFrame after label mapping and after filtering
    """

    df_original, df_processed = ut_d.undersample_group_filter(
        df_clean, label_mapping=label_mapping, long_limit_deg=config.LONG_LIMIT_DEG, undersample=config.UNDERSAMPLE
    )
    logging.info(f"Label mapping: {label_mapping}")
    logging.info(f"Label mapping applied: {len(df_original):,} → {len(df_processed):,}")
    class_dist = df_processed["grouped_labels"].value_counts()
    logging.info(
        "Final class distribution:"
        + "".join(
            f"\n  {label}: {count:,} ({count / len(df_processed) * 100:.1f}%)" for label, count in class_dist.items()
        )
    )
    return df_original, df_processed


def validate_fold_separation(df: pd.DataFrame) -> None:
    """
    Ensure AR numbers don't overlap between train/val/test sets to prevent data leakage.

    Args:
        df: DataFrame with fold assignment columns
    """
    problematic_folds = []
    for fold_column_name in [col for col in df.columns if col.startswith("Fold ")]:
        fold_num = fold_column_name.split()[-1]
        masks = [df[fold_column_name] == split for split in ["train", "val", "test"]]
        counts = [mask.sum() for mask in masks]
        ars = [set(df[mask]["number"].unique()) for mask in masks]
        ar_counts = [len(a) for a in ars]
        total = sum(counts)
        total_ars = sum(ar_counts)
        pct = [100 * c / total if total else 0 for c in counts]
        ar_pct = [100 * c / total_ars if total_ars else 0 for c in ar_counts]
        overlaps = {
            name: ars[i] & ars[j] for (name, i, j) in [("Train-Val", 0, 1), ("Train-Test", 0, 2), ("Val-Test", 1, 2)]
        }
        msg = (
            f"Fold {fold_num}: "
            f"Train={counts[0]:,} ({pct[0]:.1f}%), "
            f"Val={counts[1]:,} ({pct[1]:.1f}%), "
            f"Test={counts[2]:,} ({pct[2]:.1f}%) | "
            f"ARs: Train={ar_counts[0]} ({ar_pct[0]:.1f}%), "
            f"Val={ar_counts[1]} ({ar_pct[1]:.1f}%), "
            f"Test={ar_counts[2]} ({ar_pct[2]:.1f}%)"
        )
        logging.info(msg)
        if any(overlaps.values()):
            logging.error(f"  ERROR - AR number overlaps detected in Fold {fold_num}:")
            [logging.error(f"    {name} overlap: {sorted(overlap)}") for name, overlap in overlaps.items() if overlap]
            problematic_folds.append(fold_num)
    if problematic_folds:
        logging.error(f"Summary: Overlaps detected in folds: {', '.join(problematic_folds)}")
    else:
        logging.info("Summary: No AR number overlaps detected in any fold.")


def create_and_validate_cv_folds(
    df_processed: pd.DataFrame, n_splits: int = config.N_FOLDS, random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """
    Create stratified CV folds with AR number separation.

    Args:
        df_processed: DataFrame with grouped labels
        n_splits: Number of folds
        random_state: Random seed

    Returns:
        DataFrame with fold assignment columns
    """
    logging.info("Creating cross-validation folds...")

    ut_d.split_data(
        df_processed,
        label_col="grouped_labels",
        group_col="number",
        n_splits=n_splits,
        random_state=random_state,
    )

    validate_fold_separation(df_processed)

    return df_processed


def prepare_dataset(
    save_path: str = None,
    n_splits: int = config.N_FOLDS,
    random_state: int = config.RANDOM_STATE,
    label_mapping: dict[str, Any] = None,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
) -> pd.DataFrame:
    """
    Complete dataset preparation pipeline with NaN filtering.

    Args:
        save_path: Path to save processed dataset
        n_splits: Number of cross-validation folds
        random_state: Random seed
        label_mapping: Custom label mapping dict
        nan_threshold: Maximum allowed NaN fraction

    Returns:
        Processed dataset with fold assignments
    """
    _, df_clean = load_and_clean_dataset(nan_threshold=nan_threshold)

    if label_mapping is None:
        label_mapping = config.label_mapping
    _, df_processed = apply_label_mapping_and_filter(df_clean, label_mapping=label_mapping)
    df_with_folds = create_and_validate_cv_folds(df_processed, n_splits=n_splits, random_state=random_state)
    if save_path:
        df_with_folds.to_parquet(save_path, index=False)
        logging.info(f"Processed dataset saved to: {save_path}")
    return df_with_folds


def main():
    """CLI entry point for dataset preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare dataset for ARCCNet model training.")
    parser.add_argument("--n_splits", type=int, default=config.N_FOLDS, help="Number of splits for cross-validation.")
    parser.add_argument(
        "--random_state", type=int, default=config.RANDOM_STATE, help="Random state for reproducibility."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the processed dataset. If not provided, a default name will be generated.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

    save_path = args.save_path
    if save_path is None:
        save_path = (
            Path(config.DATA_FOLDER)
            / f"processed_dataset_{config.classes}_{args.n_splits}-splits_rs-{args.random_state}.parquet"
        )

    prepare_dataset(
        save_path=save_path, n_splits=args.n_splits, random_state=args.random_state, label_mapping=config.label_mapping
    )


if __name__ == "__main__":
    main()

"""Generate YOLO datasets from full-disk solar observations (mag/cont)."""

from __future__ import annotations

import logging
from typing import Any
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml
from p_tqdm import p_map

from arccnet.models.fulldisk.yolo import dataset_config as cfg
from arccnet.models.fulldisk.yolo import yolo_utils as ut

logger = logging.getLogger("yolo.dataset")


def _load_filtered_regions() -> pd.DataFrame:
    """Load full-disk catalog and apply base filtering."""
    # Local import avoids importing heavy full-disk plotting deps when not needed.
    from arccnet.models.fulldisk import utils as fd_utils

    return fd_utils.prepare_fulldisk_dataset(
        cfg.DATA_FOLDER,
        cfg.DATASET_ROOT,
        cfg.DATASET_FOLDER,
        cfg.DATAFRAME_NAME,
        longitude_threshold=cfg.LONGITUDE_THRESHOLD,
        min_size=cfg.MIN_SIZE,
        filter_selected=cfg.FILTER_SELECTED,
    )


def _filter_existing_files(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with both magnetogram and continuum files present locally."""
    from arccnet.models.fulldisk import utils as fd_utils

    initial_count = len(df)
    df = df.copy()
    df["mag_exists"] = df["processed_path_image_mag"].apply(
        lambda path: fd_utils.check_file_exists(path, cfg.DATA_FOLDER, cfg.DATASET_ROOT)
    )
    df["cont_exists"] = df["processed_path_image_cont"].apply(
        lambda path: fd_utils.check_file_exists(path, cfg.DATA_FOLDER, cfg.DATASET_ROOT)
    )
    df = df[df["mag_exists"] & df["cont_exists"]].copy()
    logger.info(
        "File existence: present %d/%d (%.1f%%)",
        len(df),
        initial_count,
        (len(df) / initial_count * 100.0) if initial_count else 0.0,
    )
    return df


def _all_images_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return unique image table before label filtering."""
    result = (
        df.groupby("processed_path_image_mag")
        .agg(
            {
                "processed_path_image_cont": "first",
                "datetime": "first",
                "instrument": "first",
            }
        )
        .reset_index()
    )
    return result.rename(columns={"processed_path_image_mag": "path_mag", "processed_path_image_cont": "path_cont"})


def _encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map magnetic classes, drop excluded labels, and encode class indices."""
    df = df.copy()
    df["grouped_label"] = df["magnetic_class"].map(cfg.LABEL_MAPPING)
    logger.info("Label distribution (before filtering 'None'):")
    for label, count in df["grouped_label"].value_counts().items():
        logger.info("%s: %d (%.1f%%)", label, count, count / len(df) * 100.0 if len(df) else 0.0)

    df_with_labels = df[df["grouped_label"] != "None"].copy()
    logger.info("After removing 'None' labels: %d regions", len(df_with_labels))
    if df_with_labels.empty:
        raise ValueError("No labeled regions remain after applying LABEL_MAPPING")

    unique_labels = sorted(df_with_labels["grouped_label"].unique())
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    df_with_labels["encoded_label"] = df_with_labels["grouped_label"].map(label_to_index)

    logger.info("Label mapping (grouped_label -> YOLO class index):")
    for label, idx in label_to_index.items():
        count = int((df_with_labels["grouped_label"] == label).sum())
        logger.info("%s: %d (%d regions)", label, idx, count)

    return df_with_labels, label_to_index


def _validate_bbox(row: pd.Series, img_size_by_instrument: dict[str, int]) -> bool:
    """Validate that bbox is in-bounds with positive normalized dimensions."""
    x1, y1 = row["bottom_left_cutout"]
    x2, y2 = row["top_right_cutout"]
    img_size = img_size_by_instrument[row["instrument"]]

    if x1 < 0 or y1 < 0 or x2 > img_size or y2 > img_size:
        return False
    if x2 <= x1 or y2 <= y1:
        return False

    x_center = ((x1 + x2) / 2) / img_size
    y_center = ((y1 + y2) / 2) / img_size
    width = (x2 - x1) / img_size
    height = (y2 - y1) / img_size
    return 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1


def _create_yolo_labels(df_with_labels: pd.DataFrame, img_size_by_instrument: dict[str, int]) -> pd.DataFrame:
    """Build YOLO label strings for each labeled region."""
    df_with_labels = df_with_labels.copy()
    df_with_labels["valid_bbox"] = df_with_labels.apply(lambda row: _validate_bbox(row, img_size_by_instrument), axis=1)

    invalid_count = int((~df_with_labels["valid_bbox"]).sum())
    if invalid_count > 0:
        logger.warning("%d invalid bounding boxes detected and removed", invalid_count)
        df_with_labels = df_with_labels[df_with_labels["valid_bbox"]].copy()
    else:
        logger.info("All %d bounding boxes are valid", len(df_with_labels))

    if df_with_labels.empty:
        raise ValueError("No valid bounding boxes remain after validation")

    df_with_labels["yolo_label"] = df_with_labels.apply(
        lambda row: ut.to_yolo(
            row["encoded_label"],
            row["top_right_cutout"],
            row["bottom_left_cutout"],
            img_size_by_instrument[row["instrument"]],
            img_size_by_instrument[row["instrument"]],
        ),
        axis=1,
    )
    return df_with_labels


def _build_image_level_dataset(df_with_labels: pd.DataFrame, all_images: pd.DataFrame) -> pd.DataFrame:
    """Aggregate region labels per image and optionally include empty-label images."""
    with_labels = (
        df_with_labels.groupby("processed_path_image_mag")
        .agg(
            {
                "yolo_label": "\n".join,
                "processed_path_image_cont": "first",
                "datetime": "first",
                "instrument": "first",
            }
        )
        .reset_index()
        .rename(columns={"processed_path_image_mag": "path_mag", "processed_path_image_cont": "path_cont"})
    )

    if cfg.INCLUDE_EMPTY_LABELS:
        images_with_labels = set(with_labels["path_mag"])
        images_without_labels = all_images[~all_images["path_mag"].isin(images_with_labels)].copy()
        images_without_labels["yolo_label"] = ""

        logger.info("Images WITH valid labels: %d", len(with_labels))
        logger.info("Images WITHOUT valid labels (negative examples): %d", len(images_without_labels))

        df_yolo = pd.concat([with_labels, images_without_labels], ignore_index=True)
    else:
        logger.info("Using only images with labels: %d", len(with_labels))
        df_yolo = with_labels

    if df_yolo.empty:
        raise ValueError("No images available for YOLO dataset after aggregation")

    return df_yolo.sort_values("datetime").reset_index(drop=True)


def _split_temporal(df_yolo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split image-level dataset into train/val with temporal gap."""
    logger.info("Splitting dataset with temporal gap...")
    logger.info("Date range: %s to %s", df_yolo["datetime"].min(), df_yolo["datetime"].max())
    logger.info("Duration: %d days", (df_yolo["datetime"].max() - df_yolo["datetime"].min()).days)

    if len(df_yolo) < 2:
        raise ValueError("Need at least 2 images to split train/val")

    initial_split_idx = int(cfg.TRAIN_SPLIT_RATIO * len(df_yolo))
    initial_split_idx = min(max(initial_split_idx, 1), len(df_yolo) - 1)

    train_end_date = df_yolo.iloc[initial_split_idx - 1]["datetime"]
    val_start_threshold = train_end_date + timedelta(days=cfg.TEMPORAL_GAP_DAYS)

    candidate_indices = df_yolo.index[df_yolo["datetime"] >= val_start_threshold]
    if len(candidate_indices) == 0:
        logger.warning(
            "No validation start found with full %d-day temporal gap; falling back to ratio split index %d",
            cfg.TEMPORAL_GAP_DAYS,
            initial_split_idx,
        )
        val_start_idx = initial_split_idx
    else:
        val_start_idx = int(candidate_indices[0])

    train_df = df_yolo.iloc[:val_start_idx].copy()
    val_df = df_yolo.iloc[val_start_idx:].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            "Invalid split produced empty subset. "
            f"train={len(train_df)}, val={len(val_df)}, val_start_idx={val_start_idx}"
        )

    actual_gap = int((val_df["datetime"].min() - train_df["datetime"].max()).days)
    logger.info("Train set: %d images (%.1f%%)", len(train_df), len(train_df) / len(df_yolo) * 100.0)
    logger.info("  Date range: %s to %s", train_df["datetime"].min(), train_df["datetime"].max())
    logger.info("Validation set: %d images (%.1f%%)", len(val_df), len(val_df) / len(df_yolo) * 100.0)
    logger.info("  Date range: %s to %s", val_df["datetime"].min(), val_df["datetime"].max())
    logger.info("Temporal gap: %d days", actual_gap)

    if actual_gap < cfg.TEMPORAL_GAP_DAYS:
        logger.warning("Actual gap (%d days) < requested (%d days)", actual_gap, cfg.TEMPORAL_GAP_DAYS)

    return train_df, val_df, actual_gap


def _compute_bbox_stats(labels_df: pd.DataFrame, img_size_by_instrument: dict[str, int]) -> dict[str, float]:
    """Compute normalized bbox summary stats for a subset."""
    if labels_df.empty:
        return {
            k: 0.0
            for k in (
                "mean_width",
                "mean_height",
                "median_width",
                "median_height",
                "min_width",
                "min_height",
                "max_width",
                "max_height",
            )
        }

    widths = [
        (row["top_right_cutout"][0] - row["bottom_left_cutout"][0]) / img_size_by_instrument[row["instrument"]]
        for _, row in labels_df.iterrows()
    ]
    heights = [
        (row["top_right_cutout"][1] - row["bottom_left_cutout"][1]) / img_size_by_instrument[row["instrument"]]
        for _, row in labels_df.iterrows()
    ]

    return {
        "mean_width": float(np.mean(widths)),
        "mean_height": float(np.mean(heights)),
        "median_width": float(np.median(widths)),
        "median_height": float(np.median(heights)),
        "min_width": float(np.min(widths)),
        "min_height": float(np.min(heights)),
        "max_width": float(np.max(widths)),
        "max_height": float(np.max(heights)),
    }


def _log_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    df_with_labels: pd.DataFrame,
    label_to_index: dict[str, int],
    img_size_by_instrument: dict[str, int],
) -> None:
    """Log class distributions and bbox stats for train/val subsets."""
    train_labels = df_with_labels[df_with_labels["processed_path_image_mag"].isin(train_df["path_mag"])]
    val_labels = df_with_labels[df_with_labels["processed_path_image_mag"].isin(val_df["path_mag"])]

    logger.info("TRAIN SET STATISTICS:")
    logger.info("  Total images: %d", len(train_df))
    logger.info("  Images with labels: %d", len(train_df[train_df["yolo_label"] != ""]))
    logger.info("  Images without labels: %d", len(train_df[train_df["yolo_label"] == ""]))
    logger.info("  Total regions: %d", len(train_labels))
    logger.info("  Label distribution:")
    for label, idx in sorted(label_to_index.items(), key=lambda item: item[1]):
        count = int((train_labels["grouped_label"] == label).sum())
        pct = count / len(train_labels) * 100 if len(train_labels) > 0 else 0.0
        logger.info("    Class %d (%s): %5d (%5.1f%%)", idx, label, count, pct)

    logger.info("VALIDATION SET STATISTICS:")
    logger.info("  Total images: %d", len(val_df))
    logger.info("  Images with labels: %d", len(val_df[val_df["yolo_label"] != ""]))
    logger.info("  Images without labels: %d", len(val_df[val_df["yolo_label"] == ""]))
    logger.info("  Total regions: %d", len(val_labels))
    logger.info("  Label distribution:")
    for label, idx in sorted(label_to_index.items(), key=lambda item: item[1]):
        count = int((val_labels["grouped_label"] == label).sum())
        pct = count / len(val_labels) * 100 if len(val_labels) > 0 else 0.0
        logger.info("    Class %d (%s): %5d (%5.1f%%)", idx, label, count, pct)

    logger.info("CLASS IMBALANCE CHECK:")
    for label, idx in sorted(label_to_index.items(), key=lambda item: item[1]):
        train_pct = (
            (train_labels["grouped_label"] == label).sum() / len(train_labels) * 100 if len(train_labels) > 0 else 0.0
        )
        val_pct = (val_labels["grouped_label"] == label).sum() / len(val_labels) * 100 if len(val_labels) > 0 else 0.0
        diff = abs(train_pct - val_pct)
        status = "CRITICAL" if diff > 10 else "WARNING" if diff > 5 else "OK"
        logger.info(
            "[%s] Class %d (%s): Train %.1f%% vs Val %.1f%% (Δ %.1f%%)", status, idx, label, train_pct, val_pct, diff
        )

    train_bbox_stats = _compute_bbox_stats(train_labels, img_size_by_instrument)
    val_bbox_stats = _compute_bbox_stats(val_labels, img_size_by_instrument)

    logger.info("BOUNDING BOX STATISTICS:")
    logger.info(
        "Train width: mean=%.4f median=%.4f range=[%.4f, %.4f]",
        train_bbox_stats["mean_width"],
        train_bbox_stats["median_width"],
        train_bbox_stats["min_width"],
        train_bbox_stats["max_width"],
    )
    logger.info(
        "Train height: mean=%.4f median=%.4f range=[%.4f, %.4f]",
        train_bbox_stats["mean_height"],
        train_bbox_stats["median_height"],
        train_bbox_stats["min_height"],
        train_bbox_stats["max_height"],
    )
    logger.info(
        "Val width: mean=%.4f median=%.4f range=[%.4f, %.4f]",
        val_bbox_stats["mean_width"],
        val_bbox_stats["median_width"],
        val_bbox_stats["min_width"],
        val_bbox_stats["max_width"],
    )
    logger.info(
        "Val height: mean=%.4f median=%.4f range=[%.4f, %.4f]",
        val_bbox_stats["mean_height"],
        val_bbox_stats["median_height"],
        val_bbox_stats["min_height"],
        val_bbox_stats["max_height"],
    )


def _save_config(label_to_index: dict[str, int]) -> Path:
    """Write YOLO class names YAML file."""
    config_data = {
        "names": {idx: label for label, idx in label_to_index.items()},
        "nc": len(label_to_index),
        "train": str(cfg.YOLO_OUTPUT_MAG / "images" / "train"),
        "val": str(cfg.YOLO_OUTPUT_MAG / "images" / "val"),
    }

    config_path = Path(__file__).with_name("config.yaml")
    with open(config_path, "w", encoding="utf-8") as stream:
        yaml.dump(config_data, stream, default_flow_style=False, sort_keys=False)
    logger.info("Saved configuration to: %s", config_path)
    return config_path


def _process_images(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Generate PNG/label files for train and validation splits."""
    logger.info("Processing and saving FITS files...")
    local_root = str(cfg.DATA_FOLDER / cfg.DATASET_ROOT)
    mag_root = str(cfg.YOLO_OUTPUT_MAG)
    cont_root = str(cfg.YOLO_OUTPUT_CONT)

    def process(row: pd.Series, split: str) -> None:
        ut.process_fits_pair(row, local_root, mag_root, cont_root, split, cfg.RESIZE_DIM, cfg.USE_COLORMAP_MAG)

    logger.info("Processing %d train images...", len(train_df))
    p_map(lambda row: process(row, "train"), [row for _, row in train_df.iterrows()], num_cpus=cfg.NUM_CPUS)

    logger.info("Processing %d validation images...", len(val_df))
    p_map(lambda row: process(row, "val"), [row for _, row in val_df.iterrows()], num_cpus=cfg.NUM_CPUS)


def generate_yolo_dataset() -> dict[str, Any]:
    """Run the YOLO dataset generation pipeline."""
    from arccnet.models.fulldisk import utils as fd_utils

    logger.info("=" * 80)
    logger.info("YOLO DATASET GENERATION")
    logger.info("=" * 80)

    logger.info("Loading and filtering dataset...")
    df = _load_filtered_regions()
    df = _filter_existing_files(df)

    logger.info("Processing labels and creating YOLO annotations...")
    for label, count in df["magnetic_class"].value_counts().items():
        logger.info("%s: %d (%.1f%%)", label, count, count / len(df) * 100.0 if len(df) else 0.0)

    all_images = _all_images_table(df)
    logger.info("Total unique images (before label filter): %d", len(all_images))

    df_with_labels, label_to_index = _encode_labels(df)
    df_with_labels = _create_yolo_labels(df_with_labels, fd_utils.IMG_SIZE_BY_INSTRUMENT)
    df_yolo = _build_image_level_dataset(df_with_labels, all_images)
    logger.info("Total images in dataset: %d", len(df_yolo))

    train_df, val_df, actual_gap = _split_temporal(df_yolo)
    _log_split_stats(train_df, val_df, df_with_labels, label_to_index, fd_utils.IMG_SIZE_BY_INSTRUMENT)
    config_path = _save_config(label_to_index)
    _process_images(train_df, val_df)

    logger.info("Dataset generation complete!")
    logger.info("=" * 80)
    logger.info("SUMMARY:")
    logger.info("Total images: %d", len(df_yolo))
    logger.info("Train: %d (%.1f%%)", len(train_df), len(train_df) / len(df_yolo) * 100.0)
    logger.info("Validation: %d (%.1f%%)", len(val_df), len(val_df) / len(df_yolo) * 100.0)
    logger.info("Temporal gap: %d days", actual_gap)
    logger.info("Classes: %d", len(label_to_index))
    logger.info("Output directories:")
    logger.info("  Magnetogram: %s", cfg.YOLO_OUTPUT_MAG)
    logger.info("  Continuum: %s", cfg.YOLO_OUTPUT_CONT)
    logger.info("Configuration saved to: %s", config_path)
    logger.info("=" * 80)

    return {
        "df_yolo": df_yolo,
        "train_df": train_df,
        "val_df": val_df,
        "label_to_index": label_to_index,
        "config_path": config_path,
        "actual_gap_days": actual_gap,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_yolo_dataset()


if __name__ == "__main__":
    main()

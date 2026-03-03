"""Dataset utilities for McIntosh classification."""

import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

from arccnet.models import dataset_utils as ut_d
from arccnet.models import preprocessing_common as pp_common
from arccnet.visualisation import utils as ut_v

logger = logging.getLogger(__name__)


def display_sample_image(data_folder: str, dataset_folder: str, df: pd.DataFrame, index: int = 15):
    """
    Display a sample image from the dataset.

    Args:
        data_folder: Path to data directory
        dataset_folder: Dataset subdirectory name
        df: DataFrame with dataset info
        index: Sample index to display
    """

    row = df.iloc[index]
    fits_file_path = pp_common.resolve_preferred_cutout_fits_path(
        row,
        data_folder=data_folder,
        dataset_folder=dataset_folder,
    )
    if fits_file_path is None:
        raise FileNotFoundError(f"No valid cutout FITS path found for row index {index}")

    image_data = pp_common.load_fits_hdu_data(fits_file_path, hdu_index=1, dtype=np.float32)

    plt.figure(figsize=(10, 6))
    vlim = np.max(np.abs(image_data))
    plt.imshow(image_data, cmap=ut_v.magnetic_map, vmin=-vlim, vmax=vlim)
    plt.colorbar()
    plt.title(f"{row['date_only']} - {row['magnetic_class']} - McI: {row['mcintosh_class']} ")
    plt.show()


def process_ar_dataset(
    data_folder,
    dataset_folder="arccnet-v20251017/04_final",
    df_name="data/cutout_classification/region_classification.parq",
    plot_histograms=True,
    histogram_params=None,
    nan_threshold: float | None = None,
    verbose=False,
):
    """
    Process AR dataset: load, filter, group, encode, and optionally visualize.

    Args:
        data_folder: Path to data directory
        dataset_folder: Dataset subdirectory name
        df_name: Parquet catalog filename
        plot_histograms: Whether to plot distributions
        histogram_params: Histogram plotting parameters
        nan_threshold: Maximum NaN fraction (None to skip filtering)
        verbose: Enable verbose logging

    Returns:
        Tuple of (processed DataFrame, encoders dict, mappings dict)
    """

    if data_folder is None:
        data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../../data/")

    _, AR_df, _ = ut_d.make_dataframe(data_folder, dataset_folder, df_name)

    AR_df = ut_d.cleanup_df(AR_df, log_level=logging.INFO if verbose else None)

    AR_df = AR_df[AR_df["magnetic_class"] != ""].copy()
    AR_df = AR_df[AR_df["mcintosh_class"].notna()].copy()
    AR_df["mcintosh_class"] = AR_df["mcintosh_class"].astype(str).str.strip()
    AR_df = AR_df[AR_df["mcintosh_class"].str.len() >= 3].copy()

    if nan_threshold is not None:
        before_nan_filter = len(AR_df)
        AR_df, _ = pp_common.filter_cutouts_by_nan_threshold(
            AR_df,
            nan_threshold=nan_threshold,
            data_folder=data_folder,
            dataset_folder=dataset_folder,
            hdu_index=1,
        )
        if verbose:
            logger.info(
                "NaN filtering retained %s/%s samples (threshold=%.3f)",
                len(AR_df),
                before_nan_filter,
                nan_threshold,
            )

    AR_df["Z_component"] = AR_df["mcintosh_class"].str[0]
    AR_df["p_component"] = AR_df["mcintosh_class"].str[1]
    AR_df["c_component"] = AR_df["mcintosh_class"].str[2]

    default_hist_params = {
        "Z_component": {"y_off": 50, "ylim": 6600, "figsz": (10, 6), "title": "Z McIntosh Component"},
        "p_component": {"y_off": 50, "ylim": None, "figsz": (9, 6), "title": "p McIntosh Component"},
        "c_component": {"y_off": 50, "ylim": None, "figsz": (6, 6), "title": "c McIntosh Component"},
    }

    if histogram_params is not None:
        default_hist_params.update(histogram_params)

    if plot_histograms:
        ut_v.make_classes_histogram(
            AR_df["Z_component"],
            y_off=default_hist_params["Z_component"].get("y_off", 50),
            ylim=default_hist_params["Z_component"].get("ylim", None),
            figsz=default_hist_params["Z_component"].get("figsz", (10, 6)),
            title=default_hist_params["Z_component"].get("title", "Z McIntosh Component"),
        )
        ut_v.make_classes_histogram(
            AR_df["p_component"],
            y_off=default_hist_params["p_component"].get("y_off", 50),
            ylim=default_hist_params["p_component"].get("ylim", None),
            figsz=default_hist_params["p_component"].get("figsz", (9, 6)),
            title=default_hist_params["p_component"].get("title", "p McIntosh Component"),
        )
        ut_v.make_classes_histogram(
            AR_df["c_component"],
            y_off=default_hist_params["c_component"].get("y_off", 50),
            ylim=default_hist_params["c_component"].get("ylim", None),
            figsz=default_hist_params["c_component"].get("figsz", (6, 6)),
            title=default_hist_params["c_component"].get("title", "c McIntosh Component"),
        )

    z_component_mapping = {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "LG",
        "E": "LG",
        "F": "LG",
        "H": "H",
    }

    p_component_mapping = {
        "x": "x",
        "r": "r",
        "s": "sym",
        "h": "sym",
        "a": "asym",
        "k": "asym",
    }

    c_component_mapping = {"x": "x", "o": "o", "i": "frag", "c": "frag"}

    mappings = {
        "Z_component": z_component_mapping,
        "p_component": p_component_mapping,
        "c_component": c_component_mapping,
    }

    AR_df["Z_component_grouped"] = AR_df["Z_component"].map(z_component_mapping)
    AR_df["p_component_grouped"] = AR_df["p_component"].map(p_component_mapping)
    AR_df["c_component_grouped"] = AR_df["c_component"].map(c_component_mapping)
    grouped_cols = ["Z_component_grouped", "p_component_grouped", "c_component_grouped"]
    before_group_filter = len(AR_df)
    AR_df = AR_df.dropna(subset=grouped_cols).copy()
    if verbose and len(AR_df) != before_group_filter:
        logger.info(
            "Dropped %s samples with unsupported McIntosh grouped components.",
            before_group_filter - len(AR_df),
        )

    if plot_histograms:
        ut_v.make_classes_histogram(
            AR_df["Z_component_grouped"], y_off=50, figsz=(7, 6), title="Z McIntosh Component (Grouped)"
        )
        ut_v.make_classes_histogram(
            AR_df["p_component_grouped"], y_off=50, figsz=(6, 6), title="p McIntosh Component (Grouped)"
        )
        ut_v.make_classes_histogram(
            AR_df["c_component_grouped"], y_off=50, figsz=(5, 6), title="c McIntosh Component (Grouped)"
        )

    z_encoder = LabelEncoder()
    p_encoder = LabelEncoder()
    c_encoder = LabelEncoder()

    AR_df["Z_grouped_encoded"] = z_encoder.fit_transform(AR_df["Z_component_grouped"])
    AR_df["p_grouped_encoded"] = p_encoder.fit_transform(AR_df["p_component_grouped"])
    AR_df["c_grouped_encoded"] = c_encoder.fit_transform(AR_df["c_component_grouped"])

    if verbose:
        print("Z Component Label Encoding:", dict(zip(z_encoder.classes_, z_encoder.transform(z_encoder.classes_))))
        print("p Component Label Encoding:", dict(zip(p_encoder.classes_, p_encoder.transform(p_encoder.classes_))))
        print("c Component Label Encoding:", dict(zip(c_encoder.classes_, c_encoder.transform(c_encoder.classes_))))

    encoders = {"Z_encoder": z_encoder, "p_encoder": p_encoder, "c_encoder": c_encoder}

    return AR_df, encoders, mappings


def split_dataset(
    df: pd.DataFrame,
    group_column: str,
    plot_histograms: bool = False,
    histogram_params: dict | None = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test with group constraints.

    Args:
        df: Input DataFrame
        group_column: Column to group by (e.g., 'number')
        plot_histograms: Whether to plot distributions
        histogram_params: Histogram plotting parameters
        train_size: Training set proportion
        val_size: Validation set proportion
        test_size: Test set proportion
        random_state: Random seed
        verbose: Print split sizes and verification

    Returns:
        Train, validation, and test DataFrames

    Raises:
        ValueError: If split proportions don't sum to 1.0
        AssertionError: If groups overlap across splits
    """
    total = train_size + val_size + test_size
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"The sum of train_size, val_size, and test_size must be 1.0. Got {total}")

    train_prop = train_size
    remaining_prop = 1.0 - train_prop

    gss_train = GroupShuffleSplit(n_splits=1, test_size=remaining_prop, random_state=random_state)

    train_idx, remaining_idx = next(gss_train.split(df, groups=df[group_column]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    remaining_df = df.iloc[remaining_idx].reset_index(drop=True)

    val_prop = val_size / (val_size + test_size)

    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=1 - val_prop, random_state=random_state)

    val_idx, test_idx = next(gss_val_test.split(remaining_df, groups=remaining_df[group_column]))

    val_df = remaining_df.iloc[val_idx].reset_index(drop=True)
    test_df = remaining_df.iloc[test_idx].reset_index(drop=True)

    if verbose:
        total_samples = len(df)
        print(f"Train set: {len(train_df)} ({len(train_df) / total_samples * 100:.2f}%)")
        print(f"Validation set: {len(val_df)} ({len(val_df) / total_samples * 100:.2f}%)")
        print(f"Test set: {len(test_df)} ({len(test_df) / total_samples * 100:.2f}%)\n")

        train_groups = set(train_df[group_column])
        val_groups = set(val_df[group_column])
        test_groups = set(test_df[group_column])

        assert train_groups.isdisjoint(val_groups), "Overlap found between Train and Validation sets."
        assert train_groups.isdisjoint(test_groups), "Overlap found between Train and Test sets."
        assert val_groups.isdisjoint(test_groups), "Overlap found between Validation and Test sets."

        print("No overlap in groups across Train, Validation, and Test sets.")

    if plot_histograms:
        default_split_hist_params = {
            "train": {"y_off": 50, "ylim": None, "figsz": (10, 6), "title_prefix": "Train"},
            "val": {"y_off": 10, "ylim": None, "figsz": (9, 6), "title_prefix": "Validation"},
            "test": {"y_off": 10, "ylim": None, "figsz": (6, 6), "title_prefix": "Test"},
        }

        if histogram_params is not None:
            for split, params in histogram_params.items():
                if split in default_split_hist_params:
                    default_split_hist_params[split].update(params)
                else:
                    default_split_hist_params[split] = params

        splits = {"train": train_df, "val": val_df, "test": test_df}

        for split_name, split_df in splits.items():
            params = default_split_hist_params.get(split_name, {})
            prefix = params.pop("title_prefix", split_name.capitalize())

            ut_v.make_classes_histogram(
                split_df["Z_component"],
                y_off=params.get("y_off", 50),
                ylim=params.get("ylim", None),
                figsz=params.get("figsz", (10, 6)),
                title=f"{prefix} - Z McIntosh Component",
            )
            ut_v.make_classes_histogram(
                split_df["p_component"],
                y_off=params.get("y_off", 50),
                ylim=params.get("ylim", None),
                figsz=params.get("figsz", (9, 6)),
                title=f"{prefix} - p McIntosh Component",
            )
            ut_v.make_classes_histogram(
                split_df["c_component"],
                y_off=params.get("y_off", 50),
                ylim=params.get("ylim", None),
                figsz=params.get("figsz", (6, 6)),
                title=f"{prefix} - c McIntosh Component",
            )

            ut_v.make_classes_histogram(
                split_df["Z_component_grouped"],
                y_off=params.get("y_off", 50),
                figsz=(7, 6),
                title=f"{prefix} - Z McIntosh Component (Grouped)",
            )
            ut_v.make_classes_histogram(
                split_df["p_component_grouped"],
                y_off=params.get("y_off", 50),
                figsz=(6, 6),
                title=f"{prefix} - p McIntosh Component (Grouped)",
            )
            ut_v.make_classes_histogram(
                split_df["c_component_grouped"],
                y_off=params.get("y_off", 50),
                figsz=(5, 6),
                title=f"{prefix} - c McIntosh Component (Grouped)",
            )

    return train_df, val_df, test_df


class SunspotDataset(Dataset):
    """PyTorch Dataset for sunspot AR images and hierarchical labels."""

    def __init__(
        self,
        data_folder,
        dataset_folder,
        df,
        transform=None,
        target_height=100,
        target_width=200,
        divisor=800.0,
    ):
        """
        Initialize the SunspotDataset.

        Args:
            data_folder: Path to data directory
            dataset_folder: Dataset subdirectory name
            df: DataFrame with dataset info
            transform: Transformations to apply
            target_height: Target height for resizing
            target_width: Target width for resizing
            divisor: Normalization divisor
        """
        self.df = df
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.target_height = target_height
        self.target_width = target_width
        self.divisor = divisor

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, row: pd.Series) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Load and preprocess image from FITS file."""
        fits_file_path = pp_common.resolve_preferred_cutout_fits_path(
            row,
            data_folder=self.data_folder,
            dataset_folder=self.dataset_folder,
        )
        if fits_file_path is None:
            raise FileNotFoundError(f"No valid cutout FITS path found for row index {row.name}")

        image_data = pp_common.load_fits_hdu_data(fits_file_path, hdu_index=1, dtype=np.float32)

        image_data = np.nan_to_num(image_data, nan=0.0)

        image_data = ut_v.hardtanh_transform_npy(image_data, divisor=self.divisor, min_val=-1.0, max_val=1.0)
        image_data = ut_v.pad_resize_normalize(
            image_data, target_height=self.target_height, target_width=self.target_width
        )

        image = torch.from_numpy(image_data).unsqueeze(0)

        label = (int(row["Z_grouped_encoded"]), int(row["p_grouped_encoded"]), int(row["c_grouped_encoded"]))

        return image, label

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Get image and label tuple (Z, P, C) at specified index."""
        row = self.df.iloc[idx]
        image, label = self._load_image(row)

        if self.transform:
            image = self.transform(image)

        return image, label


def compute_weights(labels, num_classes):
    """
    Compute class weights for balanced training.

    Args:
        labels: Array of labels
        num_classes: Total number of classes

    Returns:
        Tensor of class weights
    """
    labels = np.asarray(labels, dtype=np.int64)
    class_weights = np.ones(int(num_classes), dtype=np.float32)
    unique_labels = np.unique(labels)

    if unique_labels.size > 0:
        present_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_labels,
            y=labels,
        )
        class_weights[unique_labels] = present_weights.astype(np.float32)

    missing_labels = sorted(set(range(int(num_classes))) - set(unique_labels.tolist()))
    if missing_labels:
        logger.warning(
            "Training split is missing classes %s; assigning zero class weight for those classes.",
            missing_labels,
        )
        class_weights[missing_labels] = 0.0

    return torch.tensor(class_weights, dtype=torch.float32)

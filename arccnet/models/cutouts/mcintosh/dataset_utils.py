import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

from astropy.io import fits

from arccnet.models import dataset_utils as ut_d
from arccnet.visualisation import utils as ut_v


def display_sample_image(data_folder: str, dataset_folder: str, df: pd.DataFrame, index: int = 15):
    """
    Displays a sample image from the dataset at the specified index.

    Args:
        data_folder (str): Path to the data directory.
        dataset_folder (str): Name of the dataset subdirectory.
        df (pd.DataFrame): DataFrame containing dataset information.
        index (int, optional): Index of the sample to display. Defaults to 15.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from astropy.io import fits

    row = df.iloc[index]
    path_key = "path_image_cutout_hmi" if row["path_image_cutout_hmi"] != "" else "path_image_cutout_mdi"
    fits_file_path = os.path.join(data_folder, dataset_folder, row[path_key])

    with fits.open(fits_file_path, memmap=True) as img_fits:
        image_data = np.array(img_fits[1].data, dtype=np.float32)

    plt.figure(figsize=(10, 6))
    vlim = np.max(np.abs(image_data))
    plt.imshow(image_data, cmap=ut_v.magnetic_map, vmin=-vlim, vmax=vlim)
    plt.colorbar()
    plt.title(f"{row['date_only']} - {row['magnetic_class']} - McI: {row['mcintosh_class']} ")
    plt.show()


def process_ar_dataset(
    data_folder: Optional[str] = None,
    dataset_folder: str = "arccnet-cutout-dataset-v20240715",
    df_name: str = "cutout-magnetic-catalog-v20240715.parq",
    plot_histograms: bool = True,
    histogram_params: Optional[dict[str, Any]] = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], dict[str, dict[str, str]]]:
    """
    Processes the AR dataset by loading, filtering, grouping, encoding, and optionally visualizing class distributions.

    Args:
        data_folder (str, optional): Path to the data directory. If None, it uses the environment variable
                                     "ARCAFF_DATA_FOLDER" or defaults to "../../../../data/".
        dataset_folder (str): Name of the dataset subdirectory.
        df_name (str): Filename of the Parquet file containing the catalog.
        plot_histograms (bool): Whether to plot histograms of class distributions. Defaults to True.
        histogram_params (dict, optional): Parameters for histogram plotting such as figure sizes,
                                           y-axis offsets, and limits.

    Returns:
        Tuple containing:
            - AR_df (pd.DataFrame): Processed DataFrame with grouped and encoded labels.
            - encoders (dict): Dictionary with LabelEncoders for Z, P, and C components.
            - mappings (dict): Dictionary with mapping rules applied to Z, P, and C components.
    """

    # Set default data_folder if not provided
    if data_folder is None:
        data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../../data/")

    # Load the DataFrame using utility function
    df, _ = ut_d.make_dataframe(data_folder, dataset_folder, df_name)

    # Filter out rows where 'magnetic_class' is empty
    AR_df = df[df["magnetic_class"] != ""].copy()

    # Extract McIntosh classification components
    AR_df["Z_component"] = AR_df["mcintosh_class"].str[0]
    AR_df["p_component"] = AR_df["mcintosh_class"].str[1]
    AR_df["c_component"] = AR_df["mcintosh_class"].str[2]

    # Default histogram parameters
    default_hist_params = {
        "Z_component": {"y_off": 50, "ylim": 6600, "figsz": (10, 6), "title": "Z McIntosh Component"},
        "p_component": {"y_off": 50, "ylim": None, "figsz": (9, 6), "title": "p McIntosh Component"},
        "c_component": {"y_off": 50, "ylim": None, "figsz": (6, 6), "title": "c McIntosh Component"},
    }

    # Update histogram parameters if provided
    if histogram_params is not None:
        default_hist_params.update(histogram_params)

    # Plot histograms for original components
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

    # Define grouping mappings
    z_component_mapping = {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "LG",  # Merge D, E, F into LG (LargeGroup)
        "E": "LG",
        "F": "LG",
        "H": "H",
    }

    p_component_mapping = {
        "x": "x",
        "r": "r",
        "s": "sym",  # Merge s and h into sym
        "h": "sym",
        "a": "asym",  # Merge a and k into asym
        "k": "asym",
    }

    c_component_mapping = {"x": "x", "o": "o", "i": "frag", "c": "frag"}  # Merge i and c into frag

    mappings = {
        "Z_component": z_component_mapping,
        "p_component": p_component_mapping,
        "c_component": c_component_mapping,
    }

    # Apply the mappings to the respective columns
    AR_df["Z_component_grouped"] = AR_df["Z_component"].map(z_component_mapping)
    AR_df["p_component_grouped"] = AR_df["p_component"].map(p_component_mapping)
    AR_df["c_component_grouped"] = AR_df["c_component"].map(c_component_mapping)

    # Plot histograms for grouped components
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

    # Initialize LabelEncoders
    z_encoder = LabelEncoder()
    p_encoder = LabelEncoder()
    c_encoder = LabelEncoder()

    # Fit and transform the grouped labels for each component
    AR_df["Z_grouped_encoded"] = z_encoder.fit_transform(AR_df["Z_component_grouped"])
    AR_df["p_grouped_encoded"] = p_encoder.fit_transform(AR_df["p_component_grouped"])
    AR_df["c_grouped_encoded"] = c_encoder.fit_transform(AR_df["c_component_grouped"])

    # Optionally, inspect the mappings
    print("Z Component Label Encoding:", dict(zip(z_encoder.classes_, z_encoder.transform(z_encoder.classes_))))
    print("p Component Label Encoding:", dict(zip(p_encoder.classes_, p_encoder.transform(p_encoder.classes_))))
    print("c Component Label Encoding:", dict(zip(c_encoder.classes_, c_encoder.transform(c_encoder.classes_))))

    # Compile encoders into a dictionary for easy access
    encoders = {"Z_encoder": z_encoder, "p_encoder": p_encoder, "c_encoder": c_encoder}

    return AR_df, encoders, mappings


def split_dataset(
    df: pd.DataFrame,
    group_column: str,
    plot_histograms: bool = False,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into training, validation, and testing subsets based on group constraints.
    Optionally plots histograms of class distributions for each split.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        group_column (str): The column name to group by (e.g., 'number').
        plot_histograms (bool, optional): Whether to plot histograms for each split. Defaults to False.
        histogram_params (dict, optional): Parameters for histogram plotting such as figure sizes,
                                           y-axis offsets, and limits for each split. Expected keys:
                                           'train', 'val', 'test', each mapping to a dict of parameters.
        train_size (float, optional): Proportion of the dataset to include in the training set. Defaults to 0.7.
        val_size (float, optional): Proportion of the dataset to include in the validation set. Defaults to 0.15.
        test_size (float, optional): Proportion of the dataset to include in the testing set. Defaults to 0.15.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        verbose (bool, optional): If True, prints the sizes and checks after splitting. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and testing DataFrames.

    Raises:
        ValueError: If the sum of train_size, val_size, and test_size does not equal 1.0.
        AssertionError: If there is an overlap in groups across the splits.
    """
    histogram_params = {
        "train": {"figsz": (12, 8), "title_prefix": "Train Set"},
        "val": {"figsz": (10, 6), "title_prefix": "Validation Set"},
        "test": {"figsz": (8, 5), "title_prefix": "Test Set"},
    }

    # Validate split ratios
    total = train_size + val_size + test_size
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"The sum of train_size, val_size, and test_size must be 1.0. Got {total}")

    # Calculate proportions for the first split (train vs. remaining)
    train_prop = train_size
    remaining_prop = 1.0 - train_prop

    # Initialize GroupShuffleSplit for the first split (train vs. remaining)
    gss_train = GroupShuffleSplit(n_splits=1, test_size=remaining_prop, random_state=random_state)

    # Perform the first split based on the group_column
    train_idx, remaining_idx = next(gss_train.split(df, groups=df[group_column]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    remaining_df = df.iloc[remaining_idx].reset_index(drop=True)

    # Calculate proportions for the second split (validation vs. test)
    val_prop = val_size / (val_size + test_size)  # Proportion within the remaining data

    # Initialize GroupShuffleSplit for the second split (val vs. test)
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=1 - val_prop, random_state=random_state)

    # Perform the second split based on the group_column
    val_idx, test_idx = next(gss_val_test.split(remaining_df, groups=remaining_df[group_column]))

    val_df = remaining_df.iloc[val_idx].reset_index(drop=True)
    test_df = remaining_df.iloc[test_idx].reset_index(drop=True)

    if verbose:
        # Display split sizes and proportions
        total_samples = len(df)
        print(f"Train set: {len(train_df)} ({len(train_df)/total_samples*100:.2f}%)")
        print(f"Validation set: {len(val_df)} ({len(val_df)/total_samples*100:.2f}%)")
        print(f"Test set: {len(test_df)} ({len(test_df)/total_samples*100:.2f}%)\n")

        # Verify that no groups are shared between splits
        train_groups = set(train_df[group_column])
        val_groups = set(val_df[group_column])
        test_groups = set(test_df[group_column])

        assert train_groups.isdisjoint(val_groups), "Overlap found between Train and Validation sets."
        assert train_groups.isdisjoint(test_groups), "Overlap found between Train and Test sets."
        assert val_groups.isdisjoint(test_groups), "Overlap found between Validation and Test sets."

        print("No overlap in groups across Train, Validation, and Test sets.")

    if plot_histograms:
        # Default histogram parameters for splits
        default_split_hist_params = {
            "train": {"y_off": 50, "ylim": None, "figsz": (10, 6), "title_prefix": "Train"},
            "val": {"y_off": 10, "ylim": None, "figsz": (9, 6), "title_prefix": "Validation"},
            "test": {"y_off": 10, "ylim": None, "figsz": (6, 6), "title_prefix": "Test"},
        }

        # Update histogram parameters if provided
        if histogram_params is not None:
            for split, params in histogram_params.items():
                if split in default_split_hist_params:
                    default_split_hist_params[split].update(params)
                else:
                    default_split_hist_params[split] = params

        # Define the splits and corresponding DataFrames
        splits = {"train": train_df, "val": val_df, "test": test_df}

        # Iterate over each split and plot histograms
        for split_name, split_df in splits.items():
            params = default_split_hist_params.get(split_name, {})
            prefix = params.pop("title_prefix", split_name.capitalize())

            # Plot histograms for original components in the split
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

            # Plot histograms for grouped components in the split
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
    """
    PyTorch Dataset for Sunspot AR images and hierarchical labels.
    """

    def __init__(
        self,
        data_folder: str,
        dataset_folder: str,
        df: pd.DataFrame,
        transform: Optional[callable] = None,
        target_height: int = 100,
        target_width: int = 200,
        divisor: float = 800.0,
    ):
        """
        Initializes the SunspotDataset.

        Args:
            data_folder (str): Path to the data directory.
            dataset_folder (str): Name of the dataset subdirectory.
            df (pd.DataFrame): DataFrame containing dataset information.
            transform (callable, optional): Transformations to apply to the images. Defaults to None.
            target_height (int, optional): Target height for image resizing. Defaults to 100.
            target_width (int, optional): Target width for image resizing. Defaults to 200.
            divisor (float, optional): Divisor for normalization. Defaults to 800.0.
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
        """
        Loads and preprocesses an image from a FITS file.
        """
        path_key = "path_image_cutout_hmi" if row["path_image_cutout_hmi"] != "" else "path_image_cutout_mdi"
        fits_file_path = os.path.join(self.data_folder, self.dataset_folder, row[path_key])

        with fits.open(fits_file_path, memmap=True) as img_fits:
            image_data = np.array(img_fits[1].data, dtype=np.float32)

        # Handle NaN values
        image_data = np.nan_to_num(image_data, nan=0.0)

        # Apply transformations
        image_data = ut_v.hardtanh_transform_npy(image_data, divisor=self.divisor, min_val=-1.0, max_val=1.0)
        image_data = ut_v.pad_resize_normalize(
            image_data, target_height=self.target_height, target_width=self.target_width
        )

        # Convert to tensor and add channel dimension
        image = torch.from_numpy(image_data).unsqueeze(0)  # Shape: (1, H, W)

        # Extract labels
        label = (int(row["Z_grouped_encoded"]), int(row["p_grouped_encoded"]), int(row["c_grouped_encoded"]))

        return image, label

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """
        Retrieves the image and label at the specified index, returning processed image tensor and label tuple (Z, P, C).
        """
        row = self.df.iloc[idx]
        image, label = self._load_image(row)

        if self.transform:
            image = self.transform(image)

        return image, label


def compute_weights(labels, num_classes):
    """
    Compute class weights using sklearn's compute_class_weight function.

    Args:
        labels (list or array-like): Array of labels for the dataset.
        num_classes (int): Total number of unique classes.

    Returns:
        torch.Tensor: Tensor of class weights.
    """
    class_weights = compute_class_weight(
        class_weight="balanced",  # Option for balanced weights
        classes=np.arange(num_classes),  # All class indices
        y=labels,  # Labels for the dataset
    )
    return torch.tensor(class_weights, dtype=torch.float)

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample

from astropy.time import Time

deg = np.pi / 180

label_to_index = {
    "QS": 0,
    "IA": 1,
    "Alpha": 2,
    "Beta": 3,
    "Beta-Gamma": 4,
    "Beta-Delta": 5,
    "Beta-Gamma-Delta": 6,
    "Gamma": 7,
    "Gamma-Delta": 8,
}


def make_dataframe(
    data_folder="../../data/",
    dataset_folder="arccnet-cutout-dataset-v20240715",
    file_name="cutout-mcintosh-catalog-v20240715.parq",
):
    """
    Processes the ARCCNet cutout dataset by loading a parquet file, converting Julian dates to datetime objects,
    filtering out problematic magnetograms, and categorizing the regions based on their magnetic class or type.

    Parameters:
    - data_folder (str): The base directory where the dataset folder is located. Default is '../../data/'.
    - dataset_folder (str): The folder containing the dataset. Default is 'arccnet-cutout-dataset-v20240715'.
    - file_name (str): The name of the parquet file to read. Default is 'cutout-mcintosh-catalog-v20240715.parq'.

    Returns:
    - df (pd.DataFrame): The processed DataFrame containing all regions with additional date and label columns.
    - AR_df (pd.DataFrame): A DataFrame filtered to include only active regions (AR) and intermediate regions (IA).
    """
    # Set the data folder using environment variable or default
    data_folder = os.getenv("ARCAFF_DATA_FOLDER", data_folder)

    # Read the parquet file
    df = pd.read_parquet(os.path.join(data_folder, dataset_folder, file_name))

    # Convert Julian dates to datetime objects
    df["time"] = df["target_time.jd1"] + df["target_time.jd2"]
    times = Time(df["time"], format="jd")
    dates = pd.to_datetime(times.iso)  # Convert to datetime objects
    df["dates"] = dates

    # Remove problematic magnetograms from the dataset
    problematic_quicklooks = ["20010116_000028_MDI.png", "20001130_000028_MDI.png", "19990420_235943_MDI.png"]

    filtered_df = []
    for ql in problematic_quicklooks:
        row = df["quicklook_path_mdi"] == "quicklook/" + ql
        filtered_df.append(df[row])
    filtered_df = pd.concat(filtered_df)
    df = df.drop(filtered_df.index).reset_index(drop=True)

    # Label the data
    df["label"] = np.where(df["magnetic_class"] == "", df["region_type"], df["magnetic_class"])
    df["date_only"] = df["dates"].dt.date

    # Filter AR and IA regions
    AR_df = pd.concat([df[df["region_type"] == "AR"], df[df["region_type"] == "IA"]])

    return df, AR_df


def undersample_group_filter(df, label_mapping, long_limit_deg=60, undersample=True, buffer_percentage=0.1):
    """
    This function filters the data based on a specified longitude limit, assigns 'front' or 'rear' locations, and
    groups labels according to a provided mapping.
    If undersampling is enabled, it reduces the majority class to the size of the second-largest class plus a
    specified buffer percentage.
    The function returns both the modified original dataframe with location and grouped labels and the undersampled dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data to be undersampled, grouped, and filtered.
    - label_mapping (dict): A dictionary mapping original labels to grouped labels.
    - long_limit_deg (int, optional): The longitude limit for filtering to determine 'front' or 'rear' location.
                                      Defaults to 60 degrees.
    - undersample (bool, optional): Flag to enable or disable undersampling of the majority class. Defaults to True.
    - buffer_percentage (float, optional): The percentage buffer added to the second-largest class size when undersampling
                                           the majority class. Defaults to 0.1 (10%).

    Returns:
    - pd.DataFrame: The modified original dataframe with 'location', 'grouped_labels' and 'encoded_labels' columns added.
    - pd.DataFrame: The undersampled and grouped dataframe, with rows from the 'rear' location filtered out.
    """
    lonV = np.deg2rad(np.where(df["processed_path_image_hmi"] != "", df["longitude_hmi"], df["longitude_mdi"]))
    condition = (lonV < -long_limit_deg * deg) | (lonV > long_limit_deg * deg)
    df_filtered = df[~condition]
    df_rear = df[condition]
    df.loc[df_filtered.index, "location"] = "front"
    df.loc[df_rear.index, "location"] = "rear"

    # Apply label mapping to the dataframe
    df["grouped_labels"] = df["label"].map(label_mapping)
    df["encoded_labels"] = df["grouped_labels"].map(label_to_index)

    if undersample:
        class_counts = df["grouped_labels"].value_counts()
        majority_class = class_counts.idxmax()
        second_largest_class_count = class_counts.iloc[1]
        n_samples = int(second_largest_class_count * (1 + buffer_percentage))

        # Perform undersampling on the majority class
        df_majority = df[df["grouped_labels"] == majority_class]
        df_majority_undersampled = resample(df_majority, replace=False, n_samples=n_samples, random_state=42)

        df_list = [df[df["grouped_labels"] == label] for label in class_counts.index if label != majority_class]
        df_list.append(df_majority_undersampled)

        df_du = pd.concat(df_list)
    else:
        df_du = df.copy()

    # Filter out rows with 'rear' location
    df_du = df_du[df_du["location"] != "rear"]

    return df, df_du


def split_data(df_du, label_col, group_col, random_state=42):
    """
    Split the data into training, validation, and test sets using stratified group k-fold cross-validation.

    Parameters:
    - df_du (pd.DataFrame): The dataframe to be split. It must contain the columns specified by `label_col` and `group_col`.
    - label_col (str): The name of the column to be used for stratification, ensuring balanced class distribution across folds.
    - group_col (str): The name of the column to be used for grouping, ensuring that all instances of a group are in the same fold.
    - random_state (int, optional): The random seed for reproducibility of the splits. Defaults to 42.

    Returns:
    - list of tuples containing:
        - fold (int): The fold number (1 to n_splits).
        - train_df (pd.DataFrame): The training set for the fold.
        - val_df (pd.DataFrame): The validation set for the fold.
        - test_df (pd.DataFrame): The test set for the fold.
    """
    fold_df = []
    inner_fold_choice = [0, 1, 2, 3, 4]
    sgkf = StratifiedGroupKFold(n_splits=5, random_state=random_state, shuffle=True)
    X = df_du

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_du, df_du[label_col], df_du[group_col]), 1):
        temp_df = X.iloc[train_idx]
        val_df = X.iloc[val_idx]
        inner_sgkf = StratifiedGroupKFold(n_splits=10)
        inner_splits = list(inner_sgkf.split(temp_df, temp_df[label_col], temp_df[group_col]))
        inner_train_idx, inner_test_idx = inner_splits[inner_fold_choice[fold - 1]]
        train_df = temp_df.iloc[inner_train_idx]
        test_df = temp_df.iloc[inner_test_idx]

        fold_df.append((fold, train_df, val_df, test_df))

    for fold, train_df, val_df, test_df in fold_df:
        X.loc[train_df.index, f"Fold {fold}"] = "train"
        X.loc[val_df.index, f"Fold {fold}"] = "val"
        X.loc[test_df.index, f"Fold {fold}"] = "test"

    return fold_df


def assign_fold_sets(df, fold_df):
    """
    Assigns training, validation, and test sets to the dataframe based on fold information.

    Parameters:
    - df (pd.DataFrame): Dataframe to be annotated with set information.
    - fold_df (list of tuples): List containing tuples for each fold.
      Each tuple consists of:
        - fold (int): The fold number.
        - train_df (pd.DataFrame): The training set for the fold.
        - val_df (pd.DataFrame): The validation set for the fold.
        - test_df (pd.DataFrame): The test set for the fold.

    Returns:
    - pd.DataFrame: The original dataframe with an additional 'set' column indicating training, validation, or test set.
    """
    for fold, train_set, val_set, test_set in fold_df:
        df.loc[train_set.index, f"Fold {fold}"] = "train"
        df.loc[val_set.index, f"Fold {fold}"] = "val"
        df.loc[test_set.index, f"Fold {fold}"] = "test"
    return df

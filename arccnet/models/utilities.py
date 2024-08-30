import os
import time
import random
import socket

import matplotlib  # noqa: F401
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from comet_ml.integration.pytorch import log_model
from matplotlib import pyplot as plt
from skimage import transform
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sunpy.visualization import colormaps as cm  # noqa: F401
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from astropy.io import fits
from astropy.time import Time

deg = np.pi / 180

magnetic_map = matplotlib.colormaps["hmimag"]

greek_mapping = {
    "Alpha": "α",
    "Beta": "β",
    "Gamma": "γ",
    "Delta": "δ",
}

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

index_to_label = {v: k for k, v in label_to_index.items()}


def convert_to_greek_label(names_array):
    def map_to_greek(name):
        parts = name.split("-")
        greek_parts = [greek_mapping.get(part, part) for part in parts]
        return "-".join(greek_parts)

    return np.array([map_to_greek(name) for name in names_array])


### Data Handling ###
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


### NN Training ###
class FITSDataset(Dataset):
    """
    Dataset class for loading and transforming magnetograms along with their corresponding labels.
    This class inherits from `torch.utils.data.Dataset`, making it compatible with PyTorch's DataLoader.

    Attributes:
    -----------
    data_folder : str
        The root directory containing the data.
    dataset_folder : str
        Directory containing inside the fits folder the FITS files.
    df : pd.DataFrame
        A DataFrame containing the file paths and corresponding labels for the images.
    transform : callable, optional
        A function/transform that takes in an image tensor and returns a transformed version.
        This can be used for data augmentation or normalization.
    cache : dict
        A dictionary used to cache loaded images in memory. The keys are the indices of the images,
        and the values are tuples containing the image tensors and their corresponding labels.
    target_height : int
        The target height to resize the images.
    target_width : int
        The target width to resize the images.
    divisor : float
        The divisor used for normalizing image pixel values.

    Methods:
    --------
    __init__(self, data_folder, dataset_folder, df, transform=None, target_height=224, target_width=224, divisor=1600.0)
        Initializes the dataset with the provided directories, DataFrame, and optional transformations.
        Initializes an empty cache for storing images in memory.

    __len__(self)
        Returns the number of samples in the dataset.

    __getitem__(self, idx)
        Retrieves the image and label at the specified index.
        If the image is cached, it retrieves the image from the cache.
        Otherwise, it loads the image from the FITS file, caches it, and
        then returns the transformed image along with its label.

    _load_image(self, row)
        Loads an image from the FITS file specified in the DataFrame row,
        converts it to a tensor, and returns it along with its label.

    Example Usage:
    --------------
    dataset = FITSDataset(data_folder='path/to/data/', dataset_folder='dataset_folder', df=df, transform=your_transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    """

    def __init__(
        self, data_folder, dataset_folder, df, transform=None, target_height=224, target_width=224, divisor=800.0
    ):
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        self.df = df
        self.transform = transform
        self.cache = {}
        self.target_height = target_height
        self.target_width = target_width
        self.divisor = divisor

    def _load_image(self, row):
        path = "path_image_cutout_hmi" if row["path_image_cutout_hmi"] != "" else "path_image_cutout_mdi"
        fits_file_path = os.path.join(self.data_folder, self.dataset_folder, row[path])
        with fits.open(fits_file_path, memmap=True) as img_fits:
            image_data = np.array(img_fits[1].data, dtype=np.float32)
        image_data = np.nan_to_num(image_data, nan=0.0)
        image_data = hardtanh_transform_npy(image_data, divisor=self.divisor, min_val=-1.0, max_val=1.0)
        image_data = pad_resize_normalize(image_data, target_height=self.target_height, target_width=self.target_width)
        image = torch.from_numpy(image_data).unsqueeze(0)
        return image, row["encoded_labels"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx in self.cache:
            image, label = self.cache[idx]
        else:
            row = self.df.iloc[idx]
            image, label = self._load_image(row)
            self.cache[idx] = (image, label)

        if self.transform:
            image = self.transform(image)

        return image, label


def replace_activations(module, old_act, new_act, **kwargs):
    """
    Recursively replace activation functions in a given module.

    Parameters:
    -----------
    module : torch.nn.Module
        The neural network in which to replace activation functions.

    old_act : type
        The class of the activation function to be replaced.
        For example, torch.nn.ReLU or torch.nn.Tanh.

    new_act : type
        The class of the new activation function to use as a replacement.
        For example, torch.nn.LeakyReLU.
    """
    for name, child in module.named_children():
        if isinstance(child, old_act):
            setattr(module, name, new_act(**kwargs))
        else:
            replace_activations(child, old_act, new_act, **kwargs)


def generate_run_id(config):
    """
    Generate a unique run ID for the current experiment.

    Parameters:
    - config: Configuration object containing information about the experiment.

    Returns:
    - A tuple containing the run ID and the directory path.
    """
    t = time.localtime()
    current_time = time.strftime("%Y%m%d-%H%M%S", t)
    run_id = f"{current_time}_{config.model_name}_GPU{torch.cuda.get_device_name()}_{socket.gethostname()}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        weights_dir = os.path.join(script_dir, "weights", f"{run_id}")
        os.makedirs(weights_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {weights_dir}: {e}")
        raise

    return run_id, weights_dir


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, scaler=None):
    """
    Train the model for one epoch.
    The autocast context manager is used to enable mixed precision for the forward pass.
    The GradScaler scales the loss to prevent underflow during backpropagation.
    After backpropagation, the gradients are unscaled before updating the model parameters.

    Args:
    - epoch (int): The current epoch number.
    - model (torch.nn.Module): The model to be trained.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
    - criterion (torch.nn.Module): The loss function.
    - optimizer (torch.optim.Optimizer): The optimizer used for model training.
    - device (torch.device): The device (CPU or GPU) on which to perform the training.
    - scaler (torch.amp.GradScaler()): GradScaler for mixed precision training.

    Returns:
    - avg_loss (float): The average training loss over the epoch.
    - accuracy (float): The training accuracy over the epoch.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_images = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
        labels = labels.long()
        inputs, labels = inputs.to(device), labels.to(device)

        # Check for NaNs or Infs in inputs
        assert not torch.isnan(inputs).any(), "Input contains NaNs"
        assert not torch.isinf(inputs).any(), "Input contains Infs"

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast(device_type="cuda"):  # Mixed precision training
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_images += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_images
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """
    Evaluate the model on the validation set.

    Args:
    - model (torch.nn.Module): The model to be evaluated.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
    - criterion (torch.nn.Module): The loss function.
    - device (torch.device): The device (CPU or GPU) on which to perform the evaluation.

    Returns:
    - avg_loss (float): The average validation loss.
    - accuracy (float): The validation accuracy.
    - precision (float): The validation precision score (macro-averaged).
    - recall (float): The validation recall score (macro-averaged).
    - f1 (float): The validation F1 score (macro-averaged).
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_images = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", unit="batch"):
            labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_images
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, precision, recall, f1


def check_early_stopping(val_metric, best_val_metric, patience_counter, model, weights_dir, config, fold_n=None):
    """
    Check for early stopping and save the model if the validation metric improves.

    Args:
    - val_metric (float): The current validation metric.
    - best_val_metric (float): The best validation metric so far.
    - patience_counter (int): Counter for the number of epochs without improvement.
    - model (torch.nn.Module): The model being trained.
    - weights_dir (str): Directory to save the model weights.
    - config (module): Configuration module containing various parameters like patience.
    - fold_n (int, optional): The current fold number. If None, cross-validation is not used.

    Returns:
    - best_val_metric (float): Updated best validation metric.
    - patience_counter (int): Updated patience counter.
    - stop_training (bool): Whether to stop training due to early stopping.
    """
    stop_training = False
    if val_metric > best_val_metric:
        best_val_metric = val_metric
        patience_counter = 0
        if fold_n is not None:
            model_save_path = os.path.join(weights_dir, f"best_model_fold{fold_n}.pth")
        else:
            model_save_path = os.path.join(weights_dir, "best_model.pth")
        torch.save(model.state_dict(), model_save_path)
    else:
        patience_counter += 1
        print(f"Early Stopping: {patience_counter}/{config.patience} without improvement.")
        if patience_counter >= config.patience:
            print("Stopping early due to no improvement in validation metric.")
            stop_training = True

    return best_val_metric, patience_counter, stop_training


def print_epoch_summary(
    epoch, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy, val_precision, val_recall, val_f1
):
    """
    Prints a summary of the training and validation metrics for a given epoch.

    Parameters:
    - epoch (int): The current epoch number (0-indexed).
    - avg_train_loss (float): The average training loss for the current epoch.
    - train_accuracy (float): The training accuracy for the current epoch.
    - avg_val_loss (float): The average validation loss for the current epoch.
    - val_accuracy (float): The validation accuracy for the current epoch.
    - val_precision (float): The validation precision for the current epoch.
    - val_recall (float): The validation recall for the current epoch.
    - val_f1 (float): The validation F1 score for the current epoch.
    """
    print(
        f"Epoch Summary {epoch+1}: "
        f"Train Loss: {avg_train_loss:.4f}, Train Acc.: {train_accuracy:.4f}, "
        f"Val. Loss: {avg_val_loss:.4f}, Val. Acc.: {val_accuracy:.4f}, "
        f"Val. Precision: {val_precision:.4f}, Val. Recall: {val_recall:.4f}, "
        f"Val. F1: {val_f1:.4f}"
    )


def load_model_test(weights_dir, model, device, fold_n=None):
    """
    Loads the best model weights from a specified directory and prepares the model for testing.

    Args:
        weights_dir (str): The directory where the model weights are stored.
        model (torch.nn.Module): The model to load the weights into.
        device (torch.device): The device to which the model is moved.
        fold_n (int, optional): The fold number in cross-validation. Defaults to None.

    Returns:
        torch.nn.Module: The model with the loaded weights.
    """
    if fold_n is not None:
        model_path = os.path.join(weights_dir, f"best_model_fold{fold_n}.pth")
    else:
        model_path = os.path.join(weights_dir, "best_model.pth")

    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def calculate_metrics(all_labels, all_preds):
    """
    Calculates evaluation metrics for the predictions made by the model.

    Args:
        all_labels (list): The ground truth labels.
        all_preds (list): The predicted labels by the model.

    Returns:
        tuple: A tuple containing test precision, recall, F1 score, confusion matrix, and classification report dataframe.
    """
    test_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm_test = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    return test_precision, test_recall, test_f1, cm_test, report_df


def test_model(model, test_loader, device, criterion):
    """
    Tests the model on the test dataset and calculates various evaluation metrics.

    Args:
        model (torch.nn.Module): The trained model to be tested.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device on which the model and data are loaded.
        criterion (torch.nn.Module): The loss function.

    Returns:
        tuple: A tuple containing average test loss, test accuracy, all labels, all predictions,
        test precision, recall, F1 score, confusion matrix, and classification report dataframe.
    """
    test_loss = 0
    total_correct = 0
    total_images = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            labels = labels.long()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            loss = criterion(outputs, labels)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = total_correct / total_images

    test_precision, test_recall, test_f1, cm_test, report_df = calculate_metrics(all_labels, all_preds)

    print(f"Average Test Loss: {avg_test_loss}")
    print("Confusion Matrix:")
    print(cm_test)
    print("Classification Report:")
    print(report_df)

    return avg_test_loss, test_accuracy, all_labels, all_preds, test_precision, test_recall, test_f1, cm_test, report_df


def count_trainable_parameters(model, print_num=False):
    """
    Calculate the total number of trainable parameters in a PyTorch model by iterating over
    all the parameters in the given model, checking if each parameter is trainable (i.e.,
    `requires_grad` is `True`), and sums up the total number of elements (parameters) in
    these tensors.

    Args:
        model (torch.nn.Module): The PyTorch model for which the number of trainable parameters
                                 is to be calculated.
        print_num (bool, optional): If set to True, the function will print the total number of
                                    trainable parameters. Default is False.

    Returns:
        num_params (int): The total number of trainable parameters in the model.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if print_num:
        print(f"Number of trainable parameters: {num_params}")
    return num_params


def train_model(config, df, weights_dir, experiment=None, fold=1):
    """
    This function handles data preparation, model initialization, training, validation, and testing
    across a specified fold of a given dataset.

    The function performs the following steps:
    1. Splits the dataset into training, validation, and test sets based on the specified fold.
    2. Optionally logs class distribution histograms for each set.
    3. Initializes the model using the timm library, along with the criterion, optimizer, and data loaders.
    4. Trains the model using a training loop with early stopping based on validation performance.
    5. Logs metrics and model summaries to the experiment object if provided.
    6. Evaluates the best model on the test set and logs results.
    7. Returns the evaluation metrics and classification report.

    Parameters:
    - config (object): A configuration object containing settings for data paths, model parameters,
        training hyperparameters, and device configuration (e.g., data folder, transforms, model name,
        batch size, number of epochs, learning rate).
    - df (pandas.DataFrame): The dataset containing all samples, along with fold assignments
        ('train', 'val', 'test').
    - weights_dir (str): The directory where model weights will be saved.
    - experiment (object, optional): A Comet experiment for logging metrics, visualizations, and other artifacts.
      If None, no logging will occur. Default is None.
    - fold (int, optional): The fold number to use for training and validation. Default is 1.

    Returns a tuple containing:
    - avg_test_loss (float): The average loss on the test set.
    - test_accuracy (float): The accuracy on the test set.
    - test_precision (float): The precision score on the test set.
    - test_recall (float): The recall score on the test set.
    - test_f1 (float): The F1 score on the test set.
    - cm_test (numpy.ndarray): The confusion matrix of the test set predictions.
    - report_df (pandas.DataFrame): A dataframe containing the classification report.
    """

    df_train = df[df[f"Fold {fold}"] == "train"]
    df_val = df[df[f"Fold {fold}"] == "val"]
    df_test = df[df[f"Fold {fold}"] == "test"]

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if experiment:
        train_image_path = os.path.join(script_dir, "temp", "train_dataset.png")
        val_image_path = os.path.join(script_dir, "temp", "val_dataset.png")
        test_image_path = os.path.join(script_dir, "temp", "test_dataset.png")
        make_classes_histogram(
            df_train["grouped_labels"], title="Train Dataset", y_off=100, figsz=(7, 5), save_path=train_image_path
        )
        experiment.log_image(train_image_path)
        make_classes_histogram(
            df_val["grouped_labels"], title="Val Dataset", y_off=100, figsz=(7, 5), save_path=val_image_path
        )
        experiment.log_image(val_image_path)
        make_classes_histogram(
            df_test["grouped_labels"], title="Test Dataset", y_off=100, figsz=(7, 5), save_path=test_image_path
        )
        experiment.log_image(test_image_path)

    num_classes = len(np.unique(df_train["encoded_labels"].values))

    train_dataset = FITSDataset(config.data_folder, config.dataset_folder, df_train, config.train_transforms)
    val_dataset = FITSDataset(config.data_folder, config.dataset_folder, df_val, config.val_transforms)
    test_dataset = FITSDataset(config.data_folder, config.dataset_folder, df_test, config.val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    model = timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=num_classes, in_chans=1)
    replace_activations(model, nn.ReLU, nn.LeakyReLU, negative_slope=0.01)
    device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    num_params = count_trainable_parameters(model, print_num=True)
    if experiment:
        log_model(experiment, model=model, model_name=config.model_name)
        experiment.set_model_graph(str(model))
        experiment.log_metric("trainable_parameters", num_params)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(df_train["encoded_labels"].values), y=df_train["encoded_labels"].values
    )
    alpha_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=alpha_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # scaler = torch.amp.GradScaler()
    scaler = GradScaler()

    # Training Loop
    best_val_metric = 0.0
    patience_counter = 0

    for epoch in range(config.num_epochs):
        avg_train_loss, train_accuracy = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device, scaler
        )
        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)
        val_metric = val_accuracy

        if experiment:
            experiment.log_metrics(
                {
                    "avg_train_loss": avg_train_loss,
                    "train_accuracy": train_accuracy,
                    "avg_val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                },
                epoch=epoch,
            )

        # early stopping
        best_val_metric, patience_counter, stop_training = check_early_stopping(
            val_metric, best_val_metric, patience_counter, model, weights_dir, config
        )
        if stop_training:
            break

        # Print epoch summary
        print_epoch_summary(
            epoch, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy, val_precision, val_recall, val_f1
        )

    # Evaluate the best model on the test set
    print("Testing...")
    model = load_model_test(weights_dir, model, device)
    (
        avg_test_loss,
        test_accuracy,
        all_labels,
        all_preds,
        test_precision,
        test_recall,
        test_f1,
        cm_test,
        report_df,
    ) = test_model(model, test_loader, device, criterion)

    if experiment:
        lbls = [value for value in config.label_mapping.values() if value is not None]
        unique_lbls = []
        for item in lbls:
            if item not in unique_lbls:
                unique_lbls.append(item)
        experiment.log_metrics(
            {
                "avg_test_loss": avg_test_loss,
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
            }
        )
        experiment.log_confusion_matrix(
            matrix=cm_test,
            title="Confusion Matrix at best val epoch",
            file_name="test_confusion_matrix_best_epoch.json",
            labels=unique_lbls,
        )
        experiment.log_text(report_df.to_string(), metadata={"type": "Classification Report"})
        csv_file_path = os.path.join(weights_dir, "classification_report.csv")
        report_df.to_csv(csv_file_path, index=False)
        experiment.log_table("classification_report.csv", tabular_data=report_df)

    # Log some misclassified examples
    if experiment:
        misclassified_indices = [i for i, (pred, label) in enumerate(zip(all_preds, all_labels)) if pred != label]
        random.shuffle(misclassified_indices)  # Shuffle to select random samples
        for idx in misclassified_indices[:20]:  # Log 20 misclassified examples
            img, true_label = test_dataset[idx]
            pred_label = all_preds[idx]
            experiment.log_image(
                img,
                name=f"Misclassified_{idx}_true{true_label}_pred{pred_label}",
                metadata={
                    "predicted_label": index_to_label[pred_label].title(),
                    "true_label": index_to_label[true_label].title(),
                },
            )

    return (avg_test_loss, test_accuracy, test_precision, test_recall, test_f1, cm_test, report_df)


### IMAGES ###
def pad_resize_normalize(image, target_height=224, target_width=224):
    """
    Adds padding to and resizes an image to specified target height and width.
    The function maintains the aspect ratio of the image by calculating the necessary padding.
    The image is padded with a constant value (default is 0.0) and then resized to the target dimensions.

    Parameters:
    - image (ndarray): The input image to be processed, can be in grayscale or RGB format.
    - target_height (int, optional): The target height of the image after resizing. Defaults to 224.
    - target_width (int, optional): The target width of the image after resizing. Defaults to 224.

    Returns:
    - ndarray: The processed image resized to the target dimensions with padding added as necessary.
    """

    original_height, original_width = image.shape[:2]
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    if original_aspect > target_aspect:
        # Image is wider than the target aspect ratio
        new_width = original_width
        new_height = int(original_width / target_aspect)
        padding_vertical = (new_height - original_height) // 2
        padding_horizontal = 0
    else:
        # Image is taller than the target aspect ratio
        new_height = original_height
        new_width = int(original_height * target_aspect)
        padding_horizontal = (new_width - original_width) // 2
        padding_vertical = 0

    # Adjust padding based on the image's dimensions
    if image.ndim == 3:  # Color image
        padding = ((padding_vertical, padding_vertical), (padding_horizontal, padding_horizontal), (0, 0))
    else:  # Grayscale image
        padding = ((padding_vertical, padding_vertical), (padding_horizontal, padding_horizontal))

    padded_image = np.pad(image, padding, "constant", constant_values=0.0)

    # Resize the padded image to the target size
    resized_image = transform.resize(padded_image, (target_height, target_width), anti_aliasing=True)

    return resized_image


def make_classes_histogram(
    series,
    figsz=(13, 5),
    y_off=300,
    title=None,
    titlesize=14,
    x_rotation=0,
    fontsize=11,
    bar_color="#4C72B0",
    edgecolor="black",
    text_fontsize=11,
    style="seaborn-v0_8-darkgrid",
    show_percentages=True,
    ax=None,
    save_path=None,
):
    """
    Creates and displays a bar chart (histogram) that visualizes the distribution of classes in a given pandas Series.

    Parameters:
    - series (pandas.Series):
        The input series containing the class labels.
    - figsz (tuple, optional):
        A tuple representing the size of the figure (width, height) in inches.
        Default is (13, 5).
    - y_off (int, optional):
        The vertical offset for the text labels above the bars.
        Default is 300.
    - title (str, optional):
        The title of the histogram plot. If `None`, no title will be displayed.
        Default is None.
    - titlesize (int, optional):
        The font size of the title text. Ignored if `title` is `None`.
        Default is 14.
    - x_rotation (int, optional):
        The rotation angle for the x-axis labels.
        Default is 0.
    - fontsize (int, optional):
        The font size of the x and y axis labels.
        Default is 11.
    - bar_color (str, optional):
        The color of the bars in the histogram.
        Default is '#4C72B0'.
    - edgecolor (str, optional):
        The color of the edges of the bars.
        Default is 'black'.
    - text_fontsize (int, optional):
        The font size of the text displayed above the bars.
        Default is 11.
    - style (str, optional):
        The matplotlib style to be used for the plot.
        Default is 'seaborn-v0_8-darkgrid'.
    - show_percentages (bool, optional):
        Whether to display percentages on top of the bars.
        Default is True.
    - ax (matplotlib.axes.Axes, optional):
        An existing matplotlib Axes object to plot on. If `None`, a new figure and Axes will be created.
        Default is None.
    - save_path (str, optional):
        Path to save the figure. If `None`, the plot will be displayed instead of saved.
        Default is None.
    """

    # Remove None values before sorting
    classes_names = sorted(filter(lambda x: x is not None, series.unique()))

    greek_labels = convert_to_greek_label(classes_names)
    classes_counts = series.value_counts().reindex(classes_names)
    values = classes_counts.values
    total = np.sum(values)

    with plt.style.context(style):
        if ax is None:
            plt.figure(figsize=figsz)
            bars = plt.bar(greek_labels, values, color=bar_color, edgecolor=edgecolor)
        else:
            bars = ax.bar(greek_labels, values, color=bar_color, edgecolor=edgecolor)

        # Add text on top of the bars
        for bar in bars:
            yval = bar.get_height()
            if show_percentages:
                percentage = f"{yval/total*100:.2f}%" if total > 0 else "0.00%"
                if ax is None:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        yval + y_off,
                        f"{yval} ({percentage})",
                        ha="center",
                        va="bottom",
                        fontsize=text_fontsize,
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        yval + y_off,
                        f"{yval} ({percentage})",
                        ha="center",
                        va="bottom",
                        fontsize=text_fontsize,
                    )
            else:
                if ax is None:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        yval + y_off,
                        f"{yval}",
                        ha="center",
                        va="bottom",
                        fontsize=text_fontsize,
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        yval + y_off,
                        f"{yval}",
                        ha="center",
                        va="bottom",
                        fontsize=text_fontsize,
                    )

        # Setting x and y ticks
        if ax is None:
            plt.xticks(rotation=x_rotation, ha="center", fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
        else:
            ax.set_xticks(np.arange(len(greek_labels)))
            ax.set_xticklabels(greek_labels, rotation=x_rotation, ha="center", fontsize=fontsize)
            ax.tick_params(axis="y", labelsize=fontsize)

        if title:
            if ax is None:
                plt.title(title, fontsize=titlesize)
            else:
                ax.set_title(title, fontsize=titlesize)

        # If a new figure was created, show the plot
        if ax is None:
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()
            else:
                plt.show()


class HardTanhTransform:
    """
    This transformation first scales the input image tensor by a specified divisor and then clamps the resulting values
    using the HardTanh function, which limits the values to a specified range [min_val, max_val].

    Attributes:
    - divisor (float):
        The value by which to divide the image tensor.
        This scaling factor is applied to the image before the HardTanh operation. Default is 800.0.
    - min_val (float):
        The minimum value to clamp the image tensor to using the HardTanh function. Default is -1.0.
    - max_val (float):
        The maximum value to clamp the image tensor to using the HardTanh function. Default is 1.0.

    Methods:
    - __call__(img):
        Applies the scaling and HardTanh transformation to the input image.

        Parameters:
        - img (PIL.Image or torch.Tensor):
            The input image, which can be either a PIL image or a PyTorch tensor.
            If a PIL image is provided, it will be converted to a tensor.

        Returns:
        - torch.Tensor:
            The transformed image tensor, scaled by the divisor and clamped to the range [min_val, max_val].
    """

    def __init__(self, divisor=800.0, min_val=-1.0, max_val=1.0):
        self.divisor = divisor
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        # Convert image to tensor if it's not already one
        if not torch.is_tensor(img):
            img = to_tensor(img)

        # Scale by the divisor and apply hardtanh
        img = img / self.divisor
        img = F.hardtanh(img, min_val=self.min_val, max_val=self.max_val)
        return img


def hardtanh_transform_npy(img, divisor=800.0, min_val=-1.0, max_val=1.0):
    """
    Apply HardTanh transformation to the input image.

    Args:
    - img: Input image (numpy array).
    - divisor: Value to divide the input image by.
    - min_val: Minimum value for the HardTanh function.
    - max_val: Maximum value for the HardTanh function.

    Returns:
    - Transformed image.
    """
    # Ensure the input is a NumPy array
    if not isinstance(img, np.ndarray):
        raise TypeError("Input should be a NumPy array")

    # Scale by the divisor
    img = img / divisor

    # Apply hardtanh
    img = np.clip(img, min_val, max_val)
    return img


def visualize_transformations(images, transforms, n_samples=16):
    """
    Visualize the effect of transformations on a set of images.

    Parameters:
    images (numpy.ndarray): Array of images to be transformed and visualized.
                            The shape should be (n_images, height, width).
    transforms (torchvision.transforms.Compose): The transformations to be applied to the images.
    n_samples (int): The number of sample images to visualize. Default is 16.

    Returns:
    None: Displays a plot with the transformed images in a 4x4 grid.
    """
    plt.figure(figsize=(12, 12))

    for i in range(n_samples):
        original_image = images[i]
        original_image_tensor = torch.from_numpy(original_image).float().unsqueeze(0)
        transformed_image_tensor = transforms(original_image_tensor)
        transformed_image = transformed_image_tensor.squeeze().numpy()

        # Plot transformed image
        plt.subplot(4, 4, i + 1)
        plt.imshow(transformed_image, cmap=magnetic_map, vmin=-1, vmax=1)
        plt.title(f"Transformed Image {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_location_on_sun(df, long_limit_deg=60, experiment=None):
    """
    Analyze and plot the distribution of solar cutouts based on their longitude.

    This function filters cutouts based on longitude, categorizes them as 'front' or 'rear',
    and plots their positiosns on a solar disc.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the solar data with latitude and longitude information.
    - long_limit_deg (int, optional): The longitude limit in degrees to determine front vs. rear. Default is 60 degrees.
    """
    latV = np.deg2rad(np.where(df["processed_path_image_hmi"] != "", df["latitude_hmi"], df["latitude_mdi"]))
    lonV = np.deg2rad(np.where(df["processed_path_image_hmi"] != "", df["longitude_hmi"], df["longitude_mdi"]))

    yV = np.cos(latV) * np.sin(lonV)
    zV = np.sin(latV)

    condition = (lonV < -long_limit_deg * deg) | (lonV > long_limit_deg * deg)

    rear_latV = latV[condition]
    lonV[condition]
    rear_yV = yV[condition]
    rear_zV = zV[condition]

    front_latV = latV[~condition]
    lonV[~condition]
    front_yV = yV[~condition]
    front_zV = zV[~condition]

    # Plot ARs' location on the solar disc
    circle = plt.Circle((0, 0), 1, edgecolor="gray", facecolor="none")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_artist(circle)

    num_meridians = 12
    num_parallels = 12
    num_points = 300

    phis = np.linspace(0, 2 * np.pi, num_meridians, endpoint=False)
    lats = np.linspace(-np.pi / 2, np.pi / 2, num_parallels)
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)

    # Plot each meridian
    for phi in phis:
        y = np.cos(theta) * np.sin(phi)
        z = np.sin(theta)
        ax.plot(y, z, "k-", linewidth=0.2)

    # Plot each parallel
    for lat in lats:
        radius = np.cos(lat)
        y = radius * np.sin(theta)
        z = np.sin(lat) * np.ones(num_points)
        ax.plot(y, z, "k-", linewidth=0.2)

    ax.scatter(rear_yV, rear_zV, s=1, alpha=0.2, color="r", label="Rear")
    ax.scatter(front_yV, front_zV, s=1, alpha=0.2, color="b", label="Front")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=12)

    # Save the plot

    num_rear_cutouts = len(rear_latV)
    num_front_cutouts = len(front_latV)
    percentage_rear = 100 * num_rear_cutouts / (num_rear_cutouts + num_front_cutouts)

    text_output = (
        f"Rear: {num_rear_cutouts}\n" f"Front: {num_front_cutouts}\n" f"Percentage of Rear: {percentage_rear:.2f}%"
    )

    if experiment:
        plot_path = os.path.join("temp", "solar_disc_plot.png")
        plt.savefig(plot_path)
        experiment.log_image(plot_path, name="Solar Disc Plot")
        experiment.log_text(text_output, metadata={"description": "Solar cutouts analysis"})

    print(text_output)
    plt.show()

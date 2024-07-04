import os
import random
import socket
import time

import cv2
import matplotlib
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits
from comet_ml.integration.pytorch import log_model
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from skimage import color, transform
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

magnetic_map = matplotlib.colormaps['hmimag'] #'gray' #


### Data Handling ###
def extract_datetime_from_filename(filename):
    # Extract parts of the filename based on its structure
    parts = filename.split('.')
    date_time = parts[3]  # Expected to be like '20171030_000000_TAI'
    date, time, _ = date_time.split('_')
    return date, time

def make_dataframe(root_dir):
    # Prepare a list to store the data
    data = []

    # Walk through the directory structure
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.fits'):
                # Determine the class based on the name of the subfolder
                class_name = os.path.basename(subdir)
                # Extract date and time from the filename
                date, time = extract_datetime_from_filename(file)
                # Create a full path to the file
                full_path = os.path.join(subdir, file)
                # Append the data to the list
                data.append({'Date': date, 'Time': time, 'magnetic_class': class_name, 'Path': full_path})

    # Create a DataFrame
    return pd.DataFrame(data)

def convert_fits_to_npy(fits_base_dir, npy_base_dir):
    """
    Converts all FITS files in a given directory structure to NPY format, preserving the directory structure.

    Parameters:
    fits_base_dir (str): The base directory containing the FITS files. The directory should contain subdirectories as needed.
    npy_base_dir (str): The base directory where the converted NPY files will be saved, preserving the same subdirectory structure as the FITS files.

    This function will:
    1. Walk through the directory tree starting from `fits_base_dir`.
    2. For each FITS file found, read the image data.
    3. Create a corresponding directory structure within `npy_base_dir`.
    4. Save the image data as an NPY file in the new directory structure.

    Example:
    Suppose the directory structure is as follows:
    
    fits_base_dir/
    ├── magnetic/
    │   ├── alpha/
    │   │   ├── image1.fits
    │   │   └── image2.fits
    │   ├── beta/
    │   │   └── image3.fits
    │   └── betax/
    │       └── image4.fits
    └── continuum/
        ├── alpha/
        │   └── image5.fits
        ├── beta/
        │   └── image6.fits
        └── betax/
            └── image7.fits
    
    After running the function with npy_base_dir set to "npy_dataset", the resulting directory structure will be:

    npy_dataset/
    ├── magnetic/
    │   ├── alpha/
    │   │   ├── image1.npy
    │   │   └── image2.npy
    │   ├── beta/
    │   │   └── image3.npy
    │   └── betax/
    │       └── image4.npy
    └── continuum/
        ├── alpha/
        │   └── image5.npy
        ├── beta/
        │   └── image6.npy
        └── betax/
            └── image7.npy

    Note:
    - The function assumes that the FITS files contain image data in the first header data unit (HDU).
    - Ensure that the directory structure within `fits_base_dir` is correctly set up to reflect the desired classification categories.

    Usage:
        # Define the base directories
        fits_base_directory = 'path/to/your/fits/files'
        npy_base_directory = 'path/to/save/npy/files'

        # Perform the conversion
        convert_fits_to_npy(fits_base_directory, npy_base_directory)
    """
    # Collect all FITS files paths
    fits_files = []
    for root, dirs, files in os.walk(fits_base_dir):
        for file in files:
            if file.endswith('.fits'):
                fits_files.append(os.path.join(root, file))

    # Process each FITS file with a progress bar
    for fits_file_path in tqdm(fits_files, desc="Converting FITS to NPY"):
        # Read FITS file
        with fits.open(fits_file_path) as hdul:
            image_data = hdul[1].data

        # Create corresponding NPY directory structure
        relative_path = os.path.relpath(os.path.dirname(fits_file_path), fits_base_dir)
        npy_dir = os.path.join(npy_base_dir, relative_path)
        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)

        # Save the image data as an NPY file
        npy_file_path = os.path.join(npy_dir, os.path.splitext(os.path.basename(fits_file_path))[0] + '.npy')
        np.save(npy_file_path, image_data)

def add_npy_paths(df, fits_base_dir, npy_base_dir):
    """
    Update the dataframe df to include NPY paths corresponding to the FITS paths.

    Parameters:
    df (pd.DataFrame): The dataframe to update.
    fits_base_dir (str): The base directory containing the FITS files.
    npy_base_dir (str): The base directory containing the converted NPY files.
    """
    def convert_to_npy_path(fits_path, fits_base_dir, npy_base_dir):
        relative_path = os.path.relpath(fits_path, fits_base_dir)
        npy_path = os.path.join(npy_base_dir, os.path.splitext(relative_path)[0] + '.npy')
        return npy_path

    df['continuum_npy_path'] = df['continuum_path'].apply(lambda x: convert_to_npy_path(x, fits_base_dir, npy_base_dir))
    df['magnetic_npy_path'] = df['magnetic_path'].apply(lambda x: convert_to_npy_path(x, fits_base_dir, npy_base_dir))

def get_fold_indices(df, fold_n):
    """
    Retrieve the train, validation, and test indices for a given fold.

    This function extracts the indices corresponding to the training, validation, 
    and test sets for a specified fold from a DataFrame. The fold information is 
    assumed to be stored in columns named 'Fold 1', 'Fold 2', etc.

    Parameters:
    df (pd.DataFrame): The DataFrame containing fold assignment information.
    fold_n (int): The fold number for which to retrieve the indices.

    Returns:
    tuple: A tuple containing three elements:
        - train_indices (pd.Index): Indices of the training set for the specified fold.
        - val_indices (pd.Index): Indices of the validation set for the specified fold.
        - test_indices (pd.Index): Indices of the test set for the specified fold.
    """
    fold_column = f'Fold {fold_n}'
    train_indices = df[df[fold_column] == 'train'].index
    val_indices = df[df[fold_column] == 'val'].index
    test_indices = df[df[fold_column] == 'test'].index
    return train_indices, val_indices, test_indices

def get_data_splits(magn_norm, cropped_magn, encoded_labels, indices):
    """
    Retrieve data splits for given indices.

    Parameters:
    magn_norm (np.ndarray): Array of normalized magnetograms.
    cropped_magn (np.ndarray): Array of cropped magnetograms.
    encoded_labels (np.ndarray): Array of encoded labels.
    indices (pd.Index or np.ndarray): Indices for the data split.

    Returns:
    tuple: Tuple containing the data splits:
        - X (np.ndarray): Normalized magnetograms for the split.
        - X_cropped (np.ndarray): Cropped magnetograms for the split.
        - y (np.ndarray): Encoded labels for the split.
    """
    X = magn_norm[indices]
    X_cropped = cropped_magn[indices]
    y = encoded_labels[indices]
    return X, X_cropped, y

def np2tensor(X_data, y_data, expand=True):
    """
    Convert numpy arrays to PyTorch tensors.

    Parameters:
    X_data (np.ndarray): Input data array.
    y_data (np.ndarray): Labels array.
    expand (bool, optional): If True, expand the dimensions of X_data. 
        This is used to add a channel dimension for grayscale images (default is True).

    Returns:
    tuple: A tuple containing:
        - X_data_tensor (torch.Tensor): The input data converted to a PyTorch tensor of type float.
        - y_data_tensor (torch.Tensor): The labels converted to a PyTorch tensor of type long.
    """
    if expand:
        X_data = np.expand_dims(X_data, axis = 1)
    X_data_tensor = torch.from_numpy(X_data).float()
    y_data_tensor = torch.from_numpy(y_data).long()
    return X_data_tensor, y_data_tensor

### NN Training ###
class Magnetograms(Dataset):
    """
    A custom dataset class for loading and transforming magnetogram images along with their corresponding labels.
    
    This class inherits from `torch.utils.data.Dataset`, making it compatible with PyTorch's DataLoader.

    Attributes:
    -----------
    images : torch.Tensor
        A tensor containing the magnetogram images. The shape of the tensor is (N, C, H, W), 
        where N is the number of images, C is the number of channels, H is the height, and W is the width.
    labels : torch.Tensor
        A tensor containing the labels for the images. The length of this tensor should be equal to N.
    transform : callable, optional
        A function/transform that takes in an image and returns a transformed version. 
        If None, no transformation is applied.

    Methods:
    --------
    __init__(self, images, labels, transform=None)
        Initializes the dataset with the provided images, labels, and optional transformations.
    __len__(self)
        Returns the number of samples in the dataset.
    __getitem__(self, idx)
        Retrieves the image and label at the specified index, 
        applying the optional transformation to the image if provided.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

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

    Returns:
    --------
    None
        This function modifies the module in place and does not return anything.
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
    run_id = f'{current_time}_{config.model_name}_GPU{str(config.gpu_index)}_{torch.cuda.get_device_name()}_{socket.gethostname()}'
    
    try:
        weights_dir = f"weights/{run_id}"
        os.makedirs(weights_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {weights_dir}: {e}")
        raise
    
    return run_id, weights_dir

def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, config, scaler):
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
    - config (module): Configuration module containing various parameters like number of epochs.
    - scaler (torch.cuda.amp.GradScaler): GradScaler for mixed precision training.

    Returns:
    - avg_loss (float): The average training loss over the epoch.
    - accuracy (float): The training accuracy over the epoch.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_images = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():  # Mixed precision training
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()* inputs.size(0)
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

def print_epoch_summary(epoch, fold_n, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy, val_precision, val_recall, val_f1):
    print(
        f"Epoch Summary {epoch+1} (Fold {fold_n}): "
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
        
    state_dict = torch.load(model_path)
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
    report_dict = classification_report(all_labels, all_preds, target_names=["Alpha", "Beta", "Beta-X"], output_dict=True)
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


def run_fold(fold_n, weights_dir, config, df_dataset, magn_norm, cropped_magn, encoded_labels, train_transforms, experiment = None):
    """
    Runs a single fold of cross-validation, including data loading, model training, and evaluation.

    Args:
        fold_n (int): The fold number in cross-validation.
        weights_dir (path): the folder where model's weights are stored.
        config (module): The configuration module containing parameters for training.
        df_dataset (pandas.DataFrame): DataFrame containing dataset information.
        magn_norm (numpy.ndarray): Normalized magnetic data.
        cropped_magn (numpy.ndarray): Cropped magnetic data.
        encoded_labels (numpy.ndarray): Encoded labels for the dataset.
        train_transforms (torchvision.transforms.Compose): Transformations to apply to the training data.
        experiment (comet_ml.Experiment, optional): Comet experiment object for logging.

    Returns:
        tuple: A tuple containing average test loss, test accuracy, test precision, 
            recall, F1 score, confusion matrix, and classification report dataframe.
    """
    
    label_name = ['alpha', 'beta', 'betax']
    print('\n--- Fold ' + str(fold_n) + '---\n')
    train_indices, val_indices, test_indices = get_fold_indices(df_dataset, fold_n)
    X_train, X_train_cropped, y_train = get_data_splits(magn_norm, cropped_magn, encoded_labels, train_indices)
    X_val, _, y_val = get_data_splits(magn_norm, cropped_magn, encoded_labels, val_indices)
    X_test, _, y_test = get_data_splits(magn_norm, cropped_magn, encoded_labels, test_indices)

    X_train = np.concatenate((X_train, X_train_cropped), axis=0)
    y_train = np.concatenate([y_train]*2, axis=0)

    # Convert to PyTorch tensors
    X_train, y_train = np2tensor(X_train, y_train)
    X_val, y_val = np2tensor(X_val, y_val)
    X_test, y_test = np2tensor(X_test, y_test)

    # Data loaders
    train_dataset = Magnetograms(X_train, y_train, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = Magnetograms(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    test_dataset = Magnetograms(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Create Model
    model = timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=len(label_name), in_chans=1)
    replace_activations(model, nn.ReLU, nn.LeakyReLU, negative_slope=0.01)
    device = torch.device("cuda:" + str(config.gpu_index)) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}\n")
    model.to(device)
    log_model(experiment, model=model, model_name=config.model_name)

    # Loss function and optimizer
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train.numpy()), y=y_train.numpy())
    alpha_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=alpha_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler = GradScaler()  # For mixed precision training

    if experiment:
        experiment.log_parameter('fold', fold_n)
    
    # Training Loop
    best_val_metric = 0.0
    patience_counter = 0

    for epoch in range(config.num_epochs):
        avg_train_loss, train_accuracy = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, config, scaler)
        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)
        val_metric = val_accuracy

        if experiment:
            with experiment.context_manager(f"fold {fold_n}"):
                experiment.log_metrics({
                    f'avg_train_loss_fold{fold_n}': avg_train_loss,
                    f'train_accuracy_fold{fold_n}': train_accuracy,
                    f'avg_val_loss_fold{fold_n}': avg_val_loss,
                    f'val_accuracy_fold{fold_n}': val_accuracy,
                    f'val_precision_fold{fold_n}': val_precision,
                    f'val_recall_fold{fold_n}': val_recall,
                    f'val_f1_fold{fold_n}': val_f1
                }, epoch=epoch)

        # early stopping
        best_val_metric, patience_counter, stop_training = check_early_stopping(val_metric, best_val_metric, patience_counter, model, weights_dir, config, fold_n)
        if stop_training:
            break

        # Print epoch summary
        print_epoch_summary(epoch, fold_n, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy, val_precision, val_recall, val_f1)

    # Evaluate the best model on the test set
    print('Testing...')
    model = load_model_test(weights_dir, model, device, fold_n)
    (avg_test_loss, test_accuracy, 
     all_labels, all_preds, 
     test_precision, test_recall, test_f1, 
     cm_test, report_df) = test_model(model, test_loader, device, criterion)

    if experiment:
        with experiment.context_manager(f"fold {fold_n}"):
            experiment.log_metrics({
                'avg_test_loss': avg_test_loss,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1})
            experiment.log_confusion_matrix(
                matrix=cm_test, 
                labels=["Alpha", "Beta", "Beta-X"], 
                title=f'Fold{fold_n} Confusion Matrix at best val epoch',
                file_name=f"fold{fold_n}_test_confusion_matrix_best_epoch.json")
            experiment.log_text(report_df.to_string(), metadata={"type": f"Fold {fold_n} Classification Report"})
            csv_file_path = os.path.join(weights_dir, f"fold{fold_n}_classification_report.csv")
            report_df.to_csv(csv_file_path, index=False)
            experiment.log_table(f"fold{fold_n}_classification_report.csv", tabular_data=report_df)
    
    # Log some misclassified examples
    if experiment:
        misclassified_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l]
        random.shuffle(misclassified_indices)  # Shuffle to select random samples
        for idx in misclassified_indices[:20]:  # Log 20 misclassified examples
            img = X_test[idx]
            pred_label = all_preds[idx]
            true_label = all_labels[idx]
            experiment.log_image(
                img,
                name=f"Misclassified_{idx}_true{true_label}_pred{pred_label}",
                metadata={"predicted_label": label_name[pred_label].title(), "true_label": label_name[true_label].title()}
            )
    
    return (avg_test_loss, test_accuracy, test_precision, test_recall, test_f1, cm_test, report_df)

def run_all_folds(weights_dir, config, df_dataset, magn_norm, cropped_magn, encoded_labels, train_transforms, experiment = None):
    """
    Run all folds for cross-validation, compute metrics, and print the results.

    Parameters:
    weights_dir (str): Directory where the model weights are saved.
    config (module): Configuration settings for training.
    df_dataset (pd.DataFrame): DataFrame containing the dataset information.
    magn_norm (np.array): Normalized magnetogram data.
    cropped_magn (np.array): Cropped magnetogram data.
    encoded_labels (np.array): Encoded labels for the dataset.
    train_transforms (torchvision.transforms.Compose): Composed data augmentation transforms.
    experiment (comet_ml.Experiment): Comet experiment object for logging.

    Returns:
    tuple: Contains lists of the following metrics for all folds:
        - test_losses (list): List of average test losses.
        - test_accuracies (list): List of test accuracies.
        - test_precisions (list): List of test precisions.
        - test_recalls (list): List of test recalls.
        - test_f1s (list): List of test F1 scores.
        - cms_test (list): List of confusion matrices for the test sets.
        - reports_df (list): List of classification reports as DataFrames.
    """
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []
    cms_test = []
    reports_df = []

    def compute_mean_min_max(metrics):
        mean_metric = np.mean(metrics)
        min_metric = np.min(metrics)
        max_metric = np.max(metrics)
        return mean_metric, min_metric, max_metric
    
    for fold_n in range(1, 5+1):
        (avg_test_loss, test_accuracy, test_precision, 
         test_recall, test_f1, cm_test, report_df) = run_fold(
            fold_n, weights_dir, config, df_dataset, magn_norm, 
            cropped_magn, encoded_labels, train_transforms, experiment)
    
        # Append results to lists
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1s.append(test_f1)
        cms_test.append(cm_test)
        reports_df.append(report_df)
    
    # Calculate mean, min, and max for each metric
    metrics = {
        'Test Loss': test_losses,
        'Test Accuracy': test_accuracies,
        'Test Precision': test_precisions,
        'Test Recall': test_recalls,
        'Test F1 Score': test_f1s
    }

    print('\n--- Averages over all Folds ---\n')
    for metric_name, metric_values in metrics.items():
        mean_metric, min_metric, max_metric = compute_mean_min_max(metric_values)
        print(f'Average {metric_name}: {mean_metric:.4f} (Min: {min_metric:.4f}, Max: {max_metric:.4f})')
        experiment.log_metric(f'average_{metric_name}', mean_metric)
        experiment.log_metric(f'min_{metric_name}', min_metric)
        experiment.log_metric(f'max_{metric_name}', max_metric)

    # Calculate and print F1 scores for alpha, beta, and betax
    alpha_f1s = []
    beta_f1s = []
    betax_f1s = []


    mag_class = ['Alpha', 'Beta', 'Beta-X']
    f1_scores = {category: [] for category in mag_class}

    for report in reports_df:
        for category in mag_class:
            f1_scores[category].append(report.loc[category, 'f1-score'])

    metrics = {}
    for category in mag_class:
        metrics[category] = compute_mean_min_max(f1_scores[category])

    for category in mag_class:
        mean, min_val, max_val = metrics[category]
        print(f'Average F1 Score for {category}: {mean:.4f} (Min: {min_val:.4f}, Max: {max_val:.4f})')

    if experiment:
        for category in mag_class:
            mean, min_val, max_val = metrics[category]
            experiment.log_metric(f'average_f1_{category.lower()}', mean)
            experiment.log_metric(f'min_f1_{category.lower()}', min_val)
            experiment.log_metric(f'max_f1_{category.lower()}', max_val)

    return test_losses, test_accuracies, test_precisions, test_recalls, test_f1s, cms_test, reports_df 

### IMAGES ###
def random_rotation(image, max_angle, crop=True):
    """
    Apply a random rotation to the image within 
    the range of -max_angle to max_angle degrees, and optionally crop to fill.
    
    Parameters:
    - image: numpy array representing the image to be rotated.
    - max_angle: max angle in deg for the rotation 
    - crop: boolean to determine if the image should be cropped after rotation
    Returns:
    - Rotated (and optionally cropped) image as a numpy array.
    """
    angle = random.uniform(-max_angle, max_angle)  # Random angle between -max_angle and max_angle degrees
    rotated_image = rotate(image, angle, reshape=False)
    
    if crop:
        # Find bounding box of non-zero pixels
        non_zero_coords = np.argwhere(rotated_image)
        top_left = non_zero_coords.min(axis=0)
        bottom_right = non_zero_coords.max(axis=0)
        cropped_image = rotated_image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
        return cropped_image
    
    return rotated_image

def random_crop(image, crop_percent=0.75, center_variation=0.1, seed=None):
    """
    Apply a random crop to the image with a crop size 
    up to crop_percent of the original image size.
    The center of the crop is varied slightly.
    
    Parameters:
    - image: numpy array representing the image to be cropped.
    - crop_percent: maximum percentage of the image to be cropped (default 0.75).
    - center_variation: percentage of variation for the crop center (default 0.1).
    - seed: optional random seed for reproducibility.
    
    Returns:
    - Cropped image as a numpy array.
    """
    if seed is not None:
        random.seed(seed)
    
    height, width = image.shape[:2]
    crop_height = int(height * crop_percent)
    crop_width = int(width * crop_percent)

    # Calculate variation ranges for the crop center
    center_y = height // 2
    center_x = width // 2
    var_y = int(center_variation * height)
    var_x = int(center_variation * width)

    # Randomly select the crop center within the variation range
    crop_center_y = random.randint(center_y - var_y, center_y + var_y)
    crop_center_x = random.randint(center_x - var_x, center_x + var_x)

    # Ensure the crop area is within the image bounds
    start_y = max(0, crop_center_y - crop_height // 2)
    start_x = max(0, crop_center_x - crop_width // 2)
    end_y = min(height, start_y + crop_height)
    end_x = min(width, start_x + crop_width)

    # Adjust start position if crop area exceeds image dimensions
    start_y = max(0, end_y - crop_height)
    start_x = max(0, end_x - crop_width)

    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image

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

def visualize_magnetograms(plt_idx, magn_norm, cropped_magn, encoded_labels,magnetic_map):
    label_name = ['alpha', 'beta', 'betax']
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.imshow(
        magn_norm[plt_idx], 
        origin='lower', cmap=magnetic_map, vmin=-1, vmax=1)
    plt.title('Normalized Magnetogram')
    plt.colorbar(shrink = 0.75)

    plt.subplot(1,2,2)
    plt.imshow(
        cropped_magn[plt_idx], 
        origin='lower', cmap=magnetic_map, vmin=-1, vmax=1)
    plt.title('Cropped Magnetogram')
    plt.colorbar(shrink = 0.75)
    plt.suptitle(label_name[encoded_labels[plt_idx]].title(), fontsize=16)

    plt.tight_layout()
    plt.show()

def convert_to_rgb(image):
    # Normalize the image to range [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    # Convert to RGB by repeating the single channel three times
    image = torch.stack([image, image, image], dim=0)
    return image

def make_classes_histogram(df, title):
    # Labels and values from the DataFrame
    classes_counts = df['magnetic_class'].value_counts().reindex(['alpha', 'beta', 'betax'], fill_value=0)
    labels = classes_counts.index.tolist()
    values = classes_counts.values
    total = np.sum(values)

    # Greek labels substitution
    greek_labels = [r'$\alpha$', r'$\beta$', r'$\beta-x$']

    # Setting the plot style and creating the plot
    with plt.style.context('seaborn-v0_8-darkgrid'):
        bars = plt.bar(greek_labels, values, edgecolor='black')

        # Add text on top of the bars
        for bar in bars:
            yval = bar.get_height()
            percentage = f"{yval/total*100:.2f}%" if total > 0 else "0.00%"
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval} ({percentage})", ha='center', va='bottom', fontsize=11)

        # Setting x and y ticks
        plt.xticks(rotation=0, ha='center', fontsize=11)
        plt.yticks(fontsize=12)
        plt.title(title, fontsize=16)


class HardTanhTransform:
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

def normalize_continuum_image(image):
    """
    Normalize the continuum image using MinMaxScaler approach.

    Args:
    - image: Input continuum image (numpy array).

    Returns:
    - Normalized image with values in the range [0, 1].
    """

    # Calculate the min and max values of the image
    min_val = np.min(image)
    max_val = np.max(image)

    # Apply MinMax normalization
    normalized_image = 1 - (image - min_val) / (max_val - min_val)

    return normalized_image

def uint8_conversion(image):
    """
    Scale the image to t0-255 range using convertScaleAbs.

    Args:
    - image: Input image (numpy array).

    Returns:
    - Scaled image in uint8 format.
    """
    # Calculate the min and max values of the image
    min_val = np.min(image)
    max_val = np.max(image)

    # Calculate the scaling factor
    if max_val != min_val:
        alpha = 255.0 / (max_val - min_val)
    else:
        alpha = 1.0  # Prevent division by zero

    # Apply the scaling and convert to uint8
    scaled_image = cv2.convertScaleAbs(image, alpha=alpha, beta=-min_val * alpha)
    
    return scaled_image
    
def find_AR_bbox(normalized_continuum_image, threshold_value=150, margin=50):
    """
    Process and visualize the steps to find and crop the ROI containing contours in a thresholded image.

    Parameters:
    - normalized_continuum_image: np.ndarray - The input image to be processed.
    - normalized_magnetic_image: np.ndarray - The magnetic image to be cropped.
    - threshold_value: int - The threshold value for binary thresholding. Default is 150.
    - margin: int - The margin to add around the bounding box. Default is 50.

    Returns:
    roi: np.ndarray - The region of interest containing the contours.
    """
    
    image_uint8 = uint8_conversion(normalized_continuum_image)
    _, binary_image = cv2.threshold(image_uint8, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the bounding box that contains all the contours
        all_contours = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_contours)
        
        # Add a margin to the bounding box
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image_uint8.shape[1] - x, w + 2 * margin)
        h = min(image_uint8.shape[0] - y, h + 2 * margin)
        bbox = (x, y, w, h)
    else:
        print("No contours found!")
        bbox = None

    return bbox

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
        plt.axis('off')

    plt.tight_layout()
    plt.show()

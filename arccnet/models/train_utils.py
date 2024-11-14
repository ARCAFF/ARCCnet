import os
import time
import random
import socket

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from comet_ml.integration.pytorch import log_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from astropy.io import fits

import arccnet.visualisation.utils as ut
from arccnet.models import labels as lbs


class FITSDataset(Dataset):
    """
    Dataset class for loading and transforming magnetograms along with their corresponding labels.
    This class inherits from `torch.utils.data.Dataset`, making it compatible with PyTorch's DataLoader.

    Parameters
    ----------
    data_folder : str
        The root directory containing the data.
    dataset_folder : str
        Directory containing the FITS files inside the folder.
    df : pandas.DataFrame
        A DataFrame containing the file paths and corresponding labels for the images.
    transform : callable, optional
        A function/transform that takes in an image tensor and returns a transformed version.
        This can be used for data augmentation or normalization. Default is None.
    target_height : int, optional
        The target height to resize the images. Default is 224.
    target_width : int, optional
        The target width to resize the images. Default is 224.
    divisor : float, optional
        The divisor used for normalizing image pixel values. Default is 1600.0.

    Attributes
    ----------
    data_folder : str
        The root directory containing the data.
    dataset_folder : str
        Directory containing the FITS files.
    df : pandas.DataFrame
        A DataFrame containing the file paths and corresponding labels for the images.
    transform : callable or None
        A transformation function for the images, if provided.
    target_height : int
        The target height to resize the images.
    target_width : int
        The target width to resize the images.
    divisor : float
        The divisor used for normalizing image pixel values.

    Methods
    -------
    __init__(self, data_folder, dataset_folder, df, transform=None, target_height=224, target_width=224, divisor=1600.0)
        Initializes the dataset with the provided directories, DataFrame, and optional transformations.
    __len__(self)
        Returns the number of samples in the dataset.
    __getitem__(self, idx)
        Retrieves the image and label at the specified index.
        Loads the image from the FITS file and applies any specified transformations.
    _load_image(self, row)
        Loads an image from the FITS file specified in the DataFrame row,
        converts it to a tensor, and returns it along with its label.

    Examples
    --------
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
        self.target_height = target_height
        self.target_width = target_width
        self.divisor = divisor

    def _load_image(self, row):
        path = "path_image_cutout_hmi" if row["path_image_cutout_hmi"] != "" else "path_image_cutout_mdi"
        fits_file_path = os.path.join(self.data_folder, self.dataset_folder, row[path])
        with fits.open(fits_file_path, memmap=True) as img_fits:
            image_data = np.array(img_fits[1].data, dtype=np.float32)
        image_data = np.nan_to_num(image_data, nan=0.0)
        image_data = ut.hardtanh_transform_npy(image_data, divisor=self.divisor, min_val=-1.0, max_val=1.0)
        image_data = ut.pad_resize_normalize(
            image_data, target_height=self.target_height, target_width=self.target_width
        )
        image = torch.from_numpy(image_data).unsqueeze(0)
        return image, row["encoded_labels"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image, label = self._load_image(row)

        if self.transform:
            image = self.transform(image)

        return image, label


def replace_activations(module, old_act, new_act, **kwargs):
    """
    Recursively replace activation functions in a given module.

    Parameters
    ----------
    module : torch.nn.Module
        The neural network module in which to replace activation functions.
    old_act : type
        The class of the activation function to be replaced.
        For example, `torch.nn.ReLU` or `torch.nn.Tanh`.
    new_act : type
        The class of the new activation function to use as a replacement.
        For example, `torch.nn.LeakyReLU`.
    **kwargs :
        Additional keyword arguments to pass to the constructor of the new activation function.

    Notes
    -----
    This function performs an in-place replacement of activation functions.
    It traverses the module hierarchy recursively and replaces instances of `old_act`
    with instances of `new_act`.

    Examples
    --------
    model = MyModel()
    replace_activations(model, torch.nn.ReLU, torch.nn.LeakyReLU, negative_slope=0.01)
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

    The `torch.amp.autocast` context manager is used to enable mixed precision for the forward pass.
    The `torch.amp.GradScaler` is used to scale the loss to prevent underflow during backpropagation.
    Gradients are unscaled before updating the model parameters when using mixed precision training.

    Parameters
    ----------
    epoch : int
        The current epoch number.
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training data.
    criterion : torch.nn.Module
        The loss function to be used during training.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model's parameters.
    device : torch.device
        The device (CPU or GPU) on which the training is performed.
    scaler : torch.amp.GradScaler, optional
        GradScaler for mixed precision training. Default is None.

    Returns
    -------
    avg_loss : float
        The average training loss over the epoch.
    accuracy : float
        The training accuracy over the epoch.

    Notes
    -----
    This function sets the model to training mode (`model.train()`) and processes each batch in
    the DataLoader using the specified loss function and optimizer. It supports mixed precision
    training if a `scaler` is provided.

    Examples
    --------
    avg_loss, accuracy = train_one_epoch(
            epoch=10,
            model=my_model,
            train_loader=my_train_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(my_model.parameters()),
            device=torch.device('cuda'),
            scaler=torch.amp.GradScaler()
            )
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

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation data.
    criterion : torch.nn.Module
        The loss function used to compute validation loss.
    device : torch.device
        The device (CPU or GPU) on which the evaluation is performed.

    Returns
    -------
    avg_loss : float
        The average validation loss over all batches.
    accuracy : float
        The validation accuracy as the percentage of correctly predicted labels.
    precision : float
        The validation precision score (macro-averaged) across all classes.
    recall : float
        The validation recall score (macro-averaged) across all classes.
    f1 : float
        The validation F1 score (macro-averaged) across all classes.

    Notes
    -----
    This function evaluates the model without modifying its parameters.
    It sets the model to evaluation mode (`model.eval()`) and ensures that no gradients are computed by using `torch.no_grad()`.
    After computing the predictions, various evaluation metrics are computed.

    Examples
    --------
    avg_loss, accuracy, precision, recall, f1 = evaluate(
         model=my_model,
         val_loader=my_val_loader,
         criterion=torch.nn.CrossEntropyLoss(),
         device=torch.device('cuda')
        s)
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
    Check for early stopping based on the validation metric and save the model if the metric improves.

    Parameters
    ----------
    val_metric : float
        The current validation metric (e.g., validation loss or accuracy).
    best_val_metric : float
        The best validation metric observed so far.
    patience_counter : int
        The counter tracking how many epochs have passed without improvement in the validation metric.
    model : torch.nn.Module
        The model being trained.
    weights_dir : str
        Directory where the model weights will be saved if the validation metric improves.
    config : module
        Configuration module containing various parameters, including `patience` for early stopping.
    fold_n : int, optional
        The current fold number for cross-validation. If `None`, cross-validation is not being used.
        Default is None.

    Returns
    -------
    best_val_metric : float
        Updated best validation metric.
    patience_counter : int
        Updated patience counter.
    stop_training : bool
        Whether to stop training due to early stopping criteria being met.

    Notes
    -----
    - Early stopping is triggered when the validation metric does not improve after a certain number
      of epochs (defined by `config.patience`).
    - If the validation metric improves, the model's weights are saved to the specified directory.
    - The function supports both single training runs and cross-validation.

    Examples
    --------
    best_val_metric, patience_counter, stop_training = check_early_stopping(
         val_metric=0.85,
         best_val_metric=0.80,
         patience_counter=2,
         model=my_model,
         weights_dir="./model_weights/",
         config=my_config,
         fold_n=1
     )
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
    Print a summary of the training and validation metrics for a given epoch.

    Parameters
    ----------
    epoch : int
        The current epoch number (0-indexed).
    avg_train_loss : float
        The average training loss for the current epoch.
    train_accuracy : float
        The training accuracy for the current epoch.
    avg_val_loss : float
        The average validation loss for the current epoch.
    val_accuracy : float
        The validation accuracy for the current epoch.
    val_precision : float
        The validation precision score for the current epoch.
    val_recall : float
        The validation recall score for the current epoch.
    val_f1 : float
        The validation F1 score for the current epoch.
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
    Load the best model weights from a specified directory and prepare the model for testing.

    Parameters
    ----------
    weights_dir : str
        The directory where the model weights are stored.
    model : torch.nn.Module
        The model into which the weights will be loaded.
    device : torch.device
        The device (CPU or GPU) to which the model is moved after loading the weights.
    fold_n : int, optional
        The fold number for cross-validation. If provided, the weights for the specific fold are loaded.
        Defaults to None.

    Returns
    -------
    torch.nn.Module
        The model with the loaded weights, ready for testing.

    Notes
    -----
    This function loads the best model weights saved during training from a specified file and
    prepares the model for testing by moving it to the specified device (CPU or GPU) and setting it
    to evaluation mode (`model.eval()`).
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
        ut.make_classes_histogram(
            df_train["grouped_labels"], title="Train Dataset", y_off=100, figsz=(7, 5), save_path=train_image_path
        )
        experiment.log_image(train_image_path)
        ut.make_classes_histogram(
            df_val["grouped_labels"], title="Val Dataset", y_off=100, figsz=(7, 5), save_path=val_image_path
        )
        experiment.log_image(val_image_path)
        ut.make_classes_histogram(
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
        experiment.set_model_graph(str(model))
        experiment.log_metric("trainable_parameters", num_params)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(df_train["encoded_labels"].values), y=df_train["encoded_labels"].values
    )
    alpha_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=alpha_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Get the CUDA version
    cuda_version = torch.version.cuda
    if cuda_version and float(cuda_version) < 11.8:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = torch.amp.GradScaler("cuda")

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
    print("Testing")
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
        log_model(experiment, model=model, model_name=config.model_name)
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
                    "predicted_label": lbs.index_to_label[pred_label].title(),
                    "true_label": lbs.index_to_label[true_label].title(),
                },
            )

    return (avg_test_loss, test_accuracy, test_precision, test_recall, test_f1, cm_test, report_df)

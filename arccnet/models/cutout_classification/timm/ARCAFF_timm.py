# %%
import os
import time

import config_timm as config
import neptune
import numpy as np
import timm
import torch
import wandb
from comet_ml import Experiment
from joblib import Parallel, delayed
from neptune.integrations.tensorflow_keras import NeptuneCallback
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from arccnet.models.cutout_classification import utilities_cutout as ut

os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

# Logging initialization
logging = config.logging
if isinstance(logging, str):
    logging = [logging]

if "wandb" in logging:
    wandb.login()
    run = wandb.init(project=config.project_name, entity=config.entity)
    wandb.save("ARCAFF_CNN.py")
    wandb.save("config_CNN.py")

if "neptune" in logging:
    run_neptune = neptune.init_run(project=config.project_name)
    neptune_callback = NeptuneCallback(run=run_neptune)

if "comet" in logging:
    run_comet = Experiment(project_name=config.project_name, workspace=config.workspace)
    run_comet.add_tags([config.model_name, config.loss])

# %% load data
print("Loading dataset...")
start_time = time.time()
loaded_data = np.load(config.data_path)
X_train = loaded_data["X_train"]
Y_train = loaded_data["Y_train"]
X_val = loaded_data["X_val"]
Y_val = loaded_data["Y_val"]
num_classes = len(np.unique(Y_train))
end_time = time.time()
print(f"done, in {end_time - start_time:.2f}s\n")

# %% make model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

model = timm.create_model(config.model_name, in_chans=1, pretrained=config.model_name, num_classes=num_classes)

model = model.to(device)
# %%
# PREPROCESSING
print("Start preprocessing...")
start_time = time.time()

# remove bars
X_train, Y_train = ut.remove_bars(X_train, Y_train)
X_val, Y_val = ut.remove_bars(X_val, Y_val)

if config.normalize:
    if config.normalize_type == "each_image":
        # Normalize training images
        X_train_min = X_train.min(axis=(1, 2), keepdims=True)
        X_train_max = X_train.max(axis=(1, 2), keepdims=True)
        X_train_range = X_train_max - X_train_min
        X_train = (X_train - X_train_min) / X_train_range

        # Normalize validation images
        X_val_min = X_val.min(axis=(1, 2), keepdims=True)
        X_val_max = X_val.max(axis=(1, 2), keepdims=True)
        X_val_range = X_val_max - X_val_min
        X_val = (X_val - X_val_min) / X_val_range

    if config.normalize_type == "dataset":
        # Compute the global minimum and maximum values from the training dataset
        global_min = X_train.min()
        global_max = X_train.max()

        # Normalize the training dataset
        X_train = (X_train - global_min) / (global_max - global_min)

        # Normalize the validation dataset using the same global minimum and maximum
        X_val = (X_val - global_min) / (global_max - global_min)

if config.augmentation:
    X_train, Y_train = ut.augment_data(X_train, Y_train)

# rescale and pad
if config.data_resize:
    X_train = np.array(
        Parallel(n_jobs=-1)(
            delayed(ut.pad_resize_normalize)(
                image,
                target_height=config.target_height,
                target_width=config.target_width,
                to_RGB=False,
                normalize=True,
            )
            for image in X_train
        )
    )
    X_val = np.array(
        Parallel(n_jobs=-1)(
            delayed(ut.pad_resize_normalize)(
                image,
                target_height=config.target_height,
                target_width=config.target_width,
                to_RGB=False,
                normalize=True,
            )
            for image in X_val
        )
    )

label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_val_encoded = label_encoder.transform(Y_val)

# Convert arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float).unsqueeze(1)
Y_train_tensor = torch.tensor(Y_train_encoded, dtype=torch.long)
Y_val_tensor = torch.tensor(Y_val_encoded, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

end_time = time.time()
print(f"done, in {end_time - start_time:.2f}s\n")

# %% train
if config.compute_class_weights:
    class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(Y_train_encoded), y=Y_train_encoded)
    alpha_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    if config.loss == "focal_loss":
        focal_loss = torch.hub.load(
            "adeelh/pytorch-multi-class-focal-loss",
            model="FocalLoss",
            alpha=alpha_tensor,
            gamma=config.gamma,
            reduction="mean",
            force_reload=False,
        )
        criterion = focal_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=alpha_tensor)
else:
    if config.loss == "focal_loss":
        focal_loss = torch.hub.load(
            "adeelh/pytorch-multi-class-focal-loss",
            model="FocalLoss",
            gamma=config.gamma,
            reduction="mean",
            force_reload=False,
        )
        criterion = focal_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

best_val_accuracy = 0.0  # Best validation accuracy observed
patience = config.patience  # Number of epochs to wait for improvement before stopping
patience_counter = 0  # Counter to track epochs without improvement

num_epochs = config.num_epochs

if "comet" in logging:
    run_comet.log_parameters(
        {
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
            "model": config.model_name,
            "patience": config.patience,
        }
    )

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_total_correct = 0
    train_total_images = 0
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for inputs, labels in train_progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        train_total_images += labels.size(0)
        train_total_correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = train_total_correct / train_total_images

    if "comet" in logging:
        run_comet.log_metric("loss", avg_train_loss, epoch=epoch)
        run_comet.log_metric("accuracy", train_accuracy, epoch=epoch)

    # Validation phase
    model.eval()
    val_loss = 0
    total_correct = 0
    total_images = 0
    all_labels = []
    all_preds = []
    val_progress_bar = tqdm(val_loader, desc="Validating", unit="batch")

    with torch.no_grad():
        for inputs, labels in val_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = total_correct / total_images
    val_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    # Early Stopping check
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        patience_counter = 0
        torch.save(model.state_dict(), "weights/best_model.pth")

        # Calculate and log the confusion matrix for the best model
        cm = confusion_matrix(all_labels, all_preds)
        if "comet" in logging:
            run_comet.log_confusion_matrix(
                matrix=cm,
                labels=["QS", "Alpha", "Beta", "Beta-Delta", "Beta-Gamma", "Beta-Gamma-Delta"],
                step=epoch,
                title="Confusion Matrix at Best Val. Epoch",
                file_name="confusion_matrix_best_epoch.json",
            )

        # log Binary classification
        binary_labels = np.array(all_labels) != 0
        binary_preds = np.array(all_preds) != 0
        # Calculate and log the binary classification confusion matrix
        binary_cm = confusion_matrix(binary_labels, binary_preds)
        TN, FP, FN, TP = binary_cm.ravel()
        binary_scores = ut.compute_scores(TN, FP, FN, TP)
        if "comet" in logging:
            run_comet.log_confusion_matrix(
                matrix=binary_cm,
                labels=["QS", "AR"],
                step=epoch,
                title="Binary Confusion Matrix at Best Model",
                file_name="binary_confusion_matrix_best_model.json",
            )

        score_names = [
            "TSS",
            "HSS",
            "CSI",
            "Recall",
            "Precision",
            "Specificity",
            "Accuracy",
            "F1 Score",
            "Balanced Accuracy",
        ]
        formatted_scores = zip(score_names, binary_scores)

        table = "| Metric            | Value     |\n"
        table += "|-------------------|-----------|\n"
        for name, score in formatted_scores:
            table += f"| {name:<17} | {score:<9.4f} |\n"

        print(table)
        if "comet" in logging:
            run_comet.log_text(table, metadata={"epoch": epoch})
            # upload best model
            run_comet.log_model("best_model", "weights/best_model.pth")

    else:
        patience_counter += 1
        print(f"Early Stopping: {patience_counter}/{patience} without improvement.")

        if patience_counter >= patience:
            print("Stopping early due to no improvement in validation accuracy.")
            break
    if "comet" in logging:
        run_comet.log_metric("val_loss", avg_val_loss, epoch=epoch)
        run_comet.log_metric("val_accuracy", val_accuracy, epoch=epoch)
        run_comet.log_metric("precision", val_precision, epoch=epoch)
        run_comet.log_metric("recall", val_recall, epoch=epoch)

    # Print epoch summary
    print(
        f"Epoch Summary {epoch+1}: "
        f"Train Loss: {avg_train_loss:.4f}, Train Acc.: {train_accuracy:.4f}, "
        f"Val. Loss: {avg_val_loss:.4f}, Val. Acc.: {val_accuracy:.4f}, "
        f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
    )

if "comet" in logging:
    run_comet.end()

# %%

# %%
# isort:skip_file
import os
import config_resnet as config
import numpy as np
import torch
from comet_ml import Experiment
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from tqdm import tqdm

from arccnet.models.cutout_classification import utilities_cutout as ut

os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
run_comet = Experiment(project_name=config.project_name, workspace=config.workspace)
run_comet.log_code("ARCAFF_ResNet.py")
run_comet.log_code("config_resnet.py")
run_comet.add_tags([config.model_name, config.loss])


# %%
loaded_data = np.load(config.data_path)
X_train = loaded_data["X_train"]
Y_train = loaded_data["Y_train"]
X_val = loaded_data["X_val"]
Y_val = loaded_data["Y_val"]

# %%
# PREPROCESSING
print("start preprocessing...\n")

# remove bars
X_train, Y_train = ut.remove_bars(X_train, Y_train)
X_val, Y_val = ut.remove_bars(X_val, Y_val)

if config.augmentation:
    X_train, Y_train = ut.augment_data(X_train, Y_train)

# rescale and pad
X_train_rescaled = np.array(
    Parallel(n_jobs=-1)(
        delayed(ut.pad_resize_normalize)(
            image, target_height=config.target_height, target_width=config.target_width, to_RGB=False, normalize=True
        )
        for image in X_train
    )
)

X_val_rescaled = np.array(
    Parallel(n_jobs=-1)(
        delayed(ut.pad_resize_normalize)(
            image, target_height=config.target_height, target_width=config.target_width, to_RGB=False, normalize=True
        )
        for image in X_val
    )
)

print("finished preprocessing.\n")

label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_val_encoded = label_encoder.transform(Y_val)

num_classes = len(np.unique(Y_train_encoded))

# %%
# Initialize the model
if config.model_name == "resnet18":
    if config.pretrained:
        from torchvision.models import ResNet18_Weights

        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18()

if config.model_name == "resnet34":
    if config.pretrained:
        from torchvision.models import ResNet34_Weights

        model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    else:
        model = models.resnet34()

if config.model_name == "resnet50":
    if config.pretrained:
        from torchvision.models import ResNet50_Weights

        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50()

model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

model = model.to(device)

# %%
# Convert arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_rescaled, dtype=torch.float).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_rescaled, dtype=torch.float).unsqueeze(1)
Y_train_tensor = torch.tensor(Y_train_encoded, dtype=torch.long)
Y_val_tensor = torch.tensor(Y_val_encoded, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# %%
# compute class weights by taking the inverse of the class frequencies,
# then normalizes the weights so they sum up to the number of classes.
class_sample_count = np.array([len(np.where(Y_train_encoded == t)[0]) for t in np.unique(Y_train_encoded)])
class_weights = 1.0 / class_sample_count
normalized_weights = class_weights / np.sum(class_weights) * len(np.unique(Y_train_encoded))

alpha_tensor = torch.tensor(normalized_weights, dtype=torch.float).to(device)  # Move alpha to device

if config.loss == "focal_loss":
    focal_loss = torch.hub.load(
        "adeelh/pytorch-multi-class-focal-loss",
        model="FocalLoss",
        alpha=alpha_tensor,
        gamma=2,
        reduction="mean",
        force_reload=False,
    )
    criterion = focal_loss
else:
    criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
# %%
best_val_accuracy = 0.0  # Best validation accuracy observed
patience = config.patience  # Number of epochs to wait for improvement before stopping
patience_counter = 0  # Counter to track epochs without improvement

num_epochs = config.num_epochs

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
        run_comet.log_text(table, metadata={"epoch": epoch})

        # upload best model
        run_comet.log_model("best_model", "weights/best_model.pth")

    else:
        patience_counter += 1
        print(f"Early Stopping: {patience_counter}/{patience} without improvement.")

        if patience_counter >= patience:
            print("Stopping early due to no improvement in validation accuracy.")
            break

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

run_comet.end()

# %%

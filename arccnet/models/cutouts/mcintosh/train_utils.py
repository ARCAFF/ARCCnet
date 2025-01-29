import os

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_z: nn.Module,
    criterion_p: nn.Module,
    criterion_c: nn.Module,
    teacher_forcing_ratio=None,
    scaler: torch.cuda.amp.GradScaler = None,
) -> dict:
    """
    Trains the model for one epoch with optional Teacher Forcing.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to run the training on.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        criterion_z (nn.Module): Loss function for Z component.
        criterion_p (nn.Module): Loss function for P component.
        criterion_c (nn.Module): Loss function for C component.
        teacher_forcing_ratio (float or None): Probability of using ground truth labels for Teacher Forcing.
                                               If None, Teacher Forcing is disabled.
        scaler (torch.cuda.amp.GradScaler, optional): GradScaler for mixed precision. Defaults to None.

    Returns:
        dict: Dictionary containing average loss, accuracies for Z, P, and C components, and average accuracy.
    """
    model.train()
    total_loss = 0.0

    correct_z, correct_p, correct_c = 0, 0, 0
    total_z, total_p, total_c = 0, 0, 0

    for inputs, (labels_z, labels_p, labels_c) in tqdm(train_loader, desc="Training", unit="batch"):
        inputs, labels_z, labels_p, labels_c = (
            inputs.to(device),
            labels_z.to(device),
            labels_p.to(device),
            labels_c.to(device),
        )

        optimizer.zero_grad()

        use_teacher_forcing = teacher_forcing_ratio is not None

        if scaler:
            with torch.amp.autocast("cuda"):
                output_z, output_p, output_c = model(
                    inputs,
                    Z_true=labels_z if use_teacher_forcing else None,
                    P_true=labels_p if use_teacher_forcing else None,
                    teacher_forcing_ratio=teacher_forcing_ratio if use_teacher_forcing else 0.0,
                )
                loss_z = criterion_z(output_z, labels_z)
                loss_p = criterion_p(output_p, labels_p)
                loss_c = criterion_c(output_c, labels_c)
                loss = loss_z + loss_p + loss_c
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output_z, output_p, output_c = model(
                inputs,
                Z_true=labels_z if use_teacher_forcing else None,
                P_true=labels_p if use_teacher_forcing else None,
                teacher_forcing_ratio=teacher_forcing_ratio if use_teacher_forcing else 0.0,
            )
            loss_z = criterion_z(output_z, labels_z)
            loss_p = criterion_p(output_p, labels_p)
            loss_c = criterion_c(output_c, labels_c)
            loss = loss_z + loss_p + loss_c

            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        # Compute accuracies for each sub-task
        _, predicted_z = torch.max(output_z.data, 1)
        _, predicted_p = torch.max(output_p.data, 1)
        _, predicted_c = torch.max(output_c.data, 1)

        correct_z += (predicted_z == labels_z).sum().item()
        correct_p += (predicted_p == labels_p).sum().item()
        correct_c += (predicted_c == labels_c).sum().item()

        total_z += labels_z.size(0)
        total_p += labels_p.size(0)
        total_c += labels_c.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy_z = correct_z / total_z
    accuracy_p = correct_p / total_p
    accuracy_c = correct_c / total_c
    avg_accuracy = (accuracy_z + accuracy_p + accuracy_c) / 3

    return {
        "train_loss": avg_loss,
        "train_accuracy_z": accuracy_z,
        "train_accuracy_p": accuracy_p,
        "train_accuracy_c": accuracy_c,
        "avg_train_accuracy": avg_accuracy,
    }


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion_z: nn.Module,
    criterion_p: nn.Module,
    criterion_c: nn.Module,
    teacher_forcing_ratio=None,
) -> dict:
    """
    Evaluates the model on a validation or test set with optional Teacher Forcing.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to run the evaluation on.
        loader (DataLoader): DataLoader for validation/test data.
        criterion_z (nn.Module): Loss function for Z component.
        criterion_p (nn.Module): Loss function for P component.
        criterion_c (nn.Module): Loss function for C component.
        teacher_forcing_ratio (float or None): Probability of using ground truth labels for Teacher Forcing.
                                               If None, Teacher Forcing is disabled.

    Returns:
        dict: Dictionary containing average loss, accuracies for Z, P, and C components, and average accuracy.
    """
    model.eval()
    total_loss = 0.0

    correct_z, correct_p, correct_c = 0, 0, 0
    total_z, total_p, total_c = 0, 0, 0

    with torch.no_grad():
        for inputs, (labels_z, labels_p, labels_c) in tqdm(loader, desc="Evaluating", unit="batch"):
            inputs, labels_z, labels_p, labels_c = (
                inputs.to(device),
                labels_z.to(device),
                labels_p.to(device),
                labels_c.to(device),
            )

            use_teacher_forcing = teacher_forcing_ratio is not None

            output_z, output_p, output_c = model(
                inputs,
                Z_true=labels_z if use_teacher_forcing else None,
                P_true=labels_p if use_teacher_forcing else None,
                teacher_forcing_ratio=teacher_forcing_ratio if use_teacher_forcing else 0.0,
            )

            # Compute loss
            loss_z = criterion_z(output_z, labels_z)
            loss_p = criterion_p(output_p, labels_p)
            loss_c = criterion_c(output_c, labels_c)
            loss = loss_z + loss_p + loss_c
            total_loss += loss.item() * inputs.size(0)

            # Compute accuracies
            _, predicted_z = torch.max(output_z.data, 1)
            _, predicted_p = torch.max(output_p.data, 1)
            _, predicted_c = torch.max(output_c.data, 1)

            correct_z += (predicted_z == labels_z).sum().item()
            correct_p += (predicted_p == labels_p).sum().item()
            correct_c += (predicted_c == labels_c).sum().item()

            total_z += labels_z.size(0)
            total_p += labels_p.size(0)
            total_c += labels_c.size(0)

    avg_loss = total_loss / len(loader.dataset)
    accuracy_z = correct_z / total_z
    accuracy_p = correct_p / total_p
    accuracy_c = correct_c / total_c
    avg_accuracy = (accuracy_z + accuracy_p + accuracy_c) / 3

    return {
        "val_loss": avg_loss,
        "val_accuracy_z": accuracy_z,
        "val_accuracy_p": accuracy_p,
        "val_accuracy_c": accuracy_c,
        "avg_val_accuracy": avg_accuracy,
    }


def apply_mask_at_evaluation(output_logits, z_pred, p_pred=None, valid_dict=None):
    """
    Applies a mask to logits at evaluation based on predicted Z or (Z, P).

    Args:
        output_logits (torch.Tensor): Logits for the P or C component (B, num_classes).
        z_pred (torch.Tensor): Predicted Z label (B,).
        p_pred (torch.Tensor or None): Predicted P label (B,) (optional, for C-component only).
        valid_dict (dict): Dictionary mapping Z or (Z, P) to valid classes.

    Returns:
        torch.Tensor: Masked logits with invalid classes set to -1e4.
    """
    batch_size, num_classes = output_logits.size()
    mask = torch.zeros((batch_size, num_classes), dtype=torch.bool, device=output_logits.device)

    for i in range(batch_size):
        if p_pred is None:  # Mask for P-component
            valid_classes = valid_dict.get(z_pred[i].item(), set())
        else:  # Mask for C-component
            valid_classes = valid_dict.get((z_pred[i].item(), p_pred[i].item()), set())

        if valid_classes:
            mask[i, list(valid_classes)] = True

    # Check if all classes are masked out
    all_masked = (~mask).all(dim=1)
    if all_masked.any():
        mask[all_masked] = True  # Allow all classes for these samples to avoid empty logits

    # Apply the mask to logits
    masked_logits = output_logits.masked_fill(~mask, -1e4)
    return masked_logits


def test(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    valid_p_for_z: dict,
    valid_c_for_zp: dict,
    teacher_forcing_ratio=None,
) -> tuple:
    """
    Tests the model and computes accuracy and F1 scores for each component with optional Teacher Forcing.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to run the testing on.
        loader (DataLoader): DataLoader for test data.
        valid_p_for_z (dict): Mapping of Z-labels to valid P-labels.
        valid_c_for_zp (dict): Mapping of (Z, P)-labels to valid C-labels.
        teacher_forcing_ratio (float or None): Probability of using ground truth labels for Teacher Forcing.
                                               If None, Teacher Forcing is disabled.

    Returns:
        tuple: Accuracy and F1 scores for Z, P, and C components respectively,
               and lists of true and predicted labels for each component.
    """
    model.eval()
    correct_z = 0
    correct_p = 0
    correct_c = 0

    total_z = 0
    total_p = 0
    total_c = 0

    true_labels_z = []
    pred_labels_z = []
    true_labels_p = []
    pred_labels_p = []
    true_labels_c = []
    pred_labels_c = []

    with torch.no_grad():
        for inputs, (labels_z, labels_p, labels_c) in tqdm(loader, desc="Testing", unit="batch"):
            inputs, labels_z, labels_p, labels_c = (
                inputs.to(device),
                labels_z.to(device),
                labels_p.to(device),
                labels_c.to(device),
            )

            use_teacher_forcing = teacher_forcing_ratio is not None

            output_z, output_p, output_c = model(
                inputs,
                Z_true=labels_z if use_teacher_forcing else None,
                P_true=labels_p if use_teacher_forcing else None,
                teacher_forcing_ratio=teacher_forcing_ratio if use_teacher_forcing else 0.0,
            )

            # Compute predictions for Z
            _, predicted_z = torch.max(output_z, 1)

            # Apply masking for P logits
            masked_logits_p = apply_mask_at_evaluation(output_p, predicted_z, valid_dict=valid_p_for_z)
            _, predicted_p = torch.max(masked_logits_p, 1)

            # Apply masking for C logits
            masked_logits_c = apply_mask_at_evaluation(output_c, predicted_z, predicted_p, valid_dict=valid_c_for_zp)
            _, predicted_c = torch.max(masked_logits_c, 1)

            # Count correct predictions
            correct_z += (predicted_z == labels_z).sum().item()
            correct_p += (predicted_p == labels_p).sum().item()
            correct_c += (predicted_c == labels_c).sum().item()

            # Total samples for accuracy computation
            total_z += labels_z.size(0)
            total_p += labels_p.size(0)
            total_c += labels_c.size(0)

            # Collect true and predicted labels for F1 score computation
            true_labels_z.extend(labels_z.cpu().numpy())
            pred_labels_z.extend(predicted_z.cpu().numpy())

            true_labels_p.extend(labels_p.cpu().numpy())
            pred_labels_p.extend(predicted_p.cpu().numpy())

            true_labels_c.extend(labels_c.cpu().numpy())
            pred_labels_c.extend(predicted_c.cpu().numpy())

    # Compute accuracy
    accuracy_z = correct_z / total_z if total_z > 0 else 0
    accuracy_p = correct_p / total_p if total_p > 0 else 0
    accuracy_c = correct_c / total_c if total_c > 0 else 0

    # Compute F1 scores
    f1_score_z = f1_score(true_labels_z, pred_labels_z, average="weighted")
    f1_score_p = f1_score(true_labels_p, pred_labels_p, average="weighted")
    f1_score_c = f1_score(true_labels_c, pred_labels_c, average="weighted")

    return (
        accuracy_z,
        accuracy_p,
        accuracy_c,
        f1_score_z,
        f1_score_p,
        f1_score_c,
        true_labels_z,
        pred_labels_z,
        true_labels_p,
        pred_labels_p,
        true_labels_c,
        pred_labels_c,
    )


def check_early_stopping(val_metric, best_val_metric, patience_counter, model, weights_dir, patience):
    stop_training = False
    if val_metric > best_val_metric:
        best_val_metric = val_metric
        patience_counter = 0
        model_save_path = os.path.join(weights_dir, "best_model.pth")
        torch.save(model.state_dict(), model_save_path)
    else:
        patience_counter += 1
        print(f"Early Stopping: {patience_counter}/{patience} without improvement.")
        if patience_counter >= patience:
            print("Stopping early due to no improvement in validation metric.")
            stop_training = True

    return best_val_metric, patience_counter, stop_training


def print_test_scores(accuracy_z, accuracy_p, accuracy_c, f1_z, f1_p, f1_c):
    """
    Prints test scores (accuracy and F1 scores) in a structured format using a DataFrame.

    Args:
        accuracy_z (float): Accuracy for Z component.
        accuracy_p (float): Accuracy for P component.
        accuracy_c (float): Accuracy for C component.
        f1_z (float): F1 score for Z component.
        f1_p (float): F1 score for P component.
        f1_c (float): F1 score for C component.
    """
    # Create a DataFrame
    data = {
        "Component": ["Z", "P", "C"],
        "Accuracy": [accuracy_z, accuracy_p, accuracy_c],
        "F1 Score": [f1_z, f1_p, f1_c],
    }
    df = pd.DataFrame(data)
    return df


def predict_region_class(model, region_input, encoders, device):
    model.eval()
    with torch.no_grad():
        outputs_z, outputs_p, outputs_c = model(region_input.to(device))

    # Get predicted indices
    _, pred_z = torch.max(outputs_z, 1)
    _, pred_p = torch.max(outputs_p, 1)
    _, pred_c = torch.max(outputs_c, 1)

    # Map indices to class labels
    final_class_z = encoders["Z_encoder"].inverse_transform(pred_z.cpu().numpy())
    final_class_p = encoders["p_encoder"].inverse_transform(pred_p.cpu().numpy())
    final_class_c = encoders["c_encoder"].inverse_transform(pred_c.cpu().numpy())

    # Combine predictions into a final class
    final_class = [(z, p, c) for z, p, c in zip(final_class_z, final_class_p, final_class_c)]

    return final_class


def calculate_f1_macro(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """
    Computes macro-averaged F1 scores for Z, P, and C components on the given DataLoader.

    Args:
        model (nn.Module): The trained neural network model.
        loader (DataLoader): DataLoader for the dataset.
        device (torch.device): The device to run inference on.

    Returns:
        tuple: Macro-averaged F1 scores for Z, P, and C components (f1_z, f1_p, f1_c).
    """
    model.eval()
    true_labels_z = []
    pred_labels_z = []
    true_labels_p = []
    pred_labels_p = []
    true_labels_c = []
    pred_labels_c = []

    with torch.no_grad():
        for inputs, (labels_z, labels_p, labels_c) in loader:
            inputs = inputs.to(device)
            labels_z = labels_z.to(device)
            labels_p = labels_p.to(device)
            labels_c = labels_c.to(device)

            # Get predictions
            output_z, output_p, output_c = model(inputs)
            _, predicted_z = torch.max(output_z, 1)
            _, predicted_p = torch.max(output_p, 1)
            _, predicted_c = torch.max(output_c, 1)

            # Collect true and predicted labels
            true_labels_z.extend(labels_z.cpu().numpy())
            pred_labels_z.extend(predicted_z.cpu().numpy())
            true_labels_p.extend(labels_p.cpu().numpy())
            pred_labels_p.extend(predicted_p.cpu().numpy())
            true_labels_c.extend(labels_c.cpu().numpy())
            pred_labels_c.extend(predicted_c.cpu().numpy())

    # Compute macro-averaged F1 scores
    f1_z = f1_score(true_labels_z, pred_labels_z, average="macro")
    f1_p = f1_score(true_labels_p, pred_labels_p, average="macro")
    f1_c = f1_score(true_labels_c, pred_labels_c, average="macro")

    return f1_z, f1_p, f1_c

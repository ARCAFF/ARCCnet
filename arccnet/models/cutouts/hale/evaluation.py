"""
Evaluation utilities for Hale classification models.

This module contains functions for model evaluation, including ROC curve generation,
confusion matrix logging, and misclassified sample analysis.
"""

import io
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Map any numeric image array to uint8 [0, 255] safely."""
    image = np.nan_to_num(image, nan=0.0)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - min_val) / (max_val - min_val)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def _class_name_for_index(class_names: list[str], index: int) -> str:
    """Return a safe class name for a numeric label index."""
    return class_names[index] if 0 <= index < len(class_names) else f"Class_{index}"


def generate_roc_curves(
    y_true: np.ndarray, y_pred_proba: np.ndarray, class_names: list[str], fold_num: int | None = None
) -> tuple[Image.Image, dict[str, dict[str, list[float] | float]]]:
    """
    Generate ROC curves for multiclass classification.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for each class
        class_names: Names of the classes
        fold_num: Fold number for the title (optional)

    Returns:
        Tuple of (ROC curve image, ROC data dictionary)
    """
    n_classes = len(class_names)

    # Binarize the output for multiclass ROC
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # Create figure
    plt.figure(figsize=(12, 8))

    # Colors for different classes
    colors = ["blue", "red", "green", "purple", "orange"]

    # Plot ROC curve for each class
    roc_data = {}
    for i, class_name in enumerate(class_names):
        color = colors[i] if i < len(colors) else f"C{i}"

        # Calculate ROC curve and AUC for this class
        class_targets = y_true_bin[:, i]
        if np.unique(class_targets).size < 2:
            roc_data[class_name] = {"fpr": [], "tpr": [], "auc": float("nan")}
            plt.plot([], [], color=color, lw=2, label=f"{class_name} (AUC = N/A)")
            continue

        fpr, tpr, _ = roc_curve(class_targets, y_pred_proba[:, i])
        roc_auc = float(auc(fpr, tpr))

        # Store data for later use
        roc_data[class_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}

        # Plot the ROC curve
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{class_name} (AUC = {roc_auc:.3f})")

    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier")

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    fold_text = f" - Fold {fold_num}" if fold_num is not None else ""
    plt.title(f"ROC Curves for Classification{fold_text}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Convert plot to image buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # Convert to PIL Image
    roc_image = Image.open(buf)
    plt.close()

    return roc_image, roc_data


def find_misclassified_samples(model: torch.nn.Module, data_module, fold_num: int, num_samples: int = 10) -> list[dict]:
    """
    Find worst misclassified samples and prepare them for logging.

    Args:
        model: Trained model
        data_module: Data module containing test dataloader
        fold_num: Current fold number
        num_samples: Number of misclassified samples to return

    Returns:
        List of dictionaries containing misclassified sample information
    """
    model.eval()
    test_loader = data_module.test_dataloader()
    misclassified_data = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(model.device)
            labels = labels.to(model.device)

            # Get predictions
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predicted = torch.argmax(logits, dim=1)

            # Find misclassified samples
            misclassified_mask = predicted != labels

            if misclassified_mask.any():
                # Get confidence of wrong predictions (high confidence = worse mistake)
                wrong_confidences = probabilities[misclassified_mask].max(dim=1)[0]

                misclassified_images = images[misclassified_mask]
                misclassified_true = labels[misclassified_mask]
                misclassified_pred = predicted[misclassified_mask]

                # Store misclassified data
                for i in range(len(misclassified_images)):
                    misclassified_data.append(
                        {
                            "image": misclassified_images[i].cpu(),
                            "true_label": misclassified_true[i].cpu().item(),
                            "pred_label": misclassified_pred[i].cpu().item(),
                            "confidence": wrong_confidences[i].cpu().item(),
                            "batch_idx": batch_idx,
                            "sample_idx": i,
                        }
                    )

    # Sort by confidence (highest confidence misclassifications first)
    misclassified_data.sort(key=lambda x: x["confidence"], reverse=True)

    # Take top N samples
    return misclassified_data[:num_samples]


def log_misclassified_samples(
    misclassified_data: list[dict], loggers: list, fold_num: int, class_names: list[str]
) -> None:
    """
    Log misclassified samples to experiment trackers.

    Args:
        misclassified_data: List of misclassified sample information
        loggers: List of experiment loggers
        fold_num: Current fold number
        class_names: Names of the classes
    """
    if not misclassified_data:
        logging.info(f"No misclassified samples found for fold {fold_num}")
        return

    for logger in loggers:
        try:
            if hasattr(logger, "experiment"):
                _log_to_comet_ml(logger, misclassified_data, fold_num, class_names)
                _log_to_tensorboard(logger, misclassified_data, fold_num, class_names)
        except Exception as e:
            logging.warning(f"Could not log misclassified samples to {type(logger).__name__}: {e}")


def _log_to_comet_ml(logger, misclassified_data: list[dict], fold_num: int, class_names: list[str]) -> None:
    """Log misclassified samples to Comet ML."""
    if not hasattr(logger.experiment, "log_image"):
        return

    for i, sample in enumerate(misclassified_data):
        img_tensor = sample["image"]

        # Handle single channel or multi-channel images
        if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
            # Single channel - squeeze and convert to grayscale
            img_np = _normalize_to_uint8(img_tensor.squeeze(0).numpy())
            img_pil = Image.fromarray(img_np, mode="L")
        else:
            # Multi-channel - assume RGB
            img_np = _normalize_to_uint8(img_tensor.permute(1, 2, 0).numpy())
            img_pil = Image.fromarray(img_np)

        true_class = _class_name_for_index(class_names, int(sample["true_label"]))
        pred_class = _class_name_for_index(class_names, int(sample["pred_label"]))
        confidence = sample["confidence"]

        logger.experiment.log_image(
            img_pil,
            name=f"fold_{fold_num}_misclassified_{i + 1}",
            step=fold_num,
            image_format="png",
            image_caption=f"True: {true_class}, Pred: {pred_class}, Conf: {confidence:.3f}",
        )


def _log_to_tensorboard(logger, misclassified_data: list[dict], fold_num: int, class_names: list[str]) -> None:
    """Log misclassified samples to TensorBoard."""
    if not hasattr(logger.experiment, "add_image"):
        return

    for i, sample in enumerate(misclassified_data):
        img_tensor = sample["image"]

        # Ensure tensor is in correct format for TensorBoard (C, H, W)
        display_tensor = img_tensor

        true_class = _class_name_for_index(class_names, int(sample["true_label"]))
        pred_class = _class_name_for_index(class_names, int(sample["pred_label"]))

        logger.experiment.add_image(
            f"fold_{fold_num}/misclassified_{i + 1}_true_{true_class}_pred_{pred_class}",
            display_tensor,
            global_step=fold_num,
            dataformats="CHW",
        )


def log_roc_curves(roc_image: Image.Image, roc_data: dict[str, dict], loggers: list, fold_num: int) -> None:
    """
    Log ROC curves to experiment trackers.

    Args:
        roc_image: PIL Image of the ROC curve plot
        roc_data: Dictionary containing ROC data for each class
        loggers: List of experiment loggers
        fold_num: Current fold number
    """
    for logger in loggers:
        try:
            if hasattr(logger, "experiment"):
                _log_roc_to_comet_ml(logger, roc_image, roc_data, fold_num)
                _log_roc_to_tensorboard(logger, roc_image, roc_data, fold_num)
        except Exception as e:
            logging.warning(f"Could not log ROC curves to {type(logger).__name__}: {e}")


def _log_roc_to_comet_ml(logger, roc_image: Image.Image, roc_data: dict, fold_num: int) -> None:
    """Log ROC curves to Comet ML."""
    if not hasattr(logger.experiment, "log_image"):
        return

    logger.experiment.log_image(roc_image, name=f"roc_curves_fold_{fold_num}", step=fold_num, image_format="png")

    # Log AUC values as metrics
    for class_name, data in roc_data.items():
        auc_value = data.get("auc")
        if isinstance(auc_value, (int, float)) and np.isfinite(float(auc_value)):
            logger.experiment.log_metric(
                f"fold_{fold_num}_auc_{class_name.lower().replace('-', '_')}",
                float(auc_value),
                step=fold_num,
            )


def _log_roc_to_tensorboard(logger, roc_image: Image.Image, roc_data: dict, fold_num: int) -> None:
    """Log ROC curves to TensorBoard."""
    if not hasattr(logger.experiment, "add_image"):
        return

    # Convert PIL image to tensor for TensorBoard
    img_array = np.array(roc_image.convert("RGB"))
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC to CHW

    logger.experiment.add_image(f"fold_{fold_num}/roc_curves", img_tensor, global_step=fold_num, dataformats="CHW")

    # Log AUC values as scalars
    for class_name, data in roc_data.items():
        auc_value = data.get("auc")
        if isinstance(auc_value, (int, float)) and np.isfinite(float(auc_value)):
            logger.experiment.add_scalar(
                f"fold_{fold_num}/auc_{class_name.lower().replace('-', '_')}",
                float(auc_value),
                global_step=fold_num,
            )


def log_confusion_matrix_and_classification_report(
    model, loggers: list, fold_num: int, class_names: list[str] | None = None
) -> None:
    """
    Log confusion matrix and classification report to all configured loggers.

    Args:
        model: The trained model with collected test predictions
        loggers: List of PyTorch Lightning loggers
        fold_num: Current fold number
        class_names: List of class names (e.g., ['Alpha', 'Beta', 'Beta-Gamma'])
    """
    # Get confusion matrix and classification report
    cm, class_report = model.get_confusion_matrix_and_classification_report(class_names)

    if cm is None or class_report is None:
        logging.warning("No test predictions found. Skipping confusion matrix and classification report.")
        return

    # Log confusion matrix as text
    _log_confusion_matrix_to_console(cm, class_report, class_names)

    # Log to each logger
    for logger in loggers:
        if hasattr(logger, "experiment"):
            _log_confusion_matrix_to_comet_ml(logger, cm, class_report, fold_num, class_names, model)
            _log_confusion_matrix_to_tensorboard(logger, cm, class_report, fold_num, class_names)

    logging.info("=" * 50)


def _log_confusion_matrix_to_console(cm, class_report, class_names: list[str] | None) -> None:
    """Log confusion matrix and classification report to console."""
    logging.info("=" * 50)
    logging.info("CONFUSION MATRIX")
    logging.info("=" * 50)
    logging.info("Confusion Matrix (rows=true, cols=predicted):")

    for i, row in enumerate(cm):
        class_name = class_names[i] if class_names and i < len(class_names) else f"Class_{i}"
        logging.info(f"{class_name:>12}: {row}")

    logging.info("=" * 50)
    logging.info("CLASSIFICATION REPORT")
    logging.info("=" * 50)

    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict):
            if class_name in ["accuracy", "macro avg", "weighted avg"]:
                if class_name == "accuracy":
                    logging.info(f"Overall Accuracy: {metrics:.4f}")
                else:
                    logging.info(
                        f"{class_name}: precision={metrics.get('precision', 0):.4f}, "
                        f"recall={metrics.get('recall', 0):.4f}, f1={metrics.get('f1-score', 0):.4f}"
                    )
            else:
                logging.info(
                    f"{class_name}: precision={metrics.get('precision', 0):.4f}, "
                    f"recall={metrics.get('recall', 0):.4f}, f1={metrics.get('f1-score', 0):.4f}, "
                    f"support={metrics.get('support', 0)}"
                )


def _log_confusion_matrix_to_comet_ml(
    logger, cm, class_report, fold_num: int, class_names: list[str] | None, model
) -> None:
    """Log confusion matrix to Comet ML."""
    if not hasattr(logger.experiment, "log_confusion_matrix"):
        return

    try:
        y_true = model.test_targets
        y_pred = model.test_predictions

        logger.experiment.log_confusion_matrix(
            y_true=y_true,
            y_predicted=y_pred,
            labels=class_names or [f"Class_{i}" for i in range(len(cm))],
            title=f"Confusion Matrix - Fold {fold_num}",
        )

        # Log classification report metrics individually
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict) and class_name not in ["accuracy"]:
                prefix = f"fold_{fold_num}_class_report_{class_name.replace(' ', '_')}"
                for metric_name, value in metrics.items():
                    if isinstance(value, int | float):
                        logger.experiment.log_metric(f"{prefix}_{metric_name}", value)

    except Exception as e:
        logging.warning(f"Could not log confusion matrix to Comet: {e}")


def _log_confusion_matrix_to_tensorboard(
    logger, cm, class_report, fold_num: int, class_names: list[str] | None
) -> None:
    """Log confusion matrix to TensorBoard."""
    if not hasattr(logger.experiment, "add_text"):
        return

    try:
        # Create text representation of confusion matrix
        cm_text = "Confusion Matrix:\n"
        for i, row in enumerate(cm):
            class_name = class_names[i] if class_names else f"Class_{i}"
            cm_text += f"{class_name}: {row}\n"

        # Create text representation of classification report
        report_text = "\nClassification Report:\n"
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict):
                if class_name == "accuracy":
                    report_text += f"Overall Accuracy: {metrics:.4f}\n"
                else:
                    report_text += f"{class_name}: "
                    report_text += f"precision={metrics.get('precision', 0):.4f}, "
                    report_text += f"recall={metrics.get('recall', 0):.4f}, "
                    report_text += f"f1={metrics.get('f1-score', 0):.4f}\n"

        step = fold_num if fold_num is not None else 0
        logger.experiment.add_text(
            f"Fold_{fold_num}/Confusion_Matrix_and_Classification_Report",
            cm_text + report_text,
            global_step=step,
        )

    except Exception as e:
        logging.warning(f"Could not log to TensorBoard: {e}")

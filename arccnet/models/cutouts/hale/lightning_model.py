"""
PyTorch Lightning model for Hale classification.
"""

import heapq
import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics import Accuracy, F1Score

import arccnet.models.cutouts.hale.config as config
from arccnet.models.train_utils import replace_activations

logger = logging.getLogger(__name__)


class HaleLightningModel(pl.LightningModule):
    """
    Simple Lightning model for Hale classification (Alpha, Beta, Beta-Gamma).
    Supports magnetogram, continuum, or both as input.
    """

    def __init__(
        self,
        num_classes: int = 3,
        learning_rate: float = 1e-3,
        model_name: str = "resnet18",
        class_weights: torch.Tensor = None,
        data_type: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.data_type = data_type or config.DATA_TYPE

        # Determine input channels based on data type
        if self.data_type == "both":
            input_channels = 2  # magnetogram + continuum
        else:
            input_channels = 1  # magnetogram or continuum only

        # Load pretrained ResNet and fall back to random init if weights are unavailable.
        resnet_configs = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        }
        if model_name not in resnet_configs:
            raise ValueError(f"Unsupported model: {model_name}")
        model_fn, pretrained_weights = resnet_configs[model_name]
        try:
            self.backbone = model_fn(weights=pretrained_weights)
        except Exception as exc:
            logger.warning(
                "Could not load pretrained weights for %s (%s). Falling back to random initialization.",
                model_name,
                exc,
            )
            self.backbone = model_fn(weights=None)

        # Replace ReLU with LeakyReLU activations
        replace_activations(self.backbone, nn.ReLU, nn.LeakyReLU, negative_slope=config.LEAKY_RELU_NEGATIVE_SLOPE)

        # Modify first conv layer for our input channels
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            input_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )

        # If using pretrained weights and input_channels=1, copy the RGB weights
        if input_channels == 1:
            self.backbone.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        elif input_channels == 2:
            # For 2 channels, use first 2 channels of RGB weights
            self.backbone.conv1.weight.data = original_conv1.weight.data[:, :2, :, :]

        # Modify final layer for our number of classes with dropout
        self.backbone.fc = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE), nn.Linear(self.backbone.fc.in_features, num_classes)
        )

        # Loss function with optional class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # Store for confusion matrix and classification report
        self.test_predictions = []
        self.test_targets = []
        self.test_logits = []
        self.test_misclassified_samples = []

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, labels)
        self.train_f1.update(preds, labels)

        # Log metrics
        batch_size = labels.size(0)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def on_train_epoch_end(self):
        """Log training epoch summary."""
        train_acc = float(self.train_accuracy.compute().detach().cpu())
        train_f1 = float(self.train_f1.compute().detach().cpu())
        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_f1", train_f1)
        print(f"Training Epoch {self.current_epoch} completed - Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        self.train_accuracy.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, labels)
        self.val_f1.update(preds, labels)

        # Log metrics
        batch_size = labels.size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def on_validation_epoch_end(self):
        """Log validation epoch summary."""
        val_acc = float(self.val_accuracy.compute().detach().cpu())
        val_f1 = float(self.val_f1.compute().detach().cpu())
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_f1", val_f1)
        print(f"Validation Epoch {self.current_epoch} completed - Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        self.val_accuracy.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, labels)
        self.test_f1.update(preds, labels)

        # Store predictions and targets for confusion matrix and classification report
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(labels.cpu().numpy())
        self.test_logits.append(logits.detach().cpu())
        self._update_top_misclassified_samples(images, labels, preds, logits, batch_idx)

        # Log metrics
        batch_size = labels.size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)

        return {"test_loss": loss, "preds": preds, "targets": labels}

    def on_test_epoch_end(self) -> None:
        """Reset aggregated test metrics after epoch end."""
        test_acc = float(self.test_accuracy.compute().detach().cpu())
        test_f1 = float(self.test_f1.compute().detach().cpu())
        self.log("test_acc", test_acc)
        self.log("test_f1", test_f1)
        self.test_accuracy.reset()
        self.test_f1.reset()

    def get_confusion_matrix_and_classification_report(self, class_names=None):
        """
        Compute confusion matrix and classification report from collected test predictions.

        Args:
            class_names: List of class names for the classification report

        Returns:
            tuple: (confusion_matrix, classification_report_dict)
        """
        if not self.test_predictions or not self.test_targets:
            return None, None

        # Convert to numpy arrays
        y_true = np.array(self.test_targets)
        y_pred = np.array(self.test_predictions)

        labels_idx = list(range(self.num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

        # Compute classification report
        if class_names is None or len(class_names) != self.num_classes:
            class_names = [f"Class_{i}" for i in range(self.num_classes)]

        # Get classification report as dictionary for better logging
        class_report = classification_report(
            y_true,
            y_pred,
            labels=labels_idx,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        return cm, class_report

    def _update_top_misclassified_samples(self, images, labels, preds, logits, batch_idx: int) -> None:
        """Track top-confidence misclassifications during the existing test pass using a heap for efficiency."""
        misclassified_mask = preds != labels
        if not misclassified_mask.any():
            return

        probabilities = torch.softmax(logits.detach(), dim=1)
        wrong_confidences = probabilities[misclassified_mask].max(dim=1)[0]
        misclassified_images = images[misclassified_mask].detach().cpu()
        misclassified_true = labels[misclassified_mask].detach().cpu()
        misclassified_pred = preds[misclassified_mask].detach().cpu()

        max_samples = max(1, int(getattr(config, "MISCLASSIFIED_SAMPLES_TO_LOG", 10)))

        for i in range(len(misclassified_images)):
            sample = {
                "image": misclassified_images[i],
                "true_label": misclassified_true[i].item(),
                "pred_label": misclassified_pred[i].item(),
                "confidence": wrong_confidences[i].detach().cpu().item(),
                "batch_idx": batch_idx,
                "sample_idx": i,
            }

            # Use heap to maintain only top-K samples without sorting every batch
            if len(self.test_misclassified_samples) < max_samples:
                # Negative confidence for max-heap behavior (Python has min-heap by default)
                heapq.heappush(self.test_misclassified_samples, (sample["confidence"], sample))
            elif sample["confidence"] > self.test_misclassified_samples[0][0]:
                heapq.heapreplace(self.test_misclassified_samples, (sample["confidence"], sample))

    def get_top_misclassified_samples(self, num_samples: int = 10) -> list[dict]:
        """Return cached misclassified samples sorted by confidence descending.

        Extracts samples from heap and sorts them once at the end.
        """
        if not self.test_misclassified_samples:
            return []

        # Extract samples from heap and sort by confidence (highest first)
        samples = [sample for _, sample in self.test_misclassified_samples]
        samples.sort(key=lambda x: x["confidence"], reverse=True)
        return samples[:num_samples]

    def reset_test_collections(self):
        """Reset the test predictions and targets collections."""
        self.test_predictions = []
        self.test_targets = []
        self.test_logits = []
        self.test_misclassified_samples = []  # Reset heap

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=config.WEIGHT_DECAY,  # Add L2 regularization
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.LR_SCHEDULER_MODE,
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": config.LR_SCHEDULER_MONITOR,
            },
        }

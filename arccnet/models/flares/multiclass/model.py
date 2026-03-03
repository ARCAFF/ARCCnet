"""PyTorch Lightning model for multiclass flare classification."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from arccnet.models import train_utils as ut_t
from arccnet.models.flares.multiclass import config

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal loss for multiclass classification."""

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss from logits and class-index targets."""
        weight = self.weight.to(inputs.device) if self.weight is not None else None
        ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1.0 - pt).pow(self.gamma) * ce_loss

        if self.reduction == "sum":
            return focal_loss.sum()
        if self.reduction == "none":
            return focal_loss
        return focal_loss.mean()


class FlareClassifier(pl.LightningModule):
    """Lightning module for multiclass flare classification."""

    def __init__(
        self,
        num_classes: int,
        class_names: list[str],
        class_weights: torch.Tensor | None = None,
        model_name: str | None = None,
        pretrained: bool | None = None,
        learning_rate: float | None = None,
    ) -> None:
        super().__init__()

        if model_name is None:
            model_name = config.MODEL_NAME
        if pretrained is None:
            pretrained = bool(getattr(config, "PRETRAINED", False))
        if learning_rate is None:
            learning_rate = float(config.LEARNING_RATE)

        self.save_hyperparameters(ignore=["class_weights"])
        self.register_buffer(
            "class_weights",
            class_weights.float() if class_weights is not None else None,
        )

        self.model = timm.create_model(
            self.hparams.model_name,
            pretrained=self.hparams.pretrained,
            num_classes=self.hparams.num_classes,
            in_chans=1,
        )
        ut_t.replace_activations(self.model, torch.nn.ReLU, torch.nn.LeakyReLU, negative_slope=0.01)

        self.loss_fn = self._build_loss_function()

        self.train_metrics = self._build_metric_collection(prefix="train_")
        self.val_metrics = self._build_metric_collection(prefix="val_")
        self.test_metrics = self._build_metric_collection(prefix="test_")

        self._clear_test_buffers()

    def _build_loss_function(self) -> nn.Module:
        """Build configured loss function."""
        loss_type = str(config.LOSS_TYPE).lower()

        if loss_type == "focal":
            return FocalLoss(
                alpha=config.FOCAL_ALPHA,
                gamma=config.FOCAL_GAMMA,
                weight=self.class_weights,
            )

        if loss_type == "weighted_focal":
            focal_weights = None
            if self.class_weights is not None:
                focal_weights = self.class_weights * float(config.FOCAL_ALPHA)
            return FocalLoss(
                alpha=1.0,
                gamma=config.FOCAL_GAMMA,
                weight=focal_weights,
            )

        if loss_type == "weighted_ce":
            return nn.CrossEntropyLoss(weight=self.class_weights)

        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss()

        raise ValueError(
            f"Unsupported LOSS_TYPE='{config.LOSS_TYPE}'. "
            "Expected one of: cross_entropy, weighted_ce, focal, weighted_focal."
        )

    def _build_metric_collection(self, prefix: str) -> MetricCollection:
        """Create per-stage multiclass metric collection."""
        metric_args = {
            "num_classes": int(self.hparams.num_classes),
            "average": "macro",
        }
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(**metric_args),
                "precision": MulticlassPrecision(**metric_args),
                "recall": MulticlassRecall(**metric_args),
                "f1": MulticlassF1Score(**metric_args),
            }
        )
        return metrics.clone(prefix=prefix)

    def _safe_experiment_call(self, fn: str, *args, **kwargs) -> None:
        """Run Comet experiment calls safely without interrupting training."""
        experiment = getattr(self.logger, "experiment", None) if self.logger is not None else None
        method = getattr(experiment, fn, None) if experiment is not None else None
        if method is None:
            return
        try:
            method(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Comet logging failed for %s: %s", fn, exc)

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass and return tensors used across steps."""
        images, labels = batch
        labels = labels.long().reshape(-1)
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        probs = torch.softmax(logits.detach(), dim=1)
        preds = torch.argmax(probs, dim=1)
        return loss, preds, labels, probs, images

    def _clear_test_buffers(self) -> None:
        """Clear test-epoch prediction buffers."""
        self.test_predictions: list[int] = []
        self.test_labels: list[int] = []
        self.test_images: list[np.ndarray] = []
        self.test_image_labels: list[int] = []
        self.test_image_predictions: list[int] = []
        self.test_image_confidences: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning training step."""
        loss, preds, labels, _, _ = self._shared_step(batch)
        self.train_metrics(preds, labels)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning validation step."""
        loss, preds, labels, _, _ = self._shared_step(batch)
        self.val_metrics(preds, labels)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss.detach()

    def on_test_epoch_start(self) -> None:
        """Reset test buffers at test start."""
        self._clear_test_buffers()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning test step."""
        loss, preds, labels, probs, images = self._shared_step(batch)

        self.test_metrics(preds, labels)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.test_predictions.extend(preds.detach().cpu().tolist())
        self.test_labels.extend(labels.detach().cpu().tolist())

        max_examples = max(int(getattr(config, "MAX_MISCLASSIFIED_EXAMPLES", 20)) * 5, 100)
        remaining = max_examples - len(self.test_images)
        if remaining > 0:
            take = min(remaining, images.shape[0])
            selected_images = images[:take].detach().cpu().numpy()
            selected_labels = labels[:take].detach().cpu().tolist()
            selected_preds = preds[:take].detach().cpu().tolist()
            selected_conf = probs[:take].detach().cpu().max(dim=1).values.tolist()

            self.test_images.extend(selected_images)
            self.test_image_labels.extend(selected_labels)
            self.test_image_predictions.extend(selected_preds)
            self.test_image_confidences.extend(selected_conf)

        return loss.detach()

    def on_test_epoch_end(self) -> None:
        """Log final test artifacts and clear buffers."""
        try:
            if not self.test_predictions:
                return

            if not (self.logger and config.ENABLE_COMET_LOGGING):
                return

            y_true = np.asarray(self.test_labels, dtype=np.int64)
            y_pred = np.asarray(self.test_predictions, dtype=np.int64)

            if bool(getattr(config, "LOG_CONFUSION_MATRIX", True)):
                self._log_confusion_matrix(y_true, y_pred)

            if bool(getattr(config, "LOG_CLASSIFICATION_REPORT", True)):
                self._log_classification_report(y_true, y_pred)

            if bool(getattr(config, "LOG_MISCLASSIFIED_EXAMPLES", True)):
                self._log_misclassified_examples(max_examples=int(getattr(config, "MAX_MISCLASSIFIED_EXAMPLES", 20)))
        finally:
            self._clear_test_buffers()

    def _log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Log confusion matrix artifacts to Comet."""
        labels_idx = list(range(int(self.hparams.num_classes)))
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

        self._safe_experiment_call(
            "log_confusion_matrix",
            y_true=y_true,
            y_predicted=y_pred,
            labels=self.hparams.class_names,
            title="Test Set Confusion Matrix",
            file_name="confusion_matrix.json",
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.hparams.class_names,
            yticklabels=self.hparams.class_names,
            cbar_kws={"label": "Count"},
        )
        plt.title("Confusion Matrix - Test Set")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        self._safe_experiment_call("log_figure", "confusion_matrix_heatmap", plt.gcf())
        plt.close()

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums > 0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.hparams.class_names,
            yticklabels=self.hparams.class_names,
            cbar_kws={"label": "Normalized Count"},
        )
        plt.title("Normalized Confusion Matrix - Test Set")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        self._safe_experiment_call("log_figure", "confusion_matrix_normalized", plt.gcf())
        plt.close()

    def _log_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Log per-class and summary classification metrics."""
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=self.hparams.class_names,
            output_dict=True,
            zero_division=0,
        )

        for class_name in self.hparams.class_names:
            class_metrics = report_dict.get(class_name)
            if not isinstance(class_metrics, dict):
                continue
            self._safe_experiment_call("log_metric", f"test_precision_{class_name}", class_metrics["precision"])
            self._safe_experiment_call("log_metric", f"test_recall_{class_name}", class_metrics["recall"])
            self._safe_experiment_call("log_metric", f"test_f1_{class_name}", class_metrics["f1-score"])
            self._safe_experiment_call("log_metric", f"test_support_{class_name}", class_metrics["support"])

        macro_metrics = report_dict.get("macro avg")
        if isinstance(macro_metrics, dict):
            self._safe_experiment_call("log_metric", "test_macro_precision", macro_metrics["precision"])
            self._safe_experiment_call("log_metric", "test_macro_recall", macro_metrics["recall"])
            self._safe_experiment_call("log_metric", "test_macro_f1", macro_metrics["f1-score"])

        weighted_metrics = report_dict.get("weighted avg")
        if isinstance(weighted_metrics, dict):
            self._safe_experiment_call("log_metric", "test_weighted_precision", weighted_metrics["precision"])
            self._safe_experiment_call("log_metric", "test_weighted_recall", weighted_metrics["recall"])
            self._safe_experiment_call("log_metric", "test_weighted_f1", weighted_metrics["f1-score"])

        accuracy = report_dict.get("accuracy")
        if isinstance(accuracy, (float, int)):
            self._safe_experiment_call("log_metric", "test_accuracy", float(accuracy))

        report_str = classification_report(y_true, y_pred, target_names=self.hparams.class_names, zero_division=0)
        self._safe_experiment_call("log_text", report_str, metadata={"name": "classification_report"})

        self._log_classification_report_figure(report_dict)

    def _log_classification_report_figure(self, report_dict: dict[str, object]) -> None:
        """Log a heatmap view of per-class precision/recall/F1."""
        metrics = ["precision", "recall", "f1-score"]
        rows: list[list[float]] = []

        for class_name in self.hparams.class_names:
            class_metrics = report_dict.get(class_name)
            if isinstance(class_metrics, dict):
                rows.append([float(class_metrics.get(metric, 0.0)) for metric in metrics])
            else:
                rows.append([0.0, 0.0, 0.0])

        data = np.asarray(rows, dtype=np.float32)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            data,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            xticklabels=metrics,
            yticklabels=self.hparams.class_names,
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Score"},
        )
        plt.title("Per-Class Classification Metrics")
        plt.xlabel("Metrics")
        plt.ylabel("Classes")
        plt.tight_layout()
        self._safe_experiment_call("log_figure", "classification_report_heatmap", plt.gcf())
        plt.close()

    def _log_misclassified_examples(self, max_examples: int = 20) -> None:
        """Log a figure of misclassified test examples."""
        if not self.test_images:
            return

        mistakes = [
            idx
            for idx, (true_label, pred_label) in enumerate(zip(self.test_image_labels, self.test_image_predictions))
            if true_label != pred_label
        ]
        if not mistakes:
            return

        selected = mistakes[: max(1, int(max_examples))]
        cols = 4
        rows = (len(selected) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = np.array(axes, dtype=object).reshape(rows, cols)

        for plot_idx, data_idx in enumerate(selected):
            row = plot_idx // cols
            col = plot_idx % cols
            image = np.asarray(self.test_images[data_idx]).squeeze()
            true_label = int(self.test_image_labels[data_idx])
            pred_label = int(self.test_image_predictions[data_idx])
            confidence = float(self.test_image_confidences[data_idx])

            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].set_title(
                f"True: {self.hparams.class_names[true_label]}\n"
                f"Pred: {self.hparams.class_names[pred_label]}\n"
                f"Conf: {confidence:.3f}",
                fontsize=10,
            )
            axes[row, col].axis("off")

        for empty_idx in range(len(selected), rows * cols):
            row = empty_idx // cols
            col = empty_idx % cols
            axes[row, col].axis("off")

        plt.tight_layout()
        self._safe_experiment_call("log_figure", "misclassified_examples", fig)
        plt.close(fig)

    def configure_optimizers(self) -> dict[str, object]:
        """Configure optimizer and LR scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": config.CHECKPOINT_METRIC,
            },
        }

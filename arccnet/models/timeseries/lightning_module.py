"""PyTorch Lightning module for timeseries flare forecasting."""

import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from .config import FLARE_CLASS_NAMES, NUM_CLASSES, REGRESSION_TARGETS
from .flare_forecaster import FlareForecaster
from .focal_loss import FocalLoss
from .metrics import compute_binary_skill, compute_regression_metrics, get_operational_class_layout

logger = logging.getLogger(__name__)


class FlareForecasterLightning(pl.LightningModule):
    """PyTorch Lightning module for flare forecasting."""

    def __init__(
        self,
        task_type="multiclass",
        num_channels=10,
        output_dim=None,
        learning_rate=1e-4,
        weight_decay=1e-5,
        class_weights=None,
        flare_class_names=None,
        loss_function="cross_entropy",
        focal_alpha=1.0,
        focal_gamma=2.0,
        tune_threshold_on_val=True,
        threshold_search_points=181,
        **kwargs,
    ):
        """
        Initialize the Lightning module.

        Parameters
        ----------
        task_type : str
            'multiclass' or 'regression'
        num_channels : int
            Number of input channels
        output_dim : int
            Output dimension (num_classes for multiclass, 3 for regression)
        learning_rate : float
            Learning rate for optimizer
        weight_decay : float
            Weight decay for optimizer
        class_weights : list or None
            Class weights for multiclass classification
        loss_function : str
            Loss function to use: 'cross_entropy' or 'focal'
        focal_alpha : float or list
            Alpha parameter for focal loss. Use 1.0 for neutral scaling, or a
            per-class list for class-specific weighting.
        focal_gamma : float
            Gamma parameter for focal loss
        **kwargs
            Additional parameters for FlareForecaster
        """
        super().__init__()
        self.save_hyperparameters()

        self.task_type = task_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_function
        self.tune_threshold_on_val = bool(tune_threshold_on_val)
        self.threshold_search_points = max(3, int(threshold_search_points))
        self.flare_class_names = list(flare_class_names) if flare_class_names is not None else list(FLARE_CLASS_NAMES)
        if output_dim is None:
            output_dim = NUM_CLASSES if task_type == "multiclass" else REGRESSION_TARGETS

        # Create model
        self.model = FlareForecaster(task_type=task_type, num_channels=num_channels, output_dim=output_dim, **kwargs)

        # Loss function
        if task_type == "multiclass":
            if loss_function == "focal":
                # For focal loss, convert class weights to alpha if provided
                if class_weights is not None:
                    weight = torch.tensor(class_weights, dtype=torch.float32)
                else:
                    weight = None
                self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=weight)
            else:  # cross_entropy
                if class_weights is not None:
                    weight = torch.tensor(class_weights, dtype=torch.float32)
                    self.criterion = nn.CrossEntropyLoss(weight=weight)
                else:
                    self.criterion = nn.CrossEntropyLoss()
        else:  # regression
            self.criterion = nn.MSELoss()

        # For collecting predictions
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.register_buffer("m_plus_threshold", torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x, mask=None):
        """Forward pass."""
        return self.model(x, mask=mask)

    def _iter_active_loggers(self):
        """Yield active loggers attached to this module/trainer."""
        try:
            active_loggers = list(getattr(self, "loggers", []) or [])
        except Exception:
            active_loggers = []

        if active_loggers:
            for active_logger in active_loggers:
                if active_logger is not None:
                    yield active_logger
            return

        single_logger = getattr(self, "logger", None)
        if single_logger is None:
            return
        if isinstance(single_logger, (list, tuple)):
            for active_logger in single_logger:
                if active_logger is not None:
                    yield active_logger
            return
        yield single_logger

    def _log_confusion_matrix_to_comet(self, y_true, y_pred, split_name="test"):
        """Log confusion matrix to any attached Comet logger."""
        if self.task_type != "multiclass":
            return

        trainer = getattr(self, "trainer", None)
        if trainer is not None and not bool(getattr(trainer, "is_global_zero", True)):
            return

        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)
        if y_true.size == 0:
            return

        num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)
        labels = list(self.flare_class_names)
        if len(labels) != num_classes:
            labels = [f"class_{idx}" for idx in range(num_classes)]

        for active_logger in self._iter_active_loggers():
            experiment = getattr(active_logger, "experiment", None)
            log_cm = getattr(experiment, "log_confusion_matrix", None) if experiment is not None else None
            if log_cm is None:
                continue
            try:
                log_cm(
                    y_true=y_true.tolist(),
                    y_predicted=y_pred.tolist(),
                    labels=labels,
                    title=f"{split_name.capitalize()} Confusion Matrix",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not log %s confusion matrix to Comet: %s", split_name, exc)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x = batch["x"]
        y = batch["y"]
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(x.device)
        output = self(x, mask=mask)
        loss = self.criterion(output, y)

        # Log training loss
        batch_size = x.size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._shared_eval_step(batch, self.validation_step_outputs)

    @staticmethod
    def _collect_epoch_outputs(step_outputs):
        """Aggregate per-step outputs into epoch-level tensors/arrays."""
        avg_loss = torch.stack([item["loss"] for item in step_outputs]).mean()
        all_preds = torch.cat([item["preds"] for item in step_outputs]).cpu().numpy()
        all_targets = torch.cat([item["targets"] for item in step_outputs]).cpu().numpy()
        return avg_loss, all_preds, all_targets

    def _shared_eval_step(self, batch, output_store):
        """Shared validation/test step logic."""
        x = batch["x"]
        y = batch["y"]
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(x.device)
        output = self(x, mask=mask)
        loss = self.criterion(output, y)
        output_store.append({"loss": loss, "preds": output.detach(), "targets": y.detach()})
        return loss

    def on_validation_epoch_end(self):
        """Compute validation metrics at the end of epoch."""
        if not self.validation_step_outputs:
            return

        avg_loss, all_preds, all_targets = self._collect_epoch_outputs(self.validation_step_outputs)

        # Compute task-specific metrics
        if self.task_type == "multiclass":
            calibrated_threshold = None
            if self.tune_threshold_on_val:
                calibrated_threshold = self._calibrate_m_plus_threshold(all_targets, all_preds)
            if calibrated_threshold is not None:
                self.m_plus_threshold.fill_(float(calibrated_threshold))

            active_threshold = float(self.m_plus_threshold.item())
            metrics = self._compute_multiclass_metrics(
                all_targets,
                all_preds,
                m_plus_threshold=active_threshold,
            )
            metrics["m_plus_threshold"] = float(active_threshold)
            # Fixed baseline retained for comparability with historical runs.
            fixed_metrics = self._compute_multiclass_metrics(
                all_targets,
                all_preds,
                m_plus_threshold=0.5,
            )
            metrics["m_plus_tss_fixed_050"] = fixed_metrics.get("m_plus_tss", 0.0)
            primary_metric = metrics.get("m_plus_tss", metrics["balanced_accuracy"])
        else:  # regression
            metrics = compute_regression_metrics(all_targets, all_preds)
            primary_metric = -metrics["rmse"]  # negative RMSE for maximization

        # Log metrics
        batch_size = len(all_targets)
        self.log("val/loss", avg_loss, prog_bar=True, batch_size=batch_size)
        self.log("val/primary_metric", primary_metric, prog_bar=True, batch_size=batch_size)

        for metric_name, metric_value in metrics.items():
            self.log(f"val/{metric_name}", metric_value, batch_size=batch_size)

        # Clear outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._shared_eval_step(batch, self.test_step_outputs)

    def on_test_epoch_end(self):
        """Compute test metrics at the end of testing."""
        if not self.test_step_outputs:
            return

        avg_loss, all_preds, all_targets = self._collect_epoch_outputs(self.test_step_outputs)

        # Compute task-specific metrics
        if self.task_type == "multiclass":
            active_threshold = float(self.m_plus_threshold.item())
            metrics = self._compute_multiclass_metrics(
                all_targets,
                all_preds,
                m_plus_threshold=active_threshold,
            )
            metrics["m_plus_threshold"] = float(active_threshold)
            fixed_metrics = self._compute_multiclass_metrics(
                all_targets,
                all_preds,
                m_plus_threshold=0.5,
            )
            metrics["m_plus_tss_fixed_050"] = fixed_metrics.get("m_plus_tss", 0.0)
            probs = torch.softmax(torch.from_numpy(np.asarray(all_preds, dtype=np.float32)), dim=1).numpy()
            y_pred_class = np.argmax(probs, axis=1).astype(np.int64)
            self._log_confusion_matrix_to_comet(
                y_true=np.asarray(all_targets, dtype=np.int64),
                y_pred=y_pred_class,
                split_name="test",
            )
        else:  # regression
            metrics = compute_regression_metrics(all_targets, all_preds)

        # Log metrics
        batch_size = len(all_targets)
        self.log("test/loss", avg_loss, batch_size=batch_size)
        for metric_name, metric_value in metrics.items():
            self.log(f"test/{metric_name}", metric_value, batch_size=batch_size)

        # Clear outputs
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/primary_metric",
            },
        }

    def _compute_multiclass_metrics(self, y_true, y_pred, m_plus_threshold=0.5):
        """Compute metrics for multiclass classification."""
        y_true = np.asarray(y_true, dtype=np.int64)
        logits = np.asarray(y_pred, dtype=np.float32)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        y_pred_class = np.argmax(probs, axis=1)

        num_classes = int(probs.shape[1])
        all_classes = list(range(num_classes))
        layout = get_operational_class_layout(num_classes, class_names=self.flare_class_names)

        acc = accuracy_score(y_true, y_pred_class)
        bal_acc = balanced_accuracy_score(y_true, y_pred_class)
        cm = confusion_matrix(y_true, y_pred_class, labels=all_classes)

        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

        metrics = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
        }
        for idx in range(num_classes):
            metrics[f"class_{idx}_acc"] = float(per_class_acc[idx]) if idx < len(per_class_acc) else 0.0

        m_plus_indices = layout["m_plus_indices"]
        m_plus_true = np.isin(y_true, m_plus_indices).astype(int)
        m_plus_score = probs[:, m_plus_indices].sum(axis=1)

        for prefix, y_bin, y_score in [("m_plus", m_plus_true, m_plus_score)]:
            skill = compute_binary_skill(y_bin, y_score, threshold=float(m_plus_threshold))
            metrics[f"{prefix}_tss"] = skill["tss"]
            metrics[f"{prefix}_hss"] = skill["hss"]
            metrics[f"{prefix}_tpr"] = skill["tpr"]
            metrics[f"{prefix}_fpr"] = skill["fpr"]
            metrics[f"{prefix}_pr_auc"] = skill["pr_auc"] if skill["pr_auc"] is not None else 0.0
            metrics[f"{prefix}_roc_auc"] = skill["roc_auc"] if skill["roc_auc"] is not None else 0.0

        x_index = layout["x_index"]
        if x_index is not None:
            x_plus_true = (y_true == x_index).astype(int)
            x_plus_score = probs[:, x_index]
            skill = compute_binary_skill(x_plus_true, x_plus_score, threshold=0.5)
            metrics["x_plus_tss"] = skill["tss"]
            metrics["x_plus_hss"] = skill["hss"]
            metrics["x_plus_tpr"] = skill["tpr"]
            metrics["x_plus_fpr"] = skill["fpr"]
            metrics["x_plus_pr_auc"] = skill["pr_auc"] if skill["pr_auc"] is not None else 0.0
            metrics["x_plus_roc_auc"] = skill["roc_auc"] if skill["roc_auc"] is not None else 0.0

        return metrics

    def _calibrate_m_plus_threshold(self, y_true, y_pred):
        """
        Select the M+ operating threshold that maximizes validation TSS.
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        logits = np.asarray(y_pred, dtype=np.float32)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

        layout = get_operational_class_layout(int(probs.shape[1]), class_names=self.flare_class_names)
        m_plus_indices = layout["m_plus_indices"]
        m_plus_true = np.isin(y_true, m_plus_indices).astype(int)
        m_plus_score = probs[:, m_plus_indices].sum(axis=1)

        best_threshold = self._find_optimal_binary_threshold(m_plus_true, m_plus_score)
        return best_threshold

    def _find_optimal_binary_threshold(self, y_true, y_score):
        """
        Find threshold maximizing TSS over a fixed search grid.
        """
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score, dtype=np.float32)

        # Cannot calibrate threshold without both classes present.
        if len(np.unique(y_true)) < 2:
            return 0.5

        thresholds = np.linspace(0.05, 0.95, self.threshold_search_points)
        best_threshold = 0.5
        best_tss = -np.inf
        best_tpr = -np.inf
        best_fpr = np.inf

        for threshold in thresholds:
            skill = compute_binary_skill(y_true, y_score, threshold=float(threshold))
            tss = skill["tss"]
            tpr = skill["tpr"]
            fpr = skill["fpr"]

            # Tie-breakers: higher TPR, then lower FPR.
            if (tss > best_tss) or (
                np.isclose(tss, best_tss) and (tpr > best_tpr or (np.isclose(tpr, best_tpr) and fpr < best_fpr))
            ):
                best_tss = tss
                best_tpr = tpr
                best_fpr = fpr
                best_threshold = float(threshold)

        return float(best_threshold)

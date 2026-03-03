"""Binary flare classification Lightning model."""

import logging

import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from torchvision.ops import sigmoid_focal_loss

from arccnet.models import train_utils as ut_t

logger = logging.getLogger(__name__)


class FlareClassifier(pl.LightningModule):
    """Lightning module wrapping a timm image classifier for binary flares."""

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        num_classes: int = 1,
        in_chans: int = 1,
        pretrained: bool = False,
        learning_rate: float = 1e-4,
        loss_function: str = "bce",
        pos_weight: torch.Tensor = None,
        auto_compute_pos_weight: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        decision_threshold: float = 0.5,
    ):
        """Initialize model, loss, and metrics for binary classification."""
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.auto_compute_pos_weight = bool(auto_compute_pos_weight)
        self.decision_threshold = float(decision_threshold)
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be between 0 and 1.")

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.register_buffer("pos_weight", torch.tensor([1.0], dtype=torch.float32))

        if loss_function not in {"bce", "focal", "weighted_bce"}:
            raise ValueError(f"Unknown loss function: {loss_function}. Options are: 'bce', 'focal', 'weighted_bce'")

        self.model = timm.create_model(
            self.hparams.model_name,
            pretrained=self.hparams.pretrained,
            num_classes=self.hparams.num_classes,
            in_chans=self.hparams.in_chans,
        )
        ut_t.replace_activations(self.model, torch.nn.ReLU, torch.nn.LeakyReLU, negative_slope=0.01)

        self.train_metrics = self._build_metric_collection(prefix="train_", include_ranking_metrics=False)
        self.val_metrics = self._build_metric_collection(prefix="val_", include_ranking_metrics=True)
        self.test_metrics = self._build_metric_collection(prefix="test_", include_ranking_metrics=True)

        self.val_confusion_matrix = self._build_confusion_matrix()
        self.test_confusion_matrix = self._build_confusion_matrix()

    def _build_metric_collection(self, prefix: str, include_ranking_metrics: bool) -> MetricCollection:
        metrics_dict = {
            "acc": BinaryAccuracy(threshold=self.decision_threshold),
            "precision": BinaryPrecision(threshold=self.decision_threshold),
            "recall": BinaryRecall(threshold=self.decision_threshold),
            "f1": BinaryF1Score(threshold=self.decision_threshold),
        }
        if include_ranking_metrics:
            metrics_dict["auroc"] = BinaryAUROC()
            metrics_dict["avg_precision"] = BinaryAveragePrecision()

        metrics = MetricCollection(metrics_dict)
        return metrics.clone(prefix=prefix)

    def _build_confusion_matrix(self) -> BinaryConfusionMatrix:
        return BinaryConfusionMatrix(threshold=self.decision_threshold)

    def set_decision_threshold(self, threshold: float) -> None:
        """Update decision threshold and rebuild validation/test metrics."""
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("decision_threshold must be between 0 and 1.")

        self.decision_threshold = threshold
        self.hparams.decision_threshold = threshold

        self.train_metrics = self._build_metric_collection(prefix="train_", include_ranking_metrics=False).to(
            self.device
        )
        self.val_metrics = self._build_metric_collection(prefix="val_", include_ranking_metrics=True).to(self.device)
        self.test_metrics = self._build_metric_collection(prefix="test_", include_ranking_metrics=True).to(self.device)
        self.val_confusion_matrix = self._build_confusion_matrix().to(self.device)
        self.test_confusion_matrix = self._build_confusion_matrix().to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss based on the configured loss function."""
        logits = logits.reshape(-1)
        targets = targets.float().reshape(-1)

        if self.loss_function == "focal":
            return sigmoid_focal_loss(
                logits,
                targets,
                alpha=self.hparams.focal_alpha,
                gamma=self.hparams.focal_gamma,
                reduction="mean",
            )
        elif self.loss_function == "weighted_bce":
            return F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weight,
            )
        elif self.loss_function == "bce":
            return F.binary_cross_entropy_with_logits(
                logits,
                targets,
            )
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")

    def _shared_eval_tensors(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert model outputs and labels to detached 1D tensors for metrics."""
        probs = torch.sigmoid(logits.reshape(-1).detach()).to(dtype=torch.float32)
        labels = targets.detach().to(dtype=torch.int64).reshape(-1)
        return probs, labels

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute training step."""
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        probs, targets = self._shared_eval_tensors(logits, y)
        self.train_metrics(probs, targets)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute validation step."""
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        probs, targets = self._shared_eval_tensors(logits, y)
        self.val_metrics(probs, targets)
        self.val_confusion_matrix(probs, targets)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss.detach(), prog_bar=True, on_epoch=True, sync_dist=True)
        return loss.detach()

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute test step."""
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        probs, targets = self._shared_eval_tensors(logits, y)
        self.test_metrics(probs, targets)
        self.test_confusion_matrix(probs, targets)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_loss", loss.detach(), prog_bar=True, on_epoch=True, sync_dist=True)
        return loss.detach()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def on_fit_start(self):
        """Optionally derive weighted-BCE pos_weight from the prepared training split."""
        if self.loss_function != "weighted_bce" or not self.auto_compute_pos_weight:
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or getattr(datamodule, "train_df", None) is None:
            raise RuntimeError("Cannot auto-compute pos_weight: trainer.datamodule.train_df is not available.")

        target_column = getattr(datamodule, "target_column", None)
        if target_column is None:
            raise RuntimeError("Cannot auto-compute pos_weight: datamodule.target_column is missing.")

        train_df = datamodule.train_df
        positives = int(train_df[target_column].sum())
        negatives = len(train_df) - positives
        if positives <= 0:
            raise ValueError(f"No positive samples found in training split for target '{target_column}'.")
        if negatives <= 0:
            raise ValueError(f"No negative samples found in training split for target '{target_column}'.")

        new_pos_weight = torch.tensor(
            [negatives / positives],
            dtype=self.pos_weight.dtype,
            device=self.pos_weight.device,
        )
        self.pos_weight.copy_(new_pos_weight)

    def on_train_epoch_end(self):
        """No-op training hook."""
        pass

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        cm = self.val_confusion_matrix.compute().to(dtype=torch.float32)
        self.log("val_tss", self._compute_tss_from_cm(cm), prog_bar=False, sync_dist=True)
        self.val_confusion_matrix.reset()

    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        cm = self.test_confusion_matrix.compute().to(dtype=torch.float32)
        self.log("test_tss", self._compute_tss_from_cm(cm), prog_bar=True, sync_dist=True)
        experiment = getattr(self.logger, "experiment", None) if self.logger is not None else None
        if experiment is not None and hasattr(experiment, "log_confusion_matrix"):
            try:
                experiment.log_confusion_matrix(
                    matrix=cm.cpu().numpy(),
                    labels=["No Flare", "Flare"],
                    title="Test Set Confusion Matrix",
                    row_label="Actual",
                    column_label="Predicted",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Comet confusion-matrix logging failed: %s", exc)
        self.test_confusion_matrix.reset()

    @staticmethod
    def _compute_tss_from_cm(cm: torch.Tensor) -> torch.Tensor:
        """Compute True Skill Statistic (TSS) from a 2x2 confusion matrix."""
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        tpr_denom = tp + fn
        fpr_denom = fp + tn
        tpr = torch.where(tpr_denom > 0, tp / tpr_denom, torch.zeros_like(tp))
        fpr = torch.where(fpr_denom > 0, fp / fpr_denom, torch.zeros_like(fp))
        return tpr - fpr

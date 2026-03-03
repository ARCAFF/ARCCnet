"""
Checkpoint management utilities for saving model checkpoints with metadata and logging.

Handles:
- Creating hierarchical directory structure in data folder
- Saving checkpoints with descriptive names (date, time, model, metadata)
- Saving classification reports and training summaries
- Storing configuration used for training
"""

import json
from typing import Any
from pathlib import Path
from datetime import datetime

from pytorch_lightning.callbacks import ModelCheckpoint


class CheckpointManager:
    """Manages checkpoint saving with metadata and logging."""

    def __init__(
        self,
        root_name: str,
        data_folder: str = "/ARCAFF/data",
        model_name: str = "unknown",
        loss_function: str = "unknown",
    ):
        """
        Initialize CheckpointManager.

        Parameters:
        -----------
        root_name : str
            Hierarchical root name for checkpoints (e.g., "flares/binary_classification",
            "flares/multiclass", "cutouts/hale", etc.)
        data_folder : str
            Base data folder where checkpoints will be saved (checkpoints go under
            {data_folder}/checkpoints/{root_name})
        model_name : str
            Name of the model being trained
        loss_function : str
            Loss function used for training
        """
        self.root_name = root_name
        self.data_folder = data_folder
        self.model_name = model_name
        self.loss_function = loss_function
        self.timestamp = datetime.now()

        # Create checkpoint directory structure
        self.checkpoint_dir = self._create_checkpoint_directory()

        # Store training metadata
        self.training_metadata: dict[str, Any] = {
            "start_time": self.timestamp.isoformat(),
            "model_name": model_name,
            "loss_function": loss_function,
            "root_name": root_name,
        }

    def _create_checkpoint_directory(self) -> Path:
        """
        Create checkpoint directory with timestamp and metadata.

        Directory structure:
        /ARCAFF/data/checkpoints/flares/binary_classification/2025-02-12_14-30-45_resnet18_weighted_bce/

        Returns:
        --------
        Path
            Path to the created checkpoint directory
        """
        # Format: YYYY-MM-DD_HH-MM-SS
        timestamp_str = self.timestamp.strftime("%Y-%m-%d_%H-%M-%S")

        # Build the checkpoint directory name with metadata
        dir_name = f"{timestamp_str}_{self.model_name}_{self.loss_function}"
        checkpoint_path = Path(self.data_folder) / "checkpoints" / self.root_name / dir_name

        # Create directory
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        return checkpoint_path

    def get_checkpoint_callback(self, monitor: str = "val_f1", mode: str = "max") -> ModelCheckpoint:
        """
        Create a PyTorch Lightning ModelCheckpoint callback.

        Parameters:
        -----------
        monitor : str
            Metric to monitor for checkpointing
        mode : str
            "max" or "min" for metric monitoring

        Returns:
        --------
        ModelCheckpoint
            Configured checkpoint callback
        """
        return ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename=f"best-{{epoch:02d}}-{{{monitor}:.3f}}",
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            auto_insert_metric_name=False,
        )

    def save_training_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Save training metadata to JSON.

        Parameters:
        -----------
        metadata : Dict[str, Any]
            Dictionary containing training metadata (epochs, final metrics, etc.)
        """
        # Merge with existing metadata
        full_metadata = {**self.training_metadata, **metadata}
        full_metadata["end_time"] = datetime.now().isoformat()

        metadata_path = self.checkpoint_dir / "training_summary.json"
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2, default=str)

    def save_classification_report(self, report: dict[str, Any]) -> None:
        """
        Save classification report to JSON.

        Parameters:
        -----------
        report : Dict[str, Any]
            Classification report containing metrics per class, overall metrics, etc.
        """
        report_path = self.checkpoint_dir / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def save_threshold_tuning_report(self, report: dict[str, Any]) -> None:
        """
        Save threshold tuning report to JSON.

        Parameters:
        -----------
        report : Dict[str, Any]
            Threshold tuning data including selected threshold and validation metrics.
        """
        report_path = self.checkpoint_dir / "threshold_tuning.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def save_config(self, config_vars: dict[str, Any]) -> None:
        """
        Save configuration used for training.

        Parameters:
        -----------
        config_vars : Dict[str, Any]
            Configuration dictionary (can be from config module)
        """
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_vars, f, indent=2, default=str)

    def save_minimal_logging(
        self,
        best_epoch: int | None = None,
        best_metric_value: float | None = None,
        best_metric_name: str | None = None,
        num_epochs_trained: int | None = None,
        early_stopping_triggered: bool = False,
        additional_metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Save minimal logging data for quick reference.

        This complements the full Comet logging by providing a local snapshot
        of key training information.

        Parameters:
        -----------
        best_epoch : int, optional
            Epoch at which best model was found
        best_metric_value : float, optional
            Value of the best metric
        best_metric_name : str, optional
            Name of the monitored metric
        num_epochs_trained : int, optional
            Total number of epochs trained
        early_stopping_triggered : bool
            Whether early stopping was triggered
        additional_metrics : Dict[str, Any], optional
            Any additional metrics to log
        """
        logging_data = {
            "timestamp": datetime.now().isoformat(),
            "best_epoch": best_epoch,
            f"best_{best_metric_name}": best_metric_value if best_metric_name else None,
            "num_epochs_trained": num_epochs_trained,
            "early_stopping_triggered": early_stopping_triggered,
        }

        if additional_metrics:
            logging_data.update(additional_metrics)

        logging_path = self.checkpoint_dir / "minimal_logging.json"
        with open(logging_path, "w") as f:
            json.dump(logging_data, f, indent=2, default=str)

    def get_checkpoint_path(self) -> Path:
        """Get the path to the checkpoint directory."""
        return self.checkpoint_dir

    def __str__(self) -> str:
        """String representation of checkpoint manager."""
        return f"CheckpointManager(root={self.root_name}, dir={self.checkpoint_dir})"


class BinaryClassificationCheckpointManager(CheckpointManager):
    """
    Specialized checkpoint manager for binary classification flare models.

    Default root name: flares/binary_classification
    """

    def __init__(
        self,
        data_folder: str = "/ARCAFF/data",
        model_name: str = "resnet18",
        loss_function: str = "weighted_bce",
    ):
        super().__init__(
            root_name="flares/binary_classification",
            data_folder=data_folder,
            model_name=model_name,
            loss_function=loss_function,
        )


class MulticlassFlareCheckpointManager(CheckpointManager):
    """
    Specialized checkpoint manager for multiclass flare models.

    Default root name: flares/multiclass
    """

    def __init__(
        self,
        data_folder: str = "/ARCAFF/data",
        model_name: str = "vit_base_patch32_224",
        loss_function: str = "focal",
    ):
        super().__init__(
            root_name="flares/multiclass",
            data_folder=data_folder,
            model_name=model_name,
            loss_function=loss_function,
        )


class CutoutHaleCheckpointManager(CheckpointManager):
    """
    Specialized checkpoint manager for Hale classification cutout models.

    Default root name: cutouts/hale
    """

    def __init__(
        self,
        data_folder: str = "/ARCAFF/data",
        model_name: str = "unknown",
        loss_function: str = "cross_entropy",
    ):
        super().__init__(
            root_name="cutouts/hale",
            data_folder=data_folder,
            model_name=model_name,
            loss_function=loss_function,
        )


class CutoutMcintoshCheckpointManager(CheckpointManager):
    """
    Specialized checkpoint manager for McIntosh classification cutout models.

    Default root name: cutouts/mcintosh
    """

    def __init__(
        self,
        data_folder: str = "/ARCAFF/data",
        model_name: str = "unknown",
        loss_function: str = "cross_entropy",
    ):
        super().__init__(
            root_name="cutouts/mcintosh",
            data_folder=data_folder,
            model_name=model_name,
            loss_function=loss_function,
        )

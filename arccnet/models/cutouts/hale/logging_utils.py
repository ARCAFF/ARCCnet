"""
Logging utilities for Hale classification models.

This module contains logging setup, custom loggers, and logging helper functions.
"""

import shutil
import logging
from typing import Any
from pathlib import Path

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.loggers.comet import CometLogger

import arccnet.models.cutouts.hale.config as config


def log_hyperparameters(logger, params: dict[str, Any]) -> None:
    """
    Log hyperparameters to a PyTorch Lightning logger with graceful fallbacks.

    Args:
        logger: Lightning logger instance
        params: Dictionary of hyperparameters to log
    """
    if not params:
        return

    if isinstance(logger, TensorBoardLogger):
        _log_tensorboard_hyperparams(logger, params)
        return

    if hasattr(logger, "log_hyperparams"):
        try:
            logger.log_hyperparams(params)
            logging.info(f"Hyperparameters logged to {type(logger).__name__}")
        except Exception as e:
            logging.warning(f"Could not log hyperparameters to {type(logger).__name__}: {e}")
        return

    if hasattr(logger, "experiment") and hasattr(logger.experiment, "log_parameters"):
        try:
            logger.experiment.log_parameters(params)
            logging.info(f"Hyperparameters logged via experiment API for {type(logger).__name__}")
        except Exception as e:
            logging.warning(f"Could not log hyperparameters via experiment API for {type(logger).__name__}: {e}")


def _log_tensorboard_hyperparams(logger: TensorBoardLogger, params: dict[str, Any]) -> None:
    """Handle TensorBoard hyperparameter logging with NumPy 2.0 fallback."""
    if not getattr(config, "TENSORBOARD_LOG_HYPERPARAMS", True):
        logging.info("TensorBoard hyperparameter logging disabled in config")
        return

    try:
        logger.log_hyperparams(params=params)
        logging.info("TensorBoard hyperparameters logged successfully")
    except (AttributeError, ImportError) as e:
        if "np.string_" in str(e) or "numpy" in str(e).lower():
            logging.warning(
                "TensorBoard hyperparameter logging failed due to NumPy 2.0 compatibility. "
                "Logging hyperparameters as text instead."
            )
            _log_hyperparams_as_text(logger, params)
        else:
            raise e
    except Exception as e:
        logging.warning(f"TensorBoard hyperparameter logging failed: {e}")
        _log_hyperparams_as_text(logger, params)


def _log_hyperparams_as_text(logger: TensorBoardLogger, params: dict[str, Any]) -> None:
    """Log hyperparameters as text summary instead of structured hyperparameters."""
    if not params or not hasattr(logger, "experiment") or not hasattr(logger.experiment, "add_text"):
        return

    hparam_text = "Hyperparameters:\n"
    for key, value in params.items():
        hparam_text += f"- {key}: {value}\n"

    try:
        logger.experiment.add_text("Hyperparameters", hparam_text, global_step=0)
        logging.info("Hyperparameters logged as text to TensorBoard")
    except Exception as text_error:
        logging.warning(f"Could not log hyperparameters as text: {text_error}")


def setup_loggers(experiment_name: str, fold_num: int | None = None) -> list:
    """
    Set up logging based on configuration.

    Args:
        experiment_name: Name for the experiment
        fold_num: Fold number (if applicable)

    Returns:
        List of configured loggers
    """
    loggers = []

    # Add fold number to experiment name if provided
    if fold_num is not None:
        experiment_name = f"{experiment_name}_fold_{fold_num}"

    # Add loggers based on config

    log_dir = getattr(config, "LOG_DIR", "logs")

    if getattr(config, "ENABLE_TENSORBOARD", True):
        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name=experiment_name,
            version=None,  # Auto-increment version
        )
        loggers.append(tb_logger)

    if getattr(config, "ENABLE_CSV", True):
        csv_logger = CSVLogger(
            save_dir=log_dir,
            name=experiment_name,
        )
        loggers.append(csv_logger)

    if getattr(config, "ENABLE_COMET", False):
        comet_logger = _setup_comet_logger(experiment_name)
        if comet_logger:
            loggers.append(comet_logger)

    # Use at least CSV logger if no loggers are enabled
    if not loggers:
        csv_logger = CSVLogger(
            save_dir=log_dir,
            name=experiment_name,
        )
        loggers.append(csv_logger)
        logging.warning("No loggers enabled in config. Using CSV logger as fallback.")

    return loggers


def _setup_comet_logger(experiment_name: str) -> CometLogger | None:
    """Set up Comet ML logger if credentials are available."""
    project_name = getattr(config, "COMET_PROJECT_NAME", getattr(config, "PROJECT_NAME", ""))
    common_kwargs = {
        "name": f"{experiment_name}_rs{config.RANDOM_STATE}",
        "offline_directory": getattr(config, "LOG_DIR", "logs"),
    }
    workspace = getattr(config, "COMET_WORKSPACE", None)
    if workspace:
        common_kwargs["workspace"] = workspace

    try:
        return CometLogger(project=project_name, **common_kwargs)
    except TypeError:
        try:
            return CometLogger(project_name=project_name, **common_kwargs)
        except Exception as e:
            logging.warning(f"Could not initialize Comet ML logger: {e}")
            return None
    except Exception as e:
        logging.warning(f"Could not initialize Comet ML logger: {e}")
        return None


def log_hyperparameters_to_comet(comet_logger: CometLogger, fold_num: int | None = None) -> None:
    """
    Log hyperparameters to Comet ML.

    Args:
        comet_logger: Comet ML logger instance
        fold_num: Fold number (if applicable)
    """
    hyperparams = {
        "model_name": config.MODEL_NAME,
        "classes": config.classes,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "max_epochs": config.MAX_EPOCHS,
        "n_folds": config.N_FOLDS,
        "random_state": config.RANDOM_STATE,
        "data_type": config.DATA_TYPE,
        "dropout_rate": getattr(config, "DROPOUT_RATE", 0.0),
        "weight_decay": getattr(config, "WEIGHT_DECAY", 0.0),
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "use_augmentation": getattr(config, "USE_AUGMENTATION", False),
        "num_classes": config.NUM_CLASSES,
    }

    if fold_num is not None:
        hyperparams["fold_num"] = fold_num

    try:
        comet_logger.log_hyperparams(hyperparams)
        logging.info("Hyperparameters logged to Comet ML")
    except Exception as e:
        logging.warning(f"Could not log hyperparameters to Comet ML: {e}")


def log_final_metrics(
    loggers: list, test_metrics: dict[str, float], best_model_path: str | None = None, fold_num: int | None = None
) -> None:
    """
    Log final metrics to all configured loggers.

    Args:
        loggers: List of experiment loggers
        test_metrics: Dictionary of test metrics
        best_model_path: Path to the best model checkpoint
        fold_num: Fold number (if applicable)
    """
    final_metrics = {
        "final_test_loss": test_metrics.get("test_loss", 0),
        "final_test_accuracy": test_metrics.get("test_acc", 0),
        "final_test_f1": test_metrics.get("test_f1", 0),
    }

    best_checkpoint = Path(best_model_path) if best_model_path else None
    for logger in loggers:
        try:
            metrics_logged = False

            if hasattr(logger, "log_metrics"):
                logger.log_metrics(final_metrics)
                metrics_logged = True

            experiment = getattr(logger, "experiment", None)
            if experiment:
                if not metrics_logged and hasattr(experiment, "log_metric"):
                    for metric_name, metric_value in final_metrics.items():
                        experiment.log_metric(metric_name, metric_value)

                if best_checkpoint and best_checkpoint.exists() and fold_num is not None:
                    if hasattr(experiment, "log_model"):
                        experiment.log_model(
                            name=f"hale_model_fold_{fold_num}",
                            file_or_folder=str(best_checkpoint),
                        )
                        logging.info(f"Model artifact logged to Comet for fold {fold_num}")

            if isinstance(logger, TensorBoardLogger) and best_checkpoint and best_checkpoint.exists():
                _backup_checkpoint_to_tensorboard(logger, best_checkpoint, fold_num)

        except Exception as e:
            logging.warning(f"Could not log final metrics to {type(logger).__name__}: {e}")


def log_experiment_summary(loggers: list, summary_metrics: dict[str, float]) -> None:
    """
    Log experiment summary metrics to all configured loggers.

    Args:
        loggers: List of experiment loggers
        summary_metrics: Dictionary of summary metrics
    """
    for logger in loggers:
        try:
            if hasattr(logger, "log_metrics"):
                logger.log_metrics(summary_metrics)
            elif hasattr(logger, "experiment") and hasattr(logger.experiment, "log_metric"):
                # For Comet ML
                for metric_name, metric_value in summary_metrics.items():
                    logger.experiment.log_metric(metric_name, metric_value)

            logging.info(f"Summary metrics logged to {type(logger).__name__}")

        except Exception as e:
            logging.warning(f"Could not log summary to {type(logger).__name__}: {e}")


def setup_basic_logging() -> None:
    """Set up basic logging configuration."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _backup_checkpoint_to_tensorboard(
    tb_logger: TensorBoardLogger, checkpoint_path: Path, fold_num: int | None
) -> None:
    """Copy the best checkpoint into the TensorBoard log directory for local access."""
    try:
        log_dir = Path(tb_logger.log_dir)
        if fold_num is not None:
            backup_dir = log_dir / "checkpoints" / f"fold_{fold_num}"
        else:
            backup_dir = log_dir / "checkpoints"

        backup_dir.mkdir(parents=True, exist_ok=True)
        target_path = backup_dir / checkpoint_path.name
        shutil.copy2(checkpoint_path, target_path)
        logging.info(f"Checkpoint copied to TensorBoard log dir: {target_path}")

        experiment = getattr(tb_logger, "experiment", None)
        if experiment and hasattr(experiment, "add_text"):
            step = fold_num if fold_num is not None else 0
            experiment.add_text(
                tag=f"fold_{fold_num}/checkpoint_path" if fold_num is not None else "checkpoint_path",
                text_string=str(target_path),
                global_step=step,
            )
    except Exception as e:
        logging.warning(f"Could not back up checkpoint to TensorBoard directory: {e}")


def log_dataset_statistics(
    train_size: int,
    val_size: int,
    test_size: int,
    total_size: int,
    class_distribution: dict[str, int],
    class_weights: list[float] | None = None,
) -> None:
    """
    Log dataset statistics to console.

    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        total_size: Total number of samples
        class_distribution: Dictionary of class distribution
        class_weights: List of class weights (optional)
    """
    logging.info("=" * 50)
    logging.info("DATASET STATISTICS")
    logging.info("=" * 50)
    logging.info(f"Train samples: {train_size}")
    logging.info(f"Val samples: {val_size}")
    logging.info(f"Test samples: {test_size}")
    logging.info(f"Total samples: {total_size}")

    logging.info("Train class distribution:")
    for class_name, count in class_distribution.items():
        logging.info(f"  {class_name}: {count} samples")

    if class_weights:
        logging.info(f"Class weights: {class_weights}")

    logging.info("=" * 50)


def log_training_info(
    max_epochs: int, learning_rate: float, batch_size: int, data_type: str, precision: str, device_info: str
) -> None:
    """
    Log training configuration information.

    Args:
        max_epochs: Maximum number of epochs
        learning_rate: Learning rate
        batch_size: Batch size
        data_type: Data type description
        precision: Training precision
        device_info: Device information
    """
    logging.info("Starting training...")
    logging.info(f"Max epochs: {max_epochs}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Data type: {data_type}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Device: {device_info}")


def log_test_results(test_metrics: dict[str, float], best_model_path: str) -> None:
    """
    Log test results to console.

    Args:
        test_metrics: Dictionary of test metrics
        best_model_path: Path to the best model checkpoint
    """
    logging.info("=" * 50)
    logging.info("FINAL TEST RESULTS")
    logging.info("=" * 50)
    test_loss = test_metrics.get("test_loss")
    test_acc = test_metrics.get("test_acc")
    test_f1 = test_metrics.get("test_f1")
    loss_str = f"{float(test_loss):.4f}" if isinstance(test_loss, int | float) else "N/A"
    acc_str = f"{float(test_acc):.4f}" if isinstance(test_acc, int | float) else "N/A"
    f1_str = f"{float(test_f1):.4f}" if isinstance(test_f1, int | float) else "N/A"
    logging.info(f"Test Loss: {loss_str}")
    logging.info(f"Test Accuracy: {acc_str}")
    logging.info(f"Test F1: {f1_str}")
    logging.info(f"Best model checkpoint: {best_model_path}")
    logging.info("=" * 50)

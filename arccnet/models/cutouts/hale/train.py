"""
Training script for Hale classification with cross-validation support.
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime

import arccnet.models.cutouts.hale.config as config
from arccnet.models.cutouts.hale.cross_validation import CrossValidationManager
from arccnet.models.cutouts.hale.trainer import HaleTrainer


def log_section(title: str, width: int = 50) -> None:
    """Log a banner-style section header."""
    logging.info("=" * width)
    logging.info(title)
    logging.info("=" * width)


def log_metric_summary(label: str, stats: dict) -> None:
    """Log mean and std stats for a metric dict."""
    mean = stats.get("mean", 0)
    std = stats.get("std", 0)
    logging.info(f"{label}: {mean:.4f} ± {std:.4f}")


def _format_metric_value(metrics: dict, key: str) -> str:
    """Format a metric value safely for logging output."""
    value = metrics.get(key)
    return f"{float(value):.4f}" if isinstance(value, int | float) else "N/A"


def train_single_fold_mode(df) -> None:
    """
    Train on a single fold for quick testing.

    Args:
        df: Processed dataset
    """
    logging.info("Starting training on fold 1 (single fold mode)...")

    trainer = HaleTrainer()
    trainer_obj, _, test_results = trainer.train_single_fold(df, fold_num=1)

    logging.info("Single fold training completed!")

    # Log final results
    if test_results:
        test_metrics = test_results[0]
        log_section("SINGLE FOLD TRAINING COMPLETED")
        logging.info(f"Final Test Accuracy: {_format_metric_value(test_metrics, 'test_acc')}")
        logging.info(f"Final Test F1: {_format_metric_value(test_metrics, 'test_f1')}")
        logging.info(f"Final Test Loss: {_format_metric_value(test_metrics, 'test_loss')}")
        logging.info(f"Best checkpoint: {trainer_obj.checkpoint_callback.best_model_path}")
        logging.info("=" * 50)


def train_cross_validation_mode(df) -> None:
    """
    Train on all folds for cross-validation.

    Args:
        df: Processed dataset
    """
    logging.info("Starting cross-validation training on all folds...")

    cv_manager = CrossValidationManager()
    _, summary = cv_manager.run_cross_validation(df)

    logging.info("Cross-validation training completed!")

    # Log final summary
    if summary and "metrics_summary" in summary:
        metrics = summary["metrics_summary"]
        log_section("CROSS-VALIDATION COMPLETED")

        metric_labels = (
            ("Mean Test Accuracy", "test_accuracy"),
            ("Mean Test F1", "test_f1"),
            ("Mean Test Loss", "test_loss"),
        )
        for label, key in metric_labels:
            if key in metrics:
                log_metric_summary(label, metrics[key])

        exp_info = summary.get("experiment_info", {})
        total_time = exp_info.get("total_training_time", 0)
        logging.info(f"Total Training Time: {total_time:.1f}s")
        logging.info("=" * 50)


def run_training(config_module=config, args: argparse.Namespace | None = None) -> None:
    """
    Backward-compatible entry point for package CLI.

    Applies CLI argument overrides to config, then runs training.
    """
    global config
    config = config_module

    if args is not None:
        overrides = {
            "model_name": "MODEL_NAME",
            "batch_size": "BATCH_SIZE",
            "num_workers": "NUM_WORKERS",
            "num_epochs": "MAX_EPOCHS",
            "patience": "EARLY_STOPPING_PATIENCE",
            "learning_rate": "LEARNING_RATE",
            "data_folder": "DATA_FOLDER",
            "dataset_folder": "DATASET_FOLDER",
            "df_file_name": "DF_FILE_NAME",
        }
        for arg_name, config_name in overrides.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                setattr(config, config_name, value)

        gpu_index = getattr(args, "gpu_index", None)
        if gpu_index is not None:
            config.ACCELERATOR = "gpu"
            config.DEVICES = [int(gpu_index)]

    main()


def main() -> None:
    """
    Main execution function.

    Prepares dataset and runs training.
    """
    log_dir = Path(getattr(config, "LOG_DIR", Path(__file__).parent / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"hale_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )

    # Log configuration info
    log_section("HALE CLASSIFICATION TRAINING", width=60)
    logging.info(f"Model: {config.MODEL_NAME}")
    logging.info(f"Classes: {config.classes}")
    logging.info(f"Number of folds: {config.N_FOLDS}")
    logging.info(f"Batch size: {config.BATCH_SIZE}")
    logging.info(f"Learning rate: {config.LEARNING_RATE}")
    logging.info(f"Max epochs: {config.MAX_EPOCHS}")
    logging.info(f"Train all folds: {config.TRAIN_ALL_FOLDS}")
    logging.info("=" * 60)

    try:
        # Prepare dataset once at runtime
        logging.info("Preparing dataset...")
        df_processed = HaleTrainer().prepare_dataset_once()
        logging.info(f"Dataset prepared successfully. Shape: {df_processed.shape}")

        # Run training based on configuration
        if config.TRAIN_ALL_FOLDS:
            train_cross_validation_mode(df_processed)
        else:
            train_single_fold_mode(df_processed)

    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise

    logging.info("Training script completed successfully!")


if __name__ == "__main__":
    main()

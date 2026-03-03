"""
Cross-validation manager for Hale classification models.

This module contains the CrossValidationManager class that handles
cross-validation workflows, result aggregation, and summary generation.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import arccnet.models.cutouts.hale.config as config
from arccnet.models.cutouts.hale.logging_utils import (
    log_experiment_summary,
    log_hyperparameters,
    setup_basic_logging,
    setup_loggers,
)
from arccnet.models.cutouts.hale.trainer import HaleTrainer


class CrossValidationManager:
    """
    Manager class for cross-validation experiments.

    Handles the execution of cross-validation across multiple folds,
    aggregates results, and generates comprehensive summaries.
    """

    def __init__(self, class_names: list[str] | None = None):
        """
        Initialize the CrossValidationManager.

        Args:
            class_names: List of class names (defaults to configured classes)
        """
        self.class_names = class_names or config.class_names
        self.trainer = HaleTrainer(class_names=self.class_names)
        setup_basic_logging()

    @staticmethod
    def _compute_stats(values: list[float]) -> dict[str, float]:
        """Compute statistics for a list of values."""
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "values": values,
        }

    @staticmethod
    def _format_metric_line(name: str, mean: float, min_val: float, max_val: float) -> str:
        """Format metric line for logging."""
        range_half = (max_val - min_val) / 2
        return f"{name:15}: {mean:.4f} ± {range_half:.4f} (range: {min_val:.4f}-{max_val:.4f})"

    def run_cross_validation(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """
        Run cross-validation training across all folds.

        Args:
            df: Dataset DataFrame

        Returns:
            Tuple of (results dictionary, summary dictionary)
        """
        results = {}
        fold_metrics = []
        fold_confusion_matrices = []
        fold_classification_reports = []

        start_time = datetime.now()
        logging.info(f"Starting cross-validation training at {start_time}")
        logging.info(f"Training {config.N_FOLDS} folds with {config.classes} classes")

        # Create unified loggers for the entire cross-validation run
        main_loggers = self._setup_main_loggers(start_time)

        # Train each fold
        for fold_num in range(1, config.N_FOLDS + 1):
            logging.info(f"\n{'=' * 60}")
            logging.info(f"TRAINING FOLD {fold_num}/{config.N_FOLDS}")
            logging.info(f"{'=' * 60}")

            fold_start_time = datetime.now()

            try:
                fold_result = self._train_single_fold(df, fold_num, main_loggers, fold_start_time)
                results[fold_num] = fold_result

                if fold_result["status"] == "success":
                    fold_metrics.append(self._extract_fold_metrics(fold_result))
                    if fold_result.get("confusion_matrix"):
                        fold_confusion_matrices.append(np.array(fold_result["confusion_matrix"]))
                    if fold_result.get("classification_report"):
                        fold_classification_reports.append(fold_result["classification_report"])

                    logging.info(
                        f"Fold {fold_num} completed in {fold_result.get('training_time', 0):.1f}s | "
                        f"Accuracy: {fold_result['test_accuracy']:.4f}, F1: {fold_result['test_f1']:.4f}"
                    )

            except Exception as e:
                logging.error(f"Fold {fold_num} failed with error: {e}")
                results[fold_num] = {
                    "fold": fold_num,
                    "status": "error",
                    "error": str(e),
                    "training_time": (datetime.now() - fold_start_time).total_seconds(),
                }

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Generate summary
        summary = self._generate_summary(fold_metrics, fold_confusion_matrices, fold_classification_reports, total_time)

        # Log summary to main loggers
        self._log_summary_to_main_loggers(main_loggers, summary)

        # Save results and log summary
        self._save_and_log_results(summary, fold_metrics, results)

        return results, summary

    def _setup_main_loggers(self, start_time: datetime) -> list:
        """Set up unified loggers for the entire cross-validation run."""
        experiment_name = f"hale_{config.classes}_cv_{start_time.strftime('%Y%m%d_%H%M%S')}"

        try:
            main_loggers = setup_loggers(experiment_name)
            experiment_config = {
                "n_folds": config.N_FOLDS,
                "classes": config.classes,
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "max_epochs": config.MAX_EPOCHS,
                "model_name": config.MODEL_NAME,
                "patience": getattr(config, "PATIENCE", config.EARLY_STOPPING_PATIENCE),
                "image_size": getattr(config, "IMAGE_SIZE", "unknown"),
            }

            # Log hyperparameters to each logger
            for logger in main_loggers:
                try:
                    log_hyperparameters(logger, experiment_config)
                except Exception as e:
                    logging.warning(f"Could not log hyperparameters to {type(logger).__name__}: {e}")

            logging.info(f"Unified experiment loggers created: {[type(logger).__name__ for logger in main_loggers]}")
            return main_loggers

        except Exception as e:
            logging.warning(f"Could not set up unified loggers: {e}")
            return []

    def _train_single_fold(
        self, df: pd.DataFrame, fold_num: int, main_loggers: list, fold_start_time: datetime
    ) -> dict:
        """Train a single fold and return results."""
        trainer, model, test_results = self.trainer.train_single_fold(df, fold_num, parent_loggers=main_loggers)
        training_time = (datetime.now() - fold_start_time).total_seconds()

        base_result = {
            "fold": fold_num,
            "trainer": trainer,
            "model": model,
            "training_time": training_time,
        }

        if test_results and len(test_results) > 0:
            test_metrics = test_results[0]
            cm, class_report = model.get_confusion_matrix_and_classification_report(self.class_names)

            return {
                **base_result,
                "status": "success",
                "test_loss": test_metrics.get("test_loss", float("nan")),
                "test_accuracy": test_metrics.get("test_acc", float("nan")),
                "test_f1": test_metrics.get("test_f1", float("nan")),
                "best_checkpoint": trainer.checkpoint_callback.best_model_path,
                "confusion_matrix": cm.tolist() if cm is not None else None,
                "classification_report": class_report,
            }

        return {**base_result, "status": "no_test_results", "error": "No test results available"}

    def _extract_fold_metrics(self, fold_result: dict) -> dict:
        """Extract metrics from fold result for summary."""
        return {
            "fold": fold_result["fold"],
            "test_loss": fold_result["test_loss"],
            "test_accuracy": fold_result["test_accuracy"],
            "test_f1": fold_result["test_f1"],
            "training_time": fold_result["training_time"],
        }

    def _generate_summary(
        self,
        fold_metrics: list[dict],
        fold_confusion_matrices: list[np.ndarray],
        fold_classification_reports: list[dict],
        total_time: float,
    ) -> dict:
        """Generate comprehensive cross-validation summary statistics."""
        summary = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "model_name": config.MODEL_NAME,
                "classes": config.classes,
                "n_folds": config.N_FOLDS,
                "total_training_time": total_time,
                "dataset_config": {
                    "batch_size": config.BATCH_SIZE,
                    "learning_rate": config.LEARNING_RATE,
                    "max_epochs": config.MAX_EPOCHS,
                    "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
                    "data_type": config.DATA_TYPE,
                },
            }
        }

        # Calculate metric statistics
        if fold_metrics:
            summary["metrics_summary"] = self._calculate_metric_statistics(fold_metrics)

        # Calculate average confusion matrix
        if fold_confusion_matrices:
            avg_cm = np.mean(fold_confusion_matrices, axis=0)
            summary["average_confusion_matrix"] = avg_cm.tolist()

        # Calculate average per-class metrics
        if fold_classification_reports:
            summary.update(self._calculate_class_metrics(fold_classification_reports))

        return summary

    def _calculate_metric_statistics(self, fold_metrics: list[dict]) -> dict:
        """Calculate statistics for main metrics across folds."""
        valid_metrics = [m for m in fold_metrics if not np.isnan(m.get("test_accuracy", float("nan")))]
        if not valid_metrics:
            return {}

        metric_keys = ["test_accuracy", "test_f1", "test_loss", "training_time"]
        result = {key: self._compute_stats([m[key] for m in valid_metrics]) for key in metric_keys}

        # Add total time for training_time metric
        result["training_time"]["total"] = float(np.sum([m["training_time"] for m in valid_metrics]))

        return result

    def _calculate_class_metrics(self, fold_classification_reports: list[dict]) -> dict:
        """Calculate per-class metrics across folds."""
        per_class_metrics = {}
        metric_names = ["precision", "recall", "f1-score", "support"]

        for class_name in self.class_names:
            class_data = {}
            for metric_name in metric_names:
                values = [
                    report[class_name].get(metric_name, 0)
                    for report in fold_classification_reports
                    if class_name in report
                ]
                if values:
                    # Compute stats without 'values' key for cleaner output
                    stats = self._compute_stats(values)
                    del stats["values"]
                    class_data[metric_name] = stats

            if class_data:
                per_class_metrics[class_name] = class_data

        result = {"per_class_metrics": per_class_metrics}

        # Overall macro/weighted averages
        for avg_type, key in [("macro avg", "macro_averaged_metrics"), ("weighted avg", "weighted_averaged_metrics")]:
            avg_metrics = self._calculate_averaged_metrics(fold_classification_reports, avg_type)
            if avg_metrics:
                result[key] = avg_metrics

        return result

    def _calculate_averaged_metrics(self, fold_classification_reports: list[dict], avg_type: str) -> dict | None:
        """Calculate macro or weighted averaged metrics."""
        metric_names = ["precision", "recall", "f1-score"]
        result = {}

        for metric_name in metric_names:
            values = [
                report[avg_type].get(metric_name, 0) for report in fold_classification_reports if avg_type in report
            ]
            if values:
                stats = self._compute_stats(values)
                del stats["values"]  # Remove values list for cleaner output
                result[metric_name] = stats

        return result if result else None

    def _log_summary_to_main_loggers(self, main_loggers: list, summary: dict) -> None:
        """Log summary metrics to main loggers."""
        if not main_loggers or not summary:
            return

        try:
            metrics_summary = summary.get("metrics_summary", {})
            metric_map = {"test_accuracy": "accuracy", "test_f1": "f1", "test_loss": "loss"}

            summary_metrics = {}
            for full_name, short_name in metric_map.items():
                if full_name in metrics_summary:
                    for stat in ["mean", "std"]:
                        summary_metrics[f"cv_{stat}_{short_name}"] = metrics_summary[full_name].get(stat, 0)

            summary_metrics.update(
                {
                    "cv_total_time": summary.get("experiment_info", {}).get("total_training_time", 0),
                    "cv_successful_folds": len(metrics_summary.get("test_accuracy", {}).get("values", [])),
                }
            )

            log_experiment_summary(main_loggers, summary_metrics)
        except Exception as e:
            logging.warning(f"Could not log summary metrics: {e}")

    def _save_and_log_results(self, summary: dict, fold_metrics: list[dict], results: dict) -> None:
        """Save cross-validation results and log summary."""
        # Save summary to files
        results_dir = self._save_results_to_files(summary, fold_metrics, results)

        # Log summary to console
        self._log_summary_to_console(summary)

        return results_dir

    def _save_results_to_files(self, summary: dict, fold_metrics: list[dict], results: dict) -> Path:
        """Save cross-validation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(config.LOG_DIR) / f"cv_results_{config.classes}_{config.MODEL_NAME}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Define files to save
        files_to_save = {
            "cross_validation_summary.json": summary,
            "detailed_results.json": {
                fold_num: {k: v for k, v in fold_result.items() if k not in ["trainer", "model"]}
                for fold_num, fold_result in results.items()
            },
            "config.json": {
                attr: getattr(config, attr)
                for attr in dir(config)
                if not attr.startswith("_") and not callable(getattr(config, attr))
            },
        }

        # Save JSON files
        for filename, data in files_to_save.items():
            with open(results_dir / filename, "w") as f:
                json.dump(data, f, indent=2, default=str)

        # Save fold metrics as CSV
        if fold_metrics:
            pd.DataFrame(fold_metrics).to_csv(results_dir / "fold_metrics.csv", index=False)

        logging.info(f"Cross-validation results saved to: {results_dir}")
        return results_dir

    def _log_summary_to_console(self, summary: dict) -> None:
        """Log cross-validation summary to console."""
        logging.info("\n" + "=" * 80)
        logging.info("CROSS-VALIDATION SUMMARY")
        logging.info("=" * 80)

        # Experiment info
        exp_info = summary.get("experiment_info", {})
        logging.info(f"Model: {exp_info.get('model_name', 'N/A')}")
        logging.info(f"Classes: {exp_info.get('classes', 'N/A')}")
        logging.info(f"Folds: {exp_info.get('n_folds', 'N/A')}")
        logging.info(f"Total Training Time: {exp_info.get('total_training_time', 0):.1f}s")

        # Metrics summary
        if "metrics_summary" in summary:
            self._log_metrics_summary(summary["metrics_summary"])

        # Per-class metrics
        if "per_class_metrics" in summary:
            self._log_per_class_metrics(summary["per_class_metrics"])

        # Macro/weighted averages
        if "macro_averaged_metrics" in summary:
            self._log_averaged_metrics("MACRO AVERAGED METRICS", summary["macro_averaged_metrics"])

        if "weighted_averaged_metrics" in summary:
            self._log_averaged_metrics("WEIGHTED AVERAGED METRICS", summary["weighted_averaged_metrics"])

        logging.info("=" * 80)

    def _log_metric_dict(self, metrics: dict, indent: str = "", skip_keys: set[str] | None = None) -> None:
        """Log metrics dictionary with formatting."""
        skip_keys = skip_keys or set()
        for metric_name, metric_data in metrics.items():
            if metric_name not in skip_keys:
                mean_val = metric_data.get("mean", 0)
                min_val = metric_data.get("min", 0)
                max_val = metric_data.get("max", 0)
                line = self._format_metric_line(metric_name, mean_val, min_val, max_val)
                logging.info(f"{indent}{line}")

    def _log_metrics_summary(self, metrics: dict) -> None:
        """Log metrics summary to console."""
        logging.info("\nMETRICS ACROSS FOLDS:")
        logging.info("-" * 40)
        self._log_metric_dict(metrics, skip_keys={"training_time"})

    def _log_per_class_metrics(self, per_class: dict) -> None:
        """Log per-class metrics to console."""
        logging.info("\nPER-CLASS METRICS (Mean ± Range/2):")
        logging.info("-" * 40)
        for class_name, metrics in per_class.items():
            logging.info(f"{class_name}:")
            self._log_metric_dict(metrics, indent="  ", skip_keys={"support"})

    def _log_averaged_metrics(self, title: str, averaged_metrics: dict) -> None:
        """Log macro/weighted averaged metrics to console."""
        logging.info(f"\n{title}:")
        logging.info("-" * 40)
        self._log_metric_dict(averaged_metrics)

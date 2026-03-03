"""
Hale Trainer class for encapsulating training logic.

This module contains the HaleTrainer class that handles the training workflow
for a single fold, making the code more organized and reusable.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import RichProgressBar
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import arccnet.models.cutouts.hale.config as config
from arccnet.models.cutouts.hale.data_preparation import prepare_dataset
from arccnet.models.cutouts.hale.evaluation import (
    find_misclassified_samples,
    generate_roc_curves,
    log_confusion_matrix_and_classification_report,
    log_misclassified_samples,
    log_roc_curves,
)
from arccnet.models.cutouts.hale.lightning_data import HaleDataModule
from arccnet.models.cutouts.hale.lightning_model import HaleLightningModel
from arccnet.models.cutouts.hale.logging_utils import (
    log_dataset_statistics,
    log_final_metrics,
    log_hyperparameters_to_comet,
    log_test_results,
    log_training_info,
    setup_loggers,
)


class HaleTrainer:
    """
    Trainer class for Hale classification models.

    Encapsulates the training workflow for a single fold, including data preparation,
    model creation, training, testing, and evaluation.
    """

    WARNING_PATTERNS = [
        ".*hipBLASLt.*",  # AMD GPUs warning
        ".*Precision.*not supported by the model summary.*",
        ".*does not have many workers.*",
        ".*Please use the new API settings to control TF32 behavior.*",
        ".*You are using a CUDA device.*Tensor Cores.*",
    ]

    def __init__(self, class_names: list[str] | None = None):
        """
        Initialize the trainer.
        """
        self.class_names = class_names or config.class_names
        self._setup_warnings()
        self.seed = int(getattr(config, "RANDOM_STATE", 42))
        self._seed_everything(self.seed)
        self.precision = self._get_precision()

    def _setup_warnings(self) -> None:
        """Suppress common warnings."""
        for pattern in self.WARNING_PATTERNS:
            warnings.filterwarnings("ignore", category=UserWarning, message=pattern)

    def _get_precision(self) -> str:
        """
        Get mixed precision setting.

        PyTorch 2.9 manages TF32/FP32 precision via backend-specific APIs that may
        clash with the deprecated `torch.set_float32_matmul_precision` helper.
        Let PyTorch keep its defaults and just report the precision Lightning will use.
        """
        accelerator = str(getattr(config, "ACCELERATOR", "auto")).lower()
        use_mixed_precision = torch.cuda.is_available() and accelerator != "cpu"
        precision = "16-mixed" if use_mixed_precision else "32-true"
        logging.info(f"Using precision: {precision}")
        return precision

    def _seed_everything(self, seed: int) -> None:
        """Seed all RNGs used by Lightning/PyTorch for reproducibility."""
        pl.seed_everything(seed, workers=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        logging.info("Global seed set to %d", seed)

    def prepare_dataset_once(self) -> pd.DataFrame:
        """
        Prepare and encode the dataset once.

        Returns:
            Processed DataFrame with encoded labels
        """
        filename = config.PROCESSED_DATASET_FILENAME
        processed_data_path = Path(config.DATA_FOLDER) / filename

        if processed_data_path.exists():
            logging.info(f"Loading previously prepared dataset: {processed_data_path.name}")
            df_processed = pd.read_parquet(processed_data_path)
        else:
            logging.info("Preparing dataset for the first time...")
            df_processed = prepare_dataset(save_path=str(processed_data_path))

        logging.info(f"Dataset shape: {df_processed.shape}")
        logging.info(f"Label distribution:\n{df_processed['grouped_labels'].value_counts()}")

        return self._encode_labels(df_processed)

    def _encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode labels consistently."""
        logging.info("Creating consistent label encoding from 'grouped_labels'...")
        label_encoder = LabelEncoder()
        df["model_labels"] = label_encoder.fit_transform(df["grouped_labels"])

        label_mapping = {label: int(index) for index, label in enumerate(label_encoder.classes_)}
        logging.info(f"Label mapping: {label_mapping}")

        model_labels_dist = df["model_labels"].value_counts().sort_index().to_dict()
        logging.info(f"Model labels distribution: {model_labels_dist}")

        # Verify class count
        actual_classes = len(label_encoder.classes_)
        if actual_classes != config.NUM_CLASSES:
            logging.warning(
                f"Expected {config.NUM_CLASSES} classes but found {actual_classes}: {list(label_encoder.classes_)}"
            )
        else:
            logging.info(f"Found expected {actual_classes} classes: {list(label_encoder.classes_)}")

        return df

    def train_single_fold(self, df: pd.DataFrame, fold_num: int = 1, parent_loggers: list | None = None) -> tuple:
        """
        Train model on a single fold.

        Args:
            df: Dataset DataFrame
            fold_num: Fold number to train
            parent_loggers: List of parent loggers to use instead of creating new ones

        Returns:
            Tuple of (trainer, model, test_results)
        """
        logging.info(f"Training on fold {fold_num}")

        # Create data module and setup
        data_module = self._create_data_module(df, fold_num)
        model = self._create_model(data_module)
        loggers = self._setup_loggers(fold_num, parent_loggers)
        trainer = self._create_trainer(loggers, fold_num)
        self._log_training_info(trainer)
        # Train and test
        trainer.fit(model, data_module)
        test_results = trainer.test(model, data_module, ckpt_path="best")
        self._evaluate_model(model, data_module, loggers, fold_num, test_results, trainer)

        return trainer, model, test_results

    def _create_data_module(self, df: pd.DataFrame, fold_num: int) -> HaleDataModule:
        """Create and setup data module."""
        data_module = HaleDataModule(df=df, fold_num=fold_num)
        data_module.setup("fit")

        # Create class distribution mapping
        train_df = data_module.train_dataset.df
        class_distribution = self._get_class_distribution(train_df)

        log_dataset_statistics(
            train_size=len(train_df),
            val_size=len(data_module.val_dataset.df),
            test_size=len(data_module.test_dataset.df),
            total_size=len(df),
            class_distribution=class_distribution,
        )

        return data_module

    def _get_class_distribution(self, df: pd.DataFrame) -> dict[str, int]:
        """Get class distribution with model labels."""
        label_counts = df["grouped_labels"].value_counts().sort_index()
        label_mapping = (
            df[["grouped_labels", "model_labels"]]
            .drop_duplicates(subset="grouped_labels")
            .set_index("grouped_labels")["model_labels"]
        )

        return {
            f"{label} (model_label={int(label_mapping[label])})": int(count) for label, count in label_counts.items()
        }

    def _create_model(self, data_module: HaleDataModule) -> HaleLightningModel:
        """Create model with class weights."""
        # Compute class weights
        train_labels = data_module.get_train_labels()
        unique_labels = np.unique(train_labels)
        num_classes = int(config.NUM_CLASSES)
        class_weights_np = np.ones(num_classes, dtype=np.float32)

        if unique_labels.size > 0:
            present_weights = compute_class_weight("balanced", classes=unique_labels, y=train_labels)
            class_weights_np[unique_labels.astype(int)] = present_weights.astype(np.float32)

        missing_labels = sorted(set(range(num_classes)) - set(unique_labels.tolist()))
        if missing_labels:
            logging.warning(
                "Training fold is missing classes %s. Assigning zero loss weight for those classes in this fold.",
                missing_labels,
            )
            class_weights_np[missing_labels] = 0.0

        class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

        logging.info(f"Class weights: {class_weights}")

        # Create model
        model = HaleLightningModel(
            num_classes=config.NUM_CLASSES,
            learning_rate=config.LEARNING_RATE,
            model_name=config.MODEL_NAME,
            class_weights=class_weights,
        )

        # Log model info
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model: {config.MODEL_NAME} with {config.NUM_CLASSES} classes")
        logging.info(f"Total trainable parameters: {num_params:,}")

        return model

    def _setup_loggers(self, fold_num: int, parent_loggers: list | None) -> list:
        """Setup loggers for training."""
        if parent_loggers is not None:
            logging.info(f"Using parent loggers for fold {fold_num}")
            return parent_loggers

        experiment_name = f"hale_fold_{fold_num}"
        loggers = setup_loggers(experiment_name)

        # Log hyperparameters to Comet if present
        for logger in loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log_parameters"):
                log_hyperparameters_to_comet(logger, fold_num)

        return loggers

    def _create_trainer(self, loggers: list, fold_num: int = 1) -> pl.Trainer:
        """Create PyTorch Lightning trainer."""
        return pl.Trainer(
            max_epochs=config.MAX_EPOCHS,
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            precision=self.precision,
            logger=loggers,
            callbacks=self._create_callbacks(fold_num),
            log_every_n_steps=config.LOG_EVERY_N_STEPS,
            enable_progress_bar=True,
            enable_model_summary=getattr(config, "ENABLE_MODEL_SUMMARY", True),
            deterministic=True,
            benchmark=False,
        )

    def _create_callbacks(self, fold_num: int) -> list:
        """Create training callbacks."""
        checkpoint_callback = ModelCheckpoint(
            monitor=config.CHECKPOINT_MONITOR,
            mode="max",
            save_top_k=1,
            filename=f"hale-fold{fold_num}-{{epoch:02d}}-{{val_acc:.3f}}",
            dirpath=getattr(config, "LOG_DIR", "logs"),
        )

        early_stop_callback = EarlyStopping(
            monitor=config.EARLY_STOPPING_MONITOR,
            patience=config.EARLY_STOPPING_PATIENCE,
            mode=config.EARLY_STOPPING_MODE,
            verbose=True,
        )

        return [checkpoint_callback, early_stop_callback, RichProgressBar(leave=True)]

    def _log_training_info(self, trainer: pl.Trainer) -> None:
        """Log training configuration information."""
        device_info = getattr(trainer, "device_ids", "auto")

        log_training_info(
            max_epochs=config.MAX_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            batch_size=config.BATCH_SIZE,
            data_type=config.DATA_TYPE,
            precision=self.precision,
            device_info=device_info,
        )

        # Additional device info
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA version: {torch.version.cuda}")
            logging.info(f"GPU device: {torch.cuda.get_device_name()}")

    def _evaluate_model(
        self,
        model: HaleLightningModel,
        data_module: HaleDataModule,
        loggers: list,
        fold_num: int,
        test_results: list[dict],
        trainer: pl.Trainer,
    ) -> None:
        """Perform comprehensive model evaluation."""
        if not test_results:
            logging.warning("No test results available for evaluation")
            return

        test_metrics = test_results[0]

        log_test_results(test_metrics, trainer.checkpoint_callback.best_model_path)
        log_confusion_matrix_and_classification_report(model, loggers, fold_num, self.class_names)
        self._generate_and_log_roc_curves(model, loggers, fold_num)
        self._find_and_log_misclassified_samples(model, data_module, loggers, fold_num)
        log_final_metrics(loggers, test_metrics, trainer.checkpoint_callback.best_model_path, fold_num)
        # Reset model's test collections for next fold
        model.reset_test_collections()

    def _generate_and_log_roc_curves(self, model: HaleLightningModel, loggers: list, fold_num: int) -> None:
        """Generate and log ROC curves."""
        try:
            y_true = model.test_targets
            logits_collection = getattr(model, "test_logits", [])
            if not logits_collection:
                logging.warning(f"No test logits available for ROC curves in fold {fold_num}")
                return

            logits = torch.cat(logits_collection, dim=0)
            y_pred_proba = torch.softmax(logits, dim=1)

            if len(y_true) == 0:
                logging.warning(f"No test predictions available for ROC curves in fold {fold_num}")
                return

            # Convert to numpy and generate ROC curves
            y_true_tensor = torch.tensor(y_true, device=logits.device)
            y_true_np = y_true_tensor.cpu().numpy()
            y_pred_proba_np = y_pred_proba.cpu().numpy()
            roc_image, roc_data = generate_roc_curves(y_true_np, y_pred_proba_np, self.class_names, fold_num)

            # Log ROC curves and AUC scores
            log_roc_curves(roc_image, roc_data, loggers, fold_num)
            logging.info(f"Generated and logged ROC curves for fold {fold_num}")

            for class_name, data in roc_data.items():
                logging.info(f"Fold {fold_num} - {class_name} AUC: {data['auc']:.4f}")

        except Exception as e:
            logging.warning(f"Could not generate ROC curves for fold {fold_num}: {e}")

    def _find_and_log_misclassified_samples(
        self, model: HaleLightningModel, data_module: HaleDataModule, loggers: list, fold_num: int
    ) -> None:
        """Find and log misclassified samples."""
        try:
            num_samples = max(1, int(getattr(config, "MISCLASSIFIED_SAMPLES_TO_LOG", 10)))

            # Prefer misclassifications cached during trainer.test() to avoid a second full test pass.
            if hasattr(model, "get_top_misclassified_samples"):
                misclassified_samples = model.get_top_misclassified_samples(num_samples=num_samples)
            else:
                misclassified_samples = find_misclassified_samples(
                    model, data_module, fold_num, num_samples=num_samples
                )

            if misclassified_samples:
                log_misclassified_samples(misclassified_samples, loggers, fold_num, self.class_names)
                logging.info(f"Found and logged {len(misclassified_samples)} misclassified samples for fold {fold_num}")
            else:
                logging.info(f"No misclassified samples found for fold {fold_num}")

        except Exception as e:
            logging.warning(f"Could not process misclassified samples for fold {fold_num}: {e}")

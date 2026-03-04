"""Training entrypoint for multiclass flare classification."""

import os
import logging
from typing import Any
from pathlib import Path

# Import comet_ml before torch/pytorch-lightning for automatic instrumentation.
try:
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from sklearn.utils.class_weight import compute_class_weight

from arccnet.models import preprocessing_common as pp_common
from arccnet.models.checkpoint_manager import MulticlassFlareCheckpointManager
from arccnet.models.flares import preprocessing
from arccnet.models.flares import split_cache_utils as cache_utils
from arccnet.models.flares import train_runtime_utils as tr_common
from arccnet.models.flares import utils as flare_utils
from arccnet.models.flares.multiclass import config
from arccnet.models.flares.multiclass.datamodule import FlareDataModule
from arccnet.models.flares.multiclass.model import FlareClassifier

torch.set_float32_matmul_precision("medium")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_split_cache_name() -> str:
    """Build deterministic split-cache key from data and preprocessing config."""
    sanitize = cache_utils.sanitize_cache_token
    return (
        f"{sanitize(config.FLARES_PARQ)}_{config.TARGET_COLUMN}"
        f"_seed{config.RANDOM_SEED}"
        f"_test{sanitize(config.TEST_SIZE)}"
        f"_val{sanitize(config.VAL_SIZE)}"
        f"_limb{int(bool(getattr(config, 'FILTER_SOLAR_LIMB', True)))}"
        f"_lon{sanitize(getattr(config, 'MAX_LONGITUDE', 65.0))}"
        f"_q{int(bool(getattr(config, 'APPLY_QUALITY_FILTER', True)))}"
        f"_p{int(bool(getattr(config, 'APPLY_PATH_FILTER', True)))}"
        f"_l{int(bool(getattr(config, 'APPLY_LONGITUDE_FILTER', False)))}"
        f"_n{int(bool(getattr(config, 'APPLY_NAN_FILTER', False)))}"
        f"_nt{sanitize(getattr(config, 'NAN_THRESHOLD', 0.05))}"
        f"_ds{sanitize(config.CUTOUT_DATASET_FOLDER)}"
    )


def save_split_cache(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: list[str],
    paths: dict[str, Path],
) -> None:
    """Persist split dataframes and metadata atomically."""
    cache_root = paths["train"].parent
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_utils.write_split_parquets(train_df, val_df, test_df, paths)

    metadata = {
        "class_names": class_names,
        "target_column": config.TARGET_COLUMN,
        "cache_name": build_split_cache_name(),
    }
    cache_utils.atomic_write_json(metadata, paths["meta"])

    logger.info("Saved multiclass split cache to: %s", cache_root)


def load_split_cache(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load split dataframes and class names from split cache."""
    metadata = cache_utils.read_json(paths["meta"])

    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Invalid class_names in split cache metadata: {paths['meta']}")

    train_df, val_df, test_df = cache_utils.read_split_parquets(paths)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Cached split contains an empty subset: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    logger.info(
        "Loaded multiclass split cache. Shapes: Train=%s, Val=%s, Test=%s",
        train_df.shape,
        val_df.shape,
        test_df.shape,
    )
    return train_df, val_df, test_df, class_names


def _derive_multiclass_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create multiclass flare target and filter out rows containing A/B events."""
    required_cols = {"C", "M", "X"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing flare count columns required for multiclass labels: {sorted(missing_cols)}")

    flare_columns = ["A", "B", "C", "M", "X"]
    flare_data = {}
    for column in flare_columns:
        if column in df.columns:
            flare_data[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        else:
            flare_data[column] = pd.Series(0.0, index=df.index)

    has_low_class_flares = (flare_data["A"] > 0) | (flare_data["B"] > 0)
    has_mx = (flare_data["M"] > 0) | (flare_data["X"] > 0)
    has_c_only = (flare_data["C"] > 0) & (~has_mx)

    targets = np.full(len(df), "Quiet", dtype=object)
    targets[has_mx.to_numpy()] = "M_X"
    targets[has_c_only.to_numpy()] = "C"

    out_df = df.copy()
    out_df[config.TARGET_COLUMN] = targets
    out_df = out_df.loc[~has_low_class_flares].copy()
    out_df = out_df.reset_index(drop=True)

    logger.info("Filtered out %d events with A/B-class flares.", int(has_low_class_flares.sum()))
    logger.info("Class distribution after filtering:\n%s", out_df[config.TARGET_COLUMN].value_counts())
    return out_df


def _filter_solar_limb(df: pd.DataFrame) -> pd.DataFrame:
    """Apply optional front-hemisphere filter by longitude."""
    if not bool(getattr(config, "FILTER_SOLAR_LIMB", True)):
        logger.info("Solar limb filtering disabled.")
        return df

    initial_count = len(df)
    df_filtered = pp_common.apply_longitude_filter(df, max_longitude=float(getattr(config, "MAX_LONGITUDE", 65.0)))
    df_filtered = df_filtered.reset_index(drop=True)
    removed = initial_count - len(df_filtered)
    pct = (removed / initial_count * 100.0) if initial_count > 0 else 0.0

    logger.info("Solar limb filtering removed %d rows (%.1f%%).", removed, pct)
    if getattr(config, "MAX_LATITUDE", None) is not None:
        logger.warning("MAX_LATITUDE is configured but latitude filtering is not currently applied.")

    return df_filtered


def _prepare_label_encoding(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Encode labels using explicit class order from config."""
    class_names = [str(name) for name in getattr(config, "CLASSES", ["Quiet", "C", "M_X"])]
    if not class_names:
        raise ValueError("config.CLASSES must define at least one class.")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    unknown_labels = sorted(set(df[config.TARGET_COLUMN].unique()) - set(class_to_idx))
    if unknown_labels:
        raise ValueError(f"Found labels not present in config.CLASSES: {unknown_labels}")

    encoded = df[config.TARGET_COLUMN].map(class_to_idx)
    if encoded.isna().any():
        raise ValueError("Label encoding produced NaN values.")

    out_df = df.copy()
    out_df[config.TARGET_COLUMN] = encoded.astype(np.int64)
    return out_df, class_names


def _format_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    """Create class count and percentage table for each split."""
    class_ids = list(range(len(class_names)))

    def _counts_by_class(df: pd.DataFrame) -> pd.Series:
        counts = df[config.TARGET_COLUMN].value_counts().reindex(class_ids, fill_value=0).to_numpy(dtype=np.int64)
        return pd.Series(counts, index=class_names)

    distribution = pd.DataFrame(
        {
            "Train": _counts_by_class(train_df),
            "Validation": _counts_by_class(val_df),
            "Test": _counts_by_class(test_df),
        },
    )

    formatted = distribution.copy()
    for split_name in formatted.columns:
        counts = formatted[split_name].astype(int)
        total = int(counts.sum())
        if total > 0:
            pct = (counts / total * 100.0).round(1)
        else:
            pct = pd.Series(0.0, index=counts.index)
        formatted[split_name] = counts.astype(str) + " (" + pct.astype(str) + "%)"

    return formatted


def prepare_multiclass_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load raw flare data, preprocess, encode labels, and create train/val/test splits."""
    input_flares_path = os.path.join(config.DATA_FOLDER, config.FLARES_PARQ)
    logger.info("Loading flare data from %s", input_flares_path)
    if not os.path.exists(input_flares_path):
        raise FileNotFoundError(f"Input file not found: {input_flares_path}")

    df = pd.read_parquet(input_flares_path)
    logger.info("Loaded %d flare events.", len(df))

    df = _derive_multiclass_targets(df)
    df = _filter_solar_limb(df)

    logger.info("Applying flare preprocessing pipeline...")
    df = preprocessing.preprocess_flare_data(
        df,
        apply_quality_filter=bool(getattr(config, "APPLY_QUALITY_FILTER", True)),
        apply_path_filter=bool(getattr(config, "APPLY_PATH_FILTER", True)),
        apply_longitude_filter=bool(
            getattr(config, "APPLY_LONGITUDE_FILTER", False) and not getattr(config, "FILTER_SOLAR_LIMB", True)
        ),
        apply_nan_filter=bool(getattr(config, "APPLY_NAN_FILTER", False)),
        max_longitude=float(getattr(config, "MAX_LONGITUDE", 65.0)),
        nan_threshold=float(getattr(config, "NAN_THRESHOLD", 0.05)),
        data_folder=config.DATA_FOLDER,
        dataset_folder=config.CUTOUT_DATASET_FOLDER,
    )

    logger.info("Verifying image file existence...")
    df_exists, missing_path_indices = flare_utils.check_fits_file_existence(
        df,
        data_folder=config.DATA_FOLDER,
        dataset_folder=config.CUTOUT_DATASET_FOLDER,
        image_type=str(getattr(config, "IMAGE_TYPE", "magnetograms")),
    )
    files_found = int(df_exists["file_exists"].sum())
    logger.info(
        "FITS verification: %d missing-path rows, %d/%d rows with existing files.",
        len(missing_path_indices),
        files_found,
        len(df_exists),
    )
    df = df_exists[df_exists["file_exists"]].copy()
    df = df.drop(columns=["file_exists", "resolved_path"], errors="ignore")

    if df.empty:
        raise ValueError("No rows remain after preprocessing and FITS-file verification.")
    if "number" not in df.columns:
        raise ValueError("Expected group column 'number' for AR-aware split, but it is missing.")

    df, class_names = _prepare_label_encoding(df)

    try:
        train_df, val_df, test_df = flare_utils.split_dataframe(
            df,
            stratify_col=config.TARGET_COLUMN,
            test_size=float(config.TEST_SIZE),
            val_size=float(config.VAL_SIZE),
            random_state=int(config.RANDOM_SEED),
        )
    except Exception as exc:  # noqa: BLE001
        class_counts = df[config.TARGET_COLUMN].value_counts().sort_index().to_dict()
        raise RuntimeError(
            "Failed to create stratified AR-aware split for multiclass training. "
            f"Class counts: {class_counts}. Original error: {exc}"
        ) from exc

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Data split produced an empty subset: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    distribution = _format_class_distribution(train_df, val_df, test_df, class_names)
    logger.info("\nMulticlass distribution (%s):\n%s", config.TARGET_COLUMN, distribution.to_string())

    return train_df.copy(), val_df.copy(), test_df.copy(), class_names


def get_or_prepare_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load cached splits when available, otherwise build and cache them."""
    cache_name = build_split_cache_name()
    split_cache_dir = getattr(config, "SPLIT_CACHE_DIR", None)
    cache_root = (
        Path(split_cache_dir) if split_cache_dir else (Path(config.DATA_FOLDER) / "cache" / "flares" / "multiclass")
    )
    cache_paths = cache_utils.build_split_cache_paths(cache_root=cache_root, cache_name=cache_name, include_meta=True)

    if cache_utils.split_cache_exists(cache_paths):
        logger.info("Detected existing multiclass split cache. Loading from %s", cache_paths["train"].parent)
        try:
            return load_split_cache(cache_paths)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load split cache (%s). Rebuilding splits.", exc)

    train_df, val_df, test_df, class_names = prepare_multiclass_splits()
    save_split_cache(train_df, val_df, test_df, class_names, cache_paths)
    return train_df, val_df, test_df, class_names


def calculate_class_weights(train_df: pd.DataFrame, num_classes: int) -> torch.Tensor | None:
    """Compute balanced class weights from training split."""
    if not bool(getattr(config, "USE_WEIGHTED_LOSS", True)):
        logger.info("Using unweighted loss.")
        return None

    train_labels = train_df[config.TARGET_COLUMN].to_numpy(dtype=np.int64)
    present_classes = np.unique(train_labels)

    class_weights_values = np.ones(num_classes, dtype=np.float32)
    if present_classes.size > 0:
        present_weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=train_labels)
        class_weights_values[present_classes.astype(int)] = present_weights.astype(np.float32)

    missing_classes = sorted(set(range(num_classes)) - set(present_classes.tolist()))
    if missing_classes:
        logger.warning(
            "Training split is missing classes %s; assigning zero class weight for those classes.",
            missing_classes,
        )
        class_weights_values[missing_classes] = 0.0

    class_weights = torch.tensor(class_weights_values, dtype=torch.float32)
    logger.info("Class weights: %s", class_weights.tolist())
    return class_weights


def log_dataset_histograms(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: list[str],
    comet_logger: CometLogger | None,
) -> None:
    """Log class-distribution metrics and a summary figure to Comet."""
    if comet_logger is None:
        return

    train_counts = train_df[config.TARGET_COLUMN].value_counts()
    val_counts = val_df[config.TARGET_COLUMN].value_counts()
    test_counts = test_df[config.TARGET_COLUMN].value_counts()

    for class_idx, class_name in enumerate(class_names):
        train_count = int(train_counts.get(class_idx, 0))
        val_count = int(val_counts.get(class_idx, 0))
        test_count = int(test_counts.get(class_idx, 0))

        tr_common.safe_comet_call(
            comet_logger,
            logger,
            "metric logging",
            "log_metric",
            f"dataset_count_train_{class_name}",
            train_count,
        )
        tr_common.safe_comet_call(
            comet_logger,
            logger,
            "metric logging",
            "log_metric",
            f"dataset_count_val_{class_name}",
            val_count,
        )
        tr_common.safe_comet_call(
            comet_logger,
            logger,
            "metric logging",
            "log_metric",
            f"dataset_count_test_{class_name}",
            test_count,
        )

    class_indices = list(range(len(class_names)))
    train_values = [int(train_counts.get(i, 0)) for i in class_indices]
    val_values = [int(val_counts.get(i, 0)) for i in class_indices]
    test_values = [int(test_counts.get(i, 0)) for i in class_indices]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(class_names, train_values, color="skyblue", alpha=0.8)
    axes[0].set_title(f"Train (n={len(train_df)})")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(class_names, val_values, color="lightgreen", alpha=0.8)
    axes[1].set_title(f"Validation (n={len(val_df)})")
    axes[1].tick_params(axis="x", rotation=45)
    axes[2].bar(class_names, test_values, color="lightcoral", alpha=0.8)
    axes[2].set_title(f"Test (n={len(test_df)})")
    axes[2].tick_params(axis="x", rotation=45)
    for axis in axes:
        axis.set_ylabel("Count")
    plt.tight_layout()
    tr_common.safe_comet_call(comet_logger, logger, "figure logging", "log_figure", "dataset_distributions", fig)
    plt.close(fig)


def create_transforms() -> tuple[T.Compose | None, T.Compose | None]:
    """Create training/validation transforms based on config."""
    if bool(getattr(config, "USE_AUGMENTATION", True)):
        train_transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=float(getattr(config, "HORIZONTAL_FLIP_PROB", 0.5))),
                T.RandomVerticalFlip(p=float(getattr(config, "VERTICAL_FLIP_PROB", 0.5))),
                T.RandomRotation(degrees=float(getattr(config, "ROTATION_DEGREES", 10))),
            ]
        )
    else:
        train_transform = None

    val_test_transform = None
    return train_transform, val_test_transform


def _dataset_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: list[str],
) -> dict[str, Any]:
    """Collect split metadata for logging and checkpoint artifacts."""
    metadata: dict[str, Any] = {
        "dataset_folder": config.CUTOUT_DATASET_FOLDER,
        "flares_parquet": config.FLARES_PARQ,
        "target_column": config.TARGET_COLUMN,
        "class_names": class_names,
        "test_size": config.TEST_SIZE,
        "val_size": config.VAL_SIZE,
        "random_seed": config.RANDOM_SEED,
        "split_group_column": "number",
        "split_rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "target_distribution": {
            "train": {
                class_names[idx]: int(train_df[config.TARGET_COLUMN].eq(idx).sum()) for idx in range(len(class_names))
            },
            "val": {
                class_names[idx]: int(val_df[config.TARGET_COLUMN].eq(idx).sum()) for idx in range(len(class_names))
            },
            "test": {
                class_names[idx]: int(test_df[config.TARGET_COLUMN].eq(idx).sum()) for idx in range(len(class_names))
            },
        },
    }
    return metadata


def main() -> None:
    """Train and evaluate multiclass flare model."""
    tr_common.set_deterministic_seed(int(config.RANDOM_SEED), logger)

    runtime = tr_common.resolve_trainer_runtime(config, logger)
    use_pin_memory = bool(getattr(config, "PIN_MEMORY", True) and str(runtime["accelerator"]).lower() != "cpu")
    logger.info(
        "Trainer runtime config: accelerator=%s, devices=%s, precision=%s, pin_memory=%s",
        runtime["accelerator"],
        runtime["devices"],
        runtime["precision"],
        use_pin_memory,
    )

    train_df, val_df, test_df, class_names = get_or_prepare_splits()
    num_classes = len(class_names)
    class_weights = calculate_class_weights(train_df, num_classes=num_classes)

    train_transform, val_test_transform = create_transforms()

    data_module = FlareDataModule(
        data_folder=config.DATA_FOLDER,
        dataset_folder=config.CUTOUT_DATASET_FOLDER,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_column=config.TARGET_COLUMN,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_target_height=config.IMG_TARGET_HEIGHT,
        img_target_width=config.IMG_TARGET_WIDTH,
        img_divisor=config.IMG_DIVISOR,
        img_min_val=config.IMG_MIN_VAL,
        img_max_val=config.IMG_MAX_VAL,
        pin_memory=use_pin_memory,
        persistent_workers=bool(getattr(config, "PERSISTENT_WORKERS", False)),
        prefetch_factor=int(getattr(config, "PREFETCH_FACTOR", 1)),
        multiprocessing_context=getattr(config, "DATALOADER_MULTIPROCESSING_CONTEXT", None),
        train_transform=train_transform,
        val_test_transform=val_test_transform,
    )

    flare_model = FlareClassifier(
        num_classes=num_classes,
        class_names=class_names,
        class_weights=class_weights,
        model_name=config.MODEL_NAME,
        pretrained=bool(getattr(config, "PRETRAINED", False)),
        learning_rate=float(config.LEARNING_RATE),
    )

    comet_logger = tr_common.init_comet_logger(config, logger, comet_ml)
    dataset_metadata = _dataset_metadata(train_df, val_df, test_df, class_names)
    run_parameters = tr_common.extract_uppercase_config_values(config)
    run_parameters.update(
        tr_common.flatten_dict(
            {
                "runtime": runtime,
                "dataset": dataset_metadata,
                "augmentation": {
                    "enabled": bool(getattr(config, "USE_AUGMENTATION", True)),
                    "horizontal_flip_prob": float(getattr(config, "HORIZONTAL_FLIP_PROB", 0.5)),
                    "vertical_flip_prob": float(getattr(config, "VERTICAL_FLIP_PROB", 0.5)),
                    "rotation_degrees": float(getattr(config, "ROTATION_DEGREES", 10.0)),
                },
            }
        )
    )
    tr_common.safe_comet_call(comet_logger, logger, "parameter logging", "log_parameters", run_parameters)
    log_dataset_histograms(train_df, val_df, test_df, class_names, comet_logger)

    checkpoint_manager = MulticlassFlareCheckpointManager(
        data_folder=config.DATA_FOLDER,
        model_name=config.MODEL_NAME,
        loss_function=config.LOSS_TYPE,
    )
    logger.info("Checkpoint directory: %s", checkpoint_manager.get_checkpoint_path())
    checkpoint_manager.save_config(tr_common.extract_uppercase_config_values(config))
    checkpoint_manager.training_metadata["class_names"] = class_names

    checkpoint_callback = checkpoint_manager.get_checkpoint_callback(
        monitor=config.CHECKPOINT_METRIC,
        mode="max",
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.CHECKPOINT_METRIC,
        patience=config.PATIENCE,
        mode="max",
        verbose=True,
    )

    callbacks: list[Callback] = [checkpoint_callback, early_stopping_callback]
    if comet_logger is not None:
        callbacks.append(
            tr_common.CometModelCheckpointCallback(
                comet_logger=comet_logger,
                logger=logger,
            )
        )

    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator=runtime["accelerator"],
        devices=runtime["devices"],
        precision=runtime["precision"],
        deterministic=True,
        enable_progress_bar=True,
        logger=comet_logger if comet_logger else False,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    trainer.fit(flare_model, datamodule=data_module)
    logger.info("Training finished.")

    tr_common.barrier_if_distributed(trainer)
    if tr_common.is_distributed_initialized():
        torch.distributed.destroy_process_group()

    if trainer.global_rank != 0:
        logger.info("Skipping post-training evaluation on global rank %s.", trainer.global_rank)
        return

    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path:
        raise RuntimeError("No best checkpoint path found after training.")

    logger.info("Loading best checkpoint weights from: %s", best_ckpt_path)
    best_checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = best_checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(f"Missing 'state_dict' in checkpoint: {best_ckpt_path}")
    flare_model.load_state_dict(state_dict)
    best_epoch = tr_common.extract_best_epoch(best_ckpt_path, best_checkpoint)

    logger.info("Starting single-device testing with best checkpoint weights...")
    eval_devices = tr_common.single_device_for_eval(runtime["devices"])
    with tr_common.single_process_env():
        eval_trainer = pl.Trainer(
            accelerator=runtime["accelerator"],
            devices=eval_devices,
            precision=runtime["precision"],
            deterministic=True,
            enable_progress_bar=True,
            logger=comet_logger if comet_logger else False,
        )
        test_results = eval_trainer.test(model=flare_model, datamodule=data_module, ckpt_path=None)

    if test_results:
        for metric_name, metric_value in test_results[0].items():
            if isinstance(metric_value, (float, int)):
                tr_common.safe_comet_call(
                    comet_logger, logger, "metric logging", "log_metric", metric_name, float(metric_value)
                )

    best_metric_value = None
    if checkpoint_callback.best_model_score is not None:
        best_metric_value = float(checkpoint_callback.best_model_score.item())

    num_epochs_trained = tr_common.num_completed_fit_epochs(trainer)
    early_stopping_triggered = bool(getattr(early_stopping_callback, "stopped_epoch", 0) > 0)

    training_metadata = {
        "best_epoch": best_epoch,
        "best_checkpoint_path": best_ckpt_path,
        f"best_{config.CHECKPOINT_METRIC}": best_metric_value,
        "num_epochs_trained": num_epochs_trained,
        "early_stopping_triggered": early_stopping_triggered,
        "class_names": class_names,
        "dataset_metadata": dataset_metadata,
        "test_results": test_results[0] if test_results else {},
    }

    checkpoint_manager.save_training_metadata(training_metadata)
    checkpoint_manager.save_minimal_logging(
        best_epoch=best_epoch,
        best_metric_value=best_metric_value,
        best_metric_name=config.CHECKPOINT_METRIC,
        num_epochs_trained=num_epochs_trained,
        early_stopping_triggered=early_stopping_triggered,
        additional_metrics=test_results[0] if test_results else {},
    )

    if test_results:
        checkpoint_manager.save_classification_report(
            {
                "test_metrics": test_results[0],
                "model_name": config.MODEL_NAME,
                "loss_function": config.LOSS_TYPE,
                "class_names": class_names,
                "best_checkpoint_path": best_ckpt_path,
                "dataset_metadata": dataset_metadata,
            }
        )

    logger.info("All checkpoints and metadata saved to: %s", checkpoint_manager.get_checkpoint_path())


def train() -> None:
    """Backward-compatible entry point."""
    main()


if __name__ == "__main__":
    main()

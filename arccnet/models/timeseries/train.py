"""Training script for timeseries flare forecasting using PyTorch Lightning."""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateFinder
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from sklearn.utils.class_weight import compute_class_weight

from arccnet.models import train_utils
from arccnet.models.checkpoint_manager import CheckpointManager

try:
    from pytorch_lightning.loggers.comet import CometLogger
except Exception:  # pragma: no cover - optional dependency
    CometLogger = None

torch.set_float32_matmul_precision("medium")

# Handle both module and direct execution
if __name__ == "__main__" and __package__ is None:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    import arccnet.models.timeseries.config as ts_config
    from arccnet.models.timeseries.data_module import FlareDataModule
    from arccnet.models.timeseries.dataset import SDOTimeseriesDataset
    from arccnet.models.timeseries.lightning_module import FlareForecasterLightning
    from arccnet.models.timeseries.manifest import build_dataset
    from arccnet.models.timeseries.splitters import get_split
else:
    from . import config as ts_config
    from .data_module import FlareDataModule
    from .dataset import SDOTimeseriesDataset
    from .lightning_module import FlareForecasterLightning
    from .manifest import build_dataset
    from .splitters import get_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = ts_config.SEED
RESIZE = ts_config.RESIZE
NUM_CHANNELS = ts_config.NUM_CHANNELS
NUM_TIMESTEPS = ts_config.NUM_TIMESTEPS
TIMESTEP_SELECTION = ts_config.TIMESTEP_SELECTION
GPU_ID = ts_config.GPU_ID
TASK_TYPE = ts_config.TASK_TYPE
SPLIT_STRATEGY = ts_config.SPLIT_STRATEGY
TRAIN_FRAC = ts_config.TRAIN_FRAC
VAL_FRAC = ts_config.VAL_FRAC
TRAIN_YEARS = ts_config.TRAIN_YEARS
VAL_YEARS = ts_config.VAL_YEARS
TEST_YEARS = ts_config.TEST_YEARS
NUM_CLASSES = ts_config.NUM_CLASSES
FLARE_CLASS_NAMES = list(getattr(ts_config, "FLARE_CLASS_NAMES", ["No-flare", "C", "M+"]))
BATCH_SIZE = ts_config.BATCH_SIZE
NUM_WORKERS = ts_config.NUM_WORKERS
USE_AUGMENTATION = ts_config.USE_AUGMENTATION
HFLIP_PROB = ts_config.HFLIP_PROB
VFLIP_PROB = ts_config.VFLIP_PROB
ROTATION_DEGREES = ts_config.ROTATION_DEGREES
REGRESSION_TARGETS = ts_config.REGRESSION_TARGETS
LEARNING_RATE = ts_config.LEARNING_RATE
WEIGHT_DECAY = ts_config.WEIGHT_DECAY
SPATIAL_FEATURE_DIM = ts_config.SPATIAL_FEATURE_DIM
TEMPORAL_NUM_LAYERS = ts_config.TEMPORAL_NUM_LAYERS
TEMPORAL_NUM_HEADS = ts_config.TEMPORAL_NUM_HEADS
TEMPORAL_DIM_FEEDFORWARD = ts_config.TEMPORAL_DIM_FEEDFORWARD
TEMPORAL_DROPOUT = ts_config.TEMPORAL_DROPOUT
TEMPORAL_POOLING = ts_config.TEMPORAL_POOLING
USE_TEMPORAL_TRANSFORMER = ts_config.USE_TEMPORAL_TRANSFORMER
PRETRAINED_SPATIAL = ts_config.PRETRAINED_SPATIAL
FREEZE_SPATIAL = ts_config.FREEZE_SPATIAL
HIDDEN_DIMS = ts_config.HIDDEN_DIMS
DROPOUT = ts_config.DROPOUT
EARLY_STOPPING_PATIENCE = ts_config.EARLY_STOPPING_PATIENCE
FIND_LR = ts_config.FIND_LR
LR_FIND_MIN = ts_config.LR_FIND_MIN
LR_FIND_MAX = ts_config.LR_FIND_MAX
LR_FIND_NUM_STEPS = ts_config.LR_FIND_NUM_STEPS
ACCELERATOR = ts_config.ACCELERATOR
MAX_EPOCHS = ts_config.MAX_EPOCHS
DEVICES = ts_config.DEVICES
GRAD_CLIP_MAX_NORM = ts_config.GRAD_CLIP_MAX_NORM
LOG_EVERY_N_STEPS = ts_config.LOG_EVERY_N_STEPS
TIMESERIES_ROOT = ts_config.TIMESERIES_ROOT
MANIFEST_PATH = ts_config.MANIFEST_PATH
LOSS_FUNCTION = ts_config.LOSS_FUNCTION
FOCAL_LOSS_ALPHA = ts_config.FOCAL_LOSS_ALPHA
FOCAL_LOSS_GAMMA = ts_config.FOCAL_LOSS_GAMMA
PRECISION = getattr(ts_config, "PRECISION", "auto")
SAFE_GPU_MODE = bool(getattr(ts_config, "SAFE_GPU_MODE", True))
PROJECT_NAME = getattr(ts_config, "PROJECT_NAME", "arcaff-timeseries-flare-forecasting")
ENABLE_COMET = bool(getattr(ts_config, "ENABLE_COMET", False))
COMET_PROJECT_NAME = getattr(ts_config, "COMET_PROJECT_NAME", PROJECT_NAME)
COMET_WORKSPACE = getattr(ts_config, "COMET_WORKSPACE", None)
MIN_AVAILABLE_PATH_FRACTION = float(os.getenv("ARCAFF_TS_MIN_AVAILABLE_PATH_FRACTION", "1.0"))

VALID_PRECISIONS = ("auto", "16-mixed", "bf16-mixed", "32-true", "64-true")


def _resolve_data_root(data_root):
    """
    Resolve data root across legacy and namespaced layouts.

    Supports:
    - /ARCAFF/data/04_final/data
    - /ARCAFF/data/timeseries/04_final/data
    """
    requested = Path(data_root)
    if requested.exists():
        return requested

    candidates = []
    requested_str = str(requested)

    if "/timeseries/04_final/data" in requested_str:
        candidates.append(Path(requested_str.replace("/timeseries/04_final/data", "/04_final/data")))
    elif requested_str.endswith("/04_final/data"):
        candidates.append(Path(requested_str.replace("/04_final/data", "/timeseries/04_final/data")))

    candidates.extend(
        [
            Path(ts_config.TIMESERIES_ROOT),
            Path("/ARCAFF/data/timeseries/04_final/data"),
            Path("/ARCAFF/data/04_final/data"),
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate_resolved = str(candidate)
        if candidate_resolved in seen:
            continue
        seen.add(candidate_resolved)
        if candidate.exists():
            logger.warning(f"Data root not found at {requested}; using existing path {candidate} instead.")
            return candidate

    raise FileNotFoundError(
        f"Data root not found: {requested}. Checked fallbacks: {', '.join(str(c) for c in candidates)}"
    )


def _preflight_data_availability(manifest_df, task_type, num_timesteps, timestep_selection, sample_count=32):
    """
    Check whether referenced FITS files are physically available before training.

    This catches cases where 04_final cutout files are broken symlinks to missing
    03_processed archives and avoids multi-worker error spam during training.
    """
    if len(manifest_df) == 0:
        raise RuntimeError("Manifest is empty after dataset build/load.")

    probe = manifest_df.sample(min(sample_count, len(manifest_df)), random_state=SEED)
    dataset_probe = SDOTimeseriesDataset(
        probe.reset_index(drop=True),
        split="test",
        task_type=task_type,
        resize=RESIZE,
        augment=False,
        num_timesteps=num_timesteps,
        timestep_selection=timestep_selection,
        norm_stats={"mean": [0.0] * NUM_CHANNELS, "std": [1.0] * NUM_CHANNELS},
    )

    checked = 0
    missing = 0
    for _, row in probe.iterrows():
        paths = dataset_probe._parse_paths(row["paths"])
        selected_timesteps = dataset_probe._select_timesteps(paths)
        for t_paths in selected_timesteps:
            for c_path in t_paths[:NUM_CHANNELS]:
                if c_path is None or c_path == "None":
                    continue
                checked += 1
                resolved = dataset_probe._resolve_existing_path(c_path)
                if not Path(resolved).exists():
                    missing += 1

    if checked == 0:
        raise RuntimeError("Preflight could not find any FITS paths to validate.")

    missing_ratio = missing / checked
    logger.info(f"Data preflight: checked={checked}, missing={missing}, missing_ratio={missing_ratio:.3f}")

    if missing_ratio >= 0.2:
        raise RuntimeError(
            "Data preflight failed: too many missing FITS files.\n"
            f"Checked {checked} paths, {missing} missing ({missing_ratio:.1%}).\n"
            "Likely cause: broken symlinks from 04_final/data/* to missing 03_processed files.\n"
            "If your processed archive exists elsewhere, set:\n"
            "  export ARCAFF_TIMESERIES_PROCESSED_ROOT=/absolute/path/to/03_processed\n"
            "or restore/create the expected path:\n"
            "  /ARCAFF/data/timeseries/03_processed"
        )


def _filter_manifest_by_data_availability(
    manifest_df,
    task_type,
    num_timesteps,
    timestep_selection,
    min_available_path_fraction=1.0,
):
    """
    Keep only samples whose referenced FITS paths are available on disk.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Built manifest with serialized per-sample path grids.
    task_type : str
        Active task type for dataset probe construction.
    min_available_path_fraction : float
        Minimum required fraction of resolvable paths per sample.
        1.0 means all referenced paths must exist.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (filtered_manifest, dropped_samples_report)
    """
    if len(manifest_df) == 0:
        raise RuntimeError("Manifest is empty after dataset build/load.")
    if not (0.0 < float(min_available_path_fraction) <= 1.0):
        raise ValueError("min_available_path_fraction must be in (0, 1].")

    dataset_probe = SDOTimeseriesDataset(
        manifest_df.head(1).reset_index(drop=True),
        split="test",
        task_type=task_type,
        resize=RESIZE,
        augment=False,
        num_timesteps=num_timesteps,
        timestep_selection=timestep_selection,
        norm_stats={"mean": [0.0] * NUM_CHANNELS, "std": [1.0] * NUM_CHANNELS},
    )

    total_samples = len(manifest_df)
    checked_counts = np.zeros(total_samples, dtype=np.int32)
    available_counts = np.zeros(total_samples, dtype=np.int32)

    for row_pos, (_, row) in enumerate(manifest_df.iterrows()):
        checked = 0
        available = 0
        paths = dataset_probe._parse_paths(row["paths"])
        selected_timesteps = dataset_probe._select_timesteps(paths)
        for t_paths in selected_timesteps:
            for c_path in t_paths[:NUM_CHANNELS]:
                if c_path is None or c_path == "None":
                    continue
                checked += 1
                resolved = dataset_probe._resolve_existing_path(c_path)
                if Path(resolved).exists():
                    available += 1
        checked_counts[row_pos] = checked
        available_counts[row_pos] = available

    available_fraction = np.divide(
        available_counts,
        checked_counts,
        out=np.zeros_like(available_counts, dtype=np.float32),
        where=checked_counts > 0,
    )
    keep_mask = (checked_counts > 0) & (available_fraction >= float(min_available_path_fraction))

    filtered_manifest = manifest_df.loc[keep_mask].reset_index(drop=True)
    dropped_report = manifest_df.loc[~keep_mask, ["sample_id", "noaa_ar", "date", "flare_class"]].copy()
    dropped_report["available_paths"] = available_counts[~keep_mask]
    dropped_report["checked_paths"] = checked_counts[~keep_mask]
    dropped_report["available_fraction"] = available_fraction[~keep_mask]

    kept = len(filtered_manifest)
    dropped = len(dropped_report)
    logger.info(
        "Availability filtering: kept=%d/%d samples (dropped=%d, min_available_path_fraction=%.3f)",
        kept,
        total_samples,
        dropped,
        float(min_available_path_fraction),
    )

    if dropped > 0:
        worst_missing_paths = int((dropped_report["checked_paths"] - dropped_report["available_paths"]).max())
        logger.warning(
            "Dropped %d sample(s) with unavailable FITS paths. Worst sample missing %d/%d paths.",
            dropped,
            worst_missing_paths,
            int(dropped_report["checked_paths"].max()),
        )

    if kept == 0:
        raise RuntimeError("All samples were dropped by availability filtering; cannot continue training.")

    return filtered_manifest, dropped_report


def _resolve_precision(requested_precision, trainer_accelerator):
    """
    Resolve trainer precision with a simple device-aware default.
    """
    precision = str(requested_precision).strip().lower()
    if precision not in VALID_PRECISIONS:
        raise ValueError(f"Invalid precision '{requested_precision}'. Expected one of: {', '.join(VALID_PRECISIONS)}")

    use_cuda = torch.cuda.is_available() and str(trainer_accelerator).lower() != "cpu"

    if precision == "auto":
        return "16-mixed" if use_cuda else "32-true"

    return precision


def _looks_like_worker_permission_error(exc):
    """Detect restricted multiprocessing environments (e.g., SemLock permission denied)."""
    msg = str(exc)
    return "[Errno 13]" in msg and "Permission denied" in msg


def _setup_comet_logger(
    run_name: str,
    enable_comet: bool,
    comet_project_name: str,
    comet_workspace: str | None,
):
    """Create an online Comet logger."""
    if not enable_comet:
        return None

    if CometLogger is None:
        logger.warning("Comet logging enabled but Comet dependencies are unavailable; skipping Comet logger.")
        return None

    common_kwargs = {"project": comet_project_name, "name": run_name, "online": True}
    if comet_workspace:
        common_kwargs["workspace"] = comet_workspace

    try:
        comet_logger = CometLogger(**common_kwargs)
        experiment = getattr(comet_logger, "experiment", None)
        exp_class = type(experiment).__name__ if experiment is not None else "None"

        # Comet may silently downgrade to an offline experiment when network/proxy fails.
        if "offline" in exp_class.lower():
            raise RuntimeError(
                "Comet initialized as offline experiment while online mode is required. "
                "Check proxy/network and COMET_API_KEY."
            )

        run_url = None
        if experiment is not None:
            run_url = getattr(experiment, "url", None)
            if callable(run_url):
                run_url = run_url()

        logger.info(
            "Comet logging enabled (online). project=%s workspace=%s run_url=%s",
            comet_project_name,
            comet_workspace or "<default>",
            run_url or "<unavailable>",
        )
        return comet_logger
    except Exception as exc:
        logger.warning("Could not initialize Comet logger in online mode: %s", exc)
        return None


def _setup_experiment_loggers(
    output_dir: Path,
    task_type: str,
    enable_comet: bool,
    comet_project_name: str,
    comet_workspace: str | None,
):
    """
    Create active experiment loggers.

    Returns:
        tuple[list, Optional[CometLogger]]
    """
    loggers = []

    try:
        tb_logger = TensorBoardLogger(
            save_dir=output_dir,
            name="lightning_logs",
        )
        loggers.append(tb_logger)
        logger.info("TensorBoard logging enabled")
    except ModuleNotFoundError:
        logger.warning("TensorBoard is unavailable; continuing without TensorBoard logger.")

    csv_logger = CSVLogger(
        save_dir=output_dir,
        name="csv_logs",
    )
    loggers.append(csv_logger)
    logger.info("CSV logging enabled")

    run_name = f"timeseries_{task_type}_{Path(output_dir).name}"
    comet_logger = _setup_comet_logger(
        run_name=run_name,
        enable_comet=enable_comet,
        comet_project_name=comet_project_name,
        comet_workspace=comet_workspace,
    )
    if enable_comet and comet_logger is None:
        raise RuntimeError(
            "Comet logging is enabled but online initialization failed. "
            "Check COMET_API_KEY, workspace/project names, and proxy/network configuration."
        )
    if comet_logger is not None:
        loggers.append(comet_logger)

    return loggers, comet_logger


def _serialize_hparam_value(value):
    """Convert values into logger-friendly scalar/list/dict representations."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [_serialize_hparam_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_hparam_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _collect_config_hyperparams(config_module) -> dict:
    """
    Capture all upper-case configuration attributes as a flat snapshot.
    """
    snapshot = {}
    for attr_name in dir(config_module):
        if not attr_name.isupper():
            continue
        try:
            value = getattr(config_module, attr_name)
        except Exception:  # pragma: no cover - defensive
            continue
        snapshot[f"config_{attr_name.lower()}"] = _serialize_hparam_value(value)
    return snapshot


def _log_hyperparameters(loggers: list, hyperparams: dict) -> None:
    """Log run hyperparameters to all active loggers with graceful fallbacks."""
    if not hyperparams:
        return

    for active_logger in loggers:
        try:
            if hasattr(active_logger, "log_hyperparams"):
                active_logger.log_hyperparams(hyperparams)
                continue

            experiment = getattr(active_logger, "experiment", None)
            if experiment is not None and hasattr(experiment, "log_parameters"):
                experiment.log_parameters(hyperparams)
        except Exception as exc:
            logger.warning("Could not log hyperparameters to %s: %s", type(active_logger).__name__, exc)


def _log_comet_artifacts(comet_logger, artifacts: list[Path]) -> None:
    """Upload run artifacts to Comet when available."""
    if comet_logger is None:
        return

    uploaded = 0
    for artifact_path in artifacts:
        path = Path(artifact_path)
        if not path.exists():
            continue
        logged = train_utils.safe_comet_call(
            comet_logger, logger, "asset logging", "log_asset", str(path), file_name=path.name
        )
        if logged:
            uploaded += 1

    if uploaded > 0:
        logger.info("Uploaded %d run artifact(s) to Comet", uploaded)


def main(args):
    """Main training function using PyTorch Lightning."""

    # Set GPU device from config
    if GPU_ID is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
        logger.info(f"Using GPU {GPU_ID}")

    # Set random seed for reproducibility without enforcing deterministic kernels.
    train_utils.set_global_seed(SEED, deterministic=False)
    logger.info(f"Random seed set to {SEED} (deterministic kernels disabled)")

    # Task type
    task_type = args.task_type if args.task_type else TASK_TYPE
    run_num_timesteps = args.num_timesteps if args.num_timesteps is not None else NUM_TIMESTEPS
    run_timestep_selection = args.timestep_selection if args.timestep_selection else TIMESTEP_SELECTION
    run_use_temporal_transformer = (
        args.use_temporal_transformer if args.use_temporal_transformer is not None else USE_TEMPORAL_TRANSFORMER
    )
    run_num_timesteps = max(1, int(run_num_timesteps))
    run_timestep_selection = str(run_timestep_selection).strip().lower()
    if run_timestep_selection not in {"first", "last"}:
        raise ValueError(
            f"Unsupported timestep_selection={run_timestep_selection!r}. Expected one of: 'first', 'last'."
        )
    logger.info(f"Task type: {task_type}")
    logger.info(
        "Input setup: num_timesteps=%s, timestep_selection=%s, use_temporal_transformer=%s",
        run_num_timesteps,
        run_timestep_selection,
        run_use_temporal_transformer,
    )

    # Setup checkpoint manager early so run artifacts can default to checkpoint directory.
    loss_fn = LOSS_FUNCTION if task_type == "multiclass" else "mse"
    checkpoint_mgr = CheckpointManager(
        root_name=f"timeseries/{task_type}",
        data_folder="/ARCAFF/data",
        model_name="resnet34_transformer",
        loss_function=loss_fn,
    )

    # Build dataset manifest for this run
    data_root = _resolve_data_root(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_mgr.checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using data root: {data_root}")
    logger.info(f"Run artifacts directory: {output_dir}")

    manifest_path = Path(args.manifest_path)
    if manifest_path.exists():
        logger.info(f"Manifest output already exists and will be overwritten: {manifest_path}")
    logger.info(f"Building dataset manifest from {data_root}")
    manifest_df = build_dataset(data_root, output_path=manifest_path)

    total_manifest_samples = len(manifest_df)
    logger.info(f"Total samples before availability filtering: {total_manifest_samples}")

    manifest_df, dropped_unavailable_samples = _filter_manifest_by_data_availability(
        manifest_df,
        task_type=task_type,
        num_timesteps=run_num_timesteps,
        timestep_selection=run_timestep_selection,
        min_available_path_fraction=args.min_available_path_fraction,
    )
    available_manifest_samples = len(manifest_df)
    dropped_manifest_samples = int(total_manifest_samples - available_manifest_samples)

    dropped_samples_path = None
    if dropped_manifest_samples > 0:
        dropped_samples_path = output_dir / "dropped_unavailable_samples.parquet"
        dropped_unavailable_samples.to_parquet(dropped_samples_path, index=False)
        logger.warning(
            "Dropped unavailable samples report saved to %s (n=%d)",
            dropped_samples_path,
            dropped_manifest_samples,
        )

    # Persist the filtered manifest used for this run so evaluation reproduces exact sample set.
    manifest_df.to_parquet(manifest_path, index=False)
    logger.info(f"Filtered manifest saved to {manifest_path} ({available_manifest_samples} samples)")

    _preflight_data_availability(
        manifest_df,
        task_type=task_type,
        num_timesteps=run_num_timesteps,
        timestep_selection=run_timestep_selection,
    )

    # Split dataset
    split_strategy = SPLIT_STRATEGY
    split_kwargs = {"seed": SEED}
    if split_strategy == "noaa":
        split_kwargs.update({"train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC})
    elif split_strategy == "time":
        split_kwargs.update({"train_years": TRAIN_YEARS, "val_years": VAL_YEARS, "test_years": TEST_YEARS})
    else:
        raise ValueError(f"Unsupported split strategy in config: {split_strategy}")

    split_data = get_split(manifest_df, strategy=split_strategy, **split_kwargs)
    train_mask = split_data["train_mask"]
    val_mask = split_data["val_mask"]
    test_mask = split_data["test_mask"]

    logger.info(f"{split_strategy.upper()} split:")
    logger.info(f"  Train: {train_mask.sum()} samples from {manifest_df[train_mask]['noaa_ar'].nunique()} ARs")
    logger.info(f"  Val:   {val_mask.sum()} samples from {manifest_df[val_mask]['noaa_ar'].nunique()} ARs")
    logger.info(f"  Test:  {test_mask.sum()} samples from {manifest_df[test_mask]['noaa_ar'].nunique()} ARs")

    if split_strategy == "noaa":
        train_noaa = set(manifest_df[train_mask]["noaa_ar"].unique().tolist())
        val_noaa = set(manifest_df[val_mask]["noaa_ar"].unique().tolist())
        test_noaa = set(manifest_df[test_mask]["noaa_ar"].unique().tolist())
        assert train_noaa.isdisjoint(val_noaa), "Leakage detected: train/val NOAA overlap"
        assert train_noaa.isdisjoint(test_noaa), "Leakage detected: train/test NOAA overlap"
        assert val_noaa.isdisjoint(test_noaa), "Leakage detected: val/test NOAA overlap"

    split_assignments = manifest_df[["sample_id", "noaa_ar", "date"]].copy()
    split_assignments["split"] = "unassigned"
    split_assignments.loc[train_mask, "split"] = "train"
    split_assignments.loc[val_mask, "split"] = "val"
    split_assignments.loc[test_mask, "split"] = "test"
    split_assignments_path = output_dir / "split_assignments.parquet"
    split_assignments.to_parquet(split_assignments_path, index=False)
    logger.info(f"Saved split assignments to {split_assignments_path}")

    # Compute class weights from training data for multiclass task
    class_weights_computed = None
    if task_type == "multiclass":
        train_labels = manifest_df[train_mask]["flare_class"].values
        unique_classes = np.unique(train_labels)

        # Compute balanced class weights
        weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=train_labels)

        # Create weight array for all classes (including those not in training set)
        class_weights_computed = np.ones(NUM_CLASSES)
        for cls, weight in zip(unique_classes, weights):
            class_weights_computed[int(cls)] = weight

        class_weights_computed = class_weights_computed.tolist()

        logger.info("Class distribution in training set:")
        for cls in range(NUM_CLASSES):
            count = (train_labels == cls).sum()
            pct = 100 * count / len(train_labels)
            weight = class_weights_computed[cls]
            class_name = FLARE_CLASS_NAMES[cls] if cls < len(FLARE_CLASS_NAMES) else f"class_{cls}"
            logger.info(f"  Class {cls} ({class_name}): {count} samples ({pct:.1f}%), weight: {weight:.3f}")

    # Compute normalization stats from training set
    logger.info("Computing normalization stats from training set...")
    train_dataset_temp = SDOTimeseriesDataset(
        manifest_df[train_mask].reset_index(drop=True),
        split="train",
        norm_stats=None,
        task_type=task_type,
        resize=RESIZE,
        augment=False,
        num_timesteps=run_num_timesteps,
        timestep_selection=run_timestep_selection,
    )
    norm_stats = train_dataset_temp.get_norm_stats()

    # Save normalization stats
    norm_stats_path = output_dir / "norm_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    logger.info(f"Normalization stats saved to {norm_stats_path}")

    effective_num_workers = args.num_workers if args.num_workers is not None else NUM_WORKERS

    def _build_datamodule(num_workers):
        return FlareDataModule(
            manifest_df=manifest_df,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            data_dir=data_root,
            norm_stats=norm_stats,
            task_type=task_type,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            resize=RESIZE,
            use_augmentation=USE_AUGMENTATION,
            num_timesteps=run_num_timesteps,
            timestep_selection=run_timestep_selection,
            hflip_prob=HFLIP_PROB,
            vflip_prob=VFLIP_PROB,
            rotation_degrees=ROTATION_DEGREES,
        )

    datamodule = _build_datamodule(effective_num_workers)

    # Create Lightning model
    model = FlareForecasterLightning(
        task_type=task_type,
        num_channels=NUM_CHANNELS,
        output_dim=NUM_CLASSES if task_type == "multiclass" else REGRESSION_TARGETS,
        flare_class_names=FLARE_CLASS_NAMES if task_type == "multiclass" else None,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        class_weights=class_weights_computed if task_type == "multiclass" else None,
        loss_function=LOSS_FUNCTION if task_type == "multiclass" else "mse",
        focal_alpha=FOCAL_LOSS_ALPHA,
        focal_gamma=FOCAL_LOSS_GAMMA,
        # Model architecture parameters
        spatial_feature_dim=SPATIAL_FEATURE_DIM,
        temporal_num_layers=TEMPORAL_NUM_LAYERS,
        temporal_num_heads=TEMPORAL_NUM_HEADS,
        temporal_dim_feedforward=TEMPORAL_DIM_FEEDFORWARD,
        temporal_dropout=TEMPORAL_DROPOUT,
        temporal_pooling=TEMPORAL_POOLING,
        use_temporal_transformer=run_use_temporal_transformer,
        pretrained_spatial=PRETRAINED_SPATIAL,
        freeze_spatial=FREEZE_SPATIAL,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    )

    # Callbacks
    checkpoint_callback = checkpoint_mgr.get_checkpoint_callback(
        monitor="val/primary_metric",
        mode="max",
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor="val/primary_metric",
            patience=EARLY_STOPPING_PATIENCE,
            mode="max",
            verbose=True,
        ),
    ]

    # Add learning rate finder if enabled in config
    if FIND_LR:
        logger.info("Learning rate finder enabled")
        callbacks.append(
            LearningRateFinder(
                min_lr=LR_FIND_MIN,
                max_lr=LR_FIND_MAX,
                num_training_steps=LR_FIND_NUM_STEPS,
            )
        )

    loggers, comet_logger = _setup_experiment_loggers(
        output_dir=output_dir,
        task_type=task_type,
        enable_comet=args.enable_comet,
        comet_project_name=args.comet_project_name,
        comet_workspace=args.comet_workspace,
    )

    trainer_accelerator = ACCELERATOR
    if trainer_accelerator == "gpu" and not torch.cuda.is_available():
        logger.warning("ACCELERATOR set to 'gpu' but CUDA not available; falling back to CPU")
        trainer_accelerator = "cpu"

    requested_precision = args.precision if args.precision is not None else PRECISION
    trainer_precision = _resolve_precision(requested_precision, trainer_accelerator)

    if trainer_accelerator != "cpu" and torch.cuda.is_available() and SAFE_GPU_MODE:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
        if trainer_precision in {"16-mixed", "bf16-mixed"}:
            logger.warning(
                "SAFE_GPU_MODE is enabled: forcing precision=32-true and disabling cuDNN to avoid CUDA engine errors."
            )
            trainer_precision = "32-true"
        else:
            logger.warning("SAFE_GPU_MODE is enabled: cuDNN disabled for training stability.")

    logger.info(
        "Trainer runtime: accelerator=%s, devices=%s, precision=%s, num_workers=%s, safe_gpu_mode=%s",
        trainer_accelerator,
        DEVICES,
        trainer_precision,
        effective_num_workers,
        SAFE_GPU_MODE,
    )

    run_hyperparams = _collect_config_hyperparams(ts_config)
    run_hyperparams.update(
        {
            "task_type": task_type,
            "project_name": PROJECT_NAME,
            "data_root": str(data_root),
            "manifest_path": str(manifest_path),
            "manifest_samples_total": total_manifest_samples,
            "manifest_samples_available": available_manifest_samples,
            "manifest_samples_dropped_unavailable": dropped_manifest_samples,
            "min_available_path_fraction": float(args.min_available_path_fraction),
            "output_dir": str(output_dir),
            "split_strategy": split_strategy,
            "train_frac": float(TRAIN_FRAC),
            "val_frac": float(VAL_FRAC),
            "train_years": _serialize_hparam_value(TRAIN_YEARS),
            "val_years": _serialize_hparam_value(VAL_YEARS),
            "test_years": _serialize_hparam_value(TEST_YEARS),
            "train_samples": int(train_mask.sum()),
            "val_samples": int(val_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "num_classes": NUM_CLASSES if task_type == "multiclass" else REGRESSION_TARGETS,
            "flare_class_names": FLARE_CLASS_NAMES if task_type == "multiclass" else [],
            "num_timesteps": int(run_num_timesteps),
            "timestep_selection": str(run_timestep_selection),
            "use_temporal_transformer": bool(run_use_temporal_transformer),
            "num_channels": int(NUM_CHANNELS),
            "resize": _serialize_hparam_value(RESIZE),
            "batch_size": int(BATCH_SIZE),
            "num_workers_config": int(NUM_WORKERS),
            "num_workers": int(effective_num_workers),
            "use_augmentation": bool(USE_AUGMENTATION),
            "hflip_prob": float(HFLIP_PROB),
            "vflip_prob": float(VFLIP_PROB),
            "rotation_degrees": float(ROTATION_DEGREES),
            "learning_rate": float(LEARNING_RATE),
            "weight_decay": float(WEIGHT_DECAY),
            "loss_function": LOSS_FUNCTION if task_type == "multiclass" else "mse",
            "focal_alpha": _serialize_hparam_value(FOCAL_LOSS_ALPHA),
            "focal_gamma": float(FOCAL_LOSS_GAMMA),
            "spatial_feature_dim": int(SPATIAL_FEATURE_DIM),
            "temporal_num_layers": int(TEMPORAL_NUM_LAYERS),
            "temporal_num_heads": int(TEMPORAL_NUM_HEADS),
            "temporal_dim_feedforward": int(TEMPORAL_DIM_FEEDFORWARD),
            "temporal_dropout": float(TEMPORAL_DROPOUT),
            "temporal_pooling": str(TEMPORAL_POOLING),
            "pretrained_spatial": bool(PRETRAINED_SPATIAL),
            "freeze_spatial": bool(FREEZE_SPATIAL),
            "hidden_dims": _serialize_hparam_value(HIDDEN_DIMS),
            "dropout": float(DROPOUT),
            "gpu_id": _serialize_hparam_value(GPU_ID),
            "seed": int(SEED),
            "max_epochs": int(MAX_EPOCHS),
            "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
            "grad_clip_max_norm": float(GRAD_CLIP_MAX_NORM),
            "log_every_n_steps": int(LOG_EVERY_N_STEPS),
            "find_lr": bool(FIND_LR),
            "lr_find_min": float(LR_FIND_MIN),
            "lr_find_max": float(LR_FIND_MAX),
            "lr_find_num_steps": int(LR_FIND_NUM_STEPS),
            "precision_requested": str(requested_precision),
            "precision": str(trainer_precision),
            "accelerator_config": str(ACCELERATOR),
            "accelerator": str(trainer_accelerator),
            "devices": _serialize_hparam_value(DEVICES),
            "safe_gpu_mode": bool(SAFE_GPU_MODE),
            "checkpoint_root_name": f"timeseries/{task_type}",
            "checkpoint_model_name": "resnet34_transformer",
            "checkpoint_loss_function": str(loss_fn),
            "enable_comet": bool(args.enable_comet),
            "comet_project_name": str(args.comet_project_name),
            "comet_workspace": _serialize_hparam_value(args.comet_workspace),
            "split_assignments_path": str(split_assignments_path),
            "norm_stats_path": str(norm_stats_path),
        }
    )
    if class_weights_computed is not None:
        run_hyperparams["class_weights"] = [float(weight) for weight in class_weights_computed]
    _log_hyperparameters(loggers, run_hyperparams)
    train_utils.safe_comet_call(
        comet_logger, logger, "tag logging", "add_tags", [task_type, split_strategy, "timeseries"]
    )

    def _build_trainer():
        return pl.Trainer(
            max_epochs=MAX_EPOCHS,
            callbacks=callbacks,
            logger=loggers if len(loggers) > 1 else loggers[0],
            accelerator=trainer_accelerator,
            devices=DEVICES,
            precision=trainer_precision,
            gradient_clip_val=GRAD_CLIP_MAX_NORM,
            log_every_n_steps=LOG_EVERY_N_STEPS,
            deterministic=False,
            benchmark=True,
        )

    trainer = _build_trainer()

    # Train
    logger.info("Starting training...")
    try:
        trainer.fit(model, datamodule)
    except PermissionError as exc:
        if effective_num_workers > 0 and _looks_like_worker_permission_error(exc):
            logger.warning(
                "DataLoader multiprocessing is not available in this environment; retrying with num_workers=0."
            )
            effective_num_workers = 0
            datamodule = _build_datamodule(effective_num_workers)
            trainer = _build_trainer()
            trainer.fit(model, datamodule)
        else:
            raise

    # Test on best model
    logger.info("Evaluating best model on test set...")
    test_results = trainer.test(model, datamodule, ckpt_path="best")

    best_checkpoint_path = checkpoint_callback.best_model_path or str(checkpoint_mgr.checkpoint_dir / "best.ckpt")
    best_checkpoint = Path(best_checkpoint_path)

    if test_results:
        final_metrics = {}
        for metric_name, metric_value in test_results[0].items():
            if isinstance(metric_value, (int, float, np.floating)):
                final_metrics[f"final_{metric_name.replace('/', '_')}"] = float(metric_value)
        if final_metrics:
            train_utils.safe_comet_call(comet_logger, logger, "final metric logging", "log_metrics", final_metrics)

    # Save final results summary
    results = {
        "task_type": task_type,
        "split_strategy": split_strategy,
        "manifest_path": str(manifest_path),
        "manifest_samples_total": total_manifest_samples,
        "manifest_samples_available": available_manifest_samples,
        "manifest_samples_dropped_unavailable": dropped_manifest_samples,
        "min_available_path_fraction": float(args.min_available_path_fraction),
        "dropped_unavailable_samples_path": str(dropped_samples_path) if dropped_samples_path else None,
        "norm_stats_path": str(norm_stats_path),
        "split_assignments_path": str(split_assignments_path),
        "checkpoint_dir": str(checkpoint_mgr.checkpoint_dir),
        "best_checkpoint": str(best_checkpoint_path),
    }

    results_path = output_dir / "training_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    if best_checkpoint.exists():
        logged_model = train_utils.safe_comet_call(
            comet_logger,
            logger,
            "model logging",
            "log_model",
            name=f"timeseries_{task_type}_best",
            file_or_folder=str(best_checkpoint),
        )
        if not logged_model:
            train_utils.safe_comet_call(
                comet_logger, logger, "checkpoint asset logging", "log_asset", str(best_checkpoint)
            )

    run_artifacts = [
        manifest_path,
        split_assignments_path,
        norm_stats_path,
        results_path,
        best_checkpoint,
    ]
    if dropped_samples_path is not None:
        run_artifacts.append(dropped_samples_path)
    _log_comet_artifacts(comet_logger, artifacts=run_artifacts)
    train_utils.safe_comet_call(comet_logger, logger, "run finalization", "end")

    logger.info(f"Training complete! Results saved to {results_path}")
    logger.info(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train timeseries flare forecasting model with PyTorch Lightning")

    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        choices=["multiclass", "regression"],
        help="Task type (default: from config.TASK_TYPE)",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=None,
        help="Number of timesteps per sample to load (default: from config.NUM_TIMESTEPS).",
    )
    parser.add_argument(
        "--timestep_selection",
        type=str,
        default=None,
        choices=["first", "last"],
        help="Select timesteps from the beginning or end of each sample (default: from config.TIMESTEP_SELECTION).",
    )
    parser.add_argument(
        "--use_temporal_transformer",
        type=train_utils.parse_bool_cli,
        nargs="?",
        const=True,
        default=None,
        help=(
            "Enable/disable temporal transformer. "
            "Examples: --use_temporal_transformer true|false. "
            "Default: from config.USE_TEMPORAL_TRANSFORMER."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=TIMESERIES_ROOT,
        help="Root directory containing sample folders",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=MANIFEST_PATH,
        help="Path where train.py writes the manifest built from --data_root for this run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory for run artifacts (split assignments, norm stats, logs, summary). "
            "Defaults to the checkpoint run folder under /ARCAFF/data/checkpoints/timeseries."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=VALID_PRECISIONS,
        help=(
            "Lightning precision mode. "
            "Use 'auto' to choose based on device availability (default from config.PRECISION)."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override DataLoader worker count (defaults to config.NUM_WORKERS).",
    )
    parser.add_argument(
        "--min_available_path_fraction",
        type=float,
        default=MIN_AVAILABLE_PATH_FRACTION,
        help=(
            "Drop samples with fewer than this fraction of resolvable FITS paths. "
            "Default 1.0 keeps only fully available samples."
        ),
    )
    parser.add_argument(
        "--enable_comet",
        type=train_utils.parse_bool_cli,
        nargs="?",
        const=True,
        default=ENABLE_COMET,
        help="Enable Comet logging (default: from config.ENABLE_COMET).",
    )
    parser.add_argument(
        "--comet_project_name",
        type=str,
        default=COMET_PROJECT_NAME,
        help="Comet project name (default: from config.COMET_PROJECT_NAME).",
    )
    parser.add_argument(
        "--comet_workspace",
        type=str,
        default=COMET_WORKSPACE,
        help="Comet workspace (default: from config.COMET_WORKSPACE).",
    )

    args = parser.parse_args()
    main(args)

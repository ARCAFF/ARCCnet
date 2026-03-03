"""Evaluation script for timeseries flare forecasting model."""

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as ts_config
from .dataset import SDOTimeseriesDataset
from .flare_forecaster import FlareForecaster
from .lightning_module import FlareForecasterLightning
from .metrics import (
    class_semantic_tag,
    compute_multiclass_metrics,
    compute_regression_metrics,
    get_operational_class_layout,
    normalize_class_name,
)
from .splitters import get_split

warnings.filterwarnings("ignore")

SEED = ts_config.SEED
SPLIT_STRATEGY = ts_config.SPLIT_STRATEGY
TRAIN_FRAC = ts_config.TRAIN_FRAC
VAL_FRAC = ts_config.VAL_FRAC
TRAIN_YEARS = ts_config.TRAIN_YEARS
VAL_YEARS = ts_config.VAL_YEARS
TEST_YEARS = ts_config.TEST_YEARS
NUM_CHANNELS = ts_config.NUM_CHANNELS
SPATIAL_FEATURE_DIM = ts_config.SPATIAL_FEATURE_DIM
TEMPORAL_NUM_LAYERS = ts_config.TEMPORAL_NUM_LAYERS
TEMPORAL_NUM_HEADS = ts_config.TEMPORAL_NUM_HEADS
TEMPORAL_DIM_FEEDFORWARD = ts_config.TEMPORAL_DIM_FEEDFORWARD
TEMPORAL_DROPOUT = ts_config.TEMPORAL_DROPOUT
TEMPORAL_POOLING = ts_config.TEMPORAL_POOLING
NUM_CLASSES = ts_config.NUM_CLASSES
REGRESSION_TARGETS = ts_config.REGRESSION_TARGETS
FREEZE_SPATIAL = ts_config.FREEZE_SPATIAL
HIDDEN_DIMS = ts_config.HIDDEN_DIMS
DROPOUT = ts_config.DROPOUT
TASK_TYPE = ts_config.TASK_TYPE
RESIZE = ts_config.RESIZE
NUM_WORKERS = ts_config.NUM_WORKERS
PIN_MEMORY = ts_config.PIN_MEMORY
PERSISTENT_WORKERS = ts_config.PERSISTENT_WORKERS
BATCH_SIZE = ts_config.BATCH_SIZE

# Backward-compatible aliases for existing helper imports.
_normalize_class_name = normalize_class_name
_class_semantic_tag = class_semantic_tag
_get_operational_class_layout = get_operational_class_layout


def _select_split_dataframe(manifest_df, split_name, split_assignments_path=None):
    """
    Select split dataframe for evaluation.

    Uses persisted split assignments when available to guarantee exact train/eval consistency.
    """
    if split_assignments_path:
        split_assignments_path = Path(split_assignments_path)
        if split_assignments_path.exists():
            if split_assignments_path.suffix.lower() == ".csv":
                split_assignments = pd.read_csv(split_assignments_path)
            else:
                split_assignments = pd.read_parquet(split_assignments_path)

            required_cols = {"sample_id", "split"}
            if not required_cols.issubset(split_assignments.columns):
                raise ValueError(
                    f"Split assignment file missing required columns: {required_cols}. "
                    f"Found: {set(split_assignments.columns)}"
                )

            merged = manifest_df.merge(
                split_assignments[["sample_id", "split"]],
                on="sample_id",
                how="inner",
                validate="one_to_one",
            )
            eval_df = merged[merged["split"] == split_name].drop(columns=["split"]).reset_index(drop=True)
            if len(eval_df) == 0:
                raise ValueError(f"No samples found for split '{split_name}' in {split_assignments_path}")
            return eval_df

        print(f"Warning: split assignments path not found: {split_assignments_path}")

    # Fallback path: deterministic split recreation from config.
    split_kwargs = {"seed": SEED}
    if SPLIT_STRATEGY == "noaa":
        split_kwargs.update({"train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC})
    elif SPLIT_STRATEGY == "time":
        split_kwargs.update({"train_years": TRAIN_YEARS, "val_years": VAL_YEARS, "test_years": TEST_YEARS})
    split_data = get_split(manifest_df, strategy=SPLIT_STRATEGY, **split_kwargs)
    return split_data[f"{split_name}_df"].reset_index(drop=True)


def _load_norm_stats(checkpoint_dir=None, norm_stats_path=None):
    """Load normalization statistics from explicit path or checkpoint-adjacent files."""
    candidates = []
    if norm_stats_path:
        candidates.append(Path(norm_stats_path))
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        candidates.append(checkpoint_dir / "norm_stats.json")
        candidates.append(checkpoint_dir.parent / "norm_stats.json")

    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f), candidate

    return None, None


def _load_model(checkpoint_path, task_type, device):
    """
    Load model from checkpoint and return optional calibrated threshold.

    Returns
    -------
    tuple[torch.nn.Module, float | None]
        (inference_model, m_plus_threshold_if_available)
    """
    checkpoint_path = Path(checkpoint_path)

    # Preferred: Lightning checkpoint.
    try:
        lightning_model = FlareForecasterLightning.load_from_checkpoint(
            str(checkpoint_path),
            map_location=device,
            strict=False,
        )
        lightning_model = lightning_model.to(device).eval()
        threshold = None
        if hasattr(lightning_model, "m_plus_threshold"):
            try:
                threshold = float(lightning_model.m_plus_threshold.detach().cpu().item())
            except Exception:
                threshold = None
        return lightning_model.model, threshold
    except Exception as lightning_error:
        print(f"Lightning checkpoint load failed, trying legacy format: {lightning_error}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # Legacy pure model checkpoint.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = FlareForecaster(
            task_type=task_type,
            num_channels=NUM_CHANNELS,
            spatial_feature_dim=SPATIAL_FEATURE_DIM,
            temporal_num_layers=TEMPORAL_NUM_LAYERS,
            temporal_num_heads=TEMPORAL_NUM_HEADS,
            temporal_dim_feedforward=TEMPORAL_DIM_FEEDFORWARD,
            temporal_dropout=TEMPORAL_DROPOUT,
            temporal_pooling=TEMPORAL_POOLING,
            output_dim=NUM_CLASSES if task_type == "multiclass" else REGRESSION_TARGETS,
            pretrained_spatial=False,
            freeze_spatial=FREEZE_SPATIAL,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device).eval()
        return model, None

    # Lightning checkpoint loaded manually.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        lightning_model = FlareForecasterLightning(
            task_type=task_type,
            num_channels=NUM_CHANNELS,
            output_dim=NUM_CLASSES if task_type == "multiclass" else REGRESSION_TARGETS,
            flare_class_names=list(getattr(ts_config, "FLARE_CLASS_NAMES", [])) if task_type == "multiclass" else None,
        )
        lightning_model.load_state_dict(checkpoint["state_dict"], strict=False)
        lightning_model = lightning_model.to(device).eval()
        threshold = None
        if hasattr(lightning_model, "m_plus_threshold"):
            try:
                threshold = float(lightning_model.m_plus_threshold.detach().cpu().item())
            except Exception:
                threshold = None
        return lightning_model.model, threshold

    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def _resolve_eval_threshold(threshold_arg, checkpoint_threshold=None):
    """Resolve evaluation threshold from CLI input or checkpoint metadata."""
    token = str(threshold_arg).strip().lower()
    if token == "auto":
        if checkpoint_threshold is not None and np.isfinite(float(checkpoint_threshold)):
            return float(checkpoint_threshold), "checkpoint"
        return 0.5, "default"

    try:
        threshold = float(threshold_arg)
    except ValueError as exc:
        raise ValueError(f"Invalid threshold '{threshold_arg}'. Use a float or 'auto'.") from exc

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be within [0, 1], got {threshold}")

    return float(threshold), "cli"


@torch.no_grad()
def evaluate(model, dataloader, device, task_type):
    """Run model inference and collect predictions."""
    model.eval()

    all_preds = []
    all_labels = []
    all_meta = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device)
        meta = batch["meta"]

        logits = model(x, mask=mask)
        if task_type == "multiclass":
            preds = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            preds = logits.cpu().numpy()

        all_preds.append(preds)
        all_labels.append(y)
        all_meta.append(meta)

    predictions = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    metadata = {"sample_id": [], "noaa_ar": [], "date": []}
    for meta_batch in all_meta:
        for key in metadata:
            values = meta_batch[key]
            if isinstance(values, list):
                metadata[key].extend(values)
            elif hasattr(values, "tolist"):
                as_list = values.tolist()
                if isinstance(as_list, list):
                    metadata[key].extend(as_list)
                else:
                    metadata[key].append(as_list)
            else:
                metadata[key].append(values)

    return predictions, labels, metadata


def main(args):
    """Main evaluation function."""
    task_type = args.task_type or TASK_TYPE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Task type: {task_type}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {args.manifest_path}")
    manifest = pd.read_parquet(args.manifest_path)

    eval_df = _select_split_dataframe(
        manifest_df=manifest,
        split_name=args.split,
        split_assignments_path=args.split_assignments_path,
    )
    print(f"Evaluating on {args.split} split: {len(eval_df)} samples")

    norm_stats, norm_stats_loaded_from = _load_norm_stats(
        checkpoint_dir=args.checkpoint_dir,
        norm_stats_path=args.norm_stats_path,
    )
    if norm_stats is None:
        print("Warning: normalization stats not found, computing from evaluation split (fallback only).")
    else:
        print(f"Loaded normalization stats from {norm_stats_loaded_from}")

    dataset = SDOTimeseriesDataset(
        eval_df,
        split=args.split,
        task_type=task_type,
        resize=RESIZE,
        augment=False,
        norm_stats=norm_stats,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
    )

    print(f"Loading model from {args.checkpoint_path}")
    model, checkpoint_threshold = _load_model(
        checkpoint_path=args.checkpoint_path,
        task_type=task_type,
        device=device,
    )
    eval_threshold, threshold_source = _resolve_eval_threshold(
        threshold_arg=args.threshold,
        checkpoint_threshold=checkpoint_threshold,
    )
    print(f"Using evaluation threshold={eval_threshold:.4f} (source={threshold_source})")

    print("Running inference...")
    predictions, labels, metadata = evaluate(model, dataloader, device, task_type=task_type)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")

    print("Computing metrics...")
    if task_type == "multiclass":
        configured_class_names = list(getattr(ts_config, "FLARE_CLASS_NAMES", []))
        metrics = compute_multiclass_metrics(
            labels,
            predictions,
            threshold=eval_threshold,
            class_names=configured_class_names,
        )
        metrics["evaluation_threshold"] = float(eval_threshold)
        metrics["evaluation_threshold_source"] = threshold_source
    else:
        metrics = compute_regression_metrics(labels, predictions)

    metrics_path = output_dir / f"metrics_{args.split}_{task_type}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    if args.save_predictions:
        pred_df = pd.DataFrame(
            {
                "sample_id": metadata["sample_id"],
                "noaa_ar": metadata["noaa_ar"],
                "date": metadata["date"],
            }
        )
        if task_type == "multiclass":
            num_classes = int(predictions.shape[1])
            configured_class_names = list(getattr(ts_config, "FLARE_CLASS_NAMES", []))
            layout = get_operational_class_layout(num_classes, class_names=configured_class_names)
            x_index = layout["x_index"]
            m_plus_indices = layout["m_plus_indices"]

            for class_idx, class_name in enumerate(layout["class_names"]):
                class_key = normalize_class_name(class_name)
                pred_df[f"pred_prob_{class_key}"] = predictions[:, class_idx]

            pred_df["pred_label"] = np.argmax(predictions, axis=1)
            pred_df["label"] = labels.astype(int)
            pred_df["pred_prob_m_plus"] = predictions[:, m_plus_indices].sum(axis=1)
            pred_df["label_m_plus"] = np.isin(labels.astype(int), m_plus_indices).astype(int)
            if x_index is not None:
                pred_df["pred_prob_x_plus"] = predictions[:, x_index]
                pred_df["label_x_plus"] = (labels == x_index).astype(int)
        else:
            pred_df["pred_log_ca"] = predictions[:, 0]
            pred_df["pred_log_ma"] = predictions[:, 1]
            pred_df["pred_log_xa"] = predictions[:, 2]
            pred_df["label_log_ca"] = labels[:, 0]
            pred_df["label_log_ma"] = labels[:, 1]
            pred_df["label_log_xa"] = labels[:, 2]

        pred_path = output_dir / f"predictions_{args.split}_{task_type}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved predictions to {pred_path}")

    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate timeseries flare forecasting model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint (.ckpt or .pt)")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing checkpoint artifacts (used to find norm_stats.json fallback)",
    )
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest parquet file")
    parser.add_argument(
        "--split_assignments_path",
        type=str,
        default=None,
        help="Path to split assignment parquet/csv saved during training (recommended)",
    )
    parser.add_argument("--task_type", type=str, default=None, choices=["multiclass", "regression"])
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default=None,
        help="Explicit path to normalization statistics JSON (overrides checkpoint-dir lookup)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ARCAFF/ARCCnet/outputs/timeseries/eval",
        help="Directory for outputs",
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument(
        "--threshold",
        type=str,
        default="auto",
        help="Threshold for derived binary metrics. Use float in [0,1] or 'auto' to use checkpoint/default.",
    )
    parser.add_argument("--save_predictions", action="store_true", help="Save per-sample predictions")

    args = parser.parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(args.checkpoint_path).parent)

    main(args)

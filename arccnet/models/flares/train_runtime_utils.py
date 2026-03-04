"""Shared runtime helpers for flare training entrypoints."""

import os
import re
import random
from typing import Any
from contextlib import contextmanager

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CometLogger

from arccnet.models import comet_utils


def safe_comet_call(comet_logger: CometLogger | None, logger, action: str, fn: str, *args: Any, **kwargs: Any) -> bool:
    """Call Comet methods safely so logging failures never crash a run."""
    return comet_utils.safe_comet_call(comet_logger, logger, action, fn, *args, **kwargs)


def extract_uppercase_config_values(config_module) -> dict[str, Any]:
    """Collect serializable, explicit module-level config values."""
    return {
        key: value
        for key, value in vars(config_module).items()
        if key.isupper() and not key.startswith("_") and not callable(value)
    }


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries for parameter logging."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=flat_key))
        elif isinstance(value, (list, tuple, set)):
            flat[flat_key] = ",".join(map(str, value))
        else:
            flat[flat_key] = value
    return flat


def set_deterministic_seed(seed: int, logger) -> None:
    """Set deterministic random seeds across Python, NumPy, Torch, and Lightning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed, workers=True)
    logger.info("Global seed set to %d.", seed)


def is_distributed_initialized() -> bool:
    """Return whether torch.distributed is available and initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def single_device_for_eval(devices: Any) -> Any:
    """Select a single-device setting for post-training evaluation."""
    if isinstance(devices, (list, tuple)) and len(devices) > 0:
        return [devices[0]]
    return 1


@contextmanager
def single_process_env():
    """Temporarily remove distributed env vars for isolated single-process eval."""
    dist_keys = (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "NODE_RANK",
        "GROUP_RANK",
        "ROLE_RANK",
    )
    saved = {key: os.environ.get(key) for key in dist_keys}
    for key in dist_keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def barrier_if_distributed(trainer: pl.Trainer) -> None:
    """Synchronize all ranks for distributed trainers."""
    if getattr(trainer, "world_size", 1) > 1:
        trainer.strategy.barrier()


def extract_best_epoch(best_ckpt_path: str, checkpoint_payload: dict[str, Any]) -> int | None:
    """Resolve best epoch from checkpoint payload or checkpoint filename."""
    epoch_value = checkpoint_payload.get("epoch")
    if isinstance(epoch_value, (int, float)):
        return int(epoch_value)

    match = re.search(r"best-(\d+)-", os.path.basename(best_ckpt_path))
    if match:
        return int(match.group(1))
    return None


def num_completed_fit_epochs(trainer: pl.Trainer) -> int:
    """Return number of completed fit epochs using Lightning progress trackers."""
    fit_loop = getattr(trainer, "fit_loop", None)
    if fit_loop is not None:
        epoch_progress = getattr(fit_loop, "epoch_progress", None)
        if epoch_progress is not None:
            current = getattr(epoch_progress, "current", None)
            completed = getattr(current, "completed", None) if current is not None else None
            if isinstance(completed, int):
                return completed
            if isinstance(completed, float):
                return int(completed)

    current_epoch = getattr(trainer, "current_epoch", 0)
    if isinstance(current_epoch, int):
        return max(current_epoch, 0)
    if isinstance(current_epoch, float):
        return max(int(current_epoch), 0)
    return 0


def resolve_trainer_runtime(config_module, logger) -> dict[str, Any]:
    """Validate runtime device availability and return Trainer settings."""
    runtime = {
        "accelerator": getattr(config_module, "ACCELERATOR", "auto"),
        "devices": getattr(config_module, "DEVICES", "auto"),
        "precision": getattr(config_module, "PRECISION", "16-mixed"),
    }

    accel = str(runtime["accelerator"]).lower()

    if accel in {"gpu", "cuda"}:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() returned False")
            if torch.cuda.device_count() < 1:
                raise RuntimeError("CUDA reports zero visible devices")
            # Force CUDA initialization now to avoid delayed crash at trainer.fit().
            torch.cuda.get_device_capability(0)
            logger.info(
                "CUDA preflight passed. Visible devices: %d. Primary device: %s",
                torch.cuda.device_count(),
                torch.cuda.get_device_name(0),
            )
        except Exception as exc:
            raise RuntimeError(f"CUDA initialization failed: {exc}") from exc

    if accel == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            runtime["accelerator"] = "gpu"
            logger.info(
                "Auto accelerator selected GPU. Visible devices: %d. Primary device: %s",
                torch.cuda.device_count(),
                torch.cuda.get_device_name(0),
            )
        else:
            runtime["accelerator"] = "cpu"
            runtime["devices"] = 1
            runtime["precision"] = "32-true"
            logger.info("Auto accelerator selected CPU (CUDA unavailable).")

    if str(runtime["accelerator"]).lower() == "cpu":
        devices = runtime["devices"]
        if devices == "auto":
            runtime["devices"] = 1
        elif isinstance(devices, int):
            runtime["devices"] = max(1, devices)
        elif isinstance(devices, (list, tuple)):
            runtime["devices"] = max(1, len(devices))
        else:
            runtime["devices"] = 1
        if str(runtime["precision"]).endswith("16-mixed"):
            logger.warning("Mixed precision is not supported on CPU. Falling back to 32-true precision.")
            runtime["precision"] = "32-true"

    return runtime


def init_comet_logger(config_module, logger, comet_ml_module) -> CometLogger | None:
    """Initialize Comet logger when enabled."""
    if not bool(getattr(config_module, "ENABLE_COMET_LOGGING", False)):
        return None

    if comet_ml_module is None:
        logger.warning("Comet logging enabled but `comet_ml` is not installed. Continuing without Comet.")
        return None

    logger.info("Comet logging is enabled. Initializing CometLogger...")
    common_kwargs = {"workspace": getattr(config_module, "COMET_WORKSPACE", None)}
    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        common_kwargs["api_key"] = api_key

    project_name = getattr(config_module, "COMET_PROJECT_NAME", None)
    try:
        comet_logger = CometLogger(project=project_name, **common_kwargs)
    except TypeError:
        comet_logger = CometLogger(project_name=project_name, **common_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to initialize Comet logger: %s", exc)
        return None

    logger.info("CometLogger initialized.")
    return comet_logger


class CometModelCheckpointCallback(Callback):
    """Log only the final best checkpoint artifact to Comet."""

    def __init__(self, comet_logger: CometLogger, logger, verbose: bool = False) -> None:
        super().__init__()
        self.comet_logger = comet_logger
        self.logger = logger
        self.verbose = bool(verbose)
        self._last_logged_best_path: str | None = None

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return

        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        best_path = getattr(checkpoint_callback, "best_model_path", None)
        if not best_path or not os.path.exists(best_path):
            return
        if best_path == self._last_logged_best_path:
            return

        if self.verbose:
            self.logger.info("Logging final best model to Comet...")
        logged = safe_comet_call(
            self.comet_logger,
            self.logger,
            "model logging",
            "log_model",
            "best_model",
            best_path,
            overwrite=True,
        )
        if logged:
            self._last_logged_best_path = best_path
            if self.verbose:
                self.logger.info("Final best model logged to Comet successfully.")

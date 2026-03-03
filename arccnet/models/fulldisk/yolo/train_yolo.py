"""Train a YOLO model on full-disk datasets."""

import os
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from arccnet.models.fulldisk.yolo import dataset_config as ds_cfg

try:
    import comet_ml
except Exception:  # pragma: no cover - optional dependency
    comet_ml = None

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"


def _setup_workspace() -> tuple[Path, Path, Path]:
    """Configure local temporary directories used by Ultralytics."""
    temp_workspace = Path(__file__).parent / "temp"
    temp_workspace.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(temp_workspace))
    os.environ.setdefault("ULTRALYTICS_CACHE_DIR", str(temp_workspace / "weights"))

    temp_configs_dir = temp_workspace / "configs"
    temp_runs_dir = temp_workspace / "runs"
    temp_weights_dir = temp_workspace / "weights"
    temp_datasets_dir = temp_workspace / "datasets"

    for directory in (temp_configs_dir, temp_runs_dir, temp_weights_dir, temp_datasets_dir):
        directory.mkdir(parents=True, exist_ok=True)

    SETTINGS.update(
        {
            "weights_dir": str(temp_weights_dir),
            "runs_dir": str(temp_runs_dir),
            "datasets_dir": str(temp_datasets_dir),
        }
    )
    return temp_workspace, temp_configs_dir, temp_runs_dir


def _maybe_login_comet(enable_comet: bool) -> None:
    """Attempt Comet login without hardcoded credentials."""
    if not enable_comet:
        return
    if comet_ml is None:
        print("Comet logging requested but `comet_ml` is not installed. Continuing without Comet.")
        return
    try:
        comet_ml.login()
        print("Comet login successful.")
    except Exception as exc:
        print(f"Comet login failed. Continuing without Comet. Error: {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on magnetogram or continuum data")
    parser.add_argument(
        "--data-type",
        type=str,
        default="mag",
        choices=["mag", "cont"],
        help="Type of data to train on: 'mag' for magnetogram or 'cont' for continuum (default: mag)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use for training: '' (auto), 'cpu', '0', '0,1', etc. (default: auto)",
    )
    parser.add_argument(
        "--enable-comet",
        action="store_true",
        help="Attempt Comet login with environment-based credentials.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    temp_workspace, temp_configs_dir, temp_runs_dir = _setup_workspace()

    config_path = Path(__file__).with_name("config.yaml")
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    data_root = ds_cfg.DATA_FOLDER / "YOLO" / args.data_type
    config["train"] = str(data_root / "images" / "train")
    config["val"] = str(data_root / "images" / "val")

    temp_config_path = temp_configs_dir / f"config_{args.data_type}_temp.yaml"
    with open(temp_config_path, "w", encoding="utf-8") as stream:
        yaml.dump(config, stream)

    print(f"Training on {args.data_type.upper()} data")
    print(f"  Train: {config['train']}")
    print(f"  Val: {config['val']}")
    print(f"  Classes: {config.get('nc', 'unknown')}")
    print(f"  Temp workspace: {temp_workspace}")

    _maybe_login_comet(args.enable_comet)

    model = YOLO("yolo11l.pt")
    train_args = {
        "data": str(temp_config_path),
        "imgsz": 1024,
        "batch": 32,
        "epochs": 500,
        "device": args.device,
        "patience": 25,
        "dropout": 0.25,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "project": str(temp_runs_dir),
        "name": f"{args.data_type}_train",
    }
    model.train(**train_args)


if __name__ == "__main__":
    main()

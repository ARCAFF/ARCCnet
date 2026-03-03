"""Configuration for full-disk YOLO dataset generation."""

import os
from pathlib import Path

DATA_FOLDER = Path(os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data/"))
DATASET_ROOT = Path("arccnet-v20251017")
DATASET_FOLDER = DATASET_ROOT / "04_final"
DATAFRAME_NAME = "data/region_detection/region_detection_noaa-xarp.parq"

YOLO_OUTPUT_MAG = DATA_FOLDER / "YOLO" / "mag"
YOLO_OUTPUT_CONT = DATA_FOLDER / "YOLO" / "cont"

LONGITUDE_THRESHOLD = 65.0
MIN_SIZE = 0.03
FILTER_SELECTED = True

LABEL_MAPPING = {
    "IA": "None",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma",
    "Gamma": "None",
    "Gamma-Delta": "None",
}

TRAIN_SPLIT_RATIO = 0.8
TEMPORAL_GAP_DAYS = 14
INCLUDE_EMPTY_LABELS = True

RESIZE_DIM = (1024, 1024)
USE_COLORMAP_MAG = False
NUM_CPUS = 30

import os

from arccnet.models.labels import CLASS_NAMES_DICT, LABEL_MAPPING_DICT

# Classes & Data Type
classes = "a-b-bg"  # Options: "qs-ia-ar", "ia-ar", "qs-ia", "qs-ia-a-b-bg", "a-b-bg"
class_names = CLASS_NAMES_DICT.get(classes, ["Unknown"])
label_mapping = LABEL_MAPPING_DICT[classes]
NUM_CLASSES = len(class_names)
DATA_TYPE = "magnetogram"  # Options: "magnetogram", "continuum", "both"

# Model Architecture
MODEL_NAME = "resnet50"  # Options: "resnet18", "resnet34", "resnet50"
LEAKY_RELU_NEGATIVE_SLOPE = 0.01
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-4

# Training Configuration
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
N_FOLDS = 8
RANDOM_STATE = 42
TRAIN_ALL_FOLDS = True  # False trains only fold 1 for testing

# Hardware
ACCELERATOR = "gpu"  # Options: "gpu", "cpu", "auto"
DEVICES = 1
NUM_WORKERS = 16

# Image Preprocessing
IMAGE_TARGET_HEIGHT = 200
IMAGE_TARGET_WIDTH = 200
IMAGE_DIVISOR = 800.0
HARDTANH_MIN_VAL = -1.0
HARDTANH_MAX_VAL = 1.0

# Data Augmentation (only applied to training data)
USE_AUGMENTATION = True
ROTATION_DEGREES = 20
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
PERSPECTIVE_DISTORTION_SCALE = 0.05
PERSPECTIVE_PROB = 0.25
AFFINE_TRANSLATE = (0.03, 0.03)
AFFINE_SCALE = (0.98, 1.03)
AFFINE_SHEAR = 3

# Optimization & Callbacks
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MONITOR = "val_acc"
EARLY_STOPPING_MODE = "max"  # For accuracy
CHECKPOINT_MONITOR = "val_acc"
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_MODE = "min"  # For loss
LR_SCHEDULER_MONITOR = "val_acc"  # Use loss for LR

# DataLoader
DATALOADER_PREFETCH_FACTOR = 2
DATALOADER_PIN_MEMORY = True
DATALOADER_PERSISTENT_WORKERS = True
DATALOADER_MULTIPROCESSING_CONTEXT = "spawn"

# Logging & Experiment Tracking
LOG_EVERY_N_STEPS = 10
MISCLASSIFIED_SAMPLES_TO_LOG = 10  # Top-confidence misclassified samples to log per fold
ENABLE_MODEL_SUMMARY = False
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
PROJECT_NAME = f"arcaff-v20250805-{classes}"
ENABLE_COMET = True
ENABLE_TENSORBOARD = True
ENABLE_CSV = True
TENSORBOARD_LOG_HYPERPARAMS = True

# Dataset Paths & Processing
DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
DATASET_FOLDER = "arcnet-v20251017/04_final"
DF_FILE_NAME = "data/cutout_classification/region_classification.parq"
PROCESSED_DATASET_FILENAME = f"processed_dataset_{classes}_{N_FOLDS}-splits_rs-{RANDOM_STATE}.parquet"
LONG_LIMIT_DEG = 65
UNDERSAMPLE = False

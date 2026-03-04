import os

NUM_TIMESTEPS = 6
TIMESTEP_SELECTION = "first"

NUM_AIA_CHANNELS = 9
NUM_HMI_CHANNELS = 1
NUM_CHANNELS = NUM_AIA_CHANNELS + NUM_HMI_CHANNELS
SPATIAL_HEIGHT = 400
SPATIAL_WIDTH = 800

AIA_WAVELENGTHS = [94, 131, 171, 193, 211, 304, 335, 1600, 1700]
HMI_WAVELENGTH = 6173
CHANNEL_ORDER = AIA_WAVELENGTHS + [HMI_WAVELENGTH]

TASK_TYPE = "multiclass"  # Options: "multiclass" (class label), "regression" (flare magnitude)

# Multiclass: Predict highest flare class in 24h window
# Classes: 0=No-flare, 1=C-class, 2=M+-class (M or X)
FLARE_CLASS_NAMES = ["No-flare", "C", "M+"]
NUM_CLASSES = len(FLARE_CLASS_NAMES)
M_PLUS_CLASS_START_INDEX = 2  # M+ means M or X
X_CLASS_INDEX = None  # No dedicated X class in the default 3-class setup

# Regression: Predict log10(peak flux) for each class
# Targets: [log10(C_peak), log10(M_peak), log10(X_peak)]
REGRESSION_TARGETS = 3

# Model architecture
MODEL_NAME = "resnet34"
SPATIAL_FEATURE_DIM = 512
FEATURE_DIM = 512
TEMPORAL_NUM_LAYERS = 3
TEMPORAL_NUM_HEADS = 8
TEMPORAL_DIM_FEEDFORWARD = 2048
TEMPORAL_DROPOUT = 0.1
TEMPORAL_POOLING = "mean"
USE_TEMPORAL_TRANSFORMER = True
PRETRAINED_SPATIAL = True
FREEZE_SPATIAL = False
HIDDEN_DIMS = [256]
DROPOUT = 0.2
HEAD_DROPOUT = 0.2

# Image preprocessing
RESIZE_HEIGHT = 256
RESIZE_WIDTH = 512
RESIZE = (RESIZE_HEIGHT, RESIZE_WIDTH)

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
GRAD_CLIP_MAX_NORM = 1.0

# Loss function
# Cross-entropy uses class weights computed from the training split in train.py.
LOSS_FUNCTION = "cross_entropy"  # Options: "cross_entropy", "focal"
FOCAL_LOSS_ALPHA = 1.0
FOCAL_LOSS_GAMMA = 2.0

# Data splitting
SPLIT_STRATEGY = "noaa"  # Options: "noaa", "time"
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42
RANDOM_SEED = SEED

TRAIN_YEARS = [2011, 2017, 2018, 2019, 2020]
VAL_YEARS = [2021]
TEST_YEARS = [2022]

# Data augmentation
USE_AUGMENTATION = False
HFLIP_PROB = 0.5
VFLIP_PROB = 0.5
HORIZONTAL_FLIP_PROB = HFLIP_PROB
VERTICAL_FLIP_PROB = VFLIP_PROB
ROTATION_DEGREES = 10

TEMPORAL_DROP_PROB = 0.0  # Probability to drop one timestep
CHANNEL_DROP_PROB = 0.0  # Probability to drop one AIA channel

ACCELERATOR = "gpu"
DEVICES = 1
GPU_ID = 0  # Which GPU to use
NUM_WORKERS = 24
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Training settings
FIND_LR = True  # Run learning rate finder before training
LR_FIND_MIN = 1e-7
LR_FIND_MAX = 1.0
LR_FIND_NUM_STEPS = 100

DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
TIMESERIES_ROOT = os.getenv(
    "ARCAFF_TIMESERIES_ROOT",
    os.path.join(DATA_FOLDER, "timeseries", "04_final", "data"),
)

# default file location train writes to (and eval reads from).
MANIFEST_PATH = os.path.join(DATA_FOLDER, "timeseries_manifest.parquet")

NORM_STATS_MAX_SAMPLES = 200
NORM_STATS_MAX_TIMESTEPS = NUM_TIMESTEPS
NORM_STATS_MAX_PIXELS_PER_IMAGE = 8192
PRECISION = "32-true"
SAFE_GPU_MODE = True

PROJECT_NAME = "arcaff-timeseries"
ENABLE_COMET = True
COMET_PROJECT_NAME = "arcaff-timeseries"
COMET_WORKSPACE = "arcaff"
LOG_EVERY_N_STEPS = 10

"""Configuration for McIntosh classification training."""

import os

import torchvision.transforms as v2

resnet_version = "resnet18"
gpu_index = 0
epochs = 500
patience = 15
batch_size = 32
num_workers = 12
learning_rate = 1e-5
random_state = 42

train_transforms = v2.Compose(
    [
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.98, 1.02)),
    ]
)

initial_teacher_forcing_ratio = 0.75
min_teacher_forcing_ratio = 0.0
teacher_forcing_decay = 0.9
teacher_forcing = True

data_folder = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
dataset_folder = "arccnet-v20251017/04_final"
df_name = "data/cutout_classification/region_classification.parq"
long_limit_deg = 65
nan_threshold = None
train_size = 0.7
val_size = 0.15
test_size = 0.15

plot_histograms = False

use_comet = True
project_name = "arcaff-mcintosh-v20251017"
workspace = "arcaff"

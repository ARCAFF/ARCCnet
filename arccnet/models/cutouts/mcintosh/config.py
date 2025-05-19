import os

import torchvision.transforms as v2

### General ###
resnet_version = "resnet18"
gpu_index = 0
epochs = 500
patience = 15
batch_size = 32
num_workers = 12  # os.cpu_count()
learning_rate = 1e-5
random_state = 42

train_transforms = v2.Compose(
    [
        #         v2.RandomVerticalFlip(),
        #         v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.98, 1.02)),
    ]
)

### Teacher Forcing ###
initial_teacher_forcing_ratio = 0.75
min_teacher_forcing_ratio = 0.0
teacher_forcing_decay = 0.9
teacher_forcing = True

### Dataset ###
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../../data/")
dataset_folder = "arccnet-cutout-dataset-v20240715"
df_name = "cutout-magnetic-catalog-v20240715.parq"
long_limit_deg = 65
train_size = 0.7
val_size = 0.15
test_size = 0.15

### Logging ###
plot_histograms = False

### Comet ###
use_comet = True
project_name = "arcaff-mcintosh"
workspace = "arcaff"

import os

from torchvision.transforms import v2

project_name = "arcaff-v2-qs-ia-a-b-bg"

batch_size = 32
num_workers = 12
num_epochs = 200
patience = 10
pretrained = True
learning_rate = 1e-5

model_name = "beit_base_patch16_224"
gpu_index = 0
device = "cuda:" + str(gpu_index)

data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../data/")
dataset_folder = "arccnet-cutout-dataset-v20240715"
df_file_name = "cutout-mcintosh-catalog-v20240715.parq"

label_mapping = {
    "QS": "QS",
    "IA": "IA",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma",
    "Gamma": None,
    "Gamma-Delta": None,
}

train_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomResizedCrop(size=(224, 224), scale=(0.9, 0.9), antialias=True),
        v2.RandomRotation(35),
    ]
)

val_transforms = None

import os

from torchvision.transforms import v2

mode = "qs-ia-a-b-bg"
project_name = "arcaff-v2-" + mode

batch_size = 16
num_workers = 32
num_epochs = 300
patience = 15
pretrained = True
learning_rate = 1e-5

model_name = "resnet18"
gpu_index = 0
device = "cuda:" + str(gpu_index)

data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../data")
dataset_folder = "arccnet-cutout-dataset-v20240715"
df_file_name = "cutout-mcintosh-catalog-v20240715.parq"


if mode == "qs-ia-ar":
    label_mapping = {
        "QS": "QS",
        "IA": "IA",
        "Alpha": "AR",
        "Beta": "AR",
        "Beta-Delta": "AR",
        "Beta-Gamma": "AR",
        "Beta-Gamma-Delta": "AR",
        "Gamma": "AR",
        "Gamma-Delta": "AR",
    }

elif mode == "ia-ar":
    label_mapping = {
        "QS": None,
        "IA": "IA",
        "Alpha": "AR",
        "Beta": "AR",
        "Beta-Delta": "AR",
        "Beta-Gamma": "AR",
        "Beta-Gamma-Delta": "AR",
        "Gamma": "AR",
        "Gamma-Delta": "AR",
    }

elif mode == "qs-ia":
    label_mapping = {
        "QS": "QS",
        "IA": "IA",
        "Alpha": None,
        "Beta": None,
        "Beta-Delta": None,
        "Beta-Gamma": None,
        "Beta-Gamma-Delta": None,
        "Gamma": None,
        "Gamma-Delta": None,
    }

elif mode == "qs-ia-a-b-bg":
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
        v2.RandomVerticalFlip(),
        v2.RandomHorizontalFlip(),
        v2.RandomPerspective(distortion_scale=0.1, p=0.25),
        v2.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
    ]
)

val_transforms = None

import os

from torchvision.transforms import v2

from arccnet.models import labels

classes = "qs-ia-a-b-bg"
project_name = "arcaff-v2-" + classes
label_mapping = labels.label_mapping_dict.get(classes)

batch_size = 16
num_workers = 32
num_epochs = 300
patience = 15
learning_rate = 1e-5

model_name = "resnet18"
pretrained = True
gpu_index = 0
device = "cuda:" + str(gpu_index)

data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../data")
dataset_folder = "arccnet-cutout-dataset-v20240715"
df_file_name = "cutout-mcintosh-catalog-v20240715.parq"


train_transforms = v2.Compose(
    [
        v2.RandomVerticalFlip(),
        v2.RandomHorizontalFlip(),
        v2.RandomPerspective(distortion_scale=0.1, p=0.25),
        v2.RandomAffine(degrees=30, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
    ]
)

val_transforms = None

# Logging
logging = ["comet"]

if "comet" in logging:
    project_name = "active-region-cutout-classification"
    workspace = "arcaff"
elif "wandb" in logging:
    project_name = "Active Region Cutout Classification"
    entity = "arcaff"
elif "neptune" in logging:
    project_name = "ARCAFF/Active-Region-Cutout-Classification"

# GPU Configuration
cuda_visible_devices = "1"

# Data
data_path = "/ARCAFF/data/sgkf_split.npz"
augmentation = True
data_resize = True
target_height = 224
target_width = 224
to_RGB = False
normalize = True
# normalization types:
#   - each_image: normalize each image based on its own minimum and maximum values.
#   - dataset: normalize the entire dataset based on the global minimum and maximum values
#           (computed on train dataset, the same scaling is then applied to the validation one)
normalize_type = "dataset"  # 'each_image'

# Model
model_name = "resnet18"
pretrained = False
batch_size = 128
loss = "cross_entropy"  # "focal_loss"
compute_class_weights = False
gamma = 3.0
learning_rate = 1e-5

# Training
num_epochs = 500
patience = 30

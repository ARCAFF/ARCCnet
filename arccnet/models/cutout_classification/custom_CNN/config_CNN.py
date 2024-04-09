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
normalize = True
# normalization types:
#   - each_image: normalize each image based on its own minimum and maximum values.
#   - dataset: normalize the entire dataset based on the global minimum and maximum values
#           (computed on train dataset, the same scaling is then applied to the validation one)
normalize_type = "dataset"  # 'each_image'

# Model
model_name = "custom_CNN"
number_of_conv_layers = 5
conv_layers_units = [64, 128, 128, 128, 128]
kernel_sizes = [(7, 7), (5, 5), (3, 3), (3, 3), (3, 3)]
padding_types = "same"
MLP_layers = [512, 256, 128]
MLP_activation = "relu"
batch_size = 64
dropout_rate = 0.3
regularization_lambda = 0.02
initializer = "glorot_uniform"
loss = "focal_loss"  #'cross_entropy'
gamma = 3.0
learning_rate = 1e-5

# Training
optimizer = "adam"
epochs = 500
patience = 30

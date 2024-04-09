# Comet Logging
project_name = "active-region-cutout-classification"
workspace = "arcaff"

# GPU Configuration
cuda_visible_devices = "1"

# Data
data_path = "/ARCAFF/data/sgkf_split.npz"
augmentation = True
target_height = 224
target_width = 224
to_RGB = False

# Model
model_name = "resnet34"
pretrained = True
batch_size = 128
loss = "focal_loss"  #'cross_entropy'
learning_rate = 1e-5

# Training
num_epochs = 500
patience = 30

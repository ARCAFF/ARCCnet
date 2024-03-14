# %%
import comet_ml
from ultralytics import YOLO

import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ['NEPTUNE_DISABLED'] = 'true'

# %%
comet_ml.init(
    project_name="fulldisk-detection-classification", 
    workspace="arcaff")
  
# %%
model = YOLO("yolov8x.pt")  # load a pretrained model

# Define training arguments
train_args = {
    'data': 'yolo_config/fulldisk640.yaml',
    'imgsz': 640,  # Image size
    'batch': 64, 
    'epochs': 1000,
    'device': [2, 3],  
    'patience': 200,
    'dropout': 0.1,
    'fliplr': 0.5
}

results = model.train(**train_args)

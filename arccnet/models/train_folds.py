# %%
import gc
import random
import time

import config_train as config
import matplotlib
import numpy as np
import pandas as pd
import torch
import utilities as ut
from comet_ml import Experiment
from torchvision.transforms import v2

magnetic_map = matplotlib.colormaps['hmimag'] #'gray' #
label_name = ['alpha', 'beta', 'betax']

random.seed(42)
torch.manual_seed(42)

run_comet = Experiment(
  project_name="alpha-beta-beta-x-classification",
  workspace="arcaff"
)

run_comet.add_tags([config.model_name])
run_comet.log_parameters({
    'model_name': config.model_name,
    'batch_size': config.batch_size,
    'GPU': f'GPU{config.gpu_index}_{torch.cuda.get_device_name()}',
    'num_epochs': config.num_epochs,
    'patience': config.patience
})

run_comet.log_code("config_train.py")
run_comet.log_code("utilities.py")

# %% load data
print('Loading data...')
start_time = time.time()
data = np.load(config.dataset_path)
df_dataset = pd.read_csv(config.dataset_df_path)

magn_norm = data['magn_norm']
cropped_magn = data['cropped_magn']
encoded_labels = data['labels']
del data # Delete the data variable to clear RAM
gc.collect()  # Force garbage collection
end_time = time.time()
print(f'Done, in {end_time-start_time:.2f} s.')
# %%
# Define augmentations
train_transforms = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(40)
])
# %%
run_id, weights_dir = ut.generate_run_id(config)
(test_losses, test_accuracies, test_precisions, 
 test_recalls, test_f1s, cms_test, reports_df) = ut.run_all_folds(
     weights_dir, config, df_dataset, magn_norm, cropped_magn, 
     encoded_labels, train_transforms, experiment = run_comet)

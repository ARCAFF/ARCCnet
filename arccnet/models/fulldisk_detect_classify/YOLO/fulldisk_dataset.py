# %%
# Operating System
import sys
import glob
import time

# Data Handling
import numpy as np
import pandas as pd
import utilities_fulldisk as ut

# sklearn
from sklearn.model_selection import train_test_split

# Images
from astropy.io import fits

sys.path.insert(0, "../")
# %%
folder_path = ut.folder_path
folder_fits = ut.folder_fits

# %%
df_magnetic = pd.read_parquet(folder_path + "/extraction-magnetic-catalog-v20231207.parq", engine="pyarrow")
df_mc = pd.read_parquet(folder_path + "/extraction-mcintosh-catalog-v20231207.parq", engine="pyarrow")
files_name_fits = sorted(glob.glob(folder_fits + "/*fits"))
file_img_fits = files_name_fits[0]
img_fits = fits.open(file_img_fits)
data = np.array(img_fits[1].data, dtype=float)

# %%
image_width, image_height = data.shape
class_dict = ut.class_dict

df_magnetic["yolo_label"] = df_magnetic.apply(
    lambda row: ut.to_yolo(
        row["magnetic_class"], row["top_right_cutout"], row["bottom_left_cutout"], image_width, image_height
    ),
    axis=1,
)

df_yolo = df_magnetic.groupby("processed_path")["yolo_label"].apply(lambda x: "\n".join(x)).reset_index()

# Split the data into training and validation sets
train_df, val_df = train_test_split(df_yolo, test_size=0.2, shuffle=True, random_state=42)

# %%
base_dir = "/ARCAFF/full_disk/YOLO/yolo640x640"
# %%
print("Processing validation dataset... \n")
start_time = time.time()
ut.process_and_save_fits(val_df, base_dir, "val", resize_dim=(640, 640))
end_time = time.time()
duration = end_time - start_time
minutes, seconds = divmod(duration, 60)
print(f"Processing completed. Time taken: {int(minutes):02}:{int(seconds):02}. \n\n")

# %%
print("Processing train dataset... \n")
start_time = time.time()
ut.process_and_save_fits(train_df, base_dir, "train", resize_dim=(640, 640))
end_time = time.time()
duration = end_time - start_time
minutes, seconds = divmod(duration, 60)
print(f"Processing completed. Time taken: {int(minutes):02}:{int(seconds):02}. \n\n")

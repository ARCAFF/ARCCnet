"""Quick visual checks for YOLO label overlays."""

# %%
import os
import random

from matplotlib import pyplot as plt

from arccnet.models.fulldisk.yolo import yolo_utils as ut

# %%

data_folder = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data/")
dataset_folder = "arccnet-v20251017/04_final"
df_name = "srs_clean_catalog.parq"

local_path_root = os.path.join(data_folder, dataset_folder)
YOLO_root_path = os.path.join(data_folder, "YOLO/mag")
YOLO_cont_root_path = os.path.join(data_folder, "YOLO/cont")


def main() -> None:
    mag_image_dir = os.path.join(YOLO_root_path, "images", "train")
    mag_images = [f for f in os.listdir(mag_image_dir) if f.endswith(".png")]
    image_name = random.choice(mag_images)
    image_path = os.path.join(mag_image_dir, image_name)
    img_mag = ut.draw_yolo_labels_on_image(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_mag)
    plt.title(f"Magnetogram: {image_name}")
    plt.axis("off")
    plt.show()

    cont_image_dir = os.path.join(YOLO_cont_root_path, "images", "train")
    cont_image_path = os.path.join(cont_image_dir, image_name)
    img_cont = ut.draw_yolo_labels_on_image(cont_image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_cont)
    plt.title(f"Continuum: {image_name}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()

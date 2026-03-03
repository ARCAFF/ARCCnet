"""Helpers for building YOLO datasets from full-disk observations."""

import os
from pathlib import Path

import numpy as np
import yaml
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import rotate

from astropy.io import fits

from arccnet.models import preprocessing_common as pp_common
from arccnet.models.fulldisk.yolo import dataset_config as cfg
from arccnet.visualisation import utils as ut_v

MAGNETOGRAM_DIVISOR = 800.0


def normalize_magnetogram(data: np.ndarray, divisor: float = MAGNETOGRAM_DIVISOR) -> np.ndarray:
    """Normalize magnetogram data to [0, 1] using hardtanh-style scaling."""
    data = np.nan_to_num(data, nan=0.0)
    data = data / divisor
    data = np.clip(data, -1.0, 1.0)
    data = (data + 1.0) / 2.0
    return data.astype(np.float32)


def normalize_continuum(data: np.ndarray) -> np.ndarray:
    """Normalize continuum data to [0, 1] with per-image min-max scaling."""
    data = np.nan_to_num(data, nan=0.0)
    data_min = np.min(data)
    data_max = np.max(data)

    if data_max - data_min > 0:
        data = (data - data_min) / (data_max - data_min)
    else:
        data = np.zeros_like(data)

    return data.astype(np.float32)


def to_yolo(encoded_label, top_right, bottom_left, img_width, img_height):
    """Convert bounding box coordinates to YOLO format string."""
    x1, y1 = bottom_left
    x2, y2 = top_right
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return f"{encoded_label} {x_center} {y_center} {width} {height}"


def process_fits_pair(
    row,
    local_path_root,
    base_dir_mag,
    base_dir_cont,
    dataset_type,
    resize_dim=(640, 640),
    cmap=False,
):
    """Process paired magnetogram/continuum FITS files for a single row."""
    mag_path = pp_common.resolve_project_path(row["path_mag"], local_root=local_path_root)
    cont_path = pp_common.resolve_project_path(row["path_cont"], local_root=local_path_root)
    if mag_path is None:
        print(f"Error processing magnetogram {row['path_mag']}: path could not be resolved")
        return
    if cont_path is None:
        print(f"Error processing continuum {row['path_cont']}: path could not be resolved")
        return

    label = row["yolo_label"]
    basename = Path(mag_path).name
    png_filename = os.path.splitext(basename)[0] + ".png"
    label_filename = os.path.splitext(basename)[0] + ".txt"
    target_width, target_height = resize_dim

    base_image_dir_mag = os.path.join(base_dir_mag, "images", dataset_type)
    base_label_dir_mag = os.path.join(base_dir_mag, "labels", dataset_type)
    os.makedirs(base_image_dir_mag, exist_ok=True)
    os.makedirs(base_label_dir_mag, exist_ok=True)
    output_image_path_mag = os.path.join(base_image_dir_mag, png_filename)

    try:
        with fits.open(mag_path) as img_fit:
            mag_data = img_fit[1].data
            header = img_fit[1].header

        crota2 = header.get("CROTA2", 0)
        if crota2 != 0:
            mag_data = rotate(mag_data, crota2, reshape=False, mode="constant", cval=0)
        mag_data = normalize_magnetogram(mag_data)
        mag_data = ut_v.pad_resize_normalize(mag_data, target_height=target_height, target_width=target_width)

        if cmap:
            plt.imshow(mag_data, cmap=ut_v.magnetic_map)
            plt.axis("off")
            plt.savefig(output_image_path_mag, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close()
        else:
            mag_data = (mag_data * 255).astype(np.uint8)
            img = Image.fromarray(mag_data, mode="L")
            img.save(output_image_path_mag)

        with open(os.path.join(base_label_dir_mag, label_filename), "w") as label_file:
            label_file.write(label)
    except Exception as e:
        print(f"Error processing magnetogram {mag_path}: {e}")

    base_image_dir_cont = os.path.join(base_dir_cont, "images", dataset_type)
    base_label_dir_cont = os.path.join(base_dir_cont, "labels", dataset_type)
    os.makedirs(base_image_dir_cont, exist_ok=True)
    os.makedirs(base_label_dir_cont, exist_ok=True)
    output_image_path_cont = os.path.join(base_image_dir_cont, png_filename)

    try:
        with fits.open(cont_path) as img_fit:
            cont_data = img_fit[1].data
            header = img_fit[1].header

        crota2 = header.get("CROTA2", 0)
        if crota2 != 0:
            cont_data = rotate(cont_data, crota2, reshape=False, mode="constant", cval=0)
        cont_data = normalize_continuum(cont_data)
        cont_data = ut_v.pad_resize_normalize(cont_data, target_height=target_height, target_width=target_width)
        cont_data = (cont_data * 255).astype(np.uint8)
        img = Image.fromarray(cont_data, mode="L")
        img.save(output_image_path_cont)
        with open(os.path.join(base_label_dir_cont, label_filename), "w") as label_file:
            label_file.write(label)

    except Exception as e:
        print(f"Error processing continuum {cont_path}: {e}")


def _class_names_from_yolo_config() -> list[str]:
    """Load class names from generated config.yaml; fall back to LABEL_MAPPING order."""
    config_path = Path(__file__).with_name("config.yaml")
    if config_path.exists():
        with open(config_path, encoding="utf-8") as stream:
            config_data = yaml.safe_load(stream) or {}
        names = config_data.get("names", {})
        if isinstance(names, dict):
            try:
                return [names[key] for key in sorted(names, key=lambda value: int(value))]
            except Exception:
                return [str(names[key]) for key in sorted(names)]

    fallback = sorted({value for value in cfg.LABEL_MAPPING.values() if value != "None"})
    return list(fallback)


def draw_yolo_labels_on_image(image_path, output_path=None):
    """Draw YOLO labels on an image using the matching label file."""
    class_names = _class_names_from_yolo_config()
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    label_path = image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    with open(label_path) as file:
        for line in file:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
            x1 = (x_center - bbox_width / 2) * width
            y1 = (y_center - bbox_height / 2) * height
            x2 = (x_center + bbox_width / 2) * width
            y2 = (y_center + bbox_height / 2) * height
            draw.rectangle([x1, y1, x2, y2], outline="orange", width=1)
            text_x = x1
            text_y = y1 - 28 if y1 - 28 > 0 else y1 + 2  # 28px above, or just below if too close to top
            class_index = int(class_id)
            class_label = class_names[class_index] if 0 <= class_index < len(class_names) else f"class_{class_index}"
            draw.text((text_x, text_y), class_label, fill="yellow", font=font)

    if output_path is not None:
        img.save(output_path)
    return img

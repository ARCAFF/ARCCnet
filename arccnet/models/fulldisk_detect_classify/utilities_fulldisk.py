import os
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from astropy.io import fits

class_dict = {
    'Alpha': 0,
    'Beta': 1,
    'Beta-Gamma': 2,
    'Beta-Delta': 3,
    'Beta-Gamma-Delta': 4
}

folder_path = '/ARCAFF/full_disk'
folder_fits = folder_path + '/fits'

def to_yolo(class_name, top_right, bottom_left, img_width, img_height):
    class_id = class_dict.get(class_name, -1)  # Returns -1 if class_name is not found
    x1, y1 = bottom_left
    x2, y2 = top_right
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return f"{class_id} {x_center} {y_center} {width} {height}"

def process_and_save_fits(dataframe, base_dir, dataset_type, resize_dim=(640, 640)):
    """
    Processes FITS images and YOLO labels from a given DataFrame, 
    resizes images to a specified dimension, and saves them in specified directories
    following a structure suited for YOLO training and validation.

    Parameters:
    - dataframe: The DataFrame containing the 'processed_path' and 'yolo_label' columns.
    - base_dir: The base directory where the dataset structure will be created.
    - dataset_type: A string specifying the type of dataset ('train' or 'val').
    - resize_dim: A tuple specifying the width and height (width, height) to resize the images to. 
        Default is (640, 640), needed for using pretrained models.

    YOLO folder structure:
    dataset/
    │
    ├── images/
    │   ├── train/
    │   └── val/
    │
    └── labels/
        ├── train/
        └── val/
    """

    # Create directories for images and labels according to the dataset type (train or val)
    base_image_dir = os.path.join(base_dir, 'images', dataset_type)
    base_label_dir = os.path.join(base_dir, 'labels', dataset_type)
    os.makedirs(base_image_dir, exist_ok=True)
    os.makedirs(base_label_dir, exist_ok=True)

    for index, row in dataframe.iterrows():
        current_image_path = folder_path + '/' + row['processed_path']
        label = row['yolo_label']
        
        # Process FITS file
        with fits.open(current_image_path) as image:
            image.verify('fix')
            data = image[1].data  # Adjust as necessary
            
            # Normalize and scale the image data
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            data = (data * 255).astype(np.uint8)
            
            # Convert to PIL Image and resize
            img = Image.fromarray(data)
            img_resized = img.resize(resize_dim)
            
            # Save the resized image as PNG
            basename = os.path.basename(current_image_path)
            png_filename = os.path.splitext(basename)[0] + '.png'
            img_resized.save(os.path.join(base_image_dir, png_filename))
            
            # Save the YOLO label in a .txt file
            label_filename = os.path.splitext(basename)[0] + '.txt'
            with open(os.path.join(base_label_dir, label_filename), 'w') as label_file:
                label_file.write(label)

def draw_yolo_labels_on_image(image_path, output_path=None):
    """
    Draws YOLO labels on the image by finding the corresponding label file,
    which assumes the label file is in a parallel 'labels' directory and has 
    the same base name as the image file but with a '.txt' extension.
    
    Parameters:
    - image_path: Path to the input image.
    - output_path: Path where the output image will be saved. 
                   If None, display the image.
    """
    class_names = [name for name, _ in sorted(class_dict.items(), key=lambda item: item[1])]

    # Load the image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Normalize the image path and replace 'images' with 'labels', change the extension to .txt
    label_path = image_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'

    # Read the label file
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
            
            # Convert YOLO coordinates to PIL rectangle format
            x1 = (x_center - bbox_width / 2) * width
            y1 = (y_center - bbox_height / 2) * height
            x2 = (x_center + bbox_width / 2) * width
            y2 = (y_center + bbox_height / 2) * height
            
            # Draw the bounding box rectangle and the label
            draw.rectangle([x1, y1, x2, y2], outline="orange", width=1)
            draw.text((x1, y1), class_names[int(class_id)], fill="yellow")

    # Save or show the output image
    if output_path:
        img.save(output_path)
    else:
        img.show()

# Data Handling
import numpy as np
import pandas as pd
import io

# Images
from skimage.transform import resize
from skimage import data, color
import matplotlib.pyplot as plt

# image processing

def add_padding_and_resize(image, target_height=224, target_width=224):
    # Convert grayscale images to RGB (if necessary)
    if len(image.shape) < 3:
        image = color.gray2rgb(image)
    
    # Calculate the aspect ratio of the target and original images
    original_height, original_width = image.shape[:2]
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height
    
    # Determine padding
    if original_aspect > target_aspect:
        # Image is wider than the target aspect ratio
        new_width = original_width
        new_height = int(original_width / target_aspect)
        padding_vertical = (new_height - original_height) // 2
        padding_horizontal = 0
    else:
        # Image is taller than the target aspect ratio
        new_height = original_height
        new_width = int(original_height * target_aspect)
        padding_horizontal = (new_width - original_width) // 2
        padding_vertical = 0
    
    # Add padding to the original image
    padded_image = np.pad(image, ((padding_vertical, padding_vertical), 
                                  (padding_horizontal, padding_horizontal), 
                                  (0, 0)), 
                          'constant', constant_values=np.min(image))
    
    # Resize the padded image to the target size
    resized_image = resize(padded_image, (target_height, target_width), anti_aliasing=True)
    
    # Normalize the pixel values dynamically based on the actual min and max
    min_val = resized_image.min()
    max_val = resized_image.max()
    if max_val - min_val > 0:  # Avoid division by zero
        rescaled_image = (resized_image - min_val) / (max_val - min_val)
    else:
        rescaled_image = resized_image - min_val
    
    return rescaled_image

def compute_scores(TN, FP, FN, TP):
    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2
    
    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))
    
    if TP+FP+FN==0:
        CSI = 0
    else:
        CSI = TP/(TP+FP+FN)
        

    if (TP + FN) == 0.:
        if TP == 0.:
            recall = 0  # float('NaN')
        else:
            recall = -100  # float('Inf')
    else:
        recall = TP / (TP + FN)


    if (TP + FP) == 0.:
        if TP == 0.:
            precision = 0.  # float('NaN')
        else:
            precision = -100  # float('Inf')
    else:
        precision = TP / (TP + FP)
        
    if (TN + FP) == 0.:
        if TN == 0.:
            spec = 0.  # float('NaN')
        else:
            spec = -100  # float('Inf')
    else:
        spec = TN / (TN + FP)
        
    acc = (TP+TN)/(TN+FP+FN+TP)
    
    if precision+recall == 0:
        if precision== 0 or recall==0:
            f1s=0
        else:
            f1s=-100
    else:
        f1s =  2.*(precision*recall)/(precision + recall)
        
    balanced_acc = (1/2)*(recall + spec)

    
    return tss, hss, CSI, recall, precision, spec, acc, f1s, balanced_acc
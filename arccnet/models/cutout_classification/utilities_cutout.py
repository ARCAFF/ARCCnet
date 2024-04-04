# Data Handling
import numpy as np

# Images
from skimage import color, transform


def pad_resize_normalize(image, target_height=224, target_width=224, to_RGB=False, normalize=True):
    """
    Adds padding to and resizes an image to a specified target height and width.

    This function first checks if the image needs to be converted from grayscale to RGB,
    based on the `to_RGB` flag. It then calculates the necessary padding to maintain the
    aspect ratio of the image when resizing it to the target dimensions. The image is padded
    with the minimum pixel value found in the image.
    After padding, the image is resized to the target dimensions, and the pixel values are
    normalized to a range between 0 and 1.
    If the `normalize` flag is set to True, the pixel values are normalized between 0 and 1.

    Parameters:
    - image (ndarray): The input image to be processed, can be in grayscale or RGB format.
    - target_height (int, optional): The target height of the image after resizing. Defaults to 224.
    - target_width (int, optional): The target width of the image after resizing. Defaults to 224.
    - to_RGB (bool, optional): Flag indicating whether to convert a grayscale image to RGB format. Defaults to False.
    - normalize (bool, optional): Flag indicating whether to normalize pixel values to the range 0-1. Defaults to True.

    Returns:
    - ndarray: The processed image resized to the target dimensions with padding added as necessary,
               optionally converted to RGB (if applicable), and with pixel values normalized (if enabled).
    """
    # Convert grayscale images to RGB (if necessary)
    if to_RGB and image.ndim == 2:  # Check if image is grayscale
        image = color.gray2rgb(image)

    original_height, original_width = image.shape[:2]
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

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

    # Adjust padding based on the image's dimensions
    if image.ndim == 3:  # Color image
        padding = ((padding_vertical, padding_vertical), (padding_horizontal, padding_horizontal), (0, 0))
    else:  # Grayscale image
        padding = ((padding_vertical, padding_vertical), (padding_horizontal, padding_horizontal))

    padded_image = np.pad(image, padding, "constant", constant_values=np.min(image))

    # Resize the padded image to the target size
    resized_image = transform.resize(padded_image, (target_height, target_width), anti_aliasing=True)

    # Normalize the pixel values
    if normalize:
        min_val, max_val = resized_image.min(), resized_image.max()
        if max_val > min_val:  # Avoid division by zero
            resized_image = (resized_image - min_val) / (max_val - min_val)
        else:
            resized_image = resized_image - min_val

    return resized_image


def compute_scores(TN, FP, FN, TP):
    """
    This function computes several metrics for binary classification.

    Parameters:
    - TN (float): True Negatives  - the number of correctly identified negative samples.
    - FP (float): False Positives - the number of negative samples incorrectly identified as positive.
    - FN (float): False Negatives - the number of positive samples incorrectly identified as negative.
    - TP (float): True Positives  - the number of correctly identified positive samples.

    Returns:
    - tuple: A tuple containing the computed metrics in the following order:
        - tss (float): True Skill Statistics, a measure of the ability to avoid false classification.
        - hss (float): Heidke Skill Score, comparing the observed accuracy with that of random chance.
        - CSI (float): Critical Success Index, also known as Threat Score, measuring the ratio of
          correctly predicted positive observations to the total predicted and actual positives.
        - recall (float): The proportion of actual positives correctly identified.
        - precision (float): The proportion of positive identifications that were actually correct.
        - spec (float): Specificity, the proportion of actual negatives correctly identified.
        - acc (float): Accuracy, the proportion of true results (both true positives and true negatives)
          in the population.
        - f1s (float): F1 score, the harmonic mean of precision and recall.
        - balanced_acc (float): Balanced accuracy, the average of recall and specificity.
    """
    if TP + FN == 0.0:
        if TP == 0.0:
            tss_aux1 = 0.0  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = TP / (TP + FN)

    if (FP + TN) == 0.0:
        if FP == 0.0:
            tss_aux2 = 0.0  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = FP / (FP + TN)

    tss = tss_aux1 - tss_aux2

    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.0:
        if (TP * TN - FN * FP) == 0:
            hss = 0.0  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))

    if TP + FP + FN == 0:
        CSI = 0
    else:
        CSI = TP / (TP + FP + FN)

    if (TP + FN) == 0.0:
        if TP == 0.0:
            recall = 0  # float('NaN')
        else:
            recall = -100  # float('Inf')
    else:
        recall = TP / (TP + FN)

    if (TP + FP) == 0.0:
        if TP == 0.0:
            precision = 0.0  # float('NaN')
        else:
            precision = -100  # float('Inf')
    else:
        precision = TP / (TP + FP)

    if (TN + FP) == 0.0:
        if TN == 0.0:
            spec = 0.0  # float('NaN')
        else:
            spec = -100  # float('Inf')
    else:
        spec = TN / (TN + FP)

    acc = (TP + TN) / (TN + FP + FN + TP)

    if precision + recall == 0:
        if precision == 0 or recall == 0:
            f1s = 0
        else:
            f1s = -100
    else:
        f1s = 2.0 * (precision * recall) / (precision + recall)

    balanced_acc = (1 / 2) * (recall + spec)

    return tss, hss, CSI, recall, precision, spec, acc, f1s, balanced_acc


def count_and_check_bars(matrix):
    """
    Determines whether a given matrix has consecutive lines (rows)
    that are entirely composed of 0 values and counts these lines.

    Returns:
    - tuple (bool, int): A tuple where the first element is a boolean indicating whether
        at least one instance of consecutive zero lines was found, and the second element
        is the total count of lines that are part of such consecutive zero sequences.
    """
    count = 0  # Initialize count of consecutive zero lines
    found_consecutive_zeros = False  # Flag to indicate if consecutive zero lines are found

    for i in range(len(matrix) - 1):
        if all(value == 0 for value in matrix[i]) and all(value == 0 for value in matrix[i + 1]):
            if not found_consecutive_zeros:  # If it's the first occurrence, count both lines
                count += 2
                found_consecutive_zeros = True
            else:  # For additional consecutive lines, count only the next line
                count += 1
    return found_consecutive_zeros, count


def remove_bars(X_train, Y_train):
    """
    Filters matrices and their corresponding labels if the number of
    consecutive zero lines greater than 5, so images with bars are removed.

    Parameters:
    - X_train: numpy.ndarray, set of matrices.
    - Y_train: numpy.ndarray or list, the labels corresponding to each matrix.

    Returns:
    - X_train_filtered: numpy.ndarray, filtered training data.
    - Y_train_filtered: numpy.ndarray, filtered labels.
    """
    valid_matrices_labels = [
        (matrix, Y_train[i]) for i, matrix in enumerate(X_train) if count_and_check_bars(matrix)[1] <= 5
    ]

    # If no valid matrices are found, return empty arrays
    if not valid_matrices_labels:
        return np.array([]), np.array([])

    valid_matrices, valid_labels = zip(*valid_matrices_labels)

    X_train_filtered = np.array(valid_matrices)
    Y_train_filtered = np.array(valid_labels)

    return X_train_filtered, Y_train_filtered


def augment_data(X_train, Y_train):
    """
    Augments the training data by flipping images horizontally, vertically, and both ways.
    """
    labels = ["Alpha", "Beta-Gamma", "Beta-Gamma-Delta", "Beta-Delta"]
    # Initialize an empty list to hold the augmented images and their labels
    augmented_images = []
    augmented_labels = []

    # Since Beta images are more, just perform one flip
    indices = np.where(Y_train == "Beta")[0]
    label_images = X_train[indices]
    label_horverflip = label_images[:, ::-1, ::-1]

    # Concatenate the flipped images
    label_augmented = np.concatenate([label_horverflip], axis=0)

    augmented_images.append(label_augmented)
    label_augmented_labels = np.array(["Beta"] * label_augmented.shape[0])
    augmented_labels.append(label_augmented_labels)

    for label in labels:
        indices = np.where(Y_train == label)[0]
        label_images = X_train[indices]
        # horizontal, vertical and both ways flip
        label_horflip = label_images[:, :, ::-1]
        label_verflip = label_images[:, ::-1, :]
        label_horverflip = label_images[:, ::-1, ::-1]

        label_augmented = np.concatenate([label_horflip, label_verflip, label_horverflip], axis=0)

        augmented_images.append(label_augmented)

        label_augmented_labels = np.array([label] * label_augmented.shape[0])

        augmented_labels.append(label_augmented_labels)

    all_augmented_images = np.concatenate(augmented_images, axis=0)
    all_augmented_labels = np.concatenate(augmented_labels, axis=0)

    X_train = np.concatenate([X_train, all_augmented_images], axis=0)
    Y_train = np.concatenate([Y_train, all_augmented_labels], axis=0)

    return X_train, Y_train

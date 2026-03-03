import os

import matplotlib  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage import transform
from sunpy.visualization import colormaps as cm  # noqa: F401
from torchvision.transforms.functional import to_tensor

from arccnet.models import labels

magnetic_map = matplotlib.colormaps["hmimag"]


def create_solar_grid(ax, num_meridians=12, num_parallels=12, num_points=300):
    """
    Add meridian and parallel grid lines to a solar disc plot.
    """
    phis = np.linspace(0, 2 * np.pi, num_meridians, endpoint=False)
    lats = np.linspace(-np.pi / 2, np.pi / 2, num_parallels)
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)

    # Meridians
    for phi in phis:
        y, z = np.cos(theta) * np.sin(phi), np.sin(theta)
        ax.plot(y, z, "k-", linewidth=0.2)

    # Parallels
    for lat in lats:
        y = np.cos(lat) * np.sin(theta)
        z = np.full(num_points, np.sin(lat))
        ax.plot(y, z, "k-", linewidth=0.2)


def pad_resize_normalize(image, target_height=224, target_width=224):
    """
    Adds padding to and resizes an image to specified target height and width.
    The function maintains the aspect ratio of the image by calculating the necessary padding.
    The image is padded with a constant value (default is 0.0) and then resized to the target dimensions.

    Parameters:
    - image (ndarray): The input image to be processed, can be in grayscale or RGB format.
    - target_height (int, optional): The target height of the image after resizing. Defaults to 224.
    - target_width (int, optional): The target width of the image after resizing. Defaults to 224.

    Returns:
    - ndarray: The processed image resized to the target dimensions with padding added as necessary.
    """

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

    padded_image = np.pad(image, padding, "constant", constant_values=0.0)

    # Resize the padded image to the target size
    resized_image = transform.resize(padded_image, (target_height, target_width), anti_aliasing=True)

    return resized_image


def make_classes_histogram(
    series,
    horizontal=False,
    figsz=(13, 5),
    y_off=None,
    ylim=None,
    title=None,
    ylabel="n° of ARs",
    titlesize=14,
    x_rotation=0,
    fontsize=11,
    bar_color="#4C72B0",
    edgecolor="black",
    text_fontsize=11,
    style="seaborn-v0_8-darkgrid",
    show_percentages=True,
    ax=None,
    save_path=None,
    transparent=False,
    categories=None,
):
    """
    Creates and displays a bar chart (histogram) that visualizes the distribution of classes in a given pandas Series.

    Parameters:
    - series (pandas.Series):
        The input series containing the class labels.
    - horizontal (bool, optional):
        Whether to create a horizontal bar chart. Default is False.
    - figsz (tuple, optional):
        A tuple representing the size of the figure (width, height) in inches.
        Default is (13, 5) for vertical, but consider (10, 16) for horizontal with many classes.
    - y_off (int or float, optional):
        The offset for the text labels. For vertical charts, this is the vertical offset;
        for horizontal charts, this is the horizontal offset. If None, defaults to 300 for vertical
        and 0.5 for horizontal. Default is None.
    - ylim (int or float, optional):
        The maximum value for the y-axis (vertical) or x-axis (horizontal). Default is None.
    - title (str, optional):
        The title of the histogram plot. If `None`, no title will be displayed.
        Default is None.
    - ylabel (str, optional):
        The label for the count axis. For vertical charts, this is the y-axis label;
        for horizontal charts, this is the x-axis label. Default is "n° of ARs".
    - titlesize (int, optional):
        The font size of the title text. Ignored if `title` is `None`.
        Default is 14.
    - x_rotation (int, optional):
        The rotation angle for the x-axis labels (vertical) or y-axis labels (horizontal).
        Default is 0.
    - fontsize (int, optional):
        The font size of the axis labels. Default is 11.
    - bar_color (str, optional):
        The color of the bars in the histogram. Default is '#4C72B0'.
    - edgecolor (str, optional):
        The color of the edges of the bars. Default is 'black'.
    - text_fontsize (int, optional):
        The font size of the text displayed above the bars. Default is 11.
    - style (str, optional):
        The matplotlib style to be used for the plot. Default is 'seaborn-v0_8-darkgrid'.
    - show_percentages (bool, optional):
        Whether to display percentages on top of the bars. Default is True.
    - ax (matplotlib.axes.Axes, optional):
        An existing matplotlib Axes object to plot on. If `None`, a new figure and Axes will be created.
        Default is None.
    - save_path (str, optional):
        Path to save the figure. If `None`, the plot will be displayed instead of saved.
        Default is None.
    - transparent (bool, optional):
        Whether to save the figure with a transparent background. Default is False.
    - categories (list, optional):
        Explicit list of categories in desired order. If provided, will use this order instead of sorting.
        Default is None (will sort alphabetically).
    """
    # Determine the default y_off based on horizontal
    if y_off is None:
        y_off = 0.5 if horizontal else 300

    # Process class names and counts
    if categories is not None:
        # Use provided categories in the given order
        classes_names = [c for c in categories if c in series.unique()]
        counts = series.value_counts().reindex(classes_names, fill_value=0)
        values = counts.values
    else:
        # Filter out None and sort based on horizontal flag (original behavior)
        classes_names = sorted(filter(lambda x: x is not None, series.unique()), reverse=horizontal)
        if horizontal:
            counts = series.value_counts().reindex(classes_names, fill_value=0)
            classes_names = counts.index.tolist()
            values = counts.values
        else:
            classes_counts = series.value_counts().reindex(classes_names)
            values = classes_counts.values

    total = np.sum(values)
    greek_labels = labels.convert_to_greek_label(classes_names)  # Ensure this is defined or imported

    with plt.style.context(style):
        # Create figure and axes if not provided
        if ax is None:
            plt.figure(figsize=figsz)
            ax = plt.gca()

        # Create bars
        if horizontal:
            bars = ax.barh(greek_labels, values, color=bar_color, edgecolor=edgecolor)
        else:
            bars = ax.bar(greek_labels, values, color=bar_color, edgecolor=edgecolor)

        # Annotate bars with values and percentages
        for bar in bars:
            if horizontal:
                value = bar.get_width()
                y_center = bar.get_y() + bar.get_height() / 2
                text_x = value + y_off
                text_y = y_center
                ha = "left"
                va = "center"
            else:
                value = bar.get_height()
                text_x = bar.get_x() + bar.get_width() / 2
                text_y = value + y_off
                ha = "center"
                va = "bottom"

            if show_percentages:
                percentage = f"{value / total * 100:.2f}%" if total > 0 else "0.00%"
                text = f"{value} ({percentage})"
            else:
                text = f"{value}"

            ax.text(text_x, text_y, text, ha=ha, va=va, fontsize=text_fontsize)

        # Set axis labels and ticks
        if horizontal:
            ax.set_yticks(np.arange(len(greek_labels)))
            ax.set_yticklabels(greek_labels, rotation=x_rotation, ha="right", fontsize=fontsize)
            ax.set_xlabel(ylabel, fontsize=fontsize)
            ax.set_ylabel("Class", fontsize=fontsize)
            if ylim is not None:
                ax.set_xlim([0, ylim])
        else:
            ax.set_xticks(np.arange(len(greek_labels)))
            ax.set_xticklabels(greek_labels, rotation=x_rotation, ha="center", fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            if ylim is not None:
                ax.set_ylim([0, ylim])

        # Set title
        if title:
            ax.set_title(title, fontsize=titlesize)

        # Set tick parameters for the other axis
        if horizontal:
            ax.tick_params(axis="x", labelsize=fontsize)
        else:
            ax.tick_params(axis="y", labelsize=fontsize)

        # Save or show the plot if ax is None
        if ax is None:
            if save_path:
                plt.savefig(save_path, bbox_inches="tight", transparent=transparent)
                plt.close()
            else:
                plt.show()

    return ax


class HardTanhTransform:
    """
    This transformation first scales the input image tensor by a specified divisor and then clamps the resulting values
    using the HardTanh function, which limits the values to a specified range [min_val, max_val].

    Attributes:
    - divisor (float):
        The value by which to divide the image tensor.
        This scaling factor is applied to the image before the HardTanh operation. Default is 800.0.
    - min_val (float):
        The minimum value to clamp the image tensor to using the HardTanh function. Default is -1.0.
    - max_val (float):
        The maximum value to clamp the image tensor to using the HardTanh function. Default is 1.0.

    Methods:
    - __call__(img):
        Applies the scaling and HardTanh transformation to the input image.

        Parameters:
        - img (PIL.Image or torch.Tensor):
            The input image, which can be either a PIL image or a PyTorch tensor.
            If a PIL image is provided, it will be converted to a tensor.

        Returns:
        - torch.Tensor:
            The transformed image tensor, scaled by the divisor and clamped to the range [min_val, max_val].
    """

    def __init__(self, divisor=800.0, min_val=-1.0, max_val=1.0):
        self.divisor = divisor
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        # Convert image to tensor if it's not already one
        if not torch.is_tensor(img):
            img = to_tensor(img)

        # Scale by the divisor and apply hardtanh
        img = img / self.divisor
        img = F.hardtanh(img, min_val=self.min_val, max_val=self.max_val)
        return img


def hardtanh_transform_npy(img, divisor=800.0, min_val=-1.0, max_val=1.0):
    """
    Apply HardTanh transformation to the input image.

    Args:
    - img: Input image (numpy array).
    - divisor: Value to divide the input image by.
    - min_val: Minimum value for the HardTanh function.
    - max_val: Maximum value for the HardTanh function.

    Returns:
    - Transformed image.
    """
    # Ensure the input is a NumPy array
    if not isinstance(img, np.ndarray):
        raise TypeError("Input should be a NumPy array")

    # Scale by the divisor
    img = img / divisor

    # Apply hardtanh
    img = np.clip(img, min_val, max_val)
    return img


def visualize_transformations(images, transforms, n_samples=16):
    """
    Visualize the effect of transformations on a set of images.

    Parameters:
    images (numpy.ndarray): Array of images to be transformed and visualized.
                            The shape should be (n_images, height, width).
    transforms (torchvision.transforms.Compose): The transformations to be applied to the images.
    n_samples (int): The number of sample images to visualize. Default is 16.

    Returns:
    None: Displays a plot with the transformed images in a 4x4 grid.
    """
    plt.figure(figsize=(12, 12))

    for i in range(n_samples):
        original_image = images[i]
        original_image_tensor = torch.from_numpy(original_image).float().unsqueeze(0)
        transformed_image_tensor = transforms(original_image_tensor)
        transformed_image = transformed_image_tensor.squeeze().numpy()

        # Plot transformed image
        plt.subplot(4, 4, i + 1)
        plt.imshow(transformed_image, cmap=magnetic_map, vmin=-1, vmax=1)
        plt.title(f"Transformed Image {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_location_on_sun(df, long_limit_deg=60, experiment=None):
    """
    Analyze and plot the distribution of solar cutouts based on their longitude.

    This function filters cutouts based on longitude, categorizes them as 'front' or 'rear',
    and plots their positiosns on a solar disc.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the solar data with latitude and longitude information.
    - long_limit_deg (int, optional): The longitude limit in degrees to determine front vs. rear. Default is 60 degrees.
    """
    latV = np.deg2rad(np.where(df["processed_path_image_hmi"] != "", df["latitude_hmi"], df["latitude_mdi"]))
    lonV = np.deg2rad(np.where(df["processed_path_image_hmi"] != "", df["longitude_hmi"], df["longitude_mdi"]))

    yV = np.cos(latV) * np.sin(lonV)
    zV = np.sin(latV)

    condition = (lonV < -np.rad2deg(long_limit_deg)) | (lonV > np.rad2deg(long_limit_deg))

    rear_latV = latV[condition]
    lonV[condition]
    rear_yV = yV[condition]
    rear_zV = zV[condition]

    front_latV = latV[~condition]
    lonV[~condition]
    front_yV = yV[~condition]
    front_zV = zV[~condition]

    # Plot ARs' location on the solar disc
    circle = plt.Circle((0, 0), 1, edgecolor="gray", facecolor="none")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_artist(circle)

    num_meridians = 12
    num_parallels = 12
    num_points = 300

    phis = np.linspace(0, 2 * np.pi, num_meridians, endpoint=False)
    lats = np.linspace(-np.pi / 2, np.pi / 2, num_parallels)
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)

    # Plot each meridian
    for phi in phis:
        y = np.cos(theta) * np.sin(phi)
        z = np.sin(theta)
        ax.plot(y, z, "k-", linewidth=0.2)

    # Plot each parallel
    for lat in lats:
        radius = np.cos(lat)
        y = radius * np.sin(theta)
        z = np.sin(lat) * np.ones(num_points)
        ax.plot(y, z, "k-", linewidth=0.2)

    ax.scatter(rear_yV, rear_zV, s=1, alpha=0.2, color="r", label="Rear")
    ax.scatter(front_yV, front_zV, s=1, alpha=0.2, color="b", label="Front")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=12)

    # Save the plot

    num_rear_cutouts = len(rear_latV)
    num_front_cutouts = len(front_latV)
    percentage_rear = 100 * num_rear_cutouts / (num_rear_cutouts + num_front_cutouts)

    text_output = f"Rear: {num_rear_cutouts}\nFront: {num_front_cutouts}\nPercentage of Rear: {percentage_rear:.2f}%"

    if experiment:
        plot_path = os.path.join("temp", "solar_disc_plot.png")
        plt.savefig(plot_path)
        experiment.log_image(plot_path, name="Solar Disc Plot")
        experiment.log_text(text_output, metadata={"description": "Solar cutouts analysis"})

    print(text_output)
    plt.show()

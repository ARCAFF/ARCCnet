import os

import matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import sunpy.map
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from p_tqdm import p_map
from skimage import transform
from sunpy.visualization import colormaps as cm  # noqa: F401
from torchvision.transforms.functional import to_tensor

import astropy.units as u
from astropy.io import fits

from arccnet.models import labels

magnetic_map = matplotlib.colormaps["hmimag"]


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
    """
    # Determine the default y_off based on horizontal
    if y_off is None:
        y_off = 0.5 if horizontal else 300

    # Process class names and counts
    # Filter out None and sort based on horizontal flag
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
        plt.title(f"Transformed Image {i+1}")
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

    text_output = (
        f"Rear: {num_rear_cutouts}\n" f"Front: {num_front_cutouts}\n" f"Percentage of Rear: {percentage_rear:.2f}%"
    )

    if experiment:
        plot_path = os.path.join("temp", "solar_disc_plot.png")
        plt.savefig(plot_path)
        experiment.log_image(plot_path, name="Solar Disc Plot")
        experiment.log_text(text_output, metadata={"description": "Solar cutouts analysis"})

    print(text_output)
    plt.show()


def months_years_heatmap(df, datetime_column, title, colorbar_title, height=900, width=650):
    """
    Create a heatmap showing the number of images per month and year.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to plot. Must contain a datetime column to group by.
    datetime_column : str
        The name of the datetime column in `df` to use for grouping by year and month.
    title : str
        The title of the heatmap.
    colorbar_title : str
        The title for the colorbar, indicating what the heatmap values represent.
    height : int, optional
        The height of the figure in pixels. Default is 900.
    width : int, optional
        The width of the figure in pixels. Default is 650.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated Plotly heatmap figure object.

    Notes
    -----
    This function creates a heatmap where the rows represent the years and the columns represent the months,
    showing the number of occurrences of a particular event (e.g., images) in each month-year combination.

    Examples
    --------
    >>> fig = months_years_heatmap(selected_df, 'datetime', 'Number of Events per Month and Year', 'Number of Events')
    >>> fig.show()
    """

    grouped_df = (
        df.groupby([df[datetime_column].dt.year.rename("year"), df[datetime_column].dt.strftime("%b").rename("month")])
        .size()
        .reset_index(name="count")
    )

    # Create a pivot table where rows are years, columns are months, and values are counts
    pivot_table = grouped_df.pivot(index="year", columns="month", values="count").fillna(0)

    # Ensure correct order of months
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_table = pivot_table.reindex(columns=months_order)

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,  # Months on x-axis
            y=pivot_table.index,  # Years on y-axis
            colorscale="Portland",
            colorbar=dict(title=colorbar_title),
            text=pivot_table.values,
            texttemplate="%{text}",
            textfont={"size": 12, "color": "white"},
        )
    )

    # Update the layout
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Year",
        xaxis={"type": "category"},  # Ensure months are treated as categorical data
        yaxis={"type": "category"},  # Ensure years are treated as categorical data
        autosize=True,
        height=height,
        width=width,
    )

    return fig


def location_on_sun(df, fig=None, color="#1f77b4"):
    """
    Plot Active Regions (ARs) on a 2D representation of the Sun's disc.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing AR data with columns:
        - 'latitude' : float, Latitude values in degrees.
        - 'longitude' : float, Longitude values in degrees.
        - 'datetime' : datetime, Timestamp for each AR entry.
    fig : plotly.graph_objs.Figure, optional
        Existing Plotly Figure object to which AR points are added. If None, a new figure is created.
    color : str, optional
        Color of the AR markers on the plot. Default is '#1f77b4' (blue).

    Returns
    -------
    plotly.graph_objs.Figure
        Plotly Figure object containing the solar disc, meridians, parallels, and AR locations.

    Notes
    -----
    - This function creates a 2D projection of the Sun with optional longitude-based filtering.
    - If `fig` is provided, ARs from `df` are added to this figure. Otherwise, a new plot is created.
    - The plot includes interactive hover text for each AR with index and timestamp.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'latitude': [10, -30, 45],
    ...         'longitude': [40, 85, -60],
    ...         'datetime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])}
    >>> df = pd.DataFrame(data)
    >>> fig = location_on_sun(df)
    >>> fig.show()
    """
    # Create a new figure if one is not provided
    if fig is None:
        fig = go.Figure()

        # Plot solar disc
        theta = np.linspace(0, 2 * np.pi, 100)
        solar_disc_y = np.cos(theta)
        solar_disc_z = np.sin(theta)

        fig.add_trace(
            go.Scatter(x=solar_disc_y, y=solar_disc_z, mode="lines", line=dict(color="gray", width=1), showlegend=False)
        )

        # Add meridians and parallels
        num_meridians = 12
        num_parallels = 12
        num_points = 300

        phis = np.linspace(0, 2 * np.pi, num_meridians, endpoint=False)
        lats = np.linspace(-np.pi / 2, np.pi / 2, num_parallels)
        theta_meridian = np.linspace(-np.pi / 2, np.pi / 2, num_points)

        for phi in phis:
            y_meridian = np.cos(theta_meridian) * np.sin(phi)
            z_meridian = np.sin(theta_meridian)
            fig.add_trace(
                go.Scatter(
                    x=y_meridian, y=z_meridian, mode="lines", line=dict(color="black", width=0.2), showlegend=False
                )
            )

        for lat in lats:
            radius = np.cos(lat)
            y_parallel = radius * np.sin(theta_meridian)
            z_parallel = np.sin(lat) * np.ones(num_points)
            fig.add_trace(
                go.Scatter(
                    x=y_parallel, y=z_parallel, mode="lines", line=dict(color="black", width=0.2), showlegend=False
                )
            )

        fig.update_layout(
            title="ARs Location on the Sun",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            xaxis_range=[-1.1, 1.1],
            yaxis_range=[-1.1, 1.1],
            width=800,
            height=800,
            autosize=False,
            hovermode="closest",
            margin=dict(l=50, r=50, b=50, t=50),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        # Set equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

    # Convert latitude and longitude to radians
    latV = np.deg2rad(df["latitude"])
    lonV = np.deg2rad(df["longitude"])

    # Convert to y and z coordinates for the plot
    yV = np.cos(latV) * np.sin(lonV)
    zV = np.sin(latV)

    # Extract additional information for hover text
    indices = [j for j in range(len(df))]
    hover_text = [f"Index: {i}<br>Time: {time}" for i, time in zip(indices, df["datetime"])]

    # Add ARs locations with specified color
    fig.add_trace(
        go.Scatter(
            x=yV,
            y=zV,
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.7),
            text=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    return fig


def compute_widths_heights(df):
    """
    Compute normalized widths and heights of bounding boxes in solar images and retrieve corresponding magnetic class and datetime.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns 'instrument', 'bottom_left_cutout', 'top_right_cutout', 'magnetic_class', and 'datetime'.
        'instrument' specifies the instrument used (e.g., 'MDI' or 'HMI').
        'bottom_left_cutout' and 'top_right_cutout' are tuples representing the coordinates of the bounding box.

    Returns
    -------
    tuple of lists
        widths : list of float
            Normalized widths of the bounding boxes.
        heights : list of float
            Normalized heights of the bounding boxes.
        magnetic_classes : list of str
            Magnetic class labels for each bounding box.
        datetimes : list of datetime
            Datetime for each bounding box.

    Notes
    -----
    The normalization of widths and heights is based on the size of the instrument image, which is 1024 for 'MDI' and 4096 for 'HMI'.
    """
    img_size_dic = {"MDI": 1024, "HMI": 4096}

    def _process_row(row):
        x_min, y_min = row["bottom_left_cutout"]
        x_max, y_max = row["top_right_cutout"]

        img_sz = img_size_dic.get(row["instrument"])
        width = (x_max - x_min) / img_sz
        height = (y_max - y_min) / img_sz

        magnetic_class = row["magnetic_class"]
        datetime = row["datetime"]

        return width, height, magnetic_class, datetime

    results = p_map(_process_row, [row for _, row in df.iterrows()])
    widths, heights, magnetic_classes, datetimes = zip(*results)

    return widths, heights, magnetic_classes, datetimes


def w_h_scatterplot(df):
    """
    Create a scatter plot of normalized bounding box widths and heights, colored by magnetic class.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing bounding box details and magnetic classification.
        Requires 'instrument', 'bottom_left_cutout', 'top_right_cutout', 'magnetic_class', and 'datetime' columns.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly scatter plot showing normalized bounding box widths vs. heights, colored by magnetic class.
        The plot includes hover information with magnetic class, width, height, datetime, and index.

    Notes
    -----
    Each unique magnetic class is mapped to a distinct color. A legend entry is added for each magnetic class.
    """
    widths, heights, magnetic_classes, datetimes = compute_widths_heights(df)

    indices = [j for j in range(len(df))]
    unique_classes = list(set(magnetic_classes))

    # Updated color map to use a consistent color per class
    color_map = {
        cls: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        for i, cls in enumerate(sorted(unique_classes))
    }  # Map each class to a unique color

    # Assign colors based on the magnetic class
    colors = [color_map[cls] for cls in magnetic_classes]

    fig = go.Figure()

    # Add scatter trace for widths and heights, colored by magnetic class
    fig.add_trace(
        go.Scatter(
            x=widths,  # X-axis: width of bounding boxes
            y=heights,  # Y-axis: height of bounding boxes
            mode="markers",
            marker=dict(size=3, color=colors, opacity=0.7),  # Use categorical colors for magnetic class
            name="Bounding Box Dimensions",
            text=magnetic_classes,
            customdata=list(zip(datetimes, indices)),  # Custom data to include both datetime and index
            hovertemplate="<b>Class</b>: %{text}<br><b>Width</b>: %{x}<br><b>Height</b>: %{y}<br><b>Datetime</b>: %{customdata[0]}<br><b>Index</b>: %{customdata[1]}<extra></extra>",
        )
    )

    # Update the layout and add the legend for class labels
    fig.update_layout(
        title="Bounding Box Widths vs Heights",
        xaxis_title="Width (normalized)",
        yaxis_title="Height (normalized)",
        autosize=True,
        showlegend=True,  # Show legend
    )

    # Add class labels to the legend by adding a dummy scatter trace for each class
    for cls, color in color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                legendgroup=cls,
                showlegend=True,
                name=cls,  # Add class name to the legend
            )
        )

    return fig


def plot_fd(row, df, local_path_root):
    """
    Plot a full-disk solar image with active region bounding boxes and labels.

    This function takes a row representing a solar image, loads the image data,
    applies rotation based on the CROTA2 header value, masks regions outside the solar disk,
    and plots bounding boxes with magnetic class labels for identified active regions.

    Parameters
    ----------
    row : pandas.Series
        Row containing data for a specific solar image, including path to the FITS file.
    df : pandas.DataFrame
        DataFrame containing details of bounding boxes and magnetic classifications for all images.
        Requires columns 'path', 'bottom_left_cutout', 'top_right_cutout', and 'magnetic_class'.

    Notes
    -----
    - The image is rotated based on the CROTA2 header value to ensure proper alignment.
    - Pixels outside the solar disk are set to NaN and are not displayed in the plot.
    - Active region bounding boxes are drawn in red, with labels indicating their magnetic class.
    - This function requires `sunpy`, `astropy`, and `matplotlib` libraries to be installed.

    Returns
    -------
    None
        Displays a matplotlib plot with the full-disk solar image, active region bounding boxes,
        and corresponding magnetic class labels.
    """
    arccnet_path_root = row["path"].split("/fits")[0]
    image_path = row["path"].replace(arccnet_path_root, local_path_root)
    image_labels = df[df["path"] == row["path"]]

    with fits.open(image_path) as img_fit:
        data = img_fit[1].data
        header = img_fit[1].header

        sunpy_map = sunpy.map.Map(data, header)

        # Generate a grid of coordinates for each pixel
        x, y = np.meshgrid(np.arange(sunpy_map.data.shape[1]), np.arange(sunpy_map.data.shape[0]))
        coordinates = sunpy_map.pixel_to_world(x * u.pix, y * u.pix)

        # Check if the coordinates are on the solar disk
        on_disk = coordinates.separation(sunpy_map.reference_coordinate) <= sunpy.map.solar_angular_radius(coordinates)

        # Mask data that is outside the solar disk
        sunpy_map.data[~on_disk] = np.nan  # Set off-disk pixels to NaN

        # Extract CROTA2 value from the header
        crota2 = sunpy_map.meta.get("CROTA2", 0)  # Default to 0 if CROTA2 is not present

        # Apply the rotation based on the CROTA2 value
        rotated_map = sunpy_map.rotate(angle=-crota2 * u.deg)  # Apply the rotation -CROTA2 to align it correctly

        # Plot the rotated map
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection=rotated_map)

        # Plot the map with the adjusted rotation
        rotated_map.plot(axes=ax, cmap="hmimag")

        # Draw grid if needed
        rotated_map.draw_grid(axes=ax)

        for _, label_row in image_labels.iterrows():
            x_min, y_min = label_row["bottom_left_cutout"]
            x_max, y_max = label_row["top_right_cutout"]

            width = x_max - x_min
            height = y_max - y_min

            rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor="red", facecolor="none")  # (x, y)

            ax.add_patch(rect)

            label_text = label_row["magnetic_class"]
            center_x = (x_min + x_max) / 2

            ax.text(
                center_x,
                y_max + 5,
                label_text,
                color="white",
                fontsize=10,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.5),
            )

        plt.show()


def plot_confusion_matrix(cmc, labels, title, figsize=(10, 10), save_path=None):
    """
    Plots a confusion matrix with counts and percentages, and optionally saves the figure.

    Args:
        cmc (numpy.ndarray): Confusion matrix counts.
        labels (list): List of label names.
        title (str): Title of the plot.
        figsize (tuple): Figure size for the plot.
        save_path (str): Path to save the figure. If None, the figure will not be saved.
    """

    # Calculate the row sums
    row_sums = cmc.sum(axis=1, keepdims=True)

    # Avoid division by zero by setting row sums that are zero to one temporarily
    # This prevents NaNs in cm_percentage
    safe_row_sums = np.where(row_sums == 0, 1, row_sums)

    # Calculate the percentages
    cm_percentage = cmc / safe_row_sums * 100

    # For rows where the original sum was zero, set percentages to zero
    cm_percentage = np.where(row_sums == 0, 0, cm_percentage)

    # Create annotations with count and percentage
    annotations = np.empty_like(cmc).astype(str)

    for i in range(cmc.shape[0]):
        for j in range(cmc.shape[1]):
            annotations[i, j] = f"{cmc[i, j]}\n({cm_percentage[i, j]:.1f}%)"

    # Initialize the matplotlib figure
    plt.figure(figsize=figsize)

    # Create the heatmap
    sns.heatmap(
        cm_percentage,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        linewidths=0,
        linecolor="none",
    )

    # Set plot labels and title
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Improve layout
    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    # Show the plot
    plt.show()

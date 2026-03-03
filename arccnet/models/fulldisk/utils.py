"""Utilities for working with full-disk observations and annotations."""

from __future__ import annotations

from typing import Any
from pathlib import Path
from collections.abc import Mapping, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sunpy.map
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Patch, Rectangle
from p_tqdm import p_map

import astropy.units as u
from astropy.time import Time

from arccnet.models import preprocessing_common as pp_common
from arccnet.visualisation.utils import create_solar_grid

FitsPath = str | Path
LabelFormatter = Callable[[Any], str]
IMG_SIZE_BY_INSTRUMENT: dict[str, int] = {"MDI": 1024, "HMI": 4096}

CLASS_COLOR_MAP = {
    "IA": "#828282",
    "Alpha": "#00BFFF",  # bright sky blue
    "Beta": "#00B45A",  # bright spring green
    "Beta-Gamma": "#FFDD00",  # bright gold
    "Beta-Delta": "#FF69B4",  # hot pink
    "Beta-Gamma-Delta": "#FF4500",  # vivid orange-red
    "Gamma-Delta": "#8A2BE2",  # bright violet (blue-purple)
    "Gamma": "#1E90FF",  # dodger blue
}


def load_rotated_map(fits_path: FitsPath, mask_off_disk: bool = True) -> sunpy.map.Map:
    """Return a rotated SunPy map, optionally masking off-disk pixels."""
    sun_map = sunpy.map.Map(fits_path)

    if mask_off_disk:
        x_pix, y_pix = np.meshgrid(np.arange(sun_map.data.shape[1]), np.arange(sun_map.data.shape[0]))
        coords = sun_map.pixel_to_world(x_pix * u.pix, y_pix * u.pix)
        mask = coords.separation(sun_map.reference_coordinate) > sunpy.map.solar_angular_radius(coords)
        sun_map.data[mask] = np.nan

    crota2 = sun_map.meta.get("CROTA2", 0)
    return sun_map.rotate(angle=-crota2 * u.deg)


def format_mag_class(mag_class: Any) -> str:
    """Map 0.0 to IA while returning other labels as strings."""
    return "IA" if str(mag_class) == "0.0" else str(mag_class)


def draw_bounding_boxes(
    ax: Axes,
    image_labels: pd.DataFrame,
    color_map: Mapping[Any, str],
    label_formatter: LabelFormatter = format_mag_class,
    text_offset: float = 5.0,
) -> None:
    """Draw class-coloured bounding boxes and labels."""
    for _, label_row in image_labels.iterrows():
        x_min, y_min = label_row["bottom_left_cutout"]
        x_max, y_max = label_row["top_right_cutout"]

        cls_key = str(label_row["magnetic_class"])
        colour = color_map.get(cls_key, "red")
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=colour,
                facecolor="none",
            )
        )
        ax.text(
            (x_min + x_max) / 2,
            y_max + text_offset,
            label_formatter(label_row["magnetic_class"]),
            color="white",
            fontsize=10,
            ha="center",
            va="bottom",
            bbox=dict(facecolor=colour, alpha=0.6),
        )


def add_class_legend(
    fig: Figure,
    color_map: Mapping[Any, str],
    label_formatter: LabelFormatter = format_mag_class,
    **legend_kwargs: Any,
) -> None:
    """Attach a legend describing the magnetic class colours."""
    handles = [
        Patch(facecolor=color, edgecolor=color, label=label_formatter(label))
        for label, color in sorted(color_map.items())
    ]
    if handles:
        fig.legend(handles=handles, **legend_kwargs)


def plot_full_disk_pair(
    fits_magnetogram: FitsPath,
    fits_continuum: FitsPath,
    image_labels: pd.DataFrame,
    color_map: Mapping[Any, str] | None = None,
    label_formatter: LabelFormatter = format_mag_class,
    figsize: tuple[int, int] = (16, 8),
    cmap_magnetogram: str = "hmimag",
    cmap_continuum: str = "gray",
) -> tuple[Figure, tuple[Axes, Axes], dict[Any, str]]:
    """Plot magnetogram and continuum maps with bounding boxes."""
    color_map = dict(color_map or CLASS_COLOR_MAP)
    maps = (load_rotated_map(fits_magnetogram), load_rotated_map(fits_continuum))
    titles = ("Magnetogram", "Continuum")
    cmaps = (cmap_magnetogram, cmap_continuum)

    fig = plt.figure(figsize=figsize)
    axes = [fig.add_subplot(1, 2, idx + 1, projection=map_obj) for idx, map_obj in enumerate(maps)]

    for ax, map_obj, title, cmap in zip(axes, maps, titles, cmaps):
        map_obj.plot(axes=ax, cmap=cmap)
        ax.set_title(title)
        map_obj.draw_grid(axes=ax)
        draw_bounding_boxes(ax, image_labels, color_map, label_formatter=label_formatter)

    return fig, (axes[0], axes[1]), color_map


def plot_ar_locations(
    df: pd.DataFrame,
    ax: Axes | None = None,
    color: str = "#1f77b4",
    label: str | None = None,
    marker_size: float = 10.0,
    alpha: float = 0.7,
) -> Axes:
    """Scatter AR locations on a solar disc."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if not getattr(ax, "_solar_grid_drawn", False):
        create_solar_grid(ax)
        solar_disc = Circle((0, 0), radius=1.0, edgecolor="gray", facecolor="none", linewidth=1)
        ax.add_patch(solar_disc)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        ax._solar_grid_drawn = True  # type: ignore[attr-defined]

    lat_rad = np.deg2rad(df["latitude"].to_numpy())
    lon_rad = np.deg2rad(df["longitude"].to_numpy())
    x_coords = np.cos(lat_rad) * np.sin(lon_rad)
    y_coords = np.sin(lat_rad)

    ax.scatter(x_coords, y_coords, s=marker_size, color=color, alpha=alpha, label=label)
    return ax


def compute_widths_heights(
    df: pd.DataFrame,
    size_map: Mapping[str, int] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[Any, ...], tuple[pd.Timestamp, ...]]:
    """Return normalised bounding-box dimensions with associated class and datetime."""
    if df.empty:
        return (), (), (), ()

    size_map = dict(size_map or IMG_SIZE_BY_INSTRUMENT)

    def _process(row: pd.Series) -> tuple[float, float, Any, pd.Timestamp]:
        x_min, y_min = row["bottom_left_cutout"]
        x_max, y_max = row["top_right_cutout"]
        img_sz = size_map.get(row["instrument"])
        if not img_sz:
            raise KeyError(f"Unknown instrument '{row['instrument']}' in size map.")
        width = (x_max - x_min) / img_sz
        height = (y_max - y_min) / img_sz
        return width, height, row["magnetic_class"], row["datetime"]

    results = p_map(_process, [row for _, row in df.iterrows()])
    if not results:
        return (), (), (), ()

    widths, heights, magnetic_classes, datetimes = zip(*results)
    return widths, heights, magnetic_classes, datetimes


def w_h_scatterplot(
    df: pd.DataFrame,
    size_map: Mapping[str, int] | None = None,
    marker_size: int = 3,
    opacity: float = 0.7,
) -> go.Figure:
    """Plot normalised bounding-box widths versus heights, colour-coded by magnetic class."""
    widths, heights, magnetic_classes, datetimes = compute_widths_heights(df, size_map=size_map)
    indices = list(range(len(df)))
    colors = [CLASS_COLOR_MAP.get(cls, "#cccccc") for cls in magnetic_classes]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=widths,
            y=heights,
            mode="markers",
            marker=dict(size=marker_size, color=colors, opacity=opacity),
            name="Bounding Box Dimensions",
            text=magnetic_classes,
            customdata=list(zip(datetimes, indices)),
            hovertemplate="<b>Class</b>: %{text}<br><b>Width</b>: %{x}<br><b>Height</b>: %{y}<br><b>Datetime</b>: %{customdata[0]}<br><b>Index</b>: %{customdata[1]}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Bounding Box Widths vs Heights",
        xaxis_title="Width (normalized)",
        yaxis_title="Height (normalized)",
        autosize=True,
        showlegend=True,
    )

    for cls, color in CLASS_COLOR_MAP.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                legendgroup=cls,
                showlegend=True,
                name=cls,
            )
        )

    return fig


def plot_fd(
    row: pd.Series,
    df: pd.DataFrame,
    local_path_root: str,
    cmap: str = "hmimag",
    legend_kwargs: dict[str, Any] | None = None,
) -> None:
    """Plot a single full-disk image with bounding boxes for the regions present in ``row``."""
    legend_kwargs = legend_kwargs or {"loc": "lower center", "bbox_to_anchor": (0.5, -0.05), "frameon": False}
    arccnet_path_root = row["path"].split("/fits")[0]
    image_path = row["path"].replace(arccnet_path_root, local_path_root)
    image_labels = df[df["path"] == row["path"]]

    rotated_map = load_rotated_map(image_path)
    color_map = CLASS_COLOR_MAP

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection=rotated_map)

    rotated_map.plot(axes=ax, cmap=cmap)
    rotated_map.draw_grid(axes=ax)
    draw_bounding_boxes(ax, image_labels, color_map)
    add_class_legend(fig, color_map, **legend_kwargs)

    plt.show()


def load_fulldisk_dataframe(data_folder: Path, dataset_root: Path, dataset_folder: Path, df_name: str) -> pd.DataFrame:
    """Load full-disk detection catalog and add time columns."""
    df = pd.read_parquet(data_folder / dataset_folder / df_name)
    df["time"] = df["datetime.jd1"] + df["datetime.jd2"]
    times = Time(df["time"], format="jd")
    df["datetime"] = pd.to_datetime(times.iso)
    return df


def filter_front_side(df: pd.DataFrame, longitude_threshold: float = 70.0) -> pd.DataFrame:
    """Filter dataframe to include only front-side active regions."""
    return df[(df["longitude"] < longitude_threshold) & (df["longitude"] > -longitude_threshold)]


def calculate_region_sizes(df: pd.DataFrame, img_size_dic: Mapping[str, int] | None = None) -> pd.DataFrame:
    """Calculate normalized width and height for each region."""
    img_size_dic = dict(img_size_dic or IMG_SIZE_BY_INSTRUMENT)
    result_df = df.copy()
    if result_df.empty:
        result_df["width"] = pd.Series(dtype=float)
        result_df["height"] = pd.Series(dtype=float)
        return result_df

    image_sizes = result_df["instrument"].map(img_size_dic)
    if image_sizes.isna().any():
        unknown = sorted(result_df.loc[image_sizes.isna(), "instrument"].astype(str).unique().tolist())
        raise ValueError(f"Unknown instrument(s): {unknown}")

    try:
        bottom_left = np.asarray(result_df["bottom_left_cutout"].tolist(), dtype=np.float64)
        top_right = np.asarray(result_df["top_right_cutout"].tolist(), dtype=np.float64)
    except Exception as exc:
        raise ValueError("Invalid cutout coordinate format; expected iterable (x, y) pairs.") from exc

    if bottom_left.ndim != 2 or top_right.ndim != 2 or bottom_left.shape[1] != 2 or top_right.shape[1] != 2:
        raise ValueError("Invalid cutout coordinate shape; expected arrays with shape (n_rows, 2).")

    sizes = image_sizes.to_numpy(dtype=np.float64)
    result_df["width"] = (top_right[:, 0] - bottom_left[:, 0]) / sizes
    result_df["height"] = (top_right[:, 1] - bottom_left[:, 1]) / sizes
    return result_df


def filter_by_minimum_size(
    df: pd.DataFrame,
    min_size: float = 0.03,
) -> pd.DataFrame:
    """Filter dataframe to include only regions above a minimum size threshold."""
    return df[(df["width"] >= min_size) & (df["height"] >= min_size)]


def filter_by_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out regions with bad data quality or excessive NaN values."""
    initial_count = len(df)
    df_filtered = df[~df["filtered"]].copy() if "filtered" in df.columns else df.copy()

    filtered_count = len(df_filtered)
    removed_count = initial_count - filtered_count

    if removed_count > 0:
        print(f"Quality filtering: removed {removed_count} regions ({removed_count / initial_count * 100:.1f}%)")

    return df_filtered


def prepare_fulldisk_dataset(
    data_folder: Path,
    dataset_root: Path,
    dataset_folder: Path,
    df_name: str,
    longitude_threshold: float = 70.0,
    min_size: float = 0.03,
    img_size_dic: Mapping[str, int] | None = None,
    filter_selected: bool = True,
) -> pd.DataFrame:
    """Load and prepare a full-disk dataset with quality filtering using Astropy Table."""
    from astropy.table import Table

    tab = Table.read(data_folder / dataset_folder / df_name)
    if filter_selected:
        good_quality_tab = tab[~tab["filtered"]]
    else:
        good_quality_tab = tab
    names_1d = [name for name in good_quality_tab.colnames if len(good_quality_tab[name].shape) <= 1]
    df = good_quality_tab[names_1d].to_pandas()
    for col in ["top_right_cutout", "bottom_left_cutout"]:
        if col in good_quality_tab.colnames:
            df[col] = list(good_quality_tab[col])
    if "magnetic_class" in df.columns:
        df["magnetic_class"] = df["magnetic_class"].apply(lambda x: "IA" if x == 0.0 or pd.isna(x) else x)

    df = filter_front_side(df, longitude_threshold)
    df = calculate_region_sizes(df, img_size_dic)
    df = filter_by_minimum_size(df, min_size)

    return df


def get_fits_statistics(img_path: str, data_folder: Path, dataset_root: Path) -> list[float]:
    """Extract statistical properties from a FITS file using SunPy Map."""
    local_root = pp_common.dataset_root_path(data_folder, dataset_root)
    local_path = pp_common.resolve_project_path(img_path, local_root=local_root)
    if local_path is None:
        return [np.nan, np.nan, np.nan, np.nan, 0.0, 0.0]

    try:
        sun_map = sunpy.map.Map(local_path)
        data = sun_map.data
        finite_mask = np.isfinite(data)
        finite_data = data[finite_mask]
        if len(finite_data) > 0:
            return [
                float(np.min(finite_data)),
                float(np.max(finite_data)),
                float(np.mean(finite_data)),
                float(np.std(finite_data)),
                float(np.sum(~finite_mask)),
                float(data.size),
            ]
        return [np.nan, np.nan, np.nan, np.nan, 0.0, float(data.size)]
    except Exception as e:
        print(f"Error processing {local_path}: {e}")
        return [np.nan, np.nan, np.nan, np.nan, 0.0, 0.0]


def identify_outliers(
    values: np.ndarray,
    paths: np.ndarray,
    label: str,
    threshold: float = 3.0,
    data_folder: Path | None = None,
    dataset_root: Path | None = None,
    visualize: bool = False,
    max_visualize: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify outliers using z-score method and optionally visualize them."""
    if values.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    mean = np.mean(values)
    std = np.std(values)
    if std == 0 or not np.isfinite(std):
        z_scores = np.zeros_like(values, dtype=float)
    else:
        z_scores = np.abs((values - mean) / std)
    outliers = z_scores > threshold
    outlier_indices = np.where(outliers)[0]

    if np.any(outliers):
        print(f"\n{label} Outliers (z-score > {threshold}):")
        for idx in outlier_indices:
            print(f"  {paths[idx]}")
            print(f"    Value: {values[idx]:.4e}, Z-score: {z_scores[idx]:.2f}")

        if visualize and data_folder is not None and dataset_root is not None:
            n_to_show = min(len(outlier_indices), max_visualize)
            print(f"\nVisualizing first {n_to_show} outliers:")

            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()

            for i, idx in enumerate(outlier_indices[:n_to_show]):
                img_path = paths[idx]
                local_root = pp_common.dataset_root_path(data_folder, dataset_root)
                local_path = pp_common.resolve_project_path(img_path, local_root=local_root)
                if local_path is None:
                    axes[i].text(0.5, 0.5, "Missing file", ha="center", va="center")
                    axes[i].axis("off")
                    continue
                try:
                    sun_map = sunpy.map.Map(local_path)
                    data = sun_map.data
                    axes[i].imshow(data, cmap="gray", origin="lower")
                    axes[i].set_title(f"Z={z_scores[idx]:.2f}\nVal={values[idx]:.2e}", fontsize=10)
                    axes[i].axis("off")
                except Exception as e:
                    axes[i].text(0.5, 0.5, f"Error:\n{str(e)[:30]}", ha="center", va="center")
                    axes[i].axis("off")
            for i in range(n_to_show, len(axes)):
                axes[i].axis("off")
            plt.tight_layout()
            plt.show()
    else:
        print(f"\n{label}: No outliers detected")
    return outlier_indices, z_scores


def check_file_exists(path: str, data_folder: Path, dataset_root: Path) -> bool:
    """Check if file exists after path conversion."""
    local_root = pp_common.dataset_root_path(data_folder, dataset_root)
    return pp_common.resolve_project_path(path, local_root=local_root) is not None
